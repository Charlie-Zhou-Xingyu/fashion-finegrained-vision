"""
Mask-based landmark quality annotation and refinement utilities.

This module provides lightweight post-processing for predicted garment
landmarks using binary segmentation masks, e.g. SAM-HQ masks.

Main features:
    1. Check whether each landmark lies inside the mask.
    2. Compute distance from each landmark to the nearest mask pixel.
    3. Optionally project outside landmarks to the nearest mask pixel.

The functions are designed for inference-time post-processing and do not
require model retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_binary_mask(mask_path: str | Path) -> Optional[np.ndarray]:
    """
    Load a binary mask from disk.

    Args:
        mask_path: Path to mask image. Expected to be a grayscale or RGB image
            where foreground garment pixels are non-zero.

    Returns:
        Binary mask as uint8 array [H, W], values 0 or 1.
        Returns None if loading fails.
    """
    if mask_path is None:
        return None

    mask_path = Path(mask_path)

    if not mask_path.exists():
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    binary = (mask > 0).astype(np.uint8)
    return binary


def _safe_round_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """
    Round and clip point coordinates to image boundary.

    Args:
        x: x coordinate.
        y: y coordinate.
        width: Image width.
        height: Image height.

    Returns:
        Clipped integer point.
    """
    px = int(round(float(x)))
    py = int(round(float(y)))

    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))

    return px, py


def compute_distance_to_mask(
    mask: np.ndarray,
    x: float,
    y: float,
) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
    """
    Compute whether a point is inside mask and its distance to nearest mask pixel.

    Args:
        mask: Binary mask [H, W], values 0 or 1.
        x: Landmark x coordinate in image coordinates.
        y: Landmark y coordinate in image coordinates.

    Returns:
        inside_mask:
            Whether rounded landmark point is inside foreground mask.
        distance:
            Distance in pixels to nearest foreground mask pixel.
            0.0 if inside mask.
            inf if mask has no foreground pixels.
        nearest_point:
            Nearest foreground mask pixel as (x, y), or None if mask is empty.
    """
    height, width = mask.shape[:2]
    px, py = _safe_round_point(x=x, y=y, width=width, height=height)

    if mask[py, px] > 0:
        return True, 0.0, (px, py)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return False, float("inf"), None

    dx = xs.astype(np.float32) - float(px)
    dy = ys.astype(np.float32) - float(py)
    dist_sq = dx * dx + dy * dy

    nearest_idx = int(np.argmin(dist_sq))
    distance = float(np.sqrt(float(dist_sq[nearest_idx])))

    nearest_x = int(xs[nearest_idx])
    nearest_y = int(ys[nearest_idx])

    return False, distance, (nearest_x, nearest_y)


def annotate_landmarks_with_mask_quality(
    landmarks: List[Dict[str, Any]],
    mask_path: str | Path | None,
    missing_mask_quality: str = "no_mask",
) -> List[Dict[str, Any]]:
    """
    Annotate predicted landmarks with mask-based quality fields.

    This function does not change landmark coordinates. It only adds metadata.

    Added fields:
        inside_mask: bool or None
        distance_to_mask: float or None
        nearest_mask_x: float or None
        nearest_mask_y: float or None
        quality: str

    Quality values:
        ok:
            Point is inside mask.
        outside_mask:
            Point is outside mask, but mask exists.
        empty_mask:
            Mask exists but has no foreground pixels.
        no_mask:
            Mask is missing or cannot be loaded.

    Args:
        landmarks: Landmark list.
        mask_path: Binary mask path.
        missing_mask_quality: Quality label when mask is unavailable.

    Returns:
        New landmark list with added fields.
    """
    output: List[Dict[str, Any]] = []

    mask = load_binary_mask(mask_path) if mask_path else None

    if mask is None:
        for landmark in landmarks:
            item = dict(landmark)
            item["inside_mask"] = None
            item["distance_to_mask"] = None
            item["nearest_mask_x"] = None
            item["nearest_mask_y"] = None
            item["quality"] = missing_mask_quality
            output.append(item)
        return output

    for landmark in landmarks:
        item = dict(landmark)

        try:
            x = float(item["x"])
            y = float(item["y"])
        except Exception:
            item["inside_mask"] = None
            item["distance_to_mask"] = None
            item["nearest_mask_x"] = None
            item["nearest_mask_y"] = None
            item["quality"] = "invalid_point"
            output.append(item)
            continue

        inside, distance, nearest = compute_distance_to_mask(
            mask=mask,
            x=x,
            y=y,
        )

        item["inside_mask"] = bool(inside)

        if np.isinf(distance):
            item["distance_to_mask"] = None
            item["nearest_mask_x"] = None
            item["nearest_mask_y"] = None
            item["quality"] = "empty_mask"
        else:
            item["distance_to_mask"] = float(distance)

            if nearest is None:
                item["nearest_mask_x"] = None
                item["nearest_mask_y"] = None
            else:
                item["nearest_mask_x"] = float(nearest[0])
                item["nearest_mask_y"] = float(nearest[1])

            item["quality"] = "ok" if inside else "outside_mask"

        output.append(item)

    return output


def refine_landmarks_to_mask(
    landmarks: List[Dict[str, Any]],
    mask_path: str | Path | None,
    max_distance_px: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Project outside-mask landmarks to nearest mask pixel when distance is small.

    This function preserves raw coordinates as x_raw / y_raw when refinement
    is applied.

    Args:
        landmarks: Landmark list, preferably already annotated by
            annotate_landmarks_with_mask_quality.
        mask_path: Binary mask path.
        max_distance_px: Only refine if distance to mask is <= this threshold.

    Returns:
        Refined landmark list.
    """
    mask = load_binary_mask(mask_path) if mask_path else None

    if mask is None:
        output = []
        for landmark in landmarks:
            item = dict(landmark)
            item["refined_by_mask"] = False
            output.append(item)
        return output

    output: List[Dict[str, Any]] = []

    for landmark in landmarks:
        item = dict(landmark)
        item["refined_by_mask"] = False

        try:
            x = float(item["x"])
            y = float(item["y"])
        except Exception:
            output.append(item)
            continue

        inside, distance, nearest = compute_distance_to_mask(
            mask=mask,
            x=x,
            y=y,
        )

        if inside:
            item["inside_mask"] = True
            item["distance_to_mask"] = 0.0
            item["refined_by_mask"] = False
            item["quality"] = "ok"
            output.append(item)
            continue

        if nearest is not None and distance <= float(max_distance_px):
            item["x_raw"] = float(x)
            item["y_raw"] = float(y)
            item["x"] = float(nearest[0])
            item["y"] = float(nearest[1])
            item["inside_mask"] = True
            item["distance_to_mask"] = float(distance)
            item["nearest_mask_x"] = float(nearest[0])
            item["nearest_mask_y"] = float(nearest[1])
            item["refined_by_mask"] = True
            item["quality"] = "refined_by_mask"
        else:
            item["inside_mask"] = False
            item["distance_to_mask"] = None if np.isinf(distance) else float(distance)
            item["refined_by_mask"] = False
            item["quality"] = "outside_mask_far"

        output.append(item)

    return output
