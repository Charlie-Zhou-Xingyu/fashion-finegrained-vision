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

Robustness notes:
    - Mask files may be loaded as grayscale [H, W], BGR [H, W, 3],
      BGRA [H, W, 4], boolean arrays, or probability maps.
    - All public functions normalize masks to a 2D uint8 binary array
      with values 0 or 1 before geometric operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def ensure_2d_mask(mask: np.ndarray | None) -> Optional[np.ndarray]:
    """
    Convert an input mask to a 2D array.

    Args:
        mask:
            Input mask. Supported shapes:
                - [H, W]
                - [H, W, 1]
                - [H, W, 3]
                - [H, W, 4]
                - arrays with squeeze-able singleton dimensions

    Returns:
        2D mask array, or None if input is None.

    Raises:
        ValueError if mask cannot be converted to a 2D array.
    """
    if mask is None:
        return None

    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        channels = mask.shape[2]

        if channels == 1:
            return mask[:, :, 0]

        if channels >= 3:
            # cv2.imread returns BGR/BGRA. For binary masks, any foreground
            # channel should work, but grayscale conversion is safer for
            # colored overlays or anti-aliased masks.
            return cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)

    squeezed = np.squeeze(mask)

    if squeezed.ndim == 2:
        return squeezed

    raise ValueError(
        f"Expected mask convertible to 2D, got shape={getattr(mask, 'shape', None)}"
    )


def binarize_mask(mask: np.ndarray | None, threshold: float = 0.0) -> Optional[np.ndarray]:
    """
    Convert mask to uint8 binary array with values 0 or 1.

    Args:
        mask:
            Input mask. It can be bool, uint8, float, grayscale, or color.
        threshold:
            Foreground threshold. Pixels greater than threshold are foreground.

    Returns:
        Binary mask [H, W] uint8 with values 0 or 1, or None.
    """
    if mask is None:
        return None

    mask_2d = ensure_2d_mask(mask)

    if mask_2d is None:
        return None

    if mask_2d.dtype == np.bool_:
        return mask_2d.astype(np.uint8)

    # Handle NaN/inf robustly for float masks.
    if np.issubdtype(mask_2d.dtype, np.floating):
        clean = np.nan_to_num(mask_2d, nan=0.0, posinf=1.0, neginf=0.0)
        binary = clean > float(threshold)
    else:
        binary = mask_2d > threshold

    return binary.astype(np.uint8)


def load_binary_mask(mask_path: str | Path) -> Optional[np.ndarray]:
    """
    Load a binary mask from disk.

    Args:
        mask_path:
            Path to mask image. Expected to be a grayscale or RGB image
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

    # Prefer unchanged first to support grayscale / alpha / unusual PNG masks.
    # Then normalize it ourselves.
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        return None

    try:
        binary = binarize_mask(mask, threshold=0.0)
    except Exception:
        # Fallback to grayscale read if unchanged decoding gave unexpected shape.
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            return None
        binary = binarize_mask(mask_gray, threshold=0.0)

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
        Clipped integer point as (px, py).
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
        mask:
            Binary mask [H, W], values 0 or 1.
            For robustness, color masks [H, W, C] are also accepted and will
            be converted to 2D binary masks.
        x:
            Landmark x coordinate in image coordinates.
        y:
            Landmark y coordinate in image coordinates.

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
    binary_mask = binarize_mask(mask, threshold=0.0)

    if binary_mask is None:
        return False, float("inf"), None

    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape={binary_mask.shape}")

    height, width = binary_mask.shape[:2]

    if height <= 0 or width <= 0:
        return False, float("inf"), None

    px, py = _safe_round_point(x=x, y=y, width=width, height=height)

    if binary_mask[py, px] > 0:
        return True, 0.0, (px, py)

    ys, xs = np.where(binary_mask > 0)

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
        invalid_point:
            Landmark x/y is invalid.
        mask_error:
            Mask exists but cannot be processed robustly.

    Args:
        landmarks:
            Landmark list.
        mask_path:
            Binary mask path.
        missing_mask_quality:
            Quality label when mask is unavailable.

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

        try:
            inside, distance, nearest = compute_distance_to_mask(
                mask=mask,
                x=x,
                y=y,
            )
        except Exception as exc:
            item["inside_mask"] = None
            item["distance_to_mask"] = None
            item["nearest_mask_x"] = None
            item["nearest_mask_y"] = None
            item["quality"] = "mask_error"
            item["mask_error"] = repr(exc)
            output.append(item)
            continue

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
        landmarks:
            Landmark list, preferably already annotated by
            annotate_landmarks_with_mask_quality.
        mask_path:
            Binary mask path.
        max_distance_px:
            Only refine if distance to mask is <= this threshold.

    Returns:
        Refined landmark list.
    """
    mask = load_binary_mask(mask_path) if mask_path else None

    if mask is None:
        output: List[Dict[str, Any]] = []
        for landmark in landmarks:
            item = dict(landmark)
            item["refined_by_mask"] = False

            if "quality" not in item:
                item["quality"] = "no_mask"

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
            item["quality"] = "invalid_point"
            output.append(item)
            continue

        try:
            inside, distance, nearest = compute_distance_to_mask(
                mask=mask,
                x=x,
                y=y,
            )
        except Exception as exc:
            item["inside_mask"] = None
            item["distance_to_mask"] = None
            item["nearest_mask_x"] = None
            item["nearest_mask_y"] = None
            item["refined_by_mask"] = False
            item["quality"] = "mask_error"
            item["mask_error"] = repr(exc)
            output.append(item)
            continue

        if inside:
            item["inside_mask"] = True
            item["distance_to_mask"] = 0.0
            item["nearest_mask_x"] = float(_safe_round_point(x, y, mask.shape[1], mask.shape[0])[0])
            item["nearest_mask_y"] = float(_safe_round_point(x, y, mask.shape[1], mask.shape[0])[1])
            item["refined_by_mask"] = False
            item["quality"] = "ok"
            output.append(item)
            continue

        if nearest is None:
            item["inside_mask"] = False
            item["distance_to_mask"] = None
            item["nearest_mask_x"] = None
            item["nearest_mask_y"] = None
            item["refined_by_mask"] = False
            item["quality"] = "empty_mask"
            output.append(item)
            continue

        if distance <= float(max_distance_px):
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
            item["nearest_mask_x"] = float(nearest[0])
            item["nearest_mask_y"] = float(nearest[1])
            item["refined_by_mask"] = False
            item["quality"] = "outside_mask_far"

        output.append(item)

    return output
