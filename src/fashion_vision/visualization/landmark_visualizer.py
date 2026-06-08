"""
Landmark visualization utilities for DeepFashion2.

This module draws clothing landmarks, landmark indices, and optional bounding
boxes on images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _get_color_by_instance(index: int) -> Tuple[int, int, int]:
    """
    Get RGB color by instance index.

    Args:
        index: Instance index.

    Returns:
        RGB color.
    """
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    return palette[index % len(palette)]


def _draw_bbox(
    image_bgr: np.ndarray,
    bbox: List[float],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """
    Draw bbox on BGR image.

    Args:
        image_bgr: BGR image.
        bbox: xyxy bbox.
        color_bgr: BGR color.
        thickness: Line thickness.
    """
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr, thickness)


def _draw_label(
    image_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_bgr: Tuple[int, int, int],
) -> None:
    """
    Draw label with black background.

    Args:
        image_bgr: BGR image.
        text: Label text.
        x: Text x.
        y: Text y.
        color_bgr: Text color.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    text_w, text_h = text_size

    x1 = max(0, x)
    y1 = max(0, y - text_h - baseline - 4)
    x2 = min(image_bgr.shape[1] - 1, x + text_w + 8)
    y2 = min(image_bgr.shape[0] - 1, y + baseline + 4)

    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        image_bgr,
        text,
        (x1 + 4, y2 - baseline - 2),
        font,
        font_scale,
        color_bgr,
        thickness,
        cv2.LINE_AA,
    )


def _draw_landmark_point(
    image_bgr: np.ndarray,
    x: float,
    y: float,
    index: int,
    color_bgr: Tuple[int, int, int],
    visibility: int = 2,
    draw_index: bool = True,
) -> None:
    """
    Draw one landmark point.

    Args:
        image_bgr: BGR image.
        x: Landmark x.
        y: Landmark y.
        index: Landmark index.
        color_bgr: BGR color.
        visibility: DeepFashion2 visibility flag.
        draw_index: Whether to draw landmark index.
    """
    px = int(round(x))
    py = int(round(y))

    if visibility == 2:
        # Strictly visible point: solid instance color.
        cv2.circle(image_bgr, (px, py), 4, color_bgr, -1)
        cv2.circle(image_bgr, (px, py), 6, (0, 0, 0), 1)
        text_color = color_bgr
    elif visibility == 1:
        # Occluded point: hollow yellow point.
        cv2.circle(image_bgr, (px, py), 5, (0, 255, 255), 2)
        text_color = (0, 255, 255)
    else:
        return

    if draw_index:
        cv2.putText(
            image_bgr,
            str(index),
            (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

def _is_landmark_drawable(
    point: Dict[str, Any],
    visible_only: bool = True,
) -> bool:
    """
    Decide whether one landmark should be drawn.

    Supports both formats:
        1. DeepFashion2-style processed format:
            visible: bool
            present: bool
            visibility: int

        2. Pipeline predicted format:
            visibility: 0/1/2
            source: predicted

    Args:
        point: Landmark point dictionary.
        visible_only: If True, draw visibility > 0 / visible points only.

    Returns:
        Whether this point should be drawn.
    """
    try:
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
    except Exception:
        return False

    if x <= 0 or y <= 0:
        return False

    visibility = int(point.get("visibility", 0))

    # New / predicted landmark format usually only has visibility.
    if "visible" not in point and "present" not in point:
        if visible_only:
            return visibility > 0
        return visibility >= 0

    # Old processed format.
    if visible_only:
        if "visible" in point:
            return bool(point.get("visible", False))
        return visibility > 0

    if "present" in point:
        return bool(point.get("present", False))

    return visibility > 0


def save_landmark_visualization(
    image_rgb: np.ndarray,
    instances: List[Dict[str, Any]],
    output_path: str | Path,
    draw_bbox: bool = True,
    draw_label: bool = True,
    draw_index: bool = True,
    visible_only: bool = True,
) -> None:
    """
    Save landmark visualization image.

    Args:
        image_rgb: RGB image.
        instances: Instance dictionaries containing bbox and landmarks.
        output_path: Output visualization path.
        draw_bbox: Whether to draw instance bbox.
        draw_label: Whether to draw instance label.
        draw_index: Whether to draw landmark indices.
        visible_only: Whether to draw visible/present landmarks only.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape HxWx3.")

    image_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for instance_index, instance in enumerate(instances):
        color_rgb = _get_color_by_instance(instance_index)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        bbox = (
            instance.get("bbox")
            or instance.get("bbox_xyxy")
        )

        category = (
            instance.get("target_category")
            or instance.get("category")
            or instance.get("class_name")
            or "unknown"
        )

        instance_id = (
            instance.get("instance_id")
            or instance.get("det_id")
            or f"item{instance_index + 1}"
        )

        if draw_bbox and bbox is not None:
            _draw_bbox(
                image_bgr=image_bgr,
                bbox=bbox,
                color_bgr=color_bgr,
                thickness=2,
            )

        if draw_label and bbox is not None:
            x1, y1, _, _ = [int(round(float(value))) for value in bbox]
            _draw_label(
                image_bgr=image_bgr,
                text=f"{instance_id}: {category}",
                x=x1,
                y=max(20, y1),
                color_bgr=color_bgr,
            )

        landmarks = instance.get("landmarks", [])

        if not isinstance(landmarks, list):
            continue

        for point in landmarks:
            if not _is_landmark_drawable(
                point=point,
                visible_only=visible_only,
            ):
                continue

            visibility = int(point.get("visibility", 2))

            _draw_landmark_point(
                image_bgr=image_bgr,
                x=float(point.get("x", 0.0)),
                y=float(point.get("y", 0.0)),
                index=int(point.get("index", 0)),
                color_bgr=color_bgr,
                visibility=visibility,
                draw_index=draw_index,
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image_bgr)
