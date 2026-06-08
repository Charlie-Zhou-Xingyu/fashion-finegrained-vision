"""
Visualization utilities for fashion instance segmentation.

This module provides reusable functions to draw predicted masks, bounding
boxes, and labels on RGB images. It is used by the 3.1.1 fashion instance
segmentation baseline to generate qualitative results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from fashion_vision.utils.image import save_image_rgb


DEFAULT_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (255, 99, 71),
    (30, 144, 255),
    (50, 205, 50),
    (255, 215, 0),
    (186, 85, 211),
    (255, 140, 0),
    (0, 206, 209),
    (220, 20, 60),
    (154, 205, 50),
    (70, 130, 180),
]


def draw_instance_segmentation(
    image_rgb: np.ndarray,
    instances: List[Dict[str, Any]],
    mask_key: str = "pred_mask",
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Draw instance segmentation results on an RGB image.

    Args:
        image_rgb: Input RGB image with shape ``H x W x 3``.
        instances: List of instance result dictionaries.
        mask_key: Key used to retrieve mask from each instance.
        alpha: Transparency factor for mask overlay.

    Returns:
        Visualized RGB image.

    Raises:
        ValueError: If image format or alpha is invalid.
    """
    _validate_image_rgb(image_rgb)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got: {alpha}")

    vis_image = image_rgb.copy()

    for index, instance in enumerate(instances):
        color = DEFAULT_COLOR_PALETTE[index % len(DEFAULT_COLOR_PALETTE)]
        mask = instance.get(mask_key)
        bbox = instance.get("bbox")

        if mask is not None:
            vis_image = overlay_mask(
                image_rgb=vis_image,
                mask=np.asarray(mask),
                color=color,
                alpha=alpha,
            )

        if bbox is not None:
            label = format_instance_label(instance)
            vis_image = draw_bbox_with_label(
                image_rgb=vis_image,
                bbox=bbox,
                label=label,
                color=color,
            )

    return vis_image


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay one binary mask on an RGB image.

    Args:
        image_rgb: Input RGB image with shape ``H x W x 3``.
        mask: Binary mask with shape ``H x W``.
        color: RGB color tuple.
        alpha: Transparency factor.

    Returns:
        RGB image with mask overlay.

    Raises:
        ValueError: If mask shape does not match image size.
    """
    _validate_image_rgb(image_rgb)

    if mask.shape[:2] != image_rgb.shape[:2]:
        raise ValueError(
            "Mask shape must match image size. "
            f"Got mask={mask.shape}, image={image_rgb.shape}"
        )

    output = image_rgb.copy()
    mask_bool = mask.astype(bool)

    if not mask_bool.any():
        return output

    color_array = np.asarray(color, dtype=np.float32)
    foreground = output[mask_bool].astype(np.float32)

    blended = (1.0 - alpha) * foreground + alpha * color_array
    output[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)

    return output


def draw_bbox_with_label(
    image_rgb: np.ndarray,
    bbox: list[float] | tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw one bounding box and label on an RGB image.

    Args:
        image_rgb: Input RGB image with shape ``H x W x 3``.
        bbox: Bounding box in ``[x1, y1, x2, y2]`` format.
        label: Text label.
        color: RGB color tuple.
        thickness: Bounding box line thickness.

    Returns:
        RGB image with bounding box and label.

    Raises:
        ValueError: If bbox is invalid.
    """
    _validate_image_rgb(image_rgb)

    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 values, got: {bbox}")

    output = image_rgb.copy()
    height, width = output.shape[:2]

    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")

    color_bgr = (int(color[2]), int(color[1]), int(color[0]))

    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.rectangle(
        output_bgr,
        (x1, y1),
        (x2, y2),
        color_bgr,
        thickness=thickness,
    )

    if label:
        output_bgr = _draw_label_background(
            image_bgr=output_bgr,
            text=label,
            x=x1,
            y=y1,
            color_bgr=color_bgr,
        )

    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)


def _draw_label_background(
    image_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    """
    Draw label text with filled background.

    Args:
        image_bgr: Input BGR image.
        text: Text to draw.
        x: Left coordinate.
        y: Top coordinate of the bounding box.
        color_bgr: Background color in BGR order.

    Returns:
        BGR image with label.
    """
    output = image_bgr.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    text_width, text_height = text_size

    label_y1 = max(0, y - text_height - baseline - 4)
    label_y2 = max(text_height + baseline + 4, y)
    label_x1 = x
    label_x2 = min(output.shape[1] - 1, x + text_width + 6)

    cv2.rectangle(
        output,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color_bgr,
        thickness=-1,
    )

    text_x = label_x1 + 3
    text_y = label_y2 - baseline - 2

    cv2.putText(
        output,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )

    return output

def format_instance_label(instance: Dict[str, Any]) -> str:
    """
    Format instance label for visualization.

    Args:
        instance: Instance result dictionary.

    Returns:
        Formatted label string.
    """
    category = str(
        instance.get("target_category")
        or instance.get("target_category_zh")
        or "unknown"
    )

    parts = [category]

    iou = instance.get("iou")
    if iou is not None:
        parts.append(f"IoU:{float(iou):.2f}")

    score = instance.get("score")
    if score is not None:
        parts.append(f"S:{float(score):.2f}")

    return " ".join(parts)

def save_instance_visualization(
    image_rgb: np.ndarray,
    instances: List[Dict[str, Any]],
    output_path: str | Path,
    mask_key: str = "pred_mask",
    alpha: float = 0.45,
) -> None:
    """
    Draw and save an instance segmentation visualization.

    Args:
        image_rgb: Input RGB image.
        instances: List of instance result dictionaries.
        output_path: Output image path.
        mask_key: Key used to retrieve mask from each instance.
        alpha: Mask transparency.
    """
    vis_image = draw_instance_segmentation(
        image_rgb=image_rgb,
        instances=instances,
        mask_key=mask_key,
        alpha=alpha,
    )
    save_image_rgb(vis_image, output_path)


def _validate_image_rgb(image_rgb: np.ndarray) -> None:
    """
    Validate RGB image format.

    Args:
        image_rgb: RGB image array.

    Raises:
        ValueError: If image format is invalid.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError("image_rgb must be a numpy.ndarray.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            "image_rgb must have shape H x W x 3. "
            f"Got shape: {image_rgb.shape}"
        )

    if image_rgb.size == 0:
        raise ValueError("image_rgb must not be empty.")
