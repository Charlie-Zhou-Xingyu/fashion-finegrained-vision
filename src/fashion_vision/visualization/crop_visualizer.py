"""
Instance crop visualization utilities.

This module saves per-instance crop images with optional predicted mask overlay.
These crops are useful for:

- Manual inspection
- Fashion attribute recognition
- Local region localization debugging
- Dataset review
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


def crop_xyxy_with_padding(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding: int = 10,
) -> Tuple[int, int, int, int]:
    """
    Crop xyxy bbox with padding and clamp it to image bounds.

    Args:
        bbox: Bounding box in xyxy format.
        image_width: Image width.
        image_height: Image height.
        padding: Padding pixels around bbox.

    Returns:
        Clamped bbox as integer xyxy.
    """
    x1, y1, x2, y2 = bbox

    x1_i = max(0, int(round(x1)) - int(padding))
    y1_i = max(0, int(round(y1)) - int(padding))
    x2_i = min(int(image_width), int(round(x2)) + int(padding))
    y2_i = min(int(image_height), int(round(y2)) + int(padding))

    if x2_i <= x1_i:
        x2_i = min(int(image_width), x1_i + 1)

    if y2_i <= y1_i:
        y2_i = min(int(image_height), y1_i + 1)

    return x1_i, y1_i, x2_i, y2_i


def overlay_mask_on_crop(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Overlay a binary mask on an RGB crop.

    Args:
        crop_rgb: RGB crop image.
        crop_mask: Binary crop mask.
        alpha: Overlay alpha.
        color: RGB color for mask overlay.

    Returns:
        RGB crop with mask overlay.
    """
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise ValueError("crop_rgb must be an RGB image with shape HxWx3.")

    mask_bool = crop_mask.astype(bool)
    output = crop_rgb.copy()

    color_arr = np.array(color, dtype=np.float32)
    output_float = output.astype(np.float32)

    output_float[mask_bool] = (
        output_float[mask_bool] * (1.0 - alpha) + color_arr * alpha
    )

    return np.clip(output_float, 0, 255).astype(np.uint8)


def save_instance_crop_visualization(
    image_rgb: np.ndarray,
    instance: Dict[str, Any],
    output_path: str | Path,
    mask_key: str = "pred_mask",
    padding: int = 10,
    alpha: float = 0.45,
    draw_label: bool = True,
) -> None:
    """
    Save one instance crop visualization.

    Args:
        image_rgb: Original RGB image.
        instance: Instance dictionary containing bbox and optional mask.
        output_path: Output crop image path.
        mask_key: Key name for mask in instance dictionary.
        padding: Padding around bbox.
        alpha: Mask overlay alpha.
        draw_label: Whether to draw category label.
    """
    if "bbox" not in instance:
        raise KeyError("Instance must contain key 'bbox'.")

    height, width = image_rgb.shape[:2]
    bbox = tuple(float(v) for v in instance["bbox"])
    x1, y1, x2, y2 = crop_xyxy_with_padding(
        bbox=bbox,
        image_width=width,
        image_height=height,
        padding=padding,
    )

    crop_rgb = image_rgb[y1:y2, x1:x2].copy()

    if mask_key in instance and instance[mask_key] is not None:
        full_mask = instance[mask_key].astype(bool)
        crop_mask = full_mask[y1:y2, x1:x2]
        crop_rgb = overlay_mask_on_crop(
            crop_rgb=crop_rgb,
            crop_mask=crop_mask,
            alpha=alpha,
            color=(255, 0, 0),
        )

    if draw_label:
        category = (
            instance.get("target_category")
            or instance.get("category")
            or "unknown"
        )
        instance_id = instance.get("instance_id", "unknown")
        label = f"{instance_id}: {category}"

        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(crop_bgr, (0, 0), (min(crop_bgr.shape[1] - 1, 240), 28), (0, 0, 0), -1)
        cv2.putText(
            crop_bgr,
            label,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), crop_bgr)
