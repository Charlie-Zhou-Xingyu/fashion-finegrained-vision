"""
Visualization utilities for local fashion regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


def overlay_region_mask(
    image_rgb: np.ndarray,
    region_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay local region mask on RGB image.

    Args:
        image_rgb: RGB image.
        region_mask: Binary region mask.
        color: RGB overlay color.
        alpha: Overlay alpha.

    Returns:
        RGB image with overlay.
    """
    output = image_rgb.copy()
    mask_bool = region_mask.astype(bool)

    color_arr = np.array(color, dtype=np.float32)
    output_float = output.astype(np.float32)

    output_float[mask_bool] = (
        output_float[mask_bool] * (1.0 - alpha) + color_arr * alpha
    )

    return np.clip(output_float, 0, 255).astype(np.uint8)


def save_region_visualization(
    image_rgb: np.ndarray,
    region_result: Dict[str, Any],
    output_path: str | Path,
    alpha: float = 0.55,
) -> None:
    """
    Save local region visualization.

    Args:
        image_rgb: RGB image.
        region_result: Region result dictionary.
        output_path: Output visualization path.
        alpha: Mask overlay alpha.
    """
    if "region_mask" not in region_result:
        raise KeyError("region_result must contain 'region_mask'.")

    region_mask = region_result["region_mask"]
    output_rgb = overlay_region_mask(
        image_rgb=image_rgb,
        region_mask=region_mask,
        color=(0, 255, 255),
        alpha=alpha,
    )

    image_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    bbox = region_result.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

    label = (
        f"{region_result.get('target_instance_id', 'unknown')}: "
        f"{region_result.get('region_type', 'unknown')}"
    )

    if bbox is not None:
        text_x = max(0, int(round(bbox[0])))
        text_y = max(20, int(round(bbox[1])))
    else:
        text_x, text_y = 10, 30

    cv2.rectangle(
        image_bgr,
        (text_x, text_y - 22),
        (text_x + min(260, 12 * len(label)), text_y + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image_bgr,
        label,
        (text_x + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)
