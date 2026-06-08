"""
Mask input and output utilities.

This module provides helper functions to save binary masks as PNG files.
Binary masks are saved as 8-bit grayscale images, where foreground pixels
are 255 and background pixels are 0.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_binary_mask_png(mask: np.ndarray, output_path: str | Path) -> None:
    """
    Save a binary mask as an 8-bit PNG image.

    Args:
        mask: Input mask. Non-zero values are treated as foreground.
        output_path: Output PNG path.

    Raises:
        ValueError: If mask is not a 2D array or cannot be saved.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a numpy.ndarray.")

    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape: {mask.shape}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    success = cv2.imwrite(str(path), mask_uint8)

    if not success:
        raise ValueError(f"Failed to save mask PNG to: {path}")
