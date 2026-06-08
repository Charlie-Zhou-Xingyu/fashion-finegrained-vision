"""
Image input and output utilities.

OpenCV reads images in BGR format by default. This module provides RGB image
loading utilities for consistent model input.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_rgb(image_path: str | Path) -> np.ndarray:
    """
    Read an image file as RGB numpy array.

    Args:
        image_path: Path to the image file.

    Returns:
        RGB image as numpy array with shape ``H x W x 3``.

    Raises:
        FileNotFoundError: If image path does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Failed to read image file: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def save_image_rgb(image_rgb: np.ndarray, output_path: str | Path) -> None:
    """
    Save an RGB image to disk.

    Args:
        image_rgb: RGB image array with shape ``H x W x 3``.
        output_path: Output image path.

    Raises:
        ValueError: If the input image format is invalid.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            "Expected RGB image with shape H x W x 3, "
            f"but got shape: {image_rgb.shape}"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), image_bgr)

    if not success:
        raise ValueError(f"Failed to save image to: {path}")


def get_image_size(image_rgb: np.ndarray) -> tuple[int, int]:
    """
    Get image width and height.

    Args:
        image_rgb: RGB image array.

    Returns:
        A tuple of ``width, height``.
    """
    height, width = image_rgb.shape[:2]
    return width, height
