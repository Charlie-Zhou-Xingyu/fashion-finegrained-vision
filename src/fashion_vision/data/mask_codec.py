"""
Mask utility functions.

This module provides basic binary mask conversion and area computation
utilities. More advanced RLE encoding can be added later if needed.
"""

from __future__ import annotations

import numpy as np


def to_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert an input mask to a binary uint8 mask.

    Args:
        mask: Input mask array.
        threshold: Threshold used when mask is floating point.

    Returns:
        Binary mask with dtype uint8 and values 0 or 1.
    """
    if np.issubdtype(mask.dtype, np.floating):
        binary = mask > threshold
    else:
        binary = mask > 0

    return binary.astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    """
    Compute foreground area of a mask.

    Args:
        mask: Input mask. Non-zero values are treated as foreground.

    Returns:
        Number of foreground pixels.
    """
    return int((mask > 0).sum())


def validate_mask_shape(mask: np.ndarray, height: int, width: int) -> None:
    """
    Validate whether a mask has expected height and width.

    Args:
        mask: Input mask.
        height: Expected height.
        width: Expected width.

    Raises:
        ValueError: If the mask shape is invalid.
    """
    if mask.shape[:2] != (height, width):
        raise ValueError(
            f"Invalid mask shape: {mask.shape}. "
            f"Expected height={height}, width={width}."
        )
