"""
IoU evaluation utilities for binary masks.
"""

from __future__ import annotations

import numpy as np


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute intersection over union between predicted and ground-truth masks.

    Args:
        pred_mask: Predicted mask. Non-zero values are treated as foreground.
        gt_mask: Ground-truth mask. Non-zero values are treated as foreground.

    Returns:
        IoU value in range [0, 1].

    Raises:
        ValueError: If the two masks have different shapes.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            "Predicted mask and ground-truth mask must have the same shape. "
            f"Got pred={pred_mask.shape}, gt={gt_mask.shape}"
        )

    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_mask_area(mask: np.ndarray) -> int:
    """
    Compute foreground area of a binary mask.

    Args:
        mask: Input mask. Non-zero values are treated as foreground.

    Returns:
        Number of foreground pixels.
    """
    return int(mask.astype(bool).sum())


def compute_iou_at_threshold(iou_values: list[float], threshold: float) -> float:
    """
    Compute the ratio of IoU values above a given threshold.

    Args:
        iou_values: List of IoU values.
        threshold: IoU threshold.

    Returns:
        Ratio of IoU values greater than or equal to threshold.
    """
    if not iou_values:
        return 0.0

    passed = sum(iou >= threshold for iou in iou_values)
    return float(passed / len(iou_values))
