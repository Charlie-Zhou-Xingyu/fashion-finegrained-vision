"""
Unit tests for IoU utilities.
"""

import numpy as np
import pytest

from fashion_vision.evaluation.iou import compute_mask_iou


def test_compute_mask_iou_perfect_match() -> None:
    """Test IoU when two masks are identical."""
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    iou = compute_mask_iou(mask, mask)

    assert iou == 1.0


def test_compute_mask_iou_no_overlap() -> None:
    """Test IoU when two masks have no overlap."""
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    iou = compute_mask_iou(pred, gt)

    assert iou == 0.0


def test_compute_mask_iou_partial_overlap() -> None:
    """Test IoU when two masks partially overlap."""
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    iou = compute_mask_iou(pred, gt)

    assert iou == pytest.approx(1 / 3)


def test_compute_mask_iou_shape_mismatch() -> None:
    """Test IoU raises error when mask shapes are different."""
    pred = np.zeros((2, 2), dtype=np.uint8)
    gt = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        compute_mask_iou(pred, gt)
