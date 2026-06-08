"""
Unit tests for SAM-HQ wrapper utility logic.

These tests do not load the real SAM-HQ model. They only validate input
checking and mask selection logic.
"""

import numpy as np
import pytest

from fashion_vision.models.sam_hq_wrapper import SamHqWrapper


def test_validate_image_success() -> None:
    """Test valid RGB image input."""
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    SamHqWrapper._validate_image(image)


def test_validate_image_invalid_shape() -> None:
    """Test invalid image shape."""
    image = np.zeros((100, 120), dtype=np.uint8)

    with pytest.raises(ValueError):
        SamHqWrapper._validate_image(image)


def test_validate_box_success() -> None:
    """Test valid box conversion."""
    box = SamHqWrapper._validate_box([10, 20, 50, 80])

    assert box.shape == (4,)
    assert box.dtype == np.float32
    assert box.tolist() == [10.0, 20.0, 50.0, 80.0]


def test_validate_box_invalid_shape() -> None:
    """Test invalid box shape."""
    with pytest.raises(ValueError):
        SamHqWrapper._validate_box([10, 20, 50])


def test_validate_box_invalid_geometry() -> None:
    """Test invalid box geometry."""
    with pytest.raises(ValueError):
        SamHqWrapper._validate_box([50, 20, 10, 80])


def test_select_best_mask_single_mask() -> None:
    """Test selecting mask when only one mask is returned."""
    mask = np.ones((10, 20), dtype=np.uint8)
    scores = np.array([0.8], dtype=np.float32)

    best_mask, best_score = SamHqWrapper._select_best_mask(mask, scores)

    assert best_mask.shape == (10, 20)
    assert best_score == pytest.approx(0.8)


def test_select_best_mask_multiple_masks() -> None:
    """Test selecting the highest-score mask."""
    masks = np.zeros((3, 10, 20), dtype=np.uint8)
    masks[2, :, :] = 1
    scores = np.array([0.2, 0.5, 0.9], dtype=np.float32)

    best_mask, best_score = SamHqWrapper._select_best_mask(masks, scores)

    assert best_mask.shape == (10, 20)
    assert best_mask.sum() == 200
    assert best_score == pytest.approx(0.9)


def test_select_best_mask_invalid_shape() -> None:
    """Test invalid mask shape."""
    masks = np.zeros((1, 2, 3, 4), dtype=np.uint8)
    scores = np.array([0.5], dtype=np.float32)

    with pytest.raises(ValueError):
        SamHqWrapper._select_best_mask(masks, scores)
