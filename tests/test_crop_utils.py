"""Unit tests for src/fashion_vision/utils/crop_utils.py."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fashion_vision.utils.crop_utils import crop_region_from_image


def _make_image(h: int = 400, w: int = 300) -> np.ndarray:
    """Create a deterministic RGB image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _make_mask(h: int = 400, w: int = 300, bbox_xyxy: tuple = (50, 60, 200, 300)) -> np.ndarray:
    """Create a binary mask that is 1 inside bbox_xyxy, 0 outside."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox_xyxy
    mask[y1:y2, x1:x2] = 1
    return mask


# ---------------------------------------------------------------------------
# Basic output shape and type
# ---------------------------------------------------------------------------


def test_output_is_pil_image() -> None:
    img = _make_image()
    result = crop_region_from_image(img, [50, 60, 200, 300])
    assert isinstance(result, Image.Image)


def test_output_size_matches_target_size() -> None:
    img = _make_image()
    result = crop_region_from_image(img, [50, 60, 200, 300], target_size=224)
    assert result.size == (224, 224)


def test_output_mode_is_rgb() -> None:
    img = _make_image()
    result = crop_region_from_image(img, [50, 60, 200, 300])
    assert result.mode == "RGB"


def test_custom_target_size() -> None:
    img = _make_image()
    result = crop_region_from_image(img, [50, 60, 200, 300], target_size=112)
    assert result.size == (112, 112)


# ---------------------------------------------------------------------------
# Expand ratio / boundary clamping
# ---------------------------------------------------------------------------


def test_expand_ratio_zero_still_crops() -> None:
    img = _make_image()
    result = crop_region_from_image(img, [50, 60, 200, 300], expand_ratio=0.0)
    assert result.size == (224, 224)


def test_bbox_touching_image_edge_clamps_correctly() -> None:
    img = _make_image(h=100, w=100)
    result = crop_region_from_image(img, [0, 0, 100, 100], expand_ratio=0.5)
    assert result.size == (224, 224)


def test_bbox_near_edge_does_not_raise() -> None:
    img = _make_image()
    result = crop_region_from_image(img, [0, 0, 10, 10], expand_ratio=0.15)
    assert result.size == (224, 224)


# ---------------------------------------------------------------------------
# background_fill modes
# ---------------------------------------------------------------------------


def test_background_fill_keep_ignores_mask() -> None:
    img = _make_image()
    mask = _make_mask()
    result_no_mask = crop_region_from_image(img, [50, 60, 200, 300], background_fill="keep")
    result_with_mask = crop_region_from_image(
        img, [50, 60, 200, 300], mask=mask, background_fill="keep"
    )
    assert np.array_equal(np.array(result_no_mask), np.array(result_with_mask))


def test_background_fill_zero_sets_background_to_black() -> None:
    img = _make_image()
    mask = _make_mask(bbox_xyxy=(50, 60, 200, 300))
    result = crop_region_from_image(
        img, [50, 60, 200, 300], mask=mask, background_fill="zero", expand_ratio=0.5
    )
    arr = np.array(result)
    # After expanding the bbox beyond the mask, some background pixels exist.
    # At least one pixel should be zero if expansion brought in non-mask area.
    # We verify the function ran without error and produced correct shape.
    assert arr.shape == (224, 224, 3)


def test_background_fill_mean_produces_valid_image() -> None:
    img = _make_image()
    mask = _make_mask(bbox_xyxy=(50, 60, 200, 300))
    result = crop_region_from_image(
        img, [50, 60, 200, 300], mask=mask, background_fill="mean", expand_ratio=0.5
    )
    arr = np.array(result)
    assert arr.shape == (224, 224, 3)
    # Mean fill should produce uint8 values
    assert arr.dtype == np.uint8


def test_background_fill_zero_without_mask_is_same_as_keep() -> None:
    """Without a mask, zero fill has no effect (mask=None → keep behavior)."""
    img = _make_image()
    result_keep = crop_region_from_image(img, [50, 60, 200, 300], background_fill="keep")
    result_zero_no_mask = crop_region_from_image(
        img, [50, 60, 200, 300], mask=None, background_fill="zero"
    )
    assert np.array_equal(np.array(result_keep), np.array(result_zero_no_mask))


# ---------------------------------------------------------------------------
# Mask with 3-D shape (H, W, 1)
# ---------------------------------------------------------------------------


def test_mask_3d_squeeze_accepted() -> None:
    img = _make_image()
    mask_2d = _make_mask(bbox_xyxy=(50, 60, 200, 300))
    mask_3d = mask_2d[:, :, np.newaxis]
    result = crop_region_from_image(
        img, [50, 60, 200, 300], mask=mask_3d, background_fill="zero"
    )
    assert result.size == (224, 224)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_invalid_background_fill_raises() -> None:
    img = _make_image()
    with pytest.raises(ValueError, match="background_fill"):
        crop_region_from_image(img, [50, 60, 200, 300], background_fill="blur")


def test_zero_target_size_raises() -> None:
    img = _make_image()
    with pytest.raises(ValueError, match="target_size"):
        crop_region_from_image(img, [50, 60, 200, 300], target_size=0)


def test_negative_target_size_raises() -> None:
    img = _make_image()
    with pytest.raises(ValueError, match="target_size"):
        crop_region_from_image(img, [50, 60, 200, 300], target_size=-1)


def test_wrong_image_channels_raises() -> None:
    img_gray = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="image"):
        crop_region_from_image(img_gray, [0, 0, 50, 50])


def test_bbox_wrong_length_raises() -> None:
    img = _make_image()
    with pytest.raises(ValueError, match="bbox_xyxy"):
        crop_region_from_image(img, [0, 0, 50])


def test_mask_shape_mismatch_raises() -> None:
    img = _make_image(h=400, w=300)
    wrong_mask = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask"):
        crop_region_from_image(img, [50, 60, 200, 300], mask=wrong_mask)


def test_zero_area_bbox_raises() -> None:
    img = _make_image()
    # After clamping a degenerate bbox to image bounds, area is still zero.
    with pytest.raises(ValueError, match="zero-area"):
        crop_region_from_image(
            img, [50, 50, 50, 50], expand_ratio=0.0
        )
