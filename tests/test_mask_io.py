"""
Unit tests for mask IO utilities.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from fashion_vision.utils.mask_io import save_binary_mask_png


def test_save_binary_mask_png(tmp_path: Path) -> None:
    """Test saving binary mask as PNG."""
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 10:20] = 1

    output_path = tmp_path / "mask.png"

    save_binary_mask_png(mask, output_path)

    assert output_path.exists()
    assert output_path.is_file()

    loaded = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)

    assert loaded is not None
    assert loaded.shape == (20, 30)
    assert loaded.max() == 255
    assert loaded.min() == 0


def test_save_binary_mask_png_invalid_shape(tmp_path: Path) -> None:
    """Test saving mask with invalid shape."""
    mask = np.zeros((20, 30, 3), dtype=np.uint8)
    output_path = tmp_path / "invalid.png"

    with pytest.raises(ValueError):
        save_binary_mask_png(mask, output_path)
