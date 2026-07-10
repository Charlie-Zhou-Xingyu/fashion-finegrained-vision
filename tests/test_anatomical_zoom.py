"""Unit tests for anatomical_zoom.py — crop coordinate math and box remapping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from fashion_vision.localization.anatomical_zoom import (
    ANATOMICAL_ZOOM_CONFIG,
    apply_anatomical_zoom,
    map_box_from_zoom_to_original,
)


@pytest.fixture
def synthetic_image():
    img = np.zeros((800, 600, 3), dtype=np.uint8)
    mask = np.zeros((800, 600), dtype=np.uint8)
    mask[200:600, 150:450] = 255
    inst_bbox = [150, 200, 450, 600]
    return img, mask, inst_bbox


class TestApplyAnatomicalZoom:
    def test_zipper_zoom(self, synthetic_image):
        img, mask, bbox = synthetic_image
        zoomed, zmask, tform = apply_anatomical_zoom(img, mask, bbox, "zipper")
        assert tform["zoom_applied"] is True
        assert tform["zoom_factor"] == 2.0
        assert zoomed.shape[0] > 0 and zoomed.shape[1] > 0
        # offset should be inside the garment bbox
        assert tform["offset_x"] >= bbox[0]
        assert tform["offset_y"] >= bbox[1]

    def test_all_zoomable_parts(self, synthetic_image):
        img, mask, bbox = synthetic_image
        for part in ANATOMICAL_ZOOM_CONFIG:
            zoomed, zmask, tform = apply_anatomical_zoom(img, mask, bbox, part)
            assert tform["zoom_applied"] is True, f"{part}: zoom not applied"
            assert tform["zoom_factor"] >= 1.0, f"{part}: bad zoom factor"
            assert "offset_x" in tform
            assert "scale_x" in tform

    def test_fallback_unknown_part(self, synthetic_image):
        img, mask, bbox = synthetic_image
        _, _, tform = apply_anatomical_zoom(img, mask, bbox, "nonexistent_part")
        assert tform["zoom_applied"] is False
        assert tform["zoom_factor"] == 1.0

    def test_fallback_invalid_bbox(self, synthetic_image):
        img, mask, _ = synthetic_image
        _, _, tform = apply_anatomical_zoom(img, mask, [100, 100, 50, 50], "zipper")
        assert tform["zoom_applied"] is False

    def test_mask_none_handling(self, synthetic_image):
        img, _, bbox = synthetic_image
        zoomed, zmask, tform = apply_anatomical_zoom(img, None, bbox, "zipper")
        assert zmask is None
        assert zoomed is not None

    def test_mask_resize_alignment(self, synthetic_image):
        img, mask, bbox = synthetic_image
        # Create a mask with different spatial dimensions
        mask_small = np.zeros((400, 300), dtype=np.uint8)
        mask_small[100:300, 75:225] = 255
        zoomed, zmask, tform = apply_anatomical_zoom(img, mask_small, bbox, "zipper")
        assert zmask is not None
        assert zmask.shape[:2] == zoomed.shape[:2], (
            f"mask shape {zmask.shape} != image shape {zoomed.shape}"
        )


class TestMapBoxFromZoomToOriginal:
    def test_basic_remap(self):
        tform = {
            "offset_x": 100, "offset_y": 50,
            "scale_x": 2.0, "scale_y": 2.0,
            "crop_box": [0, 0, 200, 300],
            "garment_bbox": [100, 50, 500, 650],
            "zoom_applied": True, "zoom_factor": 2.0, "part": "zipper",
        }
        # Box at (20, 10, 100, 60) in zoomed (2x) space
        # Should map to: (20/2+100, 10/2+50, 100/2+100, 60/2+50) = (110, 55, 150, 80)
        result = map_box_from_zoom_to_original([20, 10, 100, 60], tform)
        assert result == [110, 55, 150, 80]

    def test_no_zoom_identity(self):
        tform = {
            "offset_x": 8, "offset_y": 8,
            "scale_x": 1.0, "scale_y": 1.0,
            "crop_box": [0, 0, 400, 500],
            "garment_bbox": [8, 8, 408, 508],
            "zoom_applied": False, "zoom_factor": 1.0, "part": "__fallback__",
        }
        result = map_box_from_zoom_to_original([50, 30, 200, 150], tform)
        assert result == [58, 38, 208, 158]

    def test_returns_int_list(self):
        tform = {
            "offset_x": 100, "offset_y": 50,
            "scale_x": 2.0, "scale_y": 2.0,
            "crop_box": [0, 0, 200, 300],
            "garment_bbox": [100, 50, 500, 650],
            "zoom_applied": True, "zoom_factor": 2.0, "part": "zipper",
        }
        result = map_box_from_zoom_to_original([20.5, 10.3, 100.7, 60.1], tform)
        assert all(isinstance(v, int) for v in result)


class TestZoomConfig:
    def test_all_have_required_keys(self):
        for part, cfg in ANATOMICAL_ZOOM_CONFIG.items():
            assert "x_range" in cfg, f"{part}: missing x_range"
            assert "y_range" in cfg, f"{part}: missing y_range"
            assert "zoom_factor" in cfg, f"{part}: missing zoom_factor"
            assert 0.0 <= cfg["x_range"][0] <= 1.0
            assert 0.0 <= cfg["y_range"][0] <= 1.0
            assert cfg["x_range"][0] < cfg["x_range"][1]
            assert cfg["y_range"][0] < cfg["y_range"][1]

    def test_zoom_factors_sane(self):
        for part, cfg in ANATOMICAL_ZOOM_CONFIG.items():
            assert 1.5 <= cfg["zoom_factor"] <= 4.0, (
                f"{part}: zoom_factor {cfg['zoom_factor']} out of range [1.5, 4.0]"
            )
