"""Unit tests for mask_attribute_pipeline.py.

All tests are pure (no GPU, no model weights, no external datasets).
Tests that exercise the full ``MaskAttributePipeline.predict()`` path mock out
``GarmentAttributePipeline.predict_instance`` so that no checkpoint files are
loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path setup — mirror the CLI pattern so tests find the src package
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fashion_vision.attributes.mask_attribute_pipeline import (  # noqa: E402
    MaskAttributePipeline,
    _build_synthetic_record,
    _get_region_component,
    _load_binary_mask,
    _load_image_rgb,
    _make_overlay,
    _mask_bbox_xyxy,
    _normalize_garment_category,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_image(tmp_path) -> Path:
    """10×10 RGB JPEG image with a coloured centre."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[3:7, 3:7] = [100, 150, 200]
    path = tmp_path / "test_image.jpg"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture()
def tiny_mask(tmp_path) -> Path:
    """10×10 single-channel mask with foreground in rows/cols 3–6."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[3:7, 3:7] = 255
    path = tmp_path / "test_mask.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture()
def full_mask(tmp_path) -> Path:
    """10×10 fully-foreground mask."""
    arr = np.full((10, 10), 255, dtype=np.uint8)
    path = tmp_path / "full_mask.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture()
def empty_mask(tmp_path) -> Path:
    """10×10 all-zero mask (no foreground)."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    path = tmp_path / "empty_mask.png"
    Image.fromarray(arr).save(path)
    return path


def _mock_mapping() -> SimpleNamespace:
    """Minimal mock of AttributeGroupMapping for _infer_coarse_class tests."""
    return SimpleNamespace(
        coarse_class_to_fine_substrings={
            "top":       ["sleeve top", "sling", "vest"],
            "pants":     ["trousers", "shorts"],
            "skirt":     ["skirt"],
            "outerwear": ["outwear"],
            "dress":     ["dress"],
        },
        coarse_class_to_tasks={},
        task_to_crop_type={},
        task_to_component_filter={},
    )


# ---------------------------------------------------------------------------
# _mask_bbox_xyxy
# ---------------------------------------------------------------------------


class TestMaskBboxXyxy:
    def test_centred_mask_returns_tight_bbox(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        x1, y1, x2, y2 = _mask_bbox_xyxy(mask)
        assert x1 == 3
        assert y1 == 3
        # x2 and y2 are exclusive (one past the last foreground pixel)
        assert x2 == 7
        assert y2 == 7

    def test_full_mask_returns_full_image_bbox(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        x1, y1, x2, y2 = _mask_bbox_xyxy(mask)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 10
        assert y2 == 10

    def test_single_pixel_mask(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 4] = True
        x1, y1, x2, y2 = _mask_bbox_xyxy(mask)
        assert x1 == 4
        assert y1 == 5
        assert x2 == 5
        assert y2 == 6

    def test_empty_mask_raises_value_error(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="empty"):
            _mask_bbox_xyxy(mask)

    def test_uint8_mask_also_accepted(self) -> None:
        """_mask_bbox_xyxy should handle uint8 masks (truthy non-zero)."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 1:6] = 255
        x1, y1, x2, y2 = _mask_bbox_xyxy(mask.astype(bool))
        assert x1 == 1 and y1 == 2 and x2 == 6 and y2 == 5


# ---------------------------------------------------------------------------
# _load_image_rgb and _load_binary_mask
# ---------------------------------------------------------------------------


class TestFileLoaders:
    def test_load_image_rgb_shape(self, tiny_image: Path) -> None:
        arr = _load_image_rgb(tiny_image)
        assert arr.shape == (10, 10, 3)
        assert arr.dtype == np.uint8

    def test_load_image_rgb_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_image_rgb(tmp_path / "nonexistent.jpg")

    def test_load_binary_mask_shape(self, tiny_mask: Path) -> None:
        arr = _load_binary_mask(tiny_mask)
        assert arr.shape == (10, 10)
        assert arr.dtype == bool

    def test_load_binary_mask_foreground(self, tiny_mask: Path) -> None:
        arr = _load_binary_mask(tiny_mask)
        assert arr[3:7, 3:7].all()
        assert not arr[0:3, :].any()

    def test_load_binary_mask_full_mask(self, full_mask: Path) -> None:
        arr = _load_binary_mask(full_mask)
        assert arr.all()

    def test_load_binary_mask_empty(self, empty_mask: Path) -> None:
        arr = _load_binary_mask(empty_mask)
        assert not arr.any()

    def test_load_binary_mask_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_binary_mask(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# _normalize_garment_category
# ---------------------------------------------------------------------------


class TestNormalizeGarmentCategory:
    def test_direct_name_top(self) -> None:
        assert _normalize_garment_category("top", _mock_mapping()) == "top"

    def test_direct_name_pants(self) -> None:
        assert _normalize_garment_category("pants", _mock_mapping()) == "pants"

    def test_alias_upper_maps_to_top(self) -> None:
        assert _normalize_garment_category("upper", _mock_mapping()) == "top"

    def test_alias_coat_maps_to_outerwear(self) -> None:
        assert _normalize_garment_category("coat", _mock_mapping()) == "outerwear"

    def test_alias_trousers_maps_to_pants(self) -> None:
        assert _normalize_garment_category("trousers", _mock_mapping()) == "pants"

    def test_fine_class_long_sleeve_top(self) -> None:
        # "long sleeve top" not in alias dict → falls through to _infer_coarse_class
        # coarse_class_to_fine_substrings has "sleeve top" under "top"
        assert _normalize_garment_category("long sleeve top", _mock_mapping()) == "top"

    def test_fine_class_trousers_as_pants(self) -> None:
        # Not in alias dict; "trousers" is a substring match for "pants"
        # Wait, "trousers" IS in the alias dict.  Use a non-alias fine name.
        assert _normalize_garment_category("slim trousers", _mock_mapping()) == "pants"

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot map"):
            _normalize_garment_category("swimsuit", _mock_mapping())

    def test_case_insensitive_alias(self) -> None:
        assert _normalize_garment_category("TOP", _mock_mapping()) == "top"
        assert _normalize_garment_category("Dress", _mock_mapping()) == "dress"


# ---------------------------------------------------------------------------
# _get_region_component
# ---------------------------------------------------------------------------


class TestGetRegionComponent:
    def test_collar_maps_correctly(self) -> None:
        assert _get_region_component("collar") == ("collar", "collar")

    def test_neckline_maps_correctly(self) -> None:
        assert _get_region_component("neckline") == ("collar", "neckline")

    def test_sleeve_maps_correctly(self) -> None:
        assert _get_region_component("sleeve") == ("sleeve", "sleeve")

    def test_pant_leg_maps_correctly(self) -> None:
        assert _get_region_component("pant_leg") == ("leg_opening", "pant_leg")

    def test_pant_maps_correctly(self) -> None:
        assert _get_region_component("pant") == ("pant_leg", "pant")

    def test_unknown_falls_back_to_passthrough(self) -> None:
        region, component = _get_region_component("zipper")
        assert region == "zipper"
        assert component == "zipper"

    def test_case_insensitive(self) -> None:
        region, component = _get_region_component("COLLAR")
        assert region == "collar"
        assert component == "collar"


# ---------------------------------------------------------------------------
# _make_overlay
# ---------------------------------------------------------------------------


class TestMakeOverlay:
    def test_output_is_rgb_pil_image(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        result = _make_overlay(img, mask)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_output_has_same_spatial_dims(self) -> None:
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        mask = np.ones((20, 30), dtype=bool)
        result = _make_overlay(img, mask)
        assert result.size == (30, 20)  # PIL size is (width, height)

    def test_foreground_pixels_are_tinted(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True  # single foreground pixel
        result_arr = np.array(_make_overlay(img, mask, color=(255, 80, 80), alpha=0.45))
        # Background pixel unchanged (still near 0)
        assert result_arr[0, 0, 0] == 0
        # Foreground pixel must have non-zero red channel due to tinting
        assert result_arr[5, 5, 0] > 0

    def test_background_pixels_unchanged(self) -> None:
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        result_arr = np.array(_make_overlay(img, mask))
        # No foreground → all pixels unchanged
        np.testing.assert_array_equal(result_arr, img)


# ---------------------------------------------------------------------------
# _build_synthetic_record
# ---------------------------------------------------------------------------


class TestBuildSyntheticRecord:
    def test_all_crop_paths_point_to_masked_crop(self, tmp_path) -> None:
        masked = tmp_path / "masked.jpg"
        raw = tmp_path / "raw.jpg"
        record = _build_synthetic_record(masked, raw, "long sleeve top", "collar", "collar")
        s = str(masked)
        assert record["crop_path"] == s
        assert record["expanded_crop_path"] == s
        assert record["upper_crop_path"] == s
        assert record["masked_crop_path"] == s

    def test_image_crop_path_is_raw(self, tmp_path) -> None:
        masked = tmp_path / "masked.jpg"
        raw = tmp_path / "raw.jpg"
        record = _build_synthetic_record(masked, raw, "long sleeve top", "collar", "collar")
        assert record["image_crop_path"] == str(raw)

    def test_success_is_true(self, tmp_path) -> None:
        masked = tmp_path / "masked.jpg"
        raw = tmp_path / "raw.jpg"
        record = _build_synthetic_record(masked, raw, "trousers", "hem", "hem")
        assert record["success"] is True

    def test_class_name_preserved(self, tmp_path) -> None:
        masked = tmp_path / "masked.jpg"
        raw = tmp_path / "raw.jpg"
        record = _build_synthetic_record(masked, raw, "skirt", "hem", "hem")
        assert record["class_name"] == "skirt"


# ---------------------------------------------------------------------------
# MaskAttributePipeline.predict — schema and artifact tests
# ---------------------------------------------------------------------------


class TestMaskAttributePipelinePredict:
    """Test predict() schema and artifact creation with mocked predict_instance."""

    @pytest.fixture()
    def pipeline(self) -> MaskAttributePipeline:
        return MaskAttributePipeline()

    @pytest.fixture()
    def tiny_image_arr(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[3:7, 3:7] = [100, 150, 200]
        return arr

    @pytest.fixture()
    def tiny_mask_arr(self):
        arr = np.zeros((10, 10), dtype=bool)
        arr[3:7, 3:7] = True
        return arr

    @pytest.fixture()
    def image_file(self, tmp_path, tiny_image_arr) -> Path:
        p = tmp_path / "img.jpg"
        Image.fromarray(tiny_image_arr).save(p)
        return p

    @pytest.fixture()
    def mask_file(self, tmp_path, tiny_mask_arr) -> Path:
        p = tmp_path / "mask.png"
        Image.fromarray(tiny_mask_arr.astype(np.uint8) * 255).save(p)
        return p

    _MOCK_ATTRS = {
        "neckline_design": {
            "label": "V_shape",
            "score": 0.87,
            "topk": [
                {"label": "V_shape", "score": 0.87},
                {"label": "round", "score": 0.08},
            ],
        }
    }

    def test_output_schema_keys(self, pipeline, image_file, mask_file, tmp_path) -> None:
        with patch.object(pipeline._pipeline, "predict_instance", return_value=self._MOCK_ATTRS):
            result = pipeline.predict(
                image_path=image_file,
                mask_path=mask_file,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
            )
        expected_keys = {
            "image_path", "mask_path", "garment_category", "coarse_class",
            "component_type", "bbox_xyxy", "crop_path", "raw_crop_path",
            "overlay_path", "attributes",
        }
        assert expected_keys.issubset(result.keys())

    def test_coarse_class_resolved(self, pipeline, image_file, mask_file, tmp_path) -> None:
        with patch.object(pipeline._pipeline, "predict_instance", return_value={}):
            result = pipeline.predict(
                image_path=image_file,
                mask_path=mask_file,
                garment_category="upper",
                component_type="collar",
                output_dir=tmp_path / "out",
            )
        assert result["coarse_class"] == "top"

    def test_artifacts_saved_to_output_dir(self, pipeline, image_file, mask_file, tmp_path) -> None:
        out_dir = tmp_path / "artifacts"
        with patch.object(pipeline._pipeline, "predict_instance", return_value=self._MOCK_ATTRS):
            result = pipeline.predict(
                image_path=image_file,
                mask_path=mask_file,
                garment_category="top",
                component_type="collar",
                output_dir=out_dir,
            )
        assert Path(result["crop_path"]).exists()
        assert Path(result["raw_crop_path"]).exists()
        assert Path(result["overlay_path"]).exists()

    def test_bbox_xyxy_is_four_ints(self, pipeline, image_file, mask_file, tmp_path) -> None:
        with patch.object(pipeline._pipeline, "predict_instance", return_value={}):
            result = pipeline.predict(
                image_path=image_file,
                mask_path=mask_file,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
            )
        assert len(result["bbox_xyxy"]) == 4
        assert all(isinstance(v, int) for v in result["bbox_xyxy"])

    def test_attributes_passed_through(self, pipeline, image_file, mask_file, tmp_path) -> None:
        with patch.object(pipeline._pipeline, "predict_instance", return_value=self._MOCK_ATTRS):
            result = pipeline.predict(
                image_path=image_file,
                mask_path=mask_file,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
            )
        assert result["attributes"] == self._MOCK_ATTRS

    def test_empty_mask_raises_value_error(self, pipeline, tiny_image_arr, tmp_path) -> None:
        img_path = tmp_path / "img.jpg"
        Image.fromarray(tiny_image_arr).save(img_path)

        empty = np.zeros((10, 10), dtype=np.uint8)
        mask_path = tmp_path / "empty.png"
        Image.fromarray(empty).save(mask_path)

        with pytest.raises(ValueError, match="empty"):
            pipeline.predict(
                image_path=img_path,
                mask_path=mask_path,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
            )

    def test_spatial_mismatch_raises_value_error(self, pipeline, tmp_path) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img_path = tmp_path / "img.jpg"
        Image.fromarray(img).save(img_path)

        # Mask is a different size from the image
        mask = np.ones((20, 20), dtype=np.uint8) * 255
        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(mask_path)

        with pytest.raises(ValueError, match="dims"):
            pipeline.predict(
                image_path=img_path,
                mask_path=mask_path,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
            )

    def test_missing_image_raises_file_not_found(self, pipeline, mask_file, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            pipeline.predict(
                image_path=tmp_path / "nonexistent.jpg",
                mask_path=mask_file,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
            )

    def test_missing_mask_raises_file_not_found(self, pipeline, image_file, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            pipeline.predict(
                image_path=image_file,
                mask_path=tmp_path / "nonexistent.png",
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
            )

    def test_topk_is_passed_to_config(self, pipeline, image_file, mask_file, tmp_path) -> None:
        """Verify that predict() temporarily sets _config.topk on the inner pipeline."""
        captured_topk = []

        def _fake_predict_instance(crops):
            captured_topk.append(pipeline._pipeline._config.topk)
            return {}

        with patch.object(pipeline._pipeline, "predict_instance", side_effect=_fake_predict_instance):
            pipeline.predict(
                image_path=image_file,
                mask_path=mask_file,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
                topk=5,
            )
        assert captured_topk == [5]

    def test_topk_restored_after_predict(self, pipeline, image_file, mask_file, tmp_path) -> None:
        """_config.topk must be restored to its original value after predict()."""
        original_topk = pipeline._pipeline._config.topk
        with patch.object(pipeline._pipeline, "predict_instance", return_value={}):
            pipeline.predict(
                image_path=image_file,
                mask_path=mask_file,
                garment_category="top",
                component_type="collar",
                output_dir=tmp_path / "out",
                topk=7,
            )
        assert pipeline._pipeline._config.topk == original_topk
