"""Unit tests for src/fashion_vision/attributes/garment_attribute_pipeline.py.

All tests use pure logic or synthetic data only.  No model checkpoints,
no real crop images, and no GPU access are required.

The tests are grouped as follows:

* ``_resolve_device``            — device resolution helper
* ``_infer_coarse_class``        — fine→coarse mapping via substrings
* ``_get_crop_path``             — crop-type to dict-key fallback chain
* ``_select_crop_record``        — multi-filter crop record selection
* ``_run_inference``             — forward pass with a tiny synthetic model
* ``AttributePipelineConfig``    — default values
* ``GarmentAttributePipeline``   — integration tests with monkeypatched models
* ``predict_from_json``          — JSON loading + grouping logic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torchvision import transforms

# ---------------------------------------------------------------------------
# Add src/ to sys.path (mirrors the module's own setup for direct script use)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fashion_vision.attributes.garment_attribute_pipeline import (  # noqa: E402
    AttributePipelineConfig,
    GarmentAttributePipeline,
    _get_crop_path,
    _infer_coarse_class,
    _resolve_device,
    _run_inference,
    _select_crop_record,
)
from fashion_vision.attributes.category_gate import load_attribute_group_mapping
from fashion_vision.attributes.task_registry import LoadedTask, AttributeTaskConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_GROUP_MAPPING_YAML = _PROJECT_ROOT / "configs" / "attribute_group_mapping.yaml"
_INFERENCE_YAML = _PROJECT_ROOT / "configs" / "attribute_inference.yaml"


@pytest.fixture(scope="module")
def mapping():
    """Real AttributeGroupMapping loaded from the config file."""
    return load_attribute_group_mapping(_GROUP_MAPPING_YAML)


def _make_loaded_task(num_classes: int = 5) -> LoadedTask:
    """Build a minimal LoadedTask with a tiny linear model (CPU, no checkpoint)."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, num_classes))
    model.eval()
    id_to_label = {i: f"Label{i}" for i in range(num_classes)}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    config = AttributeTaskConfig(
        task="test_task",
        checkpoint=Path("fake/ckpt.pt"),
        label_map=Path("fake/lm.json"),
        arch="linear_test",
        img_size=224,
    )
    return LoadedTask(config=config, model=model, id_to_label=id_to_label, transform=transform)


def _make_crop_record(**kwargs) -> dict[str, Any]:
    """Return a minimal crop record with sensible defaults, overridable via kwargs."""
    base: dict[str, Any] = {
        "det_id": "img001_det0",
        "class_name": "long sleeve top",
        "region": "collar",
        "component": "front_collar",
        "success": True,
        "expanded_crop_path": None,
        "upper_crop_path": None,
        "image_crop_path": None,
        "crop_path": None,
        "masked_crop_path": None,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu() -> None:
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_device() -> None:
    d = _resolve_device("auto")
    assert d.type in ("cpu", "cuda")


def test_resolve_device_auto_cpu_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA present — auto would return cuda.")
    assert _resolve_device("auto") == torch.device("cpu")


# ---------------------------------------------------------------------------
# _infer_coarse_class
# ---------------------------------------------------------------------------


def test_infer_coarse_class_long_sleeve_top(mapping) -> None:
    assert _infer_coarse_class("long sleeve top", mapping) == "top"


def test_infer_coarse_class_short_sleeve_top(mapping) -> None:
    assert _infer_coarse_class("short sleeve top", mapping) == "top"


def test_infer_coarse_class_vest(mapping) -> None:
    assert _infer_coarse_class("vest", mapping) == "top"


def test_infer_coarse_class_sling(mapping) -> None:
    assert _infer_coarse_class("sling", mapping) == "top"


def test_infer_coarse_class_outwear(mapping) -> None:
    assert _infer_coarse_class("long sleeve outwear", mapping) == "outerwear"


def test_infer_coarse_class_short_sleeve_outwear(mapping) -> None:
    assert _infer_coarse_class("short sleeve outwear", mapping) == "outerwear"


def test_infer_coarse_class_trousers(mapping) -> None:
    assert _infer_coarse_class("trousers", mapping) == "pants"


def test_infer_coarse_class_shorts(mapping) -> None:
    assert _infer_coarse_class("shorts", mapping) == "pants"


def test_infer_coarse_class_skirt(mapping) -> None:
    assert _infer_coarse_class("skirt", mapping) == "skirt"


def test_infer_coarse_class_dress(mapping) -> None:
    assert _infer_coarse_class("long sleeve dress", mapping) == "dress"


def test_infer_coarse_class_unknown_returns_none(mapping) -> None:
    assert _infer_coarse_class("hat", mapping) is None


def test_infer_coarse_class_empty_string_returns_none(mapping) -> None:
    assert _infer_coarse_class("", mapping) is None


# ---------------------------------------------------------------------------
# _get_crop_path
# ---------------------------------------------------------------------------


def test_get_crop_path_expanded_prefers_expanded_crop_path() -> None:
    record = {"expanded_crop_path": "/a/e.jpg", "image_crop_path": "/a/i.jpg"}
    assert _get_crop_path(record, "expanded_crop") == "/a/e.jpg"


def test_get_crop_path_expanded_falls_back_to_image_crop_path() -> None:
    record = {"image_crop_path": "/a/i.jpg", "crop_path": "/a/c.jpg"}
    assert _get_crop_path(record, "expanded_crop") == "/a/i.jpg"


def test_get_crop_path_expanded_falls_back_to_crop_path() -> None:
    record = {"crop_path": "/a/c.jpg"}
    assert _get_crop_path(record, "expanded_crop") == "/a/c.jpg"


def test_get_crop_path_upper_prefers_upper_crop_path() -> None:
    record = {"upper_crop_path": "/a/u.jpg", "expanded_crop_path": "/a/e.jpg"}
    assert _get_crop_path(record, "upper_crop") == "/a/u.jpg"


def test_get_crop_path_upper_falls_back_to_expanded() -> None:
    record = {"expanded_crop_path": "/a/e.jpg"}
    assert _get_crop_path(record, "upper_crop") == "/a/e.jpg"


def test_get_crop_path_masked_crop() -> None:
    record = {"masked_crop_path": "/a/m.jpg"}
    assert _get_crop_path(record, "masked_crop") == "/a/m.jpg"


def test_get_crop_path_image_crop() -> None:
    record = {"image_crop_path": "/a/i.jpg"}
    assert _get_crop_path(record, "image_crop") == "/a/i.jpg"


def test_get_crop_path_all_empty_returns_none() -> None:
    assert _get_crop_path({}, "expanded_crop") is None


def test_get_crop_path_unknown_type_falls_back_to_crop_path() -> None:
    record = {"crop_path": "/a/c.jpg"}
    assert _get_crop_path(record, "raw_region_crop") == "/a/c.jpg"


# ---------------------------------------------------------------------------
# _select_crop_record
# ---------------------------------------------------------------------------


def _make_crops(*overrides: dict[str, Any]) -> list[dict[str, Any]]:
    return [_make_crop_record(**ov) for ov in overrides]


def test_select_crop_record_basic_match() -> None:
    crops = _make_crops({"region": "collar", "success": True})
    result = _select_crop_record(crops, "collar", None, None)
    assert result is not None
    assert result["region"] == "collar"


def test_select_crop_record_region_all_skips_region_filter() -> None:
    crops = _make_crops(
        {"region": "sleeve", "success": True},
        {"region": "hem", "success": True},
    )
    result = _select_crop_record(crops, "all", None, None)
    assert result is not None
    assert result["region"] == "sleeve"


def test_select_crop_record_skips_failed_records() -> None:
    crops = _make_crops(
        {"region": "collar", "success": False},
        {"region": "collar", "success": True},
    )
    result = _select_crop_record(crops, "collar", None, None)
    assert result is not None
    assert result["success"] is True


def test_select_crop_record_class_contains_filter() -> None:
    crops = _make_crops(
        {"class_name": "long sleeve top", "region": "all", "success": True},
    )
    result = _select_crop_record(crops, "all", "outwear", None)
    assert result is None


def test_select_crop_record_class_contains_match() -> None:
    crops = _make_crops(
        {"class_name": "long sleeve outwear", "region": "all", "success": True},
    )
    result = _select_crop_record(crops, "all", "outwear", None)
    assert result is not None


def test_select_crop_record_component_contains_filter() -> None:
    crops = _make_crops(
        {"component": "front_hem", "region": "all", "success": True},
    )
    result = _select_crop_record(crops, "all", None, "sleeve")
    assert result is None


def test_select_crop_record_component_contains_match() -> None:
    crops = _make_crops(
        {"component": "left_sleeve", "region": "all", "success": True},
    )
    result = _select_crop_record(crops, "all", None, "sleeve")
    assert result is not None


def test_select_crop_record_returns_none_for_empty_list() -> None:
    assert _select_crop_record([], "all", None, None) is None


def test_select_crop_record_returns_first_matching() -> None:
    crops = _make_crops(
        {"region": "collar", "success": True, "det_id": "first"},
        {"region": "collar", "success": True, "det_id": "second"},
    )
    result = _select_crop_record(crops, "collar", None, None)
    assert result["det_id"] == "first"


# ---------------------------------------------------------------------------
# _run_inference — tiny synthetic model, no real checkpoint or image
# ---------------------------------------------------------------------------


def test_run_inference_returns_label_score_topk(tmp_path: Path) -> None:
    # Create a tiny white PNG image.
    from PIL import Image as PILImage
    import numpy as np

    img_path = tmp_path / "crop.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB").save(img_path)

    loaded = _make_loaded_task(num_classes=5)
    device = torch.device("cpu")
    result = _run_inference(loaded, img_path, topk=3, device=device)

    assert "label" in result
    assert "score" in result
    assert "topk" in result
    assert isinstance(result["label"], str)
    assert 0.0 <= result["score"] <= 1.0
    assert len(result["topk"]) == 3


def test_run_inference_topk_capped_by_num_classes(tmp_path: Path) -> None:
    from PIL import Image as PILImage
    import numpy as np

    img_path = tmp_path / "crop.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB").save(img_path)

    loaded = _make_loaded_task(num_classes=2)
    result = _run_inference(loaded, img_path, topk=10, device=torch.device("cpu"))
    assert len(result["topk"]) == 2


def test_run_inference_topk_scores_sum_to_one(tmp_path: Path) -> None:
    from PIL import Image as PILImage
    import numpy as np

    img_path = tmp_path / "crop.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB").save(img_path)

    loaded = _make_loaded_task(num_classes=4)
    result = _run_inference(loaded, img_path, topk=4, device=torch.device("cpu"))
    total = sum(item["score"] for item in result["topk"])
    assert total == pytest.approx(1.0, abs=1e-4)


def test_run_inference_top1_matches_topk_first(tmp_path: Path) -> None:
    from PIL import Image as PILImage
    import numpy as np

    img_path = tmp_path / "crop.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB").save(img_path)

    loaded = _make_loaded_task(num_classes=5)
    result = _run_inference(loaded, img_path, topk=3, device=torch.device("cpu"))
    assert result["label"] == result["topk"][0]["label"]
    assert result["score"] == result["topk"][0]["score"]


# ---------------------------------------------------------------------------
# AttributePipelineConfig defaults
# ---------------------------------------------------------------------------


def test_attribute_pipeline_config_defaults() -> None:
    cfg = AttributePipelineConfig()
    assert cfg.device == "auto"
    assert cfg.topk == 3
    assert cfg.inference_config_path.name == "attribute_inference.yaml"
    assert cfg.group_mapping_path.name == "attribute_group_mapping.yaml"


def test_attribute_pipeline_config_custom_device() -> None:
    cfg = AttributePipelineConfig(device="cpu")
    assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# GarmentAttributePipeline — with monkeypatched model loading
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline():
    """A real pipeline instance using the standard config files."""
    cfg = AttributePipelineConfig(
        inference_config_path=_INFERENCE_YAML,
        group_mapping_path=_GROUP_MAPPING_YAML,
        device="cpu",
        topk=3,
    )
    return GarmentAttributePipeline(cfg)


def test_pipeline_init_loads_inference_config(pipeline) -> None:
    assert len(pipeline._inference_config) == 8


def test_pipeline_init_loads_mapping(pipeline) -> None:
    assert "top" in pipeline._mapping.coarse_class_to_tasks


def test_pipeline_init_empty_task_cache(pipeline) -> None:
    assert pipeline._task_cache == {}


def test_predict_instance_unknown_class_returns_empty(pipeline) -> None:
    crops = [_make_crop_record(class_name="hat", success=True)]
    result = pipeline.predict_instance(crops)
    assert result == {}


def test_predict_instance_no_successful_crops_returns_empty(pipeline) -> None:
    crops = [_make_crop_record(class_name="long sleeve top", success=False)]
    result = pipeline.predict_instance(crops)
    assert result == {}


def test_predict_instance_empty_crops_returns_empty(pipeline) -> None:
    result = pipeline.predict_instance([])
    assert result == {}


def test_predict_instance_with_mock_model(pipeline, tmp_path) -> None:
    """Monkeypatch _get_tasks_for_class to avoid loading real checkpoints."""
    from PIL import Image as PILImage
    import numpy as np

    # Write a synthetic crop image.
    crop_file = tmp_path / "collar_crop.jpg"
    PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB").save(crop_file)

    loaded = _make_loaded_task(num_classes=5)

    # Stub returns a fake LoadedTask for collar_design only.
    def fake_get_tasks(coarse_class_name: str) -> dict:
        return {"collar_design": loaded}

    pipeline._get_tasks_for_class = fake_get_tasks

    crops = [
        _make_crop_record(
            class_name="long sleeve top",
            region="collar",
            success=True,
            upper_crop_path=str(crop_file),
            expanded_crop_path=str(crop_file),
        )
    ]

    result = pipeline.predict_instance(crops)

    # collar_design uses upper_crop type and collar region — should produce a prediction.
    assert "collar_design" in result
    assert "label" in result["collar_design"]
    assert "score" in result["collar_design"]
    assert "topk" in result["collar_design"]


def test_predict_instance_missing_crop_file_skips_task(pipeline, tmp_path) -> None:
    """Task is skipped gracefully when the crop file does not exist."""
    loaded = _make_loaded_task(num_classes=5)

    def fake_get_tasks(coarse_class_name: str) -> dict:
        return {"collar_design": loaded}

    pipeline._get_tasks_for_class = fake_get_tasks

    crops = [
        _make_crop_record(
            class_name="long sleeve top",
            region="collar",
            success=True,
            upper_crop_path="/nonexistent/crop.jpg",
            expanded_crop_path="/nonexistent/crop.jpg",
        )
    ]

    result = pipeline.predict_instance(crops)
    assert "collar_design" not in result


def test_predict_instance_preserves_original_dict(pipeline) -> None:
    """predict_instance must not mutate the input crop records."""
    crops = [_make_crop_record(class_name="long sleeve top", success=True)]
    original_snapshot = dict(crops[0])
    with patch.object(pipeline, "_get_tasks_for_class", return_value={}):
        pipeline.predict_instance(crops)
    assert crops[0] == original_snapshot


# ---------------------------------------------------------------------------
# predict_from_json — JSON grouping and structure
# ---------------------------------------------------------------------------


def _write_region_crops_json(
    tmp_path: Path,
    crops: list[dict[str, Any]],
) -> Path:
    p = tmp_path / "region_crops.json"
    p.write_text(json.dumps({"crops": crops}), encoding="utf-8")
    return p


def test_predict_from_json_missing_file_raises(pipeline, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.predict_from_json(tmp_path / "nonexistent.json")


def test_predict_from_json_wrong_crops_type_raises(pipeline, tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"crops": "not a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        pipeline.predict_from_json(p)


def test_predict_from_json_groups_by_det_id(pipeline, tmp_path) -> None:
    crops = [
        _make_crop_record(det_id="det0", class_name="long sleeve top", region="collar"),
        _make_crop_record(det_id="det0", class_name="long sleeve top", region="sleeve"),
        _make_crop_record(det_id="det1", class_name="trousers", region="hem"),
    ]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    det_ids = [r["det_id"] for r in results]
    assert set(det_ids) == {"det0", "det1"}


def test_predict_from_json_result_has_required_keys(pipeline, tmp_path) -> None:
    crops = [_make_crop_record(det_id="det0", class_name="long sleeve top")]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    assert len(results) == 1
    r = results[0]
    for key in ("det_id", "fine_class_name", "coarse_class_name", "num_crops", "attributes", "error"):
        assert key in r, f"Missing key: {key!r}"


def test_predict_from_json_coarse_class_inferred(pipeline, tmp_path) -> None:
    crops = [_make_crop_record(det_id="det0", class_name="trousers")]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    assert results[0]["coarse_class_name"] == "pants"


def test_predict_from_json_num_crops_correct(pipeline, tmp_path) -> None:
    crops = [
        _make_crop_record(det_id="det0"),
        _make_crop_record(det_id="det0"),
        _make_crop_record(det_id="det0"),
    ]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    assert results[0]["num_crops"] == 3


def test_predict_from_json_max_instances(pipeline, tmp_path) -> None:
    crops = [
        _make_crop_record(det_id=f"det{i}", class_name="long sleeve top")
        for i in range(10)
    ]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p, max_instances=3)
    assert len(results) == 3


def test_predict_from_json_empty_crops_returns_empty_list(pipeline, tmp_path) -> None:
    p = _write_region_crops_json(tmp_path, [])
    results = pipeline.predict_from_json(p)
    assert results == []


def test_predict_from_json_error_key_is_none_on_success(pipeline, tmp_path) -> None:
    crops = [_make_crop_record(det_id="det0", class_name="hat", success=True)]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    # Unknown class → attributes={}, but no exception → error=None
    assert results[0]["error"] is None


# ---------------------------------------------------------------------------
# predict_from_json — instance-key grouping (cross-image collision fix)
# ---------------------------------------------------------------------------


def test_predict_from_json_same_det_id_different_images_produce_separate_groups(
    pipeline, tmp_path
) -> None:
    """Same det_id from two different images must yield two separate instances."""
    crops = [
        _make_crop_record(
            det_id=0, image_path="assets/images/img001.jpg",
            class_name="long sleeve top", region="collar",
        ),
        _make_crop_record(
            det_id=0, image_path="assets/images/img002.jpg",
            class_name="trousers", region="hem",
        ),
    ]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    assert len(results) == 2
    det_ids = {r["det_id"] for r in results}
    assert "img001__det0" in det_ids
    assert "img002__det0" in det_ids


def test_predict_from_json_same_det_id_same_image_single_group(
    pipeline, tmp_path
) -> None:
    """Two crop records with the same det_id and image_path must stay in one group."""
    crops = [
        _make_crop_record(
            det_id=0, image_path="assets/images/img001.jpg",
            class_name="long sleeve top", region="collar",
        ),
        _make_crop_record(
            det_id=0, image_path="assets/images/img001.jpg",
            class_name="long sleeve top", region="sleeve",
        ),
    ]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    assert len(results) == 1
    assert results[0]["num_crops"] == 2
    assert results[0]["det_id"] == "img001__det0"


def test_predict_from_json_missing_image_path_does_not_crash(
    pipeline, tmp_path
) -> None:
    """Records without image_path fall back to raw det_id as the grouping key."""
    crops = [
        _make_crop_record(det_id="det0", class_name="long sleeve top"),
    ]
    p = _write_region_crops_json(tmp_path, crops)
    results = pipeline.predict_from_json(p)
    assert len(results) == 1
    assert results[0]["det_id"] == "det0"
    assert results[0]["error"] is None
