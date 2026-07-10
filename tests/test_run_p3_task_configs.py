"""Unit tests for build_task_configs() in scripts/run_p3_region_to_attribute_8tasks.py.

Verifies that the config-driven loader produces output identical to the former
hardcoded TASK_CONFIGS dict.  No GPU, no checkpoints, no dataset access required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "attribute_inference.yaml"
_GROUP_MAPPING_CONFIG = PROJECT_ROOT / "configs" / "attribute_group_mapping.yaml"


# ---------------------------------------------------------------------------
# Import build_task_configs from the script via importlib
# (scripts/ is not a package, so direct import needs a path-based loader)
# ---------------------------------------------------------------------------


def _load_run_p3_module():
    """Load run_p3_region_to_attribute_8tasks as a module object."""
    script_path = PROJECT_ROOT / "scripts" / "run_p3_region_to_attribute_8tasks.py"
    spec = importlib.util.spec_from_file_location("run_p3_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def task_configs() -> dict[str, dict[str, Any]]:
    """Load task configs from the real YAML files once per module."""
    mod = _load_run_p3_module()
    return mod.build_task_configs(_INFERENCE_CONFIG, _GROUP_MAPPING_CONFIG)


# ---------------------------------------------------------------------------
# Task names and completeness
# ---------------------------------------------------------------------------


_EXPECTED_TASKS = {
    "neckline_design",
    "collar_design",
    "neck_design",
    "lapel_design",
    "sleeve_length",
    "coat_length",
    "pant_length",
    "skirt_length",
}


def test_all_eight_tasks_present(task_configs: dict) -> None:
    assert set(task_configs.keys()) == _EXPECTED_TASKS


def test_each_task_has_required_keys(task_configs: dict) -> None:
    required = {"checkpoint", "label_map", "arch", "img_size",
                "region", "crop_input_type", "component_contains", "class_contains"}
    for task, cfg in task_configs.items():
        missing = required - cfg.keys()
        assert not missing, f"Task {task!r} is missing keys: {missing}"


# ---------------------------------------------------------------------------
# Model paths — verbatim from old TASK_CONFIGS
# ---------------------------------------------------------------------------


def test_neckline_design_checkpoint_uses_resnet18_seed2_not_multiview(
    task_configs: dict,
) -> None:
    """neckline_design uses the baseline checkpoint, not the multiview_v2 one."""
    ckpt = task_configs["neckline_design"]["checkpoint"]
    assert "resnet18_seed2" in ckpt
    assert "multiview_v2" not in ckpt


def test_multiview_v2_tasks_use_multiview_v2_pipeline_checkpoint(task_configs: dict) -> None:
    # sleeve/pant/skirt_length have multiview_v2_pipeline checkpoints.
    # collar/neck/lapel/coat fall back to baseline (multiview_v2 was never trained for them).
    multiview_tasks = {"sleeve_length", "pant_length", "skirt_length"}
    for task in multiview_tasks:
        ckpt = task_configs[task]["checkpoint"]
        assert "multiview_v2_pipeline" in ckpt, (
            f"Task {task!r} checkpoint expected to contain 'multiview_v2_pipeline', got {ckpt!r}"
        )


def test_baseline_only_tasks_do_not_use_multiview_v2_checkpoint(task_configs: dict) -> None:
    # These four tasks only have baseline (v1) checkpoints; multiview_v2 was not trained.
    baseline_tasks = {"collar_design", "neck_design", "lapel_design", "coat_length"}
    for task in baseline_tasks:
        ckpt = task_configs[task]["checkpoint"]
        assert "multiview_v2_pipeline" not in ckpt, (
            f"Task {task!r} unexpectedly references a multiview_v2_pipeline checkpoint: {ckpt!r}"
        )
        assert "resnet18_seed2" in ckpt, (
            f"Task {task!r} checkpoint expected to use resnet18_seed2 baseline: {ckpt!r}"
        )


def test_all_label_maps_point_to_fashionai_index(task_configs: dict) -> None:
    for task, cfg in task_configs.items():
        assert "fashionai_attribute_index" in cfg["label_map"], (
            f"Task {task!r} label_map path unexpected: {cfg['label_map']!r}"
        )


def test_all_arch_is_resnet18(task_configs: dict) -> None:
    for task, cfg in task_configs.items():
        assert cfg["arch"] == "resnet18", f"Task {task!r} arch: {cfg['arch']!r}"


def test_all_img_size_is_224(task_configs: dict) -> None:
    for task, cfg in task_configs.items():
        assert cfg["img_size"] == 224, f"Task {task!r} img_size: {cfg['img_size']}"


# ---------------------------------------------------------------------------
# Region filter — equivalent to old TASK_CONFIGS["region"]
# ---------------------------------------------------------------------------


def test_collar_tasks_use_region_collar(task_configs: dict) -> None:
    collar_tasks = ("neckline_design", "collar_design", "neck_design", "lapel_design")
    for task in collar_tasks:
        assert task_configs[task]["region"] == "collar", (
            f"Task {task!r} region: {task_configs[task]['region']!r}"
        )


def test_length_tasks_use_region_all(task_configs: dict) -> None:
    length_tasks = ("sleeve_length", "coat_length", "pant_length", "skirt_length")
    for task in length_tasks:
        assert task_configs[task]["region"] == "all", (
            f"Task {task!r} region: {task_configs[task]['region']!r}"
        )


# ---------------------------------------------------------------------------
# Crop input type — from attribute_group_mapping.yaml task_to_crop_type
# ---------------------------------------------------------------------------


def test_collar_tasks_use_upper_crop(task_configs: dict) -> None:
    collar_tasks = ("neckline_design", "collar_design", "neck_design", "lapel_design")
    for task in collar_tasks:
        assert task_configs[task]["crop_input_type"] == "upper_crop", (
            f"Task {task!r} crop_input_type: {task_configs[task]['crop_input_type']!r}"
        )


def test_length_tasks_use_expanded_crop(task_configs: dict) -> None:
    length_tasks = ("sleeve_length", "coat_length", "pant_length", "skirt_length")
    for task in length_tasks:
        assert task_configs[task]["crop_input_type"] == "expanded_crop", (
            f"Task {task!r} crop_input_type: {task_configs[task]['crop_input_type']!r}"
        )


# ---------------------------------------------------------------------------
# Component filter — from attribute_group_mapping.yaml task_to_component_filter
# ---------------------------------------------------------------------------


def test_sleeve_length_component_contains_sleeve(task_configs: dict) -> None:
    assert task_configs["sleeve_length"]["component_contains"] == "sleeve"


def test_pant_length_component_contains_pant(task_configs: dict) -> None:
    assert task_configs["pant_length"]["component_contains"] == "pant"


def test_collar_tasks_have_no_component_filter(task_configs: dict) -> None:
    for task in ("neckline_design", "collar_design", "neck_design", "lapel_design"):
        assert task_configs[task]["component_contains"] is None, (
            f"Task {task!r} unexpected component_contains: "
            f"{task_configs[task]['component_contains']!r}"
        )


def test_coat_length_has_no_component_filter(task_configs: dict) -> None:
    assert task_configs["coat_length"]["component_contains"] is None


def test_skirt_length_has_no_component_filter(task_configs: dict) -> None:
    assert task_configs["skirt_length"]["component_contains"] is None


# ---------------------------------------------------------------------------
# Class filter — bug-fix regression: coat_length must use "outwear" not "outerwear"
# ---------------------------------------------------------------------------


def test_coat_length_class_contains_outwear(task_configs: dict) -> None:
    """Regression: bug was 'outerwear'; correct value is 'outwear'."""
    assert task_configs["coat_length"]["class_contains"] == "outwear"


def test_coat_length_class_contains_is_not_outerwear(task_configs: dict) -> None:
    """Explicit check that the old wrong value is absent."""
    assert task_configs["coat_length"]["class_contains"] != "outerwear"


def test_skirt_length_class_contains_skirt(task_configs: dict) -> None:
    assert task_configs["skirt_length"]["class_contains"] == "skirt"


def test_collar_tasks_have_no_class_filter(task_configs: dict) -> None:
    for task in ("neckline_design", "collar_design", "neck_design", "lapel_design"):
        assert task_configs[task]["class_contains"] is None, (
            f"Task {task!r} unexpected class_contains: "
            f"{task_configs[task]['class_contains']!r}"
        )


def test_sleeve_length_has_no_class_filter(task_configs: dict) -> None:
    assert task_configs["sleeve_length"]["class_contains"] is None


def test_pant_length_has_no_class_filter(task_configs: dict) -> None:
    assert task_configs["pant_length"]["class_contains"] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_inference_config_raises_file_not_found() -> None:
    mod = _load_run_p3_module()
    with pytest.raises(FileNotFoundError, match="attribute_inference"):
        mod.build_task_configs(
            Path("/nonexistent/attribute_inference.yaml"),
            _GROUP_MAPPING_CONFIG,
        )


def test_missing_group_mapping_raises_file_not_found() -> None:
    mod = _load_run_p3_module()
    with pytest.raises(FileNotFoundError):
        mod.build_task_configs(
            _INFERENCE_CONFIG,
            Path("/nonexistent/attribute_group_mapping.yaml"),
        )
