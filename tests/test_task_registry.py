"""Unit tests for src/fashion_vision/attributes/task_registry.py.

All tests are designed to run without real checkpoints, a GPU, or dataset access.
Tests that exercise load_task() / load_tasks_for_class() are excluded because
they require real checkpoint files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torchvision import transforms

from fashion_vision.attributes.task_registry import (
    AttributeTaskConfig,
    LoadedTask,
    _parse_label_map_dict,
    build_inference_transform,
    infer_num_classes_from_state,
    load_checkpoint_state,
    load_id_to_label,
    load_inference_config,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_YAML = _PROJECT_ROOT / "configs" / "attribute_inference.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_TASKS = {
    "neckline_design", "collar_design", "neck_design", "lapel_design",
    "sleeve_length", "coat_length", "pant_length", "skirt_length",
}


# ---------------------------------------------------------------------------
# load_inference_config
# ---------------------------------------------------------------------------


def test_load_inference_config_returns_dict() -> None:
    cfg = load_inference_config(_INFERENCE_YAML)
    assert isinstance(cfg, dict)


def test_load_inference_config_all_eight_tasks_present() -> None:
    cfg = load_inference_config(_INFERENCE_YAML)
    assert set(cfg.keys()) == _ALL_TASKS


def test_load_inference_config_values_are_attribute_task_config() -> None:
    cfg = load_inference_config(_INFERENCE_YAML)
    for task, task_cfg in cfg.items():
        assert isinstance(task_cfg, AttributeTaskConfig), (
            f"Task {task!r} value is {type(task_cfg).__name__}"
        )


def test_load_inference_config_task_name_matches_key() -> None:
    cfg = load_inference_config(_INFERENCE_YAML)
    for key, task_cfg in cfg.items():
        assert task_cfg.task == key


def test_load_inference_config_arch_is_resnet18() -> None:
    cfg = load_inference_config(_INFERENCE_YAML)
    for task, task_cfg in cfg.items():
        assert task_cfg.arch == "resnet18", f"Task {task!r} arch: {task_cfg.arch!r}"


def test_load_inference_config_img_size_is_224() -> None:
    cfg = load_inference_config(_INFERENCE_YAML)
    for task, task_cfg in cfg.items():
        assert task_cfg.img_size == 224, f"Task {task!r} img_size: {task_cfg.img_size}"


def test_load_inference_config_checkpoint_and_label_map_are_paths() -> None:
    cfg = load_inference_config(_INFERENCE_YAML)
    for task, task_cfg in cfg.items():
        assert isinstance(task_cfg.checkpoint, Path)
        assert isinstance(task_cfg.label_map, Path)


def test_load_inference_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_inference_config("/nonexistent/attribute_inference.yaml")


def test_load_inference_config_missing_tasks_section_raises(
    tmp_path: Path,
) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("version: '1.0'\n", encoding="utf-8")
    with pytest.raises(KeyError, match="tasks"):
        load_inference_config(bad_yaml)


def test_load_inference_config_missing_checkpoint_raises(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "tasks:\n  mytask:\n    label_map: some/path.json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint"):
        load_inference_config(bad_yaml)


def test_load_inference_config_missing_label_map_raises(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "tasks:\n  mytask:\n    checkpoint: some/path.pt\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label_map"):
        load_inference_config(bad_yaml)


# ---------------------------------------------------------------------------
# AttributeTaskConfig is frozen
# ---------------------------------------------------------------------------


def test_attribute_task_config_is_immutable() -> None:
    cfg = AttributeTaskConfig(
        task="test",
        checkpoint=Path("a/b.pt"),
        label_map=Path("a/c.json"),
        arch="resnet18",
        img_size=224,
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.task = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_inference_transform
# ---------------------------------------------------------------------------


def test_build_inference_transform_returns_compose() -> None:
    t = build_inference_transform(224)
    assert isinstance(t, transforms.Compose)


def test_build_inference_transform_has_three_stages() -> None:
    t = build_inference_transform(224)
    assert len(t.transforms) == 3


def test_build_inference_transform_first_is_resize() -> None:
    t = build_inference_transform(112)
    assert isinstance(t.transforms[0], transforms.Resize)


def test_build_inference_transform_second_is_to_tensor() -> None:
    t = build_inference_transform(224)
    assert isinstance(t.transforms[1], transforms.ToTensor)


def test_build_inference_transform_third_is_normalize() -> None:
    t = build_inference_transform(224)
    assert isinstance(t.transforms[2], transforms.Normalize)


def test_build_inference_transform_normalize_imagenet_mean() -> None:
    t = build_inference_transform(224)
    norm: transforms.Normalize = t.transforms[2]
    assert list(norm.mean) == pytest.approx([0.485, 0.456, 0.406])


def test_build_inference_transform_normalize_imagenet_std() -> None:
    t = build_inference_transform(224)
    norm: transforms.Normalize = t.transforms[2]
    assert list(norm.std) == pytest.approx([0.229, 0.224, 0.225])


def test_build_inference_transform_zero_size_raises() -> None:
    with pytest.raises(ValueError, match="img_size"):
        build_inference_transform(0)


def test_build_inference_transform_negative_size_raises() -> None:
    with pytest.raises(ValueError, match="img_size"):
        build_inference_transform(-1)


def test_build_inference_transform_output_tensor_shape() -> None:
    """Apply transform to a synthetic PIL image and check output shape."""
    from PIL import Image as PILImage
    import numpy as np

    t = build_inference_transform(224)
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# _parse_label_map_dict — all five formats
# ---------------------------------------------------------------------------


def test_parse_format1_id_to_label() -> None:
    data = {"id_to_label": {"0": "Invisible", "1": "Short Length"}}
    result = _parse_label_map_dict(data)
    assert result == {0: "Invisible", 1: "Short Length"}


def test_parse_format2_idx_to_label() -> None:
    data = {"idx_to_label": {"0": "Invisible", "1": "Short Length"}}
    result = _parse_label_map_dict(data)
    assert result == {0: "Invisible", 1: "Short Length"}


def test_parse_format3_classes_list() -> None:
    data = {"classes": ["Invisible", "Short Length", "Knee Length"]}
    result = _parse_label_map_dict(data)
    assert result == {0: "Invisible", 1: "Short Length", 2: "Knee Length"}


def test_parse_format4_plain_digit_keys() -> None:
    data = {"0": "Invisible", "1": "Short Length", "2": "Knee Length"}
    result = _parse_label_map_dict(data)
    assert result == {0: "Invisible", 1: "Short Length", 2: "Knee Length"}


def test_parse_format5_label_to_id() -> None:
    data = {"label_to_id": {"Invisible": 0, "Short Length": 1, "Knee Length": 2}}
    result = _parse_label_map_dict(data)
    assert result == {0: "Invisible", 1: "Short Length", 2: "Knee Length"}


def test_parse_unsupported_format_raises() -> None:
    data = {"something_else": {"a": 1}}
    with pytest.raises(ValueError, match="Unsupported label-map format"):
        _parse_label_map_dict(data)


def test_parse_format1_takes_priority_over_digit_keys() -> None:
    """id_to_label key takes priority over the plain digit-keys fallback."""
    data = {"id_to_label": {"0": "A", "1": "B"}, "0": "IGNORED"}
    result = _parse_label_map_dict(data)
    assert result[0] == "A"


# ---------------------------------------------------------------------------
# load_id_to_label — file-based tests using tmp_path
# ---------------------------------------------------------------------------


def test_load_id_to_label_id_to_label_format(tmp_path: Path) -> None:
    p = tmp_path / "lm.json"
    p.write_text(json.dumps({"id_to_label": {"0": "A", "1": "B"}}), encoding="utf-8")
    assert load_id_to_label(p) == {0: "A", 1: "B"}


def test_load_id_to_label_classes_list_format(tmp_path: Path) -> None:
    p = tmp_path / "lm.json"
    p.write_text(json.dumps({"classes": ["A", "B", "C"]}), encoding="utf-8")
    assert load_id_to_label(p) == {0: "A", 1: "B", 2: "C"}


def test_load_id_to_label_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_id_to_label(Path("/nonexistent/label_map.json"))


# ---------------------------------------------------------------------------
# load_checkpoint_state — synthetic checkpoints in tmp_path
# ---------------------------------------------------------------------------


def _make_fake_state() -> dict[str, torch.Tensor]:
    return {
        "fc.weight": torch.zeros(5, 512),
        "fc.bias": torch.zeros(5),
        "layer1.weight": torch.zeros(64, 3, 3, 3),
    }


def test_load_checkpoint_state_model_state_dict_wrapper(tmp_path: Path) -> None:
    fake = _make_fake_state()
    p = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": fake, "epoch": 10}, p)
    state = load_checkpoint_state(p, torch.device("cpu"))
    assert "fc.weight" in state
    assert state["fc.weight"].shape == (5, 512)


def test_load_checkpoint_state_dict_wrapper(tmp_path: Path) -> None:
    fake = _make_fake_state()
    p = tmp_path / "ckpt.pt"
    torch.save({"state_dict": fake}, p)
    state = load_checkpoint_state(p, torch.device("cpu"))
    assert "fc.weight" in state


def test_load_checkpoint_state_model_wrapper(tmp_path: Path) -> None:
    fake = _make_fake_state()
    p = tmp_path / "ckpt.pt"
    torch.save({"model": fake}, p)
    state = load_checkpoint_state(p, torch.device("cpu"))
    assert "fc.weight" in state


def test_load_checkpoint_state_bare_dict(tmp_path: Path) -> None:
    fake = _make_fake_state()
    p = tmp_path / "ckpt.pt"
    torch.save(fake, p)
    state = load_checkpoint_state(p, torch.device("cpu"))
    assert "fc.weight" in state


def test_load_checkpoint_state_strips_module_prefix(tmp_path: Path) -> None:
    fake = {
        "module.fc.weight": torch.zeros(5, 512),
        "module.fc.bias": torch.zeros(5),
    }
    p = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": fake}, p)
    state = load_checkpoint_state(p, torch.device("cpu"))
    assert "fc.weight" in state
    assert "module.fc.weight" not in state


def test_load_checkpoint_state_non_dict_raises(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.pt"
    torch.save([1, 2, 3], p)  # a list, not a dict
    with pytest.raises(ValueError, match="expected a dict"):
        load_checkpoint_state(p, torch.device("cpu"))


def test_load_checkpoint_state_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint_state(Path("/nonexistent/ckpt.pt"), torch.device("cpu"))


# ---------------------------------------------------------------------------
# infer_num_classes_from_state
# ---------------------------------------------------------------------------


def test_infer_num_classes_from_fc_weight() -> None:
    state = {"fc.weight": torch.zeros(8, 512), "fc.bias": torch.zeros(8)}
    assert infer_num_classes_from_state(state) == 8


def test_infer_num_classes_from_classifier_weight() -> None:
    state = {"classifier.weight": torch.zeros(10, 1280)}
    assert infer_num_classes_from_state(state) == 10


def test_infer_num_classes_from_head_weight() -> None:
    state = {"head.weight": torch.zeros(5, 768)}
    assert infer_num_classes_from_state(state) == 5


def test_infer_num_classes_returns_none_when_no_key() -> None:
    state = {"layer1.weight": torch.zeros(64, 3, 3, 3)}
    assert infer_num_classes_from_state(state) is None


def test_infer_num_classes_fc_weight_takes_priority() -> None:
    """fc.weight is checked before classifier.weight."""
    state = {
        "fc.weight": torch.zeros(5, 512),
        "classifier.weight": torch.zeros(10, 1280),
    }
    assert infer_num_classes_from_state(state) == 5
