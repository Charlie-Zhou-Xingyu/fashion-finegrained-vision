"""Per-task model registry for PRD 3.1.3 attribute classification inference.

Provides config loading, model instantiation, checkpoint loading, and label-map
parsing for each FashionAI attribute task.  All inference-time transforms are
deterministic (no augmentation); training-time augmentation stays in the dataset.

Public API
----------
AttributeTaskConfig
    Frozen dataclass for per-task fields from ``attribute_inference.yaml``.

LoadedTask
    Live inference bundle: model (eval, on device) + label map + transform.

load_inference_config(yaml_path)
    Parse ``attribute_inference.yaml`` → ``dict[str, AttributeTaskConfig]``.

load_checkpoint_state(checkpoint_path, device)
    Load and return a cleaned model state dict.  Supports all four checkpoint
    formats used by this project.  Public so the latency benchmark can reuse it.

infer_num_classes_from_state(state)
    Inspect the state dict's final-layer weight shape to derive ``num_classes``.

build_inference_transform(img_size)
    Deterministic ``Resize → ToTensor → Normalize(ImageNet)`` pipeline.

load_id_to_label(label_map_path)
    Parse a label-map JSON into ``dict[int, str]``.  Supports all five formats
    used by this project's data scripts.

load_task(config, device)
    Combine the above into a :class:`LoadedTask`.

load_tasks_for_class(coarse_class_name, inference_config, mapping, device)
    Convenience loader: uses :func:`~fashion_vision.attributes.category_gate.get_enabled_tasks`
    then calls :func:`load_task` for each enabled task.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
from torch import nn
from torchvision import transforms
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so models/ is importable.
# (This module lives at src/fashion_vision/attributes/task_registry.py;
#  parents[3] resolves to the project root.)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from models.attribute_classifier import build_attribute_classifier  # noqa: E402
except ImportError as _exc:
    raise ImportError(
        "Cannot import build_attribute_classifier from models/attribute_classifier.py. "
        f"Ensure the project root ({_PROJECT_ROOT}) is on sys.path. "
        f"Original error: {_exc}"
    ) from _exc

# ImageNet normalisation constants — must match the training pipeline.
_IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributeTaskConfig:
    """Per-task configuration parsed from ``configs/attribute_inference.yaml``.

    All fields are read-only after construction.

    Attributes:
        task: Task name, e.g. ``"sleeve_length"``.
        checkpoint: Path to the model checkpoint, relative to project root.
        label_map: Path to the label-map JSON, relative to project root.
        arch: Model architecture name, e.g. ``"resnet18"``.
        img_size: Square input image size in pixels, e.g. ``224``.
    """

    task: str
    checkpoint: Path
    label_map: Path
    arch: str
    img_size: int


@dataclass
class LoadedTask:
    """Live inference bundle for one attribute task.

    Attributes:
        config: Source :class:`AttributeTaskConfig` for this task.
        model: Loaded PyTorch module in eval mode, on *device*.
        id_to_label: Mapping from contiguous training label id to label string.
        transform: Deterministic inference transform (no augmentation).
    """

    config: AttributeTaskConfig
    model: nn.Module
    id_to_label: dict[int, str]
    transform: transforms.Compose


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_inference_config(
    yaml_path: Union[str, Path],
) -> dict[str, AttributeTaskConfig]:
    """Parse ``configs/attribute_inference.yaml`` into per-task config objects.

    Args:
        yaml_path: Path to ``attribute_inference.yaml``
            (absolute or relative to CWD).

    Returns:
        Dict mapping task name → :class:`AttributeTaskConfig`.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        KeyError: If the YAML lacks a ``tasks`` section.
        ValueError: If any task entry is missing ``checkpoint`` or ``label_map``.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Inference config not found: {path}\n"
            "Ensure configs/attribute_inference.yaml exists at the project root."
        )

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    if "tasks" not in raw:
        raise KeyError(
            f"'tasks' section missing from {path}. "
            "Expected a YAML mapping of task names under the 'tasks' key."
        )

    result: dict[str, AttributeTaskConfig] = {}
    for task, entry in raw["tasks"].items():
        _validate_inference_entry(task, entry, path)
        result[task] = AttributeTaskConfig(
            task=task,
            checkpoint=Path(entry["checkpoint"]),
            label_map=Path(entry["label_map"]),
            arch=str(entry.get("arch", "resnet18")),
            img_size=int(entry.get("img_size", 224)),
        )
    return result


def _validate_inference_entry(
    task: str,
    entry: dict[str, Any],
    config_path: Path,
) -> None:
    """Raise ValueError if required per-task YAML fields are absent.

    Args:
        task: Task name string (for error messages).
        entry: Raw task dict parsed from the YAML.
        config_path: YAML file path (for error messages).

    Raises:
        ValueError: If ``checkpoint`` or ``label_map`` are missing.
    """
    for required_field in ("checkpoint", "label_map"):
        if required_field not in entry:
            raise ValueError(
                f"Task {task!r} in {config_path} is missing "
                f"required field '{required_field}'."
            )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint_state(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load and return the cleaned model state dict from a checkpoint file.

    Supports all four checkpoint formats produced by this project's training
    code:

    1. ``{"model_state_dict": {...}}``
    2. ``{"state_dict": {...}}``
    3. ``{"model": {...}}``
    4. A plain dict that is itself the state dict (bare format).

    DataParallel ``"module."`` key prefixes are stripped automatically.

    This function is **public** so that the latency benchmark
    (``tools/eval/benchmark_attribute_latency.py``) can reuse it without
    duplicating the loading logic.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Device to map tensors to during loading.

    Returns:
        Cleaned model state dict (``dict[str, torch.Tensor]``).

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        ValueError: If the checkpoint file is not a dict
            (unsupported format).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unsupported checkpoint format in {path}: "
            f"expected a dict, got {type(ckpt).__name__}. "
            "Supported formats: model_state_dict wrapper, state_dict wrapper, "
            "model wrapper, or a bare state dict."
        )

    # Extract inner state dict from the wrapper, or use the dict directly.
    if "model_state_dict" in ckpt:
        raw_state: dict[str, Any] = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        raw_state = ckpt["state_dict"]
    elif "model" in ckpt:
        raw_state = ckpt["model"]
    else:
        raw_state = ckpt  # bare state dict

    # Strip DataParallel "module." prefix from all keys.
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        clean_key = key[len("module."):] if key.startswith("module.") else key
        state[clean_key] = value

    return state


def infer_num_classes_from_state(
    state: dict[str, torch.Tensor],
) -> int | None:
    """Try to infer ``num_classes`` from the final-layer weight shape.

    Inspects common final-layer key names for supported architectures:

    * ``fc.weight`` — ResNet (resnet18, resnet34, …)
    * ``classifier.weight`` — VGG, EfficientNet
    * ``head.weight`` — ViT, Swin Transformer

    Returns ``None`` if no recognized key is found (caller must handle this).

    Args:
        state: Cleaned model state dict from :func:`load_checkpoint_state`.

    Returns:
        Number of output classes (first dimension of the weight tensor),
        or ``None`` if no recognized key is found.
    """
    for key in ("fc.weight", "classifier.weight", "head.weight"):
        if key in state:
            return int(state[key].shape[0])
    return None


# ---------------------------------------------------------------------------
# Label-map loading
# ---------------------------------------------------------------------------


def _parse_label_map_dict(data: dict[str, Any]) -> dict[int, str]:
    """Parse an already-loaded label-map dict into ``{int: str}``.

    Supports the five formats produced by this project's data scripts,
    mirroring ``predict_region_attribute_batch.py:load_label_map`` exactly:

    1. ``{"id_to_label":  {"0": "Invisible", ...}}``
    2. ``{"idx_to_label": {"0": "Invisible", ...}}``
    3. ``{"classes": ["Invisible", "Short Length", ...]}``  (list → enumerate)
    4. Plain dict with all-digit string keys: ``{"0": "Invisible", ...}``
    5. ``{"label_to_id": {"Invisible": 0, ...}}``  (inverted)

    Args:
        data: Dict loaded from a label-map JSON file.

    Returns:
        Mapping from integer label id to label string.

    Raises:
        ValueError: If none of the five formats match.
    """
    # Format 1: id_to_label dict
    if "id_to_label" in data:
        return {int(k): str(v) for k, v in data["id_to_label"].items()}

    # Format 2: idx_to_label dict
    if "idx_to_label" in data:
        return {int(k): str(v) for k, v in data["idx_to_label"].items()}

    # Format 3: classes list
    if "classes" in data and isinstance(data["classes"], list):
        return {i: str(name) for i, name in enumerate(data["classes"])}

    # Format 4: plain dict with all-digit string keys
    if data and all(str(k).isdigit() for k in data.keys()):
        return {int(k): str(v) for k, v in data.items()}

    # Format 5: label_to_id dict (inverted)
    if "label_to_id" in data:
        label_to_id: dict[str, int] = {
            str(k): int(v) for k, v in data["label_to_id"].items()
        }
        return {v: k for k, v in label_to_id.items()}

    raise ValueError(
        "Unsupported label-map format. Recognized top-level keys: "
        "'id_to_label', 'idx_to_label', 'classes', 'label_to_id', "
        "or a plain dict with all-digit string keys. "
        f"Got keys: {sorted(str(k) for k in data.keys())}"
    )


def load_id_to_label(label_map_path: Union[str, Path]) -> dict[int, str]:
    """Load a label-map JSON file and return ``{int: str}``.

    Delegates format detection to :func:`_parse_label_map_dict`.

    Args:
        label_map_path: Path to the label-map JSON file.

    Returns:
        Mapping from integer label id to label string.

    Raises:
        FileNotFoundError: If *label_map_path* does not exist.
        ValueError: If the JSON format is not recognized.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(label_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    return _parse_label_map_dict(data)


# ---------------------------------------------------------------------------
# Inference transform
# ---------------------------------------------------------------------------


def build_inference_transform(img_size: int = 224) -> transforms.Compose:
    """Build the deterministic inference transform.

    Matches the validation/test transform used during training:
    ``Resize((img_size, img_size)) → ToTensor → Normalize(ImageNet)``.
    No random augmentation is applied.

    Args:
        img_size: Square output size in pixels.  Default ``224``.

    Returns:
        ``torchvision.transforms.Compose`` pipeline.

    Raises:
        ValueError: If *img_size* is not positive.
    """
    if img_size <= 0:
        raise ValueError(f"img_size must be positive, got {img_size}.")
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
    ])


# ---------------------------------------------------------------------------
# Task loader
# ---------------------------------------------------------------------------


def load_task(
    config: AttributeTaskConfig,
    device: torch.device,
) -> LoadedTask:
    """Load a single attribute task: label map → model → checkpoint → transform.

    ``num_classes`` is derived from ``len(id_to_label)`` after loading the
    label map, which is always authoritative for contiguous training ids.

    Args:
        config: :class:`AttributeTaskConfig` specifying paths and arch.
        device: Target device for the model.

    Returns:
        :class:`LoadedTask` with model in eval mode on *device*.

    Raises:
        FileNotFoundError: If checkpoint or label-map path does not exist.
        ValueError: If the label-map format is unrecognized or empty.
        RuntimeError: If ``model.load_state_dict()`` fails (e.g. shape mismatch).
    """
    id_to_label = load_id_to_label(config.label_map)
    num_classes = len(id_to_label)
    if num_classes == 0:
        raise ValueError(
            f"Label map for task {config.task!r} is empty: {config.label_map}"
        )

    model = build_attribute_classifier(
        arch=config.arch,
        num_classes=num_classes,
        pretrained=False,
    )

    state = load_checkpoint_state(config.checkpoint, device)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint for task {config.task!r} from "
            f"{config.checkpoint}. "
            "This usually means the checkpoint architecture does not match "
            f"arch={config.arch!r} with num_classes={num_classes}. "
            f"Original error: {exc}"
        ) from exc

    model.to(device)
    model.eval()

    return LoadedTask(
        config=config,
        model=model,
        id_to_label=id_to_label,
        transform=build_inference_transform(config.img_size),
    )


def load_tasks_for_class(
    coarse_class_name: str,
    inference_config: dict[str, AttributeTaskConfig],
    mapping: Any,
    device: torch.device,
) -> dict[str, LoadedTask]:
    """Load all attribute task models enabled for a coarse garment class.

    Queries :func:`~fashion_vision.attributes.category_gate.get_enabled_tasks`
    to determine the task list, then calls :func:`load_task` for each task
    present in *inference_config*.

    Args:
        coarse_class_name: PRD coarse class name, e.g. ``"top"``, ``"dress"``.
        inference_config: Pre-loaded config dict from :func:`load_inference_config`.
        mapping: Loaded :class:`~fashion_vision.attributes.category_gate.AttributeGroupMapping`.
        device: Target device for all loaded models.

    Returns:
        Dict mapping task name → :class:`LoadedTask`.
        Tasks not present in *inference_config* are skipped with a warning.
    """
    # Local import avoids a circular import at module level.
    from fashion_vision.attributes.category_gate import get_enabled_tasks

    loaded: dict[str, LoadedTask] = {}
    for task in get_enabled_tasks(coarse_class_name, mapping):
        if task not in inference_config:
            logger.warning(
                "Task %r is enabled for class %r but has no entry in "
                "attribute_inference.yaml — skipping.",
                task,
                coarse_class_name,
            )
            continue
        loaded[task] = load_task(inference_config[task], device)
    return loaded
