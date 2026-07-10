"""Config-driven category-to-attribute-task gating for PRD 3.1.3.

Loads ``configs/attribute_group_mapping.yaml`` and exposes a small, pure API
for resolving which attribute tasks to run given a coarse garment class name.

No model loading is performed here; this module is config + logic only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

logger = logging.getLogger(__name__)

# The set of valid PRD 5-class coarse category names.
_VALID_COARSE_CLASSES: frozenset[str] = frozenset(
    {"top", "outerwear", "pants", "skirt", "dress"}
)


@dataclass(frozen=True)
class AttributeGroupMapping:
    """Parsed content of ``configs/attribute_group_mapping.yaml``.

    All fields are read-only after construction.

    Attributes:
        coarse_class_to_tasks: Maps coarse class name → ordered list of task names.
        task_to_region: Maps task name → garment region type (e.g. "collar", "hem").
        task_to_crop_type: Maps task name → crop_input_type string for inference.
        coarse_class_to_fine_substrings: Maps coarse class name → list of fine-class
            substrings used to filter records that lack ``coarse_class_name``.
        task_to_component_filter: Maps task name → component substring filter used
            to filter region-crop records by their ``component`` field.  Only tasks
            that require filtering appear here; absent tasks have no component filter.
        version: Version string from the YAML.
    """

    coarse_class_to_tasks: dict[str, list[str]]
    task_to_region: dict[str, str]
    task_to_crop_type: dict[str, str]
    coarse_class_to_fine_substrings: dict[str, list[str]]
    task_to_component_filter: dict[str, str]
    version: str = "1.0"


def load_attribute_group_mapping(path: Union[str, Path]) -> AttributeGroupMapping:
    """Load and validate ``attribute_group_mapping.yaml``.

    Args:
        path: Path to the YAML file (absolute or relative to CWD).

    Returns:
        Validated :class:`AttributeGroupMapping` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required sections are missing or task names are inconsistent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Attribute group mapping not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    _validate_raw_mapping(raw, path)

    return AttributeGroupMapping(
        coarse_class_to_tasks=raw["coarse_class_to_tasks"],
        task_to_region=raw["task_to_region"],
        task_to_crop_type=raw["task_to_crop_type"],
        coarse_class_to_fine_substrings=raw.get(
            "coarse_class_to_fine_class_substrings", {}
        ),
        task_to_component_filter=raw.get("task_to_component_filter", {}),
        version=str(raw.get("version", "1.0")),
    )


def _validate_raw_mapping(raw: dict, path: Path) -> None:
    """Validate structural consistency of the raw YAML dict.

    Args:
        raw: Parsed YAML content.
        path: Source path (used in error messages only).

    Raises:
        ValueError: On any structural inconsistency.
    """
    required_keys = {"coarse_class_to_tasks", "task_to_region", "task_to_crop_type"}
    missing = required_keys - raw.keys()
    if missing:
        raise ValueError(
            f"attribute_group_mapping.yaml ({path}) is missing sections: {missing}"
        )

    task_region_keys: set[str] = set(raw["task_to_region"].keys())
    task_crop_keys: set[str] = set(raw["task_to_crop_type"].keys())

    for coarse_cls, tasks in raw["coarse_class_to_tasks"].items():
        if coarse_cls not in _VALID_COARSE_CLASSES:
            logger.warning(
                "Unknown coarse class %r in attribute_group_mapping.yaml — "
                "expected one of %s",
                coarse_cls,
                sorted(_VALID_COARSE_CLASSES),
            )
        for task in tasks:
            if task not in task_region_keys:
                raise ValueError(
                    f"Task {task!r} (under coarse class {coarse_cls!r}) has no "
                    f"entry in task_to_region section of {path}."
                )
            if task not in task_crop_keys:
                raise ValueError(
                    f"Task {task!r} (under coarse class {coarse_cls!r}) has no "
                    f"entry in task_to_crop_type section of {path}."
                )


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------


def get_enabled_tasks(
    coarse_class_name: str,
    mapping: AttributeGroupMapping,
) -> list[str]:
    """Return the ordered list of attribute tasks enabled for a coarse class.

    Args:
        coarse_class_name: PRD coarse class, e.g. ``"top"``, ``"dress"``.
        mapping: Loaded :class:`AttributeGroupMapping`.

    Returns:
        List of task name strings in display order.  Empty list for unknown
        class names (never raises so callers can safely skip unknown classes).
    """
    tasks = mapping.coarse_class_to_tasks.get(coarse_class_name)
    if tasks is None:
        logger.debug("get_enabled_tasks: unknown coarse class %r → []", coarse_class_name)
        return []
    return list(tasks)


def get_region_for_task(task: str, mapping: AttributeGroupMapping) -> str:
    """Return the garment region type required by *task*.

    Args:
        task: Attribute task name, e.g. ``"sleeve_length"``.
        mapping: Loaded :class:`AttributeGroupMapping`.

    Returns:
        Region type string, e.g. ``"collar"``, ``"hem"``.

    Raises:
        KeyError: If *task* is not present in the mapping.
    """
    try:
        return mapping.task_to_region[task]
    except KeyError:
        raise KeyError(
            f"Task {task!r} not found in task_to_region. "
            f"Available tasks: {sorted(mapping.task_to_region)}"
        ) from None


def get_crop_type_for_task(task: str, mapping: AttributeGroupMapping) -> str:
    """Return the preferred crop input type for *task* at inference time.

    Args:
        task: Attribute task name.
        mapping: Loaded :class:`AttributeGroupMapping`.

    Returns:
        Crop type string matching ``crop_input_type`` choices in
        ``predict_region_attribute_batch.py``.

    Raises:
        KeyError: If *task* is not present in the mapping.
    """
    try:
        return mapping.task_to_crop_type[task]
    except KeyError:
        raise KeyError(
            f"Task {task!r} not found in task_to_crop_type. "
            f"Available tasks: {sorted(mapping.task_to_crop_type)}"
        ) from None


def get_fine_class_filter(
    coarse_class_name: str,
    mapping: AttributeGroupMapping,
) -> list[str]:
    """Return fine-class substrings for backward-compatible record filtering.

    Use these substrings to filter region-crop records whose ``fine_class_name``
    field (13-class DeepFashion2) is available but ``coarse_class_name`` is not.

    Args:
        coarse_class_name: PRD coarse class name.
        mapping: Loaded :class:`AttributeGroupMapping`.

    Returns:
        List of substrings.  Empty list for unknown class names.
    """
    return list(mapping.coarse_class_to_fine_substrings.get(coarse_class_name, []))


def get_component_filter_for_task(
    task: str,
    mapping: AttributeGroupMapping,
) -> str | None:
    """Return the component substring filter for *task*, or ``None`` if not set.

    The returned value maps directly to the ``--component-contains`` CLI argument
    of ``predict_region_attribute_batch.py``.  A ``None`` return means no
    component filtering should be applied (pass an empty string or omit the flag).

    Args:
        task: Attribute task name, e.g. ``"sleeve_length"``.
        mapping: Loaded :class:`AttributeGroupMapping`.

    Returns:
        Component substring string (e.g. ``"sleeve"``), or ``None`` if the task
        requires no component filtering.
    """
    return mapping.task_to_component_filter.get(task)
