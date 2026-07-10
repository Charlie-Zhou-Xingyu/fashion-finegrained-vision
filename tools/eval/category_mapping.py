"""Utilities for loading and validating garment category mappings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CategoryMapping:
    """Container for DeepFashion2-to-PRD category mapping.

    Attributes:
        deepfashion2_13cls: Mapping from 13-class id to class name.
        prd_5cls: Mapping from 5-class id to class name.
        prd_5cls_zh: Mapping from 5-class id to Chinese display name.
        map_13_to_5: Mapping from DeepFashion2 13-class id to PRD 5-class id.
    """

    deepfashion2_13cls: dict[int, str]
    prd_5cls: dict[int, str]
    prd_5cls_zh: dict[int, str]
    map_13_to_5: dict[int, int]


def _to_int_key_dict(data: dict[Any, Any], value_type: type) -> dict[int, Any]:
    """Convert dictionary keys to integers and validate value types.

    Args:
        data: Source dictionary loaded from YAML.
        value_type: Expected Python type of dictionary values.

    Returns:
        A dictionary with integer keys.

    Raises:
        ValueError: If keys cannot be converted to integers.
        TypeError: If a value does not match the expected type.
    """
    result: dict[int, Any] = {}

    for key, value in data.items():
        try:
            int_key = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid non-integer mapping key: {key}") from exc

        if not isinstance(value, value_type):
            raise TypeError(
                f"Invalid value type for key {key}: "
                f"expected {value_type.__name__}, got {type(value).__name__}"
            )

        result[int_key] = value

    return result


def load_category_mapping(path: str | Path) -> CategoryMapping:
    """Load and validate category mapping YAML.

    Args:
        path: Path to the category mapping YAML file.

    Returns:
        A validated CategoryMapping object.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If required fields are missing.
        ValueError: If mapping ids are incomplete or invalid.
    """
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Category mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid mapping file format: {mapping_path}")

    required_keys = [
        "deepfashion2_13cls",
        "prd_5cls",
        "prd_5cls_zh",
        "map_13_to_5",
    ]
    for key in required_keys:
        if key not in raw:
            raise KeyError(f"Missing required mapping field: {key}")

    deepfashion2_13cls = _to_int_key_dict(raw["deepfashion2_13cls"], str)
    prd_5cls = _to_int_key_dict(raw["prd_5cls"], str)
    prd_5cls_zh = _to_int_key_dict(raw["prd_5cls_zh"], str)
    map_13_to_5 = _to_int_key_dict(raw["map_13_to_5"], int)

    _validate_category_mapping(
        deepfashion2_13cls=deepfashion2_13cls,
        prd_5cls=prd_5cls,
        prd_5cls_zh=prd_5cls_zh,
        map_13_to_5=map_13_to_5,
    )

    return CategoryMapping(
        deepfashion2_13cls=deepfashion2_13cls,
        prd_5cls=prd_5cls,
        prd_5cls_zh=prd_5cls_zh,
        map_13_to_5=map_13_to_5,
    )


def _validate_category_mapping(
    deepfashion2_13cls: dict[int, str],
    prd_5cls: dict[int, str],
    prd_5cls_zh: dict[int, str],
    map_13_to_5: dict[int, int],
) -> None:
    """Validate category mapping id coverage.

    Args:
        deepfashion2_13cls: DeepFashion2 13-class names.
        prd_5cls: PRD 5-class names.
        prd_5cls_zh: PRD 5-class Chinese names.
        map_13_to_5: Mapping from 13-class ids to 5-class ids.

    Raises:
        ValueError: If class ids are incomplete or invalid.
    """
    expected_13_ids = set(range(13))
    expected_5_ids = set(range(5))

    if set(deepfashion2_13cls.keys()) != expected_13_ids:
        raise ValueError("deepfashion2_13cls must contain ids 0-12.")

    if set(map_13_to_5.keys()) != expected_13_ids:
        raise ValueError("map_13_to_5 must contain ids 0-12.")

    if set(prd_5cls.keys()) != expected_5_ids:
        raise ValueError("prd_5cls must contain ids 0-4.")

    if set(prd_5cls_zh.keys()) != expected_5_ids:
        raise ValueError("prd_5cls_zh must contain ids 0-4.")

    mapped_values = set(map_13_to_5.values())
    invalid_values = mapped_values - expected_5_ids
    if invalid_values:
        raise ValueError(f"Invalid mapped 5-class ids: {sorted(invalid_values)}")
