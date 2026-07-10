"""Unit tests for src/fashion_vision/attributes/category_gate.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from fashion_vision.attributes.category_gate import (
    AttributeGroupMapping,
    get_component_filter_for_task,
    get_crop_type_for_task,
    get_enabled_tasks,
    get_fine_class_filter,
    get_region_for_task,
    load_attribute_group_mapping,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MAPPING_YAML = _PROJECT_ROOT / "configs" / "attribute_group_mapping.yaml"


@pytest.fixture(scope="module")
def mapping() -> AttributeGroupMapping:
    """Load the real attribute_group_mapping.yaml once per test module."""
    return load_attribute_group_mapping(_MAPPING_YAML)


# ---------------------------------------------------------------------------
# load_attribute_group_mapping
# ---------------------------------------------------------------------------


def test_load_returns_attribute_group_mapping(mapping: AttributeGroupMapping) -> None:
    assert isinstance(mapping, AttributeGroupMapping)


def test_load_version_is_string(mapping: AttributeGroupMapping) -> None:
    assert isinstance(mapping.version, str)
    assert mapping.version  # non-empty


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_attribute_group_mapping("/nonexistent/path/attribute_group_mapping.yaml")


# ---------------------------------------------------------------------------
# get_enabled_tasks
# ---------------------------------------------------------------------------


def test_top_has_four_tasks(mapping: AttributeGroupMapping) -> None:
    tasks = get_enabled_tasks("top", mapping)
    assert tasks == ["neckline_design", "collar_design", "neck_design", "sleeve_length"]


def test_pants_has_one_task(mapping: AttributeGroupMapping) -> None:
    tasks = get_enabled_tasks("pants", mapping)
    assert tasks == ["pant_length"]


def test_skirt_has_one_task(mapping: AttributeGroupMapping) -> None:
    tasks = get_enabled_tasks("skirt", mapping)
    assert tasks == ["skirt_length"]


def test_outerwear_has_three_tasks(mapping: AttributeGroupMapping) -> None:
    tasks = get_enabled_tasks("outerwear", mapping)
    assert "lapel_design" in tasks
    assert "coat_length" in tasks
    assert "sleeve_length" in tasks


def test_dress_tasks_present(mapping: AttributeGroupMapping) -> None:
    tasks = get_enabled_tasks("dress", mapping)
    assert "neckline_design" in tasks
    assert "skirt_length" in tasks


def test_unknown_class_returns_empty_list(mapping: AttributeGroupMapping) -> None:
    assert get_enabled_tasks("shoes", mapping) == []
    assert get_enabled_tasks("", mapping) == []
    assert get_enabled_tasks("UNKNOWN", mapping) == []


def test_all_enabled_tasks_are_strings(mapping: AttributeGroupMapping) -> None:
    for coarse_cls in mapping.coarse_class_to_tasks:
        for task in get_enabled_tasks(coarse_cls, mapping):
            assert isinstance(task, str)


# ---------------------------------------------------------------------------
# get_region_for_task
# ---------------------------------------------------------------------------


def test_coat_length_region_is_hem(mapping: AttributeGroupMapping) -> None:
    assert get_region_for_task("coat_length", mapping) == "hem"


def test_collar_tasks_region_is_collar(mapping: AttributeGroupMapping) -> None:
    for task in ("neckline_design", "collar_design", "neck_design", "lapel_design"):
        assert get_region_for_task(task, mapping) == "collar"


def test_pant_length_region(mapping: AttributeGroupMapping) -> None:
    assert get_region_for_task("pant_length", mapping) == "leg_opening"


def test_skirt_length_region_is_hem(mapping: AttributeGroupMapping) -> None:
    assert get_region_for_task("skirt_length", mapping) == "hem"


def test_region_for_unknown_task_raises_key_error(mapping: AttributeGroupMapping) -> None:
    with pytest.raises(KeyError):
        get_region_for_task("nonexistent_task", mapping)


# ---------------------------------------------------------------------------
# get_crop_type_for_task
# ---------------------------------------------------------------------------


def test_collar_tasks_use_upper_crop(mapping: AttributeGroupMapping) -> None:
    for task in ("neckline_design", "collar_design", "neck_design", "lapel_design"):
        assert get_crop_type_for_task(task, mapping) == "upper_crop"


def test_length_tasks_use_expanded_crop(mapping: AttributeGroupMapping) -> None:
    for task in ("sleeve_length", "coat_length", "pant_length", "skirt_length"):
        assert get_crop_type_for_task(task, mapping) == "expanded_crop"


def test_crop_type_for_unknown_task_raises_key_error(mapping: AttributeGroupMapping) -> None:
    with pytest.raises(KeyError):
        get_crop_type_for_task("nonexistent_task", mapping)


# ---------------------------------------------------------------------------
# get_fine_class_filter — critical: "outwear" not "outerwear"
# ---------------------------------------------------------------------------


def test_outerwear_filter_contains_outwear(mapping: AttributeGroupMapping) -> None:
    substrings = get_fine_class_filter("outerwear", mapping)
    assert "outwear" in substrings


def test_outerwear_filter_does_not_contain_outerwear_with_er(
    mapping: AttributeGroupMapping,
) -> None:
    """Regression: the bug used 'outerwear'; correct substring is 'outwear'."""
    substrings = get_fine_class_filter("outerwear", mapping)
    assert "outerwear" not in substrings


def test_outwear_substring_matches_fine_class_names(mapping: AttributeGroupMapping) -> None:
    """'outwear' must match both fine class names from DeepFashion2."""
    substrings = get_fine_class_filter("outerwear", mapping)
    fine_classes = ["short sleeve outwear", "long sleeve outwear"]
    for name in fine_classes:
        assert any(s in name for s in substrings), (
            f"Fine class {name!r} not matched by any substring in {substrings}"
        )


def test_top_filter_includes_vest_and_sling(mapping: AttributeGroupMapping) -> None:
    substrings = get_fine_class_filter("top", mapping)
    assert "vest" in substrings
    assert "sling" in substrings


def test_unknown_coarse_class_returns_empty_filter(mapping: AttributeGroupMapping) -> None:
    assert get_fine_class_filter("shoes", mapping) == []
    assert get_fine_class_filter("unknown", mapping) == []


# ---------------------------------------------------------------------------
# get_component_filter_for_task
# ---------------------------------------------------------------------------


def test_sleeve_length_has_component_filter(mapping: AttributeGroupMapping) -> None:
    assert get_component_filter_for_task("sleeve_length", mapping) == "sleeve"


def test_pant_length_has_component_filter(mapping: AttributeGroupMapping) -> None:
    assert get_component_filter_for_task("pant_length", mapping) == "pant"


def test_collar_task_has_no_component_filter(mapping: AttributeGroupMapping) -> None:
    """Collar tasks use class/region filters, not component filters."""
    for task in ("neckline_design", "collar_design", "neck_design", "lapel_design"):
        assert get_component_filter_for_task(task, mapping) is None


def test_length_tasks_without_component_filter_return_none(
    mapping: AttributeGroupMapping,
) -> None:
    """coat_length and skirt_length use class_contains, not component_contains."""
    assert get_component_filter_for_task("coat_length", mapping) is None
    assert get_component_filter_for_task("skirt_length", mapping) is None


def test_component_filter_for_unknown_task_returns_none(
    mapping: AttributeGroupMapping,
) -> None:
    assert get_component_filter_for_task("nonexistent_task", mapping) is None
    assert get_component_filter_for_task("", mapping) is None


def test_task_to_component_filter_values_are_strings(
    mapping: AttributeGroupMapping,
) -> None:
    for task, filt in mapping.task_to_component_filter.items():
        assert isinstance(task, str) and isinstance(filt, str)
        assert filt  # non-empty string
