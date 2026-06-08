"""
Unit tests for category mapping utilities.
"""

import pytest

from fashion_vision.data.class_mapping import (
    get_all_target_categories,
    get_supplementary_dataset_plan,
    map_deepfashion2_category,
)


def test_target_taxonomy_has_eight_categories() -> None:
    """Test that the unified target taxonomy contains 8 categories."""
    categories = get_all_target_categories()

    assert len(categories) == 8
    assert "top" in categories
    assert "pants" in categories
    assert "skirt" in categories
    assert "outwear" in categories
    assert "dress" in categories
    assert "shoes" in categories
    assert "bag" in categories
    assert "accessory" in categories


def test_map_deepfashion2_top() -> None:
    """Test mapping DeepFashion2 top category."""
    result = map_deepfashion2_category(1)

    assert result["raw_category_name"] == "short sleeve top"
    assert result["target_category"] == "top"
    assert result["target_category_zh"] == "上衣"


def test_map_deepfashion2_outwear() -> None:
    """Test mapping DeepFashion2 outwear category."""
    result = map_deepfashion2_category(3)

    assert result["raw_category_name"] == "short sleeve outwear"
    assert result["target_category"] == "outwear"
    assert result["target_category_zh"] == "外套"


def test_map_deepfashion2_vest_as_outwear() -> None:
    """Test mapping DeepFashion2 vest category to outwear."""
    result = map_deepfashion2_category(5)

    assert result["raw_category_name"] == "vest"
    assert result["target_category"] == "outwear"
    assert result["target_category_zh"] == "外套"


def test_map_deepfashion2_invalid_category() -> None:
    """Test invalid DeepFashion2 category ID."""
    with pytest.raises(KeyError):
        map_deepfashion2_category(999)


def test_supplementary_dataset_plan_for_shoes() -> None:
    """Test supplementary dataset plan for shoes."""
    plan = get_supplementary_dataset_plan("shoes")

    assert "UT Zappos" in plan
    assert "Custom Label Studio annotations" in plan


def test_supplementary_dataset_plan_for_bag() -> None:
    """Test supplementary dataset plan for bag."""
    plan = get_supplementary_dataset_plan("bag")

    assert "Custom Label Studio annotations" in plan


def test_supplementary_dataset_plan_invalid_category() -> None:
    """Test supplementary dataset plan with invalid category."""
    with pytest.raises(KeyError):
        get_supplementary_dataset_plan("top")
