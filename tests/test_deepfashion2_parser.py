"""
Unit tests for DeepFashion2 parser.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from fashion_vision.data.deepfashion2_parser import DeepFashion2Parser


def _create_dummy_deepfashion2_sample(root: Path) -> None:
    """
    Create a minimal DeepFashion2-like sample for parser tests.

    Args:
        root: Temporary dataset root.
    """
    image_dir = root / "validation" / "image"
    annotation_dir = root / "validation" / "annos"

    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(image_dir / "000001.jpg"), image)

    annotation = {
        "source": "user",
        "pair_id": 1,
        "item1": {
            "category_id": 5,
            "category_name": "vest",
            "bounding_box": [10, 10, 60, 80],
            "segmentation": [[10, 10, 60, 10, 60, 80, 10, 80]],
        },
    }

    annotation_path = annotation_dir / "000001.json"

    import json

    with annotation_path.open("w", encoding="utf-8") as file:
        json.dump(annotation, file)


def test_list_image_ids(tmp_path: Path) -> None:
    """Test listing image IDs from DeepFashion2-like structure."""
    _create_dummy_deepfashion2_sample(tmp_path)

    parser = DeepFashion2Parser(root=tmp_path)
    image_ids = parser.list_image_ids()

    assert image_ids == ["000001"]


def test_load_sample(tmp_path: Path) -> None:
    """Test loading and parsing one sample."""
    _create_dummy_deepfashion2_sample(tmp_path)

    parser = DeepFashion2Parser(root=tmp_path)
    sample = parser.load_sample("000001")

    assert sample["image_id"] == "000001"
    assert sample["width"] == 120
    assert sample["height"] == 100
    assert len(sample["instances"]) == 1

    instance = sample["instances"][0]

    assert instance["instance_id"] == "item1"
    assert instance["raw_category_id"] == 5
    assert instance["raw_category_name"] == "vest"
    assert instance["target_category"] == "outwear"
    assert instance["target_category_zh"] == "外套"
    assert instance["bbox"] == [10.0, 10.0, 60.0, 80.0]
    assert instance["gt_mask"].shape == (100, 120)
    assert instance["gt_mask"].sum() > 0


def test_polygon_to_mask() -> None:
    """Test polygon-to-mask conversion."""
    segmentation = [[10, 10, 50, 10, 50, 50, 10, 50]]

    mask = DeepFashion2Parser.polygon_to_mask(
        segmentation=segmentation,
        height=80,
        width=100,
    )

    assert mask.shape == (80, 100)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


def test_invalid_root(tmp_path: Path) -> None:
    """Test parser raises error for invalid dataset root."""
    invalid_root = tmp_path / "not_exist"

    with pytest.raises(FileNotFoundError):
        DeepFashion2Parser(root=invalid_root)
