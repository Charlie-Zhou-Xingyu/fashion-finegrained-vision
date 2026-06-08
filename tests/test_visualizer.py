"""
Unit tests for visualization utilities.
"""

from pathlib import Path

import numpy as np
import pytest

from fashion_vision.visualization.visualizer import (
    draw_bbox_with_label,
    draw_instance_segmentation,
    format_instance_label,
    overlay_mask,
    save_instance_visualization,
)


def test_overlay_mask_changes_foreground_pixels() -> None:
    """Test that mask overlay changes foreground pixels."""
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 10:20] = 1

    output = overlay_mask(
        image_rgb=image,
        mask=mask,
        color=(255, 0, 0),
        alpha=0.5,
    )

    assert output.shape == image.shape
    assert output[10, 15, 0] > 0
    assert output[0, 0].sum() == 0


def test_overlay_mask_shape_mismatch() -> None:
    """Test that overlay_mask raises error for shape mismatch."""
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((10, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        overlay_mask(
            image_rgb=image,
            mask=mask,
            color=(255, 0, 0),
        )


def test_draw_bbox_with_label() -> None:
    """Test drawing bbox and label."""
    image = np.zeros((40, 50, 3), dtype=np.uint8)

    output = draw_bbox_with_label(
        image_rgb=image,
        bbox=[5, 5, 30, 30],
        label="top IoU:0.90",
        color=(255, 0, 0),
    )

    assert output.shape == image.shape
    assert output.sum() > 0


def test_draw_bbox_invalid_geometry() -> None:
    """Test invalid bbox geometry."""
    image = np.zeros((40, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        draw_bbox_with_label(
            image_rgb=image,
            bbox=[30, 5, 5, 30],
            label="invalid",
            color=(255, 0, 0),
        )


def test_format_instance_label() -> None:
    """Test formatting instance label."""
    instance = {
        "target_category_zh": "外套",
        "iou": 0.876,
        "score": 0.932,
    }

    label = format_instance_label(instance)

    assert "外套" in label
    assert "IoU:0.88" in label
    assert "S:0.93" in label


def test_draw_instance_segmentation() -> None:
    """Test drawing full instance segmentation visualization."""
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[10:30, 15:35] = 1

    instances = [
        {
            "bbox": [15, 10, 35, 30],
            "target_category_zh": "上衣",
            "iou": 0.9,
            "score": 0.95,
            "pred_mask": mask,
        }
    ]

    output = draw_instance_segmentation(image, instances)

    assert output.shape == image.shape
    assert output.sum() > 0


def test_save_instance_visualization(tmp_path: Path) -> None:
    """Test saving visualization image."""
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[10:30, 15:35] = 1

    instances = [
        {
            "bbox": [15, 10, 35, 30],
            "target_category_zh": "上衣",
            "pred_mask": mask,
        }
    ]

    output_path = tmp_path / "vis.jpg"

    save_instance_visualization(
        image_rgb=image,
        instances=instances,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.is_file()
