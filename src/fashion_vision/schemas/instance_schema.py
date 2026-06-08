"""
Standard schemas for fashion instance segmentation outputs.

This module provides lightweight helper functions to build consistent JSON
records across different pipelines, including:

- DeepFashion2 GT-box + SAM-HQ baseline
- Label Studio annotations + SAM-HQ
- GroundingDINO detection + SAM-HQ
- Future local region localization and attribute extraction modules
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _to_float_list(values: List[Any]) -> List[float]:
    """
    Convert a list-like object to a list of floats.

    Args:
        values: Input list-like values.

    Returns:
        List of floats.
    """
    return [float(value) for value in values]


def build_instance_record(
    image_id: str,
    instance_id: str,
    category: str,
    bbox: List[Any],
    pred_mask_path: Optional[str] = None,
    gt_mask_path: Optional[str] = None,
    category_zh: Optional[str] = None,
    category_id: Optional[int] = None,
    source_item_id: Optional[str] = None,
    score: Optional[float] = None,
    iou: Optional[float] = None,
    source_type: str = "unknown",
    segmentor: str = "sam_hq",
    bbox_format: str = "xyxy",
    status: str = "success",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a standardized instance-level prediction record.

    Args:
        image_id: Image identifier.
        instance_id: Instance identifier.
        category: Unified target category name.
        bbox: Bounding box in xyxy format by default.
        pred_mask_path: Saved predicted mask path.
        gt_mask_path: Saved ground-truth mask path.
        category_zh: Chinese category name.
        category_id: Original dataset category id if available.
        source_item_id: Original annotation item id.
        score: SAM or detector confidence score.
        iou: IoU between predicted mask and ground-truth mask if available.
        source_type: Source of bbox/mask, e.g. deepfashion2_gt_box.
        segmentor: Segmentation model name.
        bbox_format: Bounding box format.
        status: Instance processing status.
        extra: Extra fields to merge into the record.

    Returns:
        Standardized instance record.
    """
    record: Dict[str, Any] = {
        "image_id": str(image_id),
        "instance_id": str(instance_id),
        "source_item_id": str(source_item_id or instance_id),
        "category": str(category),
        "category_zh": str(category_zh or category),
        "category_id": int(category_id) if category_id is not None else None,
        "target_category": str(category),
        "bbox": _to_float_list(list(bbox)),
        "bbox_format": str(bbox_format),
        "pred_mask_path": str(pred_mask_path) if pred_mask_path else None,
        "gt_mask_path": str(gt_mask_path) if gt_mask_path else None,
        "score": float(score) if score is not None else None,
        "iou": float(iou) if iou is not None else None,
        "source_type": str(source_type),
        "segmentor": str(segmentor),
        "status": str(status),
    }

    if extra:
        record.update(extra)

    return record


def build_image_prediction_record(
    image_id: str,
    image_path: str,
    width: int,
    height: int,
    instances: List[Dict[str, Any]],
    source: str = "deepfashion2",
    task: str = "instance_segmentation",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a standardized image-level prediction record.

    Args:
        image_id: Image identifier.
        image_path: Original image path.
        width: Image width.
        height: Image height.
        instances: List of standardized instance records.
        source: Data source.
        task: Task name.
        extra: Extra fields to merge into image-level record.

    Returns:
        Standardized image-level prediction record.
    """
    record: Dict[str, Any] = {
        "image_id": str(image_id),
        "image_path": str(image_path),
        "width": int(width),
        "height": int(height),
        "source": str(source),
        "task": str(task),
        "num_instances": int(len(instances)),
        "instances": instances,
    }

    if extra:
        record.update(extra)

    return record


def build_index_record(
    experiment_name: str,
    output_root: str,
    num_images: int,
    num_instances: int,
    images: List[Dict[str, Any]],
    mean_iou: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an experiment-level index record.

    Args:
        experiment_name: Experiment name.
        output_root: Output root directory.
        num_images: Number of processed images.
        num_instances: Number of processed instances.
        images: List of image-level index entries.
        mean_iou: Mean IoU if available.
        extra: Extra fields to merge into index record.

    Returns:
        Standardized experiment index record.
    """
    record: Dict[str, Any] = {
        "experiment": str(experiment_name),
        "output_root": str(output_root),
        "num_images": int(num_images),
        "num_instances": int(num_instances),
        "mean_iou": float(mean_iou) if mean_iou is not None else None,
        "images": images,
    }

    if extra:
        record.update(extra)

    return record


def build_image_index_entry(
    image_id: str,
    image_path: str,
    prediction_json: Optional[str] = None,
    visualization: Optional[str] = None,
    num_instances: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build one image-level entry for index.json.

    Args:
        image_id: Image identifier.
        image_path: Original image path.
        prediction_json: Prediction JSON path.
        visualization: Visualization image path.
        num_instances: Number of instances for this image.
        extra: Extra fields to merge into entry.

    Returns:
        Image index entry.
    """
    entry: Dict[str, Any] = {
        "image_id": str(image_id),
        "image_path": str(image_path),
        "prediction_json": str(prediction_json) if prediction_json else None,
        "visualization": str(visualization) if visualization else None,
        "num_instances": int(num_instances) if num_instances is not None else None,
    }

    if extra:
        entry.update(extra)

    return entry
