"""
Metric summarization utilities.

This module aggregates instance-level segmentation results into dataset-level
summary metrics.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, List


def _safe_mean(values: list[float]) -> float:
    """
    Compute mean safely.

    Args:
        values: List of numeric values.

    Returns:
        Mean value or 0.0 if input list is empty.
    """
    return float(mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    """
    Compute median safely.

    Args:
        values: List of numeric values.

    Returns:
        Median value or 0.0 if input list is empty.
    """
    return float(median(values)) if values else 0.0


def _ratio_at_threshold(values: list[float], threshold: float) -> float:
    """
    Compute ratio of values greater than or equal to threshold.

    Args:
        values: List of numeric values.
        threshold: Threshold value.

    Returns:
        Ratio in range [0, 1].
    """
    if not values:
        return 0.0

    count = sum(value >= threshold for value in values)
    return float(count / len(values))


def summarize_instance_results(
    instance_results: List[Dict[str, Any]],
    dataset_name: str,
    split: str,
    method: str,
    num_images: int,
) -> Dict[str, Any]:
    """
    Summarize instance-level prediction results.

    Args:
        instance_results: List of instance-level result dictionaries.
        dataset_name: Dataset name.
        split: Dataset split.
        method: Method description.
        num_images: Number of processed images.

    Returns:
        Summary dictionary containing overall and per-category metrics.
    """
    iou_values = [
        float(result["iou"])
        for result in instance_results
        if "iou" in result and result["iou"] is not None
    ]

    total_latencies = [
        float(result["total_latency_ms"])
        for result in instance_results
        if "total_latency_ms" in result
        and result["total_latency_ms"] is not None
    ]

    sam_latencies = [
        float(result["sam_latency_ms"])
        for result in instance_results
        if "sam_latency_ms" in result and result["sam_latency_ms"] is not None
    ]

    category_to_ious: dict[str, list[float]] = defaultdict(list)

    for result in instance_results:
        category = str(result.get("target_category", "unknown"))
        iou = result.get("iou")

        if iou is not None:
            category_to_ious[category].append(float(iou))

    per_category: Dict[str, Dict[str, float | int]] = {}

    for category, values in sorted(category_to_ious.items()):
        per_category[category] = {
            "num_instances": len(values),
            "mean_iou": _safe_mean(values),
            "median_iou": _safe_median(values),
            "iou_at_0_5": _ratio_at_threshold(values, 0.5),
            "iou_at_0_75": _ratio_at_threshold(values, 0.75),
            "iou_at_0_85": _ratio_at_threshold(values, 0.85),
        }

    return {
        "dataset": dataset_name,
        "split": split,
        "method": method,
        "num_images": int(num_images),
        "num_instances": len(instance_results),
        "mean_iou": _safe_mean(iou_values),
        "median_iou": _safe_median(iou_values),
        "iou_at_0_5": _ratio_at_threshold(iou_values, 0.5),
        "iou_at_0_75": _ratio_at_threshold(iou_values, 0.75),
        "iou_at_0_85": _ratio_at_threshold(iou_values, 0.85),
        "mean_total_latency_ms": _safe_mean(total_latencies),
        "mean_sam_latency_ms": _safe_mean(sam_latencies),
        "per_category": per_category,
    }
