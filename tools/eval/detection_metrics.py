"""Detection confusion matrix metric computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PerClassMetric:
    """Per-class precision and recall result."""

    class_id: int
    class_name: str
    true_positive: int
    gt_total: int
    pred_total: int
    precision: float
    recall: float


@dataclass(frozen=True)
class FiveClassMetricResult:
    """Aggregated 5-class detection metric result."""

    foreground_correct: int
    foreground_matched_total: int
    foreground_accuracy_ignore_background: float
    foreground_gt_total: int
    foreground_correct_detection_rate: float
    overall_total_with_background: int
    overall_accuracy_with_background: float
    per_class: list[PerClassMetric]

    def to_dict(self) -> dict:
        """Convert metric result to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the metric result.
        """
        result = asdict(self)
        result["per_class"] = [asdict(item) for item in self.per_class]
        return result


def compute_five_class_metrics(
    matrix_14x14: np.ndarray,
    matrix_5x5: np.ndarray,
    map_13_to_5: dict[int, int],
    class_names_5: list[str],
) -> FiveClassMetricResult:
    """Compute 5-class metrics from original and aggregated matrices.

    Args:
        matrix_14x14: Original YOLO-style 14x14 confusion matrix.
        matrix_5x5: Aggregated 5x5 foreground confusion matrix.
        map_13_to_5: Mapping from 13-class ids to 5-class ids.
        class_names_5: Names of 5 foreground classes.

    Returns:
        FiveClassMetricResult containing summary and per-class metrics.
    """
    diagonal = np.diag(matrix_5x5)
    foreground_correct = int(diagonal.sum())
    foreground_matched_total = int(matrix_5x5.sum())

    foreground_accuracy = _safe_divide(
        foreground_correct,
        foreground_matched_total,
    )

    foreground_gt_total = int(matrix_14x14[:, :13].sum())
    foreground_correct_detection_rate = _safe_divide(
        foreground_correct,
        foreground_gt_total,
    )

    overall_total = int(matrix_14x14.sum())
    overall_correct = foreground_correct + int(matrix_14x14[13, 13])
    overall_accuracy = _safe_divide(overall_correct, overall_total)

    per_class = []
    for class_id, class_name in enumerate(class_names_5):
        per_class.append(
            _compute_single_class_metric(
                class_id=class_id,
                class_name=class_name,
                matrix_14x14=matrix_14x14,
                matrix_5x5=matrix_5x5,
                map_13_to_5=map_13_to_5,
            )
        )

    return FiveClassMetricResult(
        foreground_correct=foreground_correct,
        foreground_matched_total=foreground_matched_total,
        foreground_accuracy_ignore_background=foreground_accuracy,
        foreground_gt_total=foreground_gt_total,
        foreground_correct_detection_rate=foreground_correct_detection_rate,
        overall_total_with_background=overall_total,
        overall_accuracy_with_background=overall_accuracy,
        per_class=per_class,
    )


def _compute_single_class_metric(
    class_id: int,
    class_name: str,
    matrix_14x14: np.ndarray,
    matrix_5x5: np.ndarray,
    map_13_to_5: dict[int, int],
) -> PerClassMetric:
    """Compute precision and recall for one 5-class category.

    Args:
        class_id: Target 5-class id.
        class_name: Target 5-class name.
        matrix_14x14: Original YOLO-style confusion matrix.
        matrix_5x5: Aggregated foreground confusion matrix.
        map_13_to_5: Mapping from 13-class ids to 5-class ids.

    Returns:
        PerClassMetric for the target class.
    """
    gt_13_indices = [
        source_id for source_id, target_id in map_13_to_5.items()
        if target_id == class_id
    ]
    pred_13_indices = [
        source_id for source_id, target_id in map_13_to_5.items()
        if target_id == class_id
    ]

    true_positive = int(matrix_5x5[class_id, class_id])
    gt_total = int(matrix_14x14[:, gt_13_indices].sum())
    pred_total = int(matrix_14x14[pred_13_indices, :].sum())

    return PerClassMetric(
        class_id=class_id,
        class_name=class_name,
        true_positive=true_positive,
        gt_total=gt_total,
        pred_total=pred_total,
        precision=_safe_divide(true_positive, pred_total),
        recall=_safe_divide(true_positive, gt_total),
    )


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Safely divide two numbers.

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        Division result. Returns 0.0 if denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
