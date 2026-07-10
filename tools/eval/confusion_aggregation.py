"""Confusion matrix aggregation utilities."""

from __future__ import annotations

import numpy as np


def validate_yolo_confusion_matrix(matrix: np.ndarray) -> None:
    """Validate a YOLO-style 14x14 confusion matrix.

    The expected layout is:
        - rows: predicted classes
        - columns: ground-truth classes
        - ids 0-12: DeepFashion2 foreground classes
        - id 13: background

    Args:
        matrix: Input confusion matrix.

    Raises:
        ValueError: If matrix shape or values are invalid.
    """
    if matrix.shape != (14, 14):
        raise ValueError(f"Expected a 14x14 matrix, got {matrix.shape}.")

    if np.any(matrix < 0):
        raise ValueError("Confusion matrix cannot contain negative values.")


def aggregate_13cls_to_5cls(
    matrix_14x14: np.ndarray,
    map_13_to_5: dict[int, int],
    num_classes_5: int = 5,
) -> np.ndarray:
    """Aggregate DeepFashion2 13-class confusion matrix to PRD 5-class matrix.

    Background row and column are ignored in the returned 5x5 matrix.

    Args:
        matrix_14x14: YOLO-style 14x14 confusion matrix.
        map_13_to_5: Mapping from 13-class ids to 5-class ids.
        num_classes_5: Number of output foreground classes.

    Returns:
        A 5x5 confusion matrix where rows are predicted classes and columns
        are ground-truth classes.
    """
    validate_yolo_confusion_matrix(matrix_14x14)

    matrix_5x5 = np.zeros((num_classes_5, num_classes_5), dtype=np.int64)

    for pred_13 in range(13):
        for gt_13 in range(13):
            pred_5 = map_13_to_5[pred_13]
            gt_5 = map_13_to_5[gt_13]
            matrix_5x5[pred_5, gt_5] += int(matrix_14x14[pred_13, gt_13])

    return matrix_5x5
