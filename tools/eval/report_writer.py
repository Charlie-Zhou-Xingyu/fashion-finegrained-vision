"""Report writing utilities for detection evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.detection_metrics import FiveClassMetricResult


def save_json_report(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a formatted JSON file.

    Args:
        data: JSON-serializable dictionary.
        path: Output JSON path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_five_class_markdown_report(
    path: str | Path,
    matrix_5x5: np.ndarray,
    metric_result: FiveClassMetricResult,
    class_names_5: list[str],
) -> None:
    """Save a Markdown report for 5-class aggregated evaluation.

    Args:
        path: Output Markdown path.
        matrix_5x5: Aggregated 5x5 confusion matrix.
        metric_result: Computed 5-class metric result.
        class_names_5: Names of 5 foreground classes.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 13-class Detector Aggregated as 5-class Report",
        "",
        "## 1. Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        _format_metric_row(
            "Foreground 5-class accuracy, ignoring background",
            metric_result.foreground_accuracy_ignore_background,
        ),
        _format_metric_row(
            "Foreground correct detection rate, counting missed foreground as wrong",
            metric_result.foreground_correct_detection_rate,
        ),
        _format_metric_row(
            "Overall accuracy, treating background as ordinary class",
            metric_result.overall_accuracy_with_background,
        ),
        f"| Foreground correct | {metric_result.foreground_correct} |",
        f"| Foreground matched total | {metric_result.foreground_matched_total} |",
        f"| Foreground GT total | {metric_result.foreground_gt_total} |",
        "",
        "## 2. Aggregated 5-class Confusion Matrix",
        "",
    ]

    lines.extend(_format_confusion_matrix(matrix_5x5, class_names_5))
    lines.extend(
        [
            "",
            "## 3. Per-class Metrics",
            "",
            "| Class | TP | GT Total | Pred Total | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for item in metric_result.per_class:
        lines.append(
            f"| {item.class_name} | "
            f"{item.true_positive} | "
            f"{item.gt_total} | "
            f"{item.pred_total} | "
            f"{item.precision:.4f} | "
            f"{item.recall:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric_row(name: str, value: float) -> str:
    """Format a float metric as a Markdown table row.

    Args:
        name: Metric name.
        value: Metric value.

    Returns:
        Markdown table row.
    """
    return f"| {name} | {value:.4f} |"


def _format_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
) -> list[str]:
    """Format a confusion matrix as Markdown rows.

    Args:
        matrix: Confusion matrix.
        class_names: Class names.

    Returns:
        List of Markdown lines.
    """
    lines = []
    header = "| Pred \\\\ GT | " + " | ".join(class_names) + " |"
    separator = "|---|" + "|".join(["---:"] * len(class_names)) + "|"

    lines.append(header)
    lines.append(separator)

    for row_idx, class_name in enumerate(class_names):
        row_values = [str(int(value)) for value in matrix[row_idx].tolist()]
        lines.append(f"| {class_name} | " + " | ".join(row_values) + " |")

    return lines
