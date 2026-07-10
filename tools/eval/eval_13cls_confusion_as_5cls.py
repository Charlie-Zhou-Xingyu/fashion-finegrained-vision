"""CLI for evaluating a 13-class confusion matrix as 5 PRD classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.category_mapping import load_category_mapping
from tools.eval.confusion_aggregation import aggregate_13cls_to_5cls
from tools.eval.detection_metrics import compute_five_class_metrics
from tools.eval.report_writer import (
    save_five_class_markdown_report,
    save_json_report,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate a DeepFashion2 13-class confusion matrix "
        "into PRD 5-class detection metrics."
    )
    parser.add_argument(
        "--matrix-json",
        required=True,
        type=Path,
        help="Path to JSON file containing labels and 14x14 matrix.",
    )
    parser.add_argument(
        "--mapping-yaml",
        default=Path("configs/category_mapping.yaml"),
        type=Path,
        help="Path to category mapping YAML.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for JSON and Markdown reports.",
    )
    return parser.parse_args()


def load_confusion_matrix_json(path: str | Path) -> np.ndarray:
    """Load a confusion matrix JSON file.

    Expected JSON format:
        {
          "labels": [...],
          "matrix": [[...], ...]
        }

    Args:
        path: Path to the confusion matrix JSON file.

    Returns:
        A numpy array representing the confusion matrix.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        KeyError: If the matrix field is missing.
        ValueError: If the matrix cannot be converted to a 2D array.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Confusion matrix JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    if "matrix" not in data:
        raise KeyError(f"Missing 'matrix' field in {json_path}")

    matrix = np.asarray(data["matrix"], dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    return matrix


def main() -> None:
    """Run 13-class to 5-class confusion matrix evaluation."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    category_mapping = load_category_mapping(args.mapping_yaml)
    matrix_14x14 = load_confusion_matrix_json(args.matrix_json)

    matrix_5x5 = aggregate_13cls_to_5cls(
        matrix_14x14=matrix_14x14,
        map_13_to_5=category_mapping.map_13_to_5,
        num_classes_5=5,
    )

    class_names_5 = [
        category_mapping.prd_5cls[class_id] for class_id in range(5)
    ]

    metric_result = compute_five_class_metrics(
        matrix_14x14=matrix_14x14,
        matrix_5x5=matrix_5x5,
        map_13_to_5=category_mapping.map_13_to_5,
        class_names_5=class_names_5,
    )

    output_data = {
        "class_names_5": class_names_5,
        "matrix_5x5": matrix_5x5.tolist(),
        "metrics": metric_result.to_dict(),
    }

    json_path = args.out_dir / "eval_13cls_as_5cls_metrics.json"
    markdown_path = args.out_dir / "eval_13cls_as_5cls_report.md"

    save_json_report(output_data, json_path)
    save_five_class_markdown_report(
        path=markdown_path,
        matrix_5x5=matrix_5x5,
        metric_result=metric_result,
        class_names_5=class_names_5,
    )

    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {markdown_path}")
    print(
        "Foreground 5-class accuracy ignoring background: "
        f"{metric_result.foreground_accuracy_ignore_background:.4f}"
    )
    print(
        "Foreground correct detection rate: "
        f"{metric_result.foreground_correct_detection_rate:.4f}"
    )


if __name__ == "__main__":
    main()
