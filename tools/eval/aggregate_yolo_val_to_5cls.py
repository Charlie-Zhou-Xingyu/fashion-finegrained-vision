"""
Aggregate YOLO val per-class metrics from 13 DeepFashion2 classes
to 5 PRD garment categories required by PRD 3.1.1.

Metric aggregation method
-------------------------
For each coarse class (group of fine classes):
  - TP_i  = recall_i  * instances_i            (per fine-class approximate TP)
  - FP_i  = TP_i * (1 - precision_i) / precision_i  (per fine-class approximate FP)
  - coarse precision = sum(TP_i) / (sum(TP_i) + sum(FP_i))
  - coarse recall    = sum(TP_i) / sum(instances_i)
  - coarse F1        = 2 * P * R / (P + R)
  - coarse mAP50     = instance-weighted average of fine-class mAP50 values

This decomposition is an approximation because only point-estimate P/R values
are available (not the full prediction list).  For exact counts use the
confusion matrix path (extract_yolo_confusion_matrix.py +
eval_13cls_confusion_as_5cls.py).

Usage
-----
Step 1: redirect YOLO val stdout to a text file::

    yolo detect val \\
        data=data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml \\
        model=runs/detect/yolov8n_df2_13cls_balanced/weights/best.pt \\
        imgsz=640 device=0 2>&1 | tee outputs/yolo_val_log.txt

Step 2: aggregate to 5 PRD classes::

    python tools/eval/aggregate_yolo_val_to_5cls.py \\
        --val-txt outputs/yolo_val_log.txt \\
        --mapping-yaml configs/category_mapping.yaml \\
        --out-dir outputs/eval_prd_5cls

Step 3 (optional): add 5x5 confusion matrix section::

    # First extract the confusion matrix JSON:
    python tools/eval/extract_yolo_confusion_matrix.py \\
        --weights runs/detect/yolov8n_df2_13cls_balanced/weights/best.pt \\
        --data data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml \\
        --out outputs/confusion_matrix_14x14.json

    # Then re-run with --confusion-json:
    python tools/eval/aggregate_yolo_val_to_5cls.py \\
        --val-txt outputs/yolo_val_log.txt \\
        --confusion-json outputs/confusion_matrix_14x14.json \\
        --out-dir outputs/eval_prd_5cls
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.eval.category_mapping import CategoryMapping, load_category_mapping  # noqa: E402
from tools.eval.confusion_aggregation import aggregate_13cls_to_5cls  # noqa: E402

_DEFAULT_MAPPING_YAML = _PROJECT_ROOT / "configs" / "category_mapping.yaml"

# Strips ANSI colour codes that YOLO sometimes emits to terminals.
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FineClassRow:
    """Single per-class row parsed from YOLO val text output."""

    class_name: str
    images: int
    instances: int
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass
class CoarseClassMetric:
    """Aggregated metrics for one PRD 5-class category."""

    class_id: int
    class_name: str
    instances: int
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float


@dataclass
class OverallMetrics:
    """Micro and macro averaged metrics across all 5 PRD classes."""

    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_map50: float
    weighted_map50_95: float
    total_instances: int


@dataclass
class AggregationResult:
    """Full 5-class aggregation result."""

    per_class: list[CoarseClassMetric]
    overall: OverallMetrics
    matrix_5x5: Optional[list[list[int]]]
    class_names_5: list[str]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_yolo_val_text(path: Path) -> list[FineClassRow]:
    """Parse per-class rows from a YOLO val text output file.

    Expected row format (space-aligned, class name may contain spaces)::

        short sleeve top   12371   12556   0.892   0.864   0.945   0.827

    The ``all`` summary row and any header lines are skipped.

    Args:
        path: Path to the YOLO val text file.

    Returns:
        List of parsed fine-class rows.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid per-class rows were found.
    """
    if not path.exists():
        raise FileNotFoundError(f"YOLO val text file not found: {path}")

    rows: list[FineClassRow] = []

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = _ANSI_ESCAPE.sub("", raw_line).strip()
            if not line:
                continue
            # Skip header lines (contain "Class" or "Box(P")
            if "Class" in line or "Box(P" in line:
                continue
            # Skip the overall summary row
            if line.startswith("all ") or line == "all":
                continue

            # Split from the right to extract 6 numeric columns;
            # everything remaining on the left is the (possibly multi-word) class name.
            parts = line.rsplit(maxsplit=6)
            if len(parts) != 7:
                continue

            class_name = parts[0].strip()
            try:
                images = int(parts[1])
                instances = int(parts[2])
                precision = float(parts[3])
                recall = float(parts[4])
                map50 = float(parts[5])
                map50_95 = float(parts[6])
            except ValueError:
                continue

            rows.append(
                FineClassRow(
                    class_name=class_name,
                    images=images,
                    instances=instances,
                    precision=precision,
                    recall=recall,
                    map50=map50,
                    map50_95=map50_95,
                )
            )

    if not rows:
        raise ValueError(
            f"No valid per-class rows found in {path}. "
            "Ensure the file contains YOLO val per-class metric lines."
        )

    return rows


def load_confusion_matrix_json(path: Path) -> np.ndarray:
    """Load a 14x14 confusion matrix from JSON.

    Expected format::

        {"labels": [...], "matrix": [[...], ...]}

    Args:
        path: Path to the confusion matrix JSON file.

    Returns:
        Numpy array of shape (14, 14) with integer counts.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the ``matrix`` field is absent.
        ValueError: If the matrix is not 2-D.
    """
    if not path.exists():
        raise FileNotFoundError(f"Confusion matrix JSON not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "matrix" not in data:
        raise KeyError(f"Missing 'matrix' field in {path}")

    matrix = np.asarray(data["matrix"], dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {matrix.shape}.")

    return matrix


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        Division result or 0.0 on zero denominator.
    """
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall.

    Args:
        precision: Precision value in [0, 1].
        recall: Recall value in [0, 1].

    Returns:
        F1 score, or 0.0 when both are zero.
    """
    return _safe_divide(2.0 * precision * recall, precision + recall)


def aggregate_to_5cls(
    rows: list[FineClassRow],
    mapping: CategoryMapping,
) -> list[CoarseClassMetric]:
    """Aggregate 13 fine-class metrics to 5 PRD coarse-class metrics.

    For each coarse class the TP and FP counts are approximated from the
    per-fine-class precision and recall values, then re-combined.

    Args:
        rows: Parsed per-fine-class metric rows.
        mapping: Loaded category mapping (13-class → 5-class).

    Returns:
        List of 5 CoarseClassMetric objects, ordered by coarse class id.

    Raises:
        ValueError: If a row class name is not in the mapping.
    """
    name_to_fine_id = {v: k for k, v in mapping.deepfashion2_13cls.items()}

    groups: dict[int, list[FineClassRow]] = {i: [] for i in range(5)}
    unrecognised: list[str] = []

    for row in rows:
        fine_id = name_to_fine_id.get(row.class_name)
        if fine_id is None:
            unrecognised.append(row.class_name)
            continue
        coarse_id = mapping.map_13_to_5[fine_id]
        groups[coarse_id].append(row)

    if unrecognised:
        raise ValueError(
            f"Class names not found in category mapping: {unrecognised}. "
            "Verify that --mapping-yaml matches the YOLO model's class names."
        )

    results: list[CoarseClassMetric] = []

    for coarse_id in range(5):
        group = groups[coarse_id]
        coarse_name = mapping.prd_5cls[coarse_id]

        if not group:
            results.append(
                CoarseClassMetric(
                    class_id=coarse_id,
                    class_name=coarse_name,
                    instances=0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    map50=0.0,
                    map50_95=0.0,
                )
            )
            continue

        tp_sum = 0.0
        fp_sum = 0.0
        gt_sum = 0
        map50_weighted = 0.0
        map50_95_weighted = 0.0

        for row in group:
            tp = row.recall * row.instances
            fp = tp * (1.0 - row.precision) / row.precision if row.precision > 0 else 0.0
            tp_sum += tp
            fp_sum += fp
            gt_sum += row.instances
            map50_weighted += row.map50 * row.instances
            map50_95_weighted += row.map50_95 * row.instances

        prec = _safe_divide(tp_sum, tp_sum + fp_sum)
        rec = _safe_divide(tp_sum, gt_sum)
        map50_agg = _safe_divide(map50_weighted, gt_sum)
        map50_95_agg = _safe_divide(map50_95_weighted, gt_sum)

        results.append(
            CoarseClassMetric(
                class_id=coarse_id,
                class_name=coarse_name,
                instances=gt_sum,
                precision=prec,
                recall=rec,
                f1=_f1(prec, rec),
                map50=map50_agg,
                map50_95=map50_95_agg,
            )
        )

    return results


def compute_overall_metrics(
    rows: list[FineClassRow],
    per_class: list[CoarseClassMetric],
) -> OverallMetrics:
    """Compute micro and macro aggregated metrics across all 5 PRD classes.

    Micro metrics aggregate TP/FP/GT counts across all fine classes before
    dividing, giving equal weight to each detection instance.
    Macro metrics average the per-coarse-class values, giving equal weight
    to each PRD category.

    Args:
        rows: All 13 fine-class rows (used for micro computation).
        per_class: Aggregated coarse-class metrics (used for macro computation).

    Returns:
        OverallMetrics containing micro, macro, and weighted-mAP values.
    """
    # Micro: aggregate directly over all 13 fine classes
    tp_total = 0.0
    fp_total = 0.0
    gt_total = 0
    map50_total = 0.0
    map50_95_total = 0.0

    for row in rows:
        tp = row.recall * row.instances
        fp = tp * (1.0 - row.precision) / row.precision if row.precision > 0 else 0.0
        tp_total += tp
        fp_total += fp
        gt_total += row.instances
        map50_total += row.map50 * row.instances
        map50_95_total += row.map50_95 * row.instances

    micro_p = _safe_divide(tp_total, tp_total + fp_total)
    micro_r = _safe_divide(tp_total, gt_total)

    # Macro: average over 5 coarse classes
    non_empty = [m for m in per_class if m.instances > 0]
    macro_p = sum(m.precision for m in non_empty) / len(non_empty) if non_empty else 0.0
    macro_r = sum(m.recall for m in non_empty) / len(non_empty) if non_empty else 0.0
    macro_f1 = sum(m.f1 for m in non_empty) / len(non_empty) if non_empty else 0.0

    return OverallMetrics(
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=_f1(micro_p, micro_r),
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_map50=_safe_divide(map50_total, gt_total),
        weighted_map50_95=_safe_divide(map50_95_total, gt_total),
        total_instances=gt_total,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def save_json_report(result: AggregationResult, path: Path) -> None:
    """Save aggregation result as a formatted JSON file.

    Args:
        result: Aggregation result to serialise.
        path: Output JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "class_names_5": result.class_names_5,
        "per_class": [asdict(m) for m in result.per_class],
        "overall": asdict(result.overall),
    }
    if result.matrix_5x5 is not None:
        data["matrix_5x5"] = result.matrix_5x5

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv_report(result: AggregationResult, path: Path) -> None:
    """Save per-class metrics as a CSV file.

    Args:
        result: Aggregation result to serialise.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "class_id",
        "class_name",
        "instances",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
    ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for m in result.per_class:
            writer.writerow(
                {
                    "class_id": m.class_id,
                    "class_name": m.class_name,
                    "instances": m.instances,
                    "precision": f"{m.precision:.4f}",
                    "recall": f"{m.recall:.4f}",
                    "f1": f"{m.f1:.4f}",
                    "map50": f"{m.map50:.4f}",
                    "map50_95": f"{m.map50_95:.4f}",
                }
            )

        # Append micro/macro summary rows
        ov = result.overall
        for label, p, r, f1 in (
            ("__micro__", ov.micro_precision, ov.micro_recall, ov.micro_f1),
            ("__macro__", ov.macro_precision, ov.macro_recall, ov.macro_f1),
        ):
            writer.writerow(
                {
                    "class_id": "",
                    "class_name": label,
                    "instances": ov.total_instances,
                    "precision": f"{p:.4f}",
                    "recall": f"{r:.4f}",
                    "f1": f"{f1:.4f}",
                    "map50": f"{ov.weighted_map50:.4f}",
                    "map50_95": f"{ov.weighted_map50_95:.4f}",
                }
            )


def save_markdown_report(result: AggregationResult, path: Path) -> None:
    """Save a Markdown evaluation report.

    Args:
        result: Aggregation result to serialise.
        path: Output Markdown path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# PRD 3.1.1 Garment Detection — 5-Class Evaluation Report",
        "",
        "> Metrics aggregated from 13 DeepFashion2 classes to 5 PRD garment categories.",
        "> Precision and recall are approximate (derived from per-class P/R point estimates).",
        "> For exact TP/FP/FN counts use the confusion matrix path (`eval_13cls_confusion_as_5cls.py`).",
        "",
        "## 1. Per-class Metrics",
        "",
        "| Class | Instances | Precision | Recall | F1 | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for m in result.per_class:
        lines.append(
            f"| {m.class_name} | {m.instances} | "
            f"{m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | "
            f"{m.map50:.4f} | {m.map50_95:.4f} |"
        )

    ov = result.overall
    lines += [
        "",
        "## 2. Overall Metrics",
        "",
        "| Metric | Precision | Recall | F1 | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Micro (instance-level) | {ov.micro_precision:.4f} | "
            f"{ov.micro_recall:.4f} | {ov.micro_f1:.4f} | "
            f"{ov.weighted_map50:.4f} | {ov.weighted_map50_95:.4f} |"
        ),
        (
            f"| Macro (class-level average) | {ov.macro_precision:.4f} | "
            f"{ov.macro_recall:.4f} | {ov.macro_f1:.4f} | — | — |"
        ),
    ]

    if result.matrix_5x5 is not None:
        lines += [
            "",
            "## 3. 5-Class Confusion Matrix",
            "",
            "> Rows: predicted class.  Columns: ground-truth class.",
            "",
        ]
        names = result.class_names_5
        header = "| Pred \\\\ GT | " + " | ".join(names) + " |"
        sep = "|---|" + "|".join(["---:"] * len(names)) + "|"
        lines += [header, sep]
        for row_idx, name in enumerate(names):
            cells = [str(result.matrix_5x5[row_idx][col_idx]) for col_idx in range(len(names))]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate YOLO val 13-class per-class metrics to 5 PRD "
            "garment categories (PRD 3.1.1)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--val-txt",
        type=Path,
        required=True,
        help=(
            "Path to YOLO val stdout redirected to a text file. "
            "Tip: yolo detect val ... 2>&1 | tee outputs/yolo_val_log.txt"
        ),
    )
    parser.add_argument(
        "--mapping-yaml",
        type=Path,
        default=_DEFAULT_MAPPING_YAML,
        help=(
            "Path to category mapping YAML. "
            f"Default: {_DEFAULT_MAPPING_YAML}"
        ),
    )
    parser.add_argument(
        "--confusion-json",
        type=Path,
        default=None,
        help=(
            "Optional: path to 14x14 confusion matrix JSON "
            "(from extract_yolo_confusion_matrix.py). "
            "When provided, the report includes a 5x5 confusion matrix section."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for JSON, CSV, and Markdown reports.",
    )
    return parser.parse_args()


def main() -> None:
    """Run 13-to-5-class metric aggregation and save reports."""
    args = parse_args()

    mapping = load_category_mapping(args.mapping_yaml)

    rows = parse_yolo_val_text(args.val_txt)
    print(f"Parsed {len(rows)} fine-class rows from {args.val_txt}")

    per_class = aggregate_to_5cls(rows, mapping)
    overall = compute_overall_metrics(rows, per_class)

    matrix_5x5_list: Optional[list[list[int]]] = None
    if args.confusion_json is not None:
        matrix_14x14 = load_confusion_matrix_json(args.confusion_json)
        matrix_5x5 = aggregate_13cls_to_5cls(
            matrix_14x14=matrix_14x14,
            map_13_to_5=mapping.map_13_to_5,
            num_classes_5=5,
        )
        matrix_5x5_list = matrix_5x5.tolist()
        print(f"Loaded confusion matrix from {args.confusion_json}")

    class_names_5 = [mapping.prd_5cls[i] for i in range(5)]
    result = AggregationResult(
        per_class=per_class,
        overall=overall,
        matrix_5x5=matrix_5x5_list,
        class_names_5=class_names_5,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "prd_5cls_metrics.json"
    csv_path = args.out_dir / "prd_5cls_metrics.csv"
    md_path = args.out_dir / "prd_5cls_report.md"

    save_json_report(result, json_path)
    save_csv_report(result, csv_path)
    save_markdown_report(result, md_path)

    print(f"\nOutputs saved to {args.out_dir}/")
    print(f"  {json_path.name}")
    print(f"  {csv_path.name}")
    print(f"  {md_path.name}")

    print("\n--- 5-Class PRD Metrics (micro) ---")
    print(
        f"  Precision: {overall.micro_precision:.4f}  "
        f"Recall: {overall.micro_recall:.4f}  "
        f"F1: {overall.micro_f1:.4f}  "
        f"mAP50: {overall.weighted_map50:.4f}"
    )
    print("\n--- Per-class ---")
    for m in result.per_class:
        print(
            f"  {m.class_name:<16}  "
            f"P={m.precision:.4f}  R={m.recall:.4f}  "
            f"F1={m.f1:.4f}  mAP50={m.map50:.4f}"
        )


if __name__ == "__main__":
    main()
