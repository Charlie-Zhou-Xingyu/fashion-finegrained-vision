"""Batch evaluation script for all 8 FashionAI attribute classification tasks.

Reads checkpoint and label_map paths from configs/attribute_inference.yaml.
Val/test JSONL files are expected at:
    data/fashionai_attribute_index/{task}_{split}.jsonl

For each task, runs eval_attribute_classifier_checkpoint.py and writes:
    outputs/attr_eval/{task}/metrics.json
    outputs/attr_eval/{task}/confusion_matrix.csv
    outputs/attr_eval/{task}/predictions.jsonl  (if --save-predictions)

Generates a consolidated Markdown summary at:
    outputs/attr_eval/eval_summary.md

Usage
-----
::

    python scripts/run_all_attribute_eval.py [OPTIONS]

    python scripts/run_all_attribute_eval.py --split val --save-predictions
    python scripts/run_all_attribute_eval.py --tasks sleeve_length skirt_length
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INFERENCE_CONFIG = _PROJECT_ROOT / "configs" / "attribute_inference.yaml"
DEFAULT_JSONL_ROOT = _PROJECT_ROOT / "data" / "fashionai_attribute_index"
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "attr_eval"
PRD_MACRO_F1_TARGET = 0.88

TASK_ORDER = [
    "collar_design",
    "neckline_design",
    "neck_design",
    "lapel_design",
    "sleeve_length",
    "coat_length",
    "pant_length",
    "skirt_length",
]

_EVAL_SCRIPT = _PROJECT_ROOT / "scripts" / "eval_attribute_classifier_checkpoint.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation for all FashionAI attribute classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=DEFAULT_INFERENCE_CONFIG,
        help="Path to attribute_inference.yaml.",
    )
    parser.add_argument(
        "--jsonl-root",
        type=Path,
        default=DEFAULT_JSONL_ROOT,
        help="Directory containing {task}_{split}.jsonl files.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory; per-task results go into subdirectories.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample predictions.jsonl for error inspection.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Subset of tasks to run. Default: all 8 tasks.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=5,
        help="Number of top misclassification pairs to report per task.",
    )
    return parser.parse_args()


def load_inference_config(path: Path) -> Dict[str, Any]:
    """Load and return the attribute_inference.yaml contents.

    Args:
        path: Path to the YAML config.

    Returns:
        Parsed YAML dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Inference config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_single_task(
    task: str,
    checkpoint: Path,
    label_map: Path,
    jsonl: Path,
    output_dir: Path,
    arch: str,
    img_size: int,
    device: str,
    batch_size: int,
    save_predictions: bool,
) -> bool:
    """Invoke eval_attribute_classifier_checkpoint.py for one task.

    Args:
        task: Short task name (e.g. 'sleeve_length').
        checkpoint: Path to the .pt checkpoint.
        label_map: Path to the label map JSON.
        jsonl: Path to the evaluation JSONL file.
        output_dir: Directory to write metrics and confusion matrix.
        arch: Model architecture string.
        img_size: Input image size.
        device: Torch device string.
        batch_size: DataLoader batch size.
        save_predictions: Whether to save per-sample predictions.

    Returns:
        True if the subprocess exited with code 0, False otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        str(_EVAL_SCRIPT),
        "--jsonl", str(jsonl),
        "--checkpoint", str(checkpoint),
        "--label-map", str(label_map),
        "--arch", arch,
        "--img-size", str(img_size),
        "--device", device,
        "--batch-size", str(batch_size),
        "--output-dir", str(output_dir),
    ]
    if save_predictions:
        cmd.append("--save-predictions")

    print(f"\n{'=' * 60}")
    print(f"[TASK] {task}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  jsonl      : {jsonl}")
    print(f"  output     : {output_dir}")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"[ERROR] {task} failed (exit {result.returncode})")
        return False
    return True


def _load_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics.json produced by eval_attribute_classifier_checkpoint.py.

    Args:
        output_dir: Task output directory.

    Returns:
        Metrics dict, or None if file missing.
    """
    path = output_dir / "metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_confusion_matrix(
    cm_path: Path,
) -> Tuple[List[str], List[List[int]]]:
    """Load the CSV confusion matrix produced by eval_attribute_classifier_checkpoint.py.

    CSV format:
        header row: gt/pred, 0:ClassA, 1:ClassB, ...
        data rows:  0:ClassA, count00, count01, ...

    Args:
        cm_path: Path to confusion_matrix.csv.

    Returns:
        Tuple of (class_names, matrix) where matrix[gt][pred] = count.
    """
    with cm_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        class_names = [
            cell.split(":", 1)[1] if ":" in cell else cell
            for cell in header[1:]
        ]
        matrix: List[List[int]] = []
        for row in reader:
            matrix.append([int(x) for x in row[1:]])
    return class_names, matrix


def _find_top_errors(
    class_names: List[str],
    matrix: List[List[int]],
    top_n: int,
) -> List[str]:
    """Return a sorted list of the most frequent misclassification pairs.

    Args:
        class_names: List of class label strings.
        matrix: Confusion matrix where matrix[gt][pred] = count.
        top_n: How many pairs to return.

    Returns:
        List of strings describing top error pairs, e.g.
        'GT=No Sleeve → PRED=Sleeveless (142×)'.
    """
    errors: List[Tuple[int, str, str]] = []
    for gt_idx, row in enumerate(matrix):
        total_gt = sum(row)
        for pred_idx, count in enumerate(row):
            if gt_idx != pred_idx and count > 0:
                pct = count / total_gt * 100 if total_gt > 0 else 0.0
                errors.append((count, class_names[gt_idx], class_names[pred_idx], pct))

    errors.sort(key=lambda x: x[0], reverse=True)

    return [
        f"GT={gt} → PRED={pred}  ({n}× / {p:.1f}% of GT class)"
        for n, gt, pred, p in errors[:top_n]
    ]


def _class_recalls(
    class_names: List[str],
    matrix: List[List[int]],
) -> List[Tuple[str, float]]:
    """Compute per-class recall from the confusion matrix.

    Args:
        class_names: Class labels.
        matrix: Confusion matrix where matrix[gt][pred] = count.

    Returns:
        List of (class_name, recall) sorted by recall ascending.
    """
    recalls = []
    for gt_idx, row in enumerate(matrix):
        total = sum(row)
        recall = matrix[gt_idx][gt_idx] / total if total > 0 else 0.0
        recalls.append((class_names[gt_idx], recall))
    recalls.sort(key=lambda x: x[1])
    return recalls


def _write_summary_markdown(
    rows: List[Dict[str, Any]],
    output_path: Path,
    split: str,
    top_errors: int,
) -> None:
    """Write a consolidated Markdown evaluation report.

    Args:
        rows: List of per-task result dicts.
        output_path: Where to write the Markdown file.
        split: 'val' or 'test'.
        top_errors: Number of top error pairs shown per task.
    """
    lines = [
        "# FashionAI Attribute Classifier — Batch Evaluation Report",
        "",
        f"> Split: **{split}** | PRD target: macro-F1 ≥ {PRD_MACRO_F1_TARGET}",
        "",
        "## Summary Table",
        "",
        "| Task | Classes | Samples | Accuracy | Macro-F1 | Weighted-F1 | Gap-to-PRD | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in rows:
        task = r["task"]
        f1 = r.get("macro_f1")
        if f1 is None:
            lines.append(f"| {task} | — | — | — | — | — | — | FAILED/SKIPPED |")
            continue
        gap = f1 - PRD_MACRO_F1_TARGET
        status = ":white_check_mark: OK" if gap >= 0 else f":x: {gap:+.3f}"
        lines.append(
            f"| {task} | {r.get('num_classes', '—')} | {r.get('num_rows', '—')} "
            f"| {r.get('accuracy', 0):.3f} | {f1:.3f} | {r.get('weighted_f1', 0):.3f} "
            f"| {gap:+.3f} | {status} |"
        )

    lines += ["", f"## Top-{top_errors} Misclassification Pairs", ""]

    for r in rows:
        task = r["task"]
        f1 = r.get("macro_f1")
        if f1 is None:
            continue
        lines.append(f"### {task}  (macro-F1 = {f1:.3f})")
        lines.append("")
        top_err = r.get("top_errors", [])
        if top_err:
            for err in top_err:
                lines.append(f"- {err}")
        else:
            lines.append("_No error data available._")
        lines.append("")
        per_cls = r.get("per_class_recall", [])
        if per_cls:
            lines.append("**Per-class recall (sorted ascending):**")
            lines.append("")
            lines.append("| Class | Recall |")
            lines.append("|---|---:|")
            for cls, rec in per_cls:
                lines.append(f"| {cls} | {rec:.3f} |")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary_table(rows: List[Dict[str, Any]]) -> None:
    """Print a compact summary table to stdout."""
    header = f"{'Task':<22} {'Macro-F1':>10} {'Accuracy':>10} {'W-F1':>8} {'Gap':>8}"
    print("\n" + "=" * 65)
    print("ATTRIBUTE EVAL SUMMARY")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for r in rows:
        f1 = r.get("macro_f1")
        if f1 is None:
            print(f"{r['task']:<22} {'FAILED/SKIPPED':>10}")
            continue
        acc = r.get("accuracy", 0.0)
        wf1 = r.get("weighted_f1", 0.0)
        gap = f1 - PRD_MACRO_F1_TARGET
        print(f"{r['task']:<22} {f1:>10.3f} {acc:>10.3f} {wf1:>8.3f} {gap:>+8.3f}")
    print("=" * 65)
    print(f"PRD target: macro-F1 ≥ {PRD_MACRO_F1_TARGET}")


def main() -> None:
    """Entry point: run batch attribute evaluation for all configured tasks."""
    args = parse_args()

    config = load_inference_config(args.inference_config)
    tasks_config: Dict[str, Any] = config.get("tasks", {})

    tasks_to_run = args.tasks if args.tasks is not None else TASK_ORDER
    summary_rows: List[Dict[str, Any]] = []

    for task in tasks_to_run:
        if task not in tasks_config:
            print(f"[SKIP] '{task}' not found in {args.inference_config}")
            summary_rows.append({"task": task})
            continue

        task_cfg = tasks_config[task]
        checkpoint = _PROJECT_ROOT / task_cfg["checkpoint"]
        label_map = _PROJECT_ROOT / task_cfg["label_map"]
        jsonl = args.jsonl_root / f"{task}_{args.split}.jsonl"
        output_dir = args.output_root / task
        arch = task_cfg.get("arch", "resnet18")
        img_size = int(task_cfg.get("img_size", 224))

        # Pre-flight checks
        if not checkpoint.exists():
            print(f"[SKIP] Checkpoint not found: {checkpoint}")
            summary_rows.append({"task": task})
            continue
        if not label_map.exists():
            print(f"[SKIP] Label map not found: {label_map}")
            summary_rows.append({"task": task})
            continue
        if not jsonl.exists():
            print(f"[SKIP] JSONL not found: {jsonl}")
            summary_rows.append({"task": task})
            continue

        success = _run_single_task(
            task=task,
            checkpoint=checkpoint,
            label_map=label_map,
            jsonl=jsonl,
            output_dir=output_dir,
            arch=arch,
            img_size=img_size,
            device=args.device,
            batch_size=args.batch_size,
            save_predictions=args.save_predictions,
        )

        metrics = _load_metrics(output_dir) if success else None
        row: Dict[str, Any] = {"task": task}
        if metrics:
            row.update(metrics)

        # Load confusion matrix and compute error analysis
        cm_path = output_dir / "confusion_matrix.csv"
        if cm_path.exists():
            try:
                class_names, matrix = _load_confusion_matrix(cm_path)
                row["top_errors"] = _find_top_errors(class_names, matrix, args.top_errors)
                row["per_class_recall"] = _class_recalls(class_names, matrix)
            except Exception as exc:
                print(f"[WARN] Could not parse confusion matrix for {task}: {exc}")

        summary_rows.append(row)

    # Output
    summary_md = args.output_root / "eval_summary.md"
    _write_summary_markdown(summary_rows, summary_md, args.split, args.top_errors)
    _print_summary_table(summary_rows)

    print(f"\nSaved summary: {summary_md}")
    print(f"Per-task outputs: {args.output_root}/{{task}}/")


if __name__ == "__main__":
    main()
