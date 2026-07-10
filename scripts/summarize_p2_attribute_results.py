# scripts/summarize_p2_attribute_results.py
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TASKS = [
    "sleeve_length",
    "pant_length",
    "neckline_design",
    "collar_design",
    "neck_design",
    "lapel_design",
    "skirt_length",
    "coat_length",
]


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_output_dir(outputs_root: Path, task: str, arch: str, seed: int) -> Path:
    preferred = outputs_root / f"p2_{task}_{arch}_seed{seed}"
    if preferred.exists():
        return preferred

    matches = sorted(outputs_root.glob(f"p2_{task}_*"))
    if matches:
        return matches[0]

    return preferred


def get_num_classes(label_map: Optional[Dict[str, Any]]) -> Optional[int]:
    if not label_map:
        return None

    if "num_classes" in label_map:
        return int(label_map["num_classes"])

    if "id_to_label" in label_map:
        return len(label_map["id_to_label"])

    if "idx_to_label" in label_map:
        return len(label_map["idx_to_label"])

    if "classes" in label_map and isinstance(label_map["classes"], list):
        return len(label_map["classes"])

    if "label_to_id" in label_map:
        return len(label_map["label_to_id"])

    numeric_keys = [k for k in label_map.keys() if str(k).isdigit()]
    if numeric_keys:
        return len(numeric_keys)

    return None


def pct(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x * 100:.2f}%"


def num(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:.4f}"


def build_row(
    task: str,
    output_dir: Path,
    index_dir: Path,
) -> Dict[str, Any]:
    test_metrics = load_json_if_exists(output_dir / "test_metrics.json")
    stats = load_json_if_exists(index_dir / f"stats_{task}.json")
    label_map = load_json_if_exists(index_dir / f"label_map_{task}.json")

    split_distribution = {}
    if stats:
        split_distribution = stats.get("split_distribution", {}) or {}

    row = {
        "task": task,
        "status": "ok" if test_metrics is not None else "missing_metrics",
        "num_classes": get_num_classes(label_map),
        "valid_samples": stats.get("valid_samples") if stats else None,
        "train_samples": split_distribution.get("train"),
        "val_samples": split_distribution.get("val"),
        "test_samples": split_distribution.get("test"),
        "loss": test_metrics.get("loss") if test_metrics else None,
        "accuracy": test_metrics.get("accuracy") if test_metrics else None,
        "macro_precision": test_metrics.get("macro_precision") if test_metrics else None,
        "macro_recall": test_metrics.get("macro_recall") if test_metrics else None,
        "macro_f1": test_metrics.get("macro_f1") if test_metrics else None,
        "weighted_f1": test_metrics.get("weighted_f1") if test_metrics else None,
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "test_metrics.json"),
        "stats_path": str(index_dir / f"stats_{task}.json"),
        "label_map_path": str(index_dir / f"label_map_{task}.json"),
    }

    return row


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task",
        "status",
        "num_classes",
        "valid_samples",
        "train_samples",
        "val_samples",
        "test_samples",
        "loss",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "output_dir",
        "metrics_path",
        "stats_path",
        "label_map_path",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get("macro_f1") is not None, r.get("macro_f1") or -1),
        reverse=True,
    )

    lines = []
    lines.append("# P2 FashionAI Attribute Baseline Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Task | Classes | Valid | Train | Val | Test | Accuracy | Macro-F1 | Weighted-F1 | Loss |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            "| {task} | {num_classes} | {valid_samples} | {train_samples} | {val_samples} | {test_samples} | {accuracy} | {macro_f1} | {weighted_f1} | {loss} |".format(
                task=r["task"],
                num_classes=r.get("num_classes") if r.get("num_classes") is not None else "",
                valid_samples=r.get("valid_samples") if r.get("valid_samples") is not None else "",
                train_samples=r.get("train_samples") if r.get("train_samples") is not None else "",
                val_samples=r.get("val_samples") if r.get("val_samples") is not None else "",
                test_samples=r.get("test_samples") if r.get("test_samples") is not None else "",
                accuracy=pct(r.get("accuracy")),
                macro_f1=pct(r.get("macro_f1")),
                weighted_f1=pct(r.get("weighted_f1")),
                loss=num(r.get("loss")),
            )
        )

    lines.append("")
    lines.append("## Ranking by Macro-F1")
    lines.append("")
    lines.append("| Rank | Task | Macro-F1 | Accuracy | Weighted-F1 |")
    lines.append("|---:|---|---:|---:|---:|")

    rank = 1
    for r in sorted_rows:
        if r.get("macro_f1") is None:
            continue
        lines.append(
            f"| {rank} | {r['task']} | {pct(r.get('macro_f1'))} | {pct(r.get('accuracy'))} | {pct(r.get('weighted_f1'))} |"
        )
        rank += 1

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- All metrics are based on FashionAI in-domain test splits.")
    lines.append("- `sleeve_length`, `pant_length`, and `neckline_design` were completed earlier.")
    lines.append("- `collar_design`, `neck_design`, `lapel_design`, `skirt_length`, and `coat_length` were added in the current P2 completion round.")
    lines.append("- Crop-based weak evaluation is task-dependent and currently should be reported separately from FashionAI in-domain metrics.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/fashionai_attribute_index", type=str)
    parser.add_argument("--outputs-root", default="outputs", type=str)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--output-csv", default="outputs/p2_attribute_baseline_summary.csv", type=str)
    parser.add_argument("--output-md", default="outputs/p2_attribute_baseline_summary.md", type=str)
    parser.add_argument("--output-json", default="outputs/p2_attribute_baseline_summary.json", type=str)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    outputs_root = Path(args.outputs_root)

    rows = []
    for task in args.tasks:
        output_dir = find_output_dir(outputs_root, task, args.arch, args.seed)
        row = build_row(task=task, output_dir=output_dir, index_dir=index_dir)
        rows.append(row)

    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))
    write_json(rows, Path(args.output_json))

    print("[OK] P2 attribute summary generated.")
    print(f"[OK] CSV:  {args.output_csv}")
    print(f"[OK] MD:   {args.output_md}")
    print(f"[OK] JSON: {args.output_json}")

    print("\n[SUMMARY]")
    for r in rows:
        print(
            f"{r['task']}: "
            f"classes={r.get('num_classes')}, "
            f"test={r.get('test_samples')}, "
            f"acc={pct(r.get('accuracy'))}, "
            f"macro_f1={pct(r.get('macro_f1'))}, "
            f"status={r.get('status')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
