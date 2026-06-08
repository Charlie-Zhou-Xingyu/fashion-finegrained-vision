from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple landmark mask quality summary JSON files."
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        nargs="+",
        required=True,
        help="One or more mask_quality_summary.json files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ratio_to_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "N/A"


def get_count(data: Dict[str, Any], section: str, key: str) -> int:
    value = data.get(section, {}).get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


def get_ratio(data: Dict[str, Any], section: str, key: str) -> float:
    value = data.get(section, {}).get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, Any]] = []

    for path_str in args.summary_json:
        path = Path(path_str)
        data = load_json(path)

        name = path.parent.name

        total_landmarks = int(data.get("total_landmarks", 0))

        row = {
            "name": name,
            "total": total_landmarks,
            "ok": get_count(data, "quality_counts", "ok"),
            "outside_mask": get_count(data, "quality_counts", "outside_mask"),
            "refined_by_mask": get_count(data, "quality_counts", "refined_by_mask"),
            "outside_mask_far": get_count(data, "quality_counts", "outside_mask_far"),
            "inside_true": get_count(data, "inside_mask_counts", "True"),
            "inside_false": get_count(data, "inside_mask_counts", "False"),
            "ok_ratio": get_ratio(data, "quality_ratios", "ok"),
            "outside_mask_ratio": get_ratio(data, "quality_ratios", "outside_mask"),
            "refined_ratio": get_ratio(data, "quality_ratios", "refined_by_mask"),
            "outside_far_ratio": get_ratio(data, "quality_ratios", "outside_mask_far"),
            "inside_ratio": get_ratio(data, "inside_mask_ratios", "True"),
        }

        rows.append(row)

    print(
        "| version | total | ok | outside | refined | far | inside | ok% | refined% | far% | inside% |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for row in rows:
        print(
            f"| {row['name']} "
            f"| {row['total']} "
            f"| {row['ok']} "
            f"| {row['outside_mask']} "
            f"| {row['refined_by_mask']} "
            f"| {row['outside_mask_far']} "
            f"| {row['inside_true']} "
            f"| {ratio_to_percent(row['ok_ratio'])} "
            f"| {ratio_to_percent(row['refined_ratio'])} "
            f"| {ratio_to_percent(row['outside_far_ratio'])} "
            f"| {ratio_to_percent(row['inside_ratio'])} |"
        )


if __name__ == "__main__":
    main()
