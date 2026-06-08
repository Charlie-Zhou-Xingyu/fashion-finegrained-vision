from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize region crop records."
    )
    parser.add_argument(
        "--region-crops-json",
        type=str,
        required=True,
        help="Path to region_crops.json.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output summary JSON path.",
    )
    parser.add_argument(
        "--print-fallback-records",
        action="store_true",
        help="Print fallback crop records.",
    )
    parser.add_argument(
        "--print-low-reliable-records",
        action="store_true",
        help="Print records with num_reliable_landmarks <= 1.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return float(a) / float(b)


def main() -> None:
    args = parse_args()

    path = Path(args.region_crops_json)
    data = load_json(path)

    crops = data.get("crops", [])
    if not isinstance(crops, list):
        raise RuntimeError("Invalid region_crops.json: 'crops' must be a list.")

    source_counter = Counter()
    region_counter = Counter()
    class_counter = Counter()
    success_counter = Counter()
    fallback_counter = Counter()
    reliable_count_counter = Counter()
    landmark_count_counter = Counter()

    by_class_source = defaultdict(Counter)
    by_region_source = defaultdict(Counter)
    by_class_region_source = defaultdict(lambda: defaultdict(Counter))

    fallback_records = []
    low_reliable_records = []

    for record in crops:
        class_name = str(record.get("class_name", "unknown"))
        region = str(record.get("region", "unknown"))
        source = str(record.get("source", "unknown"))
        success = bool(record.get("success", False))
        fallback = bool(record.get("fallback", False))

        try:
            num_reliable = int(record.get("num_reliable_landmarks", 0))
        except Exception:
            num_reliable = 0

        try:
            num_landmarks = int(record.get("num_landmarks", 0))
        except Exception:
            num_landmarks = 0

        source_counter[source] += 1
        region_counter[region] += 1
        class_counter[class_name] += 1
        success_counter[str(success)] += 1
        fallback_counter[str(fallback)] += 1
        reliable_count_counter[str(num_reliable)] += 1
        landmark_count_counter[str(num_landmarks)] += 1

        by_class_source[class_name][source] += 1
        by_region_source[region][source] += 1
        by_class_region_source[class_name][region][source] += 1

        if fallback:
            fallback_records.append(record)

        if num_reliable <= 1:
            low_reliable_records.append(record)

    total = len(crops)
    num_fallback = fallback_counter["True"]
    num_success = success_counter["True"]

    summary: Dict[str, Any] = {
        "region_crops_json": str(path),
        "total_records": total,
        "num_success": num_success,
        "num_fallback": num_fallback,
        "success_ratio": safe_div(num_success, total),
        "fallback_ratio": safe_div(num_fallback, total),
        "source_counts": dict(source_counter),
        "region_counts": dict(region_counter),
        "class_counts": dict(class_counter),
        "success_counts": dict(success_counter),
        "fallback_counts": dict(fallback_counter),
        "num_reliable_landmarks_counts": dict(reliable_count_counter),
        "num_landmarks_counts": dict(landmark_count_counter),
        "by_class_source": {
            k: dict(v) for k, v in sorted(by_class_source.items())
        },
        "by_region_source": {
            k: dict(v) for k, v in sorted(by_region_source.items())
        },
        "by_class_region_source": {
            class_name: {
                region: dict(counter)
                for region, counter in sorted(region_map.items())
            }
            for class_name, region_map in sorted(by_class_region_source.items())
        },
        "fallback_records": fallback_records,
        "low_reliable_records": low_reliable_records,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        save_json(summary, output_path)
        print(f"[INFO] Saved summary to: {output_path}")

    if args.print_fallback_records:
        print("[INFO] Fallback records:")
        for record in fallback_records:
            print(
                f"- image={record.get('image_path')} "
                f"class={record.get('class_name')} "
                f"det={record.get('det_id')} "
                f"region={record.get('region')} "
                f"num_lm={record.get('num_landmarks')} "
                f"num_rel={record.get('num_reliable_landmarks')} "
                f"crop={record.get('crop_path')}"
            )

    if args.print_low_reliable_records:
        print("[INFO] Low reliable records:")
        for record in low_reliable_records:
            print(
                f"- image={record.get('image_path')} "
                f"class={record.get('class_name')} "
                f"det={record.get('det_id')} "
                f"region={record.get('region')} "
                f"source={record.get('source')} "
                f"num_lm={record.get('num_landmarks')} "
                f"num_rel={record.get('num_reliable_landmarks')} "
                f"crop={record.get('crop_path')}"
            )


if __name__ == "__main__":
    main()
