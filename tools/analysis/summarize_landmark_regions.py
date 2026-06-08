from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize landmark region and reliability statistics."
    )
    parser.add_argument(
        "--landmarks-json",
        type=str,
        required=True,
        help="Path to landmarks_results.json.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output summary JSON path.",
    )
    parser.add_argument(
        "--max-outside-distance",
        type=float,
        default=5.0,
        help="Max distance for outside_mask landmark to be considered reliable.",
    )
    return parser.parse_args()


def is_reliable_landmark(
    landmark: Dict[str, Any],
    max_outside_distance: float,
) -> bool:
    if landmark.get("valid_for_class", True) is False:
        return False

    quality = str(landmark.get("quality", ""))

    if quality == "ok":
        return True

    if quality == "refined_by_mask":
        return True

    if quality == "outside_mask":
        distance = landmark.get("distance_to_mask", None)
        if isinstance(distance, (int, float)):
            return float(distance) <= float(max_outside_distance)

    return False


def safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return float(a) / float(b)


def main() -> None:
    args = parse_args()

    path = Path(args.landmarks_json)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    total_counter = Counter()
    region_counter = defaultdict(Counter)
    class_region_counter = defaultdict(lambda: defaultdict(Counter))

    for image_record in data.get("images", []):
        for instance in image_record.get("instances", []):
            class_name = str(
                instance.get("class_name", instance.get("category_name", "unknown"))
            )

            landmarks = instance.get("landmarks", [])
            if not isinstance(landmarks, list):
                continue

            for lm in landmarks:
                region = lm.get("region")
                if not region:
                    region = "unknown"

                quality = str(lm.get("quality", "missing"))
                reliable = is_reliable_landmark(
                    lm,
                    max_outside_distance=float(args.max_outside_distance),
                )

                total_counter["total_landmarks"] += 1
                total_counter[f"quality:{quality}"] += 1
                total_counter[f"region:{region}"] += 1

                if reliable:
                    total_counter["reliable_landmarks"] += 1

                region_counter[region]["total"] += 1
                region_counter[region][f"quality:{quality}"] += 1
                if reliable:
                    region_counter[region]["reliable"] += 1

                class_region_counter[class_name][region]["total"] += 1
                class_region_counter[class_name][region][f"quality:{quality}"] += 1
                if reliable:
                    class_region_counter[class_name][region]["reliable"] += 1

    summary: Dict[str, Any] = {
        "landmarks_json": str(path),
        "max_outside_distance": float(args.max_outside_distance),
        "total": dict(total_counter),
        "regions": {},
        "by_class": {},
    }

    total_landmarks = int(total_counter["total_landmarks"])
    total_reliable = int(total_counter["reliable_landmarks"])

    summary["total"]["reliable_ratio"] = safe_div(total_reliable, total_landmarks)

    for region in sorted(region_counter.keys()):
        stats = dict(region_counter[region])
        total = int(stats.get("total", 0))
        reliable = int(stats.get("reliable", 0))
        stats["reliable_ratio"] = safe_div(reliable, total)
        summary["regions"][region] = stats

    for class_name in sorted(class_region_counter.keys()):
        summary["by_class"][class_name] = {}
        for region in sorted(class_region_counter[class_name].keys()):
            stats = dict(class_region_counter[class_name][region])
            total = int(stats.get("total", 0))
            reliable = int(stats.get("reliable", 0))
            stats["reliable_ratio"] = safe_div(reliable, total)
            summary["by_class"][class_name][region] = stats

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
