from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize mask quality fields in landmark results JSON."
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
    return parser.parse_args()


def safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return float(a) / float(b)


def percentile(values: List[float], p: float) -> float | None:
    if not values:
        return None

    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return float(values_sorted[0])

    rank = (len(values_sorted) - 1) * p
    low = int(rank)
    high = min(low + 1, len(values_sorted) - 1)
    weight = rank - low

    return float(values_sorted[low] * (1.0 - weight) + values_sorted[high] * weight)


def main() -> None:
    args = parse_args()

    path = Path(args.landmarks_json)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    quality_counter = Counter()
    inside_counter = Counter()
    refined_counter = Counter()
    valid_for_class_counter = Counter()

    class_quality = defaultdict(Counter)
    class_inside = defaultdict(Counter)
    class_refined = defaultdict(Counter)
    class_valid = defaultdict(Counter)

    total_landmarks = 0
    total_instances = 0
    total_images = 0

    distance_values: List[float] = []

    for image_record in data.get("images", []):
        total_images += 1

        for instance in image_record.get("instances", []):
            total_instances += 1
            class_name = str(
                instance.get("class_name", instance.get("category_name", "unknown"))
            )

            landmarks = instance.get("landmarks", [])
            if not isinstance(landmarks, list):
                continue

            for landmark in landmarks:
                total_landmarks += 1

                quality = str(landmark.get("quality", "missing"))
                quality_counter[quality] += 1
                class_quality[class_name][quality] += 1

                inside = landmark.get("inside_mask", None)
                inside_key = str(inside)
                inside_counter[inside_key] += 1
                class_inside[class_name][inside_key] += 1

                refined = bool(landmark.get("refined_by_mask", False))
                refined_key = str(refined)
                refined_counter[refined_key] += 1
                class_refined[class_name][refined_key] += 1

                valid_for_class = landmark.get("valid_for_class", None)
                valid_key = str(valid_for_class)
                valid_for_class_counter[valid_key] += 1
                class_valid[class_name][valid_key] += 1

                dist = landmark.get("distance_to_mask", None)
                if isinstance(dist, (int, float)):
                    distance_values.append(float(dist))

    summary = {
        "landmarks_json": str(path),
        "total_images": total_images,
        "total_instances": total_instances,
        "total_landmarks": total_landmarks,
        "quality_counts": dict(quality_counter),
        "inside_mask_counts": dict(inside_counter),
        "refined_by_mask_counts": dict(refined_counter),
        "valid_for_class_counts": dict(valid_for_class_counter),
        "quality_ratios": {
            k: safe_div(v, total_landmarks) for k, v in quality_counter.items()
        },
        "inside_mask_ratios": {
            k: safe_div(v, total_landmarks) for k, v in inside_counter.items()
        },
        "refined_by_mask_ratios": {
            k: safe_div(v, total_landmarks) for k, v in refined_counter.items()
        },
        "valid_for_class_ratios": {
            k: safe_div(v, total_landmarks)
            for k, v in valid_for_class_counter.items()
        },
        "distance_to_mask": {
            "count": len(distance_values),
            "mean": sum(distance_values) / len(distance_values)
            if distance_values
            else None,
            "min": min(distance_values) if distance_values else None,
            "p50": percentile(distance_values, 0.50),
            "p75": percentile(distance_values, 0.75),
            "p90": percentile(distance_values, 0.90),
            "p95": percentile(distance_values, 0.95),
            "max": max(distance_values) if distance_values else None,
        },
        "by_class": {},
    }

    for class_name in sorted(class_quality.keys()):
        class_total = sum(class_quality[class_name].values())
        summary["by_class"][class_name] = {
            "total_landmarks": class_total,
            "quality_counts": dict(class_quality[class_name]),
            "inside_mask_counts": dict(class_inside[class_name]),
            "refined_by_mask_counts": dict(class_refined[class_name]),
            "valid_for_class_counts": dict(class_valid[class_name]),
            "quality_ratios": {
                k: safe_div(v, class_total)
                for k, v in class_quality[class_name].items()
            },
            "inside_mask_ratios": {
                k: safe_div(v, class_total)
                for k, v in class_inside[class_name].items()
            },
            "refined_by_mask_ratios": {
                k: safe_div(v, class_total)
                for k, v in class_refined[class_name].items()
            },
            "valid_for_class_ratios": {
                k: safe_div(v, class_total)
                for k, v in class_valid[class_name].items()
            },
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
