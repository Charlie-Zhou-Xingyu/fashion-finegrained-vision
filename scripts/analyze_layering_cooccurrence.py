#!/usr/bin/env python3
"""
Quick analysis: how often does YOLO detect outerwear + inner-garment (top/dress)
in the same image? This checks feasibility of inner/outer layering detection.

Usage:
    python scripts/analyze_layering_cooccurrence.py \
        --detections outputs/calibration_v3/yolo/detections.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# DeepFashion2 coarse class IDs (from configs/category_mapping.yaml)
_OUTERWEAR_IDS = {2, 3}       # short/long sleeve outwear
_INNER_IDS = {0, 1, 4, 5}     # top, vest, sling (anything coarse=top)
_DRESS_IDS = {9, 10, 11, 12}  # dresses (layering with outerwear also valid)


def main(det_path: Path) -> None:
    data = json.loads(det_path.read_text(encoding="utf-8"))
    images = data["images"]

    total = len(images)
    # Counters
    has_outer = 0
    has_outer_and_inner = 0
    has_outer_and_dress = 0
    has_multi_outer = 0
    instance_counts: Counter = Counter()

    for img in images:
        dets = img.get("detections", [])
        fine_ids = [d["fine_class_id"] for d in dets]
        coarse_ids = [d["coarse_class_id"] for d in dets]

        outer_count = sum(1 for c in fine_ids if c in _OUTERWEAR_IDS)
        inner_count = sum(1 for c in fine_ids if c in _INNER_IDS)
        dress_count = sum(1 for c in fine_ids if c in _DRESS_IDS)

        instance_counts[len(dets)] += 1

        if outer_count >= 1:
            has_outer += 1
        if outer_count >= 1 and inner_count >= 1:
            has_outer_and_inner += 1
        if outer_count >= 1 and dress_count >= 1:
            has_outer_and_dress += 1
        if outer_count >= 2:
            has_multi_outer += 1

    def pct(n: int) -> str:
        return f"{n}/{total} ({100*n/total:.1f}%)"

    print(f"\n=== Layering co-occurrence analysis ===")
    print(f"Total images:                  {total}")
    print(f"Has outerwear:                 {pct(has_outer)}")
    print(f"Outerwear + inner top:         {pct(has_outer_and_inner)}")
    print(f"Outerwear + dress:             {pct(has_outer_and_dress)}")
    print(f"Multi-outer (≥2 outwear):      {pct(has_multi_outer)}")
    print(f"\nInstance count distribution:")
    for k in sorted(instance_counts):
        print(f"  {k} instance(s): {instance_counts[k]} images")

    feasibility = has_outer_and_inner / total if total else 0
    verdict = (
        "✓ FEASIBLE (>50% co-detection)" if feasibility > 0.5
        else "△ MARGINAL (20–50%)" if feasibility > 0.2
        else "✗ LOW SIGNAL (<20%) — occlusion likely blocking inner detection"
    )
    print(f"\nLayering feasibility: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", type=Path,
                    default=Path("outputs/calibration_v3/yolo/detections.json"))
    main(ap.parse_args().detections)
