# scripts/find_region_crops_json.py
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_json_safe(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def path_exists(value: Any) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).exists()
    except Exception:
        return False


def is_region_crop_record(record: Dict[str, Any]) -> bool:
    keys = set(record.keys())

    region_like_keys = {
        "region",
        "component",
        "crop_path",
        "image_crop_path",
        "masked_crop_path",
        "mask_crop_path",
        "bbox_xyxy",
        "det_id",
        "class_name",
    }

    return len(keys.intersection(region_like_keys)) >= 3


def summarize_candidate(path: Path, data: Dict[str, Any], max_samples: int = 5) -> Dict[str, Any]:
    crops = data.get("crops", [])
    region_counter = Counter()
    component_counter = Counter()
    class_counter = Counter()
    success_counter = Counter()
    masked_success_counter = Counter()

    existing_crop_paths = 0
    existing_image_crop_paths = 0
    existing_masked_crop_paths = 0

    sample_records = []

    for idx, record in enumerate(crops):
        if not isinstance(record, dict):
            continue

        region_counter[str(record.get("region", ""))] += 1
        component_counter[str(record.get("component", ""))] += 1
        class_counter[str(record.get("class_name", ""))] += 1
        success_counter[str(record.get("success", ""))] += 1
        masked_success_counter[str(record.get("masked_success", ""))] += 1

        if path_exists(record.get("crop_path")):
            existing_crop_paths += 1
        if path_exists(record.get("image_crop_path")):
            existing_image_crop_paths += 1
        if path_exists(record.get("masked_crop_path")):
            existing_masked_crop_paths += 1

        if len(sample_records) < max_samples:
            sample_records.append({
                "region": record.get("region"),
                "component": record.get("component"),
                "class_name": record.get("class_name"),
                "success": record.get("success"),
                "masked_success": record.get("masked_success"),
                "crop_path": record.get("crop_path"),
                "image_crop_path": record.get("image_crop_path"),
                "masked_crop_path": record.get("masked_crop_path"),
            })

    return {
        "json_path": str(path),
        "num_crops": len(crops),
        "top_regions": region_counter.most_common(20),
        "top_components": component_counter.most_common(20),
        "top_classes": class_counter.most_common(20),
        "success_counts": success_counter.most_common(),
        "masked_success_counts": masked_success_counter.most_common(),
        "existing_crop_paths": existing_crop_paths,
        "existing_image_crop_paths": existing_image_crop_paths,
        "existing_masked_crop_paths": existing_masked_crop_paths,
        "sample_records": sample_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs", type=str)
    parser.add_argument("--output-json", default="outputs/region_crops_json_candidates.json", type=str)
    parser.add_argument("--min-crops", default=1, type=int)
    parser.add_argument("--max-files", default=0, type=int, help="0 means no limit")
    parser.add_argument("--print-top", default=30, type=int)
    args = parser.parse_args()

    root = Path(args.root)
    json_paths = list(root.rglob("*.json"))

    candidates: List[Dict[str, Any]] = []
    scanned = 0

    for path in json_paths:
        if args.max_files > 0 and scanned >= args.max_files:
            break
        scanned += 1

        data = load_json_safe(path)
        if not isinstance(data, dict):
            continue

        crops = data.get("crops")
        if not isinstance(crops, list):
            continue

        if len(crops) < args.min_crops:
            continue

        # Check whether records look like region crops.
        dict_records = [r for r in crops[:20] if isinstance(r, dict)]
        if not dict_records:
            continue

        if not any(is_region_crop_record(r) for r in dict_records):
            continue

        summary = summarize_candidate(path, data)
        candidates.append(summary)

    candidates.sort(
        key=lambda x: (
            x["existing_image_crop_paths"] + x["existing_crop_paths"] + x["existing_masked_crop_paths"],
            x["num_crops"],
        ),
        reverse=True,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "num_scanned_json_files": len(json_paths),
                "num_candidates": len(candidates),
                "candidates": candidates,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] scanned json files: {len(json_paths)}")
    print(f"[OK] region crop json candidates: {len(candidates)}")
    print(f"[OK] saved: {output_path}")

    print("\n[TOP CANDIDATES]")
    for i, c in enumerate(candidates[: args.print_top]):
        print(f"\n#{i}")
        print(f"path: {c['json_path']}")
        print(f"num_crops: {c['num_crops']}")
        print(f"existing image_crop/crop/masked: {c['existing_image_crop_paths']} / {c['existing_crop_paths']} / {c['existing_masked_crop_paths']}")
        print(f"regions: {c['top_regions'][:10]}")
        print(f"components: {c['top_components'][:10]}")
        print(f"classes: {c['top_classes'][:10]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
