from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a balanced image evaluation set from landmark JSONL."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to validation/train JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=20,
        help="Number of images to sample per category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy sampled images to output-dir/images.",
    )
    return parser.parse_args()


def normalize_category_name(value: Any) -> str:
    if value is None:
        return "unknown"
    name = str(value).strip().lower()
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name if name else "unknown"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception as exc:
                print(f"[WARN] Failed to parse {path}:{line_idx}: {exc}")

    return records


def main() -> None:
    args = parse_args()

    random.seed(int(args.seed))

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(jsonl_path)
    print(f"[INFO] Loaded records: {len(records)}")

    image_to_categories: Dict[str, Set[str]] = defaultdict(set)
    image_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in records:
        image_path = record.get("image_path")
        category_name = normalize_category_name(record.get("category_name"))

        if not isinstance(image_path, str) or not image_path:
            continue

        image_to_categories[image_path].add(category_name)
        image_to_records[image_path].append(record)

    category_to_images: Dict[str, List[str]] = defaultdict(list)

    for image_path, categories in image_to_categories.items():
        for category_name in categories:
            category_to_images[category_name].append(image_path)

    selected_images: Set[str] = set()
    selected_by_class: Dict[str, List[str]] = {}

    for category_name in sorted(category_to_images.keys()):
        candidates = sorted(set(category_to_images[category_name]))
        random.shuffle(candidates)

        selected = candidates[: int(args.samples_per_class)]
        selected_by_class[category_name] = selected
        selected_images.update(selected)

        print(
            f"[INFO] {category_name}: selected {len(selected)} / "
            f"{len(candidates)} candidates"
        )

    selected_images_sorted = sorted(selected_images)

    copied_records = []
    missing_images = []

    for idx, image_path_str in enumerate(selected_images_sorted, start=1):
        src_path = Path(image_path_str)

        if not src_path.exists():
            missing_images.append(image_path_str)
            continue

        if args.copy_images:
            suffix = src_path.suffix.lower()
            if not suffix:
                suffix = ".jpg"

            dst_name = f"{idx:06d}{suffix}"
            dst_path = images_dir / dst_name
            shutil.copy2(src_path, dst_path)
            output_image_path = str(dst_path)
        else:
            output_image_path = image_path_str

        copied_records.append(
            {
                "eval_image_id": idx - 1,
                "source_image_path": image_path_str,
                "image_path": output_image_path,
                "categories": sorted(image_to_categories[image_path_str]),
                "num_instances_in_jsonl": len(image_to_records[image_path_str]),
            }
        )

    manifest = {
        "jsonl": str(jsonl_path),
        "output_dir": str(output_dir),
        "samples_per_class": int(args.samples_per_class),
        "seed": int(args.seed),
        "copy_images": bool(args.copy_images),
        "num_unique_selected_images": len(selected_images_sorted),
        "num_written_images": len(copied_records),
        "num_missing_images": len(missing_images),
        "selected_by_class": selected_by_class,
        "images": copied_records,
        "missing_images": missing_images,
    }

    manifest_path = output_dir / "eval_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[INFO] Sampling finished.")
    print(f"[INFO] Unique selected images: {len(selected_images_sorted)}")
    print(f"[INFO] Written images: {len(copied_records)}")
    print(f"[INFO] Missing images: {len(missing_images)}")
    print(f"[INFO] Manifest: {manifest_path}")
    if args.copy_images:
        print(f"[INFO] Images dir: {images_dir}")


if __name__ == "__main__":
    main()
