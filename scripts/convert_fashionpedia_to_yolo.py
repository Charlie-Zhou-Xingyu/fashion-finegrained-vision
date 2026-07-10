#!/usr/bin/env python3
"""
Convert Fashionpedia HuggingFace parquet dataset to YOLO detection format.

Reads train/val parquet files from E:/fashionpedia/data/, extracts images and
bounding-box annotations for selected categories, and writes:

    E:/fashionpedia_yolo/
        images/train/  *.jpg
        images/val/    *.jpg
        labels/train/  *.txt   (YOLO format: class cx cy w h, normalized)
        labels/val/    *.txt
        fashionpedia_parts.yaml  (dataset config for YOLOv8)

Default target categories are the apparel parts useful for 3.1.2 part detection.
Adjust TARGET_CATS to add/remove categories.

Usage:
    python scripts/convert_fashionpedia_to_yolo.py
    python scripts/convert_fashionpedia_to_yolo.py --cats zipper pocket collar
    python scripts/convert_fashionpedia_to_yolo.py --src E:/fashionpedia --dst E:/fashionpedia_yolo
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

# ── Category config ───────────────────────────────────────────────────────────

# All 46 Fashionpedia categories (index = original ID)
ALL_CATEGORIES: list[str] = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress",
    "jumpsuit", "cape", "glasses", "hat",
    "headband, head covering, hair accessory", "tie", "glove", "watch",
    "belt", "leg warmer", "tights, stockings", "sock", "shoe", "bag, wallet",
    "scarf", "umbrella",
    # apparel parts (IDs 27-45)
    "hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline",
    "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe",
    "ribbon", "rivet", "ruffle", "sequin", "tassel",
]

# Default: parts that map directly to 3.1.2 pipeline targets
DEFAULT_TARGET_CATS: list[str] = [
    "zipper",    # 35 — dedicated detector to replace DINO (complete failure)
    "pocket",    # 32 — dedicated detector to improve precision
    "collar",    # 28 — supplement landmark fast path
    "hood",      # 27
    "sleeve",    # 31
    "bow",       # 38
    "fringe",    # 40
    "ruffle",    # 43
    "sequin",    # 44
]


def _build_cat_map(target_names: list[str]) -> dict[int, int]:
    """Map original Fashionpedia category IDs → contiguous YOLO class IDs."""
    name_to_orig = {n: i for i, n in enumerate(ALL_CATEGORIES)}
    result: dict[int, int] = {}
    for new_id, name in enumerate(target_names):
        if name not in name_to_orig:
            raise ValueError(f"Unknown category: '{name}'. Choose from: {ALL_CATEGORIES}")
        result[name_to_orig[name]] = new_id
    return result


def _convert_split(
    parquet_paths: list[Path],
    images_dir: Path,
    labels_dir: Path,
    cat_map: dict[int, int],
    split: str,
) -> dict[str, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "labels": 0, "annotations": 0, "skipped_no_target": 0}

    for pq_path in parquet_paths:
        print(f"  reading {pq_path.name} ...")
        df = pd.read_parquet(pq_path)

        for _, row in df.iterrows():
            img_id = row["image_id"]
            W, H = row["width"], row["height"]
            objects = row["objects"]

            # Filter to target categories
            lines: list[str] = []
            for cat_id, bbox in zip(objects["category"], objects["bbox"]):
                if cat_id not in cat_map:
                    continue
                yolo_cls = cat_map[cat_id]
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                stats["skipped_no_target"] += 1
                continue

            # Save image
            img_path = images_dir / f"{img_id}.jpg"
            if not img_path.exists():
                img_bytes = row["image"]["bytes"]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(img_path, quality=95)
                stats["images"] += 1

            # Save label
            label_path = labels_dir / f"{img_id}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")
            stats["labels"] += 1
            stats["annotations"] += len(lines)

    print(f"  [{split}] images={stats['images']} labels={stats['labels']} "
          f"annotations={stats['annotations']} skipped={stats['skipped_no_target']}")
    return stats


def _write_yaml(dst: Path, target_names: list[str]) -> None:
    yaml_lines = [
        f"path: {dst}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(target_names)}",
        "names:",
    ]
    for name in target_names:
        yaml_lines.append(f"  - \"{name}\"")
    (dst / "fashionpedia_parts.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")


def main(src: Path, dst: Path, target_names: list[str]) -> None:
    cat_map = _build_cat_map(target_names)
    print(f"Target categories ({len(target_names)}):")
    orig_id_map = {v: k for k, v in {i: n for i, n in enumerate(ALL_CATEGORIES)}.items()}
    for new_id, name in enumerate(target_names):
        orig_id = next(k for k, v in cat_map.items() if v == new_id)
        print(f"  {new_id}: {name}  (fashionpedia id={orig_id})")
    print()

    train_parquets = sorted((src / "data").glob("train-*.parquet"))
    val_parquets = sorted((src / "data").glob("val-*.parquet"))

    print(f"Train parquets: {len(train_parquets)}, Val parquets: {len(val_parquets)}")

    _convert_split(train_parquets, dst / "images/train", dst / "labels/train", cat_map, "train")
    _convert_split(val_parquets,   dst / "images/val",   dst / "labels/val",   cat_map, "val")

    _write_yaml(dst, target_names)
    print(f"\nDone. Dataset config: {dst / 'fashionpedia_parts.yaml'}")
    print("Train with: yolo detect train data=E:/fashionpedia_yolo/fashionpedia_parts.yaml model=yolov8n.pt epochs=50")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("E:/fashionpedia"),
                    help="Root of downloaded Fashionpedia dataset")
    ap.add_argument("--dst", type=Path, default=Path("E:/fashionpedia_yolo"),
                    help="Output directory for YOLO-format dataset")
    ap.add_argument("--cats", nargs="+", default=None,
                    help="Category names to include (default: 9 parts for 3.1.2)")
    args = ap.parse_args()
    main(args.src, args.dst, args.cats or DEFAULT_TARGET_CATS)
