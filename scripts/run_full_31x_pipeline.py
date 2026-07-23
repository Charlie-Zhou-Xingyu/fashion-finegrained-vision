#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full 3.1.x pipeline runner — YOLO → SAM-HQ → Landmark → Region Crops → Masked Crops → Attributes.

Runs the complete 6-stage garment analysis pipeline on a batch of images using the
function-call ``GarmentPipeline`` API (not subprocess), with attribute inference enabled.

Usage::

    conda activate fashion-demo2
    python scripts/run_full_31x_pipeline.py \
        --image-dir D:/Aliintern/fashion-ai-data/deepfashion2/validation/image \
        --output-dir outputs/full_31x_demo \
        --num-images 35 \
        --seed 42

Output structure per image::

    outputs/full_31x_demo/
        000001/
            01_yolo/          detections.json, visualizations/
            02_samhq/         segmentation_results.json, overlays/, masks/
            03_landmarks/     landmarks_results.json
            04_region_crops/  region_crops.json, crops/
            05_region_masked_crops/  region_masked_crops.json, masked_crops/
            06_attributes/    predictions.jsonl
            pipeline_summary.json
        ...
        batch_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Image picker
# ---------------------------------------------------------------------------

FINE_TO_COARSE: dict[str, str] = {
    "short sleeve top": "top",
    "long sleeve top": "top",
    "vest": "top",
    "sling": "top",
    "short sleeve outwear": "outerwear",
    "long sleeve outwear": "outerwear",
    "shorts": "pants",
    "trousers": "pants",
    "skirt": "skirt",
    "short sleeve dress": "dress",
    "long sleeve dress": "dress",
    "vest dress": "dress",
    "sling dress": "dress",
}


def _pick_diverse_images(
    image_dir: Path,
    anno_dir: Path,
    num_images: int,
    seed: int,
) -> list[Path]:
    """Pick *num_images* images covering all 5 garment coarse classes.

    Scans annotation JSONs to identify garment categories, then selects images
    proportionally across the 13 fine classes so that tops, outerwear, pants,
    skirts, and dresses are all represented.

    Args:
        image_dir: Directory containing ``.jpg`` images.
        anno_dir: Directory containing matching ``.json`` annotation files.
        num_images: Target number of images to pick.
        seed: Random seed for reproducibility.

    Returns:
        List of absolute paths to selected image files.
    """
    random.seed(seed)

    # Collect up to 4 images per fine class from the first 3000 annotations.
    cat_samples: dict[str, list[str]] = defaultdict(list)
    max_per_cat = max(3, num_images // 13 + 1)
    anno_files = sorted(os.listdir(anno_dir))[:3000]
    random.shuffle(anno_files)

    for fn in anno_files:
        anno_path = anno_dir / fn
        try:
            data: dict[str, Any] = json.loads(anno_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        for key in ("item1", "item2"):
            item = data.get(key)
            if not isinstance(item, dict):
                continue
            cat = item.get("category_name", "")
            if cat in FINE_TO_COARSE and len(cat_samples[cat]) < max_per_cat:
                img_name = fn.replace(".json", ".jpg")
                img_path = image_dir / img_name
                if img_path.exists():
                    cat_samples[cat].append(str(img_path))

    # Flatten all collected images, shuffle, and take num_images.
    all_images: list[Path] = []
    for imgs in cat_samples.values():
        all_images.extend(Path(p) for p in imgs)

    random.shuffle(all_images)
    picked = all_images[:num_images]

    # Report distribution.
    coarse_counts: dict[str, int] = defaultdict(int)
    for p in picked:
        # Read annotation to get the actual coarse class.
        anno_path = anno_dir / f"{p.stem}.json"
        try:
            data = json.loads(anno_path.read_text(encoding="utf-8"))
            for key in ("item1", "item2"):
                item = data.get(key)
                if isinstance(item, dict):
                    fine = item.get("category_name", "")
                    coarse = FINE_TO_COARSE.get(fine, "unknown")
                    coarse_counts[coarse] += 1
        except Exception:
            pass

    print(f"Picked {len(picked)} images:")
    for coarse in ("top", "outerwear", "pants", "skirt", "dress"):
        print(f"  {coarse}: {coarse_counts.get(coarse, 0)}")

    return picked


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_batch(
    image_paths: list[Path],
    output_dir: Path,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    attribute_topk: int = 3,
) -> dict[str, Any]:
    """Run the full 6-stage pipeline on *image_paths*.

    Each image is processed independently via ``GarmentPipeline.run_image()``.
    Errors on individual images are caught and recorded; they do not stop the
    batch.

    Args:
        image_paths: List of absolute paths to input images.
        output_dir: Root directory for all outputs.
        yolo_conf: YOLO confidence threshold.
        yolo_iou: YOLO NMS IoU threshold.
        attribute_topk: Number of top-k attribute predictions to return.

    Returns:
        Batch summary dict with keys: ``status``, ``total``, ``success``,
        ``failed``, ``total_time_s``, ``per_image``.
    """
    # Lazy import so argparse --help is fast.
    from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    config = GarmentPipelineConfig(
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        run_attribute_inference=True,
        attribute_device="cuda",
        attribute_topk=attribute_topk,
    )

    pipeline = GarmentPipeline(config)

    per_image: list[dict[str, Any]] = []
    n_success = 0
    n_failed = 0
    t_start = time.perf_counter()

    for idx, img_path in enumerate(image_paths):
        img_stem = img_path.stem
        img_out = output_dir / img_stem
        t_img = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(image_paths)}] {img_path.name}  ({_now()})")
        print(f"{'='*60}")

        try:
            result = pipeline.run_image(str(img_path), str(img_out))

            elapsed = time.perf_counter() - t_img
            attr_summary = result.get("attributes_summary", {})
            detections = _count_detections(img_out)

            print(f"  ✓ Done in {elapsed:.1f}s")
            print(f"  Garments detected: {detections}")
            print(f"  Attributes: {attr_summary.get('num_with_attributes', 0)}/"
                  f"{attr_summary.get('num_instances', 0)} instances")

            per_image.append({
                "image": img_path.name,
                "output_dir": str(img_out),
                "status": "success",
                "elapsed_s": round(elapsed, 1),
                "garments_detected": detections,
                "attributes": attr_summary,
            })
            n_success += 1

        except Exception as exc:
            elapsed = time.perf_counter() - t_img
            trace = traceback.format_exc()
            print(f"  ✗ FAILED in {elapsed:.1f}s: {exc}")
            print(f"  {trace[:300]}")

            per_image.append({
                "image": img_path.name,
                "output_dir": str(img_out),
                "status": "failed",
                "elapsed_s": round(elapsed, 1),
                "error": repr(exc),
                "traceback": trace,
            })
            n_failed += 1

    total_elapsed = time.perf_counter() - t_start

    summary: dict[str, Any] = {
        "task": "run_full_31x_pipeline",
        "status": "success" if n_failed == 0 else "partial",
        "started_at": _now(),
        "total_images": len(image_paths),
        "success": n_success,
        "failed": n_failed,
        "total_time_s": round(total_elapsed, 1),
        "avg_time_per_image_s": round(total_elapsed / len(image_paths), 1) if image_paths else 0,
        "config": {
            "yolo_conf": yolo_conf,
            "yolo_iou": yolo_iou,
            "attribute_topk": attribute_topk,
        },
        "per_image": per_image,
    }

    # Save batch summary.
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {n_success} success, {n_failed} failed "
          f"in {total_elapsed:.1f}s")
    print(f"Summary: {summary_path}")

    return summary


def _count_detections(img_out: Path) -> int:
    """Count YOLO detections from an image's pipeline output."""
    det_json = img_out / "01_yolo" / "detections.json"
    if not det_json.exists():
        return 0
    try:
        data = json.loads(det_json.read_text(encoding="utf-8"))
        # Format: {"task": "...", "images": [{"image": "...", "detections": [...]}]}
        images = data.get("images", [])
        total = 0
        for img in images:
            dets = img.get("detections", [])
            if isinstance(dets, list):
                total += len(dets)
        return total
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full 6-stage fashion garment analysis pipeline with attribute inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing input images (JPG/PNG).",
    )
    parser.add_argument(
        "--anno-dir",
        type=Path,
        default=None,
        help="Directory containing DeepFashion2 annotation JSONs (auto-detected).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/full_31x_demo"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=35,
        help="Number of images to process (0 = all images in directory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.7,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--attribute-topk",
        type=int,
        default=3,
        help="Number of top-k attribute predictions per task.",
    )
    parser.add_argument(
        "--single-image",
        type=Path,
        default=None,
        help="Process a single image instead of batch (overrides --image-dir).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Single-image mode.
    if args.single_image:
        image_paths = [args.single_image.resolve()]
        output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/single_test")
        print(f"Single-image mode: {image_paths[0]}")
    else:
        image_dir = Path(args.image_dir).resolve()
        if not image_dir.is_dir():
            print(f"ERROR: image-dir not found: {image_dir}", file=sys.stderr)
            return 1

        # Auto-detect annotation directory.
        anno_dir = args.anno_dir
        if anno_dir is None:
            anno_candidate = image_dir.parent / "annos"
            if anno_candidate.is_dir():
                anno_dir = anno_candidate
            else:
                # No annotations available — pick images randomly.
                anno_dir = None

        output_dir = Path(args.output_dir).resolve()

        if anno_dir and anno_dir.is_dir():
            image_paths = _pick_diverse_images(
                image_dir, anno_dir, args.num_images, args.seed
            )
        else:
            # Random pick from image directory.
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            all_imgs = sorted(
                p for p in image_dir.iterdir()
                if p.suffix.lower() in exts
            )
            random.seed(args.seed)
            random.shuffle(all_imgs)
            image_paths = all_imgs[: args.num_images] if args.num_images > 0 else all_imgs
            print(f"Picked {len(image_paths)} images randomly (no annotations available).")

    if not image_paths:
        print("ERROR: No images found.", file=sys.stderr)
        return 1

    summary = run_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        attribute_topk=args.attribute_topk,
    )

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
