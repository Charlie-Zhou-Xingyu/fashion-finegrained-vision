"""Batch-run GarmentPipeline (YOLO + SAM-HQ) on selected images for IoU eval.

Usage:
    python scripts/run_iou_eval_batch.py --image-ids outputs/iou_eval_100train/selected_image_ids.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", required=True, help="JSON file with image_ids list")
    parser.add_argument("--image-dir", default="D:/Aliintern/fashion-ai-data/deepfashion2/train/image")
    parser.add_argument("--output-dir", default="outputs/iou_eval_100train")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp image dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image IDs
    with open(args.image_ids) as f:
        data = json.load(f)
    image_ids = data["image_ids"]
    print(f"Running pipeline on {len(image_ids)} images...")

    image_dir = Path(args.image_dir)

    # Copy images to temp dir so run_source() works on a directory
    temp_dir = output_dir / "_temp_images"
    temp_dir.mkdir(exist_ok=True)
    copied = 0
    for img_id in image_ids:
        src = image_dir / f"{img_id}.jpg"
        if src.exists():
            shutil.copy2(src, temp_dir / f"{img_id}.jpg")
            copied += 1
    print(f"Copied {copied}/{len(image_ids)} images to {temp_dir}")

    # Configure pipeline (segmentation only, no landmarks/crops)
    config = GarmentPipelineConfig()
    config.run_landmark_and_crops = False
    config.run_attribute_inference = False
    config.save_yolo_vis = False
    config.save_yolo_crops = False

    pipeline = GarmentPipeline(config)

    result = pipeline.run_source(
        source=str(temp_dir),
        output_dir=str(output_dir),
        max_images=len(image_ids),
    )

    print(f"Pipeline done. Timing: {result.get('timing', {})}")

    # Cleanup temp dir
    if not args.keep_temp:
        shutil.rmtree(temp_dir)
        print(f"Removed temp dir: {temp_dir}")

    seg_json = output_dir / "02_samhq" / "segmentation_results.json"
    if seg_json.exists():
        print(f"Segmentation results: {seg_json}")
    else:
        print("WARNING: segmentation_results.json not found!")


if __name__ == "__main__":
    main()
