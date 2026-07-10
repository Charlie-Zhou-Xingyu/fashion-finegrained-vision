#!/usr/bin/env python3
"""
Run Fashionpedia balanced YOLO on random FashionAI images and save visualizations.

Samples N random images from a source dir, runs 19-class part detector,
and saves annotated images with bounding boxes.

Usage:
    python scripts/visualize_fashionpedia_on_fashionai.py \
        --image-dir "D:/Aliintern/fashion-ai-data/fashionai_attributes/round1_fashionAI_attributes_test_a/Images/collar_design_labels" \
        --model models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt \
        --num-samples 50 \
        --output-dir outputs/fashionpedia_balanced_collar_viz \
        --device cuda
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 19 Fashionpedia part class colors (BGR)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (192, 192, 0),
    (192, 0, 192), (0, 192, 192), (128, 128, 128),
]


def draw_boxes(image: np.ndarray, results, class_names: dict, conf: float = 0.25) -> np.ndarray:
    """Draw YOLO detection boxes on image."""
    img = image.copy()
    if results[0].boxes is None:
        return img

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, score, cls_id in zip(boxes, scores, cls_ids):
        if score < conf:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(cls_id, cls_id)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--model", "-m", type=Path, required=True)
    ap.add_argument("--num-samples", "-n", type=int, default=50)
    ap.add_argument("--output-dir", "-o", type=Path, default=Path("outputs/fashionpedia_balanced_collar_viz"))
    ap.add_argument("--device", "-d", type=str, default="cuda")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ── Gather images ──
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(args.image_dir.glob(ext)))
    if not image_paths:
        print(f"[ERROR] No images found in {args.image_dir}")
        return

    sampled = random.sample(image_paths, min(args.num_samples, len(image_paths)))
    print(f"Sampled {len(sampled)} / {len(image_paths)} images from {args.image_dir}")

    # ── Load model ──
    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))
    class_names = model.names
    print(f"  Classes: {class_names}")

    # ── Output dirs ──
    viz_dir = args.output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-class counts ──
    det_counts: dict[str, int] = {name: 0 for name in class_names.values()}
    images_with_detections = 0

    # ── Run inference ──
    print(f"\nRunning inference on {len(sampled)} images...")
    for i, img_path in enumerate(sampled):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{i+1:3d}/{len(sampled)}] SKIP (unreadable): {img_path.name}")
            continue

        results = model(img_path, device=args.device, verbose=False, imgsz=640, conf=args.conf)

        has_det = results[0].boxes is not None and len(results[0].boxes) > 0
        if has_det:
            images_with_detections += 1
            boxes = results[0].boxes
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()
            for cid, s in zip(cls_ids, scores):
                name = class_names.get(cid, str(cid))
                det_counts[name] = det_counts.get(name, 0) + 1

            annotated = draw_boxes(image, results, class_names, args.conf)
            out_path = viz_dir / f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(out_path), annotated)

        if (i + 1) % 10 == 0 or i == len(sampled) - 1:
            print(f"  [{i+1:3d}/{len(sampled)}] processed, {images_with_detections} with detections")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images    : {len(sampled)}")
    print(f"With detections : {images_with_detections} ({100*images_with_detections/len(sampled):.1f}%)")
    print(f"Output dir      : {viz_dir}")

    if det_counts:
        print(f"\nPer-class detection counts (conf > {args.conf}):")
        for name, count in sorted(det_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 40)
            print(f"  {name:12s} {count:4d}  {bar}")

    # Write summary txt
    with open(args.output_dir / "summary.txt", "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Image dir: {args.image_dir}\n")
        f.write(f"Samples: {len(sampled)}\n")
        f.write(f"With detections: {images_with_detections}\n\n")
        f.write("Per-class counts:\n")
        for name, count in sorted(det_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {name}: {count}\n")

    print(f"\nDone. Summary saved to {args.output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
