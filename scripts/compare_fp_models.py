#!/usr/bin/env python3
"""
Quick v1 vs v2 Fashionpedia YOLO comparison on FP-core validation parts.

Runs Fashionpedia YOLO on validation images using both v1 and v2 models,
computes per-part recall@IoU>0.01, and prints side-by-side comparison.

Usage:
    python scripts/compare_fp_models.py --device cuda --max-images 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

V1_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
V2_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v2_best.pt"
YOLO_WEIGHTS = "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
PER_PART_DIR = PROJECT_ROOT / "data/validation/per_part"
ANNOTATIONS_JSON = (
    PROJECT_ROOT / "data/validation/project-10-at-2026-07-09-16-01-2504b933/result.json"
)

# FP-core parts to evaluate
FP_CORE = [
    "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline",
    "buckle", "zipper", "bow", "fringe", "ruffle", "hood",
]


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _resolve_image_path(base_filename: str) -> Optional[Path]:
    import os, re
    basename = os.path.basename(base_filename.replace("\\", "/"))
    m = re.match(r"^[^-]+-(.+?)__(\d+)\.jpg$", basename)
    if m:
        return PER_PART_DIR / m.group(1) / f"{m.group(2)}.jpg"
    m = re.match(r"^[^-]+-(.+?)__([0-9a-f]{20,})\.jpg$", basename)
    if m:
        return PER_PART_DIR / m.group(1) / f"{m.group(2)}.jpg"
    m = re.match(r"^[^-]+__[^-]+-(.+?)__(\d+)", basename)
    if m:
        return PER_PART_DIR / m.group(1) / f"{m.group(2)}.jpg"
    return None


def load_fp_images(max_per_part: int = 50) -> Dict[str, List[Tuple[Path, List[List[float]]]]]:
    """Return {part: [(image_path, [gt_bbox, ...]), ...]}."""
    data = json.loads(ANNOTATIONS_JSON.read_text(encoding="utf-8"))
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    id_to_fn = {img["id"]: img["file_name"] for img in data["images"]}

    by_part: Dict[str, List[Tuple[Path, List[List[float]]]]] = defaultdict(list)
    for ann in data["annotations"]:
        part = cat_map.get(ann["category_id"], "")
        if part not in FP_CORE:
            continue
        fn = id_to_fn.get(ann["image_id"])
        if fn is None:
            continue
        resolved = _resolve_image_path(fn)
        if resolved is None or not resolved.exists():
            continue
        x, y, w, h = ann["bbox"]
        by_part[part].append((resolved, [x, y, x + w, y + h]))

    # Limit per part
    for part in by_part:
        if len(by_part[part]) > max_per_part:
            by_part[part] = by_part[part][:max_per_part]

    return dict(by_part)


def evaluate_model(model_path: str, images_by_part: Dict, device: str) -> Dict:
    """Run Fashionpedia YOLO on all images, return per-part TP/FP/FN."""
    from ultralytics import YOLO
    from fashion_vision.localization.fashionpedia_part_detector import FashionpediaPartDetector

    if not Path(model_path).exists():
        return {"error": f"Model not found: {model_path}"}

    yolo = YOLO(str(PROJECT_ROOT / YOLO_WEIGHTS))
    yolo.to(device)
    fp = FashionpediaPartDetector(model_path, device=device)

    results: Dict[str, Dict] = {}
    for part, img_list in sorted(images_by_part.items()):
        tp = fp_count = fn = 0
        for img_path, gt_list in img_list:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            H, W = image.shape[:2]

            # Garment detection
            yolo_results = yolo(image, device=device, conf=0.25, verbose=False)
            if yolo_results[0].boxes is not None:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                confs = yolo_results[0].boxes.conf.cpu().numpy()
                best = max(range(len(confs)), key=lambda i: confs[i]) if len(confs) > 0 else 0
                gx1, gy1, gx2, gy2 = [int(v) for v in boxes[best]]
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(W, gx2), min(H, gy2)
                crop = image[gy1:gy2, gx1:gx2]
                ox, oy = gx1, gy1
            else:
                crop, ox, oy = image, 0, 0

            if crop.size == 0:
                continue

            dets = fp.detect(crop, part, conf=0.25)
            pred_boxes = [[d["bbox_xyxy"][0] + ox, d["bbox_xyxy"][1] + oy,
                          d["bbox_xyxy"][2] + ox, d["bbox_xyxy"][3] + oy]
                         for d in dets]

            for gt in gt_list:
                fn += 1
                for pb in pred_boxes:
                    if _box_iou(pb, gt) > 0.01:
                        tp += 1
                        fn -= 1
                        break
            fp_count += len(pred_boxes)
            # Subtract TPs that were counted in preds
            matched_preds = 0
            for pb in pred_boxes:
                for gt in gt_list:
                    if _box_iou(pb, gt) > 0.01:
                        matched_preds += 1
                        break
            fp_count -= matched_preds

        total = tp + fn
        recall = tp / max(1, total)
        precision = tp / max(1, tp + fp_count)
        results[part] = {
            "tp": tp, "fp": fp_count, "fn": fn, "total": total,
            "recall": round(recall, 4), "precision": round(precision, 4),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare v1 vs v2 FP YOLO models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-per-part", type=int, default=30)
    parser.add_argument("--v1-model", default=V1_MODEL)
    parser.add_argument("--v2-model", default=V2_MODEL)
    args = parser.parse_args()

    print("Loading validation images...")
    images = load_fp_images(args.max_per_part)
    total_imgs = sum(len(v) for v in images.values())
    print(f"  {total_imgs} images across {len(images)} parts")

    # ── v1 ──
    print(f"\nEvaluating v1: {args.v1_model}")
    t0 = time.perf_counter()
    v1 = evaluate_model(args.v1_model, images, args.device)
    t1 = time.perf_counter()

    # ── v2 ──
    print(f"Evaluating v2: {args.v2_model}")
    v2 = evaluate_model(args.v2_model, images, args.device)
    t2 = time.perf_counter()

    if "error" in v1 or "error" in v2:
        print(f"\nv1 error: {v1.get('error', 'none')}")
        print(f"v2 error: {v2.get('error', 'none')}")
        return

    # ── Report ──
    total_v1_tp = total_v1_fn = 0
    total_v2_tp = total_v2_fn = 0
    regressions = []

    print(f"\n{'='*80}")
    print(f"FASHIONPEDIA YOLO v1 vs v2 COMPARISON (IoU>0.01)")
    print(f"{'='*80}")
    print(f"{'Part':<14s} {'v1 R':>7s} {'v2 R':>7s} {'Δ R':>7s} {'v1 P':>7s} {'v2 P':>7s} {'Δ P':>7s} {'v1 n':>5s} {'v2 n':>5s}")
    print("-" * 80)

    for part in sorted(v1.keys()):
        m1 = v1[part]; m2 = v2.get(part, {})
        if not m2:
            continue
        dr = m2["recall"] - m1["recall"]
        dp = m2["precision"] - m1["precision"]
        total_v1_tp += m1["tp"]; total_v1_fn += m1["fn"]
        total_v2_tp += m2["tp"]; total_v2_fn += m2["fn"]

        flag = ""
        if dr < -0.02:
            flag = " ⚠️ REGRESSION"
            regressions.append(part)
        elif dr > 0.05:
            flag = " ✅ GAIN"
        print(f"{part:<14s} {m1['recall']:7.4f} {m2['recall']:7.4f} {dr:+7.4f} "
              f"{m1['precision']:7.4f} {m2['precision']:7.4f} {dp:+7.4f} "
              f"{m1['total']:>5d} {m2['total']:>5d}{flag}")

    r1 = total_v1_tp / max(1, total_v1_tp + total_v1_fn)
    r2 = total_v2_tp / max(1, total_v2_tp + total_v2_fn)
    print("-" * 80)
    print(f"{'OVERALL':<14s} {r1:7.4f} {r2:7.4f} {r2-r1:+7.4f}")
    print(f"\nv1 time: {t1-t0:.1f}s | v2 time: {t2-t1:.1f}s")
    print(f"Regressions: {len(regressions)} parts — {regressions if regressions else 'none'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
