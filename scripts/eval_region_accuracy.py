#!/usr/bin/env python3
"""
3.1.2 Region localization accuracy evaluation against Label Studio GT annotations.

Measures "区域定位准确率": given N images with ground-truth bboxes for a part,
what fraction have the pipeline correctly localize the part (IoU > threshold)?

Usage::

    conda activate fashion-demo2
    python scripts/eval_region_accuracy.py

Outputs ``data/validation/eval_312/``:
  - metrics.json       per-part accuracy summary
  - per_result.jsonl   per-annotation results (consumed by visualization script)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Paths ────────────────────────────────────────────────────────────────────────

ANNOTATIONS_JSON = (
    PROJECT_ROOT / "data/validation/project-5-at-2026-07-07-17-55-adf237c4/result.json"
)
IMAGES_DIR = PROJECT_ROOT / "data/validation/to_annotate"
GARMENT_DETECTIONS_JSON = PROJECT_ROOT / "data/validation/eval_312/detections.json"
OUTPUT_DIR = PROJECT_ROOT / "data/validation/eval_312"

YOLO_WEIGHTS = "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
FP_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
DINO_MODEL = "models/grounding_dino_tiny"
SAM_CHECKPOINT = "checkpoints/sam_hq/sam_hq_vit_b.pth"

IOU_THRESHOLD = 0.3  # IoU > this → counted as "found correctly"

# ── Which parts are Fashionpedia-core vs DINO-only ─────────────────────────────

# Fashionpedia-core parts: FP YOLO can detect these directly.
# Non-FP parts: need Grounding DINO.
# inner_garment: SAM-based special path.
FP_CORE_PARTS = {
    "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline",
    "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe",
    "ribbon", "rivet", "ruffle", "sequin", "tassel", "hood",
}

# Label Studio label name → internal part name (where they differ)
LABEL_TO_PART: dict[str, str] = {
    "button_cluster": "button",
}


def _box_iou(a: List[float], b: List[float]) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalize_fname(raw: str) -> str:
    """Strip Label Studio hash prefix: 'abc123-001.jpg' -> '001.jpg'."""
    base = Path(raw.replace("\\", "/")).name
    if "-" in base:
        base = base.split("-", 1)[1]
    return base


def load_gt_annotations(coco_path: Path) -> Tuple[Dict[str, List[Dict]], List[str]]:
    """
    Parse Label Studio COCO JSON.

    Returns:
        annotations: {file_name: [{"part": str, "bbox_xyxy": [...]}]}
        parts: sorted unique part names
    """
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    id_to_fname = {
        img["id"]: _normalize_fname(img["file_name"]) for img in data["images"]
    }
    cat_map = {
        c["id"]: LABEL_TO_PART.get(c["name"], c["name"])
        for c in data["categories"]
    }
    annotations: Dict[str, List[Dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        fname = id_to_fname[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        annotations[fname].append({
            "part": cat_map[ann["category_id"]],
            "bbox_xyxy": [x, y, x + w, y + h],
        })
    parts = sorted(set(cat_map.values()))
    return dict(annotations), parts


def load_garment_detections(path: Path) -> Dict[str, List[Dict]]:
    """Parse predict_garments_yolo.py output."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        Path(rec["file_name"]).name: rec.get("detections", [])
        for rec in data.get("images", [])
    }


def _best_garment(ref_box: List, instances: List[Dict]) -> Optional[Dict]:
    """Return the garment instance whose bbox best overlaps ref_box."""
    if not instances:
        return None
    best = max(instances, key=lambda inst: _box_iou(ref_box, inst["bbox_xyxy"]))
    return best if _box_iou(ref_box, best["bbox_xyxy"]) > 0.05 else None


def _build_instance_dict(garment_det: Dict) -> Dict[str, Any]:
    """Build a minimal instance dict compatible with downstream consumers."""
    bbox = garment_det["bbox_xyxy"]
    x1, y1, x2, y2 = bbox
    return {
        "bbox_xyxy": bbox,
        "bbox_x": x1, "bbox_y": y1,
        "bbox_w": x2 - x1, "bbox_h": y2 - y1,
        "fine_class_id": garment_det.get("class_id", -1),
        "fine_class_name": garment_det.get("class_name", "unknown"),
        "coarse_class_id": garment_det.get("coarse_class_id", -1),
        "coarse_class_name": garment_det.get("coarse_class_name", "unknown"),
        "score": garment_det.get("score", 0.0),
        "mask": None,
    }


# ── Model loading ────────────────────────────────────────────────────────────────


def load_fp_detector(device: str = "cuda"):
    """Load Fashionpedia YOLOv8s 19-class part detector."""
    from fashion_vision.localization.fashionpedia_part_detector import (
        FashionpediaPartDetector,
    )
    fp_path = str(PROJECT_ROOT / FP_MODEL)
    if not Path(fp_path).exists():
        print(f"[WARN] Fashionpedia model not found: {fp_path}")
        return None
    det = FashionpediaPartDetector(fp_path, device=device)
    print(f"[INFO] Fashionpedia detector loaded: {fp_path}")
    return det


def load_dino_locator(device: str = "cuda"):
    """Load Grounding DINO from local weights."""
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    dino_path = str(PROJECT_ROOT / DINO_MODEL)
    if not Path(dino_path).exists():
        print(f"[WARN] DINO model not found: {dino_path}")
        return None
    locator = GroundingDINOLocator(model_id=dino_path, device=device)
    print(f"[INFO] DINO locator loaded: {dino_path}")
    return locator


def load_sam_wrapper(device: str = "cuda"):
    """Load SAM-HQ wrapper for inner-garment detection and mask refinement."""
    from fashion_vision.models.sam_hq_wrapper import SamHqWrapper
    ckpt = str(PROJECT_ROOT / SAM_CHECKPOINT)
    if not Path(ckpt).exists():
        print(f"[WARN] SAM checkpoint not found: {ckpt}")
        return None
    sam = SamHqWrapper(
        checkpoint=ckpt,
        model_type="vit_b",
        device=device,
    )
    print(f"[INFO] SAM-HQ loaded: {ckpt}")
    return sam


# ── Per-part detection runners ───────────────────────────────────────────────────


def _part_to_query(part: str) -> str:
    """Convert internal part name to the simplest Chinese query for evaluation."""
    from fashion_vision.localization.intent_parser import PART_VOCAB
    if part in PART_VOCAB:
        return PART_VOCAB[part][0]  # first Chinese keyword
    return part.replace("_", " ")


def run_fp_detection(
    crop: np.ndarray,
    part: str,
    fp_detector,
    conf: float = 0.25,
) -> List[Dict]:
    """Run Fashionpedia YOLO on a garment crop, return detections in crop coords."""
    try:
        dets = fp_detector.detect(crop, part, garment_mask=None, conf=conf)
    except Exception as e:
        print(f"  [WARN] FP detection error for {part}: {e}")
        return []
    return dets


def run_dino_detection(
    crop: np.ndarray,
    part: str,
    dino_locator,
    threshold: float = 0.28,
) -> List[Dict]:
    """Run Grounding DINO on a garment crop, return detections in crop coords."""
    from fashion_vision.localization.part_detection_config import get_part_prompts
    from fashion_vision.localization.part_shape_priors import filter_by_shape_priors

    prompts = get_part_prompts(part)
    try:
        dets = dino_locator.detect_multi_prompt(
            crop, prompts,
            garment_mask=None,
            threshold=threshold,
        )
    except Exception as e:
        print(f"  [WARN] DINO error for {part}: {e}")
        return []

    # Apply shape priors
    h, w = crop.shape[:2]
    dets = filter_by_shape_priors(dets, part, garment_bbox=[0, 0, w, h])
    return dets


def evaluate_inner_garment_overlap(
    image_bgr: np.ndarray,
    garment_det: Dict,
    gt_bbox: List[float],
    sam_wrapper,
) -> Tuple[bool, float, str]:
    """
    Evaluate inner_garment by checking GT bbox overlap with SAM mask complement.

    Runs SAM-HQ on the outerwear garment bbox, then computes what fraction of the
    GT bbox falls OUTSIDE the outerwear mask (i.e., is visible inner garment).
    If >= 30% overlap, the inner garment region is considered correctly identified.

    Returns:
        (is_hit, overlap_ratio, reason_string)
    """
    coarse = (garment_det.get("coarse_class_name") or "").strip().lower()
    fine = (garment_det.get("class_name") or "").strip().lower()
    _OUTWEAR_FINE = {"short sleeve outwear", "long sleeve outwear"}
    is_outer = coarse == "outerwear" or fine in _OUTWEAR_FINE
    if not is_outer:
        return False, 0.0, f"not_outerwear (coarse={coarse}, fine={fine})"

    # SAM expects RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        result = sam_wrapper.predict_with_box(
            image_rgb,
            garment_det["bbox_xyxy"],
        )
        sam_mask = result["mask"]  # H x W binary uint8
    except Exception as e:
        return False, 0.0, f"SAM error: {e}"

    if sam_mask is None or sam_mask.sum() == 0:
        return False, 0.0, "SAM produced empty mask"

    # Compute overlap: GT area NOT covered by outerwear mask / GT area
    gx1, gy1, gx2, gy2 = (int(v) for v in gt_bbox)
    h, w = sam_mask.shape[:2]
    gx1, gy1 = max(0, gx1), max(0, gy1)
    gx2, gy2 = min(w, gx2), min(h, gy2)

    if gx2 <= gx1 or gy2 <= gy1:
        return False, 0.0, "GT bbox outside image bounds"

    gt_region = sam_mask[gy1:gy2, gx1:gx2]
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    # Pixels NOT in outerwear mask = complement (visible inner garment)
    complement_pixels = int((gt_region == 0).sum())
    overlap_ratio = complement_pixels / gt_area if gt_area > 0 else 0.0

    is_hit = overlap_ratio >= 0.3
    reason = f"complement_overlap={overlap_ratio:.3f}"
    return is_hit, overlap_ratio, reason


# ── Main evaluation ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="3.1.2 Region Localization Accuracy Evaluation")
    parser.add_argument("--annotations", type=str, default=str(ANNOTATIONS_JSON))
    parser.add_argument("--images-dir", type=str, default=str(IMAGES_DIR))
    parser.add_argument("--garment-dets", type=str, default=str(GARMENT_DETECTIONS_JSON))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp-conf", type=float, default=0.25,
                        help="Fashionpedia YOLO confidence threshold")
    parser.add_argument("--dino-threshold", type=float, default=0.28,
                        help="Grounding DINO box threshold")
    parser.add_argument("--skip-inner", action="store_true",
                        help="Skip inner_garment evaluation (needs SAM)")
    parser.add_argument("--skip-dino", action="store_true",
                        help="Skip DINO evaluation (for quick FP-only test)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit to first N images for quick test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("3.1.2 Region Localization Accuracy Evaluation")
    print(f"IoU threshold: {args.iou_threshold}")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n--- Loading annotations ---")
    gt_annotations, all_parts = load_gt_annotations(Path(args.annotations))
    print(f"  {len(gt_annotations)} images, {len(all_parts)} parts")
    for p in all_parts:
        n = sum(1 for anns in gt_annotations.values() for a in anns if a["part"] == p)
        if n:
            print(f"    {p}: {n}")

    print("\n--- Loading garment detections ---")
    garment_dets = load_garment_detections(Path(args.garment_dets))
    print(f"  {len(garment_dets)} images with garment detections")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n--- Loading models ---")
    fp_detector = load_fp_detector(args.device)
    dino_locator = None if args.skip_dino else load_dino_locator(args.device)
    sam_wrapper = None if args.skip_inner else load_sam_wrapper(args.device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n--- Evaluating ---")

    images_dir = Path(args.images_dir)
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.max_images:
        image_files = image_files[:args.max_images]

    # Per-part counters
    counters: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "hits": 0, "detections": 0, "misses": 0}
    )

    # Per-result records for visualization
    results: List[Dict] = []
    jsonl_path = output_dir / "per_result.jsonl"
    jsonl_fh = open(jsonl_path, "w", encoding="utf-8")

    t_start = time.perf_counter()

    for img_idx, img_path in enumerate(image_files):
        fname = img_path.name
        if fname not in gt_annotations:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Cannot read {fname}")
            continue
        H, W = image.shape[:2]

        gt_parts_in_image = gt_annotations[fname]
        instances = garment_dets.get(fname, [])

        if img_idx < 3 or img_idx % 20 == 0:
            print(f"\n[{img_idx+1:03d}/{len(image_files)}] {fname} "
                  f"({len(gt_parts_in_image)} GTs, {len(instances)} garments)")

        for gt in gt_parts_in_image:
            part = gt["part"]
            gt_bbox = gt["bbox_xyxy"]
            counters[part]["total"] += 1

            # ── Find best garment instance ──────────────────────────────────
            garment = _best_garment(gt_bbox, instances)

            # ── Crop to garment bbox ────────────────────────────────────────
            if garment is not None:
                gx1, gy1, gx2, gy2 = (int(v) for v in garment["bbox_xyxy"])
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(W, gx2), min(H, gy2)
                crop = image[gy1:gy2, gx1:gx2]
                ox, oy = gx1, gy1
            else:
                crop = image
                ox, oy = 0, 0

            # ── Detect ──────────────────────────────────────────────────────
            pred_bboxes: List[List[float]] = []
            backend = "none"
            status = "not_detected"
            pred_score = None

            if part == "inner_garment":
                # SAM-HQ based: GT bbox overlap with outerwear mask complement.
                if sam_wrapper is not None and garment is not None:
                    is_hit, overlap, reason = evaluate_inner_garment_overlap(
                        image, garment, gt_bbox, sam_wrapper,
                    )
                    best_iou = overlap  # store overlap ratio in place of IoU
                    if is_hit:
                        counters[part]["hits"] += 1
                    else:
                        counters[part]["misses"] += 1
                    backend = "sam_mask_complement"
                    status = "success" if is_hit else "not_detected"
                    pred_score = overlap

                    record = {
                        "image": fname,
                        "image_idx": img_idx,
                        "part": part,
                        "gt_bbox": gt_bbox,
                        "pred_bboxes": [],
                        "best_iou": round(overlap, 4),
                        "is_hit": is_hit,
                        "backend": backend,
                        "status": status,
                        "pred_score": pred_score,
                        "garment_class": garment.get("class_name") if garment else None,
                        "garment_bbox": garment["bbox_xyxy"] if garment else None,
                        "crop_offset": [0, 0],
                        "inner_reason": reason,
                    }
                    results.append(record)
                    jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue  # skip the normal flow below
                else:
                    backend = "inner_garment_skipped"
                    status = "skipped"

            elif part in FP_CORE_PARTS:
                # Fashionpedia YOLO path
                if fp_detector is not None:
                    dets = run_fp_detection(crop, part, fp_detector, conf=args.fp_conf)
                    # Remap coords to full image
                    for d in dets:
                        b = d["bbox_xyxy"]
                        pred_bboxes.append([
                            b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy,
                        ])
                    if pred_bboxes:
                        backend = "fashionpedia_yolo"
                        status = "success"
                        pred_score = dets[0].get("score")
                    else:
                        backend = "fashionpedia_yolo"
                else:
                    backend = "fp_not_loaded"

            else:
                # DINO path (non-FP parts: button, placket, drawstring, etc.)
                if dino_locator is not None:
                    dets = run_dino_detection(crop, part, dino_locator, threshold=args.dino_threshold)
                    for d in dets:
                        b = d["bbox_xyxy"]
                        pred_bboxes.append([
                            b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy,
                        ])
                    if pred_bboxes:
                        backend = "grounding_dino"
                        status = "success"
                        pred_score = dets[0].get("score")
                    else:
                        backend = "grounding_dino"
                else:
                    backend = "dino_not_loaded"

            # ── Compute IoU ──────────────────────────────────────────────────
            best_iou = 0.0
            for pb in pred_bboxes:
                iou = _box_iou(pb, gt_bbox)
                if iou > best_iou:
                    best_iou = iou

            is_hit = best_iou > args.iou_threshold
            if is_hit:
                counters[part]["hits"] += 1
            else:
                counters[part]["misses"] += 1

            # ── Record ──────────────────────────────────────────────────────
            record = {
                "image": fname,
                "image_idx": img_idx,
                "part": part,
                "gt_bbox": gt_bbox,
                "pred_bboxes": pred_bboxes,
                "best_iou": round(best_iou, 4),
                "is_hit": is_hit,
                "backend": backend,
                "status": status,
                "pred_score": pred_score,
                "garment_class": garment.get("class_name") if garment else None,
                "garment_bbox": garment["bbox_xyxy"] if garment else None,
                "crop_offset": [ox, oy],
            }
            results.append(record)
            jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    jsonl_fh.close()
    elapsed = time.perf_counter() - t_start
    print(f"\nEvaluation done in {elapsed:.1f}s ({len(results)} annotations)")

    # ── Compute metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS: Region Localization Accuracy (IoU > {:.1f})".format(args.iou_threshold))
    print("=" * 70)
    print(f"{'Part':<22s} {'Total':>6s} {'Hits':>6s} {'Miss':>6s} {'Acc%':>7s}")
    print("-" * 55)

    metrics: Dict[str, Dict] = {}
    total_hits = 0
    total_total = 0

    for part in sorted(counters.keys()):
        c = counters[part]
        acc = c["hits"] / c["total"] * 100 if c["total"] > 0 else 0.0
        total_hits += c["hits"]
        total_total += c["total"]
        metrics[part] = {
            "total": c["total"],
            "hits": c["hits"],
            "misses": c["misses"],
            "accuracy": round(acc, 2),
        }
        flag = " ★" if acc >= 92 else " ✗" if acc < 50 else ""
        print(f"{part:<22s} {c['total']:>6d} {c['hits']:>6d} {c['misses']:>6d} {acc:>6.1f}%{flag}")

    overall_acc = total_hits / total_total * 100 if total_total > 0 else 0.0
    print("-" * 55)
    print(f"{'OVERALL':<22s} {total_total:>6d} {total_hits:>6d} "
          f"{total_total - total_hits:>6d} {overall_acc:>6.1f}%")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics_output = {
        "iou_threshold": args.iou_threshold,
        "num_images": len(image_files),
        "num_annotations": total_total,
        "overall_accuracy": round(overall_acc, 2),
        "per_part": metrics,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Per-result JSONL: {jsonl_path}")

    # ── Pass/Fail against PRD target ─────────────────────────────────────────
    prd_target = 92.0
    passed = sum(1 for m in metrics.values() if m["accuracy"] >= prd_target)
    total_parts_evaluated = len(metrics)
    print(f"\nPRD target ≥ {prd_target}%: {passed}/{total_parts_evaluated} parts meet target")

    return metrics_output


if __name__ == "__main__":
    main()
