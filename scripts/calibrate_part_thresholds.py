#!/usr/bin/env python3
"""
Threshold calibration sweep for Grounding DINO part detection (Phase 2, 3.1.2).

Sweeps box_threshold from 0.20 to 0.55 (step 0.05).
Computes precision, recall, F1 per part type against a COCO JSON validation set.
Outputs metrics.csv, pr_curves.png, and optionally per_image_detections.jsonl.

Usage:
    python scripts/calibrate_part_thresholds.py \
        --images data/calibration/coat_100/images/ \
        --annotations data/calibration/coat_100/annotations_v2.json \
        --output outputs/calibration_v3/ \
        --garment-detections outputs/calibration_v3/yolo/detections.json \
        --sam-segmentation-results outputs/calibration_v3/samhq/segmentation_results.json \
        --save-detections-jsonl \
        [--device cuda] \
        [--model-id IDEA-Research/grounding-dino-base]

--garment-detections: output of tools/infer/predict_garments_yolo.py
    Format: {"images": [{"file_name": "...", "detections": [{"bbox_xyxy": [...]}]}]}
    If omitted, DINO runs on the full image (conservative baseline).

--sam-segmentation-results: output of tools/infer/segment_garments_samhq.py
    Format: {"images": [{"image_path": "...", "segments": [{"bbox_xyxy": [...], "mask_path": "..."}]}]}
    If omitted, DINO runs without garment mask gating.

--save-detections-jsonl: write per_image_detections.jsonl alongside metrics.csv.
    Each line: {"image", "part", "threshold", "garment_bbox", "has_mask",
                "dino_candidates", "gt_bboxes", "tp", "fp", "fn"}
    Used by visualize_calibration_failures.py to build the failure gallery.

Label Studio exports "button_cluster"; this script maps it to "button" in
PART_DETECTION_CONFIG automatically via LABEL_TO_PART.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fashion_vision.localization.part_shape_priors import filter_by_shape_priors

# Label Studio label name → PART_DETECTION_CONFIG key (only where they differ)
LABEL_TO_PART: dict[str, str] = {
    "button_cluster": "button",
}

IOU_MATCH_THRESHOLD = 0.5   # IoU with GT to count as TP


def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if not inter:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _best_garment(ref_box: list, instances: list[dict]) -> Optional[list]:
    """Return the garment bbox with highest overlap to ref_box."""
    if not instances:
        return None
    best = max(instances, key=lambda inst: _iou(ref_box, inst["bbox_xyxy"]))
    return best["bbox_xyxy"] if _iou(ref_box, best["bbox_xyxy"]) > 0.05 else None


def _crop(image: np.ndarray, garment_bbox: Optional[list]) -> tuple[np.ndarray, int, int]:
    """Crop image to garment_bbox; returns (crop, offset_x, offset_y)."""
    if garment_bbox is None:
        return image, 0, 0
    x1, y1, x2, y2 = (int(v) for v in garment_bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return image, 0, 0
    return image[y1:y2, x1:x2], x1, y1


def _normalize_fname(raw_fname: str) -> str:
    """
    Extract the original image filename from a Label Studio export path.

    Label Studio prepends a random hash when importing images, producing paths like:
        ..\\label-studio\\media\\upload\\1\\1e8a1d03-000004.jpg
    We extract just the base filename and strip the hash prefix to recover "000004.jpg".
    """
    base = Path(raw_fname.replace("\\", "/")).name   # "1e8a1d03-000004.jpg"
    # Strip the "<hash>-" prefix (first segment before the first "-")
    if "-" in base:
        base = base.split("-", 1)[1]
    return base


def load_coco(path: Path) -> tuple[dict[str, list], list[str]]:
    """
    Parse Label Studio COCO JSON.

    Returns:
        annotations: {file_name: [{"part": str, "bbox_xyxy": [x1,y1,x2,y2]}]}
        parts:       sorted list of part names (using PART_DETECTION_CONFIG keys)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    id_to_fname = {img["id"]: _normalize_fname(img["file_name"]) for img in data["images"]}
    cat_map = {c["id"]: LABEL_TO_PART.get(c["name"], c["name"]) for c in data["categories"]}
    annotations: dict[str, list] = {}
    for ann in data.get("annotations", []):
        fname = id_to_fname[ann["image_id"]]
        x, y, w, h = ann["bbox"]   # COCO format: [x, y, width, height]
        annotations.setdefault(fname, []).append({
            "part": cat_map[ann["category_id"]],
            "bbox_xyxy": [x, y, x + w, y + h],
        })
    parts = sorted(set(cat_map.values()))
    return annotations, parts


def load_garment_detections(path: Path) -> dict[str, list]:
    """
    Parse predict_garments_yolo.py output JSON.

    Returns: {file_name: [{"bbox_xyxy": [...], ...}]}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {rec["file_name"]: rec.get("detections", []) for rec in data.get("images", [])}


def load_sam_masks(path: Path) -> dict[str, list[dict]]:
    """
    Parse segment_garments_samhq.py segmentation_results.json.

    Returns: {file_name: [{"bbox_xyxy": [...], "mask_path": str}]}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[dict]] = {}
    for img_rec in data.get("images", []):
        fname = Path(img_rec["image_path"]).name
        result[fname] = [
            {"bbox_xyxy": seg["bbox_xyxy"], "mask_path": seg["mask_path"]}
            for seg in img_rec.get("segments", [])
        ]
    return result


def _load_mask_for_bbox(
    garment_bbox: Optional[list],
    sam_records: list[dict],
) -> Optional[np.ndarray]:
    """
    Find the SAM segment whose bbox best overlaps garment_bbox, load its mask PNG.

    Returns a full-image binary uint8 mask, or None when no suitable record is found.
    Uses the existing _iou() helper — no duplicate IoU logic.
    """
    if not sam_records or garment_bbox is None:
        return None
    best = max(sam_records, key=lambda r: _iou(garment_bbox, r["bbox_xyxy"]))
    if _iou(garment_bbox, best["bbox_xyxy"]) < 0.3:
        return None
    mask_p = Path(best["mask_path"])
    if not mask_p.exists():
        return None
    m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.uint8) if m is not None else None


def sweep(
    images_dir: Path,
    annotations: dict[str, list],
    parts: list[str],
    garment_dets: dict[str, list],
    sam_masks: dict[str, list],
    locator,
    part_config: dict,
    thresholds: list[float],
    detections_jsonl_path: Optional[Path] = None,
) -> list[dict]:
    """
    Sweep box_threshold values and compute per-part P/R/F1.

    Args:
        sam_masks: {file_name: [{"bbox_xyxy": [...], "mask_path": str}]}
            from load_sam_masks(). Empty dict when SAM results are unavailable.
        detections_jsonl_path: When set, each image×part×threshold record is
            appended to this JSONL file for use by visualize_calibration_failures.py.
    """
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rows: list[dict] = []

    jsonl_fh = open(detections_jsonl_path, "w", encoding="utf-8") if detections_jsonl_path else None

    try:
        for thresh in thresholds:
            print(f"\n--- threshold={thresh:.2f} ---")
            counters: dict[str, dict[str, int]] = {p: {"tp": 0, "fp": 0, "fn": 0} for p in parts}

            for img_path in image_files:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                fname = img_path.name
                gt_by_part: dict[str, list] = {}
                for gt in annotations.get(fname, []):
                    gt_by_part.setdefault(gt["part"], []).append(gt["bbox_xyxy"])

                instances = garment_dets.get(fname, [])

                for part in parts:
                    cfg = part_config.get(part, {})
                    prompts = cfg.get("prompts", [part.replace("_", " ")])
                    dilation = cfg.get("shape", {}).get("mask_dilation_px", 0)
                    gt_boxes = gt_by_part.get(part, [])

                    # Choose garment crop: instance that best overlaps the first GT box
                    ref_box = gt_boxes[0] if gt_boxes else None
                    if ref_box is not None:
                        garment_bbox = _best_garment(ref_box, instances)
                    else:
                        garment_bbox = instances[0]["bbox_xyxy"] if instances else None

                    crop, ox, oy = _crop(image, garment_bbox)
                    crop_h, crop_w = crop.shape[:2]

                    # Load SAM mask for this garment and crop it to match image crop.
                    # _crop() uses [y1:y2, x1:x2] without padding, so the mask slice
                    # uses identical bounds and stays pixel-aligned with the image crop.
                    garment_mask_crop: Optional[np.ndarray] = None
                    if garment_bbox is not None:
                        mask_full = _load_mask_for_bbox(garment_bbox, sam_masks.get(fname, []))
                        if mask_full is not None:
                            x1c, y1c, x2c, y2c = (int(v) for v in garment_bbox)
                            x1c, y1c = max(0, x1c), max(0, y1c)
                            x2c, y2c = min(image.shape[1], x2c), min(image.shape[0], y2c)
                            if x2c > x1c and y2c > y1c:
                                garment_mask_crop = mask_full[y1c:y2c, x1c:x2c]

                    dets = locator.detect_multi_prompt(
                        crop, prompts,
                        garment_mask=garment_mask_crop,
                        threshold=thresh,
                        dilation_px=dilation,
                    )
                    # Shape priors in crop-space (garment fills the crop → [0,0,w,h])
                    dets = filter_by_shape_priors(dets, part, garment_bbox=[0, 0, crop_w, crop_h])
                    # Remap detection coords to full-image space
                    dets_full = [
                        [d["bbox_xyxy"][0]+ox, d["bbox_xyxy"][1]+oy,
                         d["bbox_xyxy"][2]+ox, d["bbox_xyxy"][3]+oy]
                        for d in dets
                    ]

                    if not gt_boxes:
                        counters[part]["fp"] += len(dets_full)
                        if jsonl_fh:
                            _write_jsonl_record(
                                jsonl_fh, fname, part, thresh, garment_bbox,
                                garment_mask_crop is not None, dets, dets_full, gt_boxes,
                                tp=0, fp=len(dets_full), fn=0,
                            )
                        continue

                    # Greedy TP matching (GT-first)
                    matched_gt: set[int] = set()
                    matched_det: set[int] = set()
                    for gi, gb in enumerate(gt_boxes):
                        for di, db in enumerate(dets_full):
                            if di in matched_det:
                                continue
                            if _iou(gb, db) >= IOU_MATCH_THRESHOLD:
                                matched_gt.add(gi)
                                matched_det.add(di)
                                break

                    tp = len(matched_gt)
                    fp = len(dets_full) - len(matched_det)
                    fn = len(gt_boxes) - len(matched_gt)

                    counters[part]["tp"] += tp
                    counters[part]["fn"] += fn
                    counters[part]["fp"] += fp

                    if jsonl_fh:
                        _write_jsonl_record(
                            jsonl_fh, fname, part, thresh, garment_bbox,
                            garment_mask_crop is not None, dets, dets_full, gt_boxes,
                            tp=tp, fp=fp, fn=fn,
                        )

            for part, c in counters.items():
                tp, fp, fn = c["tp"], c["fp"], c["fn"]
                prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
                print(f"  {part:20s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
                      f"TP={tp} FP={fp} FN={fn}")
                rows.append({"threshold": thresh, "part": part,
                             "precision": round(prec, 4), "recall": round(rec, 4),
                             "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn})
    finally:
        if jsonl_fh:
            jsonl_fh.close()

    return rows


def _write_jsonl_record(
    fh,
    fname: str,
    part: str,
    threshold: float,
    garment_bbox: Optional[list],
    has_mask: bool,
    dets_crop: list[dict],
    dets_full: list[list],
    gt_boxes: list,
    tp: int,
    fp: int,
    fn: int,
) -> None:
    """Append one line to the per-image detections JSONL file."""
    candidates = []
    for d, full_box in zip(dets_crop, dets_full):
        candidates.append({
            "bbox_xyxy_crop": [round(v) for v in d["bbox_xyxy"]],
            "bbox_xyxy_full": [round(v) for v in full_box],
            "score": round(float(d["score"]), 4),
            "_shape_prior_status": d.get("_shape_prior_status", "unknown"),
            "_shape_prior_reasons": d.get("_shape_prior_reasons", []),
            "prompt": d.get("prompt", ""),
        })
    record = {
        "image": fname,
        "part": part,
        "threshold": threshold,
        "garment_bbox": [round(v) for v in garment_bbox] if garment_bbox else None,
        "has_mask": has_mask,
        "dino_candidates": candidates,
        "gt_bboxes": [[round(v) for v in b] for b in gt_boxes],
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "metrics.csv"
    fields = ["threshold", "part", "precision", "recall", "f1", "tp", "fp", "fn"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_curves(rows: list[dict], parts: list[str], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    ncols = max(1, len(parts))
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    for i, part in enumerate(parts):
        part_rows = sorted([r for r in rows if r["part"] == part], key=lambda r: r["threshold"])
        recalls    = [r["recall"] for r in part_rows]
        precisions = [r["precision"] for r in part_rows]
        thresholds = [r["threshold"] for r in part_rows]
        ax = axes[0][i]
        ax.plot(recalls, precisions, "o-")
        for t, r, p in zip(thresholds, recalls, precisions):
            ax.annotate(f"{t:.2f}", (r, p), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.axhline(0.65, color="green",  linestyle="--", alpha=0.6, label="P≥0.65 target")
        ax.axvline(0.50, color="orange", linestyle="--", alpha=0.6, label="R≥0.50 target")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(part); ax.legend(fontsize=7)
    plt.tight_layout()
    out = output_dir / "pr_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"PR curves → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate DINO box_threshold per part type.")
    ap.add_argument("--images",       required=True,  type=Path, help="Dir of validation images")
    ap.add_argument("--annotations",  required=True,  type=Path, help="COCO JSON from Label Studio")
    ap.add_argument("--output",       required=True,  type=Path, help="Output dir for CSV + PNG")
    ap.add_argument("--garment-detections", type=Path, default=None,
                    help="detections.json from predict_garments_yolo.py (optional)")
    ap.add_argument("--sam-segmentation-results", type=Path, default=None,
                    help="segmentation_results.json from segment_garments_samhq.py (optional)")
    ap.add_argument("--save-detections-jsonl", action="store_true",
                    help="Write per_image_detections.jsonl for use by visualize_calibration_failures.py")
    ap.add_argument("--model-id",  default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--min-threshold", type=float, default=0.20)
    ap.add_argument("--max-threshold", type=float, default=0.55)
    ap.add_argument("--step",          type=float, default=0.05)
    args = ap.parse_args()

    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    from fashion_vision.localization.part_detection_config import PART_DETECTION_CONFIG

    annotations, parts = load_coco(args.annotations)
    garment_dets = load_garment_detections(args.garment_detections) if args.garment_detections else {}
    sam_masks = load_sam_masks(args.sam_segmentation_results) if args.sam_segmentation_results else {}

    thresholds: list[float] = []
    t = args.min_threshold
    while t <= args.max_threshold + 1e-9:
        thresholds.append(round(t, 2))
        t += args.step

    print(f"Parts to calibrate : {parts}")
    print(f"Threshold sweep    : {thresholds}")
    print(f"Annotated images   : {len(annotations)}")
    if garment_dets:
        print(f"Garment detections : {len(garment_dets)} images")
    else:
        print("Garment detections : none — running DINO on full image")
    if sam_masks:
        print(f"SAM masks          : {len(sam_masks)} images")
    else:
        print("SAM masks          : none — mask gating disabled")

    print(f"\nLoading {args.model_id} on {args.device} ...")
    locator = GroundingDINOLocator(model_id=args.model_id, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.output / "per_image_detections.jsonl" if args.save_detections_jsonl else None

    rows = sweep(
        images_dir=args.images,
        annotations=annotations,
        parts=parts,
        garment_dets=garment_dets,
        sam_masks=sam_masks,
        locator=locator,
        part_config=PART_DETECTION_CONFIG,
        thresholds=thresholds,
        detections_jsonl_path=jsonl_path,
    )

    csv_path = save_csv(rows, args.output)
    print(f"\nMetrics CSV → {csv_path}")
    if jsonl_path:
        print(f"Detections JSONL   → {jsonl_path}")
    plot_curves(rows, parts, args.output)


if __name__ == "__main__":
    main()
