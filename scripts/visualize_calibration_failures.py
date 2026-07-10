#!/usr/bin/env python3
"""
Batch failure gallery for calibration v3 (3.1.2).

Reads per_image_detections.jsonl from calibrate_part_thresholds.py and builds
an HTML gallery so you can visually judge DINO's behaviour — what it found vs.
what the GT says.

Each image×part pair renders a three-panel row:
  Left:   Original image with GT boxes (green).
  Centre: Garment crop with all DINO candidates coloured by outcome:
            green  = TP (IoU ≥ 0.5 with any GT)
            orange = FP (passed shape priors but no GT match)
            red    = rejected by shape priors (label shows reason)
  Right:  Text summary (status, top score, IoU, rejection counts).

The HTML is structured as one <section> per part, with TP / FP / FN sub-headings.

Usage:
    python scripts/visualize_calibration_failures.py \
        --images data/calibration/coat_100/images/ \
        --annotations data/calibration/coat_100/annotations_v2.json \
        --garment-detections outputs/calibration_v3/yolo/detections.json \
        --sam-results outputs/calibration_v3/samhq/segmentation_results.json \
        --detections-jsonl outputs/calibration_v3/per_image_detections.jsonl \
        --threshold 0.35 \
        --output-dir outputs/calibration_v3_gallery/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fashion_vision.localization.viz_utils import (  # noqa: E402
    draw_text,
    draw_box,
    add_title_bar,
    resize_to_height,
    pad_width,
    panels_to_html,
    box_iou,
)

IOU_MATCH_THRESHOLD = 0.5
_TARGET_H = 320   # height of each panel in the gallery


def _load_coco_gt(path: Path) -> dict[str, dict[str, list]]:
    """Returns {file_name: {part: [bbox_xyxy, ...]}}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    id_to_fname: dict[int, str] = {}
    for img in data["images"]:
        base = Path(img["file_name"].replace("\\", "/")).name
        if "-" in base:
            base = base.split("-", 1)[1]
        id_to_fname[img["id"]] = base
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    result: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        result[id_to_fname[ann["image_id"]]][cat_map[ann["category_id"]]].append(
            [x, y, x + w, y + h]
        )
    return result


def _load_jsonl(path: Path, threshold: float) -> dict[str, dict[str, dict]]:
    """
    Returns {part: {image_name: record}} filtered to the requested threshold.

    When multiple thresholds exist for the same image+part, only the one
    closest to ``threshold`` is kept.
    """
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            raw[rec["part"]][rec["image"]].append(rec)

    result: dict[str, dict[str, dict]] = {}
    for part, by_img in raw.items():
        result[part] = {}
        for img, recs in by_img.items():
            best = min(recs, key=lambda r: abs(r["threshold"] - threshold))
            result[part][img] = best
    return result


def _load_garment_bbox(garment_dets: dict[str, list], fname: str, gt_boxes: list) -> Optional[list]:
    """Pick the YOLO detection bbox that best overlaps the first GT box."""
    instances = garment_dets.get(fname, [])
    if not instances:
        return None
    ref = gt_boxes[0] if gt_boxes else None
    if ref is None:
        return instances[0]["bbox_xyxy"]
    best = max(instances, key=lambda d: box_iou(ref, d["bbox_xyxy"]))
    return best["bbox_xyxy"] if box_iou(ref, best["bbox_xyxy"]) > 0.05 else None


def _load_garment_detections(path: Path) -> dict[str, list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {rec["file_name"]: rec.get("detections", []) for rec in data.get("images", [])}


# ── Panel builders ────────────────────────────────────────────────────────────

def _crop_to_garment(
    image: np.ndarray,
    garment_bbox: Optional[list],
    pad: int = 8,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (crop, (offset_x, offset_y))."""
    if garment_bbox is None:
        return image.copy(), (0, 0)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in garment_bbox)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return image.copy(), (0, 0)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def _build_left_panel(image: np.ndarray, gt_boxes: list) -> np.ndarray:
    panel = image.copy()
    for gb in gt_boxes:
        draw_box(panel, gb, (0, 220, 0), "GT")
    return add_title_bar(panel, "Original + GT (green)")


def _build_centre_panel(
    image: np.ndarray,
    garment_bbox: Optional[list],
    candidates: list[dict],
    gt_boxes: list,
) -> np.ndarray:
    crop, (ox, oy) = _crop_to_garment(image, garment_bbox)

    for cand in candidates:
        full = cand["bbox_xyxy_full"]
        crop_box = [full[0] - ox, full[1] - oy, full[2] - ox, full[3] - oy]
        status = cand.get("_shape_prior_status", "unknown")

        if status == "rejected":
            reasons = cand.get("_shape_prior_reasons", [])
            label = reasons[0][:22] if reasons else "rejected"
            draw_box(crop, crop_box, (0, 0, 200), label)           # red
        else:
            # Determine TP vs FP by IoU with GT
            is_tp = any(box_iou(full, gb) >= IOU_MATCH_THRESHOLD for gb in gt_boxes)
            color = (0, 200, 0) if is_tp else (0, 140, 255)         # green / orange
            label = f"TP {cand['score']:.2f}" if is_tp else f"FP {cand['score']:.2f}"
            draw_box(crop, crop_box, color, label)

    return add_title_bar(crop, "Garment crop + DINO (green=TP orange=FP red=reject)")


def _build_right_panel(
    image: np.ndarray,
    record: dict,
    gt_boxes: list,
) -> np.ndarray:
    """Text-only summary panel."""
    h, w = max(100, image.shape[0] // 2), max(200, image.shape[1] // 2)
    panel = np.full((h, w, 3), 30, dtype=np.uint8)

    passed = [c for c in record["dino_candidates"]
              if c.get("_shape_prior_status") != "rejected"]
    rejected = [c for c in record["dino_candidates"]
                if c.get("_shape_prior_status") == "rejected"]

    # Derive human-readable outcome
    if record["tp"] > 0:
        outcome = "TP"
    elif not gt_boxes:
        outcome = "NO-GT"
    elif record["fp"] > 0:
        outcome = "FP"
    else:
        outcome = "FN"

    top_score = max((c["score"] for c in passed), default=None)
    best_iou = max(
        (box_iou(c["bbox_xyxy_full"], gb) for c in passed for gb in gt_boxes),
        default=0.0,
    ) if gt_boxes and passed else 0.0

    lines = [
        f"part: {record['part']}",
        f"image: {record['image']}",
        f"status: {outcome}",
        f"threshold: {record['threshold']:.2f}",
        f"top score: {top_score:.3f}" if top_score is not None else "top score: —",
        f"best IoU with GT: {best_iou:.3f}",
        f"passed shape: {len(passed)}  rejected: {len(rejected)}",
        f"GT boxes: {len(gt_boxes)}",
        f"has_mask: {record.get('has_mask', False)}",
    ]

    y = 18
    for line in lines:
        color = (200, 255, 200) if outcome == "TP" else (
            (255, 160, 80) if outcome == "FP" else (80, 80, 255)
        )
        draw_text(panel, line, (6, y), color if "status:" in line else (200, 200, 200))
        y += 20
        if y > h - 10:
            break

    return add_title_bar(panel, f"Summary  [{outcome}]")


def _build_trio(
    image: np.ndarray,
    record: dict,
    gt_boxes: list,
    garment_bbox: Optional[list],
) -> np.ndarray:
    """Assemble a single three-panel row image."""
    p_left = _build_left_panel(image, gt_boxes)
    p_centre = _build_centre_panel(image, garment_bbox, record["dino_candidates"], gt_boxes)
    p_right = _build_right_panel(image, record, gt_boxes)

    target_h = _TARGET_H
    scaled = [resize_to_height(p, target_h) for p in (p_left, p_centre, p_right)]
    max_w = max(p.shape[1] for p in scaled)
    padded = [pad_width(p, max_w, fill=30) for p in scaled]
    sep = np.full((target_h, 4, 3), 60, dtype=np.uint8)
    return np.hstack([padded[0], sep, padded[1], sep, padded[2]])


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    images_dir: Path,
    annotations_path: Path,
    garment_dets_path: Optional[Path],
    jsonl_path: Path,
    threshold: float,
    output_dir: Path,
) -> None:
    gt_all = _load_coco_gt(annotations_path)
    garment_dets = _load_garment_detections(garment_dets_path) if garment_dets_path else {}
    by_part = _load_jsonl(jsonl_path, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

    for part in sorted(by_part):
        records = by_part[part]
        # Sort: FN first (most informative), then FP, then TP
        def _sort_key(r: dict) -> int:
            if r["fn"] > 0 and r["tp"] == 0:
                return 0   # FN
            if r["fp"] > 0 and r["tp"] == 0:
                return 1   # FP
            return 2       # TP or mixed

        sorted_records = sorted(records.values(), key=_sort_key)
        panels: list[tuple[str, np.ndarray]] = []

        for record in sorted_records:
            fname = record["image"]
            img_path = images_dir / fname
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[WARN] Cannot read {img_path} — skipping")
                continue

            gt_boxes = gt_all.get(fname, {}).get(part, [])
            garment_bbox = record.get("garment_bbox") or _load_garment_bbox(garment_dets, fname, gt_boxes)

            trio = _build_trio(image, record, gt_boxes, garment_bbox)

            tp, fp, fn = record["tp"], record["fp"], record["fn"]
            outcome = "TP" if tp > 0 else ("FP" if fp > 0 and not gt_boxes else "FN" if fn > 0 else "OK")
            label = f"{outcome} | {fname} | t={record['threshold']:.2f}"
            panels.append((label, trio))

        if panels:
            sections.append((f"part: {part}  ({len(panels)} images)", panels))
            print(f"  {part}: {len(panels)} panels")

    out_html = output_dir / "index.html"
    panels_to_html(sections, out_html)
    print(f"\nGallery → {out_html}")
    print("Open in a browser to visually judge DINO output vs. GT.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images",             required=True, type=Path, help="Calibration images directory")
    p.add_argument("--annotations",        required=True, type=Path, help="COCO JSON from Label Studio")
    p.add_argument("--detections-jsonl",   required=True, type=Path,
                   help="per_image_detections.jsonl from calibrate_part_thresholds.py --save-detections-jsonl")
    p.add_argument("--garment-detections", type=Path, default=None,
                   help="detections.json from predict_garments_yolo.py (used for garment crop display)")
    p.add_argument("--threshold",          type=float, default=0.35,
                   help="Which threshold to visualise (closest available value is used)")
    p.add_argument("--output-dir",         type=Path, default=Path("outputs/calibration_gallery"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        images_dir=args.images,
        annotations_path=args.annotations,
        garment_dets_path=args.garment_detections,
        jsonl_path=args.detections_jsonl,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )
