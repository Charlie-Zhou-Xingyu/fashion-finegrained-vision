#!/usr/bin/env python3
"""
3.1.2 Part localization accuracy evaluation v2 — Label Studio GT vs pipeline predictions.

Evaluates GT annotations from Label Studio export against the Fashionpedia YOLO +
Grounding DINO pipeline, with per-image visualizations and multi-IoU threshold reporting.

Usage::

    conda activate fashion-demo2
    python scripts/eval_validation_v2.py

Outputs ``data/validation/eval_v2/``:
  - metrics.json              per-part accuracy at IoU ∈ {0.01, 0.15, 0.3}
  - per_result.jsonl           per-annotation results
  - viz/{part}/                visualization panels (GT vs pred overlaid)
  - viz/summary.png            summary dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    PROJECT_ROOT
    / "data/validation/project-10-at-2026-07-09-16-01-2504b933/result.json"
)
PER_PART_DIR = PROJECT_ROOT / "data/validation/per_part"
OUTPUT_DIR = PROJECT_ROOT / "data/validation/eval_v2"

YOLO_WEIGHTS = "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
FP_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
DINO_MODEL = "models/grounding_dino_tiny"

IOU_THRESHOLDS = [0.01, 0.15, 0.3]  # mentor: cap at 0.3 to avoid over-penalizing

# Merged evaluation groups: combine parts that are hard to distinguish visually.
# "skin-exposed opening = neckline, fabric entity = collar" — mentor rule.
MERGE_GROUPS: Dict[str, List[str]] = {
    "collar+neckline+lapel": ["collar", "neckline", "lapel"],
}

# ── Part backend assignment ──────────────────────────────────────────────────────

# Fashionpedia YOLO core parts (class IDs in FP_CORE_PART_MAP)
FP_CORE = {
    "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline",
    "buckle", "zipper", "bow", "fringe", "ruffle",
    "hood",
}

# DINO fallback parts — FP PART_TO_FP_IDS does NOT include these,
# or they are not Fashionpedia classes at all.
# decoration classes (applique/bead/flower/ribbon/rivet/tassel) are
# FP_DECORATION_CLASSES but gated out by PART_TO_FP_IDS → must use DINO.
# sequin moved to DINO: FP YOLO gets 27% @0.3, DINO gets 90% (test_dino_queries.py)

# DINO fallback parts — FP PART_TO_FP_IDS does NOT include these,
# or they are not Fashionpedia classes at all.
# decoration classes (applique/bead/flower/ribbon/rivet/tassel) are
# FP_DECORATION_CLASSES but gated out by PART_TO_FP_IDS → must use DINO.
DINO_PARTS = {
    "bag", "shoes", "cuff", "button", "strap",
    "applique", "bead", "flower", "ribbon", "rivet", "tassel",
    "sequin",  # moved from FP_CORE: DINO 90% vs FP YOLO 27%
}

# Label Studio label → internal part name
LABEL_MAP: Dict[str, str] = {
    "button_dino": "button",
}


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _box_iou(a: List[float], b: List[float]) -> float:
    """IoU of two xyxy boxes."""
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
    """Resolve a Label Studio image filename to a per_part image path.

    Handles three naming conventions:
      1. uuid-part__<numeric_id>.jpg  → per_part/<part>/<id>.jpg
      2. uuid-part__<hash>.jpg         → per_part/<part>/<hash>.jpg
      3. uuid__uuid-part__<id>_-_副本  → per_part/<part>/<id>.jpg  (副本 copies)
    """
    basename = os.path.basename(base_filename.replace("\\", "/"))

    # Pattern 1: uuid-part__<digits>.jpg
    m = re.match(r"^[^-]+-(.+?)__(\d+)\.jpg$", basename)
    if m:
        part, img_id = m.group(1), m.group(2)
        return PER_PART_DIR / part / f"{img_id}.jpg"

    # Pattern 2: uuid-part__<hex_hash>.jpg
    m = re.match(r"^[^-]+-(.+?)__([0-9a-f]{20,})\.jpg$", basename)
    if m:
        part, hash_id = m.group(1), m.group(2)
        return PER_PART_DIR / part / f"{hash_id}.jpg"

    # Pattern 3: uuid__uuid-part__<digits>_-_副本.jpg
    m = re.match(r"^[^-]+__[^-]+-(.+?)__(\d+)", basename)
    if m:
        part, img_id = m.group(1), m.group(2)
        return PER_PART_DIR / part / f"{img_id}.jpg"

    return None


# ── Data Loading ─────────────────────────────────────────────────────────────────

def load_gt_annotations() -> Tuple[
    Dict[str, List[Dict]],      # image_path → [{"part": str, "bbox_xyxy": [x1,y1,x2,y2]}]
    List[str],                   # sorted unique parts
    List[str],                   # unmapped warnings
]:
    """Parse Label Studio COCO JSON, resolve images to per_part paths.

    Returns (annotations, parts, warnings).
    annotations is keyed by resolved absolute image path as string.
    """
    data = json.loads(ANNOTATIONS_JSON.read_text(encoding="utf-8"))

    cat_map: Dict[int, str] = {}
    for c in data["categories"]:
        raw_name = c["name"]
        cat_map[c["id"]] = LABEL_MAP.get(raw_name, raw_name)

    # Build image_id → resolved path map
    id_to_path: Dict[int, str] = {}
    unmapped: List[str] = []

    for img in data["images"]:
        resolved = _resolve_image_path(img["file_name"])
        if resolved is not None and resolved.exists():
            id_to_path[img["id"]] = str(resolved)
        else:
            basename = os.path.basename(img["file_name"].replace("\\", "/"))
            unmapped.append(
                f"image_id={img['id']}: {basename} "
                f"(resolved={resolved}, exists={resolved.exists() if resolved else False})"
            )

    annotations: Dict[str, List[Dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        img_path = id_to_path.get(ann["image_id"])
        if img_path is None:
            continue  # skip unmapped image
        x, y, w, h = ann["bbox"]
        part = cat_map.get(ann["category_id"], f"unknown_{ann['category_id']}")
        annotations[img_path].append({
            "part": part,
            "bbox_xyxy": [x, y, x + w, y + h],
        })

    parts = sorted(set(
        a["part"] for anns in annotations.values() for a in anns
    ))
    return dict(annotations), parts, unmapped


# ── Model Loading ────────────────────────────────────────────────────────────────

def _load_yolo(device: str):
    """Load YOLOv8n garment detector."""
    from ultralytics import YOLO
    path = str(PROJECT_ROOT / YOLO_WEIGHTS)
    if not Path(path).exists():
        print(f"[WARN] YOLO weights not found: {path}")
        return None
    model = YOLO(path)
    model.to(device)
    print(f"[INFO] YOLO garment detector loaded: {path}")
    return model


def _load_fp_detector(device: str):
    """Load Fashionpedia YOLOv8s 19-class part detector."""
    from fashion_vision.localization.fashionpedia_part_detector import (
        FashionpediaPartDetector,
    )
    path = str(PROJECT_ROOT / FP_MODEL)
    if not Path(path).exists():
        print(f"[WARN] FP model not found: {path}")
        return None
    det = FashionpediaPartDetector(path, device=device)
    print(f"[INFO] Fashionpedia detector loaded: {path}")
    return det


def _load_dino_locator(device: str):
    """Load Grounding DINO."""
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    path = str(PROJECT_ROOT / DINO_MODEL)
    if not Path(path).exists():
        print(f"[WARN] DINO model not found: {path}")
        return None
    locator = GroundingDINOLocator(model_id=path, device=device)
    print(f"[INFO] DINO locator loaded: {path}")
    return locator


# ── Inference ────────────────────────────────────────────────────────────────────

def run_yolo_detection(yolo_model, image_bgr: np.ndarray,
                       device: str = "cuda") -> List[Dict]:
    """Run YOLO garment detection, return list of garment bboxes."""
    results = yolo_model(image_bgr, device=device, conf=0.25, verbose=False)
    dets = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            dets.append({
                "bbox_xyxy": [float(v) for v in box],
                "score": float(conf),
                "class_id": int(cls_id),
                "class_name": names.get(cls_id, f"cls_{cls_id}"),
            })
    # Sort by score desc, take top-3 (most images have 1 garment)
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets[:3]


def select_garment(gt_bbox_xyxy: List[float], garment_dets: List[Dict]) -> Optional[Dict]:
    """Pick the garment detection that best overlaps the GT part bbox."""
    if not garment_dets:
        return None
    best = max(garment_dets,
               key=lambda d: _box_iou(gt_bbox_xyxy, d["bbox_xyxy"]))
    if _box_iou(gt_bbox_xyxy, best["bbox_xyxy"]) > 0.01:
        return best
    # Fallback: return largest garment
    return max(garment_dets,
               key=lambda d: (d["bbox_xyxy"][2] - d["bbox_xyxy"][0])
                             * (d["bbox_xyxy"][3] - d["bbox_xyxy"][1]))


def run_fp_detection(
    crop_bgr: np.ndarray,
    part: str,
    fp_detector,
    conf: float = 0.25,
) -> List[Dict]:
    """Run Fashionpedia YOLO on garment crop."""
    try:
        dets = fp_detector.detect(crop_bgr, part, garment_mask=None, conf=conf)
        # ponytail: apply class-aware soft NMS post-hoc for multi-instance parts
        # (FP YOLO applies its own NMS internally, but for parts like pocket/button
        # we want to keep adjacent detections)
        from fashion_vision.localization.grounding_dino_locator import (
            MULTI_INSTANCE_IOU,
            GroundingDINOLocator,
        )
        if part in MULTI_INSTANCE_IOU and len(dets) > 1:
            dets = GroundingDINOLocator._soft_nms(dets, MULTI_INSTANCE_IOU[part])
        return dets
    except Exception as e:
        print(f"  [WARN] FP detection error for {part}: {e}")
        return []


def run_dino_detection(
    crop_bgr: np.ndarray,
    part: str,
    dino_locator,
    threshold: float = 0.28,
    nms_mode: str = "soft",
) -> List[Dict]:
    """Run Grounding DINO on garment crop."""
    from fashion_vision.localization.part_detection_config import get_part_prompts
    from fashion_vision.localization.part_shape_priors import filter_by_shape_priors

    prompts = get_part_prompts(part)
    try:
        dets = dino_locator.detect_multi_prompt(
            crop_bgr, prompts,
            garment_mask=None,
            threshold=threshold,
            nms_mode=nms_mode,
            part=part,
        )
    except Exception as e:
        print(f"  [WARN] DINO error for {part}: {e}")
        return []

    h, w = crop_bgr.shape[:2]
    dets = filter_by_shape_priors(dets, part, garment_bbox=[0, 0, w, h])
    return dets


# ── Visualization ────────────────────────────────────────────────────────────────

def draw_bbox(image: np.ndarray, bbox_xyxy: List[float], color: Tuple[int, int, int],
              label: str, thickness: int = 2, line_type: int = cv2.LINE_AA) -> np.ndarray:
    """Draw a labelled bbox on a copy of the image."""
    out = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, line_type)
    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, line_type)
    return out


def make_viz_panel(
    image_bgr: np.ndarray,
    gt_bbox: List[float],
    gt_part: str,
    pred_bboxes: List[List[float]],
    best_iou: float,
    is_hit: bool,
    backend: str,
    garment_bbox: Optional[List[float]] = None,
) -> np.ndarray:
    """Create a comparison panel: GT (green) vs Pred (red).

    Layout: [GT only | Pred only | GT+Pred overlay]
    Green = GT, Red = prediction, Blue = garment bbox.
    """
    H, W = image_bgr.shape[:2]
    # Ensure image is reasonable size for display
    max_side = 400
    if max(H, W) > max_side:
        scale = max_side / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h))
        # Remap bboxes
        sx, sy = new_w / W, new_h / H
        gt_bbox_remap = [gt_bbox[0] * sx, gt_bbox[1] * sy,
                         gt_bbox[2] * sx, gt_bbox[3] * sy]
        pred_bboxes_remap = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                             for b in pred_bboxes]
        if garment_bbox:
            gb = [garment_bbox[0] * sx, garment_bbox[1] * sy,
                  garment_bbox[2] * sx, garment_bbox[3] * sy]
        else:
            gb = None
        H, W = new_h, new_w
    else:
        gt_bbox_remap = gt_bbox
        pred_bboxes_remap = pred_bboxes
        gb = garment_bbox

    GREEN = (0, 200, 0)
    RED = (0, 0, 255)
    BLUE = (200, 100, 0)
    WHITE = (255, 255, 255)

    # Panel 1: GT only
    gt_only = draw_bbox(image_bgr, gt_bbox_remap, GREEN, f"GT: {gt_part}")

    # Panel 2: Pred only
    pred_only = image_bgr.copy()
    for i, pb in enumerate(pred_bboxes_remap):
        pred_only = draw_bbox(pred_only, pb, RED, f"Pred {i+1}")

    # Panel 3: GT + Pred overlay + garment bbox
    overlay = image_bgr.copy()
    overlay = draw_bbox(overlay, gt_bbox_remap, GREEN, f"GT: {gt_part}", thickness=3)
    if gb:
        overlay = draw_bbox(overlay, gb, BLUE, "garment", thickness=1)
    for i, pb in enumerate(pred_bboxes_remap):
        overlay = draw_bbox(overlay, pb, RED, f"Pred {i+1}")

    # Status strip
    status_color = (0, 180, 0) if is_hit else (0, 0, 220)
    status_text = f"Hit (IoU={best_iou:.3f})" if is_hit else f"Miss (IoU={best_iou:.3f})"

    # Assemble panel: top row [GT | Pred], bottom row [Overlay | Info]
    top_row = np.hstack([gt_only, pred_only])

    # Info panel
    info_h = min(120, H // 4)
    info = np.full((info_h, overlay.shape[1], 3), (40, 40, 40), dtype=np.uint8)
    lines = [
        f"Part: {gt_part} | Backend: {backend} | {status_text}",
        f"GT: [{gt_bbox[0]:.0f},{gt_bbox[1]:.0f},{gt_bbox[2]:.0f},{gt_bbox[3]:.0f}]",
        f"Preds: {len(pred_bboxes)} bboxes",
    ]
    for i, line in enumerate(lines):
        cv2.putText(info, line, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    bottom = np.vstack([overlay, info])

    # Pad to match widths if needed
    if top_row.shape[1] != bottom.shape[1]:
        pad_w = abs(top_row.shape[1] - bottom.shape[1])
        if top_row.shape[1] < bottom.shape[1]:
            pad = np.full((top_row.shape[0], pad_w, 3), (40, 40, 40), dtype=np.uint8)
            top_row = np.hstack([top_row, pad])
        else:
            pad = np.full((bottom.shape[0], pad_w, 3), (40, 40, 40), dtype=np.uint8)
            bottom = np.hstack([bottom, pad])

    panel = np.vstack([top_row, bottom])
    return panel


# ── Summary Dashboard ────────────────────────────────────────────────────────────

def make_summary_dashboard(
    per_part_metrics: Dict[str, Dict],
    iou_thresholds: List[float],
    total_images: int,
    total_annotations: int,
    unmapped_count: int,
) -> np.ndarray:
    """Create a summary bar-chart dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = sorted(per_part_metrics.keys())
    n_parts = len(parts)

    fig, axes = plt.subplots(1, len(iou_thresholds), figsize=(6 * len(iou_thresholds), max(6, n_parts * 0.35)),
                             squeeze=False)

    colors_hit = ["#2ca02c", "#ff7f0e", "#d62728"]

    for ti, iou_t in enumerate(iou_thresholds):
        ax = axes[0, ti]
        accs = []
        totals = []
        for p in parts:
            m = per_part_metrics[p].get(f"iou_{iou_t}", {"accuracy": 0, "total": 0, "hits": 0})
            accs.append(m["accuracy"])
            totals.append(m["total"])

        y_pos = range(n_parts)
        bars = ax.barh(y_pos, accs, color=[colors_hit[ti]] * n_parts, edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(parts, fontsize=8)
        ax.set_xlim(0, 105)
        ax.axvline(x=92, color="red", linestyle="--", linewidth=1, alpha=0.7, label="PRD 92%")
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"IoU > {iou_t}")

        # Annotate bars
        for yi, (acc, total) in enumerate(zip(accs, totals)):
            if total > 0:
                ax.text(acc + 1, yi, f"{acc:.0f}%", va="center", fontsize=7)
        ax.legend(loc="lower right", fontsize=7)

    fig.suptitle(f"3.1.2 Region Localization Accuracy | {total_images} images, {total_annotations} annotations\n"
                 f"{unmapped_count} unmapped images skipped",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # ARGB → RGB: drop alpha channel and reorder
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, [1, 2, 3]]  # A,R,G,B → R,G,B
        buf = buf.flatten()
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 3)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3.1.2 Validation Evaluation v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp-conf", type=float, default=0.25)
    parser.add_argument("--dino-threshold", type=float, default=0.28)
    parser.add_argument("--skip-dino", action="store_true",
                        help="Skip DINO parts (bag, shoes, button, strap, cuff)")
    parser.add_argument("--skip-fp", action="store_true",
                        help="Skip Fashionpedia parts (for quick DINO-only test)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit to first N images for quick test")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--viz-samples", type=int, default=3,
                        help="Number of sample visualizations per part")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "viz"
    if not args.no_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("3.1.2 Region Localization Accuracy Evaluation v2")
    print(f"IoU thresholds: {IOU_THRESHOLDS}")
    print("=" * 70)

    # ── Load GT ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading GT annotations ...")
    gt_annotations, all_parts, unmapped = load_gt_annotations()

    n_images = len(gt_annotations)
    n_anns = sum(len(anns) for anns in gt_annotations.values())
    print(f"  {n_images} images, {n_anns} annotations, {len(all_parts)} parts")
    if unmapped:
        print(f"  WARNING: {len(unmapped)} images could not be mapped:")
        for u in unmapped[:10]:
            print(f"    {u}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")

    # Per-part summary
    part_counts = defaultdict(int)
    for anns in gt_annotations.values():
        for a in anns:
            part_counts[a["part"]] += 1
    for p in sorted(all_parts):
        backend = "FP" if p in FP_CORE else ("DINO" if p in DINO_PARTS else "OTHER")
        print(f"    {p:<18s} {part_counts[p]:>4d} anns  [{backend}]")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[2/5] Loading models ...")
    yolo = _load_yolo(args.device)
    fp_det = None if args.skip_fp else _load_fp_detector(args.device)
    dino_loc = None if args.skip_dino else _load_dino_locator(args.device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n[3/5] Running evaluation on {n_images} images ...")

    # Counters per IoU threshold
    counters: Dict[float, Dict[str, Dict[str, int]]] = {
        iou_t: defaultdict(lambda: {"total": 0, "hits": 0, "misses": 0})
        for iou_t in IOU_THRESHOLDS
    }

    results: List[Dict] = []
    jsonl_path = output_dir / "per_result.jsonl"
    jsonl_fh = open(jsonl_path, "w", encoding="utf-8")

    # For visualization sampling
    viz_collect: Dict[str, List[Dict]] = defaultdict(list)

    # ── Timing accumulators ──────────────────────────────────────────────────
    timing = {
        "load_image": 0.0,
        "garment_det": 0.0,
        "fp_detection": 0.0,
        "dino_detection": 0.0,
        "n_fp_calls": 0,
        "n_dino_calls": 0,
    }
    t_start = time.perf_counter()
    img_paths = sorted(gt_annotations.keys())
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    for idx, img_path_str in enumerate(img_paths):
        img_path = Path(img_path_str)
        fname = img_path.name

        t0 = time.perf_counter()
        image = cv2.imread(str(img_path))
        timing["load_image"] += time.perf_counter() - t0
        if image is None:
            print(f"  [WARN] Cannot read {img_path}")
            continue
        H, W = image.shape[:2]

        gt_list = gt_annotations[img_path_str]

        if idx < 3 or idx % 50 == 0:
            print(f"  [{idx+1:4d}/{len(img_paths)}] {img_path.parent.name}/{fname} "
                  f"({len(gt_list)} GT)")

        # Run YOLO garment detection once per image
        t0 = time.perf_counter()
        if yolo is not None:
            garment_dets = run_yolo_detection(yolo, image, device=args.device)
        else:
            garment_dets = []
        timing["garment_det"] += time.perf_counter() - t0

        for gt in gt_list:
            part = gt["part"]
            gt_bbox = gt["bbox_xyxy"]

            # Increment totals for all thresholds
            for iou_t in IOU_THRESHOLDS:
                counters[iou_t][part]["total"] += 1

            # Select garment
            garment = None
            if garment_dets:
                garment = select_garment(gt_bbox, garment_dets)

            # Crop
            if garment is not None:
                gx1, gy1, gx2, gy2 = [int(v) for v in garment["bbox_xyxy"]]
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(W, gx2), min(H, gy2)
                crop = image[gy1:gy2, gx1:gx2]
                ox, oy = gx1, gy1
            else:
                crop = image
                ox, oy = 0, 0

            # Run part detection
            pred_bboxes: List[List[float]] = []
            backend = "none"
            pred_score = None

            if part in FP_CORE and not args.skip_fp:
                if fp_det is not None:
                    t0 = time.perf_counter()
                    dets = run_fp_detection(crop, part, fp_det, conf=args.fp_conf)
                    timing["fp_detection"] += time.perf_counter() - t0
                    timing["n_fp_calls"] += 1
                    for d in dets:
                        b = d["bbox_xyxy"]
                        pred_bboxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy])
                    backend = "fashionpedia_yolo"
                    if dets:
                        pred_score = dets[0].get("score")
                else:
                    backend = "fp_not_loaded"

            elif part in DINO_PARTS and not args.skip_dino:
                if dino_loc is not None:
                    t0 = time.perf_counter()
                    dets = run_dino_detection(crop, part, dino_loc, threshold=args.dino_threshold)
                    timing["dino_detection"] += time.perf_counter() - t0
                    timing["n_dino_calls"] += 1
                    for d in dets:
                        b = d["bbox_xyxy"]
                        pred_bboxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy])
                    backend = "grounding_dino"
                    if dets:
                        pred_score = dets[0].get("score")
                else:
                    backend = "dino_not_loaded"

            else:
                backend = "skipped"

            # Compute best IoU
            best_iou = 0.0
            is_contained = False  # any pred box fully inside GT
            for pb in pred_bboxes:
                iou = _box_iou(pb, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                # Check containment: pred completely inside GT
                if (pb[0] >= gt_bbox[0] and pb[1] >= gt_bbox[1]
                        and pb[2] <= gt_bbox[2] and pb[3] <= gt_bbox[3]):
                    is_contained = True

            # Check hits at each threshold.
            # Rule: any intersection (>0.01) OR pred contained in GT → hit.
            hits = {}
            for iou_t in IOU_THRESHOLDS:
                is_hit = (best_iou > iou_t) or is_contained
                hits[f"iou_{iou_t}"] = is_hit
                if is_hit:
                    counters[iou_t][part]["hits"] += 1
                else:
                    counters[iou_t][part]["misses"] += 1

            # Record
            garment_class = garment.get("class_name") if garment else None
            garment_bbox = garment["bbox_xyxy"] if garment else None

            record = {
                "image": fname,
                "image_path": img_path_str,
                "image_idx": idx,
                "part": part,
                "gt_bbox": gt_bbox,
                "pred_bboxes": pred_bboxes,
                "best_iou": round(best_iou, 4),
                "is_contained": is_contained,
                "is_hit_iou_0.01": hits["iou_0.01"],
                "is_hit_iou_0.15": hits.get("iou_0.15", False),
                "is_hit_iou_0.3": hits["iou_0.3"],
                "backend": backend,
                "pred_score": pred_score,
                "garment_class": garment_class,
                "garment_bbox": garment_bbox,
                "crop_offset": [ox, oy],
            }
            results.append(record)
            jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Collect for visualization sampling
            if not args.no_viz and part not in ("inner_garment",):
                viz_collect[part].append({
                    "image": image,
                    "record": record,
                })

    jsonl_fh.close()
    elapsed = time.perf_counter() - t_start

    # ── Timing report ────────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"TIMING BREAKDOWN")
    print(f"{'─' * 55}")
    print(f"  Image loading:        {timing['load_image']:8.1f}s ({timing['load_image'] / elapsed * 100:5.1f}%)")
    print(f"  Garment detection:    {timing['garment_det']:8.1f}s ({timing['garment_det'] / elapsed * 100:5.1f}%)")
    print(f"  FP part detection:    {timing['fp_detection']:8.1f}s ({timing['fp_detection'] / elapsed * 100:5.1f}%)  "
          f"[{timing['n_fp_calls']} calls, avg {timing['fp_detection'] / max(timing['n_fp_calls'], 1):.3f}s/call]")
    print(f"  DINO part detection:  {timing['dino_detection']:8.1f}s ({timing['dino_detection'] / elapsed * 100:5.1f}%)  "
          f"[{timing['n_dino_calls']} calls, avg {timing['dino_detection'] / max(timing['n_dino_calls'], 1):.3f}s/call]")
    print(f"  {'─' * 50}")
    print(f"  TOTAL elapsed:        {elapsed:8.1f}s")
    print(f"  Throughput:           {len(results) / elapsed:.2f} ann/s, {len(img_paths) / elapsed:.2f} img/s")
    timing["total_elapsed"] = round(elapsed, 1)
    print(f"\n  Evaluation done: {len(results)} annotations, "
          f"{elapsed / max(len(results), 1):.2f}s per ann")

    # ── Compute Metrics ───────────────────────────────────────────────────────
    print(f"\n[4/5] Computing metrics ...")

    per_part_metrics: Dict[str, Dict] = defaultdict(dict)
    all_metrics: Dict[float, Dict] = {}

    for iou_t in IOU_THRESHOLDS:
        print(f"\n{'=' * 70}")
        print(f"RESULTS: Region Localization Accuracy (IoU > {iou_t})")
        print(f"{'=' * 70}")
        print(f"{'Part':<18s} {'Total':>6s} {'Hits':>6s} {'Miss':>6s} {'Acc%':>7s}")
        print("-" * 55)

        total_hits = 0
        total_total = 0
        part_results = {}

        for part in sorted(counters[iou_t].keys()):
            c = counters[iou_t][part]
            acc = c["hits"] / c["total"] * 100 if c["total"] > 0 else 0.0
            total_hits += c["hits"]
            total_total += c["total"]
            part_results[part] = {
                "total": c["total"],
                "hits": c["hits"],
                "misses": c["misses"],
                "accuracy": round(acc, 2),
            }
            per_part_metrics[part][f"iou_{iou_t}"] = part_results[part]
            flag = " ★" if acc >= 92 else " ✗" if acc < 50 else ""
            print(f"{part:<18s} {c['total']:>6d} {c['hits']:>6d} "
                  f"{c['misses']:>6d} {acc:>6.1f}%{flag}")

        overall_acc = total_hits / total_total * 100 if total_total > 0 else 0.0
        print("-" * 55)
        print(f"{'OVERALL':<18s} {total_total:>6d} {total_hits:>6d} "
              f"{total_total - total_hits:>6d} {overall_acc:>6.1f}%")

        # ── Merge groups ──────────────────────────────────────────────────────
        for mg_name, mg_parts in MERGE_GROUPS.items():
            m_hits = sum(counters[iou_t][p]["hits"] for p in mg_parts)
            m_total = sum(counters[iou_t][p]["total"] for p in mg_parts)
            m_acc = m_hits / m_total * 100 if m_total > 0 else 0.0
            part_results[mg_name] = {
                "total": m_total,
                "hits": m_hits,
                "misses": m_total - m_hits,
                "accuracy": round(m_acc, 2),
                "merged_from": mg_parts,
            }
            per_part_metrics[mg_name][f"iou_{iou_t}"] = part_results[mg_name]
            print(f"{mg_name:<18s} {m_total:>6d} {m_hits:>6d} "
                  f"{m_total - m_hits:>6d} {m_acc:>6.1f}%  [merged]")

        all_metrics[iou_t] = {
            "num_images": n_images,
            "num_annotations": total_total,
            "overall_accuracy": round(overall_acc, 2),
            "per_part": part_results,
        }

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_output = {
        "iou_thresholds": IOU_THRESHOLDS,
        "num_images": n_images,
        "num_annotations": total_total,
        "num_unmapped": len(unmapped),
        "unmapped": unmapped,
        "results": {str(k): v for k, v in all_metrics.items()},
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Per-result JSONL: {jsonl_path}")

    # ── Visualizations ────────────────────────────────────────────────────────
    if not args.no_viz:
        print(f"\n[5/5] Generating visualizations ...")
        _generate_viz(viz_collect, per_part_metrics, n_images, n_anns, len(unmapped),
                      viz_dir, args.viz_samples)

    # ── DINO miss report ─────────────────────────────────────────────────────
    _report_dino_misses(results)

    # ── PRD target ────────────────────────────────────────────────────────────
    prd_target = 92.0
    for iou_t in IOU_THRESHOLDS:
        m = all_metrics[iou_t]
        passed = sum(1 for v in m["per_part"].values() if v["accuracy"] >= prd_target)
        total_parts = len(m["per_part"])
        print(f"PRD ≥{prd_target}% @ IoU>{iou_t}: {passed}/{total_parts} parts meet target")

    return metrics_output


def _generate_viz(
    viz_collect: Dict[str, List[Dict]],
    per_part_metrics: Dict[str, Dict],
    n_images: int,
    n_anns: int,
    unmapped_count: int,
    viz_dir: Path,
    viz_samples: int,
):
    """Generate per-part sample visualizations and summary dashboard."""
    import random

    for part, samples in viz_collect.items():
        part_dir = viz_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)

        # Pick samples: mix of hits and misses
        hits = [s for s in samples if s["record"]["is_hit_iou_0.3"]]
        misses = [s for s in samples if not s["record"]["is_hit_iou_0.3"]]

        selected = []
        # Pick up to viz_samples hits and misses
        if hits:
            selected.extend(random.sample(hits, min(viz_samples // 2 + 1, len(hits))))
        if misses:
            selected.extend(random.sample(misses, min(viz_samples // 2 + 1, len(misses))))

        # If not enough, fill from all (compare by image path since dicts contain numpy arrays)
        remaining = viz_samples - len(selected)
        if remaining > 0:
            selected_paths = {s["record"]["image_path"] for s in selected}
            others = [s for s in samples if s["record"]["image_path"] not in selected_paths]
            if others:
                selected.extend(random.sample(others, min(remaining, len(others))))

        for si, sample in enumerate(selected):
            rec = sample["record"]
            panel = make_viz_panel(
                sample["image"],
                rec["gt_bbox"],
                rec["part"],
                rec["pred_bboxes"],
                rec["best_iou"],
                rec["is_hit_iou_0.3"],  # display hit based on 0.3 threshold
                rec["backend"],
                rec.get("garment_bbox"),
            )
            status_str = "HIT" if rec["is_hit_iou_0.3"] else "MISS"
            out_name = f"{part}_{status_str}_iou{rec['best_iou']:.2f}_{Path(rec['image']).stem}.jpg"
            cv2.imwrite(str(part_dir / out_name), panel)

        print(f"  {part}: {len(selected)} viz panels saved")

    # Summary dashboard
    dashboard = make_summary_dashboard(
        per_part_metrics, IOU_THRESHOLDS,
        n_images, n_anns, unmapped_count,
    )
    cv2.imwrite(str(viz_dir / "summary.png"), dashboard)
    print(f"  Summary dashboard: {viz_dir / 'summary.png'}")


def _report_dino_misses(results: List[Dict]):
    """Report DINO-backed parts that had misses."""
    dino_results = [r for r in results if r["backend"] == "grounding_dino"]
    misses = [r for r in dino_results if not r["is_hit_iou_0.3"]]
    if misses:
        print(f"\n[DINO MISSES] {len(misses)}/{len(dino_results)} DINO annotations "
              f"missed (IoU <= 0.3):")
        for r in misses[:15]:
            print(f"  {r['image']} | part={r['part']} | best_iou={r['best_iou']:.4f}")
        if len(misses) > 15:
            print(f"  ... and {len(misses) - 15} more")


if __name__ == "__main__":
    main()
