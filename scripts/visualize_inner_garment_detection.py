#!/usr/bin/env python3
"""
Visual evaluation of SAM-based inner-garment detection.

Reuses the existing GarmentPipeline (3.1.1) for YOLO detection + SAM
segmentation, then runs inner-garment detection on outerwear instances.

Output: per-image 4x2 panel PNGs under --output-dir.

Usage:
    set PYTHONPATH=%CD%\src && python scripts/visualize_inner_garment_detection.py ^
      --image-dir "D:/Aliintern/fashion-ai-data/.../Images/lapel_design_labels" ^
      --num-images 50 ^
      --output-dir outputs/visual_tests/inner_garment_v1 ^
      --device cuda ^
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _resize_to_height_with_scale(img: np.ndarray, h: int) -> tuple[np.ndarray, float, float]:
    ih, iw = img.shape[:2]
    sy = h / max(1, ih)
    sx = sy
    nw = max(1, int(iw * sx))
    return cv2.resize(img, (nw, h)), sx, sy


def _scale_bbox(bbox, scale_x: float, scale_y: float) -> list[int]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [int(round(x1 * scale_x)), int(round(y1 * scale_y)),
            int(round(x2 * scale_x)), int(round(y2 * scale_y))]


def _resize_panel_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    img = _to_bgr(img)
    if img.shape[0] == target_h:
        return img
    nw = max(1, int(img.shape[1] * target_h / max(1, img.shape[0])))
    return cv2.resize(img, (nw, target_h))


def _make_panel(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    bar_h = 28
    h, w = img.shape[:2]
    panel = np.full((h + bar_h, w, 3), 40, dtype=np.uint8)
    panel[bar_h:, :] = _to_bgr(img)
    cv2.putText(panel, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    if subtitle:
        for j, line in enumerate(subtitle.split("\n")[:2]):
            cv2.putText(panel, line, (4, bar_h - 4 - j * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    return panel


def _make_grid(panels: list[np.ndarray], cols: int = 2) -> np.ndarray:
    rows: list[np.ndarray] = []
    row_widths: list[int] = []
    for i in range(0, len(panels), cols):
        row_panels = panels[i:i + cols]
        row_h = max(p.shape[0] for p in row_panels)
        row_panels = [_resize_panel_to_height(p, row_h) for p in row_panels]
        row = np.hstack(row_panels)
        rows.append(row)
        row_widths.append(row.shape[1])
    max_w = max(row_widths)
    padded = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)
    return np.vstack(padded)


def _draw_mask_overlay(img: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    if mask is None or mask.sum() == 0:
        return
    overlay = img.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def is_outerwear_class(class_name: str) -> bool:
    name = str(class_name).lower().strip()
    return ("outwear" in name or "outerwear" in name or "coat" in name
            or "jacket" in name or "cardigan" in name)


def _draw_bbox(img: np.ndarray, bbox, color=(0, 255, 0), thickness=2, label=""):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)


def _draw_dashed_bbox(img: np.ndarray, bbox, color=(64, 165, 255),
                      dash_len: int = 6, gap_len: int = 3, thickness: int = 2,
                      label: str = ""):
    """Draw a dashed rectangle on a BGR image in-place."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    for x in range(x1, x2, dash_len + gap_len):
        xe = min(x + dash_len, x2)
        cv2.line(img, (x, y1), (xe, y1), color, thickness)
        cv2.line(img, (x, y2), (xe, y2), color, thickness)
    for y in range(y1, y2, dash_len + gap_len):
        ye = min(y + dash_len, y2)
        cv2.line(img, (x1, y), (x1, ye), color, thickness)
        cv2.line(img, (x2, y), (x2, ye), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(img, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, color, 1, cv2.LINE_AA)


def _blank_panel(h: int) -> np.ndarray:
    return np.full((h, 100, 3), 40, dtype=np.uint8)


def _build_summary(output_dir: Path, viz_dir: Path) -> dict:
    """Build summary dict from global counters set during processing."""
    return {
        "num_images_requested": _summary_state.get("num_requested", 0),
        "num_images_processed": _summary_state.get("n_processed", 0),
        "num_outerwear_found": _summary_state.get("n_outerwear", 0),
        "num_inner_found": _summary_state.get("n_inner_found", 0),
        "num_inner_extended": _summary_state.get("n_extended", 0),
        "num_boundary_refine_attempted": _summary_state.get("n_refine_attempted", 0),
        "num_boundary_refine_accepted": _summary_state.get("n_refine_accepted", 0),
        "num_boundary_refine_rejected": _summary_state.get("n_refine_rejected", 0),
        "num_artifact_cleanup_accepted": _summary_state.get("n_cleanup_accepted", 0),
        "num_artifact_cleanup_rejected": _summary_state.get("n_cleanup_rejected", 0),
        "avg_cleanup_removed_pixels": round(
            _summary_state.get("total_cleanup_removed", 0)
            / max(1, _summary_state.get("n_cleanup_accepted", 1))
        ),
        "num_no_inner": _summary_state.get("n_no_inner", 0),
        "output_dir": str(output_dir),
        "viz_dir": str(viz_dir),
        "top_reject_reasons": dict(
            sorted(_summary_state.get("reject_reason_counts", {}).items(),
                   key=lambda x: -x[1])[:10]
        ),
    }


# ── Module-level state for summary ────────────────────────────────────────────
_summary_state: dict = {}


# ── Main logic ─────────────────────────────────────────────────────────────────

def run_visualization(
    image_dir: Path, num_images: int, output_dir: Path,
    device: str = "cuda", seed: int = 42,
):
    from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig
    from fashion_vision.models.sam_hq_wrapper import SamHqWrapper
    from fashion_vision.localization.inner_garment_detector import (
        detect_inner_garment_from_sam, _build_complement_search_mask,
    )

    # 1. Pick random images
    all_imgs = sorted(p for p in image_dir.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    rng = random.Random(seed)
    selected = rng.sample(all_imgs, min(num_images, len(all_imgs)))
    selected_names = {p.name for p in selected}
    print(f"Selected {len(selected)} images (seed={seed})")

    # 2. Stage to temp dir
    stage_dir = Path(tempfile.mkdtemp(prefix="inner_viz_"))
    for p in selected:
        shutil.copy2(p, stage_dir / p.name)
    print(f"Staged to {stage_dir}")

    # 3. Run GarmentPipeline
    pipeline_dir = output_dir / "_pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    # Resolve YOLO weights — try multiple paths
    yolo_paths = [
        Path("models/detectors/yolov8n_deepfashion2_13cls_best.pt"),
        Path("D:/best.pt"),
    ]
    yolo_w = None
    for yp in yolo_paths:
        if yp.exists():
            yolo_w = str(yp.resolve())
            break
    if yolo_w is None:
        raise FileNotFoundError("YOLO weights not found. Checked: " + ", ".join(str(p) for p in yolo_paths))

    config = GarmentPipelineConfig(
        yolo_weights=yolo_w,
        yolo_device=device.replace("cuda", "0") if device == "cuda" else device,
        sam_device=device,
        save_yolo_vis=False, save_yolo_crops=False,
        save_landmark_visualizations=False,
    )
    pipeline = GarmentPipeline(config)
    print("Running GarmentPipeline ...")
    result = pipeline.run_source(source=str(stage_dir), output_dir=str(pipeline_dir))
    print(f"  YOLO: {result['timing']['yolo_seconds']:.1f}s  SAM: {result['timing']['sam_hq_seconds']:.1f}s")

    # 4. Read outputs
    with open(result["paths"]["detections_json"], encoding="utf-8") as f:
        det_data = json.load(f)
    with open(result["paths"]["segmentation_json"], encoding="utf-8") as f:
        seg_data = json.load(f)

    # 5. SAM wrapper
    sam = SamHqWrapper(checkpoint=config.sam_checkpoint, model_type=config.sam_model_type,
                        device=config.sam_device)

    # 6. Process
    panel_h = 220
    viz_dir = output_dir / "panels"
    viz_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_raw_bbox"
    debug_dir.mkdir(parents=True, exist_ok=True)
    n_outerwear, n_inner_found = 0, 0
    n_processed, n_extended, n_no_inner = 0, 0, 0
    n_refine_attempted, n_refine_accepted, n_refine_rejected = 0, 0, 0
    n_cleanup_accepted, n_cleanup_rejected = 0, 0
    total_cleanup_removed = 0
    reject_reason_counts: dict[str, int] = {}

    for img_rec in seg_data.get("images", []):
        image_path = img_rec["image_path"]
        img_name = Path(image_path).name
        if img_name not in selected_names:
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        n_processed += 1
        outer_segs = [s for s in img_rec.get("segments", [])
                      if is_outerwear_class(s.get("class_name", ""))]
        if not outer_segs:
            continue
        n_outerwear += 1

        seg = outer_segs[0]
        bbox = [int(float(v)) for v in seg["bbox_xyxy"]]
        outer_mask = cv2.imread(seg.get("mask_path", ""), cv2.IMREAD_GRAYSCALE) if seg.get("mask_path") else None

        panels: list[np.ndarray] = []

        # Panel 0: original + outerwear bbox
        orig_viz, sc_x, sc_y = _resize_to_height_with_scale(img.copy(), panel_h)
        _draw_bbox(orig_viz, _scale_bbox(bbox, sc_x, sc_y), (0, 255, 255), 2, seg["class_name"])
        panels.append(_make_panel(orig_viz, f"Original ({img_name})"))

        # Debug raw bbox
        debug_raw = img.copy()
        _draw_bbox(debug_raw, bbox, (0, 255, 255), 2, seg["class_name"])
        cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}_raw_bbox.png"), debug_raw)

        # Panel 1: outerwear SAM mask
        mask_viz, _, _ = _resize_to_height_with_scale(img.copy(), panel_h)
        if outer_mask is not None:
            mv = cv2.resize(outer_mask, (mask_viz.shape[1], mask_viz.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            _draw_mask_overlay(mask_viz, mv, (0, 200, 255))
        panels.append(_make_panel(mask_viz, "Outerwear SAM mask",
                                  f"{seg['class_name']} conf={seg.get('confidence',0):.2f}"))

        # ── Inner garment detection ──────────────────────────────────────
        pseudo_inst = {"bbox_xyxy": bbox, "_mask": outer_mask, "coarse_class_name": "outerwear"}
        inner_result = detect_inner_garment_from_sam(img, pseudo_inst, sam)
        debug = inner_result.get("debug", {}) if inner_result else {}
        method = debug.get("method", "none")

        # --- Collect stats ---
        if inner_result is not None:
            n_inner_found += 1
            ext_dbg = debug.get("extension", {})
            if isinstance(ext_dbg, dict) and ext_dbg.get("extended"):
                n_extended += 1
            refine_dbg = debug.get("boundary_refinement", {})
            if isinstance(refine_dbg, dict) and (refine_dbg.get("h_refined") or refine_dbg.get("v_refined")):
                n_refine_attempted += 1
                if debug.get("refine_accepted"):
                    n_refine_accepted += 1
                else:
                    n_refine_rejected += 1
            # --- Collect cleanup stats ---
            cd2 = debug.get("artifact_cleanup", {})
            if isinstance(cd2, dict) and cd2:
                if cd2.get("cleanup_accepted"):
                    n_cleanup_accepted += 1
                    total_cleanup_removed += cd2.get("removed_pixels", 0)
                else:
                    n_cleanup_rejected += 1
            # Collect reject reasons from all candidates
            for cd in debug.get("all_debug_candidates", []):
                if not cd.get("passed"):
                    for reason in cd.get("reject_reasons", []):
                        # Extract category from first word
                        cat = reason.split("=")[0].split(" ")[0]
                        reject_reason_counts[cat] = reject_reason_counts.get(cat, 0) + 1
        else:
            n_no_inner += 1

        # Panel 2: neckline + opening ROI
        nbox = debug.get("neckline_box")
        obox = (debug.get("extension", {}).get("opening_box")
                if isinstance(debug.get("extension"), dict) else None)
        if nbox:
            neck_viz, nkx, nky = _resize_to_height_with_scale(img.copy(), panel_h)
            _draw_bbox(neck_viz, _scale_bbox(nbox, nkx, nky), (255, 200, 0), 2, "neckline")
            if obox:
                _draw_bbox(neck_viz, _scale_bbox(obox, nkx, nky), (200, 100, 255), 1, "opening")
            _draw_bbox(neck_viz, _scale_bbox(bbox, nkx, nky), (0, 255, 255), 1, "outer")
            opening_str = f" opening_y=[{obox[1]},{obox[3]}]" if obox else ""
            panels.append(_make_panel(neck_viz, "ROIs (orange=neckline, purple=opening)",
                                      f"neckline y=[{nbox[1]},{nbox[3]}]{opening_str}"))
        else:
            panels.append(_make_panel(_blank_panel(panel_h), "Neckline ROI", "not computed"))

        # Panel 3: complement search mask
        search_px = debug.get("search_mask_px", 0)
        if search_px > 0 and outer_mask is not None and nbox:
            outer_bin = (outer_mask > 0).astype(np.uint8)
            comp = _build_complement_search_mask(bbox, outer_bin, nbox, h_img, w_img)
            comp_viz, _, _ = _resize_to_height_with_scale(img.copy(), panel_h)
            comp_small = cv2.resize(comp, (comp_viz.shape[1], comp_viz.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            _draw_mask_overlay(comp_viz, comp_small, (0, 255, 100), alpha=0.6)
            panels.append(_make_panel(comp_viz, "Complement mask", f"{search_px} px"))
        else:
            panels.append(_make_panel(_blank_panel(panel_h), "Complement mask",
                                      f"source={method}"))

        # Panels 4-6: top 3 candidates
        all_cands = debug.get("all_debug_candidates", [])
        all_cands_sorted = sorted(all_cands, key=lambda c: c.get("score", -999), reverse=True)
        for i in range(3):
            if i < len(all_cands_sorted):
                cd = all_cands_sorted[i]
                passed = cd.get("passed", False)
                color = (0, 255, 0) if passed else (100, 100, 255)
                cviz, ccx, ccy = _resize_to_height_with_scale(img.copy(), panel_h)
                cb = _scale_bbox(cd.get("bbox", [0, 0, 0, 0]), ccx, ccy)
                _draw_bbox(cviz, cb, color, 2)
                _draw_bbox(cviz, _scale_bbox(bbox, ccx, ccy), (0, 255, 255), 1, "outer")
                src = cd.get("source", "?")
                score_s = cd.get("score", -99)
                reasons = cd.get("reject_reasons", [])
                short = reasons[0][:50] if reasons else ""
                status = "PASS" if passed else f"REJECT:{short}"
                subtitle = (
                    f"{status}\n"
                    f"s={score_s:.2f} src={src}\n"
                    f"out_out={cd.get('outside_outer_ratio',0):.2f} "
                    f"cx={cd.get('rel_cx',0):.2f} "
                    f"core={cd.get('opening_core_overlap',0):.2f}\n"
                    f"torso={cd.get('torso_overlap',0):.3f}/{cd.get('torso_min_overlap','?'):.3f} "
                    f"bw={cd.get('bbox_w_ratio',0):.3f} "
                    f"bh={cd.get('bbox_h_ratio',0):.3f} "
                    f"a_bb={cd.get('area_ratio_bbox',0):.4f}"
                )
                panels.append(_make_panel(cviz, f"Candidate {i+1} ({src})", subtitle))
            else:
                panels.append(_make_panel(_blank_panel(panel_h), "(no candidate)"))

        # Panel 7: selected inner / fallback
        if inner_result is not None:
            inner_viz, isc_x, isc_y = _resize_to_height_with_scale(img.copy(), panel_h)
            inner_mask = inner_result.get("mask")

            # --- Draw torso ROI (magenta semi-transparent) ---
            torso_mask_full = debug.get("torso_prior", {})
            if isinstance(torso_mask_full, dict):
                torso_bbox_t = torso_mask_full.get("torso_bbox")
                if torso_bbox_t:
                    _draw_bbox(inner_viz, _scale_bbox(torso_bbox_t, isc_x, isc_y),
                              (128, 0, 255), 2, "torso")

            # --- Draw opening ROI (purple box) ---
            ext_dbg = debug.get("extension", {})
            opening_box = ext_dbg.get("opening_box") if isinstance(ext_dbg, dict) else None
            if opening_box:
                _draw_bbox(inner_viz, _scale_bbox(opening_box, isc_x, isc_y),
                          (255, 0, 255), 1, "open")

            # --- Draw before-refine bbox (orange dashed) ---
            bbox_before = debug.get("bbox_before_refine")
            if bbox_before:
                bf = _scale_bbox(bbox_before, isc_x, isc_y)
                # dashed-line effect: use dotted segments
                _draw_dashed_bbox(inner_viz, bf, (64, 165, 255), dash_len=6, gap_len=3, label="before")

            # --- Draw inner mask (teal overlay) ---
            if inner_mask is not None:
                ms = cv2.resize(inner_mask, (inner_viz.shape[1], inner_viz.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
                _draw_mask_overlay(inner_viz, ms, (255, 200, 0), alpha=0.35)

            # --- Draw after-refine bbox (blue solid) ---
            inner_bbox = inner_result.get("bbox_xyxy")
            if inner_bbox:
                _draw_bbox(inner_viz, _scale_bbox(inner_bbox, isc_x, isc_y),
                          (255, 0, 0), 2, "inner")

            # --- Draw cleanup-before bbox (green dashed) ---
            cleanup_before_box = debug.get("bbox_before_cleanup")
            cleanup_dbg = debug.get("artifact_cleanup", {})
            if not isinstance(cleanup_dbg, dict):
                cleanup_dbg = {}
            if (cleanup_before_box and cleanup_dbg.get("cleanup_accepted")
                    and inner_bbox and inner_bbox != cleanup_before_box):
                _draw_dashed_bbox(inner_viz, _scale_bbox(cleanup_before_box, isc_x, isc_y),
                                (0, 200, 100), dash_len=6, gap_len=3, label="before_cln")

            # --- Build subtitle ---
            ext = ext_dbg if isinstance(ext_dbg, dict) else {}
            refine_dbg = debug.get("boundary_refinement", {})
            refine_safety = debug.get("boundary_refine_safety", {})
            refine_acc = debug.get("refine_accepted", False)
            if not isinstance(refine_dbg, dict):
                refine_dbg = {}
            if not isinstance(refine_safety, dict):
                refine_safety = {}

            sel_scoring = debug.get("selected_scoring", {})
            subtitle_lines = [
                f"method={inner_result.get('source','?')} "
                f"score={inner_result.get('score',0):.3f} "
                f"src={debug.get('selected_source','?')}",
                f"extended={ext.get('extended', False)} "
                f"matched={ext.get('num_matched', 0)}/{ext.get('num_opening_components', 0)} "
                f"torso_ov={sel_scoring.get('torso_overlap','?'):.3f}",
            ]
            if refine_dbg.get("h_refined") or refine_dbg.get("v_refined"):
                acc_str = "ACCEPT" if refine_acc else "REJECT"
                reason_str = refine_safety.get("reason", "")
                if reason_str:
                    reason_str = reason_str[:60]
                subtitle_lines.append(
                    f"refine: {acc_str}  reason={reason_str}"
                )
            else:
                subtitle_lines.append("refine: not triggered")

            # Cleanup status
            cl_acc = "ACCEPT" if cleanup_dbg.get("cleanup_accepted") else "REJECT"
            cl_removed = cleanup_dbg.get("removed_pixels", 0)
            cl_reason = (cleanup_dbg.get("reason") or "None")
            subtitle_lines.append(
                f"cleanup={cl_acc} removed={cl_removed} reason={cl_reason}"
            )

            subtitle = "\n".join(subtitle_lines)
            panels.append(_make_panel(inner_viz, "Selected inner garment", subtitle))
        else:
            no_viz, nsc_x, nsc_y = _resize_to_height_with_scale(img.copy(), panel_h)
            _draw_bbox(no_viz, _scale_bbox(bbox, nsc_x, nsc_y), (100, 100, 100), 2, "outer only")
            panels.append(_make_panel(no_viz, "No inner garment found",
                f"method={method}"))

        # Save
        grid = _make_grid(panels, cols=2)
        out_path = viz_dir / f"{Path(image_path).stem}_inner_detection.png"
        cv2.imwrite(str(out_path), grid)

    shutil.rmtree(stage_dir, ignore_errors=True)

    # ── Generate summary ──────────────────────────────────────────────────
    _summary_state.update({
        "num_requested": num_images,
        "n_processed": n_processed,
        "n_outerwear": n_outerwear,
        "n_inner_found": n_inner_found,
        "n_extended": n_extended,
        "n_refine_attempted": n_refine_attempted,
        "n_refine_accepted": n_refine_accepted,
        "n_refine_rejected": n_refine_rejected,
        "n_no_inner": n_no_inner,
        "n_cleanup_accepted": n_cleanup_accepted,
        "n_cleanup_rejected": n_cleanup_rejected,
        "total_cleanup_removed": total_cleanup_removed,
        "reject_reason_counts": reject_reason_counts,
    })
    summary = _build_summary(output_dir, viz_dir)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")
    print(f"  outerwear found  : {summary['num_outerwear_found']}")
    print(f"  inner found      : {summary['num_inner_found']}")
    print(f"  inner extended   : {summary['num_inner_extended']}")
    print(f"  refine attempted : {summary['num_boundary_refine_attempted']}")
    print(f"  refine accepted  : {summary['num_boundary_refine_accepted']}")
    print(f"  refine rejected  : {summary['num_boundary_refine_rejected']}")
    print(f"  no inner         : {summary['num_no_inner']}")
    if summary.get("top_reject_reasons"):
        print(f"  top reject reasons: {summary['top_reject_reasons']}")
    print(f"Output: {viz_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--num-images", type=int, default=50)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/visual_tests/inner_garment_v1"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.image_dir.is_dir():
        ap.error(f"Image directory not found: {args.image_dir}")
    run_visualization(args.image_dir, args.num_images, args.output_dir,
                      args.device, args.seed)


if __name__ == "__main__":
    main()
