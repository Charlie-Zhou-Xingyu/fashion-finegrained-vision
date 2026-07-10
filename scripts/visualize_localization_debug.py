"""
Debug visualization for open-vocabulary region localization (3.1.2).

Runs Grounding DINO directly on an image + query, applies shape priors, and
generates a 6-panel debug image showing each stage of the pipeline.

Panels:
  1. Original image with garment bbox (green) overlaid
  2. Bbox crop sent to DINO (raw, no mask)
  3. Mask-gated crop (what DINO actually receives when a mask is supplied)
  4. All candidates after NMS, before shape filter (orange boxes + score)
  5. Rejected candidates with rejection reasons (red boxes + reason text)
  6. Final result: accepted box (green) or "NOT DETECTED" banner

Usage:
    python scripts/visualize_localization_debug.py \
        --image path/to/image.jpg \
        --query "拉链" \
        --output-dir outputs/debug_viz \
        [--garment-mask path/to/mask.png] \
        [--garment-bbox "x1,y1,x2,y2"] \
        [--device cpu]

The script is self-contained: it does NOT run the full garment pipeline.
It calls GroundingDINOLocator and filter_by_shape_priors directly so it can
be used without SAM-HQ or YOLO.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fashion_vision.localization.viz_utils import (  # noqa: E402
    draw_text as _draw_text,
    draw_box as _draw_box,
    add_title_bar as _add_title,
    resize_to_height as _resize_to_height,
    pad_width as _pad_width,
    panels_to_html as _panels_to_html_multi,
    PANEL_PAD as _PANEL_PAD,
)


def _crop_region(image: np.ndarray, bbox: list[int], pad_px: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w, x2 + pad_px)
    y2 = min(h, y2 + pad_px)
    if x2 <= x1 or y2 <= y1:
        return image.copy(), (0, 0)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def _apply_mask_to_crop(crop: np.ndarray, mask_crop: np.ndarray, dilation_px: int = 0) -> np.ndarray:
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    return GroundingDINOLocator.mask_to_garment(crop, mask_crop, fill_mode="grey", dilation_px=dilation_px)


def _panels_to_html(panels: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    """Save 6 panels as a self-contained HTML grid (single-section wrapper)."""
    _panels_to_html_multi([("", panels)], out_path)
    logger.info("HTML saved: %s", out_path)


def build_panels(
    image: np.ndarray,
    query: str,
    garment_mask: Optional[np.ndarray],
    garment_bbox: Optional[list[int]],
    all_candidates: list[dict],
    kept: list[dict],
    dilation_px: int,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """
    Assemble the 6-panel debug image.

    Args:
        image: Full BGR image.
        query: Original text query (Chinese or English).
        garment_mask: Binary H×W garment mask, or None.
        garment_bbox: [x1,y1,x2,y2] garment instance bbox, or None.
        all_candidates: All detections after NMS (each has bbox_xyxy, score, prompt).
            After filter_by_shape_priors, rejected ones have _shape_prior_status=="rejected".
        kept: Detections that passed shape priors.
        dilation_px: Mask dilation used (for display).

    Returns:
        (montage, panel_list) where panel_list is [(title, img), ...] for HTML output.
    """
    target_h = 480

    # ── Panel 1: original image with garment bbox ─────────────────────────────
    p1 = image.copy()
    if garment_bbox is not None:
        _draw_box(p1, garment_bbox, (0, 220, 0), "garment")
    p1 = _add_title(p1, f"1. Original  query={query!r}")

    # ── Panel 2: raw bbox crop (no mask) ─────────────────────────────────────
    if garment_bbox is not None:
        raw_crop, crop_off = _crop_region(image, garment_bbox)
    else:
        raw_crop = image.copy()
        crop_off = (0, 0)
    p2 = _add_title(raw_crop.copy(), "2. Raw crop (no mask)")

    # ── Panel 3: mask-gated crop ──────────────────────────────────────────────
    if garment_mask is not None:
        h_img, w_img = image.shape[:2]
        m = garment_mask
        if m.shape[:2] != (h_img, w_img):
            m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        if garment_bbox is not None:
            x1b, y1b, x2b, y2b = garment_bbox
            x1b = max(0, x1b - 8); y1b = max(0, y1b - 8)
            x2b = min(w_img, x2b + 8); y2b = min(h_img, y2b + 8)
            mask_crop = m[y1b:y2b, x1b:x2b]
        else:
            mask_crop = m
        p3_img = _apply_mask_to_crop(raw_crop, mask_crop, dilation_px)
        dil_label = f"dil={dilation_px}px" if dilation_px > 0 else "no dilation"
        p3 = _add_title(p3_img, f"3. Mask-gated ({dil_label})")
    else:
        p3_blank = np.full_like(raw_crop, 80)
        _draw_text(p3_blank, "no mask", (10, raw_crop.shape[0] // 2))
        p3 = _add_title(p3_blank, "3. Mask-gated (no mask)")

    # ── Panel 4: all candidates before shape filter ───────────────────────────
    p4 = raw_crop.copy()
    for d in all_candidates:
        bx = list(d["bbox_xyxy"])
        # Convert full-image coords to crop coords
        if garment_bbox is not None:
            bx = [bx[0] - crop_off[0], bx[1] - crop_off[1],
                  bx[2] - crop_off[0], bx[3] - crop_off[1]]
        label = f"{d['score']:.2f}"
        _draw_box(p4, bx, (0, 165, 255), label)   # orange
    _draw_text(p4, f"n={len(all_candidates)}", (4, p4.shape[0] - 6), (0, 165, 255))
    p4 = _add_title(p4, "4. All candidates (pre-shape-filter)")

    # ── Panel 5: rejected candidates with reasons ─────────────────────────────
    p5 = raw_crop.copy()
    rejected = [d for d in all_candidates if d.get("_shape_prior_status") == "rejected"]
    for d in rejected:
        bx = list(d["bbox_xyxy"])
        if garment_bbox is not None:
            bx = [bx[0] - crop_off[0], bx[1] - crop_off[1],
                  bx[2] - crop_off[0], bx[3] - crop_off[1]]
        reasons = d.get("_shape_prior_reasons", [])
        short = reasons[0][:24] if reasons else "?"
        _draw_box(p5, bx, (0, 0, 220), f"REJECT: {short}")   # red
    _draw_text(p5, f"n={len(rejected)}", (4, p5.shape[0] - 6), (0, 0, 220))
    p5 = _add_title(p5, "5. Rejected candidates")

    # ── Panel 6: final result ─────────────────────────────────────────────────
    p6 = raw_crop.copy()
    if kept:
        top = kept[0]
        bx = list(top["bbox_xyxy"])
        if garment_bbox is not None:
            bx = [bx[0] - crop_off[0], bx[1] - crop_off[1],
                  bx[2] - crop_off[0], bx[3] - crop_off[1]]
        _draw_box(p6, bx, (0, 220, 0), f"score={top['score']:.2f}")
    else:
        overlay = p6.copy()
        cv2.rectangle(overlay, (0, 0), (p6.shape[1], p6.shape[0]), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.45, p6, 0.55, 0, p6)
        th, tw = p6.shape[0], p6.shape[1]
        _draw_text(p6, "NOT DETECTED", (max(2, tw // 2 - 60), th // 2), (0, 0, 255))
    p6 = _add_title(p6, f"6. Final  n_kept={len(kept)}")

    # ── Assemble: scale all to same height, concatenate ──────────────────────
    panel_imgs = [p1, p2, p3, p4, p5, p6]
    panel_titles = [
        f"1. Original  query={query!r}",
        "2. Raw crop (no mask)",
        f"3. Mask-gated (dil={dilation_px}px)" if dilation_px else "3. Mask-gated (no mask)",
        "4. All candidates (pre-shape-filter)",
        "5. Rejected candidates",
        f"6. Final  n_kept={len(kept)}",
    ]
    panel_list = list(zip(panel_titles, panel_imgs))

    scaled = [_resize_to_height(p, target_h) for p in panel_imgs]
    row1 = np.hstack([
        scaled[0],
        np.full((target_h, _PANEL_PAD, 3), 30, dtype=np.uint8),
        scaled[1],
        np.full((target_h, _PANEL_PAD, 3), 30, dtype=np.uint8),
        scaled[2],
    ])
    row2 = np.hstack([
        scaled[3],
        np.full((target_h, _PANEL_PAD, 3), 30, dtype=np.uint8),
        scaled[4],
        np.full((target_h, _PANEL_PAD, 3), 30, dtype=np.uint8),
        scaled[5],
    ])
    max_w = max(row1.shape[1], row2.shape[1])
    row1 = _pad_width(row1, max_w, fill=30)
    row2 = _pad_width(row2, max_w, fill=30)
    divider = np.full((_PANEL_PAD, max_w, 3), 30, dtype=np.uint8)
    return np.vstack([row1, divider, row2]), panel_list


def run(
    image_path: str,
    query: str,
    output_dir: str,
    garment_mask_path: Optional[str],
    garment_bbox: Optional[list[int]],
    device: str,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Cannot read image: %s", image_path)
        sys.exit(1)

    garment_mask: Optional[np.ndarray] = None
    if garment_mask_path:
        m = cv2.imread(garment_mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            logger.warning("Cannot read mask: %s — running without mask", garment_mask_path)
        else:
            garment_mask = (m > 0).astype(np.uint8)

    # Parse query to get part + prompts + thresholds
    from fashion_vision.localization.intent_parser import parse_intent
    from fashion_vision.localization.part_detection_config import (
        get_part_prompts, get_part_shape_config, get_part_thresholds,
        DEFAULT_BOX_THRESHOLD,
    )

    intent = parse_intent(query)
    part = intent.part or ""

    if part:
        prompts = get_part_prompts(part, fallback_prompt=intent.grounding_text or query)
        box_t, _ = get_part_thresholds(part)
        dilation_px: int = get_part_shape_config(part).get("mask_dilation_px", 0)
    else:
        # Zero-shot: use raw query as single prompt
        prompts = [query]
        box_t = DEFAULT_BOX_THRESHOLD
        dilation_px = 0

    logger.info("Query %r → part=%r, prompts=%s, box_t=%.2f, dilation=%dpx",
                query, part, prompts, box_t, dilation_px)

    # Crop to garment bbox if provided
    if garment_bbox is not None:
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = garment_bbox
        x1c = max(0, x1 - 8); y1c = max(0, y1 - 8)
        x2c = min(w_img, x2 + 8); y2c = min(h_img, y2 + 8)
        crop = image[y1c:y2c, x1c:x2c]
        crop_off = (x1c, y1c)
        if garment_mask is not None:
            m = garment_mask
            if m.shape[:2] != (h_img, w_img):
                m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            crop_mask: Optional[np.ndarray] = m[y1c:y2c, x1c:x2c]
        else:
            crop_mask = None
    else:
        crop = image
        crop_off = (0, 0)
        crop_mask = garment_mask

    # Run DINO
    logger.info("Loading GroundingDINOLocator (device=%s) …", device)
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    locator = GroundingDINOLocator(device=device)

    detections, n_raw = locator.detect_multi_prompt(
        crop, prompts,
        garment_mask=crop_mask,
        threshold=box_t,
        return_raw_count=True,
        dilation_px=dilation_px,
    )
    logger.info("DINO: %d raw, %d after NMS", n_raw, len(detections))

    # Remap to full-image coords
    ox, oy = crop_off
    if ox or oy:
        for d in detections:
            b = d["bbox_xyxy"]
            d["bbox_xyxy"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]

    # Shape priors — fall back to full image dims so area_ratio checks are never skipped
    from fashion_vision.localization.part_shape_priors import filter_by_shape_priors
    h_img, w_img = image.shape[:2]
    prior_bbox = garment_bbox if garment_bbox is not None else [0, 0, w_img, h_img]
    all_candidates = list(detections)   # keep refs; filter mutates dicts in-place
    kept = filter_by_shape_priors(detections, part or None, prior_bbox)
    logger.info("Shape filter: %d → %d kept", len(all_candidates), len(kept))

    # Build and save visualization
    montage, panel_list = build_panels(
        image, query, garment_mask, garment_bbox, all_candidates, kept, dilation_px
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    safe_q = query.replace("/", "_").replace("\\", "_")[:20]
    jpg_path = out_dir / f"{stem}_{safe_q}_debug.jpg"
    html_path = out_dir / f"{stem}_{safe_q}_debug.html"
    cv2.imwrite(str(jpg_path), montage)
    _panels_to_html(panel_list, html_path)
    logger.info("Saved JPG: %s", jpg_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--query", required=True, help="Region query (Chinese or English)")
    p.add_argument("--output-dir", default="outputs/debug_viz", help="Output directory")
    p.add_argument("--garment-mask", default=None, help="Binary garment mask PNG path")
    p.add_argument("--garment-bbox", default=None, help="Garment bbox as x1,y1,x2,y2")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    bbox: Optional[list[int]] = None
    if args.garment_bbox:
        try:
            bbox = [int(v.strip()) for v in args.garment_bbox.split(",")]
            if len(bbox) != 4:
                raise ValueError
        except ValueError:
            logger.error("--garment-bbox must be four integers: x1,y1,x2,y2")
            sys.exit(1)

    run(
        image_path=args.image,
        query=args.query,
        output_dir=args.output_dir,
        garment_mask_path=args.garment_mask,
        garment_bbox=bbox,
        device=args.device,
    )
