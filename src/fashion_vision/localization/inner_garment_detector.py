"""
SAM-based inner-garment detection under outerwear (3.1.2).

Two-stage strategy:
  1. PRIMARY: Geometric complement analysis on the outerwear neckline region.
     The inner garment is typically OUTSIDE the outerwear mask but INSIDE the
     outerwear bbox, in the collar / chest-opening area.  Connected-component,
     Canny-edge, and SAM-multimask candidates are scored by how well they fit
     the "complement of outerwear in neckline" pattern.
  2. FALLBACK: SAM multimask on the full outerwear bbox (legacy, kept for
     cases where the neckline branch finds nothing).

Integration:
    Called on-demand from region_localization_router.locate_region() when the
    user intent targets an inner layer and the selected instance is outerwear.
    NOT run globally on every image.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# ── Neckline-rule scoring weights and thresholds ────────────────────────────

NECKLINE_RULES: dict[str, float] = {
    # ROI definition (fractions of outerwear bbox)
    "roi_x_lo": 0.18,
    "roi_x_hi": 0.82,
    "roi_y_lo": 0.03,
    "roi_y_hi": 0.58,

    # Opening core region (fractions of outerwear bbox)
    "opening_core_x_lo": 0.25,
    "opening_core_x_hi": 0.70,
    "opening_core_y_lo": 0.05,
    "opening_core_y_hi": 0.65,

    # Thresholds for candidate filtering
    "min_score": 3.0,
    "min_inside_bbox_ratio": 0.75,
    "min_outside_outer_ratio": 0.45,
    "min_neckline_overlap": 0.60,
    "min_area_ratio_bbox": 0.006,
    "max_area_ratio_bbox": 0.25,
    "min_rel_cx": 0.30,
    "max_rel_cx": 0.70,
    "min_opening_core_overlap": 0.35,
    "min_bbox_w_ratio": 0.08,
    "min_bbox_h_ratio": 0.10,

    # SAM refine acceptance
    "refine_min_iou": 0.60,
    "refine_min_area_ratio": 0.50,

    # Scoring weights
    "w_inside_bbox": 1.0,
    "w_outside_outer": 1.5,
    "w_neckline_overlap": 1.5,
    "w_center_score": 1.8,
    "w_opening_core": 2.0,
    "w_upper_position": 1.0,
    "w_solidity": 1.0,
    "w_area_ratio": 0.5,
    "penalty_side_edge": 2.0,

    # Torso constraint
    "w_torso_overlap": 2.0,
    "min_torso_overlap": 0.35,
}

# ── Opening extension rules ──────────────────────────────────────────────────

OPENING_EXTENSION_RULES: dict[str, float] = {
    "roi_x_lo": 0.22,
    "roi_x_hi": 0.78,
    "roi_y_lo": 0.08,
    "roi_y_hi": 0.85,
    "min_x_overlap_ratio": 0.20,
    "max_vertical_gap_ratio": 0.10,
    "min_area_ratio_bbox": 0.002,
    "min_rel_cx": 0.22,
    "max_rel_cx": 0.78,
}

# ── Fallback rules (legacy: SAM full-bbox multimask) ────────────────────────

FALLBACK_RULES: dict[str, float] = {
    "min_containment_ratio": 0.80,
    "max_area_ratio": 0.50,
    "min_area_ratio": 0.01,
    "min_solidity": 0.65,
    "edge_margin_px": 8,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_2d_mask(mask: np.ndarray) -> np.ndarray:
    """Force a mask array to 2-D (H, W)."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[0] == 1:
            mask = mask[0, :, :]
        else:
            mask = mask[:, :, 0]
    return mask


def _extract_bbox(instance: dict[str, Any]) -> Optional[list[int]]:
    for key in ("bbox_xyxy", "bbox", "det_bbox", "detection_bbox"):
        val = instance.get(key)
        if val is not None:
            try:
                return [int(round(float(v))) for v in val]
            except (TypeError, ValueError):
                pass
    return None


def _load_mask(instance: dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("_mask", "mask_array", "mask"):
        val = instance.get(key)
        if isinstance(val, np.ndarray):
            return ensure_2d_mask(val)
    for key in ("pred_mask_path", "gt_mask_path", "mask_path"):
        path = instance.get(key)
        if path:
            try:
                m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    return ensure_2d_mask(m)
            except Exception:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Neckline-rule candidate generation
# ═══════════════════════════════════════════════════════════════════════════════

def _construct_neckline_roi(
    outer_bbox: list[int], h_img: int, w_img: int
) -> list[int]:
    """Return [nx1, ny1, nx2, ny2] in full-image coords."""
    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    nx1 = max(0, int(gx1 + gw * NECKLINE_RULES["roi_x_lo"]))
    ny1 = max(0, int(gy1 + gh * NECKLINE_RULES["roi_y_lo"]))
    nx2 = min(w_img, int(gx1 + gw * NECKLINE_RULES["roi_x_hi"]))
    ny2 = min(h_img, int(gy1 + gh * NECKLINE_RULES["roi_y_hi"]))
    return [nx1, ny1, nx2, ny2]


def _build_complement_search_mask(
    outer_bbox: list[int],
    outer_mask_bin: np.ndarray,
    neckline_box: list[int],
    h_img: int, w_img: int,
) -> np.ndarray:
    """Return binary mask: inside outer_bbox ∩ neckline_box ∩ ¬outer_mask."""
    gx1, gy1, gx2, gy2 = outer_bbox
    nx1, ny1, nx2, ny2 = neckline_box

    # outer_bbox region
    bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    bbox_mask[gy1:gy2, gx1:gx2] = 255

    # neckline ROI
    neckline_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    neckline_mask[ny1:ny2, nx1:nx2] = 255

    # complement = inside bbox AND inside neckline AND NOT in outer_mask
    # Normalise outer_mask_bin to 0/255 for correct bitwise_not behaviour
    outer_bin_255 = (outer_mask_bin > 0).astype(np.uint8) * 255
    inside = cv2.bitwise_and(bbox_mask, neckline_mask)
    complement = cv2.bitwise_and(inside, cv2.bitwise_not(outer_bin_255))
    # Return as 0/1 mask
    return (complement > 0).astype(np.uint8)


def _build_opening_core_region(
    outer_bbox: list[int], h_img: int, w_img: int
) -> np.ndarray:
    """Return binary mask of the centre-front opening core region."""
    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    w = NECKLINE_RULES
    ox1 = max(0, int(gx1 + gw * w["opening_core_x_lo"]))
    oy1 = max(0, int(gy1 + gh * w["opening_core_y_lo"]))
    ox2 = min(w_img, int(gx1 + gw * w["opening_core_x_hi"]))
    oy2 = min(h_img, int(gy1 + gh * w["opening_core_y_hi"]))
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if ox2 > ox1 and oy2 > oy1:
        mask[oy1:oy2, ox1:ox2] = 255
    return mask


def _construct_opening_roi(
    outer_bbox: list[int], h_img: int, w_img: int
) -> list[int]:
    """Return [ox1, oy1, ox2, oy2] — the full front-opening region."""
    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    w = OPENING_EXTENSION_RULES
    ox1 = max(0, int(gx1 + gw * w["roi_x_lo"]))
    oy1 = max(0, int(gy1 + gh * w["roi_y_lo"]))
    ox2 = min(w_img, int(gx1 + gw * w["roi_x_hi"]))
    oy2 = min(h_img, int(gy1 + gh * w["roi_y_hi"]))
    return [ox1, oy1, ox2, oy2]


def _extend_inner_mask_downward(
    seed_mask: np.ndarray,
    seed_bbox: list[int],
    outer_bbox: list[int],
    outer_mask_bin: np.ndarray,
    h_img: int, w_img: int,
    torso_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Extend seed inner-garment mask downward through the front-opening region.

    Returns (extended_mask, debug_dict).
    """
    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    bbox_area = max(1, gw * gh)
    w = OPENING_EXTENSION_RULES

    # 1. Opening ROI + complement
    oroix1, oroiy1, oroix2, oroiy2 = _construct_opening_roi(outer_bbox, h_img, w_img)
    if oroix2 <= oroix1 or oroiy2 <= oroiy1:
        return seed_mask, {"extended": False, "reason": "invalid_opening_roi"}

    outer_bin_255 = (outer_mask_bin > 0).astype(np.uint8) * 255
    bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    bbox_mask[gy1:gy2, gx1:gx2] = 255
    opening_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    opening_mask[oroiy1:oroiy2, oroix1:oroix2] = 255
    opening_comp = cv2.bitwise_and(bbox_mask, opening_mask)
    opening_comp = cv2.bitwise_and(opening_comp, cv2.bitwise_not(outer_bin_255))
    opening_comp = (opening_comp > 0).astype(np.uint8)

    # Intersect with torso prior if available
    if torso_mask is not None and torso_mask.sum() > 0:
        torso_bin = (torso_mask > 0).astype(np.uint8)
        opening_comp = cv2.bitwise_and(opening_comp, torso_bin)

    if opening_comp.sum() < 30:
        return seed_mask, {"extended": False, "reason": "empty_opening_complement"}

    # 2. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(opening_comp, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 3. Extract connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=4)

    # 4. Filter components to merge
    sx1, sy1, sx2, sy2 = seed_bbox
    seed_w = sx2 - sx1
    matched_labels: list[int] = []
    debug_components: list[dict] = []
    extended = seed_mask.copy()

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        cx = float(stats[label_id, cv2.CC_STAT_LEFT])
        cy = float(stats[label_id, cv2.CC_STAT_TOP])
        cw = float(stats[label_id, cv2.CC_STAT_WIDTH])
        ch = float(stats[label_id, cv2.CC_STAT_HEIGHT])
        comp_cx = cx + cw / 2.0
        comp_cy = cy + ch / 2.0

        rel_cx_comp = (comp_cx - gx1) / max(1.0, gw)
        area_ratio_comp = area / max(1, bbox_area)
        x_overlap = max(0, min(sx2, int(cx + cw)) - max(sx1, int(cx)))
        x_overlap_ratio = x_overlap / max(1.0, seed_w)
        vertical_gap = (cy - sy2) / max(1.0, gh)  # positive = below seed

        comp_mask = (labels == label_id).astype(np.uint8) * 255

        passed = (
            x_overlap_ratio >= w["min_x_overlap_ratio"]
            and vertical_gap <= w["max_vertical_gap_ratio"]
            and w["min_rel_cx"] <= rel_cx_comp <= w["max_rel_cx"]
            and area_ratio_comp >= w["min_area_ratio_bbox"]
        )

        debug_components.append({
            "label_id": int(label_id), "area": area,
            "bbox": [int(cx), int(cy), int(cx + cw), int(cy + ch)],
            "rel_cx": round(rel_cx_comp, 3),
            "x_overlap_ratio": round(x_overlap_ratio, 3),
            "vertical_gap": round(vertical_gap, 3),
            "area_ratio": round(area_ratio_comp, 5),
            "passed": passed,
        })

        if passed:
            matched_labels.append(label_id)
            extended = cv2.bitwise_or(extended, comp_mask)

    return extended, {
        "extended": len(matched_labels) > 0,
        "opening_box": [oroix1, oroiy1, oroix2, oroiy2],
        "num_opening_components": num_labels - 1,
        "num_matched": len(matched_labels),
        "matched_labels": matched_labels,
        "all_components": debug_components,
    }


def _mask_to_bbox(mask: np.ndarray) -> list[int]:
    """Return [x1, y1, x2, y2] bounding box of non-zero pixels in mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _extract_cc_candidates(
    search_mask: np.ndarray,
) -> list[dict]:
    """Extract candidates via connected-components analysis on search_mask."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        search_mask, connectivity=4)
    candidates: list[dict] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 20:  # pixel-level noise
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[label_id, 0])
        cy = float(centroids[label_id, 1])
        candidates.append({
            "mask": (labels == label_id).astype(np.uint8) * 255,
            "bbox_xyxy": [x, y, x + w, y + h],
            "area": area,
            "centroid": (cx, cy),
            "source": "connected_component",
        })
    return candidates


def _extract_canny_candidates(
    image_bgr: np.ndarray,
    neckline_box: list[int],
) -> list[dict]:
    """Extract closed-contour candidates via Canny edges within neckline ROI."""
    nx1, ny1, nx2, ny2 = neckline_box
    roi = image_bgr[ny1:ny2, nx1:nx2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    # Morphological close to bridge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Map back to full-image coords
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        candidate = {
            "mask": mask,  # ROI-local
            "bbox_xyxy": [nx1 + x, ny1 + y, nx1 + x + w, ny1 + y + h],
            "area": area,
            "centroid": (nx1 + x + w / 2, ny1 + y + h / 2),
            "source": "canny_contour",
            "_roi_offset": (nx1, ny1),
        }
        candidates.append(candidate)
    return candidates


def _extract_sam_candidates(
    image_bgr: np.ndarray,
    neckline_box: list[int],
    sam_wrapper,
) -> list[dict]:
    """Run SAM multimask on the neckline ROI, return candidates."""
    if sam_wrapper is None:
        return []
    nx1, ny1, nx2, ny2 = neckline_box
    roi = image_bgr[ny1:ny2, nx1:nx2]
    if roi.size == 0:
        return []
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    try:
        sam_cands = sam_wrapper.predict_all_masks(roi_rgb, [0, 0, nx2 - nx1, ny2 - ny1])
    except Exception as exc:
        logger.warning("inner_garment_detector: SAM neckline multimask failed: %s", exc)
        return []

    candidates: list[dict] = []
    for i, sc in enumerate(sam_cands):
        m = ensure_2d_mask((sc["mask"] > 0).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append({
            "mask": m,
            "bbox_xyxy": [nx1 + x, ny1 + y, nx1 + x + w, ny1 + y + h],
            "area": cv2.contourArea(cnt),
            "centroid": (nx1 + x + w / 2, ny1 + y + h / 2),
            "source": f"sam_neckline_multimask_{i}",
            "sam_score": float(sc["score"]),
        })
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Unified scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _score_candidate(
    candidate: dict,
    outer_bbox: list[int],
    outer_mask_bin: np.ndarray,
    neckline_box: list[int],
    opening_core_mask: np.ndarray,
    h_img: int, w_img: int,
    torso_mask: Optional[np.ndarray] = None,
    torso_min_overlap: Optional[float] = None,
) -> dict:
    """Score a single candidate.  Returns dict with scores + reject_reasons."""
    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    nx1, ny1, nx2, ny2 = neckline_box

    mask = ensure_2d_mask(candidate["mask"])
    # If mask is ROI-local (Canny / SAM), pad to full-image for scoring
    offset = candidate.get("_roi_offset")
    if offset is not None:
        ox, oy = offset
        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        mh, mw = mask.shape
        full_mask[oy:oy + mh, ox:ox + mw] = mask
    elif mask.shape[:2] != (h_img, w_img):
        # candidate mask in neckline-local coords — pad
        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        full_mask[ny1:ny2, nx1:nx2] = mask
    else:
        full_mask = mask

    cand_area = int((full_mask > 0).sum())
    bbox_area = max(1, gw * gh)
    outer_area = int((outer_mask_bin > 0).sum())
    bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    bbox_mask[gy1:gy2, gx1:gx2] = 255
    neckline_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    neckline_mask[ny1:ny2, nx1:nx2] = 255

    if cand_area == 0:
        return {"score": 0.0, "reject_reasons": ["zero_area"]}

    # ── Metrics (use pixel counts, not 0/255 value sums) ──────────────────
    outer_bin_255 = (outer_mask_bin > 0).astype(np.uint8) * 255
    inside_bbox = int((cv2.bitwise_and(full_mask, bbox_mask) > 0).sum())
    inside_bbox_ratio = inside_bbox / max(1, cand_area)

    outside_outer = int((cv2.bitwise_and(full_mask, cv2.bitwise_not(outer_bin_255)) > 0).sum())
    outside_outer_ratio = outside_outer / max(1, cand_area)

    neckline_overlap_px = int((cv2.bitwise_and(full_mask, neckline_mask) > 0).sum())
    neckline_overlap = neckline_overlap_px / max(1, cand_area)

    # centroid
    ys, xs = np.where(full_mask > 0)
    cy = float(np.mean(ys)) if len(ys) > 0 else 0.0
    cx = float(np.mean(xs)) if len(xs) > 0 else 0.0
    center_x = gx1 + gw / 2.0
    center_score = 1.0 - min(1.0, abs(cx - center_x) / max(1.0, gw * 0.5))
    upper_position_score = 1.0 - min(1.0, max(0.0, (cy - gy1) / max(1.0, gh)))

    area_ratio_bbox = cand_area / max(1, bbox_area)
    area_ratio_outer = cand_area / max(1, outer_area)

    # solidity
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area_cv = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = cnt_area_cv / max(1.0, hull_area)

    # opening core overlap
    opening_core_overlap_px = int((cv2.bitwise_and(full_mask, opening_core_mask) > 0).sum())
    opening_core_overlap = opening_core_overlap_px / max(1, cand_area)

    # side-edge penalty
    x1c, y1c, x2c, y2c = candidate["bbox_xyxy"]
    touches_left = x1c <= gx1 + gw * 0.08
    touches_right = x2c >= gx2 - gw * 0.08
    side_edge_penalty = 1.0 if (touches_left or touches_right) else 0.0

    # Derived position metrics
    rel_cx = (cx - gx1) / max(1.0, gw)
    rel_cy = (cy - gy1) / max(1.0, gh)
    bw = x2c - x1c
    bh = y2c - y1c
    bbox_w_ratio = bw / max(1.0, gw)
    bbox_h_ratio = bh / max(1.0, gh)

    # torso overlap (optional)
    torso_overlap = 0.0
    if torso_mask is not None:
        torso_bin = (torso_mask > 0).astype(np.uint8) * 255
        torso_overlap_px = int((cv2.bitwise_and(full_mask, torso_bin) > 0).sum())
        torso_overlap = torso_overlap_px / max(1, cand_area)

    # ── Weighted score ───────────────────────────────────────────────────
    w = NECKLINE_RULES
    score = (
        w["w_inside_bbox"] * inside_bbox_ratio
        + w["w_outside_outer"] * outside_outer_ratio
        + w["w_neckline_overlap"] * neckline_overlap
        + w["w_center_score"] * center_score
        + w["w_opening_core"] * opening_core_overlap
        + w["w_upper_position"] * upper_position_score
        + w["w_solidity"] * solidity
        + w["w_area_ratio"] * (1.0 - abs(area_ratio_bbox - 0.08))
        - w["penalty_side_edge"] * side_edge_penalty
        + w["w_torso_overlap"] * torso_overlap
    )

    # ── Rejection reasons ────────────────────────────────────────────────
    reasons = []
    if inside_bbox_ratio < w["min_inside_bbox_ratio"]:
        reasons.append(f"inside_bbox={inside_bbox_ratio:.3f}<{w['min_inside_bbox_ratio']}")
    if outside_outer_ratio < w["min_outside_outer_ratio"]:
        reasons.append(f"outside_outer={outside_outer_ratio:.3f}<{w['min_outside_outer_ratio']}")
    if neckline_overlap < w["min_neckline_overlap"]:
        reasons.append(f"neckline_overlap={neckline_overlap:.3f}<{w['min_neckline_overlap']}")
    if area_ratio_bbox < w["min_area_ratio_bbox"]:
        reasons.append(f"area_bbox={area_ratio_bbox:.5f}<{w['min_area_ratio_bbox']}")
    if area_ratio_bbox > w["max_area_ratio_bbox"]:
        reasons.append(f"area_bbox={area_ratio_bbox:.4f}>{w['max_area_ratio_bbox']}")
    if rel_cx < w["min_rel_cx"] or rel_cx > w["max_rel_cx"]:
        reasons.append(f"off_center rel_cx={rel_cx:.3f} not in [{w['min_rel_cx']},{w['max_rel_cx']}]")
    if opening_core_overlap < w["min_opening_core_overlap"]:
        reasons.append(f"low_opening_core={opening_core_overlap:.3f}<{w['min_opening_core_overlap']}")
    if bbox_w_ratio < w["min_bbox_w_ratio"]:
        reasons.append(f"bbox_w={bbox_w_ratio:.3f}<{w['min_bbox_w_ratio']}")
    if bbox_h_ratio < w["min_bbox_h_ratio"]:
        reasons.append(f"bbox_h={bbox_h_ratio:.3f}<{w['min_bbox_h_ratio']}")
    _torso_min = torso_min_overlap if torso_min_overlap is not None else w["min_torso_overlap"]
    if torso_mask is not None and torso_overlap < _torso_min:
        reasons.append(f"low_torso_overlap={torso_overlap:.3f}<{_torso_min}")

    passed = score >= w["min_score"] and len(reasons) == 0

    return {
        "score": round(score, 3),
        "passed": passed,
        "reject_reasons": reasons,
        "inside_bbox_ratio": round(inside_bbox_ratio, 3),
        "outside_outer_ratio": round(outside_outer_ratio, 3),
        "neckline_overlap": round(neckline_overlap, 3),
        "opening_core_overlap": round(opening_core_overlap, 3),
        "center_score": round(center_score, 3),
        "upper_position_score": round(upper_position_score, 3),
        "area_ratio_bbox": round(area_ratio_bbox, 4),
        "area_ratio_outer": round(area_ratio_outer, 4),
        "solidity": round(solidity, 3),
        "side_edge_penalty": round(side_edge_penalty, 1),
        "rel_cx": round(rel_cx, 3),
        "rel_cy": round(rel_cy, 3),
        "bbox_w_ratio": round(bbox_w_ratio, 3),
        "bbox_h_ratio": round(bbox_h_ratio, 3),
        "cand_area": cand_area,
        "torso_overlap": round(torso_overlap, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SAM refine safety check
# ═══════════════════════════════════════════════════════════════════════════════

def _check_sam_refine_safety(
    refined_mask: np.ndarray,
    original_mask: np.ndarray,
    original_bbox: list[int],
    outer_bbox: list[int],
    h_img: int, w_img: int,
) -> dict:
    """Validate that a SAM-refined mask is consistent with the original candidate.

    Checks:
        1. IoU(refined, original) >= 0.60
        2. refined_area >= 0.5 * original_area
        3. refined bbox centre shift <= 0.15 * outer_bbox_width

    Args:
        refined_mask: SAM-refined binary mask H×W.
        original_mask: Original candidate binary mask H×W.
        original_bbox: ``[x1, y1, x2, y2]`` of the original candidate.
        outer_bbox: ``[x1, y1, x2, y2]`` of the outerwear instance.
        h_img, w_img: Full image dimensions.

    Returns:
        Debug dict with ``accept``, ``refined_iou``, ``refined_area_ratio``,
        ``center_shift``, and ``reason`` (if rejected).
    """
    gw = max(1, outer_bbox[2] - outer_bbox[0])

    refined_bin = (refined_mask > 0).astype(np.uint8)
    original_bin = (original_mask > 0).astype(np.uint8)

    inter = int((cv2.bitwise_and(refined_bin, original_bin) > 0).sum())
    union = int((cv2.bitwise_or(refined_bin, original_bin) > 0).sum())
    refined_iou = inter / max(1, union)

    refined_area = int(refined_bin.sum())
    original_area = max(1, int(original_bin.sum()))
    refined_area_ratio = refined_area / original_area

    # centre shift
    oys, oxs = np.where(original_bin > 0)
    rys, rxs = np.where(refined_bin > 0)
    ocx = float(np.mean(oxs)) if len(oxs) > 0 else 0.0
    ocy = float(np.mean(oys)) if len(oys) > 0 else 0.0
    rcx = float(np.mean(rxs)) if len(rxs) > 0 else 0.0
    rcy = float(np.mean(rys)) if len(rys) > 0 else 0.0
    center_shift = np.sqrt((rcx - ocx) ** 2 + (rcy - ocy) ** 2) / gw

    accept = (
        refined_iou >= NECKLINE_RULES["refine_min_iou"]
        and refined_area_ratio >= NECKLINE_RULES["refine_min_area_ratio"]
        and center_shift <= 0.15
    )

    reason = None if accept else (
        f"iou={refined_iou:.3f}<{NECKLINE_RULES['refine_min_iou']}" if refined_iou < NECKLINE_RULES["refine_min_iou"]
        else f"area_ratio={refined_area_ratio:.3f}<{NECKLINE_RULES['refine_min_area_ratio']}" if refined_area_ratio < NECKLINE_RULES["refine_min_area_ratio"]
        else f"center_shift={center_shift:.3f}>0.15"
    )

    return {
        "accept": accept,
        "refined_iou": round(refined_iou, 4),
        "refined_area_ratio": round(refined_area_ratio, 4),
        "center_shift": round(center_shift, 4),
        "reason": reason,
    }


def _check_boundary_refine_safety(
    refined_mask: np.ndarray,
    original_mask: np.ndarray,
    refined_bbox: list[int],
    original_bbox: list[int],
    outer_bbox: list[int],
    torso_mask: Optional[np.ndarray] = None,
) -> dict:
    """Validate that a boundary-refined result is safe to adopt.

    Checks:
        1. area_ratio = refined_area / original_area in [0.45, 2.80]
        2. bbox_area_ratio = refined_bbox_area / original_bbox_area in [0.45, 3.00]
        3. centre shift / outer_bbox_width <= 0.18
        4. If torso_mask: torso_overlap_after >= torso_overlap_before - 0.20

    Returns debug dict with ``accept``, metrics, and ``reason`` (if rejected).
    """
    gw = max(1, outer_bbox[2] - outer_bbox[0])

    refined_bin = (refined_mask > 0).astype(np.uint8)
    original_bin = (original_mask > 0).astype(np.uint8)

    refined_area = max(1, int(refined_bin.sum()))
    original_area = max(1, int(original_bin.sum()))
    area_ratio = refined_area / original_area

    def _bbox_area(b: list[int]) -> int:
        bw = max(1, b[2] - b[0])
        bh = max(1, b[3] - b[1])
        return bw * bh

    bbox_area_ratio = _bbox_area(refined_bbox) / max(1, _bbox_area(original_bbox))

    # centre shift
    oys, oxs = np.where(original_bin > 0)
    rys, rxs = np.where(refined_bin > 0)
    ocx = float(np.mean(oxs)) if len(oxs) > 0 else 0.0
    ocy = float(np.mean(oys)) if len(oys) > 0 else 0.0
    rcx = float(np.mean(rxs)) if len(rxs) > 0 else 0.0
    rcy = float(np.mean(rys)) if len(rys) > 0 else 0.0
    center_shift_ratio = np.sqrt((rcx - ocx) ** 2 + (rcy - ocy) ** 2) / gw

    # torso overlap
    torso_before = 0.0
    torso_after = 0.0
    if torso_mask is not None:
        torso_bin = (torso_mask > 0).astype(np.uint8) * 255
        torso_before = int((cv2.bitwise_and(original_bin, torso_bin) > 0).sum()) / original_area
        torso_after = int((cv2.bitwise_and(refined_bin, torso_bin) > 0).sum()) / refined_area

    # Acceptance checks
    checks = [
        (0.45 <= area_ratio <= 2.80,
         f"area_ratio={area_ratio:.3f} not in [0.45,2.80]"),
        (0.45 <= bbox_area_ratio <= 3.00,
         f"bbox_area_ratio={bbox_area_ratio:.3f} not in [0.45,3.00]"),
        (center_shift_ratio <= 0.18,
         f"center_shift={center_shift_ratio:.3f}>0.18"),
    ]
    if torso_mask is not None:
        checks.append(
            (torso_after >= torso_before - 0.20,
             f"torso_overlap_after={torso_after:.3f} < torso_before={torso_before:.3f} - 0.20"),
        )

    reject_reasons = [msg for ok, msg in checks if not ok]
    accept = len(reject_reasons) == 0

    return {
        "accept": accept,
        "area_ratio": round(area_ratio, 4),
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "center_shift_ratio": round(center_shift_ratio, 4),
        "torso_overlap_before": round(torso_before, 4),
        "torso_overlap_after": round(torso_after, 4),
        "reason": reject_reasons[0] if reject_reasons else None,
        "all_reject_reasons": reject_reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Primary: neckline-complement detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_inner_by_neckline_rules(
    image_bgr: np.ndarray,
    outer_bbox: list[int],
    outer_mask_bin: np.ndarray,
    sam_wrapper=None,
) -> Optional[dict[str, Any]]:
    """Find inner garment via geometric complement in the neckline region.

    Returns None if no candidate passes, or a pseudo-instance dict on success.
    """
    h_img, w_img = image_bgr.shape[:2]

    # 1. Construct neckline ROI
    neckline_box = _construct_neckline_roi(outer_bbox, h_img, w_img)
    nx1, ny1, nx2, ny2 = neckline_box
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    # 2. Opening core region
    opening_core = _build_opening_core_region(outer_bbox, h_img, w_img)

    # 2b. Torso prior — used to suppress off-torso candidates
    from fashion_vision.localization.torso_prior import build_proxy_torso_prior
    torso_mask, torso_bbox, torso_debug = build_proxy_torso_prior(outer_bbox, h_img, w_img)
    # Softer threshold for proxy torso (no real keypoints)
    torso_min_overlap = 0.25 if torso_debug.get("source") == "proxy" else 0.35

    # 3. Complement search mask
    search_mask = _build_complement_search_mask(
        outer_bbox, outer_mask_bin, neckline_box, h_img, w_img)
    if search_mask.sum() < 50:  # too few complement pixels
        return None

    # 4. Morphological cleanup of search mask
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, kernel3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel3)

    # 5. Generate candidates from three sources
    all_candidates: list[dict] = []
    all_candidates.extend(_extract_cc_candidates(cleaned))
    all_candidates.extend(_extract_canny_candidates(image_bgr, neckline_box))
    all_candidates.extend(_extract_sam_candidates(image_bgr, neckline_box, sam_wrapper))

    if not all_candidates:
        logger.info("inner_garment_detector (neckline): no candidates generated")
        return None

    # 6. Score all candidates
    debug_candidates: list[dict] = []
    best = None
    best_score = -999.0

    for cand in all_candidates:
        scoring = _score_candidate(
            cand, outer_bbox, outer_mask_bin, neckline_box, opening_core, h_img, w_img,
            torso_mask=torso_mask,
            torso_min_overlap=torso_min_overlap,
        )

        debug_candidates.append({
            "bbox": cand["bbox_xyxy"],
            "source": cand.get("source", "?"),
            "score": scoring["score"],
            "passed": scoring["passed"],
            "reject_reasons": scoring["reject_reasons"],
            "inside_bbox_ratio": scoring["inside_bbox_ratio"],
            "outside_outer_ratio": scoring["outside_outer_ratio"],
            "neckline_overlap": scoring["neckline_overlap"],
            "opening_core_overlap": scoring["opening_core_overlap"],
            "center_score": scoring["center_score"],
            "upper_position_score": scoring["upper_position_score"],
            "area_ratio_bbox": scoring["area_ratio_bbox"],
            "solidity": scoring["solidity"],
            "side_edge_penalty": scoring["side_edge_penalty"],
            "rel_cx": scoring["rel_cx"],
            "rel_cy": scoring["rel_cy"],
            "bbox_w_ratio": scoring["bbox_w_ratio"],
            "bbox_h_ratio": scoring["bbox_h_ratio"],
            "torso_overlap": scoring["torso_overlap"],
            "torso_min_overlap": torso_min_overlap,
        })

        if scoring["passed"] and scoring["score"] > best_score:
            best_score = scoring["score"]
            best = {"candidate": cand, "scoring": scoring}

    # 7. Build result
    if best is None:
        logger.info(
            "inner_garment_detector (neckline): no candidate passed. "
            "Checked %d candidates.", len(debug_candidates))
        return None

    cand = best["candidate"]
    # Build full-image seed mask
    seed_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    offset = cand.get("_roi_offset")
    cm = ensure_2d_mask(cand["mask"])
    if offset is not None:
        ox, oy = offset
        mh, mw = cm.shape
        seed_mask[oy:oy + mh, ox:ox + mw] = cm
    elif cm.shape[:2] != (h_img, w_img):
        seed_mask[ny1:ny2, nx1:nx2] = cm
    else:
        seed_mask = cm

    seed_bbox = cand["bbox_xyxy"]

    # ── Extension: grow seed downward through opening region ────────────
    extended_mask, ext_debug = _extend_inner_mask_downward(
        seed_mask, seed_bbox, outer_bbox, outer_mask_bin, h_img, w_img,
        torso_mask=torso_mask,
    )
    final_mask = extended_mask
    final_bbox = _mask_to_bbox(final_mask)
    final_method = ("neckline_outerwear_complement_extended"
                    if ext_debug.get("extended") else "neckline_outerwear_complement")

    # ── Boundary refinement ──────────────────────────────────────────
    bbox_before_refine = list(final_bbox)
    opening_roi = _construct_opening_roi(outer_bbox, h_img, w_img)
    from fashion_vision.localization.inner_boundary_refiner import refine_inner_boundary
    refined_mask, refined_bbox, refine_debug = refine_inner_boundary(
        image_bgr, final_mask, final_bbox, outer_bbox, outer_mask_bin,
        opening_roi=opening_roi,
        torso_mask=torso_mask,
    )
    refine_safety = _check_boundary_refine_safety(
        refined_mask, final_mask, refined_bbox, final_bbox, outer_bbox,
        torso_mask=torso_mask,
    )
    bbox_after_refine = refined_bbox if refine_safety["accept"] else list(final_bbox)
    refine_accepted = False
    if refine_debug.get("h_refined") or refine_debug.get("v_refined"):
        if refine_safety["accept"]:
            final_mask = refined_mask
            final_bbox = refined_bbox
            final_method += "_boundary_refined"
            refine_accepted = True
        else:
            # keep original result, log rejection
            logger.info(
                "inner_garment_detector: boundary refinement REJECTED — %s",
                refine_safety.get("reason"),
            )

    # ── Artifact cleanup ──────────────────────────────────────────────────
    from fashion_vision.localization.inner_mask_cleaner import clean_inner_mask_artifacts
    cleaned_mask, cleaned_bbox, cleanup_debug = clean_inner_mask_artifacts(
        image_bgr,
        final_mask,
        final_bbox,
        outer_bbox,
        outer_mask_bin,
        opening_roi=opening_roi,
        torso_mask=torso_mask,
        seed_mask=seed_mask,
    )
    bbox_before_cleanup = list(final_bbox)
    if cleanup_debug.get("cleanup_accepted"):
        final_mask = cleaned_mask
        final_bbox = cleaned_bbox
        final_method += "_artifact_cleaned"

    logger.info(
        "inner_garment_detector (neckline): found inner garment — "
        "bbox=%s, score=%.3f, source=%s",
        cand["bbox_xyxy"], best["scoring"]["score"], cand.get("source", "?"))

    return {
        "bbox_xyxy": final_bbox,
        "bbox_format": "xyxy",
        "mask": final_mask,
        "coarse_class_name": "inner_garment",
        "coarse_class_id": -1,
        "fine_class_name": "inner_garment",
        "fine_class_id": -1,
        "source": final_method,
        "score": round(best["scoring"]["score"], 4),
        "debug": {
            "method": final_method,
            "neckline_box": neckline_box,
            "search_mask_px": int(search_mask.sum()),
            "candidates_generated": len(all_candidates),
            "candidates_passed": len([d for d in debug_candidates if d["passed"]]),
            "all_debug_candidates": debug_candidates,
            "selected_score": best["scoring"]["score"],
            "selected_source": cand.get("source", "?"),
            "seed_bbox": seed_bbox,
            "extension": ext_debug,
            "torso_prior": torso_debug,
            "torso_min_overlap": torso_min_overlap,
            "boundary_refinement": refine_debug,
            "boundary_refine_safety": refine_safety,
            "bbox_before_refine": bbox_before_refine,
            "bbox_after_refine": bbox_after_refine,
            "refine_accepted": refine_accepted,
            "artifact_cleanup": cleanup_debug,
            "bbox_before_cleanup": bbox_before_cleanup,
            "selected_scoring": {
                k: v for k, v in best["scoring"].items()
                if k not in ("reject_reasons",)
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback: SAM multimask on full outerwear bbox (legacy)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_fallback_full_bbox(
    image_bgr: np.ndarray,
    outer_bbox: list[int],
    outer_mask_bin: np.ndarray,
    sam_wrapper,
) -> Optional[dict[str, Any]]:
    """Legacy: SAM multimask on the full outerwear bbox.

    Kept as fallback when neckline-complement finds nothing.
    """
    if sam_wrapper is None:
        return None

    h_img, w_img = image_bgr.shape[:2]
    gx1, gy1, gx2, gy2 = outer_bbox

    margin = 4
    cx1 = max(0, int(gx1 + margin))
    cy1 = max(0, int(gy1 + margin))
    cx2 = min(w_img, int(gx2 - margin))
    cy2 = min(h_img, int(gy2 - margin))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    search_crop = image_bgr[cy1:cy2, cx1:cx2]
    search_rgb = cv2.cvtColor(search_crop, cv2.COLOR_BGR2RGB)
    crop_h, crop_w = search_crop.shape[:2]

    try:
        candidates = sam_wrapper.predict_all_masks(search_rgb, [0, 0, crop_w, crop_h])
    except Exception as exc:
        logger.warning("inner_garment_detector (fallback): SAM failed: %s", exc)
        return None

    if not candidates:
        return None

    outer_crop = ensure_2d_mask(outer_mask_bin[cy1:cy2, cx1:cx2])
    outer_area = int((outer_mask_bin > 0).sum())
    rules = FALLBACK_RULES

    valid: list[dict] = []
    debug_cands: list[dict] = []

    for cand in candidates:
        cmask = ensure_2d_mask((cand["mask"] > 0).astype(np.uint8))
        ca = int((cmask > 0).sum())
        if ca == 0:
            continue
        contained = int((np.logical_and(cmask > 0, outer_crop > 0)).sum())
        containment = contained / max(1, ca)
        outer_crop_area = int((outer_crop > 0).sum())
        area_ratio = ca / max(1, outer_area)

        contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            solidity = cv2.contourArea(cnt) / max(1.0, cv2.contourArea(cv2.convexHull(cnt)))

        x1b, y1b, wb, hb = cv2.boundingRect(
            max(contours, key=cv2.contourArea) if contours else np.array([[0, 0, 1, 1]]))
        x2b, y2b = x1b + wb, y1b + hb
        touches_edge = (x1b <= rules["edge_margin_px"] or y1b <= rules["edge_margin_px"]
                        or x2b >= crop_w - rules["edge_margin_px"]
                        or y2b >= crop_h - rules["edge_margin_px"])
        is_whole = ca > 0.85 * outer_crop_area

        debug_cands.append({
            "bbox": [x1b, y1b, x2b, y2b],
            "score": round(cand["score"], 4),
            "containment": round(containment, 3),
            "area_ratio": round(area_ratio, 4),
            "solidity": round(solidity, 3),
            "touches_edge": touches_edge,
            "is_whole_garment": is_whole,
            "passed": (containment >= rules["min_containment_ratio"]
                       and rules["min_area_ratio"] <= area_ratio <= rules["max_area_ratio"]
                       and solidity >= rules["min_solidity"]
                       and not touches_edge and not is_whole),
        })

        if (containment >= rules["min_containment_ratio"]
                and rules["min_area_ratio"] <= area_ratio <= rules["max_area_ratio"]
                and solidity >= rules["min_solidity"]
                and not touches_edge and not is_whole):
            valid.append({
                "mask": cmask,
                "score": cand["score"],
                "bbox_xyxy": [cx1 + x1b, cy1 + y1b, cx1 + x2b, cy1 + y2b],
                "containment": containment,
                "area_ratio": area_ratio,
                "solidity": solidity,
            })

    if not valid:
        logger.info("inner_garment_detector (fallback): no candidate passed.")
        return None

    best = max(valid, key=lambda v: v["score"])
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = best["mask"]

    return {
        "bbox_xyxy": best["bbox_xyxy"],
        "bbox_format": "xyxy",
        "mask": full_mask,
        "coarse_class_name": "inner_garment",
        "coarse_class_id": -1,
        "fine_class_name": "inner_garment",
        "fine_class_id": -1,
        "source": "sam_fallback_full_bbox",
        "score": round(best["score"], 4),
        "debug": {
            "method": "sam_multimask_full_bbox_fallback",
            "search_box": [cx1, cy1, cx2, cy2],
            "candidates_checked": len(debug_cands),
            "candidates_passed": len(valid),
            "all_debug_candidates": debug_cands,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def detect_inner_garment_from_sam(
    image: np.ndarray,
    outer_instance: dict[str, Any],
    sam_wrapper=None,
) -> Optional[dict[str, Any]]:
    """Detect inner garment under outerwear.

    Strategy:
      1. PRIMARY: Neckline-complement geometric analysis (outer_mask complement
         in the collar/chest-opening region).
      2. FALLBACK: SAM multimask on the full outerwear bbox.

    Args:
        image: Full BGR uint8 H×W×3 image.
        outer_instance: Garment instance dict with bbox_xyxy/bbox and mask.
        sam_wrapper: Optional SamHqWrapper for refinement / SAM-based candidates.

    Returns:
        Pseudo-instance dict or None.
    """
    outer_bbox = _extract_bbox(outer_instance)
    if outer_bbox is None:
        logger.warning("inner_garment_detector: no bbox on outerwear instance")
        return None

    outer_mask = _load_mask(outer_instance)
    if outer_mask is None:
        logger.warning("inner_garment_detector: no mask on outerwear instance")
        return None

    h_img, w_img = image.shape[:2]
    if outer_mask.shape[:2] != (h_img, w_img):
        outer_mask = cv2.resize(outer_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    outer_mask_bin = ensure_2d_mask((outer_mask > 0).astype(np.uint8))

    # ── Primary: neckline-complement rules ──────────────────────────────
    result = detect_inner_by_neckline_rules(image, outer_bbox, outer_mask_bin, sam_wrapper)
    if result is not None:
        return result

    # ── Fallback: SAM multimask on full outerwear bbox ───────────────────
    logger.info("inner_garment_detector: neckline branch found nothing, trying fallback...")
    return _detect_fallback_full_bbox(image, outer_bbox, outer_mask_bin, sam_wrapper)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("inner_garment_detector smoke test ...")

    h, w = 300, 200
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    outer_mask[30:270, 40:160] = 255
    outer_bbox = [40, 30, 160, 270]
    outer_mask_bin = (outer_mask > 0).astype(np.uint8)

    # Test neckline ROI construction
    neckline_box = _construct_neckline_roi(outer_bbox, h, w)
    print(f"  neckline_box: {neckline_box}")
    assert neckline_box[0] >= outer_bbox[0] and neckline_box[2] <= outer_bbox[2]
    assert neckline_box[1] >= outer_bbox[1] and neckline_box[3] <= outer_bbox[3]

    # Test complement search mask
    search = _build_complement_search_mask(outer_bbox, outer_mask_bin, neckline_box, h, w)
    print(f"  search_mask pixels: {search.sum()}")
    # Inside bbox, inside neckline, outside outer_mask → should be empty for
    # this synthetic (outer_mask fills almost the whole bbox).  That's correct.
    # Let's make a more realistic case: outer_mask[70:100, 70:130] = 0 (hole)
    outer_mask[70:100, 70:130] = 0
    outer_mask_bin2 = (outer_mask > 0).astype(np.uint8)
    search2 = _build_complement_search_mask(outer_bbox, outer_mask_bin2, neckline_box, h, w)
    print(f"  search_mask after hole: {search2.sum()} px")
    assert search2.sum() > 0, "hole in outerwear should produce complement pixels"

    # Test CC extraction
    ccs = _extract_cc_candidates(search2)
    print(f"  CC candidates: {len(ccs)}")
    # At least one connected component from the hole
    assert len(ccs) >= 1, f"expected >=1 CC from hole, got {len(ccs)}"

    # Test scoring
    if ccs:
        opening = _build_opening_core_region(outer_bbox, h, w)
        s = _score_candidate(ccs[0], outer_bbox, outer_mask_bin2, neckline_box, opening, h, w)
        print(f"  score: {s['score']:.3f}, passed={s['passed']}, reasons={s['reject_reasons']}")

    # Test _extract_bbox
    assert _extract_bbox({"bbox_xyxy": [10, 20, 100, 200]}) == [10, 20, 100, 200]
    print("  _extract_bbox OK")

    print("All inner_garment_detector smoke tests passed.")
