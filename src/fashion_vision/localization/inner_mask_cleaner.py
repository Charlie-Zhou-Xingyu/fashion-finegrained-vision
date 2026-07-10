"""
Inner-garment mask artifact cleanup (3.1.2).

Removes rectangular ROI cropping artifacts — particularly upper-corner
right-angle residues and side-strip residues — while preserving the
true inner-garment shape.

Strategy:
  1. Soft opening corridor (trapezoid) replaces the rectangular ROI.
  2. Upper-corner suppression via colour-consistency with the seed region.
  3. Side-strip detection and removal (tall narrow edge components).
  4. Main-component preservation with intelligent auxiliary retention.
  5. Light morphology smoothing.
  6. Area-ratio safety gate — reject cleanup if it removes too much.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Default rules ──────────────────────────────────────────────────────────────
DEFAULT_CLEANUP_RULES: dict[str, float] = {
    # Soft opening corridor
    "corridor_top_width_ratio": 0.34,     # narrower at top
    "corridor_bottom_width_ratio": 0.58,  # wider at bottom
    "corridor_y_top_ratio": 0.05,
    "corridor_y_bottom_ratio": 0.88,

    # Upper corner suppression
    "upper_height_ratio": 0.30,           # fraction of inner bbox h
    "corner_width_ratio": 0.25,           # fraction of inner bbox w each side
    "color_delta_threshold": 38.0,        # Lab CIE76 delta

    # Side strip removal
    "side_strip_width_ratio": 0.15,       # fraction of inner bbox w
    "narrow_strip_max_w_ratio": 0.12,     # tall narrow strip max width
    "narrow_strip_min_h_ratio": 0.35,     # tall narrow strip min height
    "strip_rel_cx_edge": 0.25,            # rel_cx < this or > 1-this = edge
    "strip_min_torso_overlap": 0.15,      # below this = likely artifact

    # Small component removal
    "min_component_area_ratio": 0.03,     # smaller than 3% of inner area → remove

    # Main component preservation
    "aux_x_overlap_ratio": 0.35,          # x-overlap with main component to keep
    "aux_min_area_ratio": 0.10,           # area >10% of inner, kept if centred

    # Morphology
    "morph_open_kernel": 3,
    "morph_close_kernel": 3,

    # Safety gate
    "min_cleanup_area_ratio": 0.45,       # cleaned area must be >= 45% of original
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bin(mask: np.ndarray) -> np.ndarray:
    """Convert mask to uint8 0/255."""
    return ((mask > 0).astype(np.uint8)) * 255


def _area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def _bbox_area(bbox: list[int]) -> int:
    return max(1, bbox[2] - bbox[0]) * max(1, bbox[3] - bbox[1])


def _clip_bbox(bbox: list[int], h: int, w: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


# ═══════════════════════════════════════════════════════════════════════════════
# B2: Soft opening corridor
# ═══════════════════════════════════════════════════════════════════════════════

def _build_soft_opening_corridor(
    outer_bbox: list[int],
    h_img: int,
    w_img: int,
    opening_roi: Optional[list[int]] = None,
    torso_mask: Optional[np.ndarray] = None,
    rules: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Build a trapezoidal corridor mask instead of a rectangular ROI.

    The corridor is narrower at the top (collar) and wider at the bottom
    (chest), which naturally cuts off the rectangular upper-corner residues.
    """
    if rules is None:
        rules = DEFAULT_CLEANUP_RULES

    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    cx = gx1 + gw / 2.0

    top_w = gw * rules["corridor_top_width_ratio"]
    bottom_w = gw * rules["corridor_bottom_width_ratio"]
    y_top = gy1 + gh * rules["corridor_y_top_ratio"]
    y_bottom = gy1 + gh * rules["corridor_y_bottom_ratio"]

    # Trapezoid: four corners
    top_left = (max(0, int(cx - top_w / 2)), max(0, int(y_top)))
    top_right = (min(w_img, int(cx + top_w / 2)), max(0, int(y_top)))
    bottom_right = (min(w_img, int(cx + bottom_w / 2)), min(h_img, int(y_bottom)))
    bottom_left = (max(0, int(cx - bottom_w / 2)), min(h_img, int(y_bottom)))

    pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    corridor = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(corridor, [pts], 255)

    # Intersect with opening_roi if provided
    if opening_roi is not None:
        ox1, oy1, ox2, oy2 = opening_roi
        roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        roi_mask[oy1:oy2, ox1:ox2] = 255
        corridor = cv2.bitwise_and(corridor, roi_mask)

    # Intersect with torso_mask if provided
    if torso_mask is not None and torso_mask.sum() > 0:
        torso_bin = _bin(torso_mask)
        corridor = cv2.bitwise_and(corridor, torso_bin)

    return corridor


# ═══════════════════════════════════════════════════════════════════════════════
# Seed colour extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_seed_lab_median(
    image_bgr: np.ndarray,
    seed_mask: Optional[np.ndarray] = None,
    inner_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract median Lab colour from seed_mask or inner_mask centre region.

    Returns (3,) float32 array or None.
    """
    if seed_mask is not None and seed_mask.sum() > 10:
        region = seed_mask
    elif inner_mask is not None and inner_mask.sum() > 10:
        # Use central 40% of inner mask as seed proxy
        ys, xs = np.where(inner_mask > 0)
        if len(ys) < 10:
            return None
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        h_r = (ys.max() - ys.min()) * 0.2
        w_r = (xs.max() - xs.min()) * 0.2
        region = np.zeros_like(inner_mask, dtype=np.uint8)
        y1 = max(0, int(cy - h_r))
        y2 = min(inner_mask.shape[0], int(cy + h_r))
        x1 = max(0, int(cx - w_r))
        x2 = min(inner_mask.shape[1], int(cx + w_r))
        region[y1:y2, x1:x2] = inner_mask[y1:y2, x1:x2]
    else:
        return None

    px = image_bgr[(region > 0) & (image_bgr.sum(axis=2) > 0)]
    if len(px) < 5:
        return None
    lab_px = cv2.cvtColor(
        px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)
    return np.median(lab_px, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# B3: Upper corner suppression
# ═══════════════════════════════════════════════════════════════════════════════

def _suppress_upper_corners(
    image_bgr: np.ndarray,
    inner_mask: np.ndarray,
    inner_bbox: list[int],
    seed_mask: Optional[np.ndarray] = None,
    rules: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, int]:
    """Remove pixels in upper-left and upper-right corners that don't match
    the seed colour or aren't connected to the main component.

    Returns (cleaned_mask, removed_pixel_count).
    """
    if rules is None:
        rules = DEFAULT_CLEANUP_RULES

    h, w = inner_mask.shape[:2]
    ix1, iy1, ix2, iy2 = inner_bbox
    ibw = max(1, ix2 - ix1)
    ibh = max(1, iy2 - iy1)

    upper_y_end = iy1 + int(ibh * rules["upper_height_ratio"])
    left_x_end = ix1 + int(ibw * rules["corner_width_ratio"])
    right_x_start = ix2 - int(ibw * rules["corner_width_ratio"])

    if upper_y_end <= iy1 or left_x_end <= ix1 or right_x_start >= ix2:
        return inner_mask.copy(), 0

    # Build corner mask
    corner_mask = np.zeros((h, w), dtype=np.uint8)
    # Upper band
    corner_mask[iy1:upper_y_end, ix1:ix2] = 1
    # Keep only left + right corner columns
    corner_mask[:, left_x_end:right_x_start] = 0
    corner_mask[inner_mask == 0] = 0  # only where inner mask exists

    corner_pixels = int(corner_mask.sum())
    if corner_pixels < 5:
        return inner_mask.copy(), 0

    # Extract seed Lab colour
    seed_lab = _extract_seed_lab_median(image_bgr, seed_mask=seed_mask, inner_mask=inner_mask)
    if seed_lab is None:
        return inner_mask.copy(), 0

    # Find main connected component for connectivity test
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        _bin(inner_mask), connectivity=8)
    if num_labels <= 1:
        return inner_mask.copy(), 0

    # Largest component = main body
    sizes = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    if not sizes:
        return inner_mask.copy(), 0
    main_label = max(sizes, key=lambda x: x[0])[1]

    # Process each corner pixel
    cleaned = inner_mask.copy()
    removed = 0
    corner_ys, corner_xs = np.where(corner_mask > 0)

    for py, px in zip(corner_ys, corner_xs):
        # Check connectivity to main component
        label_here = labels[py, px]
        is_main = (label_here == main_label)

        # Check connectivity to seed_mask
        is_seed = False
        if seed_mask is not None and seed_mask.sum() > 0:
            # 8-neighbour check
            y0 = max(0, py - 1)
            y1 = min(h, py + 2)
            x0 = max(0, px - 1)
            x1 = min(w, px + 2)
            if (seed_mask[y0:y1, x0:x1] > 0).any():
                is_seed = True

        # Check colour consistency
        px_bgr = image_bgr[py, px].astype(np.float32).reshape(1, 1, 3)
        px_lab = cv2.cvtColor(px_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        delta = float(np.sqrt(np.sum((px_lab - seed_lab) ** 2)))
        color_ok = delta <= rules["color_delta_threshold"]

        # Keep if: connected to main body OR connected to seed OR colour matches
        if not (is_main or is_seed or color_ok):
            cleaned[py, px] = 0
            removed += 1

    return cleaned, removed


# ═══════════════════════════════════════════════════════════════════════════════
# B4 + B5: Side strip removal
# ═══════════════════════════════════════════════════════════════════════════════

def _remove_side_strips(
    inner_mask: np.ndarray,
    inner_bbox: list[int],
    outer_bbox: list[int],
    torso_mask: Optional[np.ndarray] = None,
    rules: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, int, dict]:
    """Remove tall narrow side strips and small edge components.

    Returns (cleaned_mask, removed_pixels, debug).
    """
    if rules is None:
        rules = DEFAULT_CLEANUP_RULES

    ix1, iy1, ix2, iy2 = inner_bbox
    ibw = max(1, ix2 - ix1)
    ibh = max(1, iy2 - iy1)
    gx1, gy1, gx2, gy2 = outer_bbox
    gw = max(1, gx2 - gx1)
    inner_area = _area(inner_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        _bin(inner_mask), connectivity=8)
    if num_labels <= 1:
        return inner_mask.copy(), 0, {"num_components": 0, "removed_labels": []}

    # Build side strip mask (left 15% + right 15% of inner bbox)
    left_edge = ix1 + int(ibw * rules["side_strip_width_ratio"])
    right_edge = ix2 - int(ibw * rules["side_strip_width_ratio"])

    torso_bin = _bin(torso_mask) if (torso_mask is not None and torso_mask.sum() > 0) else None

    cleaned = inner_mask.copy()
    removed = 0
    removed_labels: list[int] = []
    comp_debug: list[dict] = []

    for label_id in range(1, num_labels):
        area_c = int(stats[label_id, cv2.CC_STAT_AREA])
        cx_c = int(stats[label_id, cv2.CC_STAT_LEFT])
        cy_c = int(stats[label_id, cv2.CC_STAT_TOP])
        cw_c = int(stats[label_id, cv2.CC_STAT_WIDTH])
        ch_c = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        comp_cx = float(centroids[label_id, 0])
        comp_cy = float(centroids[label_id, 1])

        rel_cx_comp = (comp_cx - ix1) / ibw
        comp_w_ratio = cw_c / ibw
        comp_h_ratio = ch_c / ibh
        area_ratio_comp = area_c / max(1, inner_area)

        touches_left = cx_c <= left_edge
        touches_right = cx_c + cw_c >= right_edge
        at_side = touches_left or touches_right

        # Torso overlap
        torso_ov = 0.0
        if torso_bin is not None:
            comp_mask = (labels == label_id).astype(np.uint8) * 255
            torso_ov = _area(cv2.bitwise_and(comp_mask, torso_bin)) / max(1, area_c)

        should_remove = False
        reason = ""

        # Rule 1: very small components
        if area_ratio_comp < rules["min_component_area_ratio"]:
            should_remove = True
            reason = f"too_small area_ratio={area_ratio_comp:.4f}"

        # Rule 2: tall narrow strip at edge
        elif (comp_w_ratio < rules["narrow_strip_max_w_ratio"]
              and comp_h_ratio > rules["narrow_strip_min_h_ratio"]
              and (rel_cx_comp < rules["strip_rel_cx_edge"]
                   or rel_cx_comp > 1.0 - rules["strip_rel_cx_edge"])):
            should_remove = True
            reason = (f"tall_narrow_strip w={comp_w_ratio:.3f} h={comp_h_ratio:.3f} "
                      f"rel_cx={rel_cx_comp:.3f}")

        # Rule 3: side component with low torso overlap
        elif at_side and torso_bin is not None and torso_ov < rules["strip_min_torso_overlap"]:
            should_remove = True
            reason = f"low_torso_side torso_ov={torso_ov:.3f}"

        comp_debug.append({
            "label": int(label_id), "area": area_c,
            "bbox": [cx_c, cy_c, cx_c + cw_c, cy_c + ch_c],
            "rel_cx": round(rel_cx_comp, 3),
            "w_ratio": round(comp_w_ratio, 3),
            "h_ratio": round(comp_h_ratio, 3),
            "area_ratio": round(area_ratio_comp, 4),
            "torso_overlap": round(torso_ov, 3),
            "at_side": at_side,
            "removed": should_remove,
            "reason": reason,
        })

        if should_remove:
            cleaned[labels == label_id] = 0
            removed += area_c
            removed_labels.append(int(label_id))

    return cleaned, removed, {
        "num_components": num_labels - 1,
        "removed_labels": removed_labels,
        "components": comp_debug,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# B6: Main component + auxiliary preservation
# ═══════════════════════════════════════════════════════════════════════════════

def _preserve_main_and_aux(
    inner_mask: np.ndarray,
    inner_bbox: list[int],
    seed_mask: Optional[np.ndarray] = None,
    rules: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, int]:
    """Keep the main connected component plus valid auxiliary components.

    Removes orphan components that are unlikely to be real inner garment parts.
    """
    if rules is None:
        rules = DEFAULT_CLEANUP_RULES

    h, w = inner_mask.shape[:2]
    ix1, iy1, ix2, iy2 = inner_bbox
    ibw = max(1, ix2 - ix1)
    inner_area = _area(inner_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        _bin(inner_mask), connectivity=8)
    if num_labels <= 1:
        return inner_mask.copy(), 0

    # Largest component = main
    sizes = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    if not sizes:
        return inner_mask.copy(), 0
    sizes.sort(key=lambda x: -x[0])
    main_label = sizes[0][1]
    main_area = sizes[0][0]
    main_bbox = [int(stats[main_label, cv2.CC_STAT_LEFT]),
                 int(stats[main_label, cv2.CC_STAT_TOP]),
                 int(stats[main_label, cv2.CC_STAT_LEFT] + stats[main_label, cv2.CC_STAT_WIDTH]),
                 int(stats[main_label, cv2.CC_STAT_TOP] + stats[main_label, cv2.CC_STAT_HEIGHT])]
    main_cx = main_bbox[0] + (main_bbox[2] - main_bbox[0]) / 2.0

    cleaned = np.zeros_like(inner_mask)
    cleaned[labels == main_label] = inner_mask[labels == main_label]

    removed = 0
    for area_c, label_id in sizes[1:]:  # skip main
        comp_mask = (labels == label_id).astype(np.uint8) * 255
        cx_c = int(stats[label_id, cv2.CC_STAT_LEFT])
        cy_c = int(stats[label_id, cv2.CC_STAT_TOP])
        cw_c = int(stats[label_id, cv2.CC_STAT_WIDTH])
        ch_c = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        comp_bbox = [cx_c, cy_c, cx_c + cw_c, cy_c + ch_c]
        comp_cx = float(centroids[label_id, 0])

        keep = False

        # Check x-overlap with main component
        x_overlap = max(0, min(main_bbox[2], comp_bbox[2]) - max(main_bbox[0], comp_bbox[0]))
        x_overlap_ratio = x_overlap / max(1, cw_c)
        if x_overlap_ratio > rules["aux_x_overlap_ratio"]:
            keep = True

        # Check seed overlap
        if not keep and seed_mask is not None and seed_mask.sum() > 0:
            seed_ov = _area(cv2.bitwise_and(comp_mask, _bin(seed_mask)))
            if seed_ov > 5:
                keep = True

        # Check: large auxiliary component, centred
        if not keep:
            area_ratio_c = area_c / max(1, inner_area)
            if area_ratio_c > rules["aux_min_area_ratio"]:
                rel_cx_c = (comp_cx - ix1) / ibw
                if 0.25 <= rel_cx_c <= 0.75:
                    keep = True

        if keep:
            cleaned[labels == label_id] = inner_mask[labels == label_id]
        else:
            removed += area_c

    return cleaned, removed


# ═══════════════════════════════════════════════════════════════════════════════
# B7: Morphology smoothing
# ═══════════════════════════════════════════════════════════════════════════════

def _morphology_smooth(
    mask: np.ndarray,
    rules: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Light open + close to remove small noise and fill small holes."""
    if rules is None:
        rules = DEFAULT_CLEANUP_RULES
    ks_open = int(rules["morph_open_kernel"])
    ks_close = int(rules["morph_close_kernel"])
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_open, ks_open))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_close, ks_close))
    m = _bin(mask)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_o)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_c)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def clean_inner_mask_artifacts(
    image_bgr: np.ndarray,
    inner_mask: np.ndarray,
    inner_bbox: list[int],
    outer_bbox: list[int],
    outer_mask_bin: np.ndarray,
    opening_roi: Optional[list[int]] = None,
    torso_mask: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    rules: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, list[int], dict]:
    """Clean inner garment mask by removing ROI cropping artifacts.

    Steps:
        1. Normalise inputs.
        2. Apply soft opening corridor (trapezoid).
        3. Upper corner colour-consistency suppression.
        4. Side-strip detection and removal.
        5. Main-component preservation with aux retention.
        6. Morphology smoothing.
        7. Area-ratio safety gate.

    Args:
        image_bgr: Full BGR uint8 image H×W×3.
        inner_mask: Current inner garment mask H×W.
        inner_bbox: ``[x1, y1, x2, y2]`` of inner_mask.
        outer_bbox: ``[x1, y1, x2, y2]`` of outerwear.
        outer_mask_bin: Binary outerwear mask H×W.
        opening_roi: Front-opening ROI box.
        torso_mask: Optional torso binary mask H×W.
        seed_mask: Optional seed mask from candidate (H×W).
        rules: Cleanup rules dict (uses :data:`DEFAULT_CLEANUP_RULES` if None).

    Returns:
        ``(cleaned_mask, cleaned_bbox, debug_dict)``.
    """
    if rules is None:
        rules = dict(DEFAULT_CLEANUP_RULES)

    h_img, w_img = image_bgr.shape[:2]
    debug: dict[str, Any] = {
        "cleanup_accepted": False,
        "reason": None,
        "removed_pixels": 0,
        "removed_upper_corner_pixels": 0,
        "removed_side_strip_pixels": 0,
        "removed_orphan_pixels": 0,
        "original_area": 0,
        "cleaned_area": 0,
        "original_bbox": list(inner_bbox),
    }

    # ── B1: Normalise ──────────────────────────────────────────────────────
    inner_bin = _bin(inner_mask)
    debug["original_area"] = _area(inner_bin)
    if debug["original_area"] < 20:
        debug["reason"] = "empty_input"
        return inner_mask.copy(), list(inner_bbox), debug

    inner_bbox_clipped = _clip_bbox(inner_bbox, h_img, w_img)

    # ── B2: Soft opening corridor ──────────────────────────────────────────
    corridor = _build_soft_opening_corridor(
        outer_bbox, h_img, w_img, opening_roi=opening_roi,
        torso_mask=torso_mask, rules=rules,
    )
    before_corridor_area = _area(inner_bin)
    inner_bin = cv2.bitwise_and(inner_bin, corridor)
    debug["corridor_removed_pixels"] = before_corridor_area - _area(inner_bin)
    debug["corridor_area"] = _area(corridor)

    # ── B3: Upper corner suppression ───────────────────────────────────────
    inner_bin, removed_corners = _suppress_upper_corners(
        image_bgr, inner_bin, inner_bbox_clipped, seed_mask=seed_mask, rules=rules,
    )
    debug["removed_upper_corner_pixels"] = removed_corners

    # ── B4+B5: Side strip removal ──────────────────────────────────────────
    inner_bin, removed_sides, side_debug = _remove_side_strips(
        inner_bin, inner_bbox_clipped, outer_bbox, torso_mask=torso_mask, rules=rules,
    )
    debug["removed_side_strip_pixels"] = removed_sides
    debug["side_strip_components"] = side_debug

    # ── B6: Main component + aux preservation ──────────────────────────────
    inner_bin, removed_orphans = _preserve_main_and_aux(
        inner_bin, inner_bbox_clipped, seed_mask=seed_mask, rules=rules,
    )
    debug["removed_orphan_pixels"] = removed_orphans

    # ── B7: Morphology smoothing ───────────────────────────────────────────
    inner_bin = _morphology_smooth(inner_bin, rules=rules)

    # ── Recompute bbox ─────────────────────────────────────────────────────
    cleaned_bbox = _compute_bbox_from_mask(inner_bin)
    debug["cleaned_area"] = _area(inner_bin)
    debug["cleaned_bbox"] = cleaned_bbox
    total_removed = (debug["original_area"] - debug["cleaned_area"]
                     + debug.get("corridor_removed_pixels", 0))
    debug["removed_pixels"] = max(0, int(total_removed))

    # ── B8: Safety gate ───────────────────────────────────────────────────
    area_ratio = debug["cleaned_area"] / max(1, debug["original_area"])
    debug["area_ratio"] = round(area_ratio, 4)

    if area_ratio < rules["min_cleanup_area_ratio"]:
        debug["cleanup_accepted"] = False
        debug["reason"] = f"area_too_small area_ratio={area_ratio:.3f}<{rules['min_cleanup_area_ratio']}"
        return inner_mask.copy(), list(inner_bbox), debug

    debug["cleanup_accepted"] = True
    return inner_bin, cleaned_bbox, debug


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_bbox_from_mask(mask: np.ndarray) -> list[int]:
    """Return [x1, y1, x2, y2] bounding box of non-zero pixels."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("inner_mask_cleaner smoke test ...")

    h, w = 300, 200
    outer_bbox = [40, 30, 160, 270]
    inner_bbox = [65, 80, 135, 200]

    # Synthetic inner mask with upper-corner rectangular artifacts
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    # Main body: tapered shape
    inner_mask[120:200, 70:130] = 255
    inner_mask[100:120, 75:125] = 255
    inner_mask[80:100, 80:120] = 255
    # Rectangular corner artifacts (simulating ROI crop edges)
    inner_mask[80:100, 65:80] = 255    # left upper corner
    inner_mask[80:100, 120:135] = 255  # right upper corner

    outer_mask = np.zeros((h, w), dtype=np.uint8)
    outer_mask[30:270, 40:160] = 255
    outer_mask[80:200, 65:135] = 0  # opening
    outer_bin = (outer_mask > 0).astype(np.uint8)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[80:200, 65:135] = [60, 60, 80]  # dark inner region

    result_mask, result_bbox, debug = clean_inner_mask_artifacts(
        img, inner_mask, inner_bbox, outer_bbox, outer_bin,
    )
    print(f"  original_area: {debug['original_area']}")
    print(f"  cleaned_area: {debug['cleaned_area']}")
    print(f"  removed_upper_corner: {debug['removed_upper_corner_pixels']}")
    print(f"  removed_side_strip: {debug['removed_side_strip_pixels']}")
    print(f"  removed_orphan: {debug['removed_orphan_pixels']}")
    print(f"  cleanup_accepted: {debug['cleanup_accepted']}")
    print(f"  original_bbox: {inner_bbox}")
    print(f"  cleaned_bbox: {result_bbox}")

    # Verify: some cleanup happened
    assert debug["original_area"] > 0
    assert result_mask.shape == (h, w)
    # Main body should still be present
    assert _area(result_mask) > 0.45 * debug["original_area"], \
        f"too much removed: {_area(result_mask)} / {debug['original_area']}"

    # Test corridor construction
    corridor = _build_soft_opening_corridor(outer_bbox, h, w)
    assert corridor.sum() > 0
    # Upper region should be narrower
    mid_y = 120
    top_y = 60
    top_row = corridor[top_y, :].sum()
    mid_row = corridor[mid_y, :].sum()
    # Trapezoid should be narrower at top
    assert top_row <= mid_row + 5, f"top_row={top_row}, mid_row={mid_row} — corridor should narrow at top"

    print("All inner_mask_cleaner smoke tests passed.")
