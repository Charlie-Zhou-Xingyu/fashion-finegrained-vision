"""
Inner-garment boundary refinement (3.1.2).

Refines the inner-garment bbox and mask using edge, colour-gradient, and texture
profiles in the opening/torso region.  Operates independently in the horizontal
and vertical directions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# ── Default refinement rules ──────────────────────────────────────────────────
DEFAULT_REFINE_RULES: dict[str, float] = {
    # Horizontal search
    "h_search_margin_x": 0.10,       # search margin as fraction of outer_w
    "edge_weight": 0.40,
    "color_grad_weight": 0.35,
    "texture_weight": 0.25,
    "h_boundary_min_strength": 0.15,
    # Vertical scan
    "v_scan_step": 2,
    "v_color_change_thresh": 35.0,   # Lab CIE76 delta
    "v_texture_change_thresh": 2.0,  # Laplacian-std ratio
    "v_persist_rows": 5,             # consecutive changed rows before stop
    "v_max_extend_ratio": 0.15,      # max downward extension / outer_h
}


# ── Profile helpers ───────────────────────────────────────────────────────────

def _edge_profile_x(gray: np.ndarray) -> np.ndarray:
    """Canny vertical-edge profile summed along y, normalised [0,1]."""
    e = cv2.Canny(gray, 30, 100)
    p = e.sum(axis=0).astype(np.float32)
    mx = p.max()
    return p / mx if mx > 0 else p


def _color_grad_profile_x(lab: np.ndarray) -> np.ndarray:
    """x-gradient of a*/b* Lab channels, summed along y, normalised [0,1]."""
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    ga = np.abs(np.diff(a, axis=1))
    gb = np.abs(np.diff(b, axis=1))
    ga = np.pad(ga, ((0, 0), (1, 0)), mode="edge")
    gb = np.pad(gb, ((0, 0), (1, 0)), mode="edge")
    p = ga.sum(axis=0) + gb.sum(axis=0)
    mx = p.max()
    return p / mx if mx > 0 else p


def _texture_profile_x(gray: np.ndarray) -> np.ndarray:
    """Laplacian texture profile summed along y, normalised [0,1]."""
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    p = lap.sum(axis=0)
    mx = p.max()
    return p / mx if mx > 0 else p


# ── Main refiner ──────────────────────────────────────────────────────────────

def refine_inner_boundary(
    image_bgr: np.ndarray,
    inner_mask: np.ndarray,
    inner_bbox: list[int],
    outer_bbox: list[int],
    outer_mask_bin: np.ndarray,
    opening_roi: Optional[list[int]] = None,
    torso_mask: Optional[np.ndarray] = None,
    rules: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, list[int], dict]:
    """Refine inner-garment boundary using edge / colour / texture profiles.

    *Horizontal*: composite profile → outward peak search from inner centre.
    *Vertical*: row-wise scan downward using seed Lab colour + texture, stops
    when a persistent change streak is detected.

    Args:
        image_bgr: Full BGR uint8 image H×W×3.
        inner_mask: Current inner-garment binary mask H×W (0/255).
        inner_bbox: ``[x1, y1, x2, y2]`` of *inner_mask*.
        outer_bbox: ``[x1, y1, x2, y2]`` of the outerwear instance.
        outer_mask_bin: Binary outerwear mask H×W (0/1 or 0/255).
        opening_roi: ``[x1, y1, x2, y2]`` of the front-opening region.
        torso_mask: Optional binary torso mask H×W (0/255).
        rules: Refinement rule dict (uses :data:`DEFAULT_REFINE_RULES` if None).

    Returns:
        ``(refined_mask, refined_bbox, debug_dict)``.
        The refined mask is uint8 0/255, same shape as *image_bgr*.
    """
    h, w = image_bgr.shape[:2]
    if rules is None:
        rules = dict(DEFAULT_REFINE_RULES)

    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    ix1, iy1, ix2, iy2 = inner_bbox
    inner_cx = (ix1 + ix2) / 2.0

    # ── Build search mask: opening_roi ∩ torso ∩ ¬outer ──────────────────
    search_mask = np.zeros((h, w), dtype=np.uint8)
    if opening_roi is not None:
        ox1, oy1, ox2, oy2 = opening_roi
        search_mask[oy1:oy2, ox1:ox2] = 255
    else:
        search_mask[gy1:gy2, gx1:gx2] = 255

    if torso_mask is not None:
        search_mask = cv2.bitwise_and(search_mask, (torso_mask > 0).astype(np.uint8) * 255)

    outer_bin_255 = (outer_mask_bin > 0).astype(np.uint8) * 255
    search_mask = cv2.bitwise_and(search_mask, cv2.bitwise_not(outer_bin_255))

    debug: dict[str, Any] = {
        "old_bbox": list(inner_bbox),
        "refined_bbox": list(inner_bbox),
        "edge_profile_x": None,
        "color_grad_profile_x": None,
        "texture_profile_x": None,
        "y_change_profile": [],
        "h_refined": False,
        "v_refined": False,
    }

    if search_mask.sum() < 20:
        return inner_mask.copy(), list(inner_bbox), debug

    # ═══════════════════════════════════════════════════════════════════════
    # Horizontal refinement
    # ═══════════════════════════════════════════════════════════════════════
    refined_ix1, refined_ix2 = ix1, ix2

    h_margin = int(gw * rules["h_search_margin_x"])
    s_x1 = max(gx1, ix1 - h_margin, 0)
    s_x2 = min(gx2, ix2 + h_margin, w)
    s_y1 = max(iy1, 0)
    s_y2 = min(iy2, h)

    if s_y2 > s_y1 and s_x2 > s_x1:
        band = image_bgr[s_y1:s_y2, s_x1:s_x2]
        if band.size > 0:
            gray_b = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            lab_b = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)

            ep = _edge_profile_x(gray_b)
            cp = _color_grad_profile_x(lab_b)
            tp = _texture_profile_x(gray_b)

            blen = s_x2 - s_x1
            composite = (
                rules["edge_weight"] * ep[:blen]
                + rules["color_grad_weight"] * cp[:blen]
                + rules["texture_weight"] * tp[:blen]
            )
            # Light smoothing
            kernel = np.ones(3) / 3.0
            composite = np.convolve(composite, kernel, mode="same")

            thr = rules["h_boundary_min_strength"]
            rel_c = inner_cx - s_x1

            # Left boundary: strongest peak to the left of centre
            left_part = composite[: max(1, int(rel_c))]
            left_peaks = np.where(left_part > thr)[0]
            if len(left_peaks):
                refined_ix1 = s_x1 + int(left_peaks[-1])
            # (if no left peak, keep original ix1)

            # Right boundary: first strong peak to the right of centre
            right_part = composite[min(len(composite) - 1, int(rel_c)):]
            right_peaks = np.where(right_part > thr)[0]
            if len(right_peaks):
                refined_ix2 = s_x1 + int(rel_c) + int(right_peaks[0])
            # (if no right peak, keep original ix2)

            # Constrain within opening ROI
            if opening_roi is not None:
                refined_ix1 = max(refined_ix1, opening_roi[0])
                refined_ix2 = min(refined_ix2, opening_roi[2])
            refined_ix1 = max(gx1, refined_ix1)
            refined_ix2 = min(gx2, refined_ix2)

            debug["edge_profile_x"] = ep.tolist()
            debug["color_grad_profile_x"] = cp.tolist()
            debug["texture_profile_x"] = tp.tolist()
            debug["h_refined"] = (refined_ix1 != ix1 or refined_ix2 != ix2)

    # ═══════════════════════════════════════════════════════════════════════
    # Vertical refinement (downward only — y1 is seed from neckline, trusted)
    # ═══════════════════════════════════════════════════════════════════════
    refined_iy2 = iy2

    # Seed colour: median Lab of the existing inner mask region
    inner_region = image_bgr[iy1:iy2, ix1:ix2]
    inner_inner = inner_mask[iy1:iy2, ix1:ix2] > 0
    inner_px = inner_region[inner_inner]
    seed_median = None
    if len(inner_px) > 5:
        lab_px = cv2.cvtColor(
            inner_px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)
        seed_median = np.median(lab_px, axis=0)

    v_max_extend = int(gh * rules["v_max_extend_ratio"])
    scan_end = min(h, iy2 + v_max_extend)
    if opening_roi is not None:
        scan_end = min(scan_end, opening_roi[3])

    change_streak = 0
    y_profile: list[dict] = []

    for y in range(iy2, scan_end, int(rules["v_scan_step"])):
        # Pixels in search_mask at this row
        row_sel = (search_mask[y, refined_ix1:refined_ix2] > 0)
        n_sel = int(row_sel.sum())
        if n_sel < 3:
            break  # no relevant pixels left

        row_rgb = image_bgr[y, refined_ix1:refined_ix2][row_sel]
        row_lab = cv2.cvtColor(
            row_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)
        row_median = np.median(row_lab, axis=0)

        color_delta = 0.0
        if seed_median is not None:
            color_delta = float(np.sqrt(np.sum((row_median - seed_median) ** 2)))

        # Texture: Laplacian std on the search-mask row pixels
        row_gray = cv2.cvtColor(
            image_bgr[y:y + 1, refined_ix1:refined_ix2], cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        row_lap = np.abs(cv2.Laplacian(row_gray, cv2.CV_32F))
        row_tex = float(np.std(row_lap))

        y_profile.append({
            "y": y,
            "color_delta": round(color_delta, 1),
            "texture": round(row_tex, 3),
            "pixels": n_sel,
        })

        if (color_delta > rules["v_color_change_thresh"]
                or row_tex > rules["v_texture_change_thresh"]):
            change_streak += 1
        else:
            change_streak = 0
            refined_iy2 = y

        if change_streak >= int(rules["v_persist_rows"]):
            break

    debug["y_change_profile"] = y_profile
    debug["v_refined"] = (refined_iy2 != iy2)

    refined_bbox = [refined_ix1, iy1, refined_ix2, refined_iy2]

    # ── Build refined mask ────────────────────────────────────────────────
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    # Copy existing inner mask within the refined bbox
    refined_mask[iy1:refined_iy2, refined_ix1:refined_ix2] = (
        inner_mask[iy1:refined_iy2, refined_ix1:refined_ix2]
    )
    # Fill newly exposed area below original y2 within search_mask
    if refined_iy2 > iy2:
        new_region = search_mask[iy2:refined_iy2, refined_ix1:refined_ix2]
        refined_mask[iy2:refined_iy2, refined_ix1:refined_ix2] = new_region

    debug["refined_bbox"] = refined_bbox
    return refined_mask, refined_bbox, debug


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    h, w = 300, 200
    # Synthetic inner garment inside outerwear
    outer_bbox = [40, 30, 160, 270]
    gx1, gy1, gx2, gy2 = outer_bbox

    outer_mask = np.zeros((h, w), dtype=np.uint8)
    outer_mask[30:270, 40:160] = 255
    # Opening: remove a V-shaped region
    outer_mask[80:200, 70:130] = 0
    outer_bin = (outer_mask > 0).astype(np.uint8)

    # Opening ROI
    opening_roi = _construct_opening_roi_helper(outer_bbox, h, w)

    # Inner mask (seed from neckline, too short)
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    inner_mask[80:130, 75:125] = 255
    inner_bbox = [75, 80, 125, 130]

    # Fake BGR image
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[80:200, 75:125] = [100, 80, 60]  # dark inner region
    img[:, :] += np.random.randint(0, 10, img.shape, dtype=np.uint8)  # noise

    result_mask, result_bbox, dbg = refine_inner_boundary(
        img, inner_mask, inner_bbox, outer_bbox, outer_bin, opening_roi,
    )
    print(f"  old_bbox={dbg['old_bbox']}")
    print(f"  refined_bbox={dbg['refined_bbox']}")
    print(f"  h_refined={dbg['h_refined']}, v_refined={dbg['v_refined']}")
    assert result_bbox[2] <= outer_bbox[2], "x2 must not exceed outer_bbox"
    assert result_bbox[3] <= opening_roi[3], "y2 must not exceed opening_roi bottom"
    print("  boundary refiner smoke test OK")


def _construct_opening_roi_helper(outer_bbox, h, w):
    """Minimal helper for self-test only (avoids circular import)."""
    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1
    ox1 = max(0, int(gx1 + gw * 0.22))
    oy1 = max(0, int(gy1 + gh * 0.08))
    ox2 = min(w, int(gx1 + gw * 0.78))
    oy2 = min(h, int(gy1 + gh * 0.85))
    return [ox1, oy1, ox2, oy2]
