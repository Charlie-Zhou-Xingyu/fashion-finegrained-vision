"""
Anatomical region zoom for 3.1.2 open-vocabulary part detection.

Before running Grounding DINO on a garment crop, this module crops further to
the anatomical sub-region where the target part typically appears, then scales
up 2× so small parts (zipper, button, pocket) get more pixel budget.

Coordinate conventions (all pixel coords, xyxy format):
    full_image  — original BGR image at native resolution
    garment_crop — image cropped to the garment instance bbox
    zoom_crop    — anatomical sub-region extracted from garment_crop, then
                   resized (zoomed) by zoom_factor

Transform chain (zoom → full):
    full_x = zoom_x / scale_x + offset_x
    full_y = zoom_y / scale_y + offset_y
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ── Per-part zoom strategies ───────────────────────────────────────────────────

ANATOMICAL_ZOOM_CONFIG: dict[str, dict] = {
    # ── Vertical center-line parts ──────────────────────────────────────────
    "zipper": {
        "x_range": [0.30, 0.70],   # keep center 40% width of garment bbox
        "y_range": [0.12, 0.88],   # top 12% → bottom 88% of garment height
        "zoom_factor": 2.0,
    },
    "button": {
        "x_range": [0.30, 0.70],
        "y_range": [0.08, 0.82],
        "zoom_factor": 2.0,
    },
    "placket": {
        "x_range": [0.30, 0.70],
        "y_range": [0.08, 0.82],
        "zoom_factor": 2.0,
    },
    # ── Upper-body parts ────────────────────────────────────────────────────
    "pocket": {
        "x_range": [0.0, 1.0],     # full width (upper body)
        "y_range": [0.0, 0.55],    # top 55% of garment height
        "zoom_factor": 2.0,
    },
    "collar": {
        "x_range": [0.25, 0.75],   # center 50%
        "y_range": [0.0, 0.30],    # top 30%
        "zoom_factor": 2.0,
    },
    "hood": {
        "x_range": [0.20, 0.80],
        "y_range": [0.0, 0.40],
        "zoom_factor": 1.8,
    },
    # ── Mid-body horizontal parts ───────────────────────────────────────────
    "belt": {
        "x_range": [0.0, 1.0],     # full width
        "y_range": [0.32, 0.78],   # middle band
        "zoom_factor": 2.0,
    },
}


def apply_anatomical_zoom(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    inst_bbox: list[int],
    part: str,
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Crop to the anatomical sub-region for ``part`` and scale up.

    Args:
        image: Full BGR H×W×3 image.
        mask: Binary H×W garment mask, or None.
        inst_bbox: ``[x1, y1, x2, y2]`` detection bbox in full-image coords.
        part: Canonical part name (e.g. ``"zipper"``).

    Returns:
        ``(zoomed_image, zoomed_mask, transform)`` where ``transform`` is a dict
        with keys ``offset_x, offset_y, scale_x, scale_y, crop_box, zoom_applied``.
        ``crop_box`` is ``[cx1, cy1, cx2, cy2]`` in **garment-crop-local** coords.
        Coordinates map back to full-image via::

            full_x = zoom_x / scale_x + offset_x
            full_y = zoom_y / scale_y + offset_y
    """
    cfg = ANATOMICAL_ZOOM_CONFIG.get(part)

    h_img, w_img = image.shape[:2]
    gx1, gy1, gx2, gy2 = inst_bbox
    gx1 = max(0, gx1); gy1 = max(0, gy1)
    gx2 = min(w_img, gx2); gy2 = min(h_img, gy2)
    gw, gh = gx2 - gx1, gy2 - gy1

    if cfg is None or gw <= 0 or gh <= 0:
        # Fallback: full garment crop, no zoom
        return _fallback_garment_crop(image, mask, [gx1, gy1, gx2, gy2])

    # ── Compute anatomical sub-region in garment-crop-local coords ──────────
    x_range = cfg.get("x_range", [0.0, 1.0])
    y_range = cfg.get("y_range", [0.0, 1.0])
    zoom = float(cfg.get("zoom_factor", 2.0))

    cx1 = int(round(gw * x_range[0]))
    cy1 = int(round(gh * y_range[0]))
    cx2 = int(round(gw * x_range[1]))
    cy2 = int(round(gh * y_range[1]))
    cx1 = max(0, min(gw - 1, cx1))
    cy1 = max(0, min(gh - 1, cy1))
    cx2 = max(cx1 + 1, min(gw, cx2))
    cy2 = max(cy1 + 1, min(gh, cy2))

    # ── Crop garment-crop to anatomical region ──────────────────────────────
    image_crop = image[gy1:gy2, gx1:gx2]  # garment-level crop
    anatomical_crop = image_crop[cy1:cy2, cx1:cx2]  # sub-region

    if mask is not None:
        # Align mask to image dimensions if needed
        if mask.shape[:2] != (h_img, w_img):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_garment = mask[gy1:gy2, gx1:gx2]
        mask_crop = mask_garment[cy1:cy2, cx1:cx2]
    else:
        mask_crop = None

    # ── Zoom (resize up) ───────────────────────────────────────────────────
    ac_h, ac_w = anatomical_crop.shape[:2]
    new_w = int(round(ac_w * zoom))
    new_h = int(round(ac_h * zoom))
    zoomed_image = cv2.resize(anatomical_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if mask_crop is not None:
        zoomed_mask = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        zoomed_mask = None

    # ── Build transform ────────────────────────────────────────────────────
    # offset = garment_bbox_offset + sub_region_offset (in full-image coords)
    transform = {
        "offset_x": gx1 + cx1,
        "offset_y": gy1 + cy1,
        "scale_x": zoom,
        "scale_y": zoom,
        "crop_box": [cx1, cy1, cx2, cy2],       # garment-crop-local
        "garment_bbox": [gx1, gy1, gx2, gy2],    # full-image coords
        "zoom_applied": True,
        "zoom_factor": zoom,
        "part": part,
    }

    return zoomed_image, zoomed_mask, transform


def map_box_from_zoom_to_original(
    bbox_xyxy: list[float] | tuple,
    transform: dict,
) -> list[int]:
    """Map a DINO detection box from zoomed-image coords back to full-image coords.

    Args:
        bbox_xyxy: ``[x1, y1, x2, y2]`` in zoomed-image pixel coords.
        transform: The transform dict returned by :func:`apply_anatomical_zoom`.

    Returns:
        ``[x1, y1, x2, y2]`` in full-image pixel coords, rounded to int.
    """
    ox = transform["offset_x"]
    oy = transform["offset_y"]
    sx = transform["scale_x"]
    sy = transform["scale_y"]

    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    return [
        int(round(x1 / sx + ox)),
        int(round(y1 / sy + oy)),
        int(round(x2 / sx + ox)),
        int(round(y2 / sy + oy)),
    ]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fallback_garment_crop(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    inst_bbox: list[int],
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Full garment crop with no anatomical zoom (unknown part or invalid bbox)."""
    h_img, w_img = image.shape[:2]
    gx1, gy1, gx2, gy2 = inst_bbox
    gx1 = max(0, gx1); gy1 = max(0, gy1)
    gx2 = min(w_img, gx2); gy2 = min(h_img, gy2)

    pad = 8
    gx1p = max(0, gx1 - pad); gy1p = max(0, gy1 - pad)
    gx2p = min(w_img, gx2 + pad); gy2p = min(h_img, gy2 + pad)

    image_crop = image[gy1p:gy2p, gx1p:gx2p]

    if mask is not None:
        if mask.shape[:2] != (h_img, w_img):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_crop = mask[gy1p:gy2p, gx1p:gx2p]
    else:
        mask_crop = None

    transform = {
        "offset_x": gx1p,
        "offset_y": gy1p,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "crop_box": [0, 0, gx2p - gx1p, gy2p - gy1p],
        "garment_bbox": [gx1, gy1, gx2, gy2],
        "zoom_applied": False,
        "zoom_factor": 1.0,
        "part": "__fallback__",
    }

    return image_crop, mask_crop, transform


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify coordinate math: create a synthetic image, apply zoom, map a box back.
    img = np.zeros((800, 600, 3), dtype=np.uint8)
    mask = np.zeros((800, 600), dtype=np.uint8)
    mask[200:600, 150:450] = 255  # garment region

    inst_bbox = [150, 200, 450, 600]  # 300×400 garment

    # Test zipper zoom
    zoomed_img, zoomed_mask, tform = apply_anatomical_zoom(img, mask, inst_bbox, "zipper")

    print(f"Garment bbox: {inst_bbox}")
    print(f"Zoomed image shape: {zoomed_img.shape}")
    print(f"Transform: offset=({tform['offset_x']}, {tform['offset_y']}), "
          f"scale=({tform['scale_x']}, {tform['scale_y']}), "
          f"crop_box={tform['crop_box']}, zoom={tform['zoom_applied']}")

    # Simulate a DINO detection in zoomed space and map back
    zbox = [50, 30, 180, 60]  # some box in zoomed image
    orig_box = map_box_from_zoom_to_original(zbox, tform)
    print(f"\nZoomed box: {zbox}")
    print(f"Mapped to full-image: {orig_box}")

    # Verify: the mapped box should be inside the garment bbox
    assert orig_box[0] >= inst_bbox[0] - 1, f"x1={orig_box[0]} < gx1={inst_bbox[0]}"
    assert orig_box[2] <= inst_bbox[2] + 1, f"x2={orig_box[2]} > gx2={inst_bbox[2]}"
    print("  OK box within garment bbox")

    # Test fallback for unknown part
    _, _, tform2 = apply_anatomical_zoom(img, mask, inst_bbox, "unknown_part")
    assert tform2["zoom_applied"] is False
    assert tform2["zoom_factor"] == 1.0
    print("\nFallback test: transform.zoom_applied =", tform2["zoom_applied"], "OK")
    print("All smoke tests passed.")
