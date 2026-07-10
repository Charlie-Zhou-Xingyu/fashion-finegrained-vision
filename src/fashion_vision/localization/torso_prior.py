"""
Torso prior construction for inner garment detection (3.1.2).

Provides a proxy torso mask from the outerwear bbox when pose keypoints are
unavailable, and a keypoint-based polygon builder as a future extension point.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# ── Proxy torso fractions (of outerwear bbox) ─────────────────────────────────
_PROXY_X_LO = 0.18
_PROXY_X_HI = 0.82
_PROXY_Y_LO = 0.03
_PROXY_Y_HI = 0.88


def build_proxy_torso_prior(
    outer_bbox: list[int],
    h_img: int,
    w_img: int,
) -> Tuple[np.ndarray, list[int], dict]:
    """Build a proxy torso mask from outerwear bbox.

    When pose keypoints are unavailable, the outerwear bbox is shrunk to a
    plausible torso region:
        x: 18%–82%  of outerwear width
        y:  3%–88%  of outerwear height

    Args:
        outer_bbox: ``[x1, y1, x2, y2]`` outerwear bounding box.
        h_img, w_img: Full image dimensions.

    Returns:
        ``(torso_mask, torso_bbox, debug_dict)``.
        torso_mask is a uint8 H×W binary mask (0/255).
    """
    gx1, gy1, gx2, gy2 = outer_bbox
    gw, gh = gx2 - gx1, gy2 - gy1

    tx1 = max(0, int(gx1 + gw * _PROXY_X_LO))
    ty1 = max(0, int(gy1 + gh * _PROXY_Y_LO))
    tx2 = min(w_img, int(gx1 + gw * _PROXY_X_HI))
    ty2 = min(h_img, int(gy1 + gh * _PROXY_Y_HI))

    torso_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if tx2 > tx1 and ty2 > ty1:
        torso_mask[ty1:ty2, tx1:tx2] = 255

    torso_bbox = [tx1, ty1, tx2, ty2]
    debug = {
        "source": "proxy",
        "torso_bbox": torso_bbox,
        "outer_bbox": outer_bbox,
        "fractions": {"x_lo": _PROXY_X_LO, "x_hi": _PROXY_X_HI,
                       "y_lo": _PROXY_Y_LO, "y_hi": _PROXY_Y_HI},
    }
    return torso_mask, torso_bbox, debug


def build_torso_prior_from_keypoints(
    keypoints: Optional[Dict[str, Any]],
    image_shape: Tuple[int, int],
    outer_bbox: list[int],
) -> Tuple[np.ndarray, list[int], dict]:
    """Build torso mask from pose keypoints (shoulders + hips).

    Constructs a convex hull from left/right shoulder and left/right hip
    keypoints, dilates it, and returns the mask.  Falls back to
    :func:`build_proxy_torso_prior` when keypoints are missing or invalid.

    Args:
        keypoints: Dict mapping name → ``(x, y)`` point.  Expected keys:
            ``left_shoulder``, ``right_shoulder``, ``left_hip``, ``right_hip``.
        image_shape: ``(h, w)`` tuple.
        outer_bbox: ``[x1, y1, x2, y2]`` — used for fallback only.

    Returns:
        ``(torso_mask, torso_bbox, debug_dict)``.
    """
    h, w = image_shape[:2]
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

    def _valid_pt(v):
        return v is not None and hasattr(v, "__len__") and len(v) >= 2

    if keypoints is None or not all(
        k in keypoints and _valid_pt(keypoints[k]) for k in required
    ):
        return build_proxy_torso_prior(outer_bbox, h, w)

    pts = []
    for k in required:
        pt = keypoints[k]
        try:
            pts.append((float(pt[0]), float(pt[1])))
        except (TypeError, ValueError):
            return build_proxy_torso_prior(outer_bbox, h, w)

    pts_arr = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts_arr)

    torso_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(torso_mask, [hull], 255)

    # Dilate to capture garment extent beyond skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    torso_mask = cv2.dilate(torso_mask, kernel, iterations=2)

    ys, xs = np.where(torso_mask > 0)
    if len(ys) < 10:
        return build_proxy_torso_prior(outer_bbox, h, w)

    torso_bbox = [int(xs.min()), int(ys.min()),
                  int(xs.max() + 1), int(ys.max() + 1)]
    debug = {
        "source": "keypoints",
        "torso_bbox": torso_bbox,
        "keypoints_used": {k: list(keypoints[k]) for k in required},
    }
    return torso_mask, torso_bbox, debug


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    h, w = 300, 200
    outer_bbox = [40, 30, 160, 270]

    # Proxy
    mask, bbox, dbg = build_proxy_torso_prior(outer_bbox, h, w)
    assert mask.shape == (h, w)
    assert mask.sum() > 0
    assert dbg["source"] == "proxy"
    assert bbox[0] >= outer_bbox[0] and bbox[2] <= outer_bbox[2]
    print("  proxy torso OK")

    # Keypoint fallback (no keypoints → proxy)
    mask2, bbox2, dbg2 = build_torso_prior_from_keypoints(None, (h, w), outer_bbox)
    assert dbg2["source"] == "proxy"
    print("  fallback to proxy OK")

    # Keypoints present
    kps = {
        "left_shoulder": (55, 60),
        "right_shoulder": (145, 60),
        "left_hip": (60, 220),
        "right_hip": (140, 220),
    }
    mask3, bbox3, dbg3 = build_torso_prior_from_keypoints(kps, (h, w), outer_bbox)
    assert dbg3["source"] == "keypoints"
    assert mask3.sum() > 0
    print("  keypoints torso OK")

    print("All torso_prior smoke tests passed.")
