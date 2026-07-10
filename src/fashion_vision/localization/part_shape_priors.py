"""
Part-specific shape and spatial prior filters for open-vocabulary localization (3.1.2).

Reduces GroundingDINO false positives by rejecting detections that violate
the expected geometry of a given garment part.

All coordinates must be in the same space (full-image xyxy).

When every candidate is rejected the function returns an empty list.
Callers are responsible for emitting a ``not_detected`` status.  Rejected
candidates (with their ``_shape_prior_reasons``) remain on the detection dicts
so callers can surface them in debug metadata.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def filter_by_shape_priors(
    detections: list[dict],
    part: str | None,
    garment_bbox: Optional[list | tuple] = None,
    shape_config: Optional[dict] = None,
) -> list[dict]:
    """
    Filter detections using per-part geometric priors.

    Adds ``_shape_prior_status`` (and ``_shape_prior_reasons`` on rejection)
    directly to each detection dict, then returns only passing detections.

    Args:
        detections: DINO detections with ``"bbox_xyxy"`` and ``"score"`` keys.
            Modified in-place to add ``_shape_prior_*`` metadata.
        part: Canonical part name for config lookup (e.g. "zipper").
            Ignored if ``shape_config`` is provided explicitly.
        garment_bbox: Parent garment [x1, y1, x2, y2] in full-image coords.
            Required for area_ratio, y_band, x_band, and prefer_center_x.
            If None, those checks are skipped.
        shape_config: Explicit config dict.  Pass ``{}`` to disable all checks
            without looking up from ``part``.  ``None`` → look up from ``part``.

    Returns:
        Subset of passing detections. Empty list when all candidates are rejected;
        callers must treat this as a ``not_detected`` signal, not an error.
    """
    if not detections:
        return detections

    if shape_config is None:
        if part:
            from fashion_vision.localization.part_detection_config import get_part_shape_config
            shape_config = get_part_shape_config(part)
        else:
            shape_config = {}

    if not shape_config:
        for d in detections:
            d["_shape_prior_status"] = "not_applicable"
        return detections

    # Pre-compute garment geometry (None-safe).
    gx1 = gy1 = gx2 = gy2 = g_width = g_height = garment_area = garment_cx = None
    if garment_bbox is not None:
        gx1, gy1, gx2, gy2 = (float(v) for v in garment_bbox[:4])
        g_width = max(1.0, gx2 - gx1)
        g_height = max(1.0, gy2 - gy1)
        garment_area = g_width * g_height
        garment_cx = (gx1 + gx2) / 2.0

    kept: list[dict] = []
    rejected: list[dict] = []

    for det in detections:
        bbox = det["bbox_xyxy"]
        dx1, dy1, dx2, dy2 = (float(v) for v in bbox[:4])
        det_w = max(1.0, dx2 - dx1)
        det_h = max(1.0, dy2 - dy1)
        det_area = det_w * det_h
        det_cx = (dx1 + dx2) / 2.0
        det_cy = (dy1 + dy2) / 2.0

        reasons: list[str] = []

        # Area ratio checks (skipped when garment_bbox unavailable).
        if garment_area is not None:
            ar = det_area / garment_area
            lo = shape_config.get("min_area_ratio")
            hi = shape_config.get("max_area_ratio")
            if lo is not None and ar < lo:
                reasons.append(f"area_ratio {ar:.3f} < min {lo}")
            if hi is not None and ar > hi:
                reasons.append(f"area_ratio {ar:.3f} > max {hi}")

        # Aspect ratio checks (purely from bbox, no garment ref needed).
        h_w = det_h / det_w
        w_h = det_w / det_h

        for key, val_key, op, fmt in (
            ("min_aspect_ratio_h_over_w", "min_aspect_ratio_h_over_w", lambda v, t: v < t, "h/w"),
            ("max_aspect_ratio_h_over_w", "max_aspect_ratio_h_over_w", lambda v, t: v > t, "h/w"),
            ("min_aspect_ratio_w_over_h", "min_aspect_ratio_w_over_h", lambda v, t: v < t, "w/h"),
            ("max_aspect_ratio_w_over_h", "max_aspect_ratio_w_over_h", lambda v, t: v > t, "w/h"),
        ):
            threshold = shape_config.get(key)
            if threshold is not None:
                ratio = h_w if fmt == "h/w" else w_h
                op_label = "min" if "min" in key else "max"
                if op(ratio, threshold):
                    reasons.append(f"{fmt} {ratio:.2f} {'<' if op_label == 'min' else '>'} {op_label} {threshold}")

        # Center-x proximity (skipped when garment_bbox unavailable).
        if shape_config.get("prefer_center_x") and garment_cx is not None:
            tol = shape_config.get("center_x_tolerance", 0.30)
            offset = abs(det_cx - garment_cx) / g_width
            if offset > tol:
                reasons.append(f"center_x offset {offset:.2f} > tol {tol}")

        # Vertical band (skipped when garment_bbox unavailable).
        y_band = shape_config.get("y_band")
        if y_band is not None and gy1 is not None:
            cy_norm = (det_cy - gy1) / g_height
            if not (y_band[0] <= cy_norm <= y_band[1]):
                reasons.append(f"cy_norm {cy_norm:.2f} outside y_band {y_band}")

        # Horizontal band (skipped when garment_bbox unavailable).
        x_band = shape_config.get("x_band")
        if x_band is not None and gx1 is not None:
            cx_norm = (det_cx - gx1) / g_width
            if not (x_band[0] <= cx_norm <= x_band[1]):
                reasons.append(f"cx_norm {cx_norm:.2f} outside x_band {x_band}")

        det["_shape_prior_status"] = "rejected" if reasons else "passed"
        if reasons:
            det["_shape_prior_reasons"] = reasons
            rejected.append(det)
        else:
            kept.append(det)

    if not kept:
        logger.warning(
            "part_shape_priors: all %d candidate(s) rejected for part=%r — "
            "returning empty list (caller should emit not_detected status)",
            len(rejected),
            part,
        )
        return []

    return kept
