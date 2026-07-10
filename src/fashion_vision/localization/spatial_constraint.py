"""
Spatial constraint filters for open-vocabulary region localization (3.1.2).

Two types of spatial constraints:

1. **Side** (left/right): re-rank detections by bbox center x-coordinate.
   "left" and "right" are viewer/image perspective, not wearer perspective.

2. **Direction** (upper/lower/front_upper/back): filter detections by bbox
   center y-coordinate relative to the parent garment bounding box.
   - upper:       top 40% of garment height
   - lower:       bottom 40% of garment height
   - front_upper: top 50% of garment height (chest area proxy)
   - back:        unsupported from a single frontal image — no filter applied

In both cases, if the filter would produce an empty list the original list is
returned unchanged to avoid over-constraining.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def select_side_detection(
    detections: list[dict],
    side: str,
) -> list[dict]:
    """
    Re-rank detections so the leftmost (or rightmost) bbox comes first.

    The caller should still apply a score threshold before calling this;
    this function only re-orders, it does not filter by score.

    Args:
        detections: List of dicts with "bbox_xyxy": [x1, y1, x2, y2].
            Already sorted descending by score (output of GroundingDINOLocator.detect).
        side: "left" — prefer the detection whose center is furthest image-left.
              "right" — prefer the detection whose center is furthest image-right.

    Returns:
        Re-ordered copy of detections. Empty list if input is empty.

    Raises:
        ValueError: If side is not "left" or "right".
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    if not detections:
        return []

    reverse = side == "right"
    return sorted(
        detections,
        key=lambda d: (d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2,
        reverse=reverse,
    )


def select_direction_detection(
    detections: list[dict],
    direction: str,
    garment_bbox_xyxy: list[int] | None,
) -> list[dict]:
    """
    Filter detections to those matching a vertical/directional constraint.

    Args:
        detections: List of dicts with "bbox_xyxy": [x1, y1, x2, y2].
        direction: One of "upper", "lower", "front_upper", "back".
        garment_bbox_xyxy: Parent garment bbox [x1, y1, x2, y2] used as
            reference for computing thresholds.  If None, no filtering is done.

    Returns:
        Filtered list.  Returns original list if filtering would empty it or
        if the direction is not supported ("back").

    Raises:
        ValueError: If direction is not a recognised value.
    """
    _SUPPORTED = ("upper", "lower", "front_upper", "back")
    if direction not in _SUPPORTED:
        raise ValueError(f"direction must be one of {_SUPPORTED}, got {direction!r}")

    if not detections:
        return detections

    if direction == "back":
        logger.warning(
            "select_direction_detection: 'back' direction cannot be resolved "
            "from a single frontal image — returning all detections unchanged."
        )
        return detections

    if garment_bbox_xyxy is None:
        logger.warning(
            "select_direction_detection: garment_bbox_xyxy is None — "
            "cannot apply direction filter, returning all detections."
        )
        return detections

    gx1, gy1, gx2, gy2 = garment_bbox_xyxy
    g_height = max(1, gy2 - gy1)

    if direction == "upper":
        threshold_y = gy1 + 0.4 * g_height
        filtered = [
            d for d in detections
            if (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2 < threshold_y
        ]
    elif direction == "lower":
        threshold_y = gy1 + 0.6 * g_height
        filtered = [
            d for d in detections
            if (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2 > threshold_y
        ]
    else:  # front_upper
        threshold_y = gy1 + 0.5 * g_height
        filtered = [
            d for d in detections
            if (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2 < threshold_y
        ]

    if not filtered:
        logger.warning(
            "select_direction_detection: direction=%r filter returned 0 results — "
            "falling back to all detections.",
            direction,
        )
        return detections

    return filtered


if __name__ == "__main__":
    # --- side tests ---
    dets = [
        {"bbox_xyxy": [300, 100, 400, 200], "score": 0.8, "label": "sleeve"},  # right side
        {"bbox_xyxy": [50,  100, 150, 200], "score": 0.7, "label": "sleeve"},  # left side
    ]

    left_first = select_side_detection(dets, "left")
    assert left_first[0]["bbox_xyxy"][0] == 50,  "leftmost detection should be first"

    right_first = select_side_detection(dets, "right")
    assert right_first[0]["bbox_xyxy"][0] == 300, "rightmost detection should be first"

    assert select_side_detection([], "left") == []

    # --- direction tests ---
    garment_box = [0, 0, 200, 400]   # garment spans y=0..400
    dets_vertical = [
        {"bbox_xyxy": [50, 50,  150, 100], "score": 0.8},   # center_y=75  → upper
        {"bbox_xyxy": [50, 300, 150, 380], "score": 0.7},   # center_y=340 → lower
    ]

    upper = select_direction_detection(dets_vertical, "upper", garment_box)
    assert len(upper) == 1 and upper[0]["score"] == 0.8, "should keep upper detection"

    lower = select_direction_detection(dets_vertical, "lower", garment_box)
    assert len(lower) == 1 and lower[0]["score"] == 0.7, "should keep lower detection"

    front = select_direction_detection(dets_vertical, "front_upper", garment_box)
    assert len(front) == 1 and front[0]["score"] == 0.8, "front_upper keeps top-half"

    back = select_direction_detection(dets_vertical, "back", garment_box)
    assert len(back) == 2, "back returns all detections unchanged"

    # fallback when filter empties
    all_lower = [{"bbox_xyxy": [50, 300, 150, 380], "score": 0.7}]
    result = select_direction_detection(all_lower, "upper", garment_box)
    assert len(result) == 1, "fallback: return original when filter would empty"

    print("spatial_constraint OK")
