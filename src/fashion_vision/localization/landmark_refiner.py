"""
Landmark-aware local region refiner.

This module locates local fashion regions using DeepFashion2 garment
landmarks. It is designed as a refinement layer before falling back to
geometry-only mask rules.

Main entry:
    locate_region_by_landmarks(...)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fashion_vision.localization.geometry import intersect_with_instance_mask
from fashion_vision.localization.landmark_region_map import (
    get_landmark_region_rule,
    is_region_supported_by_landmarks,
)


Point = Tuple[float, float]


def is_mask_empty(mask: Optional[np.ndarray]) -> bool:
    """
    Check whether a mask is None or empty.

    Args:
        mask: Binary mask.

    Returns:
        Whether mask is empty.
    """
    if mask is None:
        return True

    return int(mask.sum()) <= 0


def normalize_raw_category_name(raw_category_name: str) -> str:
    """
    Normalize raw category name.

    Args:
        raw_category_name: Raw category name.

    Returns:
        Normalized category name.
    """
    return raw_category_name.lower().strip().replace("_", " ")


def get_instance_landmarks(instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get landmarks from instance.

    Args:
        instance: Instance dict.

    Returns:
        Landmark list.
    """
    landmarks = instance.get("landmarks", [])

    if not isinstance(landmarks, list):
        return []

    return landmarks


def get_bbox_from_mask(instance_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute bbox from binary mask.

    Args:
        instance_mask: Binary mask.

    Returns:
        xyxy bbox or None.
    """
    ys, xs = np.where(instance_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def get_bbox_size(instance_mask: np.ndarray) -> Tuple[float, float]:
    """
    Get bbox width and height from mask.

    Args:
        instance_mask: Binary mask.

    Returns:
        bbox width and height.
    """
    bbox = get_bbox_from_mask(instance_mask)

    if bbox is None:
        return 1.0, 1.0

    x1, y1, x2, y2 = bbox
    return float(max(1, x2 - x1)), float(max(1, y2 - y1))


def landmark_to_point(
    landmark: Dict[str, Any],
    allow_occluded: bool = True,
) -> Optional[Point]:
    """
    Convert one landmark dict to point.

    Args:
        landmark: Landmark dict.
        allow_occluded: Whether visibility == 1 can be used.

    Returns:
        Point or None.
    """
    visibility = int(landmark.get("visibility", 0))
    x = float(landmark.get("x", 0.0))
    y = float(landmark.get("y", 0.0))

    if x <= 0 or y <= 0:
        return None

    if visibility == 2:
        return x, y

    if allow_occluded and visibility == 1:
        return x, y

    return None


def select_landmark_points(
    landmarks: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    allow_occluded: bool = True,
) -> List[Point]:
    """
    Select landmark points by 1-based landmark indices.

    Args:
        landmarks: Landmark list.
        indices: 1-based landmark indices.
        allow_occluded: Whether to use occluded landmarks.

    Returns:
        Selected points.
    """
    index_set = {int(index) for index in indices}
    points: List[Point] = []

    for landmark in landmarks:
        index = int(landmark.get("index", 0))

        if index not in index_set:
            continue

        point = landmark_to_point(
            landmark=landmark,
            allow_occluded=allow_occluded,
        )

        if point is not None:
            points.append(point)

    return points


def make_bbox_window_mask(
    shape: Tuple[int, int],
    bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Make rectangular window mask from bbox.

    Args:
        shape: Mask shape, H,W.
        bbox: xyxy bbox.

    Returns:
        Binary window mask.
    """
    height, width = shape
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))

    mask = np.zeros((height, width), dtype=np.uint8)

    if x2 <= x1 or y2 <= y1:
        return mask

    mask[y1:y2, x1:x2] = 1
    return mask


def points_bbox(points: Sequence[Point]) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute bbox from points.

    Args:
        points: Points.

    Returns:
        xyxy bbox or None.
    """
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return min(xs), min(ys), max(xs), max(ys)


def expand_bbox(
    bbox: Tuple[float, float, float, float],
    expand_x: float,
    expand_y: float,
) -> Tuple[float, float, float, float]:
    """
    Expand bbox.

    Args:
        bbox: xyxy bbox.
        expand_x: Horizontal expansion.
        expand_y: Vertical expansion.

    Returns:
        Expanded bbox.
    """
    x1, y1, x2, y2 = bbox
    return x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y


def ensure_min_bbox_size(
    bbox: Tuple[float, float, float, float],
    min_w: float,
    min_h: float,
) -> Tuple[float, float, float, float]:
    """
    Ensure bbox has minimum width and height.

    Args:
        bbox: xyxy bbox.
        min_w: Minimum width.
        min_h: Minimum height.

    Returns:
        Resized bbox.
    """
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    current_w = max(1.0, x2 - x1)
    current_h = max(1.0, y2 - y1)

    target_w = max(current_w, min_w)
    target_h = max(current_h, min_h)

    return (
        cx - target_w / 2.0,
        cy - target_h / 2.0,
        cx + target_w / 2.0,
        cy + target_h / 2.0,
    )


def build_landmark_region_mask(
    points: Sequence[Point],
    region_type: str,
    instance_mask: np.ndarray,
    grouped: bool = False,
) -> np.ndarray:
    """
    Build landmark-guided region window mask.

    Important:
        Landmarks should guide where the region is, but the final shape should
        still come from the instance mask. Therefore this function creates a
        local window around landmarks, instead of drawing landmark polylines
        directly.

    Args:
        points: Landmark points.
        region_type: Local region type.
        instance_mask: Binary instance mask.
        grouped: Whether these points belong to one semantic group, e.g. one sleeve.

    Returns:
        Binary window mask.
    """
    shape = instance_mask.shape

    box_w, box_h = get_bbox_size(instance_mask)
    instance_bbox = get_bbox_from_mask(instance_mask)

    if instance_bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    inst_x1, inst_y1, inst_x2, inst_y2 = instance_bbox

    region_type = region_type.lower().strip()

    if len(points) == 0:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Cuff / leg opening:
    # If grouped=True, points are already one side, e.g. left cuff only.
    # Build one compact local window around this group.
    # ------------------------------------------------------------------
    if region_type in {"cuff", "leg_opening"}:
        bbox = points_bbox(points)

        if bbox is None:
            return np.zeros_like(instance_mask, dtype=np.uint8)

        if region_type == "cuff":
            expand_x = max(5.0, box_w * 0.020)
            expand_y = max(5.0, box_h * 0.020)

            min_w = max(12.0, box_w * 0.035)
            min_h = max(12.0, box_h * 0.035)
        else:
            expand_x = max(8.0, box_w * 0.045)
            expand_y = max(8.0, box_h * 0.045)

            min_w = max(18.0, box_w * 0.08)
            min_h = max(18.0, box_h * 0.08)

        x1, y1, x2, y2 = expand_bbox(
            bbox=bbox,
            expand_x=expand_x,
            expand_y=expand_y,
        )

        x1, y1, x2, y2 = ensure_min_bbox_size(
            bbox=(x1, y1, x2, y2),
            min_w=min_w,
            min_h=min_h,
        )

        return make_bbox_window_mask(
            shape=shape,
            bbox=(x1, y1, x2, y2),
        )


        # ------------------------------------------------------------------
    # Hem:
    # Use a thicker bottom band around hem landmarks.
    #
    # Rationale:
    #   Fashion "hem" is not just a 1-pixel bottom edge. In many images,
    #   hem includes lace/fringe/ruffle/decorative lower trim. Therefore
    #   we extend the band more upward than downward.
    # ------------------------------------------------------------------
    if region_type == "hem":
        ys = [p[1] for p in points]
        y_center = float(np.mean(ys))

        # Use a much thicker band than before.
        # Old: box_h * 0.10
        # New: box_h * 0.22
        band_h = max(28.0, box_h * 0.22)

        # Extend more upward to include fringe / lace / ruffle.
        y1 = y_center - band_h * 0.85
        y2 = y_center + band_h * 0.35

        # Do not go too far below the garment mask bbox.
        y2 = min(y2, inst_y2)

        return make_bbox_window_mask(
            shape=shape,
            bbox=(inst_x1, y1, inst_x2, y2),
        )


    # ------------------------------------------------------------------
    # Waist:
    # Use horizontal band around waist landmarks.
    # ------------------------------------------------------------------
    if region_type == "waist":
        ys = [p[1] for p in points]
        y_center = float(np.mean(ys))

        band_h = max(22.0, box_h * 0.12)

        y1 = y_center - band_h * 0.50
        y2 = y_center + band_h * 0.50

        return make_bbox_window_mask(
            shape=shape,
            bbox=(inst_x1, y1, inst_x2, y2),
        )

    # ------------------------------------------------------------------
    # Neckline:
    # Use local bbox window around neckline points.
    # ------------------------------------------------------------------
    if region_type == "neckline":
        bbox = points_bbox(points)
        if bbox is None:
            return np.zeros_like(instance_mask, dtype=np.uint8)

        expand_x = max(12.0, box_w * 0.07)
        expand_y = max(10.0, box_h * 0.055)

        x1, y1, x2, y2 = expand_bbox(
            bbox=bbox,
            expand_x=expand_x,
            expand_y=expand_y,
        )

        return make_bbox_window_mask(
            shape=shape,
            bbox=(x1, y1, x2, y2),
        )

    # ------------------------------------------------------------------
    # Shoulder:
    # If grouped=True, points are one side shoulder only.
    # Use compact bbox and clamp depth.
    # ------------------------------------------------------------------
    if region_type == "shoulder":
        bbox = points_bbox(points)
        if bbox is None:
            return np.zeros_like(instance_mask, dtype=np.uint8)

        expand_x = max(12.0, box_w * 0.07)
        expand_y = max(10.0, box_h * 0.055)

        x1, y1, x2, y2 = expand_bbox(
            bbox=bbox,
            expand_x=expand_x,
            expand_y=expand_y,
        )

        min_w = max(18.0, box_w * 0.07)
        min_h = max(18.0, box_h * 0.07)

        x1, y1, x2, y2 = ensure_min_bbox_size(
            bbox=(x1, y1, x2, y2),
            min_w=min_w,
            min_h=min_h,
        )

        max_y2 = inst_y1 + box_h * 0.32
        y2 = min(y2, max_y2)

        return make_bbox_window_mask(
            shape=shape,
            bbox=(x1, y1, x2, y2),
        )

    # ------------------------------------------------------------------
    # Fallback:
    # Local bbox around landmarks.
    # ------------------------------------------------------------------
    bbox = points_bbox(points)

    if bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    expand_x = max(12.0, box_w * 0.08)
    expand_y = max(12.0, box_h * 0.08)

    x1, y1, x2, y2 = expand_bbox(
        bbox=bbox,
        expand_x=expand_x,
        expand_y=expand_y,
    )

    return make_bbox_window_mask(
        shape=shape,
        bbox=(x1, y1, x2, y2),
    )


def clean_region_mask(
    region_mask: np.ndarray,
    min_area: int = 8,
) -> np.ndarray:
    """
    Remove tiny connected components.

    Args:
        region_mask: Binary region mask.
        min_area: Minimum component area.

    Returns:
        Cleaned mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        region_mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(region_mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned


def build_region_from_points(
    points: Sequence[Point],
    region_type: str,
    instance_mask: np.ndarray,
    min_region_area: int,
    grouped: bool = False,
) -> Optional[np.ndarray]:
    """
    Build final region mask from selected landmark points.

    Args:
        points: Selected landmark points.
        region_type: Region type.
        instance_mask: Instance binary mask.
        min_region_area: Minimum region area.
        grouped: Whether the points are from one semantic group.

    Returns:
        Final region mask or None.
    """
    if not points:
        return None

    landmark_window = build_landmark_region_mask(
        points=points,
        region_type=region_type,
        instance_mask=instance_mask,
        grouped=grouped,
    )

    if is_mask_empty(landmark_window):
        return None

    region_mask = intersect_with_instance_mask(
        instance_mask=instance_mask,
        window_mask=landmark_window,
    )

    if is_mask_empty(region_mask):
        # In some cases landmark points lie exactly on garment boundary.
        # Dilate landmark window once more before giving up.
        box_w, box_h = get_bbox_size(instance_mask)
        retry_radius = int(max(5, min(box_w, box_h) * 0.035))
        kernel_size = retry_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        expanded = cv2.dilate(landmark_window.astype(np.uint8), kernel, iterations=1)
        region_mask = intersect_with_instance_mask(
            instance_mask=instance_mask,
            window_mask=expanded,
        )

    if is_mask_empty(region_mask):
        return None

    region_mask = clean_region_mask(
        region_mask=region_mask,
        min_area=int(min_region_area),
    )

    if int(region_mask.sum()) < int(min_region_area):
        return None

    return region_mask.astype(np.uint8)


def build_grouped_region_mask(
    groups: Dict[str, List[int]],
    landmarks: Sequence[Dict[str, Any]],
    region_type: str,
    instance_mask: np.ndarray,
    allow_occluded: bool,
    min_region_area: int,
) -> Optional[np.ndarray]:
    """
    Build region mask from grouped landmark indices.

    Example:
        cuff:
            left  -> [7, 8, 9, 10, 11, 12, 13]
            right -> [19, 20, 21, 22, 23, 24]

    Each group is processed independently:
        group points -> local bbox window -> window ∩ instance mask

    Args:
        groups: Group name to landmark indices.
        landmarks: Landmark list.
        region_type: Region type.
        instance_mask: Binary instance mask.
        allow_occluded: Whether occluded landmarks can be used.
        min_region_area: Minimum output area.

    Returns:
        Merged binary region mask or None.
    """
    final_mask = np.zeros_like(instance_mask, dtype=np.uint8)

    for group_name, group_indices in groups.items():
        if not group_indices:
            continue

        points = select_landmark_points(
            landmarks=landmarks,
            indices=group_indices,
            allow_occluded=allow_occluded,
        )

        if not points:
            continue

        group_mask = build_region_from_points(
            points=points,
            region_type=region_type,
            instance_mask=instance_mask,
            min_region_area=min_region_area,
            grouped=True,
        )

        if group_mask is None:
            continue

        final_mask = ((final_mask > 0) | (group_mask > 0)).astype(np.uint8)

    if is_mask_empty(final_mask):
        return None

    final_mask = clean_region_mask(
        region_mask=final_mask,
        min_area=int(min_region_area),
    )

    if int(final_mask.sum()) < int(min_region_area):
        return None

    return final_mask.astype(np.uint8)


def flatten_groups(groups: Dict[str, List[int]]) -> List[int]:
    """
    Flatten grouped landmark indices.

    Args:
        groups: Group name to indices.

    Returns:
        Unique index list.
    """
    indices: List[int] = []

    for group_indices in groups.values():
        indices.extend([int(index) for index in group_indices])

    return sorted(set(indices))


def locate_region_by_landmarks(
    instance: Dict[str, Any],
    instance_mask: np.ndarray,
    region_type: str,
    raw_category_name: str,
    allow_occluded: bool = True,
    min_region_area: int = 8,
) -> Optional[np.ndarray]:
    """
    Locate local region using DeepFashion2 landmarks.

    Args:
        instance: Instance dict containing landmarks.
        instance_mask: Binary instance mask.
        region_type: Local region type.
        raw_category_name: Raw DeepFashion2 category name.
        allow_occluded: Whether visibility == 1 landmarks can be used.
        min_region_area: Minimum output area.

    Returns:
        Binary region mask, or None if landmark refinement is unavailable.

    Important:
        If a rule contains "groups", groups are processed independently.
        This is critical for cuff/leg opening/shoulder, because left/right
        landmarks should not be wrapped by one huge bbox.
    """
    region_type = region_type.lower().strip()
    raw_category_name = normalize_raw_category_name(raw_category_name)

    rule = get_landmark_region_rule(
        raw_category_name=raw_category_name,
        region_type=region_type,
    )

    if rule is None:
        return None

    if not is_region_supported_by_landmarks(
        raw_category_name=raw_category_name,
        region_type=region_type,
    ):
        # Explicitly unsupported by landmarks, e.g. cuff for sleeveless garments.
        return np.zeros_like(instance_mask, dtype=np.uint8)

    landmarks = get_instance_landmarks(instance)

    if not landmarks:
        return None

    groups = rule.get("groups", {})

    if isinstance(groups, dict) and len(groups) > 0:
        normalized_groups: Dict[str, List[int]] = {}

        for group_name, group_indices in groups.items():
            if not isinstance(group_indices, list):
                continue

            normalized_groups[str(group_name)] = [
                int(index) for index in group_indices
            ]

        grouped_mask = build_grouped_region_mask(
            groups=normalized_groups,
            landmarks=landmarks,
            region_type=region_type,
            instance_mask=instance_mask,
            allow_occluded=allow_occluded,
            min_region_area=min_region_area,
        )

        if grouped_mask is not None:
            return grouped_mask.astype(np.uint8)

        # For paired regions, never fall back to flattened indices.
        # Flattened indices would create one huge bbox across left/right parts.
        if region_type in {"cuff", "leg_opening", "shoulder"}:
            return None

        # If grouped version fails for non-paired regions, fall back to flattened indices.
        indices = flatten_groups(normalized_groups)

    else:
        indices = [int(index) for index in rule.get("indices", [])]

    if not indices:
        return None

    points = select_landmark_points(
        landmarks=landmarks,
        indices=indices,
        allow_occluded=allow_occluded,
    )

    if not points:
        return None

    region_mask = build_region_from_points(
        points=points,
        region_type=region_type,
        instance_mask=instance_mask,
        min_region_area=min_region_area,
        grouped=False,
    )

    if region_mask is None:
        return None

    return region_mask.astype(np.uint8)
