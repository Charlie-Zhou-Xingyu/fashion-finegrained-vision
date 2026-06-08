"""
Geometry utilities for mask-based local fashion region localization.

This module provides both fixed bbox-ratio localization and adaptive
mask-profile localization for fashion local regions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


BBox = List[float]


def clip_bbox(
    bbox: BBox,
    width: int,
    height: int,
) -> BBox:
    """
    Clip bbox to image boundary.
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0.0, min(float(x1), float(width - 1)))
    y1 = max(0.0, min(float(y1), float(height - 1)))
    x2 = max(0.0, min(float(x2), float(width - 1)))
    y2 = max(0.0, min(float(y2), float(height - 1)))

    if x2 <= x1:
        x2 = min(float(width - 1), x1 + 1.0)

    if y2 <= y1:
        y2 = min(float(height - 1), y1 + 1.0)

    return [x1, y1, x2, y2]


def bbox_to_int_window(
    bbox: BBox,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert bbox to integer window.
    """
    clipped = clip_bbox(bbox, width, height)
    x1, y1, x2, y2 = clipped

    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


def make_window_mask(
    shape: Tuple[int, int],
    window: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Make binary mask for a rectangular window.
    """
    height, width = shape
    x1, y1, x2, y2 = window

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    mask = np.zeros((height, width), dtype=np.uint8)

    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1

    return mask


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """
    Convert binary mask to bbox.
    """
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)

    return [x1, y1, x2, y2]


def intersect_with_instance_mask(
    instance_mask: np.ndarray,
    window_mask: np.ndarray,
) -> np.ndarray:
    """
    Intersect region window with instance mask.
    """
    return (
        instance_mask.astype(bool) & window_mask.astype(bool)
    ).astype(np.uint8)


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 9,
    iterations: int = 1,
) -> np.ndarray:
    """
    Dilate binary mask.

    Args:
        mask: Binary mask.
        kernel_size: Kernel size.
        iterations: Number of dilation iterations.

    Returns:
        Dilated binary mask.
    """
    if kernel_size <= 1:
        return mask.astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

    return (dilated > 0).astype(np.uint8)


def smooth_1d(
    values: np.ndarray,
    kernel_size: int = 9,
) -> np.ndarray:
    """
    Smooth 1D array using moving average.

    Args:
        values: Input values.
        kernel_size: Smoothing kernel size.

    Returns:
        Smoothed values.
    """
    if len(values) == 0:
        return values

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smoothed = np.convolve(padded, kernel, mode="valid")

    return smoothed


def compute_mask_bbox(
    instance_mask: np.ndarray,
) -> BBox | None:
    """
    Compute tight bbox from instance mask.

    Args:
        instance_mask: Binary instance mask.

    Returns:
        xyxy bbox or None.
    """
    return mask_to_bbox(instance_mask)


def compute_row_profile(
    instance_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute row-wise mask profile.

    For each y row, this records left boundary, right boundary, width,
    and center x of the foreground mask.

    Args:
        instance_mask: Binary mask.

    Returns:
        Row profile dictionary.
    """
    mask_bool = instance_mask.astype(bool)
    height, _ = mask_bool.shape

    left = np.full(height, np.nan, dtype=np.float32)
    right = np.full(height, np.nan, dtype=np.float32)
    width = np.zeros(height, dtype=np.float32)
    center = np.full(height, np.nan, dtype=np.float32)
    valid = np.zeros(height, dtype=bool)

    for y in range(height):
        xs = np.where(mask_bool[y])[0]
        if len(xs) == 0:
            continue

        l = float(xs.min())
        r = float(xs.max() + 1)

        left[y] = l
        right[y] = r
        width[y] = r - l
        center[y] = (l + r) / 2.0
        valid[y] = True

    return {
        "left": left,
        "right": right,
        "width": width,
        "center": center,
        "valid": valid,
    }


def compute_bottom_profile(
    instance_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute column-wise bottom contour profile.

    For each x column, this records the bottom-most foreground y.

    Args:
        instance_mask: Binary mask.

    Returns:
        Bottom profile dictionary.
    """
    mask_bool = instance_mask.astype(bool)
    _, width = mask_bool.shape

    bottom = np.full(width, np.nan, dtype=np.float32)
    valid = np.zeros(width, dtype=bool)

    for x in range(width):
        ys = np.where(mask_bool[:, x])[0]
        if len(ys) == 0:
            continue

        bottom[x] = float(ys.max())
        valid[x] = True

    return {
        "bottom": bottom,
        "valid": valid,
    }


def make_top_band_mask(
    instance_mask: np.ndarray,
    y_ratio_end: float,
    x_ratio_start: float = 0.0,
    x_ratio_end: float = 1.0,
) -> np.ndarray:
    """
    Make top band mask based on tight mask bbox.

    Args:
        instance_mask: Binary instance mask.
        y_ratio_end: End ratio in bbox height.
        x_ratio_start: Start ratio in bbox width.
        x_ratio_end: End ratio in bbox width.

    Returns:
        Region mask intersected with instance mask.
    """
    bbox = compute_mask_bbox(instance_mask)
    if bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    height, width = instance_mask.shape
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    window = (
        int(round(x1 + box_w * x_ratio_start)),
        int(round(y1)),
        int(round(x1 + box_w * x_ratio_end)),
        int(round(y1 + box_h * y_ratio_end)),
    )

    window_mask = make_window_mask((height, width), window)

    return intersect_with_instance_mask(instance_mask, window_mask)


def make_horizontal_band_mask(
    instance_mask: np.ndarray,
    y_center: float,
    band_height: float,
) -> np.ndarray:
    """
    Make horizontal band around y_center and intersect with instance mask.

    Args:
        instance_mask: Binary instance mask.
        y_center: Center y.
        band_height: Band height.

    Returns:
        Region mask.
    """
    height, width = instance_mask.shape
    half = band_height / 2.0

    y1 = int(round(y_center - half))
    y2 = int(round(y_center + half))

    window_mask = make_window_mask(
        (height, width),
        (0, y1, width, y2),
    )

    return intersect_with_instance_mask(instance_mask, window_mask)


def make_bottom_contour_band_mask(
    instance_mask: np.ndarray,
    band_height: int,
    min_component_width: int = 3,
) -> np.ndarray:
    """
    Make a thin band following the bottom contour of the instance mask.

    Args:
        instance_mask: Binary instance mask.
        band_height: Height above bottom contour.
        min_component_width: Minimum continuous column width.

    Returns:
        Bottom contour band mask.
    """
    mask_bool = instance_mask.astype(bool)
    height, width = mask_bool.shape
    profile = compute_bottom_profile(instance_mask)

    bottom = profile["bottom"]
    valid = profile["valid"]

    region = np.zeros_like(instance_mask, dtype=np.uint8)

    valid_xs = np.where(valid)[0]
    if len(valid_xs) == 0:
        return region

    for x in valid_xs:
        y_bottom = int(round(bottom[x]))
        y_top = max(0, y_bottom - int(band_height))
        region[y_top : y_bottom + 1, x] = 1

    region = intersect_with_instance_mask(instance_mask, region)

    # Remove tiny noise components.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        region.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(region, dtype=np.uint8)

    for label_id in range(1, num_labels):
        component_width = stats[label_id, cv2.CC_STAT_WIDTH]
        component_area = stats[label_id, cv2.CC_STAT_AREA]

        if component_width >= min_component_width and component_area > 0:
            cleaned[labels == label_id] = 1

    return cleaned


def locate_neckline_mask_adaptive(
    instance_mask: np.ndarray,
    category: str = "top",
) -> np.ndarray:
    """
    Locate neckline using adaptive top mask band.

    This is more tolerant than fixed center window, especially for round necks.

    Args:
        instance_mask: Binary instance mask.
        category: Target category.

    Returns:
        Neckline region mask.
    """
    bbox = compute_mask_bbox(instance_mask)
    if bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    _, y1, _, y2 = bbox
    mask_h = y2 - y1

    if category in {"outwear", "dress"}:
        y_ratio_end = 0.30
        x_start = 0.06
        x_end = 0.94
    else:
        y_ratio_end = 0.28
        x_start = 0.08
        x_end = 0.92

    region = make_top_band_mask(
        instance_mask=instance_mask,
        y_ratio_end=y_ratio_end,
        x_ratio_start=x_start,
        x_ratio_end=x_end,
    )

    kernel_size = int(max(5, min(17, mask_h * 0.025)))
    region = dilate_mask(region, kernel_size=kernel_size, iterations=1)

    # Keep dilation inside a slightly larger top window to avoid flooding.
    guard = make_top_band_mask(
        instance_mask=np.ones_like(instance_mask, dtype=np.uint8),
        y_ratio_end=min(0.35, y_ratio_end + 0.08),
        x_ratio_start=max(0.0, x_start - 0.05),
        x_ratio_end=min(1.0, x_end + 0.05),
    )
    region = (region.astype(bool) & guard.astype(bool)).astype(np.uint8)

    return region


def locate_hem_mask_adaptive(
    instance_mask: np.ndarray,
    category: str = "top",
) -> np.ndarray:
    """
    Locate hem using bottom contour band.

    Args:
        instance_mask: Binary instance mask.
        category: Target category.

    Returns:
        Hem region mask.
    """
    bbox = compute_mask_bbox(instance_mask)
    if bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    _, y1, _, y2 = bbox
    mask_h = y2 - y1

    if category in {"dress", "skirt"}:
        ratio = 0.07
        max_band = 36
    elif category == "pants":
        ratio = 0.055
        max_band = 30
    else:
        ratio = 0.075
        max_band = 34

    band_height = int(max(6, min(max_band, mask_h * ratio)))

    return make_bottom_contour_band_mask(
        instance_mask=instance_mask,
        band_height=band_height,
    )


def locate_leg_opening_mask_adaptive(
    instance_mask: np.ndarray,
) -> np.ndarray:
    """
    Locate pants leg opening using a thin bottom contour band.

    Args:
        instance_mask: Binary instance mask.

    Returns:
        Leg opening region mask.
    """
    bbox = compute_mask_bbox(instance_mask)
    if bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    _, y1, _, y2 = bbox
    mask_h = y2 - y1

    band_height = int(max(5, min(24, mask_h * 0.045)))

    return make_bottom_contour_band_mask(
        instance_mask=instance_mask,
        band_height=band_height,
    )


def locate_waist_mask_adaptive(
    instance_mask: np.ndarray,
    category: str,
) -> np.ndarray:
    """
    Locate waist using category-aware adaptive rules.

    Args:
        instance_mask: Binary instance mask.
        category: Target category.

    Returns:
        Waist region mask.
    """
    bbox = compute_mask_bbox(instance_mask)
    if bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    _, y1, _, y2 = bbox
    mask_h = y2 - y1

    # For pants and skirts, waist is near the top opening.
    if category in {"pants", "skirt"}:
        if category == "pants":
            y_ratio_end = 0.14
        else:
            y_ratio_end = 0.16

        return make_top_band_mask(
            instance_mask=instance_mask,
            y_ratio_end=y_ratio_end,
            x_ratio_start=0.00,
            x_ratio_end=1.00,
        )

    # For tops, "waist" usually refers to lower body/waistline near bottom.
    if category == "top":
        y_center = y1 + mask_h * 0.78
        band_height = max(8, min(36, mask_h * 0.12))

        return make_horizontal_band_mask(
            instance_mask=instance_mask,
            y_center=y_center,
            band_height=band_height,
        )

    # For dresses and outwear, find the narrowest row in middle range.
    profile = compute_row_profile(instance_mask)
    widths = profile["width"]
    valid = profile["valid"]

    search_y1 = int(round(y1 + mask_h * 0.25))
    search_y2 = int(round(y1 + mask_h * 0.68))

    search_y1 = max(0, search_y1)
    search_y2 = min(instance_mask.shape[0], search_y2)

    candidate_ys = np.arange(search_y1, search_y2)
    candidate_ys = candidate_ys[valid[candidate_ys]]

    if len(candidate_ys) == 0:
        y_center = y1 + mask_h * 0.42
    else:
        candidate_widths = widths[candidate_ys].astype(np.float32)
        smoothed = smooth_1d(candidate_widths, kernel_size=11)
        min_index = int(np.argmin(smoothed))
        y_center = float(candidate_ys[min_index])

    band_height = max(10, min(42, mask_h * 0.10))

    return make_horizontal_band_mask(
        instance_mask=instance_mask,
        y_center=y_center,
        band_height=band_height,
    )


def locate_shoulder_mask_adaptive(
    instance_mask: np.ndarray,
) -> np.ndarray:
    """
    Locate shoulder using top-wide band.

    Args:
        instance_mask: Binary instance mask.

    Returns:
        Shoulder region mask.
    """
    return make_top_band_mask(
        instance_mask=instance_mask,
        y_ratio_end=0.22,
        x_ratio_start=0.00,
        x_ratio_end=1.00,
    )


def infer_sleeve_type(
    category: str,
    raw_category_name: str = "",
) -> str:
    """
    Infer sleeve type from category and raw category name.

    Returns:
        one of: long, short, sleeveless, unknown
    """
    raw = raw_category_name.lower().strip()
    category = category.lower().strip()

    sleeveless_keywords = [
        "vest",
        "sling",
        "sleeveless",
        "tank",
    ]
    short_keywords = [
        "short sleeve",
        "short-sleeve",
        "short_sleeve",
    ]
    long_keywords = [
        "long sleeve",
        "long-sleeve",
        "long_sleeve",
        "outwear",
    ]

    if any(keyword in raw for keyword in sleeveless_keywords):
        return "sleeveless"

    if any(keyword in raw for keyword in short_keywords):
        return "short"

    if any(keyword in raw for keyword in long_keywords):
        return "long"

    if category == "outwear":
        return "long"

    # If unknown top/dress, keep conservative.
    if category in {"top", "dress"}:
        return "unknown"

    return "unknown"


def _keep_large_components(
    mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """
    Keep large connected components.

    Args:
        mask: Binary mask.
        min_area: Minimum component area.

    Returns:
        Cleaned mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned


def _extract_component_distal_band(
    component: np.ndarray,
    side: str,
    sleeve_type: str,
) -> np.ndarray:
    """
    Extract distal cuff-like band from one sleeve component.

    Args:
        component: Binary connected component mask.
        side: left or right.
        sleeve_type: long, short, unknown.

    Returns:
        Cuff-like distal band mask.
    """
    ys, xs = np.where(component > 0)
    result = np.zeros_like(component, dtype=np.uint8)

    if len(xs) == 0 or len(ys) == 0:
        return result

    x1 = int(xs.min())
    x2 = int(xs.max() + 1)
    y1 = int(ys.min())
    y2 = int(ys.max() + 1)

    comp_w = max(1, x2 - x1)
    comp_h = max(1, y2 - y1)

    if sleeve_type == "long":
        # Long sleeve cuff is usually at the lower distal end.
        y_cut = int(round(y2 - comp_h * 0.22))

        # Also restrict to the outer half to avoid selecting torso-side pixels.
        if side == "left":
            x_cut = int(round(x1 + comp_w * 0.65))
            candidate = (
                (component > 0)
                & (np.indices(component.shape)[0] >= y_cut)
                & (np.indices(component.shape)[1] <= x_cut)
            )
        else:
            x_cut = int(round(x2 - comp_w * 0.65))
            candidate = (
                (component > 0)
                & (np.indices(component.shape)[0] >= y_cut)
                & (np.indices(component.shape)[1] >= x_cut)
            )

    elif sleeve_type == "short":
        # Short sleeve cuff is on the outer side of the upper/mid sleeve.
        # It should be much thinner than the whole sleeve.
        if side == "left":
            x_cut = int(round(x1 + comp_w * 0.38))
            candidate = (
                (component > 0)
                & (np.indices(component.shape)[1] <= x_cut)
            )
        else:
            x_cut = int(round(x2 - comp_w * 0.38))
            candidate = (
                (component > 0)
                & (np.indices(component.shape)[1] >= x_cut)
            )

        # Avoid selecting shoulder-top and body-bottom too much.
        y_top = int(round(y1 + comp_h * 0.20))
        y_bottom = int(round(y1 + comp_h * 0.85))
        candidate = (
            candidate
            & (np.indices(component.shape)[0] >= y_top)
            & (np.indices(component.shape)[0] <= y_bottom)
        )

    else:
        # Unknown sleeve type: conservative distal strip.
        if comp_h >= comp_w:
            y_cut = int(round(y2 - comp_h * 0.20))
            candidate = (component > 0) & (
                np.indices(component.shape)[0] >= y_cut
            )
        else:
            if side == "left":
                x_cut = int(round(x1 + comp_w * 0.30))
                candidate = (component > 0) & (
                    np.indices(component.shape)[1] <= x_cut
                )
            else:
                x_cut = int(round(x2 - comp_w * 0.30))
                candidate = (component > 0) & (
                    np.indices(component.shape)[1] >= x_cut
                )

    result[candidate] = 1
    return result


def locate_cuff_mask_adaptive(
    instance_mask: np.ndarray,
    category: str = "top",
    raw_category_name: str = "",
) -> np.ndarray:
    """
    Locate cuff using a conservative side-window baseline.

    This is a temporary geometry baseline. It may include larger sleeve areas,
    but it is usually more stable than trying to infer sleeve distal ends from
    mask geometry only.

    Args:
        instance_mask: Binary instance mask.
        category: Target category.
        raw_category_name: Raw dataset category name.

    Returns:
        Cuff candidate mask.
    """
    raw = raw_category_name.lower().strip()

    # Sleeveless garments should not have cuff.
    sleeveless_keywords = [
        "vest",
        "sling",
        "sleeveless",
        "tank",
    ]

    if any(keyword in raw for keyword in sleeveless_keywords):
        return np.zeros_like(instance_mask, dtype=np.uint8)

    bbox = compute_mask_bbox(instance_mask)
    if bbox is None:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    height, width = instance_mask.shape
    x1, y1, x2, y2 = bbox

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    # Default stable side-window baseline.
    # This intentionally keeps a larger candidate area.
    if "short sleeve" in raw:
        y_start_ratio = 0.12
        y_end_ratio = 0.62
        side_ratio = 0.34
    elif "long sleeve" in raw or category == "outwear":
        y_start_ratio = 0.22
        y_end_ratio = 0.92
        side_ratio = 0.32
    else:
        y_start_ratio = 0.18
        y_end_ratio = 0.78
        side_ratio = 0.32

    left_window = (
        int(round(x1)),
        int(round(y1 + box_h * y_start_ratio)),
        int(round(x1 + box_w * side_ratio)),
        int(round(y1 + box_h * y_end_ratio)),
    )

    right_window = (
        int(round(x2 - box_w * side_ratio)),
        int(round(y1 + box_h * y_start_ratio)),
        int(round(x2)),
        int(round(y1 + box_h * y_end_ratio)),
    )

    left_mask = make_window_mask((height, width), left_window)
    right_mask = make_window_mask((height, width), right_window)

    window_mask = ((left_mask > 0) | (right_mask > 0)).astype(np.uint8)

    return intersect_with_instance_mask(instance_mask, window_mask)

# -------------------------------------------------------------------------
# Legacy fixed bbox-ratio functions kept for compatibility.
# -------------------------------------------------------------------------


def get_relative_window(
    bbox: BBox,
    image_width: int,
    image_height: int,
    x1_ratio: float,
    y1_ratio: float,
    x2_ratio: float,
    y2_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    Get bbox-relative window.
    """
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    window = [
        x1 + box_w * x1_ratio,
        y1 + box_h * y1_ratio,
        x1 + box_w * x2_ratio,
        y1 + box_h * y2_ratio,
    ]

    return bbox_to_int_window(window, image_width, image_height)


def locate_neckline_window(
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Locate neckline by fixed bbox-relative geometry.
    """
    return get_relative_window(
        bbox,
        image_width,
        image_height,
        x1_ratio=0.25,
        y1_ratio=0.00,
        x2_ratio=0.75,
        y2_ratio=0.25,
    )


def locate_hem_window(
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Locate hem by fixed bbox-relative geometry.
    """
    return get_relative_window(
        bbox,
        image_width,
        image_height,
        x1_ratio=0.00,
        y1_ratio=0.80,
        x2_ratio=1.00,
        y2_ratio=1.00,
    )


def locate_waist_window(
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Locate waist by fixed bbox-relative geometry.
    """
    return get_relative_window(
        bbox,
        image_width,
        image_height,
        x1_ratio=0.00,
        y1_ratio=0.35,
        x2_ratio=1.00,
        y2_ratio=0.58,
    )


def locate_shoulder_window(
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Locate shoulder by fixed bbox-relative geometry.
    """
    return get_relative_window(
        bbox,
        image_width,
        image_height,
        x1_ratio=0.00,
        y1_ratio=0.00,
        x2_ratio=1.00,
        y2_ratio=0.25,
    )


def locate_cuff_mask(
    instance_mask: np.ndarray,
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Locate cuff region by fixed side windows.
    """
    left_window = get_relative_window(
        bbox,
        image_width,
        image_height,
        x1_ratio=0.00,
        y1_ratio=0.25,
        x2_ratio=0.28,
        y2_ratio=0.82,
    )
    right_window = get_relative_window(
        bbox,
        image_width,
        image_height,
        x1_ratio=0.72,
        y1_ratio=0.25,
        x2_ratio=1.00,
        y2_ratio=0.82,
    )

    left_mask = make_window_mask(instance_mask.shape, left_window)
    right_mask = make_window_mask(instance_mask.shape, right_window)
    window_mask = ((left_mask > 0) | (right_mask > 0)).astype(np.uint8)

    return intersect_with_instance_mask(instance_mask, window_mask)


def locate_leg_opening_window(
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Locate pants leg opening by fixed bbox-relative geometry.
    """
    return get_relative_window(
        bbox,
        image_width,
        image_height,
        x1_ratio=0.00,
        y1_ratio=0.82,
        x2_ratio=1.00,
        y2_ratio=1.00,
    )
