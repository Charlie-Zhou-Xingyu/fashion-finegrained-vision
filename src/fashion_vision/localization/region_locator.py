"""
Local fashion region locator.

This module locates local regions from one garment instance using query type,
category, raw category name, instance mask, and optional garment landmarks.

Pipeline:
    1. Parse natural language query to region type.
    2. Check whether the region is supported by garment category.
    3. Load instance mask.
    4. Try landmark-aware localization first.
    5. Fall back to geometry-only localization if landmark refinement fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from fashion_vision.localization.geometry import (
    locate_cuff_mask_adaptive,
    locate_hem_mask_adaptive,
    locate_leg_opening_mask_adaptive,
    locate_neckline_mask_adaptive,
    locate_shoulder_mask_adaptive,
    locate_waist_mask_adaptive,
    mask_to_bbox,
)
from fashion_vision.localization.landmark_refiner import (
    is_mask_empty,
    locate_region_by_landmarks,
)
from fashion_vision.localization.query_parser import parse_region_type


SUPPORTED_REGIONS_BY_CATEGORY = {
    "top": {"neckline", "cuff", "hem", "waist", "shoulder"},
    "dress": {"neckline", "cuff", "hem", "waist", "shoulder"},
    "outwear": {"neckline", "cuff", "hem", "waist", "shoulder"},
    "skirt": {"waist", "hem"},
    "pants": {"waist", "hem", "leg_opening"},
}


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """
    Load binary mask from PNG path.

    Args:
        mask_path: Mask path.

    Returns:
        Binary uint8 mask.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")

    return (mask > 0).astype(np.uint8)


def get_instance_category(instance: Dict[str, Any]) -> str:
    """
    Get normalized target category from instance record.

    Args:
        instance: Standardized instance record.

    Returns:
        Target category string.
    """
    return str(
        instance.get(
            "target_category",
            instance.get("category", "unknown"),
        )
    )


def get_raw_category_name(instance: Dict[str, Any]) -> str:
    """
    Get raw dataset category name from instance record.

    Args:
        instance: Standardized instance record.

    Returns:
        Raw category name if available.
    """
    return str(
        instance.get(
            "raw_category_name",
            instance.get(
                "category_name",
                instance.get("raw_category", ""),
            ),
        )
    )


def is_region_supported_for_category(
    category: str,
    region_type: str,
) -> bool:
    """
    Check if a region is supported for a target category.

    Args:
        category: Target category.
        region_type: Region type.

    Returns:
        Whether supported.
    """
    return region_type in SUPPORTED_REGIONS_BY_CATEGORY.get(category, set())


def resolve_instance_mask_path(
    instance: Dict[str, Any],
    prefer_pred_mask: bool = True,
) -> Optional[str]:
    """
    Resolve which mask path should be used.

    Args:
        instance: Instance record.
        prefer_pred_mask: Whether to use predicted mask first.

    Returns:
        Mask path or None.
    """
    pred_mask_path = instance.get("pred_mask_path")
    gt_mask_path = instance.get("gt_mask_path")

    if prefer_pred_mask and pred_mask_path:
        return str(pred_mask_path)

    if gt_mask_path:
        return str(gt_mask_path)

    if pred_mask_path:
        return str(pred_mask_path)

    return None


def locate_region_by_geometry(
    instance_mask: np.ndarray,
    region_type: str,
    category: str,
    raw_category_name: str = "",
) -> np.ndarray:
    """
    Locate region using geometry-only adaptive mask baseline.

    Args:
        instance_mask: Binary instance mask.
        region_type: Region type.
        category: Target category.
        raw_category_name: Raw category name.

    Returns:
        Binary region mask.
    """
    if region_type == "neckline":
        return locate_neckline_mask_adaptive(
            instance_mask=instance_mask,
            category=category,
        )

    if region_type == "hem":
        return locate_hem_mask_adaptive(
            instance_mask=instance_mask,
            category=category,
        )

    if region_type == "waist":
        return locate_waist_mask_adaptive(
            instance_mask=instance_mask,
            category=category,
        )

    if region_type == "shoulder":
        return locate_shoulder_mask_adaptive(
            instance_mask=instance_mask,
        )

    if region_type == "cuff":
        return locate_cuff_mask_adaptive(
            instance_mask=instance_mask,
            category=category,
            raw_category_name=raw_category_name,
        )

    if region_type == "leg_opening":
        return locate_leg_opening_mask_adaptive(
            instance_mask=instance_mask,
        )

    return np.zeros_like(instance_mask, dtype=np.uint8)

def locate_region_from_instance(
    instance: Dict[str, Any],
    query: str,
    image_width: int,
    image_height: int,
    prefer_pred_mask: bool = True,
    use_landmarks: bool = True,
    allow_occluded_landmarks: bool = True,
) -> Dict[str, Any]:
    """
    Locate local region from one garment instance.

    Args:
        instance: Standardized instance record.
        query: Natural language query.
        image_width: Image width.
        image_height: Image height.
        prefer_pred_mask: Whether to use pred mask first.
        use_landmarks: Whether to try landmark-aware localization first.
        allow_occluded_landmarks: Whether visibility == 1 landmarks can be used.

    Returns:
        Local region result dictionary.
    """
    region_type = parse_region_type(query)
    category = get_instance_category(instance)
    raw_category_name = get_raw_category_name(instance)

    if region_type == "unknown":
        return {
            "status": "failed",
            "reason": "unknown_region_type",
            "query": query,
            "region_type": region_type,
            "target_category": category,
            "raw_category_name": raw_category_name,
        }

    if not is_region_supported_for_category(category, region_type):
        return {
            "status": "failed",
            "reason": "unsupported_region_for_category",
            "query": query,
            "region_type": region_type,
            "target_category": category,
            "raw_category_name": raw_category_name,
        }

    mask_path = resolve_instance_mask_path(
        instance=instance,
        prefer_pred_mask=prefer_pred_mask,
    )

    if mask_path is None:
        return {
            "status": "failed",
            "reason": "missing_mask_path",
            "query": query,
            "region_type": region_type,
            "target_category": category,
            "raw_category_name": raw_category_name,
        }

    try:
        instance_mask = load_binary_mask(mask_path)
    except Exception as error:
        return {
            "status": "failed",
            "reason": f"failed_to_load_mask: {error}",
            "query": query,
            "region_type": region_type,
            "target_category": category,
            "raw_category_name": raw_category_name,
            "source_mask_path": mask_path,
        }

    if (
        instance_mask.shape[0] != image_height
        or instance_mask.shape[1] != image_width
    ):
        return {
            "status": "failed",
            "reason": (
                "mask_shape_mismatch: "
                f"mask={instance_mask.shape[:2]}, "
                f"image={(image_height, image_width)}"
            ),
            "query": query,
            "region_type": region_type,
            "target_category": category,
            "raw_category_name": raw_category_name,
            "source_mask_path": mask_path,
        }

    region_mask: Optional[np.ndarray] = None
    method = "geometry_adaptive_mask_baseline"
    landmark_used = False
    fallback_used = False

    # ------------------------------------------------------------------
    # 1. Landmark-aware localization first
    # ------------------------------------------------------------------
    if use_landmarks:
        landmark_region_mask = locate_region_by_landmarks(
            instance=instance,
            instance_mask=instance_mask,
            region_type=region_type,
            raw_category_name=raw_category_name,
            allow_occluded=allow_occluded_landmarks,
        )

        if landmark_region_mask is not None and not is_mask_empty(landmark_region_mask):
            region_mask = landmark_region_mask
            method = "landmark_aware_refinement"
            landmark_used = True

        elif landmark_region_mask is not None and is_mask_empty(landmark_region_mask):
            # Explicit empty result from landmark-aware logic.
            # Example: cuff for sleeveless garment.
            region_mask = landmark_region_mask
            method = "landmark_aware_refinement_empty"
            landmark_used = True

    # ------------------------------------------------------------------
    # 2. Geometry fallback
    # ------------------------------------------------------------------
    if region_mask is None:
        landmarks = instance.get("landmarks", [])

        # Cuff is a small paired local region. If landmarks exist but
        # landmark-aware localization fails, geometry fallback often produces
        # large side bands. Returning failed is safer for visualization/demo.
        if (
            region_type == "cuff"
            and isinstance(landmarks, list)
            and len(landmarks) > 0
        ):
            return {
                "status": "failed",
                "reason": "cuff_landmark_refinement_failed_skip_geometry_fallback",
                "query": query,
                "region_type": region_type,
                "target_category": category,
                "raw_category_name": raw_category_name,
                "source_mask_path": mask_path,
                "method": "landmark_aware_refinement_failed",
                "landmark_used": False,
                "fallback_used": False,
            }

        region_mask = locate_region_by_geometry(
            instance_mask=instance_mask,
            region_type=region_type,
            category=category,
            raw_category_name=raw_category_name,
        )
        fallback_used = True

    # ------------------------------------------------------------------
    # 3. Cuff safety filter
    # ------------------------------------------------------------------
    if region_type == "cuff" and region_mask is not None:
        region_area = float(region_mask.sum())
        instance_area = float(instance_mask.sum())
        area_ratio = region_area / max(1.0, instance_area)

        if area_ratio > 0.30:
            return {
                "status": "failed",
                "reason": f"cuff_region_too_large_area_ratio={area_ratio:.4f}",
                "query": query,
                "region_type": region_type,
                "target_category": category,
                "raw_category_name": raw_category_name,
                "source_mask_path": mask_path,
                "method": method,
                "landmark_used": landmark_used,
                "fallback_used": fallback_used,
            }

    region_bbox = mask_to_bbox(region_mask)

    if region_bbox is None:
        return {
            "status": "failed",
            "reason": "empty_region_mask",
            "query": query,
            "region_type": region_type,
            "target_category": category,
            "raw_category_name": raw_category_name,
            "source_mask_path": mask_path,
            "method": method,
            "landmark_used": landmark_used,
            "fallback_used": fallback_used,
        }

    return {
        "status": "success",
        "method": method,
        "query": query,
        "region_type": region_type,
        "target_instance_id": str(instance.get("instance_id", "unknown")),
        "target_category": category,
        "raw_category_name": raw_category_name,
        "source_mask_path": mask_path,
        "bbox": region_bbox,
        "bbox_format": "xyxy",
        "landmark_used": landmark_used,
        "fallback_used": fallback_used,
        "region_mask": region_mask,
    }
