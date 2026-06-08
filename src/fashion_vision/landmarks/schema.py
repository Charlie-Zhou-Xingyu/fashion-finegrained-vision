"""
Category-aware garment landmark schema.

This module provides:
    1. Category-specific valid landmark indices.
    2. Optional landmark semantic names.
    3. Optional landmark region names.
    4. Filtering / enrichment utilities for inference outputs.

The valid index mapping below is generated from:
    data/processed/deepfashion2_landmarks/train.jsonl
    data/processed/deepfashion2_landmarks/validation.jsonl

Generated command:
    python tools/analysis/export_landmark_schema_from_jsonl.py ^
      --jsonl data/processed/deepfashion2_landmarks/train.jsonl ^
              data/processed/deepfashion2_landmarks/validation.jsonl ^
      --output-dir outputs/landmark_schema ^
      --present-threshold 0.05 ^
      --visible-threshold 0.0 ^
      --max-landmarks 39

Important:
    The current model predicts max_landmarks=39 for all categories.
    CATEGORY_TO_VALID_LANDMARK_INDICES removes padded / non-existing landmarks
    for categories with fewer than 39 real landmarks.

Manual correction:
    The semantic names originally used for some upper garments were provisional
    and did not match the actual landmark index convention. For verified classes,
    CATEGORY_TO_LANDMARK_REGIONS explicitly maps category + 1-based index to
    semantic regions and overrides name-based region inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


DEEPFASHION2_CATEGORY_NAMES = [
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear",
    "long sleeve outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress",
]


CATEGORY_TO_VALID_LANDMARK_INDICES: Dict[str, Set[int]] = {
    "long sleeve dress": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37,
    },
    "long sleeve outwear": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39,
    },
    "long sleeve top": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33,
    },
    "short sleeve dress": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29,
    },
    "short sleeve outwear": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31,
    },
    "short sleeve top": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
    },
    "shorts": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    },
    "skirt": {
        1, 2, 3, 4, 5, 6, 7, 8,
    },
    "sling": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
    },
    "sling dress": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
    },
    "trousers": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14,
    },
    "vest": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
    },
    "vest dress": {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
    },
}


def _make_generic_names(num_points: int) -> Dict[int, str]:
    return {idx: f"landmark_{idx:02d}" for idx in range(1, num_points + 1)}


# Semantic names.
#
# Note:
# For all manually checked upper garments / dresses / outwears, these names
# are aligned with the pure-index visualization you verified.
#
# Region assignment still prefers CATEGORY_TO_LANDMARK_REGIONS.
CATEGORY_TO_LANDMARK_NAMES: Dict[str, Dict[int, str]] = {
    "shorts": {
        1: "left_waist",
        2: "right_waist",
        3: "left_side",
        4: "right_side",
        5: "left_crotch",
        6: "right_crotch",
        7: "left_leg_opening_outer",
        8: "left_leg_opening_inner",
        9: "right_leg_opening_inner",
        10: "right_leg_opening_outer",
    },

    "trousers": {
        1: "left_waist",
        2: "right_waist",
        3: "left_side",
        4: "right_side",
        5: "left_crotch",
        6: "right_crotch",
        7: "left_knee_outer",
        8: "left_knee_inner",
        9: "right_knee_inner",
        10: "right_knee_outer",
        11: "left_leg_opening_outer",
        12: "left_leg_opening_inner",
        13: "right_leg_opening_inner",
        14: "right_leg_opening_outer",
    },

    "skirt": {
        1: "left_waist",
        2: "right_waist",
        3: "left_side",
        4: "right_side",
        5: "left_hem",
        6: "left_middle_hem",
        7: "right_middle_hem",
        8: "right_hem",
    },

    # Manually verified by pure-index visualization.
    "short sleeve top": {
        **_make_generic_names(25),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",

        7: "left_shoulder",
        25: "right_shoulder",

        8: "left_sleeve_01",
        9: "left_sleeve_02",
        10: "left_sleeve_03",
        11: "left_sleeve_04",
        12: "left_sleeve_05",

        14: "left_waist",
        18: "right_waist",

        15: "left_hem",
        16: "middle_hem",
        17: "right_hem",

        20: "right_sleeve_01",
        21: "right_sleeve_02",
        22: "right_sleeve_03",
        23: "right_sleeve_04",
        24: "right_sleeve_05",
    },

    # Manually verified by pure-index visualization.
    "long sleeve top": {
        **_make_generic_names(33),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        7: "left_shoulder",
        33: "right_shoulder",

        8: "left_sleeve_01",
        9: "left_sleeve_02",
        10: "left_sleeve_03",
        11: "left_sleeve_04",
        12: "left_sleeve_05",
        13: "left_sleeve_06",

        18: "left_waist",
        22: "right_waist",

        19: "left_hem",
        20: "middle_hem",
        21: "right_hem",

        26: "right_sleeve_01",
        27: "right_sleeve_02",
        28: "right_sleeve_03",
        29: "right_sleeve_04",
        30: "right_sleeve_05",
        31: "right_sleeve_06",
    },

    # Manually verified by pure-index visualization.
    "short sleeve outwear": {
        **_make_generic_names(31),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        # You marked 7 both as left sleeve and shoulder.
        # Region mapping below prioritizes 7 as shoulder.
        7: "left_shoulder",
        25: "right_shoulder",

        8: "left_sleeve_01",
        9: "left_sleeve_02",
        10: "left_sleeve_03",
        11: "left_sleeve_04",
        12: "left_sleeve_05",

        14: "left_waist_01",
        18: "right_waist_01",
        28: "right_waist_02",
        31: "left_waist_02",

        15: "left_hem",
        16: "middle_hem",
        17: "right_hem",

        21: "right_sleeve_01",
        22: "right_sleeve_02",
        23: "right_sleeve_03",
        24: "right_sleeve_04",
    },

    # Manually verified by pure-index visualization.
    "long sleeve outwear": {
        **_make_generic_names(39),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",
        34: "collar_07",

        7: "left_shoulder",
        33: "right_shoulder",

        9: "left_sleeve_01",
        10: "left_sleeve_02",
        11: "left_sleeve_03",
        12: "left_sleeve_04",
        13: "left_sleeve_05",
        14: "left_sleeve_06",

        18: "left_waist_01",
        22: "right_waist_01",
        36: "right_waist_02",
        39: "left_waist_02",

        19: "left_hem",
        20: "middle_hem_01",
        21: "middle_hem_02",
        37: "right_hem",

        26: "right_sleeve_01",
        27: "right_sleeve_02",
        28: "right_sleeve_03",
        29: "right_sleeve_04",
        30: "right_sleeve_05",
        31: "right_sleeve_06",
    },

    # Manually verified by pure-index visualization.
    "vest": {
        **_make_generic_names(15),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        7: "left_shoulder",
        15: "right_shoulder",

        9: "left_waist",
        13: "right_waist",

        10: "left_hem",
        11: "middle_hem",
        12: "right_hem",
    },

    # Manually verified by pure-index visualization.
    "sling": {
        **_make_generic_names(15),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        7: "left_shoulder",
        15: "right_shoulder",

        9: "left_waist",
        13: "right_waist",

        10: "left_hem",
        11: "middle_hem",
        12: "right_hem",
    },

    # Manually verified by pure-index visualization.
    "short sleeve dress": {
        **_make_generic_names(29),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        7: "left_shoulder",
        29: "right_shoulder",

        8: "left_sleeve_01",
        9: "left_sleeve_02",
        10: "left_sleeve_03",
        11: "left_sleeve_04",
        12: "left_sleeve_05",

        15: "left_waist",
        21: "right_waist",

        17: "left_hem",
        18: "middle_hem",
        19: "right_hem",

        24: "right_sleeve_01",
        25: "right_sleeve_02",
        26: "right_sleeve_03",
        27: "right_sleeve_04",
        28: "right_sleeve_05",
    },

    # Manually verified by pure-index visualization.
    "long sleeve dress": {
        **_make_generic_names(37),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        7: "left_shoulder",
        37: "right_shoulder",

        9: "left_sleeve_01",
        10: "left_sleeve_02",
        11: "left_sleeve_03",
        12: "left_sleeve_04",
        13: "left_sleeve_05",
        14: "left_sleeve_06",

        18: "left_waist",
        25: "right_waist",

        21: "left_hem",
        22: "middle_hem",
        23: "right_hem",

        31: "right_sleeve_01",
        32: "right_sleeve_02",
        33: "right_sleeve_03",
        34: "right_sleeve_04",
        35: "right_sleeve_05",
    },

    # Manually verified by pure-index visualization.
    "vest dress": {
        **_make_generic_names(19),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        7: "left_shoulder",
        19: "right_shoulder",

        9: "left_waist_01",
        10: "left_waist_02",
        16: "right_waist_02",
        17: "right_waist_01",

        12: "left_hem",
        13: "middle_hem",
        14: "right_hem",
    },

    # Manually verified by pure-index visualization.
    "sling dress": {
        **_make_generic_names(19),
        1: "collar_01",
        2: "collar_02",
        3: "collar_03",
        4: "collar_04",
        5: "collar_05",
        6: "collar_06",

        7: "left_shoulder",
        19: "right_shoulder",

        10: "left_waist",
        16: "right_waist",

        12: "left_hem",
        13: "middle_hem",
        14: "right_hem",
    },
}


# Explicit category-aware index -> region mapping.
#
# This mapping overrides name-based region inference for categories whose
# index convention has been manually verified.
#
# Index convention: 1-based.
CATEGORY_TO_LANDMARK_REGIONS: Dict[str, Dict[int, str]] = {
    "short sleeve top": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",

        7: "shoulder",
        25: "shoulder",

        8: "sleeve",
        9: "sleeve",
        10: "sleeve",
        11: "sleeve",
        12: "sleeve",

        14: "waist",
        18: "waist",

        15: "hem",
        16: "hem",
        17: "hem",

        20: "sleeve",
        21: "sleeve",
        22: "sleeve",
        23: "sleeve",
        24: "sleeve",
    },

    "long sleeve top": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        7: "shoulder",
        33: "shoulder",

        8: "sleeve",
        9: "sleeve",
        10: "sleeve",
        11: "sleeve",
        12: "sleeve",
        13: "sleeve",

        18: "waist",
        22: "waist",

        19: "hem",
        20: "hem",
        21: "hem",

        26: "sleeve",
        27: "sleeve",
        28: "sleeve",
        29: "sleeve",
        30: "sleeve",
        31: "sleeve",
    },

    "short sleeve outwear": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        # User marked point 7 in both sleeve and shoulder.
        # For crop semantics, shoulder wins here.
        7: "shoulder",
        25: "shoulder",

        8: "sleeve",
        9: "sleeve",
        10: "sleeve",
        11: "sleeve",
        12: "sleeve",

        14: "waist",
        18: "waist",
        28: "waist",
        31: "waist",

        15: "hem",
        16: "hem",
        17: "hem",

        21: "sleeve",
        22: "sleeve",
        23: "sleeve",
        24: "sleeve",
    },

    "long sleeve outwear": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",
        34: "collar",

        7: "shoulder",
        33: "shoulder",

        9: "sleeve",
        10: "sleeve",
        11: "sleeve",
        12: "sleeve",
        13: "sleeve",
        14: "sleeve",

        18: "waist",
        22: "waist",
        36: "waist",
        39: "waist",

        19: "hem",
        20: "hem",
        21: "hem",
        37: "hem",

        26: "sleeve",
        27: "sleeve",
        28: "sleeve",
        29: "sleeve",
        30: "sleeve",
        31: "sleeve",
    },

    "vest": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        7: "shoulder",
        15: "shoulder",

        9: "waist",
        13: "waist",

        10: "hem",
        11: "hem",
        12: "hem",
    },

    "sling": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        7: "shoulder",
        15: "shoulder",

        9: "waist",
        13: "waist",

        10: "hem",
        11: "hem",
        12: "hem",
    },

    "short sleeve dress": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        7: "shoulder",
        29: "shoulder",

        8: "sleeve",
        9: "sleeve",
        10: "sleeve",
        11: "sleeve",
        12: "sleeve",

        15: "waist",
        21: "waist",

        17: "hem",
        18: "hem",
        19: "hem",

        24: "sleeve",
        25: "sleeve",
        26: "sleeve",
        27: "sleeve",
        28: "sleeve",
    },

    "long sleeve dress": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        7: "shoulder",
        37: "shoulder",

        9: "sleeve",
        10: "sleeve",
        11: "sleeve",
        12: "sleeve",
        13: "sleeve",
        14: "sleeve",

        18: "waist",
        25: "waist",

        21: "hem",
        22: "hem",
        23: "hem",

        31: "sleeve",
        32: "sleeve",
        33: "sleeve",
        34: "sleeve",
        35: "sleeve",
    },

    "vest dress": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        7: "shoulder",
        19: "shoulder",

        9: "waist",
        10: "waist",
        16: "waist",
        17: "waist",

        12: "hem",
        13: "hem",
        14: "hem",
    },

    "sling dress": {
        1: "collar",
        2: "collar",
        3: "collar",
        4: "collar",
        5: "collar",
        6: "collar",

        7: "shoulder",
        19: "shoulder",

        10: "waist",
        16: "waist",

        12: "hem",
        13: "hem",
        14: "hem",
    },
}


LANDMARK_NAME_TO_REGION: Dict[str, str] = {
    "left_collar": "collar",
    "right_collar": "collar",
    "left_neckline": "collar",
    "right_neckline": "collar",
    "neckline": "collar",

    "left_shoulder": "shoulder",
    "right_shoulder": "shoulder",

    "left_strap": "strap",
    "right_strap": "strap",

    "left_armpit": "armpit",
    "right_armpit": "armpit",

    "left_cuff": "cuff",
    "right_cuff": "cuff",

    "left_sleeve": "sleeve",
    "right_sleeve": "sleeve",
    "left_sleeve_outer": "sleeve",
    "left_sleeve_inner": "sleeve",
    "right_sleeve_outer": "sleeve",
    "right_sleeve_inner": "sleeve",
    "left_sleeve_upper_outer": "sleeve",
    "left_sleeve_upper_inner": "sleeve",
    "right_sleeve_upper_outer": "sleeve",
    "right_sleeve_upper_inner": "sleeve",

    # generic corrected names
    "collar_01": "collar",
    "collar_02": "collar",
    "collar_03": "collar",
    "collar_04": "collar",
    "collar_05": "collar",
    "collar_06": "collar",
    "collar_07": "collar",

    "left_sleeve_01": "sleeve",
    "left_sleeve_02": "sleeve",
    "left_sleeve_03": "sleeve",
    "left_sleeve_04": "sleeve",
    "left_sleeve_05": "sleeve",
    "left_sleeve_06": "sleeve",

    "right_sleeve_01": "sleeve",
    "right_sleeve_02": "sleeve",
    "right_sleeve_03": "sleeve",
    "right_sleeve_04": "sleeve",
    "right_sleeve_05": "sleeve",
    "right_sleeve_06": "sleeve",

    "left_waist_01": "waist",
    "left_waist_02": "waist",
    "right_waist_01": "waist",
    "right_waist_02": "waist",

    "left_hem": "hem",
    "right_hem": "hem",
    "middle_hem": "hem",
    "middle_hem_01": "hem",
    "middle_hem_02": "hem",
    "left_middle_hem": "hem",
    "right_middle_hem": "hem",
    "hem": "hem",

    "left_waist": "waist",
    "right_waist": "waist",
    "waist": "waist",

    "left_side": "side",
    "right_side": "side",

    "left_crotch": "crotch",
    "right_crotch": "crotch",

    "left_knee_outer": "knee",
    "left_knee_inner": "knee",
    "right_knee_outer": "knee",
    "right_knee_inner": "knee",

    "left_pant_leg": "pant_leg",
    "right_pant_leg": "pant_leg",
    "left_leg_opening": "pant_leg",
    "right_leg_opening": "pant_leg",
    "left_leg_opening_outer": "pant_leg",
    "left_leg_opening_inner": "pant_leg",
    "right_leg_opening_outer": "pant_leg",
    "right_leg_opening_inner": "pant_leg",
}


def normalize_category_name(category_name: Any) -> str:
    """
    Normalize category name for schema lookup.
    """
    if category_name is None:
        return ""

    name = str(category_name).strip().lower()
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name


def get_valid_indices_for_category(category_name: Any) -> Optional[Set[int]]:
    """
    Get valid landmark index set for category.

    Returns:
        Set of valid indices if configured.
        None if category has no configured schema.
    """
    normalized = normalize_category_name(category_name)
    return CATEGORY_TO_VALID_LANDMARK_INDICES.get(normalized)


def get_landmark_name(category_name: Any, index: int) -> Optional[str]:
    """
    Get semantic landmark name for category and index.
    """
    normalized = normalize_category_name(category_name)
    mapping = CATEGORY_TO_LANDMARK_NAMES.get(normalized, {})
    try:
        return mapping.get(int(index))
    except Exception:
        return None


def get_landmark_region(category_name: Any, index: int) -> Optional[str]:
    """
    Get explicit semantic region for category and landmark index.

    Explicit category-index mapping is preferred over name-based inference,
    because some provisional semantic names can be wrong.
    """
    normalized = normalize_category_name(category_name)
    mapping = CATEGORY_TO_LANDMARK_REGIONS.get(normalized, {})
    try:
        return mapping.get(int(index))
    except Exception:
        return None


def infer_region_from_name(name: Optional[str]) -> Optional[str]:
    """
    Infer region name from landmark semantic name.
    """
    if not name:
        return None

    normalized = normalize_category_name(name)

    if normalized in LANDMARK_NAME_TO_REGION:
        return LANDMARK_NAME_TO_REGION[normalized]

    if "collar" in normalized or "neckline" in normalized:
        return "collar"
    if "shoulder" in normalized:
        return "shoulder"
    if "strap" in normalized:
        return "strap"
    if "armpit" in normalized:
        return "armpit"
    if "cuff" in normalized:
        return "cuff"
    if "sleeve" in normalized:
        return "sleeve"
    if "hem" in normalized:
        return "hem"
    if "waist" in normalized:
        return "waist"
    if "crotch" in normalized:
        return "crotch"
    if "knee" in normalized:
        return "knee"
    if "leg" in normalized:
        return "pant_leg"
    if "side" in normalized:
        return "side"

    return None


def enrich_landmarks_with_schema(
    landmarks: List[Dict[str, Any]],
    category_name: Any,
    mark_invalid_quality: bool = False,
) -> List[Dict[str, Any]]:
    """
    Add category-aware schema fields to landmarks.

    Added fields:
        valid_for_class: bool
        name: str or None
        region: str or None

    Args:
        landmarks: Landmark list.
        category_name: Garment category name.
        mark_invalid_quality:
            If True, invalid landmarks get quality='invalid_for_class'.

    Returns:
        New landmark list.
    """
    valid_indices = get_valid_indices_for_category(category_name)

    output: List[Dict[str, Any]] = []

    for landmark in landmarks:
        item = dict(landmark)

        try:
            index = int(item.get("index"))
        except Exception:
            index = -1

        if valid_indices is None:
            valid_for_class = True
        else:
            valid_for_class = index in valid_indices

        name = get_landmark_name(category_name, index)

        # Prefer manually verified category-index region mapping.
        # Fall back to name-based inference only when no explicit mapping exists.
        region = get_landmark_region(category_name, index)
        if region is None:
            region = infer_region_from_name(name)

        item["valid_for_class"] = bool(valid_for_class)
        item["name"] = name
        item["region"] = region

        if not valid_for_class and mark_invalid_quality:
            item["quality_before_class_filter"] = item.get("quality")
            item["quality"] = "invalid_for_class"

        output.append(item)

    return output


def filter_landmarks_by_category(
    landmarks: List[Dict[str, Any]],
    category_name: Any,
    drop_invalid: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filter or mark landmarks by category.

    Args:
        landmarks: Landmark list.
        category_name: Garment category name.
        drop_invalid:
            If True, remove invalid landmarks.
            If False, keep all but annotate valid_for_class.

    Returns:
        Filtered or annotated landmarks.
    """
    enriched = enrich_landmarks_with_schema(
        landmarks=landmarks,
        category_name=category_name,
        mark_invalid_quality=False,
    )

    if not drop_invalid:
        return enriched

    return [lm for lm in enriched if bool(lm.get("valid_for_class", False))]
