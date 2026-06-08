"""
Landmark-to-region mapping for DeepFashion2 local region localization.

This file defines manually inspected mapping rules from DeepFashion2 garment
landmark indices to semantic local regions, such as neckline, cuff, hem,
waist, shoulder, and leg opening.

Important design:
    - "indices" is the flat list of landmarks for a region.
    - "groups" is optional and is used for paired regions, e.g. left/right cuff.
    - For cuff / leg_opening, prefer "groups" to avoid producing one large bbox
      covering both sides.

Notes:
    - Landmark index is assumed to be 1-based, matching the visualization.
    - visibility == 2 means visible.
    - visibility == 1 means occluded but present.
    - visibility == 0 means absent and should not be used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


LandmarkRegionRule = Dict[str, Any]


LANDMARK_REGION_MAP: Dict[str, Dict[str, LandmarkRegionRule]] = {
    # ---------------------------------------------------------------------
    # Tops
    # ---------------------------------------------------------------------
    "short sleeve top": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Upper neckline points. User note: 3,4,5,6,2 with 1 optional.",
        },
        "cuff": {
            "indices": [8, 9, 10, 11, 12, 20, 21, 22, 23],
            "groups": {
                "left": [8, 9, 10, 11, 12],
                "right": [20, 21, 22, 23],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve cuff. Left/right groups prevent one huge bbox across body.",
        },
        "hem": {
            "indices": [15, 16, 17],
            "confidence": "high",
            "supported": True,
            "notes": "Bottom hem points.",
        },
        "shoulder": {
            "indices": [2, 5, 7, 25],
            "groups": {
                "left": [2, 7],
                "right": [5, 25],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Shoulder points grouped left/right to avoid V-shaped cross-body region.",
        },
        "waist": {
            "indices": [14, 18],
            "confidence": "high",
            "supported": True,
            "notes": "Lower body / waist-like points for top.",
        },
    },

    "long sleeve top": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Upper neckline points.",
        },
        "cuff": {
            "indices": [10, 11, 12, 13, 27, 28, 29, 30],
            "groups": {
                "left": [10, 11, 12, 13],
                "right": [27, 28, 29, 30],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve cuff points grouped left/right.",
        },
        "hem": {
            "indices": [19, 20, 21],
            "confidence": "high",
            "supported": True,
            "notes": "Bottom hem points.",
        },
        "shoulder": {
            "indices": [7, 33],
            "groups": {
                "left": [7],
                "right": [33],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Left/right shoulder points.",
        },
        "waist": {
            "indices": [18, 22],
            "confidence": "medium",
            "supported": True,
            "notes": "Waist/lower side points. Medium confidence.",
        },
    },

    "vest": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Vest neckline points.",
        },
        "cuff": {
            "indices": [],
            "groups": {},
            "confidence": "high",
            "supported": False,
            "notes": "Sleeveless garment. Cuff is unsupported.",
        },
        "hem": {
            "indices": [10, 11, 12],
            "confidence": "high",
            "supported": True,
            "notes": "Vest bottom hem points.",
        },
        "waist": {
            "indices": [8, 9, 13, 14],
            "confidence": "high",
            "supported": True,
            "notes": "Vest waist/lower side points.",
        },
        "shoulder": {
            "indices": [2, 6, 7, 15],
            "groups": {
                "left": [2, 7],
                "right": [6, 15],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Vest shoulder / armhole upper points grouped left/right.",
        },
    },

    "sling": {
        "neckline": {
            "indices": [2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Sling neckline points.",
        },
        "cuff": {
            "indices": [],
            "groups": {},
            "confidence": "high",
            "supported": False,
            "notes": "Sleeveless garment. Cuff is unsupported.",
        },
        "hem": {
            "indices": [10, 11, 12],
            "confidence": "high",
            "supported": True,
            "notes": "Sling bottom hem points.",
        },
        "waist": {
            "indices": [9, 13],
            "confidence": "high",
            "supported": True,
            "notes": "Sling waist/lower side points.",
        },
        "shoulder": {
            "indices": [7, 15],
            "groups": {
                "left": [7],
                "right": [15],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Sling strap / shoulder points.",
        },
    },

    # ---------------------------------------------------------------------
    # Outwear
    # ---------------------------------------------------------------------
    "short sleeve outwear": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve outwear neckline points.",
        },
        "cuff": {
            "indices": [9, 10, 22, 23],
            "groups": {
                "left": [9, 10],
                "right": [22, 23],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve outwear cuff points grouped left/right.",
        },
        "hem": {
            "indices": [15, 16, 17, 29],
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve outwear hem points.",
        },
        "shoulder": {
            "indices": [7, 25],
            "groups": {
                "left": [7],
                "right": [25],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve outwear shoulder points.",
        },
        "waist": {
            "indices": [14, 18, 28, 31],
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve outwear waist/lower side points.",
        },
    },

    "long sleeve outwear": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5],
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve outwear neckline points.",
        },
        "cuff": {
            "indices": [9, 10, 11, 12, 13, 14, 26, 27, 28, 29, 30, 31],
            "groups": {
                "left": [9, 10, 11, 12, 13, 14],
                "right": [26, 27, 28, 29, 30, 31],
            },
            "confidence": "medium",
            "supported": True,
            "notes": "Long sleeve outwear cuff points grouped left/right.",
        },
        "hem": {
            "indices": [19, 20, 21, 37],
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve outwear hem points.",
        },
        "shoulder": {
            "indices": [7, 33],
            "groups": {
                "left": [7],
                "right": [33],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve outwear shoulder points.",
        },
        "waist": {
            "indices": [9, 18, 35, 36, 38, 39],
            "confidence": "medium",
            "supported": True,
            "notes": "Long sleeve outwear waist / middle-body points. Medium confidence.",
        },
    },

    # ---------------------------------------------------------------------
    # Dresses
    # ---------------------------------------------------------------------
    "short sleeve dress": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve dress neckline points.",
        },
        "cuff": {
            "indices": [9, 10, 26, 27],
            "groups": {
                "left": [9, 10],
                "right": [26, 27],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve dress cuff terminal points only. Avoid using full sleeve contour." ,
        },
        "hem": {
            "indices": [17, 18, 19],
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve dress hem points.",
        },
        "waist": {
            "indices": [14, 15, 21, 22],
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve dress waist points.",
        },
        "shoulder": {
            "indices": [7, 29],
            "groups": {
                "left": [7],
                "right": [29],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Short sleeve dress shoulder points.",
        },
    },

    "long sleeve dress": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve dress neckline points.",
        },
        "cuff": {
            "indices": [9, 10, 11, 12, 13, 31, 32, 33, 34],
            "groups": {
                "left": [9, 10, 11, 12, 13],
                "right": [31, 32, 33, 34],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve dress cuff points grouped left/right.",
        },
        "hem": {
            "indices": [21, 22, 23],
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve dress bottom hem points.",
        },
        "waist": {
            "indices": [18, 19, 25, 26],
            "confidence": "high",
            "supported": True,
            "notes": "Long sleeve dress waist points.",
        },
        "shoulder": {
            "indices": [2, 6, 7, 37],
            "groups": {
                "left": [2, 7],
                "right": [6, 37],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Shoulder lines: 2-7 and 6-37 according to user notes.",
        },
    },

    "vest dress": {
        "neckline": {
            "indices": [1, 2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Vest dress neckline points.",
        },
        "cuff": {
            "indices": [],
            "groups": {},
            "confidence": "high",
            "supported": False,
            "notes": "Sleeveless dress. Cuff is unsupported.",
        },
        "hem": {
            "indices": [12, 13, 14],
            "confidence": "high",
            "supported": True,
            "notes": "Vest dress hem points.",
        },
        "waist": {
            "indices": [9, 10, 16, 17],
            "confidence": "high",
            "supported": True,
            "notes": "Vest dress waist points.",
        },
        "shoulder": {
            "indices": [6, 7, 12, 19],
            "groups": {
                "left": [6, 7],
                "right": [12, 19],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Vest dress shoulder / upper side points grouped left/right.",
        },
    },

    "sling dress": {
        "neckline": {
            "indices": [2, 3, 4, 5, 6],
            "confidence": "high",
            "supported": True,
            "notes": "Sling dress neckline points.",
        },
        "cuff": {
            "indices": [],
            "groups": {},
            "confidence": "high",
            "supported": False,
            "notes": "Sleeveless dress. Cuff is unsupported.",
        },
        "hem": {
            "indices": [12, 13, 14],
            "confidence": "high",
            "supported": True,
            "notes": "Sling dress hem points.",
        },
        "waist": {
            "indices": [3, 10, 16, 17],
            "confidence": "high",
            "supported": True,
            "notes": "Sling dress waist points. Includes index 3 from user note.",
        },
        "shoulder": {
            "indices": [7, 19],
            "groups": {
                "left": [7],
                "right": [19],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Sling dress shoulder / strap points.",
        },
    },

    # ---------------------------------------------------------------------
    # Pants / skirt
    # ---------------------------------------------------------------------
    "trousers": {
        "waist": {
            "indices": [1, 2, 3],
            "confidence": "high",
            "supported": True,
            "notes": "Trouser waist upper edge.",
        },
        "leg_opening": {
            "indices": [6, 7, 11, 12],
            "groups": {
                "left": [6, 7],
                "right": [11, 12],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Trouser leg openings grouped left/right.",
        },
        "hem": {
            "indices": [6, 7, 11, 12],
            "groups": {
                "left": [6, 7],
                "right": [11, 12],
            },
            "confidence": "high",
            "supported": True,
            "notes": "For trousers, hem is treated as leg opening.",
        },
    },

    "shorts": {
        "waist": {
            "indices": [1, 2, 3],
            "confidence": "high",
            "supported": True,
            "notes": "Shorts waist upper edge.",
        },
        "leg_opening": {
            "indices": [5, 6, 8, 9],
            "groups": {
                "left": [5, 6],
                "right": [8, 9],
            },
            "confidence": "high",
            "supported": True,
            "notes": "Shorts leg openings grouped left/right.",
        },
        "hem": {
            "indices": [5, 6, 8, 9],
            "groups": {
                "left": [5, 6],
                "right": [8, 9],
            },
            "confidence": "high",
            "supported": True,
            "notes": "For shorts, hem is treated as leg opening.",
        },
    },

    "skirt": {
        "waist": {
            "indices": [1, 2, 3],
            "confidence": "high",
            "supported": True,
            "notes": "Skirt waist upper edge.",
        },
        "hem": {
            "indices": [4, 5, 6, 7, 8],
            "confidence": "high",
            "supported": True,
            "notes": "Skirt hem. User note: 5,6,7 with 4,8 optional.",
        },
    },
}


SLEEVELESS_RAW_CATEGORIES = {
    "vest",
    "sling",
    "vest dress",
    "sling dress",
}


def normalize_raw_category_name(raw_category_name: str) -> str:
    """
    Normalize raw category name.

    Args:
        raw_category_name: Raw category name.

    Returns:
        Normalized raw category name.
    """
    return raw_category_name.lower().strip().replace("_", " ")


def get_landmark_region_rule(
    raw_category_name: str,
    region_type: str,
) -> Optional[LandmarkRegionRule]:
    """
    Get landmark region rule by raw category name and region type.

    Args:
        raw_category_name: Raw DeepFashion2 category name.
        region_type: Local region type.

    Returns:
        Rule dict or None.
    """
    normalized_category = normalize_raw_category_name(raw_category_name)
    normalized_region = region_type.lower().strip()

    category_rules = LANDMARK_REGION_MAP.get(normalized_category)
    if category_rules is None:
        return None

    return category_rules.get(normalized_region)


def get_flat_indices_from_rule(rule: LandmarkRegionRule) -> List[int]:
    """
    Get flattened landmark indices from a rule.

    Args:
        rule: Landmark region rule.

    Returns:
        Flattened unique landmark indices.
    """
    indices: List[int] = []

    raw_indices = rule.get("indices", [])
    if isinstance(raw_indices, list):
        indices.extend([int(index) for index in raw_indices])

    groups = rule.get("groups", {})
    if isinstance(groups, dict):
        for group_indices in groups.values():
            if isinstance(group_indices, list):
                indices.extend([int(index) for index in group_indices])

    return sorted(set(indices))


def is_region_supported_by_landmarks(
    raw_category_name: str,
    region_type: str,
) -> bool:
    """
    Check whether a region is supported by landmark rule.

    Args:
        raw_category_name: Raw DeepFashion2 category name.
        region_type: Local region type.

    Returns:
        Whether supported.
    """
    rule = get_landmark_region_rule(
        raw_category_name=raw_category_name,
        region_type=region_type,
    )

    if rule is None:
        return False

    if not bool(rule.get("supported", True)):
        return False

    return len(get_flat_indices_from_rule(rule)) > 0


def get_landmark_indices_for_region(
    raw_category_name: str,
    region_type: str,
) -> List[int]:
    """
    Get flattened landmark indices for a local region.

    Args:
        raw_category_name: Raw DeepFashion2 category name.
        region_type: Local region type.

    Returns:
        Landmark index list.
    """
    rule = get_landmark_region_rule(
        raw_category_name=raw_category_name,
        region_type=region_type,
    )

    if rule is None:
        return []

    if not bool(rule.get("supported", True)):
        return []

    return get_flat_indices_from_rule(rule)


def get_landmark_groups_for_region(
    raw_category_name: str,
    region_type: str,
) -> Dict[str, List[int]]:
    """
    Get grouped landmark indices for a local region.

    Args:
        raw_category_name: Raw DeepFashion2 category name.
        region_type: Local region type.

    Returns:
        Group name to landmark indices.
    """
    rule = get_landmark_region_rule(
        raw_category_name=raw_category_name,
        region_type=region_type,
    )

    if rule is None:
        return {}

    if not bool(rule.get("supported", True)):
        return {}

    groups = rule.get("groups", {})
    if not isinstance(groups, dict):
        return {}

    normalized_groups: Dict[str, List[int]] = {}

    for group_name, group_indices in groups.items():
        if not isinstance(group_indices, list):
            continue

        normalized_groups[str(group_name)] = [int(index) for index in group_indices]

    return normalized_groups


def is_sleeveless_category(raw_category_name: str) -> bool:
    """
    Check whether the raw category is sleeveless.

    Args:
        raw_category_name: Raw category name.

    Returns:
        Whether sleeveless.
    """
    normalized = normalize_raw_category_name(raw_category_name)
    return normalized in SLEEVELESS_RAW_CATEGORIES
