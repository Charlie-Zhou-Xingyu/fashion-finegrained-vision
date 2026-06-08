"""
Rule-based query parser for fashion local region localization.

This module maps natural language queries to canonical region types.
The first baseline uses keyword matching and supports both Chinese and English.
"""

from __future__ import annotations

from typing import Dict, List


REGION_KEYWORDS: Dict[str, List[str]] = {
    "neckline": [
        "领口",
        "衣领",
        "领子",
        "领部",
        "neckline",
        "collar",
        "neck",
    ],
    "cuff": [
        "袖口",
        "袖子末端",
        "袖边",
        "袖口处",
        "cuff",
        "sleeve cuff",
        "sleeve end",
    ],
    "hem": [
        "下摆",
        "衣摆",
        "裙摆",
        "底边",
        "下边缘",
        "hem",
        "bottom",
        "lower edge",
    ],
    "waist": [
        "腰部",
        "腰线",
        "收腰",
        "腰头",
        "waist",
        "waistline",
    ],
    "shoulder": [
        "肩部",
        "肩膀",
        "肩线",
        "shoulder",
    ],
    "leg_opening": [
        "裤脚",
        "裤口",
        "裤腿末端",
        "脚口",
        "leg opening",
        "pant hem",
        "trouser hem",
    ],
}


REGION_ALIASES: Dict[str, str] = {
    "collar": "neckline",
    "neck": "neckline",
    "bottom": "hem",
    "pant_hem": "leg_opening",
    "trouser_hem": "leg_opening",
}


def normalize_query(query: str) -> str:
    """
    Normalize a query string.

    Args:
        query: Raw query.

    Returns:
        Normalized query.
    """
    return query.lower().strip()


def parse_region_type(query: str) -> str:
    """
    Parse canonical region type from query.

    Args:
        query: Natural language query.

    Returns:
        Canonical region type. Returns "unknown" if no match.
    """
    normalized = normalize_query(query)

    for region_type, keywords in REGION_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in normalized:
                return region_type

    for alias, region_type in REGION_ALIASES.items():
        if alias in normalized:
            return region_type

    return "unknown"


def is_supported_region(region_type: str) -> bool:
    """
    Check whether region type is supported.

    Args:
        region_type: Region type.

    Returns:
        Whether supported.
    """
    return region_type in REGION_KEYWORDS
