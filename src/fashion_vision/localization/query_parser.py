"""
Backward-compatible region type parser.

Delegates to intent_parser.parse_intent() and returns the canonical internal
part name (neckline, cuff, hem, waist, shoulder, leg_opening).

New code should use intent_parser.parse_intent() directly.
"""
from __future__ import annotations

from typing import Dict, List

from fashion_vision.localization.intent_parser import (
    FAST_PATH_PARTS,
    PART_VOCAB,
    parse_intent,
)


# Re-exported for callers that import these names directly.
REGION_KEYWORDS: Dict[str, List[str]] = {
    k: v for k, v in PART_VOCAB.items() if k in FAST_PATH_PARTS
}

REGION_ALIASES: Dict[str, str] = {
    "collar": "neckline",
    "neck": "neckline",
    "bottom": "hem",
    "pant_hem": "leg_opening",
    "trouser_hem": "leg_opening",
}


def normalize_query(query: str) -> str:
    """Normalize a query string."""
    return query.lower().strip()


def parse_region_type(query: str) -> str:
    """
    Return canonical region type for a query, or 'unknown'.

    Delegates to parse_intent().  Prefer parse_intent() for new code.

    Args:
        query: Natural language region query.

    Returns:
        Canonical region type (neckline/cuff/hem/waist/shoulder/leg_opening)
        or 'unknown' if the query cannot be matched.
    """
    intent = parse_intent(query)
    if intent.part is None or intent.part not in FAST_PATH_PARTS:
        return "unknown"
    return intent.part


def is_supported_region(region_type: str) -> bool:
    """Check whether a region type is handled by the landmark pipeline."""
    return region_type in FAST_PATH_PARTS
