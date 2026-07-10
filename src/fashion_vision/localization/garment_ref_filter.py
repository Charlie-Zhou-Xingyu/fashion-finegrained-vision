"""
Garment reference instance filter for 3.1.2 language-guided localization.

Maps a parsed garment_ref (外套/裙子/裤子/内搭/…) to the subset of detected
garment instances that match, so downstream DINO detection runs only on the
intended garment rather than every garment in the image.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maps parsed garment_ref values → DeepFashion2 fine class names.
# Kept in sync with CLAUDE.md Section 5 category mapping.
GARMENT_REF_TO_FINE_CLASSES: Dict[str, List[str]] = {
    "outerwear": ["short sleeve outwear", "long sleeve outwear"],
    "top":       ["short sleeve top", "long sleeve top", "vest", "sling"],
    "pants":     ["shorts", "trousers"],
    "skirt":     ["skirt"],
    "dress":     ["short sleeve dress", "long sleeve dress", "vest dress", "sling dress"],
    # "inner" has no class-based signal — handled by mask-area heuristic below.
}

# Coarse class token strings as they appear in garment instance records.
# "outwear" (no 'e') matches the DeepFashion2 spelling used across the codebase.
_OUTWEAR_CLASSES = frozenset(GARMENT_REF_TO_FINE_CLASSES["outerwear"])
_TOP_CLASSES     = frozenset(GARMENT_REF_TO_FINE_CLASSES["top"])
_PANTS_CLASSES   = frozenset(GARMENT_REF_TO_FINE_CLASSES["pants"])
_SKIRT_CLASSES   = frozenset(GARMENT_REF_TO_FINE_CLASSES["skirt"])
_DRESS_CLASSES   = frozenset(GARMENT_REF_TO_FINE_CLASSES["dress"])


def _class_name(instance: Dict[str, Any]) -> str:
    """Return the fine class name from an instance record (normalised to lower)."""
    raw = (
        instance.get("fine_class_name")
        or instance.get("class_name")
        or instance.get("category_name")
        or instance.get("category")
        or ""
    )
    return str(raw).strip().lower()


def _mask_area(instance: Dict[str, Any]) -> float:
    """Return a comparable mask area proxy (pixels or ratio)."""
    for key in ("mask_area", "mask_area_ratio", "area"):
        val = instance.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return float("inf")


def filter_instances(
    instances: List[Dict[str, Any]],
    garment_ref: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Return the subset of instances that match the parsed garment_ref.

    Rules:
    - ``garment_ref is None``: all instances returned unchanged.
    - ``garment_ref == "inner"``: all instances returned, sorted ascending by
      mask area (smallest ≈ inner layer).  Each record gets ``_inner_rank`` set.
    - Recognised garment_ref: filter by fine class name.  If the filtered list
      is empty, fall back to all instances and set ``_garment_ref_mismatch=True``
      on each record.
    - Unrecognised garment_ref: fall back to all instances with a warning.

    Args:
        instances: List of garment instance dicts from the pipeline.
        garment_ref: Parsed garment reference string or None.

    Returns:
        Filtered (or full-fallback) instance list.  Never empty.
    """
    if not instances:
        return instances

    if garment_ref is None:
        return instances

    if garment_ref == "inner":
        sorted_by_area = sorted(instances, key=_mask_area)
        for rank, inst in enumerate(sorted_by_area, start=1):
            inst["_inner_rank"] = rank
        return sorted_by_area

    target_classes = GARMENT_REF_TO_FINE_CLASSES.get(garment_ref)
    if target_classes is None:
        logger.warning(
            "garment_ref_filter: unrecognised garment_ref=%r — returning all instances",
            garment_ref,
        )
        return instances

    target_set = frozenset(c.lower() for c in target_classes)
    filtered = [inst for inst in instances if _class_name(inst) in target_set]

    if not filtered:
        logger.warning(
            "garment_ref_filter: no instance matched garment_ref=%r "
            "(classes in image: %s) — falling back to all instances",
            garment_ref,
            [_class_name(i) for i in instances],
        )
        for inst in instances:
            inst["_garment_ref_mismatch"] = True
        return instances

    return filtered


def garment_ref_to_target_class(garment_ref: Optional[str]) -> str:
    """
    Map a parsed garment_ref to the ``target_class`` string expected by
    ``select_best_record()`` / ``target_class_matches()`` in the fast path.

    Special sentinel values (``__dress__``, ``__outwear__``, etc.) are handled
    by ``target_class_matches()`` in the demo.

    Args:
        garment_ref: Parsed garment reference or None.

    Returns:
        Target class string (empty string = no filter).
    """
    _MAP: Dict[Optional[str], str] = {
        None:        "",
        "outerwear": "__outwear__",
        "top":       "__top__",
        "pants":     "__pants__",
        "skirt":     "skirt",
        "dress":     "__dress__",
        "inner":     "",   # no class filter; handled by mask-area sort
    }
    return _MAP.get(garment_ref, "")


if __name__ == "__main__":
    _instances = [
        {"class_name": "long sleeve top",     "mask_area": 5000},
        {"class_name": "long sleeve outwear", "mask_area": 8000},
        {"class_name": "trousers",            "mask_area": 4000},
    ]

    result = filter_instances(_instances, "outerwear")
    assert len(result) == 1 and result[0]["class_name"] == "long sleeve outwear", result

    result = filter_instances(_instances, "pants")
    assert len(result) == 1 and result[0]["class_name"] == "trousers", result

    result = filter_instances(_instances, "inner")
    assert result[0]["class_name"] == "trousers"  # smallest area first
    assert result[0]["_inner_rank"] == 1

    result = filter_instances(_instances, None)
    assert len(result) == 3

    result = filter_instances(_instances, "skirt")  # no match → fallback
    assert len(result) == 3
    assert all(r.get("_garment_ref_mismatch") for r in result)

    assert garment_ref_to_target_class("outerwear") == "__outwear__"
    assert garment_ref_to_target_class(None) == ""
    assert garment_ref_to_target_class("dress") == "__dress__"

    print("garment_ref_filter smoke test passed.")
