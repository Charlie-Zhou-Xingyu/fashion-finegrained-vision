"""
Tests for locate_region() routing behavior — no GPU required.

Covers Phase 1.5: garment_ref mismatch detection wired into locate_region().

The fast-path with a minimal instance (no mask path) returns a failed/missing
status — that is expected and correct for these tests.  We only inspect the
garment_ref_matched field, which must be set correctly regardless of whether
the underlying localization succeeded.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2", reason="cv2 not available in test env — router imports cv2 at module level")


# Minimal 10×10 BGR image — no real content needed for structural tests.
_BLANK_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)


def _make_instance(class_name: str, category: str = "") -> dict:
    return {
        "class_name": class_name,
        "category": category or class_name,
        "instance_id": "test_inst_0",
        # no mask_path → fast-path fails gracefully with missing_mask_path
    }


# ── Phase 1.5: garment_ref mismatch detection ──────────────────────────────────

def test_garment_ref_mismatch_flagged_when_class_wrong() -> None:
    """garment_ref=outerwear but instance is a top → garment_ref_matched=False."""
    from fashion_vision.localization.region_localization_router import locate_region

    instance = _make_instance("short sleeve top", "top")
    # "外套的领口" → garment_ref="outerwear", part="neckline", is_fast_path=True
    result = locate_region(
        query="外套的领口",
        instance=instance,
        image=_BLANK_IMAGE,
        image_width=10,
        image_height=10,
    )
    assert result["garment_ref_matched"] is False, (
        f"Expected garment_ref_matched=False for top instance with outerwear ref, "
        f"got {result.get('garment_ref_matched')!r}"
    )


def test_garment_ref_mismatch_sets_flag_on_instance() -> None:
    """Mismatch detection also sets _garment_ref_mismatch on the instance dict."""
    from fashion_vision.localization.region_localization_router import locate_region

    instance = _make_instance("short sleeve top", "top")
    locate_region(
        query="外套的领口",
        instance=instance,
        image=_BLANK_IMAGE,
        image_width=10,
        image_height=10,
    )
    assert instance.get("_garment_ref_mismatch") is True


def test_garment_ref_matched_when_class_correct() -> None:
    """garment_ref=outerwear and instance is outwear → garment_ref_matched=True."""
    from fashion_vision.localization.region_localization_router import locate_region

    # DeepFashion2 spelling is "outwear" (no 'e')
    instance = _make_instance("long sleeve outwear", "outwear")
    result = locate_region(
        query="外套的领口",
        instance=instance,
        image=_BLANK_IMAGE,
        image_width=10,
        image_height=10,
    )
    assert result["garment_ref_matched"] is True


def test_garment_ref_none_no_mismatch_check() -> None:
    """No garment_ref in query → garment_ref_matched=True regardless of instance class."""
    from fashion_vision.localization.region_localization_router import locate_region

    instance = _make_instance("short sleeve top", "top")
    # "领口" → garment_ref=None, part="neckline", is_fast_path=True
    result = locate_region(
        query="领口",
        instance=instance,
        image=_BLANK_IMAGE,
        image_width=10,
        image_height=10,
    )
    assert result["garment_ref_matched"] is True


def test_garment_ref_unknown_class_no_mismatch_flag() -> None:
    """If instance has no recognisable class, mismatch check is skipped."""
    from fashion_vision.localization.region_localization_router import locate_region

    # No class_name / category → _class_name() returns ""
    instance = {"instance_id": "test_inst_1"}
    result = locate_region(
        query="外套的领口",
        instance=instance,
        image=_BLANK_IMAGE,
        image_width=10,
        image_height=10,
    )
    # Cannot determine class → no mismatch flag, defaults to True
    assert result["garment_ref_matched"] is True
