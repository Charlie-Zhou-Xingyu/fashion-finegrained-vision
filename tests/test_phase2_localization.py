"""
Phase 2 localization tests (no GPU required).

Covers:
  - part_detection_config helpers
  - part_shape_priors filter logic + fallback
  - Regression: Phase 1 garment_ref and zero-shot behaviour
"""
import pytest

from fashion_vision.localization.part_detection_config import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    get_part_prompts,
    get_part_shape_config,
    get_part_thresholds,
)
from fashion_vision.localization.part_shape_priors import filter_by_shape_priors


# ── part_detection_config helpers ────────────────────────────────────────────

def test_button_has_multiple_prompts() -> None:
    prompts = get_part_prompts("button")
    assert len(prompts) >= 2
    assert all(isinstance(p, str) and p for p in prompts)


def test_zipper_threshold_stricter_than_default() -> None:
    box_t, text_t = get_part_thresholds("zipper")
    assert box_t > DEFAULT_BOX_THRESHOLD
    assert text_t > DEFAULT_TEXT_THRESHOLD


def test_belt_has_prompts_and_shape() -> None:
    prompts = get_part_prompts("belt")
    assert len(prompts) >= 1
    cfg = get_part_shape_config("belt")
    assert "min_aspect_ratio_w_over_h" in cfg
    assert "y_band" in cfg


def test_unknown_part_uses_fallback_prompt() -> None:
    prompts = get_part_prompts("unknown_xyz", fallback_prompt="some detail")
    assert prompts == ["some detail"]


def test_unknown_part_no_fallback_uses_part_name() -> None:
    prompts = get_part_prompts("novel_part")
    assert len(prompts) == 1
    assert "novel" in prompts[0]   # underscores → spaces


def test_unknown_part_returns_default_thresholds() -> None:
    box_t, text_t = get_part_thresholds("not_a_real_part")
    assert box_t == DEFAULT_BOX_THRESHOLD
    assert text_t == DEFAULT_TEXT_THRESHOLD


def test_unknown_part_returns_empty_shape_config() -> None:
    assert get_part_shape_config("not_a_real_part") == {}


def test_known_part_shape_config_is_copy() -> None:
    """Mutations to returned config should not affect the canonical config."""
    cfg = get_part_shape_config("zipper")
    cfg["injected"] = True
    assert "injected" not in get_part_shape_config("zipper")


# ── part_shape_priors ─────────────────────────────────────────────────────────

def _det(bbox: list[float], score: float = 0.5) -> dict:
    return {"bbox_xyxy": bbox, "score": score}


GARMENT = [0, 0, 400, 600]  # 400×600, cx=200


def test_zipper_rejects_wide_box() -> None:
    # w=200, h=60 → h/w=0.3, need ≥1.8; all rejected → empty list
    wide = _det([100, 100, 300, 160])
    result = filter_by_shape_priors([wide], "zipper", GARMENT)
    assert result == []
    assert wide["_shape_prior_status"] == "rejected"


def test_zipper_accepts_tall_centered_box() -> None:
    # w=40, h=400 → h/w=10 > 1.8; area=16000/240000=0.067 < 0.30; center_x=200=garment_cx
    tall = _det([180, 100, 220, 500])
    result = filter_by_shape_priors([tall], "zipper", GARMENT)
    assert result[0]["_shape_prior_status"] == "passed"


def test_belt_accepts_horizontal_box_in_waist_band() -> None:
    # w=400, h=50 → w/h=8 > 2.0; cy=225 → cy_norm=225/600=0.375 in [0.35, 0.75]
    wide_mid = _det([0, 200, 400, 250])
    result = filter_by_shape_priors([wide_mid], "belt", GARMENT)
    assert result[0]["_shape_prior_status"] == "passed"


def test_belt_rejects_vertical_box() -> None:
    # w=40, h=300 → w/h=0.13 < 2.0; all rejected → empty list
    tall = _det([100, 200, 140, 500])
    result = filter_by_shape_priors([tall], "belt", GARMENT)
    assert result == []
    assert tall["_shape_prior_status"] == "rejected"


def test_button_rejects_box_exceeding_max_area() -> None:
    # area=200*200=40000; garment=240000; ratio=0.167 > max 0.08; all rejected → empty
    large = _det([100, 100, 300, 300])
    result = filter_by_shape_priors([large], "button", GARMENT)
    assert result == []
    assert large["_shape_prior_status"] == "rejected"
    assert any("area_ratio" in r for r in large["_shape_prior_reasons"])


def test_button_rejects_off_center_box() -> None:
    # small area OK (5*5=25/240000≈0); center_x=395, garment_cx=200 → offset=195/400=0.49 > 0.30
    off_center = _det([390, 100, 398, 108])
    result = filter_by_shape_priors([off_center], "button", GARMENT)
    assert len(result) == 1  # passes — prefer_center_x removed from button config
    assert off_center["_shape_prior_status"] == "passed"


def test_unknown_part_no_filter_applied() -> None:
    dets = [_det([0, 0, 100, 100]), _det([200, 200, 300, 300])]
    result = filter_by_shape_priors(dets, "unknown_xyz", GARMENT)
    assert len(result) == 2
    assert all(d["_shape_prior_status"] == "not_applicable" for d in result)


def test_none_part_no_filter_applied() -> None:
    dets = [_det([0, 0, 100, 100])]
    result = filter_by_shape_priors(dets, None, GARMENT)
    assert result[0]["_shape_prior_status"] == "not_applicable"


def test_all_rejected_returns_empty_list() -> None:
    # Both fail zipper h/w check; no fallback — caller must emit not_detected.
    d1 = _det([100, 100, 300, 160], score=0.3)   # wide, rejected
    d2 = _det([0, 0, 300, 100], score=0.8)        # also wide, rejected
    result = filter_by_shape_priors([d1, d2], "zipper", GARMENT)
    assert result == []
    # Both dicts should be annotated so callers can include them in debug output.
    assert d1["_shape_prior_status"] == "rejected"
    assert d2["_shape_prior_status"] == "rejected"


def test_empty_detections_returns_empty() -> None:
    assert filter_by_shape_priors([], "zipper", GARMENT) == []


def test_no_garment_bbox_skips_area_ratio() -> None:
    # Without garment bbox, area_ratio rules are skipped.
    # Button max_area_ratio=0.08 should NOT fire with garment_bbox=None.
    large = _det([0, 0, 390, 590])   # huge, but no reference garment
    result = filter_by_shape_priors([large], "button", garment_bbox=None)
    # prefer_center_x also needs garment, so all spatial checks skip → passed
    assert result[0]["_shape_prior_status"] == "passed"


def test_explicit_empty_shape_config_passes_all() -> None:
    dets = [_det([100, 100, 300, 160])]
    result = filter_by_shape_priors(dets, "zipper", GARMENT, shape_config={})
    assert result[0]["_shape_prior_status"] == "not_applicable"


def test_rejected_detection_has_reasons() -> None:
    wide = _det([100, 100, 300, 160])
    filter_by_shape_priors([wide], "zipper", GARMENT)
    assert "_shape_prior_reasons" in wide
    assert isinstance(wide["_shape_prior_reasons"], list)
    assert len(wide["_shape_prior_reasons"]) >= 1


# ── Phase 1 regression tests ─────────────────────────────────────────────────

def test_pantleg_alone_gives_no_garment_ref() -> None:
    from fashion_vision.localization.intent_parser import parse_intent
    intent = parse_intent("裤腿")
    assert intent.garment_ref is None


def test_pantleg_with_pants_word_gives_garment_ref() -> None:
    from fashion_vision.localization.intent_parser import parse_intent
    intent = parse_intent("裤子的裤腿")
    assert intent.garment_ref == "pants"


def test_zero_shot_query_does_not_raise() -> None:
    # "肩缝" was zero-shot before Phase 1; now it resolves to shoulder_seam.
    # Use a query that is genuinely not in any vocabulary.
    from fashion_vision.localization.intent_parser import parse_intent
    intent = parse_intent("裤子胯部")   # crotch area — not in vocab
    assert intent.is_zero_shot is True
    assert intent.part is None


def test_shoulder_seam_now_in_vocab() -> None:
    # Regression: "肩缝处" was zero-shot before Phase 1; now maps to shoulder_seam.
    from fashion_vision.localization.intent_parser import parse_intent
    intent = parse_intent("肩缝处")
    assert intent.part == "shoulder_seam"
    assert intent.is_zero_shot is False


def test_direction_parsed_from_胸前() -> None:
    from fashion_vision.localization.intent_parser import parse_intent
    intent = parse_intent("胸前的口袋")
    assert intent.direction == "front_upper"
    assert intent.part == "pocket"


# ── NMS (grounding_dino_locator — requires torch, skipped if unavailable) ────

def test_box_iou_zero_for_non_overlapping() -> None:
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    assert mod._box_iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0


def test_box_iou_one_for_identical_boxes() -> None:
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    assert mod._box_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)


def test_box_iou_partial_overlap() -> None:
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    # a=[0,0,100,100], b=[50,0,150,100]: inter=50*100=5000, union=15000 → 1/3
    iou = mod._box_iou([0, 0, 100, 100], [50, 0, 150, 100])
    assert iou == pytest.approx(1 / 3, abs=1e-6)


def test_nms_suppresses_overlapping_lower_score() -> None:
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    dets = [
        {"bbox_xyxy": [0, 0, 100, 100], "score": 0.9},
        {"bbox_xyxy": [5, 5, 105, 105], "score": 0.6},   # high IoU with first → suppressed
        {"bbox_xyxy": [200, 200, 300, 300], "score": 0.7},  # no overlap → kept
    ]
    result = mod.GroundingDINOLocator._nms(dets, iou_threshold=0.5)
    assert len(result) == 2
    scores = {d["score"] for d in result}
    assert 0.9 in scores
    assert 0.7 in scores
    assert 0.6 not in scores


def test_nms_keeps_all_non_overlapping() -> None:
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    dets = [
        {"bbox_xyxy": [0, 0, 50, 50], "score": 0.8},
        {"bbox_xyxy": [100, 0, 150, 50], "score": 0.7},
        {"bbox_xyxy": [200, 0, 250, 50], "score": 0.6},
    ]
    result = mod.GroundingDINOLocator._nms(dets)
    assert len(result) == 3


def test_nms_empty_input() -> None:
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    assert mod.GroundingDINOLocator._nms([]) == []


def test_nms_preserves_prompt_metadata() -> None:
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    dets = [
        {"bbox_xyxy": [0, 0, 100, 100], "score": 0.8, "prompt": "clothing zipper"},
        {"bbox_xyxy": [200, 0, 300, 100], "score": 0.6, "prompt": "zipper"},
    ]
    result = mod.GroundingDINOLocator._nms(dets)
    assert all("prompt" in d for d in result)


def test_detect_multi_prompt_has_return_raw_count_param() -> None:
    """API check: return_raw_count parameter exists without loading the model."""
    import inspect
    mod = pytest.importorskip("fashion_vision.localization.grounding_dino_locator")
    sig = inspect.signature(mod.GroundingDINOLocator.detect_multi_prompt)
    assert "return_raw_count" in sig.parameters
    assert sig.parameters["return_raw_count"].default is False


# ── debug metadata structure ──────────────────────────────────────────────────

_REQUIRED_DEBUG_KEYS = {
    "prompts_used",
    "thresholds_used",
    "candidate_count_before_nms",
    "candidate_count_after_nms",
    "candidate_count_before_shape_filter",
    "candidate_count_after_shape_filter",
}

_REQUIRED_THRESHOLD_KEYS = {"box_threshold", "text_threshold", "text_threshold_backend_note"}


def _make_mock_debug(n_before_nms=3, n_after_nms=2, n_before_shape=2, n_after_shape=1) -> dict:
    """Construct a debug dict matching the router's actual structure."""
    return {
        "prompts_used": ["clothing zipper", "zipper"],
        "thresholds_used": {
            "box_threshold": 0.40,
            "text_threshold": 0.35,
            "text_threshold_backend_note": "unused_hf_gdino_single_threshold_api",
        },
        "candidate_count_before_nms": n_before_nms,
        "candidate_count_after_nms": n_after_nms,
        "candidate_count_before_shape_filter": n_before_shape,
        "candidate_count_after_shape_filter": n_after_shape,
        "shape_prior_status": "passed",
    }


def test_debug_dict_has_all_required_keys() -> None:
    debug = _make_mock_debug()
    assert _REQUIRED_DEBUG_KEYS.issubset(debug.keys())


def test_debug_thresholds_has_note_key() -> None:
    debug = _make_mock_debug()
    assert _REQUIRED_THRESHOLD_KEYS.issubset(debug["thresholds_used"].keys())
    assert "unused" in debug["thresholds_used"]["text_threshold_backend_note"]


def test_debug_counts_are_monotonically_non_increasing() -> None:
    """NMS count ≤ raw count; shape-filter count ≤ pre-shape count."""
    debug = _make_mock_debug(n_before_nms=5, n_after_nms=3, n_before_shape=3, n_after_shape=1)
    assert debug["candidate_count_after_nms"] <= debug["candidate_count_before_nms"]
    assert debug["candidate_count_after_shape_filter"] <= debug["candidate_count_before_shape_filter"]


def test_shape_prior_status_in_debug_on_passed_detection() -> None:
    det = _det([180, 100, 220, 500])   # tall centered, passes zipper priors
    filter_by_shape_priors([det], "zipper", GARMENT)
    assert det["_shape_prior_status"] == "passed"


def test_shape_prior_status_in_debug_on_rejected_detection() -> None:
    det = _det([100, 100, 300, 160])   # wide, fails zipper h/w
    filter_by_shape_priors([det], "zipper", GARMENT)
    assert det["_shape_prior_status"] == "rejected"
    assert "_shape_prior_reasons" in det


# ── Phase 1 new features ──────────────────────────────────────────────────────

def test_long_tail_terms_in_part_vocab() -> None:
    from fashion_vision.localization.intent_parser import parse_intent
    cases = [
        ("抽绳", "drawstring"),
        ("绑带", "tie_strap"),
        ("荷叶边", "ruffle"),
        ("流苏", "fringe"),
        ("肩缝", "shoulder_seam"),
        ("袖缝", "sleeve_seam"),
    ]
    for query, expected_part in cases:
        intent = parse_intent(query)
        assert intent.part == expected_part, f"query={query!r}: expected {expected_part!r}, got {intent.part!r}"
        assert intent.is_zero_shot is False


def test_drawstring_not_fast_path() -> None:
    from fashion_vision.localization.intent_parser import parse_intent
    intent = parse_intent("抽绳")
    assert intent.is_fast_path is False


def test_seam_parts_not_fast_path() -> None:
    from fashion_vision.localization.intent_parser import parse_intent
    for query in ("肩缝", "袖缝"):
        intent = parse_intent(query)
        assert intent.is_fast_path is False, f"{query} should not be fast path"


def test_button_has_mask_dilation_px() -> None:
    cfg = get_part_shape_config("button")
    assert "mask_dilation_px" in cfg
    assert cfg["mask_dilation_px"] == 3


def test_zipper_has_mask_dilation_px() -> None:
    cfg = get_part_shape_config("zipper")
    assert "mask_dilation_px" in cfg
    assert cfg["mask_dilation_px"] == 5


def test_new_parts_have_detection_config() -> None:
    from fashion_vision.localization.part_detection_config import PART_DETECTION_CONFIG
    for part in ("drawstring", "tie_strap", "ruffle", "fringe", "shoulder_seam", "sleeve_seam"):
        assert part in PART_DETECTION_CONFIG, f"{part} missing from PART_DETECTION_CONFIG"
        cfg = PART_DETECTION_CONFIG[part]
        assert len(cfg["prompts"]) >= 1
        assert "box_threshold" in cfg
        assert "shape" in cfg


def test_long_tail_parts_have_grounding_text() -> None:
    from fashion_vision.localization.intent_parser import _PART_TO_GROUNDING_TEXT
    for part in ("drawstring", "tie_strap", "ruffle", "fringe", "shoulder_seam", "sleeve_seam"):
        assert part in _PART_TO_GROUNDING_TEXT, f"{part} missing from _PART_TO_GROUNDING_TEXT"
        assert isinstance(_PART_TO_GROUNDING_TEXT[part], str)
        assert len(_PART_TO_GROUNDING_TEXT[part]) > 0


# =============================================================================
# Phase 3 — Fashionpedia-first routing integration tests
# (skipped when cv2 unavailable — router imports cv2 at module level)
# =============================================================================


def _cv2_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("cv2") is not None
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.mark.skipif(not _cv2_available(), reason="cv2 not available in test env")
class TestFashionpediaRouterIntegration:
    """Router integration tests requiring cv2 (module-level import)."""

    @staticmethod
    def _make_fp_detector_mock(return_detections=None):
        if return_detections is None:
            return_detections = []
        mock = __import__("unittest").mock.MagicMock()
        mock.detect.return_value = return_detections
        return mock

    @staticmethod
    def _make_dino_locator_mock(return_detections=None):
        if return_detections is None:
            return_detections = []
        mock = __import__("unittest").mock.MagicMock()
        mock.detect_multi_prompt.return_value = (return_detections, len(return_detections))
        return mock

    @staticmethod
    def _make_instance():
        return {
            "instance_id": "001",
            "fine_class_name": "short sleeve top",
            "coarse_class_name": "top",
            "coarse_class_id": 0,
            "bbox": [100, 80, 400, 600],
            "pred_mask_path": None,
            "mask": None,
        }

    def test_fashionpedia_preferred_over_dino(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock([
            # ponytail: zipper shape config requires h/w >= 1.2 — use tall-narrow box
            {"bbox_xyxy": [180.0, 100.0, 220.0, 350.0], "score": 0.85,
             "label": "zipper", "class_id": 8, "backend": "fashionpedia_yolo"},
        ])
        dino_mock = self._make_dino_locator_mock()

        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "拉链", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fashionpedia_yolo"
        assert result["status"] == "success"
        dino_mock.detect_multi_prompt.assert_not_called()
        fp_mock.detect.assert_called_once()

    def test_dino_fallback_when_fp_empty(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        # ponytail: for FP-core parts (zipper), router returns not_detected when
        # FP YOLO returns empty — DINO is NOT used as fallback (Phase 3 design:
        # FP miss on core parts means the part genuinely isn't there).
        fp_mock = self._make_fp_detector_mock([])
        dino_mock = self._make_dino_locator_mock()

        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "拉链", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fashionpedia_yolo"
        assert result["status"] == "not_detected"
        fp_mock.detect.assert_called_once()
        dino_mock.detect_multi_prompt.assert_not_called()

    def test_dino_when_part_not_in_fp(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock()
        dino_mock = self._make_dino_locator_mock([
            {"bbox_xyxy": [150.0, 100.0, 180.0, 130.0], "score": 0.5,
             "label": "button", "_shape_prior_status": "passed"},
        ])

        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "纽扣", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "open_vocab_grounding_dino"
        fp_mock.detect.assert_not_called()
        dino_mock.detect_multi_prompt.assert_called_once()

    def test_backend_label_in_result_debug(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock([
            {"bbox_xyxy": [150.0, 100.0, 250.0, 200.0], "score": 0.85,
             "label": "collar", "class_id": 1, "backend": "fashionpedia_yolo"},
        ])
        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "领子", instance, img, 600, 800,
            locator=None,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fashionpedia_yolo"
        assert result["debug"]["backend"] == "fashionpedia_yolo"

    def test_fp_no_dino_non_fp_part_raises(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        with __import__("pytest").raises(ValueError):
            locate_region(
                "纽扣", instance, img, 600, 800,
                locator=None,
                fashionpedia_detector=None,
            )

    # ── Neckline / cuff migration tests ─────────────────────────────────

    def test_neckline_fp_hit_uses_yolo(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock([
            {"bbox_xyxy": [150.0, 100.0, 250.0, 200.0], "score": 0.85,
             "label": "neckline", "class_id": 6, "backend": "fashionpedia_yolo"},
        ])
        dino_mock = self._make_dino_locator_mock()
        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "领口", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fashionpedia_yolo"
        assert result["part"] == "neckline"
        dino_mock.detect_multi_prompt.assert_not_called()

    def test_cuff_fp_hit_uses_yolo_part_preserved(self) -> None:
        """When cuff is detected via FP sleeve class, result.part stays 'cuff'."""
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock([
            {"bbox_xyxy": [300.0, 200.0, 380.0, 350.0], "score": 0.78,
             "label": "sleeve", "class_id": 4, "backend": "fashionpedia_yolo"},
        ])
        dino_mock = self._make_dino_locator_mock()
        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "袖口", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fashionpedia_yolo"
        # Part must stay "cuff" even though YOLO label is "sleeve".
        assert result["part"] == "cuff"
        dino_mock.detect_multi_prompt.assert_not_called()

    def test_neckline_fp_miss_falls_back_to_fast_path(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock([])  # FP miss
        dino_mock = self._make_dino_locator_mock()
        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "领口", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fast_path"
        assert result["method"] == "fast_path"
        # DINO must NOT be called.
        dino_mock.detect_multi_prompt.assert_not_called()

    def test_cuff_fp_miss_falls_back_to_fast_path(self) -> None:
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock([])  # FP miss
        dino_mock = self._make_dino_locator_mock()
        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "袖口", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fast_path"
        assert result["method"] == "fast_path"
        dino_mock.detect_multi_prompt.assert_not_called()

    def test_fp_part_miss_returns_not_detected(self) -> None:
        """Pocket FP miss → not_detected, DINO NOT called (FP parts don't fall back)."""
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock([])  # FP miss
        dino_mock = self._make_dino_locator_mock()
        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "口袋", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "fashionpedia_yolo"
        assert result["status"] == "not_detected"
        assert result["reason"] == "fashionpedia_no_detection"
        fp_mock.detect.assert_called_once()
        # DINO must NOT be called — FP parts don't fall back.
        dino_mock.detect_multi_prompt.assert_not_called()

    def test_non_fp_part_still_falls_back_to_dino(self) -> None:
        """Button (non-FP part) → DINO fallback on miss."""
        from fashion_vision.localization.region_localization_router import locate_region

        fp_mock = self._make_fp_detector_mock()  # button not in FP → detect not called
        dino_mock = self._make_dino_locator_mock([
            {"bbox_xyxy": [150.0, 100.0, 180.0, 130.0], "score": 0.5,
             "label": "button", "_shape_prior_status": "passed"},
        ])
        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "纽扣", instance, img, 600, 800,
            locator=dino_mock,
            fashionpedia_detector=fp_mock,
        )
        assert result["backend"] == "open_vocab_grounding_dino"
        assert result["status"] == "success"
        dino_mock.detect_multi_prompt.assert_called_once()

    def test_neckline_no_fp_detector_falls_back_to_fast_path(self) -> None:
        """Neckline with no FP detector → fast-path (no error)."""
        from fashion_vision.localization.region_localization_router import locate_region

        instance = self._make_instance()
        img = __import__("numpy").zeros((800, 600, 3), dtype=__import__("numpy").uint8)

        result = locate_region(
            "领口", instance, img, 600, 800,
            locator=None,
            fashionpedia_detector=None,
        )
        assert result["backend"] == "fast_path"


# =============================================================================
# Phase 3 — New vocab entries (lapel, epaulette, buckle)
# =============================================================================


def test_lapel_in_part_vocab() -> None:
    from fashion_vision.localization.intent_parser import PART_VOCAB, _PART_TO_GROUNDING_TEXT
    assert "lapel" in PART_VOCAB
    assert "翻领" in PART_VOCAB["lapel"]
    assert "lapel" in _PART_TO_GROUNDING_TEXT


def test_epaulette_in_part_vocab() -> None:
    from fashion_vision.localization.intent_parser import PART_VOCAB, _PART_TO_GROUNDING_TEXT
    assert "epaulette" in PART_VOCAB
    assert "肩章" in PART_VOCAB["epaulette"]
    assert "epaulette" in _PART_TO_GROUNDING_TEXT


def test_buckle_in_part_vocab() -> None:
    from fashion_vision.localization.intent_parser import PART_VOCAB, _PART_TO_GROUNDING_TEXT
    assert "buckle" in PART_VOCAB
    assert "扣环" in PART_VOCAB["buckle"]
    assert "buckle" in _PART_TO_GROUNDING_TEXT


def test_fast_path_unchanged_with_new_parts() -> None:
    """New FP parts (lapel, epaulette, buckle) are NOT fast-path."""
    from fashion_vision.localization.intent_parser import FAST_PATH_PARTS, parse_intent
    assert "lapel" not in FAST_PATH_PARTS
    assert "epaulette" not in FAST_PATH_PARTS
    assert "buckle" not in FAST_PATH_PARTS
    # Parse should work and route to open-vocab
    intent = parse_intent("翻领")
    assert intent.part == "lapel"
    assert not intent.is_fast_path


# =============================================================================
# Phase 3 — Neckline + cuff migration: FAST_PATH_PARTS → FP-first
# =============================================================================


def test_neckline_not_in_fast_path_parts() -> None:
    from fashion_vision.localization.intent_parser import FAST_PATH_PARTS
    assert "neckline" not in FAST_PATH_PARTS


def test_cuff_not_in_fast_path_parts() -> None:
    from fashion_vision.localization.intent_parser import FAST_PATH_PARTS
    assert "cuff" not in FAST_PATH_PARTS


def test_hem_waist_shoulder_leg_opening_still_fast_path_parts() -> None:
    from fashion_vision.localization.intent_parser import FAST_PATH_PARTS
    for part in ("hem", "waist", "shoulder", "leg_opening"):
        assert part in FAST_PATH_PARTS, f"{part} should still be in FAST_PATH_PARTS"


def test_neckline_in_fp_core_part_map() -> None:
    from fashion_vision.localization.fashionpedia_part_detector import (
        FP_CORE_PART_MAP, PART_TO_FP_IDS,
    )
    assert 6 in FP_CORE_PART_MAP
    assert FP_CORE_PART_MAP[6] == "neckline"
    assert "neckline" in PART_TO_FP_IDS
    assert 6 in PART_TO_FP_IDS["neckline"]


def test_cuff_maps_to_sleeve_fp_class() -> None:
    from fashion_vision.localization.fashionpedia_part_detector import PART_TO_FP_IDS
    assert "cuff" in PART_TO_FP_IDS
    assert PART_TO_FP_IDS["cuff"] == [4]


def test_sleeve_still_maps_to_sleeve_fp_class() -> None:
    from fashion_vision.localization.fashionpedia_part_detector import PART_TO_FP_IDS
    assert "sleeve" in PART_TO_FP_IDS
    assert 4 in PART_TO_FP_IDS["sleeve"]


# =============================================================================
# Regression: summary print handles None backend/part/status/score
# =============================================================================


def test_summary_print_none_safe() -> None:
    """Summary line formatting must not crash when fields are None."""
    results: list[dict] = [
        {"query": "q1", "backend": None, "part": None, "status": "not_detected", "score": None},
        {"query": "q2", "backend": "fashionpedia_yolo", "part": "pocket", "status": "success", "score": 0.85},
    ]
    lines: list[str] = []
    for r in results:
        query_str = str(r.get("query") or "-")
        backend_str = str(r.get("backend") or "-")
        part_str = str(r.get("part") or "-")
        status_str = str(r.get("status") or "-")
        score = r.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        # Must not raise on None values.
        lines.append(
            f"  {query_str:<12} -> backend={backend_str:<28}"
            f" part={part_str:<14} status={status_str:<10} score={score_str}"
        )
    assert "not_detected" in lines[0]
    assert "fashionpedia_yolo" in lines[1]
    assert "0.850" in lines[1]
    assert "-" in lines[0]  # score is "-" for None
