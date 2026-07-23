"""P1.4b — Region backend integration tests.

Tests for backend adapters, normalisation, config, and serving integration.
No real models, no GPU, no DeepFashion2 data required.
"""

from __future__ import annotations

import json

import pytest

from inference.serving.qa_orchestrator import QaOrchestrator
from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.attribute_service import AttributeService
from inference.serving.rag_service import RagService


# ── Backend adapter tests ───────────────────────────────────────────────────


class TestDisabledBackend:
    def test_returns_empty(self):
        from inference.serving.region_backend import DisabledRegionLocalizationBackend
        b = DisabledRegionLocalizationBackend()
        assert b.backend_name == "disabled"
        assert b.enabled is False
        assert b.locate_regions(image=None) == []
        assert b.locate_regions(image="fake", query="test") == []


class TestBuildBackend:
    def test_disabled_by_default(self):
        from inference.serving.region_backend import build_region_backend
        b = build_region_backend()
        assert b.backend_name == "disabled"

    def test_fashionpedia_construction(self):
        from inference.serving.region_backend import build_region_backend, FashionpediaRegionBackend
        b = build_region_backend("fashionpedia", model_path="/nonexistent/model.pt", device="cpu")
        assert isinstance(b, FashionpediaRegionBackend)
        assert b.backend_name == "fashionpedia_yolo"
        # Should NOT be enabled since model file doesn't exist.
        assert b.enabled is False

    def test_full312_construction(self):
        from inference.serving.region_backend import build_region_backend, Full312RegionBackend
        b = build_region_backend("full312", model_path="/nonexistent/model.pt", device="cpu")
        assert isinstance(b, Full312RegionBackend)
        assert b.backend_name == "full312"
        # Should NOT be enabled since model file doesn't exist.
        assert b.enabled is False


# ── Normalisation tests ─────────────────────────────────────────────────────


_SAMPLE_RAW_FP = [
    {"bbox_xyxy": [100, 40, 220, 110], "score": 0.82, "label": "neckline", "class_id": 6},
    {"bbox_xyxy": [180, 260, 260, 340], "score": 0.76, "label": "pocket", "class_id": 5},
    {"bbox_xyxy": [210, 300, 230, 480], "score": 0.65, "label": "zipper", "class_id": 8},
]


class TestNormalizeRegionPredictions:
    def test_valid_predictions(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            _SAMPLE_RAW_FP, source="test", backend="fp_yolo",
        )
        assert len(result) == 3
        assert result[0]["region_id"] == "region_0"
        assert result[0]["part_type"] == "neckline"
        assert result[0]["part_group"] == "collar_area"
        assert result[0]["bbox"] == [100.0, 40.0, 220.0, 110.0]
        assert result[0]["confidence"] == 0.82
        assert result[0]["source"] == "test"
        assert result[0]["backend"] == "fp_yolo"
        assert result[0]["mask_present"] is False
        assert result[0]["mask_ref"] is None

    def test_pocket_group(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            [{"bbox_xyxy": [10, 20, 30, 40], "score": 0.9, "label": "pocket"}],
        )
        assert result[0]["part_group"] == "pocket_area"

    def test_zipper_group(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            [{"bbox_xyxy": [10, 20, 30, 40], "score": 0.9, "label": "zipper"}],
        )
        assert result[0]["part_group"] == "closure"

    def test_unknown_label(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            [{"bbox_xyxy": [10, 20, 30, 40], "score": 0.9, "label": "garbaggio"}],
        )
        assert result[0]["part_type"] == "unknown"
        assert result[0]["part_group"] == "unknown"

    def test_missing_confidence(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            [{"bbox_xyxy": [10, 20, 30, 40], "label": "neckline"}],
        )
        assert result[0]["confidence"] is None

    def test_invalid_bbox_dropped(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions([
            {"bbox_xyxy": [10, 20, 5, 40], "score": 0.9, "label": "neckline"},  # x2 < x1
            {"bbox_xyxy": [10, 20, 30, 40], "score": 0.8, "label": "pocket"},    # valid
            {"bbox_xyxy": [10, 20], "score": 0.7, "label": "zipper"},             # too short
        ])
        assert len(result) == 1
        assert result[0]["part_type"] == "pocket"

    def test_empty_list(self):
        from inference.serving.region_backend import normalize_region_predictions
        assert normalize_region_predictions([]) == []

    def test_non_dict_entries_skipped(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions([
            "not_a_dict",
            {"bbox_xyxy": [10, 20, 30, 40], "score": 0.9, "label": "neckline"},
        ])
        assert len(result) == 1

    def test_start_index(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            _SAMPLE_RAW_FP[:1], start_index=5,
        )
        assert result[0]["region_id"] == "region_5"

    def test_no_forbidden_fields(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            [{"bbox_xyxy": [10, 20, 30, 40], "score": 0.9, "label": "neckline",
              "mask": b"fake_mask_data", "crop_path": "/tmp/crop.png",
              "raw_tensor": "tensor_data"}],
        )
        r = result[0]
        for forbidden in ("mask", "crop_path", "raw_tensor", "crop", "image_bytes",
                          "temp_path", "file_path", "checkpoint"):
            assert forbidden not in r, f"Leaked {forbidden}"

    def test_cuff_label_maps_correctly(self):
        from inference.serving.region_backend import normalize_region_predictions
        result = normalize_region_predictions(
            [{"bbox_xyxy": [10, 20, 30, 40], "score": 0.9, "label": "cuff"}],
        )
        assert result[0]["part_type"] == "cuff"
        assert result[0]["part_group"] == "sleeve_area"


# ── Decode image tests ──────────────────────────────────────────────────────


class TestDecodeImageBytes:
    def test_none_returns_none(self):
        from inference.serving.region_backend import decode_image_bytes
        assert decode_image_bytes(None) is None

    def test_invalid_base64_returns_none(self):
        from inference.serving.region_backend import decode_image_bytes
        assert decode_image_bytes("not valid base64!!!") is None

    def test_empty_bytes_returns_none(self):
        from inference.serving.region_backend import decode_image_bytes
        assert decode_image_bytes(b"") is None

    def test_non_bytes_non_str_returns_none(self):
        from inference.serving.region_backend import decode_image_bytes
        assert decode_image_bytes(12345) is None


# ── Serving integration tests (fake backend) ─────────────────────────────────


class _FakeEnabledRegionBackend:
    """Fake region backend that returns pre-canned regions (simulates real FP YOLO)."""
    backend_name = "fake_test"
    enabled = True

    def __init__(self, regions=None):
        self._regions = list(regions) if regions else []
        self._call_count = 0

    def locate_regions(self, image=None, query=None, requested_part=None, **kwargs):
        self._call_count += 1
        self._last_requested_part = requested_part
        return list(self._regions)


class _FakeDisabledRegionBackend:
    """Fake region backend that is disabled (simulates disabled config)."""
    backend_name = "fake_disabled"
    enabled = False

    def locate_regions(self, image=None, query=None, requested_part=None, **kwargs):
        return []


_FAKE_REGIONS = [
    {
        "region_id": "region_0",
        "part_type": "neckline",
        "part_group": "collar_area",
        "bbox": [100.0, 40.0, 220.0, 110.0],
        "confidence": 0.82,
        "source": "fashion_vision_3_1_2",
        "backend": "fake_test",
        "mask_present": False,
        "mask_ref": None,
    },
    {
        "region_id": "region_1",
        "part_type": "pocket",
        "part_group": "pocket_area",
        "bbox": [180.0, 260.0, 260.0, 340.0],
        "confidence": 0.76,
        "source": "fashion_vision_3_1_2",
        "backend": "fake_test",
        "mask_present": False,
        "mask_ref": None,
    },
]


# ── Fake vision provider that returns NO localized_regions (forces backend call) ──


class _FakeEmptyVisionProvider:
    """Returns empty VisionAttributeResult (forces region backend call)."""

    def __init__(self, instances=None):
        from inference.serving.vision_provider import VisionAttributeResult
        self._result = VisionAttributeResult(
            attributes={},
            garment_instances=list(instances) if instances else [],
            regions=[],
            sources=[],
            warnings=[],
            used_tools=["fake_vision_provider"],
            meta={"provider": "test"},
        )

    def extract(self, **kwargs):
        return self._result


# A tiny valid JPEG (1x1 white pixel) for image_bytes decoding tests.
_VALID_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
    b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
    b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f"
    b"\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00"
    b"\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03"
    b"\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1"
    b"\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJ"
    b"STUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
    b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5"
    b"\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
    b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3"
    b"\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2"
    b"\xcf \xff\xd9"
)


def _make_orchestrator_with_region_backend(region_backend, vision_provider=None):
    """Build QaOrchestrator with a custom region backend via monkeypatch."""
    import inference.serving.qa_orchestrator as qa_mod
    orch = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=vision_provider or _FakeEmptyVisionProvider(),
    )
    # Inject fake backend by replacing the region_backend singleton.
    import inference.serving.region_backend as rb_mod
    _orig_singleton = rb_mod._region_backend
    rb_mod._region_backend = region_backend
    orch = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=vision_provider or _FakeEmptyVisionProvider(),
    )
    yield orch
    rb_mod._region_backend = _orig_singleton


@pytest.fixture
def orch_with_fake_backend():
    """Orchestrator that uses _FakeEnabledRegionBackend."""
    backend = _FakeEnabledRegionBackend(_FAKE_REGIONS)
    yield from _make_orchestrator_with_region_backend(backend)


@pytest.fixture
def orch_with_disabled_backend():
    """Orchestrator that uses _FakeDisabledRegionBackend."""
    backend = _FakeDisabledRegionBackend()
    yield from _make_orchestrator_with_region_backend(backend)


class TestServingIntegration:
    def test_region_backend_called_for_region_query(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        assert r.answer_type == "region_query_answer"
        assert "region_backend" in r.used_tools
        assert "bbox" in r.answer or "检测到" in r.answer

    def test_region_backend_not_called_for_non_region_query(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="这件衣服是什么面料？",
                        attributes={"fabric": {"value": "棉", "attribute_confidence": 0.9}})
        assert r.answer_type == "attribute_answer"
        assert "region_backend" not in r.used_tools

    def test_disabled_backend_warning(self, orch_with_disabled_backend):
        orch = orch_with_disabled_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        assert r.answer_type == "region_query_answer"
        assert any(w.code == "region_backend_disabled" for w in r.warnings)

    def test_existence_query_uses_backend(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="有没有口袋？", image_bytes=_VALID_JPEG_BYTES)
        assert r.answer_type == "region_query_answer"
        assert "检测到" in r.answer
        assert "region_backend" in r.used_tools

    def test_detail_query_uses_backend(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="这件衣服有哪些细节？", image_bytes=_VALID_JPEG_BYTES)
        assert r.answer_type == "region_query_answer"
        assert "region_backend" in r.used_tools

    def test_no_image_bytes_no_backend_call(self, orch_with_fake_backend):
        """Without image_bytes, backend should not be called (no image to analyze)."""
        orch = orch_with_fake_backend
        r = orch.answer(query="领口在哪里？")
        assert r.answer_type == "region_query_answer"
        assert "region_backend" not in r.used_tools


# ── No-leak tests for backend integration ────────────────────────────────────


class TestRegionBackendNoLeak:
    def test_response_no_raw_paths(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        for forbidden in ("D:\\\\", "/tmp/", "\\temp", "outputs/", "checkpoints/"):
            assert forbidden not in payload, f"Leaked {forbidden}"

    def test_response_no_mask_data(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        for forbidden in ("mask_bitmap", "mask_data", "crop_path", "crop_data",
                          "crop_bytes", "tensor", "checkpoint"):
            assert forbidden not in payload, f"Leaked {forbidden}"

    def test_response_no_image_bytes_leak(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        # The raw JPEG bytes should never appear.
        assert "JFIF" not in payload

    def test_localized_regions_summary_safe_keys(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        summary = r.meta.get("localized_regions_summary", [])
        SAFE = {"region_id", "part_type", "part_group", "bbox", "confidence", "source", "backend"}
        for region in summary:
            assert set(region.keys()).issubset(SAFE), f"Leaked: {set(region.keys()) - SAFE}"


# ── Missing model / error handling tests ────────────────────────────────────


class _FakeErrorRegionBackend:
    """Fake backend that always raises (simulates model load failure)."""
    backend_name = "fake_error"
    enabled = True

    def locate_regions(self, image=None, query=None, requested_part=None, **kwargs):
        raise RuntimeError("Simulated backend crash")


class _FakeMissingModelBackend:
    """Fake backend that is enabled but returns empty (simulates missing model)."""
    backend_name = "fake_missing_model"
    enabled = True

    def locate_regions(self, image=None, query=None, requested_part=None, **kwargs):
        return []


@pytest.fixture
def orch_with_error_backend():
    backend = _FakeErrorRegionBackend()
    yield from _make_orchestrator_with_region_backend(backend)


@pytest.fixture
def orch_with_missing_model_backend():
    backend = _FakeMissingModelBackend()
    yield from _make_orchestrator_with_region_backend(backend)


class TestErrorHandling:
    def test_backend_exception_returns_safe_fallback(self, orch_with_error_backend):
        orch = orch_with_error_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        assert r.answer_type == "region_query_answer"
        assert any(w.code == "region_backend_error" for w in r.warnings)
        # Must NOT crash, must return a safe fallback answer.
        assert len(r.answer) > 0

    def test_missing_model_no_crash(self, orch_with_missing_model_backend):
        orch = orch_with_missing_model_backend
        r = orch.answer(query="有没有口袋？", image_bytes=_VALID_JPEG_BYTES)
        assert r.answer_type == "region_query_answer"
        # Backend returns empty → the "unavailable" path is NOT taken since
        # backend WAS called. Instead, intent routing handles empty regions.
        assert "region_backend" in r.used_tools
        assert "没有可靠" in r.answer or "不能确认" in r.answer or "未检测到" in r.answer

    def test_disabled_backend_does_not_load_model(self, orch_with_disabled_backend):
        orch = orch_with_disabled_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        assert r.answer_type == "region_query_answer"
        assert "region_backend" not in r.used_tools
        assert any(w.code == "region_backend_disabled" for w in r.warnings)

    def test_no_stack_trace_in_response(self, orch_with_error_backend):
        orch = orch_with_error_backend
        r = orch.answer(query="领口在哪里？", image_bytes=_VALID_JPEG_BYTES)
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        assert "Traceback" not in payload
        assert "RuntimeError" not in payload
        assert "line " not in payload  # stack frames


# ── Regression tests ─────────────────────────────────────────────────────────


class TestP14bRegression:
    def test_existing_attribute_query_still_works(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(
            query="这件衣服是什么面料？",
            attributes={"fabric": {"value": "纯棉", "attribute_confidence": 0.92}},
        )
        assert r.answer_type == "attribute_answer"
        assert "纯棉" in r.answer

    def test_existing_garment_instance_query_still_works(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="图中检测到了什么？", image_bytes=_VALID_JPEG_BYTES)
        # Should not hit region backend — this is visual_instance_query.
        assert r.answer_type != "region_query_answer"

    def test_existing_knowledge_query_still_works(self, orch_with_fake_backend):
        orch = orch_with_fake_backend
        r = orch.answer(query="纤维是什么")
        assert r.answer_type in ("knowledge_answer", "unsupported")
        assert "region_backend" not in r.used_tools


# ── Full312 backend tests ──────────────────────────────────────────────────

_FAKE_LOCATE_REGION_FP = {
    "status": "success", "bbox": [100, 40, 220, 110],
    "score": 0.82, "backend": "fashionpedia_yolo",
    "query": "neckline", "part": "neckline",
}
_FAKE_LOCATE_REGION_DINO = {
    "status": "success", "bbox": [180, 260, 260, 340],
    "score": 0.71, "backend": "open_vocab_grounding_dino",
    "query": "button", "part": "button",
}
_FAKE_NOT_DETECTED = {
    "status": "not_detected", "bbox": None, "score": None,
    "backend": "fashionpedia_yolo",
}

# Synthetic 10x10 BGR image for tests that need a real numpy array.
import numpy as np
_FAKE_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)


def _patch_full312_for_test(b, monkeypatch):
    """Bypass model loading and inject fake locate_region."""
    import inference.serving.region_backend as rb_mod
    from fashion_vision.localization import region_localization_router as rlr
    b._fp_detector = object()
    b._dino = object()
    return b


class TestFull312Backend:
    def test_fp_provenance(self, monkeypatch):
        from fashion_vision.localization import region_localization_router as rlr
        monkeypatch.setattr(rlr, "locate_region", lambda *a, **kw: dict(_FAKE_LOCATE_REGION_FP))
        import inference.serving.region_backend as rb_mod; rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt", device="cpu")
        b._fp_detector = object(); b._dino = object()
        regions = b.locate_regions(image=_FAKE_IMAGE, requested_part="neckline")
        assert len(regions) > 0 and regions[0]["backend"] == "fashionpedia_yolo"

    def test_dino_provenance(self, monkeypatch):
        from fashion_vision.localization import region_localization_router as rlr
        monkeypatch.setattr(rlr, "locate_region", lambda *a, **kw: dict(_FAKE_LOCATE_REGION_DINO))
        import inference.serving.region_backend as rb_mod; rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt", device="cpu")
        b._fp_detector = object(); b._dino = object()
        regions = b.locate_regions(image=_FAKE_IMAGE, requested_part="button")
        assert len(regions) > 0 and regions[0]["backend"] == "dino"

    def test_not_detected_empty(self, monkeypatch):
        from fashion_vision.localization import region_localization_router as rlr
        monkeypatch.setattr(rlr, "locate_region", lambda *a, **kw: dict(_FAKE_NOT_DETECTED))
        import inference.serving.region_backend as rb_mod; rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt", device="cpu")
        b._fp_detector = object(); b._dino = object()
        assert b.locate_regions(image=_FAKE_IMAGE, requested_part="pocket") == []

    def test_query_all_parts(self, monkeypatch):
        from fashion_vision.localization import region_localization_router as rlr
        calls = []
        def _fake(*args, **kw):
            q = kw.get("query", args[0] if args else "")
            calls.append(q)
            if "neckline" in str(q): return dict(_FAKE_LOCATE_REGION_FP)
            return dict(_FAKE_NOT_DETECTED)
        monkeypatch.setattr(rlr, "locate_region", _fake)
        import inference.serving.region_backend as rb_mod; rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt", device="cpu")
        b._fp_detector = object(); b._dino = object()
        regions = b.locate_regions(image=_FAKE_IMAGE, query_all_parts=True)
        assert len(calls) > 1 and len(regions) >= 1

    def test_fast_path_skipped(self):
        import inference.serving.region_backend as rb_mod; rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt", device="cpu")
        b._fp_detector = object(); b._dino = object()
        for part in ("hem", "waist", "shoulder", "leg_opening"):
            assert b.locate_regions(image=_FAKE_IMAGE, requested_part=part) == []

    def test_factory_full312(self):
        from inference.serving.region_backend import build_region_backend, Full312RegionBackend
        b = build_region_backend("full312", model_path="/fake/model.pt", device="cpu")
        assert isinstance(b, Full312RegionBackend) and not b.enabled


# ── P1.4e: Instance bridge tests ────────────────────────────────────────────

_SAMPLE_311_INSTANCE = {
    "instance_id": "inst_0",
    "category": "top",
    "fine_class_name": "short sleeve top",
    "bbox": [120.0, 80.0, 300.0, 350.0],
    "bbox_format": "xyxy_abs_pixels",
    "confidence": 0.93,
    "mask_present": True,
    "mask_ref": "mask_inst_0",
}


class TestInstanceBridge:
    def test_preserves_instance_id(self):
        from inference.serving.region_backend import build_locate_region_instance
        inst = build_locate_region_instance(_SAMPLE_311_INSTANCE, 500, 600)
        assert inst["instance_id"] == "inst_0"

    def test_preserves_category(self):
        from inference.serving.region_backend import build_locate_region_instance
        inst = build_locate_region_instance(_SAMPLE_311_INSTANCE, 500, 600)
        assert inst["fine_class_name"] == "short sleeve top"
        assert inst["category"] == "top"

    def test_preserves_bbox(self):
        from inference.serving.region_backend import build_locate_region_instance
        inst = build_locate_region_instance(_SAMPLE_311_INSTANCE, 500, 600)
        assert inst["bbox"] == [120.0, 80.0, 300.0, 350.0]

    def test_clips_bbox_to_image_bounds(self):
        from inference.serving.region_backend import build_locate_region_instance
        inst = build_locate_region_instance(
            {"instance_id": "i1", "bbox": [-10, -10, 200, 200]}, 100, 100,
        )
        assert inst["bbox"] == [0.0, 0.0, 100.0, 100.0]

    def test_invalid_bbox_omitted(self):
        from inference.serving.region_backend import build_locate_region_instance
        inst = build_locate_region_instance(
            {"instance_id": "i1", "bbox": [50, 50, 40, 40]}, 100, 100,
        )
        assert "bbox" not in inst  # x2 <= x1

    def test_missing_mask_no_path(self):
        from inference.serving.region_backend import build_locate_region_instance
        inst = build_locate_region_instance(_SAMPLE_311_INSTANCE, 500, 600)
        # mask_ref is placeholder, no resolver → no pred_mask_path
        assert "pred_mask_path" not in inst

    def test_synthetic_instance(self):
        from inference.serving.region_backend import make_synthetic_instance
        inst = make_synthetic_instance(100, 200)
        assert inst["instance_id"] == "synthetic_full_image"
        assert inst["bbox"] == [0, 0, 100, 200]


class TestFull312WithInstances:
    def test_uses_real_instances(self, monkeypatch):
        import inference.serving.region_backend as rb_mod
        from fashion_vision.localization import region_localization_router as rlr
        calls = []
        def _fake(*args, **kw):
            inst = kw.get("instance", {})
            calls.append(inst.get("instance_id", "?"))
            return {"status": "success", "bbox": [10,20,30,40], "score": 0.8,
                    "backend": "fashionpedia_yolo"}
        monkeypatch.setattr(rlr, "locate_region", _fake)
        rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(
            fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
            device="cpu")
        b._fp_detector = object(); b._dino = object()
        regions = b.locate_regions(
            image=_FAKE_IMAGE, requested_part="neckline",
            garment_instances=[_SAMPLE_311_INSTANCE],
        )
        assert "inst_0" in calls
        assert not any("synthetic" in c for c in calls)

    def test_instance_id_in_output(self, monkeypatch):
        import inference.serving.region_backend as rb_mod
        from fashion_vision.localization import region_localization_router as rlr
        monkeypatch.setattr(rlr, "locate_region",
            lambda *a, **kw: {"status": "success", "bbox": [10,20,30,40],
                              "score": 0.8, "backend": "fashionpedia_yolo"})
        rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(
            fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
            device="cpu")
        b._fp_detector = object(); b._dino = object()
        regions = b.locate_regions(
            image=_FAKE_IMAGE, requested_part="neckline",
            garment_instances=[_SAMPLE_311_INSTANCE],
        )
        assert len(regions) > 0
        assert regions[0]["instance_id"] == "inst_0"

    def test_no_instance_id_for_synthetic(self, monkeypatch):
        import inference.serving.region_backend as rb_mod
        from fashion_vision.localization import region_localization_router as rlr
        monkeypatch.setattr(rlr, "locate_region",
            lambda *a, **kw: {"status": "success", "bbox": [10,20,30,40],
                              "score": 0.8, "backend": "fashionpedia_yolo"})
        rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(
            fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
            device="cpu")
        b._fp_detector = object(); b._dino = object()
        # No garment_instances → synthetic fallback
        regions = b.locate_regions(image=_FAKE_IMAGE, requested_part="neckline")
        assert len(regions) > 0
        assert regions[0].get("instance_id") is None

    def test_fallback_disabled_returns_empty(self):
        import inference.serving.region_backend as rb_mod
        rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(
            fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
            device="cpu")
        b._fp_detector = object(); b._dino = object()
        regions = b.locate_regions(
            image=_FAKE_IMAGE, requested_part="neckline",
            allow_full_image_fallback=False,
        )
        assert regions == []

    def test_fast_path_not_skipped_with_real_instances(self, monkeypatch):
        import inference.serving.region_backend as rb_mod
        from fashion_vision.localization import region_localization_router as rlr
        called = []
        monkeypatch.setattr(rlr, "locate_region",
            lambda *a, **kw: called.append(kw.get("instance", {}).get("instance_id"))
            or {"status": "success", "bbox": [10,20,30,40],
                "score": 0.8, "backend": "fast_path"})
        rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(
            fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
            device="cpu")
        b._fp_detector = object(); b._dino = object()
        regions = b.locate_regions(
            image=_FAKE_IMAGE, requested_part="hem",
            garment_instances=[_SAMPLE_311_INSTANCE],
        )
        # "hem" is a fast-path part, but with real instance it should NOT be skipped.
        assert len(called) > 0  # locate_region was called for hem

    def test_target_instance_id_filters(self, monkeypatch):
        import inference.serving.region_backend as rb_mod
        from fashion_vision.localization import region_localization_router as rlr
        calls = []
        monkeypatch.setattr(rlr, "locate_region",
            lambda *a, **kw: calls.append(kw.get("instance", {}).get("instance_id"))
            or {"status": "success", "bbox": [10,20,30,40],
                "score": 0.8, "backend": "fashionpedia_yolo"})
        rb_mod.reset_region_backend()
        b = rb_mod.Full312RegionBackend(
            fp_model_path="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
            device="cpu")
        b._fp_detector = object(); b._dino = object()
        b.locate_regions(
            image=_FAKE_IMAGE, requested_part="neckline",
            garment_instances=[
                _SAMPLE_311_INSTANCE,
                {"instance_id": "inst_1", "bbox": [10,10,50,50]},
            ],
            target_instance_id="inst_1",
        )
        # Only inst_1 should have been used.
        assert all(c == "inst_1" for c in calls)
        assert "inst_0" not in calls
