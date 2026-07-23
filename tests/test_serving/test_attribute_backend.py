"""P1.4f — 3.1.3 attribute backend tests. No real models/GPU required."""

from __future__ import annotations

import json
import pytest
import numpy as np

from inference.serving.qa_orchestrator import QaOrchestrator
from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.attribute_service import AttributeService
from inference.serving.rag_service import RagService
from inference.serving.vision_provider import MockVisionAttributeProvider


# ── Disabled backend tests ──────────────────────────────────────────────────

class TestDisabledAttributeBackend:
    def test_returns_empty(self):
        from inference.serving.attribute_backend import DisabledAttributeBackend
        b = DisabledAttributeBackend()
        assert b.backend_name == "disabled"
        assert not b.enabled
        assert b.extract_attributes() == []

    def test_singleton_is_disabled_by_default(self):
        from inference.serving.attribute_backend import get_attribute_backend, reset_attribute_backend
        reset_attribute_backend()
        b = get_attribute_backend()
        assert b.backend_name == "disabled"
        assert not b.enabled


# ── Normalisation tests ─────────────────────────────────────────────────────

class TestAttributeNormalization:
    def test_normalize_with_confidence(self):
        from inference.serving.attribute_backend import _normalize_attribute_result
        r = _normalize_attribute_result("neckline_design", {
            "label": "V-shape", "score": 0.87,
            "topk": [{"label": "V-shape", "score": 0.87}, {"label": "Round", "score": 0.10}],
        }, region_id="r0", instance_id="inst_0")
        assert r["task"] == "neckline_design"
        assert r["value"] == "V-shape"
        assert r["attribute_confidence"] == 0.87
        assert r["region_id"] == "r0"
        assert r["instance_id"] == "inst_0"

    def test_no_leak_fields(self):
        from inference.serving.attribute_backend import _normalize_attribute_result
        r = _normalize_attribute_result("sleeve_length", {"label": "Long", "score": 0.9})
        for forbidden in ("model", "state_dict", "checkpoint", "raw_logits",
                          "tensor", "file_path", "temp_path", "image_bytes"):
            assert forbidden not in r


# ── QA integration tests ────────────────────────────────────────────────────

_FAKE_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)
_VALID_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
    b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
    b"\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f"
    b"\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00"
    b"\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03"
    b"\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1"
    b"\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJ"
    b"STUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
    b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5"
    b"\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
    b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3"
    b"\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2"
    b"\xcf \xff\xd9"
)


class _FakeRegionVisionProvider:
    def __init__(self, regions=None):
        from inference.serving.vision_provider import VisionAttributeResult
        self._result = VisionAttributeResult(
            attributes={}, garment_instances=[], regions=regions or [],
            sources=[], warnings=[], used_tools=[], meta={"provider": "test"},
        )
    def extract(self, **kw): return self._result


_SAMPLE_REGIONS_WITH_CUFF = [{
    "region_id": "region_3", "part_type": "cuff", "part_group": "sleeve_area",
    "bbox": [80.0, 200.0, 130.0, 250.0], "confidence": 0.71,
    "source": "mock_3_1_2", "backend": "mock", "mask_present": False,
}]


@pytest.fixture
def orch_with_cuff():
    return QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=_FakeRegionVisionProvider(_SAMPLE_REGIONS_WITH_CUFF),
    )


class TestAttributeQAIntegration:
    def test_attribute_query_with_region_returns_answer(self, orch_with_cuff):
        """袖口是什么设计？ — cuff region exists, attribute backend disabled."""
        r = orch_with_cuff.answer(query="袖口是什么设计？", image_bytes=_VALID_JPEG)
        assert r.answer_type == "region_query_answer"
        # Backend disabled → placeholder message about 3.1.3 not enabled.
        assert "袖口" in r.answer or "cuff" in r.answer or "3.1.3" in r.answer or "暂未启用" in r.answer or "尚未接入" in r.answer

    def test_attribute_query_no_region(self, orch_with_cuff):
        """领口是什么设计？ — no neckline region in mock data."""
        r = orch_with_cuff.answer(query="领口是什么设计？", image_bytes=_VALID_JPEG)
        assert r.answer_type == "region_query_answer"
        assert "未检测到" in r.answer or "尚未接入" in r.answer

    def test_non_region_query_no_attr_backend_call(self, orch_with_cuff):
        r = orch_with_cuff.answer(query="这件衣服是什么面料？",
                                  attributes={"fabric": {"value": "棉", "attribute_confidence": 0.9}})
        assert r.answer_type == "attribute_answer"
        assert "attribute_backend" not in r.used_tools

    def test_no_leak_in_attribute_response(self, orch_with_cuff):
        r = orch_with_cuff.answer(query="袖口是什么设计？", image_bytes=_VALID_JPEG)
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        for forbidden in ("state_dict", "checkpoint", "raw_logits", "tensor_data",
                          "temp_path", "file_path", "JFIF"):
            assert forbidden not in payload


# ── Part-to-task mapping tests ──────────────────────────────────────────────

class TestPartToAttrTasks:
    def test_neckline_maps_to_neck_design(self):
        from inference.serving.attribute_backend import _PART_TO_ATTR_TASKS
        tasks = _PART_TO_ATTR_TASKS.get("neckline", [])
        assert "neckline_design" in tasks or "neck_design" in tasks

    def test_sleeve_maps_to_sleeve_length(self):
        from inference.serving.attribute_backend import _PART_TO_ATTR_TASKS
        assert "sleeve_length" in _PART_TO_ATTR_TASKS.get("sleeve", [])

    def test_collar_maps_to_collar_design(self):
        from inference.serving.attribute_backend import _PART_TO_ATTR_TASKS
        assert "collar_design" in _PART_TO_ATTR_TASKS.get("collar", [])
