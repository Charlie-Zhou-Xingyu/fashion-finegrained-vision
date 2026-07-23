"""P1.4a — Region localization QA tests.

Tests the /v1/mm/qa region_* query routing using mocked localized_regions —
no real 3.1.2 model, no GPU, no DINO, no Fashionpedia YOLO.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from inference.serving.app import app
from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.attribute_service import AttributeService
from inference.serving.qa_orchestrator import QaOrchestrator
from inference.serving.rag_service import RagService

client = TestClient(app)


# ── Sample localized regions (simulate 3.1.2 output) ───────────────────────

_SAMPLE_REGIONS = [
    {
        "region_id": "region_0",
        "part_type": "neckline",
        "part_group": "collar_area",
        "bbox": [100.0, 40.0, 220.0, 110.0],
        "confidence": 0.82,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
    {
        "region_id": "region_1",
        "part_type": "pocket",
        "part_group": "pocket_area",
        "bbox": [180.0, 260.0, 260.0, 340.0],
        "confidence": 0.76,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
    {
        "region_id": "region_2",
        "part_type": "zipper",
        "part_group": "closure",
        "bbox": [210.0, 300.0, 230.0, 480.0],
        "confidence": 0.65,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
    {
        "region_id": "region_3",
        "part_type": "cuff",
        "part_group": "sleeve_area",
        "bbox": [80.0, 200.0, 130.0, 250.0],
        "confidence": 0.71,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
]

# Low-confidence pocket (0.35, below 0.5 threshold).
_SAMPLE_REGIONS_LOW_CONF = [
    {
        "region_id": "region_0",
        "part_type": "pocket",
        "part_group": "pocket_area",
        "bbox": [180.0, 260.0, 260.0, 340.0],
        "confidence": 0.35,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
]

# Missing-confidence pocket.
_SAMPLE_REGIONS_NO_CONF = [
    {
        "region_id": "region_0",
        "part_type": "pocket",
        "part_group": "pocket_area",
        "bbox": [180.0, 260.0, 260.0, 340.0],
        "confidence": None,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
]

# Two pockets for count test.
_SAMPLE_REGIONS_TWO_POCKETS = [
    {
        "region_id": "region_0",
        "part_type": "pocket",
        "part_group": "pocket_area",
        "bbox": [100.0, 260.0, 180.0, 340.0],
        "confidence": 0.78,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
    {
        "region_id": "region_1",
        "part_type": "pocket",
        "part_group": "pocket_area",
        "bbox": [250.0, 260.0, 330.0, 340.0],
        "confidence": 0.72,
        "source": "mock_3_1_2",
        "backend": "mock",
        "mask_present": False,
        "mask_ref": None,
    },
]


# ── Fake vision provider with localized regions ────────────────────────────


class _FakeRegionVisionProvider:
    """Returns pre-canned VisionAttributeResult with localized_regions."""

    def __init__(self, regions=None, instances=None):
        from inference.serving.vision_provider import VisionAttributeResult
        import copy
        self._regions = copy.deepcopy(regions) if regions else []
        self._instances = copy.deepcopy(instances) if instances else []
        self._result = VisionAttributeResult(
            attributes={},
            garment_instances=self._instances,
            regions=self._regions,
            sources=[],
            warnings=[],
            used_tools=["fake_vision_provider"],
            meta={"provider": "test", "vision_backend": "test"},
        )

    def extract(self, **kwargs):
        return self._result


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_orchestrator(regions=None, instances=None):
    """Build a QaOrchestrator with a fake vision provider returning *regions*."""
    return QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=_FakeRegionVisionProvider(regions, instances),
    )


# ── Schema / normalization tests ───────────────────────────────────────────


class TestLocalizedRegionSchema:
    def test_valid_region_accepted(self):
        from inference.serving.schemas import LocalizedRegion
        r = LocalizedRegion(
            region_id="region_0", part_type="neckline",
            part_group="collar_area", bbox=[100, 40, 220, 110],
            confidence=0.82, source="mock_3_1_2", backend="mock",
        )
        assert r.region_id == "region_0"
        assert r.part_type == "neckline"

    def test_bbox_invalid_length_rejected(self):
        from inference.serving.schemas import LocalizedRegion
        with pytest.raises(Exception):
            LocalizedRegion(
                region_id="r0", part_type="neckline",
                bbox=[100, 40],  # too short
            )

    def test_summary_excludes_forbidden_fields(self):
        from inference.serving.schemas import LocalizedRegionSummary
        s = LocalizedRegionSummary(
            region_id="region_0", part_type="neckline",
            part_group="collar_area", bbox=[100, 40, 220, 110],
            confidence=0.82, source="mock_3_1_2", backend="mock",
        )
        d = s.model_dump()
        for forbidden in ("mask_present", "mask_ref", "mask", "crop", "image_bytes", "temp_path"):
            assert forbidden not in d, f"Leaked {forbidden} in summary"


# ── Query mapping tests ────────────────────────────────────────────────────


class TestRegionQueryMapper:
    def test_neckline(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("领口在哪里？") == "neckline"

    def test_collar(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("领子是什么") == "collar"

    def test_cuff(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("袖口在哪") == "cuff"

    def test_pocket(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("有没有口袋？") == "pocket"

    def test_zipper(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("拉链位置") == "zipper"

    def test_sequin(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("亮片装饰") == "sequin"

    def test_pattern(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("图案设计") == "pattern"

    def test_decoration(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("装饰细节") == "decoration"

    def test_no_match(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("这是什么颜色？") is None

    def test_empty_query(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("") is None

    def test_lapel(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("翻领设计") == "lapel"

    def test_hood(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("帽子在哪") == "hood"

    def test_button(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        assert extract_requested_region_part("扣子") == "button"

    def test_with_group(self):
        from inference.serving.region_query_mapper import extract_requested_region_part_with_group
        result = extract_requested_region_part_with_group("领口在哪里？")
        assert result == ("neckline", "collar_area")

    def test_first_match_wins(self):
        from inference.serving.region_query_mapper import extract_requested_region_part
        # "腰带扣" contains "腰带" and "扣环" keywords — specific "腰带扣" should win
        assert extract_requested_region_part("腰带扣") == "buckle"


# ── Intent classification tests ────────────────────────────────────────────


class TestRegionIntentClassification:
    @pytest.fixture(scope="class")
    def classifier(self):
        return RuleIntentClassifier()

    def test_location(self, classifier):
        r = classifier.classify("领口在哪里？")
        assert r.primary_intent == "region_location_query"

    def test_existence(self, classifier):
        r = classifier.classify("有没有口袋？")
        assert r.primary_intent == "region_existence_query"

    def test_detail(self, classifier):
        r = classifier.classify("这件衣服有哪些细节？")
        assert r.primary_intent == "region_detail_query"

    def test_count(self, classifier):
        r = classifier.classify("有几个口袋？")
        assert r.primary_intent == "region_count_query"

    def test_attribute(self, classifier):
        r = classifier.classify("袖口是什么设计？")
        assert r.primary_intent == "region_attribute_query"

    def test_zipper_location(self, classifier):
        r = classifier.classify("拉链在哪里？")
        assert r.primary_intent == "region_location_query"

    def test_pocket_existence_variant(self, classifier):
        r = classifier.classify("是否有口袋？")
        assert r.primary_intent == "region_existence_query"

    def test_detail_variant(self, classifier):
        r = classifier.classify("有什么装饰？")
        assert r.primary_intent == "region_detail_query"

    def test_count_long(self, classifier):
        r = classifier.classify("有多少个扣子？")
        assert r.primary_intent == "region_count_query"

    # Regression: existing intents must not be broken.
    def test_garment_fabric_regression(self, classifier):
        r = classifier.classify("这件衣服是什么面料？")
        assert r.primary_intent == "attribute_query"

    def test_garment_count_regression(self, classifier):
        r = classifier.classify("图里有几件衣服？")
        assert r.primary_intent == "visual_instance_query"

    def test_garment_location_regression(self, classifier):
        r = classifier.classify("上衣在哪里？")
        assert r.primary_intent == "visual_instance_query"


# ── QA behavior tests (orchestrator, no endpoint) ──────────────────────────


@pytest.fixture(scope="module")
def orch():
    return _make_orchestrator(regions=_SAMPLE_REGIONS)


@pytest.fixture(scope="module")
def orch_empty():
    return _make_orchestrator(regions=[])


@pytest.fixture(scope="module")
def orch_low_conf():
    return _make_orchestrator(regions=_SAMPLE_REGIONS_LOW_CONF)


@pytest.fixture(scope="module")
def orch_no_conf():
    return _make_orchestrator(regions=_SAMPLE_REGIONS_NO_CONF)


@pytest.fixture(scope="module")
def orch_two_pockets():
    return _make_orchestrator(regions=_SAMPLE_REGIONS_TWO_POCKETS)


class TestRegionQABehavior:
    # -- Location present --
    def test_location_present(self, orch):
        r = orch.answer(query="领口在哪里？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert "领口" in r.answer or "neckline" in r.answer
        assert "bbox" in r.answer or "检测到" in r.answer
        assert any("region_query_answer" in t for t in r.used_tools)
        assert "regions_used" in r.meta

    # -- Existence present --
    def test_existence_present(self, orch):
        r = orch.answer(query="有没有口袋？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert "检测到" in r.answer
        assert "pocket" in r.answer or "口袋" in r.answer

    # -- Detail summary --
    def test_detail_summary(self, orch):
        r = orch.answer(query="这件衣服有哪些细节？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert "检测到" in r.answer or "局部" in r.answer
        # Should list neckline, pocket, zipper, cuff
        summary = r.meta.get("localized_regions_summary", [])
        assert isinstance(summary, list)
        assert len(summary) >= 1

    # -- Count --
    def test_count_two_pockets(self, orch_two_pockets):
        r = orch_two_pockets.answer(query="有几个口袋？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert "2" in r.answer
        assert "pocket" in r.answer or "口袋" in r.answer

    # -- Attribute placeholder --
    def test_attribute_placeholder(self, orch):
        r = orch.answer(query="袖口是什么设计？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        # P1.4f: attribute backend is disabled by default → placeholder message.
        assert any(w.code in ("region_attribute_not_integrated", "attribute_backend_disabled") for w in r.warnings)

    # -- Missing region --
    def test_missing_region(self, orch):
        # "蝴蝶结" is NOT in _SAMPLE_REGIONS — should report not found.
        r = orch.answer(query="蝴蝶结在哪里？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert "没有可靠" in r.answer or "未能" in r.answer

    # -- Low confidence --
    def test_low_confidence(self, orch_low_conf):
        r = orch_low_conf.answer(query="有没有口袋？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert any(w.code == "region_low_confidence" for w in r.warnings)
        # Must NOT make a definitive claim.
        assert "不能确认" in r.answer or "较低" in r.answer or "可能" in r.answer

    # -- Missing confidence --
    def test_missing_confidence(self, orch_no_conf):
        r = orch_no_conf.answer(query="有哪些细节？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert any(w.code == "region_confidence_missing" for w in r.warnings)

    # -- No localized regions --
    def test_no_regions(self, orch_empty):
        r = orch_empty.answer(query="领口在哪里？", image_bytes="dGVzdA==")
        assert r.answer_type == "region_query_answer"
        assert any(w.code == "localized_regions_unavailable" for w in r.warnings)

    # -- No image (mock doesn't return regions) --
    def test_no_image_no_regions(self):
        orch_no_img = _make_orchestrator(regions=[])
        r = orch_no_img.answer(query="领口在哪里？")
        # Without image, build_vision_context won't call vision provider.
        assert any(w.code == "localized_regions_unavailable" for w in r.warnings)


# ── No-leak tests ──────────────────────────────────────────────────────────


class TestRegionNoLeak:
    def test_response_no_temp_path(self, orch):
        r = orch.answer(query="领口在哪里？", image_bytes="dGVzdA==")
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        assert "D:\\\\" not in payload
        assert "/tmp/" not in payload
        assert "\\temp" not in payload

    def test_response_no_image_bytes(self, orch):
        r = orch.answer(query="领口在哪里？", image_bytes="dGVzdA==")
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        assert "dGVzdA" not in payload

    def test_response_no_mask_bitmap(self, orch):
        r = orch.answer(query="有哪些细节？", image_bytes="dGVzdA==")
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        assert "mask_bitmap" not in payload
        assert "mask_data" not in payload
        assert "mask_path" not in payload

    def test_response_no_crop(self, orch):
        r = orch.answer(query="有哪些细节？", image_bytes="dGVzdA==")
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        assert "crop_path" not in payload
        assert "crop_data" not in payload
        assert "crop_bytes" not in payload

    def test_summary_safe_keys(self, orch):
        r = orch.answer(query="有哪些细节？", image_bytes="dGVzdA==")
        summary = r.meta.get("localized_regions_summary", [])
        SAFE = {"region_id", "part_type", "part_group", "bbox", "confidence", "source", "backend", "instance_id"}
        for region in summary:
            assert set(region.keys()).issubset(SAFE), f"Leaked: {set(region.keys()) - SAFE}"
            for forbidden in ("mask_present", "mask_ref", "mask", "crop", "image_bytes", "temp_path"):
                assert forbidden not in region, f"Leaked {forbidden} in region summary"

    def test_sources_have_localized_region_type(self, orch):
        r = orch.answer(query="有哪些细节？", image_bytes="dGVzdA==")
        lr_sources = [s for s in r.sources if s.get("type") == "localized_region"]
        assert len(lr_sources) > 0
        # No mask/path in source metadata
        for s in lr_sources:
            meta = s.get("metadata", {})
            for forbidden in ("mask", "mask_path", "crop", "crop_path", "image_bytes"):
                assert forbidden not in meta


# ── Regression: 3.1.1 garment instance QA still works ─────────────────────


_SAMPLE_INSTANCES = [
    {
        "instance_id": "inst_0",
        "category": "top",
        "fine_class_name": "short sleeve top",
        "bbox": [120.0, 80.0, 300.0, 350.0],
        "confidence": 0.93,
        "mask_present": True,
    },
]


class TestRegressionGarmentInstanceQA:
    @pytest.fixture(scope="class")
    def orch_both(self):
        """Orchestrator with both garment_instances AND localized_regions."""
        return _make_orchestrator(
            regions=_SAMPLE_REGIONS,
            instances=_SAMPLE_INSTANCES,
        )

    def test_garment_count_still_works(self, orch_both):
        r = orch_both.answer(query="图里有几件衣服？", image_bytes="dGVzdA==")
        assert r.answer_type == "visual_instance_answer"
        assert "1件" in r.answer or "检测到" in r.answer

    def test_garment_detection_still_works(self, orch_both):
        r = orch_both.answer(query="图中检测到了什么？", image_bytes="dGVzdA==")
        assert r.answer_type == "visual_instance_answer"

    def test_existing_attribute_query_still_works(self, orch_both):
        r = orch_both.answer(
            query="这件衣服是什么面料？",
            attributes={"fabric": {"value": "纯棉", "attribute_confidence": 0.92, "source": "merchant_input"}},
        )
        assert r.answer_type == "attribute_answer"
        assert "纯棉" in r.answer


# ── Endpoint integration tests ─────────────────────────────────────────────


@pytest.fixture
def _patch_region_orchestrator(monkeypatch):
    """Replace the app's orchestrator singleton with a region-aware fake."""
    orch = _make_orchestrator(regions=_SAMPLE_REGIONS)
    monkeypatch.setattr("inference.serving.app.get_qa_orchestrator", lambda: orch)
    monkeypatch.setattr("inference.serving.qa_orchestrator._orchestrator", orch)
    return orch


class TestRegionEndpoint:
    def test_location_endpoint(self, _patch_region_orchestrator):
        r = client.post("/v1/mm/qa", json={
            "query": "领口在哪里？",
            "image_bytes": "dGVzdA==",
        }).json()
        assert r["status"] == "success"
        assert r["data"]["answer_type"] == "region_query_answer"
        assert "bbox" in r["data"]["answer"] or "检测到" in r["data"]["answer"]

    def test_existence_endpoint(self, _patch_region_orchestrator):
        r = client.post("/v1/mm/qa", json={
            "query": "有没有口袋？",
            "image_bytes": "dGVzdA==",
        }).json()
        assert r["status"] == "success"
        assert "检测到" in r["data"]["answer"]

    def test_detail_endpoint(self, _patch_region_orchestrator):
        r = client.post("/v1/mm/qa", json={
            "query": "这件衣服有哪些细节？",
            "image_bytes": "dGVzdA==",
        }).json()
        assert r["status"] == "success"
        assert r["data"]["answer_type"] == "region_query_answer"

    def test_count_endpoint(self, _patch_region_orchestrator):
        r = client.post("/v1/mm/qa", json={
            "query": "有几个口袋？",
            "image_bytes": "dGVzdA==",
        }).json()
        assert r["status"] == "success"

    def test_attribute_endpoint(self, _patch_region_orchestrator):
        r = client.post("/v1/mm/qa", json={
            "query": "袖口是什么设计？",
            "image_bytes": "dGVzdA==",
        }).json()
        assert r["status"] == "success"
        assert any(w["code"] in ("region_attribute_not_integrated", "attribute_backend_disabled") for w in r["warnings"])

    def test_meta_has_summary(self, _patch_region_orchestrator):
        r = client.post("/v1/mm/qa", json={
            "query": "领口在哪里？",
            "image_bytes": "dGVzdA==",
        }).json()
        meta = r["data"].get("meta", {})
        assert "localized_regions_summary" in meta
        assert "regions_used" in meta

    def test_missing_region_endpoint(self, _patch_region_orchestrator):
        r = client.post("/v1/mm/qa", json={
            "query": "鞋子在哪里？",
            "image_bytes": "dGVzdA==",
        }).json()
        assert r["status"] == "success"
        assert "没有可靠" in r["data"]["answer"] or "未能" in r["data"]["answer"]

    def test_no_regions_endpoint(self, monkeypatch):
        orch_empty = _make_orchestrator(regions=[])
        monkeypatch.setattr("inference.serving.app.get_qa_orchestrator", lambda: orch_empty)
        monkeypatch.setattr("inference.serving.qa_orchestrator._orchestrator", orch_empty)
        r = client.post("/v1/mm/qa", json={
            "query": "领口在哪里？",
            "image_bytes": "dGVzdA==",
        }).json()
        assert r["status"] == "success"
        assert any(w["code"] == "localized_regions_unavailable" for w in r["warnings"])
