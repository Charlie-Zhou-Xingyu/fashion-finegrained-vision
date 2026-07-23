"""P0a.6 — VisionAttributeProvider tests (industrial)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from inference.serving.app import app
from inference.serving.vision_provider import (
    MockVisionAttributeProvider,
    get_vision_provider,
)


@pytest.fixture(scope="module")
def provider() -> MockVisionAttributeProvider:
    return MockVisionAttributeProvider()


# ── Unit tests ─────────────────────────────────────────────────────────────────


def test_no_image_no_attributes(provider):
    r = provider.extract()
    assert r.attributes == {}
    assert r.garment_instances == []
    assert r.regions == []
    assert any(w.code == "vision_input_missing" for w in r.warnings)
    assert r.meta["real_pipeline_enabled"] is False
    assert r.meta["has_image"] is False
    assert r.meta["has_image_url"] is False
    assert r.meta["has_image_bytes"] is False


def test_image_url_present(provider):
    r = provider.extract(image_url="http://example.com/img.jpg")
    assert r.attributes == {}
    assert any(w.code == "vision_provider_mock" for w in r.warnings)
    assert r.meta["has_image_url"] is True
    assert r.meta["has_image"] is False  # no raw `image` param
    assert r.used_tools == ["mock_vision_provider"]


def test_image_bytes_present(provider):
    r = provider.extract(image_bytes=b"fake_image_data")
    assert r.attributes == {}
    assert any(w.code == "vision_provider_mock" for w in r.warnings)
    assert r.meta["has_image_bytes"] is True


def test_provided_attributes_not_overridden(provider):
    """Case A: provided_attributes exists — do NOT override."""
    r = provider.extract(
        image_url="http://example.com/img.jpg",
        provided_attributes={"fabric": {"value": "棉"}},
    )
    assert r.attributes == {}  # not overridden
    assert r.warnings == []
    assert r.used_tools == []  # not used
    assert r.meta["provided_attributes_present"] is True


def test_regions_recorded(provider):
    r = provider.extract(image_url="http://x.com/a.jpg", regions=["collar", "sleeve"])
    assert r.meta["requested_regions"] == ["collar", "sleeve"]


def test_meta_fields(provider):
    r = provider.extract(image_url="http://x.com/a.jpg")
    assert r.meta["provider"] == "mock"
    assert r.meta["real_pipeline_enabled"] is False
    assert r.meta["has_image_url"] is True
    # used_tools is a top-level field, NOT nested in meta.
    assert r.used_tools == ["mock_vision_provider"]


def test_to_dict_json_safe(provider):
    r = provider.extract(image_url="http://example.com/img.jpg")
    d = r.to_dict()
    s = json.dumps(d, ensure_ascii=False)
    assert "vision_provider_mock" in s


def test_singleton():
    a = get_vision_provider()
    b = get_vision_provider()
    assert a is b


# ── Orchestrator integration tests ────────────────────────────────────────────


def test_orch_attributes_provided_not_calling_vision():
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(
        query="这件衣服是什么面料？",
        attributes={"fabric": {"value": "纯棉", "attribute_confidence": 0.92, "source": "merchant_input"}},
        image_url="http://example.com/img.jpg",
    )
    assert r.answer_type == "attribute_answer"
    assert "纯棉" in r.answer
    # No vision_provider_mock because attrs were provided.
    assert not any(w.code == "vision_provider_mock" for w in r.warnings)


def test_orch_image_url_no_attrs_attribute_query():
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(
        query="这件衣服是什么面料？",
        image_url="http://example.com/img.jpg",
    )
    assert r.answer_type == "attribute_answer"
    assert any(w.code == "vision_provider_mock" for w in r.warnings)
    assert any(w.code == "attribute_unavailable" for w in r.warnings)


def test_orch_knowledge_query_with_image_url():
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(query="纤维是什么", image_url="http://example.com/img.jpg")
    assert r.answer_type in ("knowledge_answer", "hybrid_answer")


def test_orch_no_attrs_no_image_knowledge_no_vision_warning():
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(query="纤维是什么")
    # Pure knowledge query with no image should NOT produce vision_input_missing.
    assert not any(w.code == "vision_input_missing" for w in r.warnings)


def test_orch_meta_vision_fields():
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(query="这件衣服是什么面料？", image_url="http://x.com/a.jpg")
    assert r.meta.get("vision_provider_used") is True
    assert r.meta.get("provided_attributes_used") is False
    assert r.meta.get("visual_attributes_present") is False
    assert r.to_dict()  # JSON-safe


def test_orch_warnings_no_duplicate():
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(query="这件衣服是什么面料？", image_url="http://x.com/a.jpg")
    codes = [w.code for w in r.warnings]
    assert len(codes) == len(set(codes))


# ── Endpoint tests ─────────────────────────────────────────────────────────────

client = TestClient(app)


def test_endpoint_qa_with_image_url():
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "image_url": "http://example.com/img.jpg",
    }).json()
    assert r["status"] == "success"
    assert any(w["code"] == "vision_provider_mock" for w in r["warnings"])


def test_endpoint_qa_with_attributes_and_image():
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "image_url": "http://example.com/img.jpg",
        "attributes": {"fabric": {"value": "棉", "attribute_confidence": 0.86, "source": "request_raw"}},
    }).json()
    assert r["status"] == "success"
    assert "棉" in r["data"]["answer"]
    assert not any(w["code"] == "vision_provider_mock" for w in r["warnings"])


def test_endpoint_qa_vision_meta_present():
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "image_url": "http://example.com/img.jpg",
    }).json()
    meta = r["data"].get("meta", {})
    assert meta is not None


def test_endpoint_warnings_top_level():
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "image_url": "http://example.com/img.jpg",
    }).json()
    assert "warnings" in r  # top-level
    data = r["data"]
    assert "warnings" not in data


def test_endpoint_rag_still_works():
    r = client.post("/v1/rag/retrieve", json={"query": "Fiber"}).json()
    assert len(r["data"]["hits"]) > 0
