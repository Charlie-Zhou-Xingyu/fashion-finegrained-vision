"""Tests for P0a.1 API schemas — Pydantic v2 model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference.serving.schemas import (
    AttributeValue,
    DialogueTurn,
    IntentClassifyData,
    IntentClassifyRequest,
    MerchantContentData,
    MerchantContentRequest,
    MultimodalQAData,
    MultimodalQARequest,
    RAGResultItem,
    RAGRetrieveData,
    RAGRetrieveRequest,
    ResponseMeta,
    SourceItem,
    UnifiedResponse,
    WarningItem,
    WarningSeverity,
    build_response,
    make_request_id,
)


# ── UnifiedResponse ──────────────────────────────────────────────────────────────


def test_unified_response_serializes():
    resp = build_response({"key": "value"})
    d = resp.model_dump()
    assert d["status"] == "success"
    assert d["request_id"].startswith("req_")
    assert d["data"] == {"key": "value"}
    assert d["elapsed_ms"] == 0.0
    assert d["used_tools"] == []
    assert d["warnings"] == []


def test_unified_response_with_warning():
    w = WarningItem(code="test", scope="test", message="test", severity=WarningSeverity.info)
    resp = build_response({}, warnings=[w])
    assert len(resp.warnings) == 1
    assert resp.warnings[0].code == "test"


def test_request_id_custom():
    resp = build_response({}, request_id="custom-123")
    assert resp.request_id == "custom-123"


# ── MultimodalQARequest ──────────────────────────────────────────────────────────


def test_qa_request_minimal_valid():
    """Only query is required — everything else is optional."""
    req = MultimodalQARequest(query="这件衣服是什么面料？")
    assert req.query == "这件衣服是什么面料？"
    assert req.product_id is None
    assert req.image_url is None
    assert req.attributes == {}


def test_qa_request_missing_query_fails():
    with pytest.raises(ValidationError):
        MultimodalQARequest()


def test_qa_request_empty_query_fails():
    with pytest.raises(ValidationError):
        MultimodalQARequest(query="")


def test_qa_request_pure_knowledge():
    """Pure knowledge QA without product_id or image should be valid."""
    req = MultimodalQARequest(query="纯棉和麻料哪个更适合夏天？")
    assert req.product_id is None
    assert req.image_url is None


def test_qa_request_max_answer_length_bounds():
    with pytest.raises(ValidationError):
        MultimodalQARequest(query="test", max_answer_length=0)
    with pytest.raises(ValidationError):
        MultimodalQARequest(query="test", max_answer_length=2001)


def test_qa_request_with_attributes():
    req = MultimodalQARequest(
        query="面料？",
        attributes={"fabric": {"value": "棉质", "attribute_confidence": 0.9}},
    )
    assert req.attributes["fabric"]["value"] == "棉质"


# ── AttributeValue ───────────────────────────────────────────────────────────────


def test_attribute_value_confidence_range():
    """attribute_confidence must be in [0, 1]."""
    AttributeValue(attribute_confidence=0.0)
    AttributeValue(attribute_confidence=1.0)
    AttributeValue(attribute_confidence=None)
    with pytest.raises(ValidationError):
        AttributeValue(attribute_confidence=-0.1)
    with pytest.raises(ValidationError):
        AttributeValue(attribute_confidence=1.1)


def test_attribute_value_has_no_bare_confidence():
    """The model must not expose a generic 'confidence' field."""
    av = AttributeValue(value="test")
    d = av.model_dump()
    assert "confidence" not in d
    assert "attribute_confidence" in d


# ── WarningItem ──────────────────────────────────────────────────────────────────


def test_warning_item_minimal():
    w = WarningItem(code="test", scope="test", message="ok", severity=WarningSeverity.warn)
    assert w.code == "test"
    assert w.term is None


def test_warning_item_full():
    w = WarningItem(
        code="exaggeration_risk",
        scope="selling_point",
        message="Exaggerated claim detected",
        severity=WarningSeverity.warn,
        term="100%纯棉",
        action="rewritten",
        reason="composition_verified=false",
    )
    d = w.model_dump()
    assert d["term"] == "100%纯棉"
    assert d["action"] == "rewritten"


# ── IntentClassifyRequest / Data ─────────────────────────────────────────────────


def test_intent_classify_request_valid():
    req = IntentClassifyRequest(query="这件适合搭配什么裤子？")
    assert req.query is not None


def test_intent_classify_request_missing_query_fails():
    with pytest.raises(ValidationError):
        IntentClassifyRequest()


def test_intent_classify_data_defaults():
    data = IntentClassifyData()
    assert data.primary_intent == "fallback_unknown"
    assert data.intent_confidence == 0.0
    assert data.classifier_level == "mock"


# ── RAGRetrieveRequest / Data ────────────────────────────────────────────────────


def test_rag_request_valid():
    req = RAGRetrieveRequest(query="纯棉面料")
    assert req.top_k == 5
    assert req.use_reranker is False


def test_rag_request_top_k_bounds():
    with pytest.raises(ValidationError):
        RAGRetrieveRequest(query="test", top_k=0)
    with pytest.raises(ValidationError):
        RAGRetrieveRequest(query="test", top_k=51)


def test_rag_data_defaults():
    data = RAGRetrieveData()
    assert data.hits == []
    assert data.query == ""
    assert data.meta == {}


# ── MerchantContentRequest / Data ────────────────────────────────────────────────


def test_merchant_content_request_defaults():
    req = MerchantContentRequest()
    assert req.style == "professional"
    assert req.generate_options.attribute_tags is True


def test_merchant_content_data_defaults():
    data = MerchantContentData()
    assert data.content_type == "selling_points"
    assert data.generated_content is None
    assert data.blocked_claims == []
    assert data.meta == {}


# ── Meta ─────────────────────────────────────────────────────────────────────────


def test_response_meta_defaults():
    meta = ResponseMeta()
    assert meta.schema_version == "1.0.0"
    assert meta.cache_hit is False


# ── SourceItem ───────────────────────────────────────────────────────────────────


def test_source_item_disambiguated_confidence():
    si = SourceItem(type="product_attribute", field="fabric",
                    attribute_confidence=0.92)
    d = si.model_dump()
    assert "confidence" not in d
    assert d["attribute_confidence"] == 0.92
    assert d["rag_score"] is None


# ── MultimodalQAData ─────────────────────────────────────────────────────────────


def test_qa_data_need_image_clarification():
    data = MultimodalQAData(
        answer=None,
        answer_type="clarification",
        need_image=True,
        clarification="请上传图片",
    )
    assert data.answer is None
    assert data.need_image is True


# ── make_request_id ──────────────────────────────────────────────────────────────


def test_make_request_id():
    rid = make_request_id()
    assert rid.startswith("req_")
    assert len(rid) == 16  # "req_" + 12 hex chars
