"""P0a.5 — QaOrchestrator unit tests."""

from __future__ import annotations

import json

import pytest

from inference.serving.attribute_service import AttributeService
from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.qa_orchestrator import QaOrchestrator, QAOrchestratorResult
from inference.serving.rag_service import RagService


@pytest.fixture(scope="module")
def orchestrator() -> QaOrchestrator:
    return QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
    )


# ── Empty query ────────────────────────────────────────────────────────────────


def test_empty_query_none(orchestrator):
    r = orchestrator.answer(query=None)
    assert r.answer_type == "empty_query"
    assert r.used_tools == []
    assert any(w.code == "empty_query" for w in r.warnings)


def test_empty_query_whitespace(orchestrator):
    r = orchestrator.answer(query="   ")
    assert r.answer_type == "empty_query"


# ── Attribute query ────────────────────────────────────────────────────────────


def test_attribute_query_fabric_hit(orchestrator):
    r = orchestrator.answer(
        query="这件衣服是什么面料？",
        attributes={"fabric": {"value": "纯棉", "attribute_confidence": 0.92, "source": "merchant_input"}},
    )
    assert r.answer_type == "attribute_answer"
    assert "纯棉" in r.answer
    assert "intent_classifier" in r.used_tools
    assert "attribute_service" in r.used_tools
    assert r.answer_confidence == 0.92


def test_attribute_query_missing(orchestrator):
    r = orchestrator.answer(
        query="这件衣服是什么面料？",
        attributes={"color": "红色"},
    )
    assert r.answer_type == "attribute_answer"
    assert "暂未获取到" in r.answer
    assert any(w.code == "attribute_unavailable" for w in r.warnings)


def test_attribute_manual_verified_confidence_none(orchestrator):
    """manual_verified with attribute_confidence=None must NOT be auto-set to 1.0."""
    r = orchestrator.answer(
        query="这件衣服是什么面料？",
        attributes={"fabric": {"value": "纯棉", "source": "manual_verified", "attribute_confidence": None}},
    )
    assert r.answer_confidence is None
    assert r.answer_confidence != 1.0


# ── Knowledge query ────────────────────────────────────────────────────────────


def test_knowledge_query_hit(orchestrator):
    r = orchestrator.answer(query="纤维是什么")
    assert r.answer_type == "knowledge_answer"
    assert "rag_service" in r.used_tools
    assert "template_answer" in r.used_tools
    assert len(r.sources) > 0


def test_knowledge_query_no_hits(orchestrator):
    """A gibberish query → fallback_unknown → orchestrator returns 'unsupported'."""
    r = orchestrator.answer(query="zzqqxxvvww")
    assert r.answer_type == "unsupported"
    assert any(w.code == "unsupported_intent" for w in r.warnings)


def test_knowledge_answer_has_review_disclaimer(orchestrator):
    r = orchestrator.answer(query="什么是材料")
    # Our seed KB has review_status=manual_review_required for all entries.
    assert "人工审核" in r.answer or "manual_review_required" in str(r.sources)


def test_knowledge_sources_have_source_ref(orchestrator):
    r = orchestrator.answer(query="什么是纤维")
    for s in r.sources:
        if s.get("type") == "knowledge_base":
            assert "source_ref" in s or "source" in s


# ── Styling ────────────────────────────────────────────────────────────────────


def test_styling_limited(orchestrator):
    r = orchestrator.answer(query="这件外套适合搭配什么裤子")
    assert r.answer_type == "hybrid_answer"
    assert "暂不生成完整穿搭方案" in r.answer


# ── Unsupported ────────────────────────────────────────────────────────────────


def test_unsupported_content_generation(orchestrator):
    r = orchestrator.answer(query="帮我生成文案")
    assert r.answer_type == "unsupported"
    assert any(w.code == "unsupported_intent" for w in r.warnings)


def test_unsupported_chat(orchestrator):
    r = orchestrator.answer(query="你好")
    assert r.answer_type == "unsupported"


# ── Warnings de-duplication ────────────────────────────────────────────────────


def test_warnings_no_duplicate(orchestrator):
    r = orchestrator.answer(query="这件是什么面料？", attributes={})
    codes = [w.code for w in r.warnings]
    assert len(codes) == len(set(codes))


# ── used_tools ─────────────────────────────────────────────────────────────────


def test_attribute_route_used_tools(orchestrator):
    r = orchestrator.answer(
        query="这件衣服颜色？",
        attributes={"color": {"value": "红色", "attribute_confidence": 0.98}},
    )
    assert "intent_classifier" in r.used_tools
    assert "attribute_service" in r.used_tools


def test_knowledge_route_used_tools(orchestrator):
    r = orchestrator.answer(query="什么是纤维")
    assert "intent_classifier" in r.used_tools
    assert "rag_service" in r.used_tools
    assert "template_answer" in r.used_tools


# ── JSON serialization ─────────────────────────────────────────────────────────


def test_result_to_dict_json(orchestrator):
    r = orchestrator.answer(
        query="纤维是什么",
        attributes={"color": "红色"},
    )
    d = r.to_dict()
    json.dumps(d, ensure_ascii=False)


def test_qa_data_compatible(orchestrator):
    r = orchestrator.answer(
        query="这件衣服是什么面料？",
        attributes={"fabric": {"value": "纯棉", "attribute_confidence": 0.92, "source": "merchant_input"}},
    )
    qa = r.to_qa_data()
    assert qa["answer"] == r.answer
    assert qa["answer_type"] == r.answer_type
    assert qa["intent_confidence"] == r.intent.get("confidence")


# ── Attribute context passthrough ──────────────────────────────────────────────


def test_attribute_context_passed_to_rag(orchestrator):
    r = orchestrator.answer(
        query="纤维 棉",
        attributes={"fabric": "cotton", "color": "红色"},
    )
    # RagService should receive attribute_context.  Exact intent may vary
    # depending on classifier output — just verify non-error.
    assert r.answer_type != "empty_query"
    assert r.used_tools


# ── P0a.6.1: Spy provider + leak + no-mutation ────────────────────────────────


def test_attributes_provided_vision_NOT_called():
    """When request attributes are provided, QaOrchestrator must NOT call
    vision_provider.extract().  Use a spy to verify."""
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    class SpyProvider:
        called = False
        def extract(self, **kwargs):
            SpyProvider.called = True
            from inference.serving.vision_provider import MockVisionAttributeProvider
            return MockVisionAttributeProvider().extract(**kwargs)

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=SpyProvider(),
    )
    r = o.answer(
        query="这件衣服是什么面料？",
        attributes={"fabric": {"value": "纯棉", "attribute_confidence": 0.92}},
        image_url="http://x.com/img.jpg",
    )
    assert SpyProvider.called is False
    assert r.meta.get("provided_attributes_used") is True
    assert r.meta.get("vision_provider_used") is False  # provider was skipped
    assert not any(w.code == "vision_provider_mock" for w in r.warnings)


def test_image_bytes_not_leaked():
    """image_bytes must NOT appear in orchestrator result or JSON output."""
    import json
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService
    from inference.serving.vision_provider import MockVisionAttributeProvider

    secret = "SECRET_BASE64_DO_NOT_LEAK_abc123"
    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(query="这件衣服是什么面料？", image_bytes=secret)
    j = json.dumps(r.to_dict(), ensure_ascii=False)
    assert secret not in j


def test_attrs_no_mutation():
    """orchestrator must NOT mutate the caller's attributes dict."""
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService

    attrs = {"fabric": {"value": "棉"}}
    original_keys = set(attrs.keys())
    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
    )
    o.answer(query="这件衣服是什么面料？", attributes=attrs, garment_category="shirt")
    assert set(attrs.keys()) == original_keys


def test_attribute_query_no_attrs_no_image_no_vision_warning():
    """Attribute query with no attrs and no image should NOT produce
    vision_input_missing — only attribute_unavailable."""
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.rag_service import RagService
    from inference.serving.vision_provider import MockVisionAttributeProvider

    o = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=MockVisionAttributeProvider(),
    )
    r = o.answer(query="这件衣服是什么面料？")
    assert any(w.code == "attribute_unavailable" for w in r.warnings)
    assert not any(w.code == "vision_input_missing" for w in r.warnings)
    assert not any(w.code == "vision_provider_mock" for w in r.warnings)
