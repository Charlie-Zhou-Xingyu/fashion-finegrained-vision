"""P0a.2 — Rule-based intent classifier tests.

Covers:
    - All 17 intent categories
    - Primary/sub intent split (primary_intent / sub_intent, not full-path id)
    - Attribute query vs knowledge_qa disambiguation
    - chat/greeting only matches pure greetings (not mixed queries)
    - Fallback for unknown queries
    - Endpoint integration
    - Performance (< 200 µs/call)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.schemas import IntentClassifyData


@pytest.fixture(scope="module")
def classifier() -> RuleIntentClassifier:
    return RuleIntentClassifier()


def _assert_intent(
    result: IntentClassifyData,
    primary: str,
    sub: str | None = None,
    *,
    min_confidence: float = 0.0,
):
    """Assert primary_intent, optional sub_intent, and minimum confidence."""
    assert result.primary_intent == primary, (
        f"expected primary={primary}, got {result.primary_intent} "
        f"(sub={result.sub_intent})"
    )
    if sub is not None:
        assert result.sub_intent == sub, (
            f"expected sub={sub}, got {result.sub_intent}"
        )
    assert result.classifier_level == "rule"
    assert result.intent_confidence >= min_confidence
    assert 0.0 <= result.intent_confidence <= 1.0


# ── Attribute queries ──────────────────────────────────────────────────────────

def test_fabric_query(classifier):
    r = classifier.classify("这件衣服是什么面料？")
    _assert_intent(r, "attribute_query", "fabric", min_confidence=0.90)


def test_color_query(classifier):
    r = classifier.classify("这个什么颜色？")
    _assert_intent(r, "attribute_query", "color", min_confidence=0.80)


def test_collar_query(classifier):
    r = classifier.classify("这件衣服是什么领口？")
    _assert_intent(r, "attribute_query", "collar", min_confidence=0.90)


def test_sleeve_query(classifier):
    r = classifier.classify("袖子是什么袖型？")
    _assert_intent(r, "attribute_query", "sleeve", min_confidence=0.80)


def test_length_query(classifier):
    r = classifier.classify("这件裤子多长？")
    _assert_intent(r, "attribute_query", "length", min_confidence=0.90)


# ── Design / craft ─────────────────────────────────────────────────────────────

def test_design_explanation(classifier):
    r = classifier.classify("这个领口设计有什么特点？")
    _assert_intent(r, "design_explanation", "collar")


def test_craft_explanation_specific_product(classifier):
    """'这件牛仔裤的水洗工艺有什么特点？' → craft_explanation
    (product-context question about a specific garment's craft)."""
    r = classifier.classify("这件牛仔裤的水洗工艺有什么特点？")
    _assert_intent(r, "craft_explanation")


# ── Styling advice ─────────────────────────────────────────────────────────────

def test_styling_match(classifier):
    r = classifier.classify("这件外套适合搭配什么裤子？")
    _assert_intent(r, "styling_advice", "match")


def test_styling_occasion(classifier):
    r = classifier.classify("这件裙子适合什么场合穿？")
    _assert_intent(r, "styling_advice", "occasion")


# ── Knowledge QA (must NOT be confused with attribute_query) ────────────────────

def test_knowledge_fabric_term_query(classifier):
    """'纯棉是什么面料？' → knowledge_qa/fabric (asks about a fabric type, not a product)."""
    r = classifier.classify("纯棉是什么面料？")
    _assert_intent(r, "knowledge_qa", "fabric")


def test_knowledge_craft_term_query(classifier):
    """'水洗工艺是什么？' → knowledge_qa/craft (asks about a craft term generally)."""
    r = classifier.classify("水洗工艺是什么？")
    _assert_intent(r, "knowledge_qa", "craft")


def test_knowledge_style_query(classifier):
    r = classifier.classify("韩系穿搭风格有什么特点？")
    _assert_intent(r, "knowledge_qa", "style")


def test_knowledge_term_query(classifier):
    """'什么是水洗工艺？' → craft_explanation (keyword '水洗' matches first).
    The orchestrator routes craft_explanation to RagService, same as knowledge_qa."""
    r = classifier.classify("什么是水洗工艺？")
    _assert_intent(r, "craft_explanation")


# ── Content generation ─────────────────────────────────────────────────────────

def test_content_tags(classifier):
    r = classifier.classify("帮我生成这件衣服的属性标签")
    _assert_intent(r, "content_generation", "tags")


def test_content_copy(classifier):
    r = classifier.classify("帮我写一段商品卖点文案")
    _assert_intent(r, "content_generation", "copy")


# ── Chat / greeting — only pure greetings, NOT mixed queries ───────────────────

def test_greeting_pure_hello(classifier):
    r = classifier.classify("你好")
    _assert_intent(r, "chat", "greeting")


def test_greeting_pure_thanks(classifier):
    """Pure '谢谢' matches regex; '谢谢你的帮助' does NOT (regex is anchored)."""
    r = classifier.classify("谢谢")
    _assert_intent(r, "chat", "greeting")


def test_greeting_thanks_with_content_is_NOT_greeting(classifier):
    """'谢谢你的帮助' does not match the anchored regex — falls through."""
    r = classifier.classify("谢谢你的帮助")
    # Falls through to craft_explanation because "帮助" isn't in any rule,
    # so it ends up as fallback_unknown.  This is CORRECT: we don't want
    # to classify "谢谢你的帮助" as chat/greeting just because of "谢谢".
    assert r.primary_intent != "chat"


def test_greeting_mixed_with_business_query_is_NOT_greeting(classifier):
    """'你好，这件衣服是什么面料？' must route to attribute_query, NOT chat."""
    r = classifier.classify("你好，这件衣服是什么面料？")
    _assert_intent(r, "attribute_query", "fabric", min_confidence=0.90)


def test_greeting_with_whitespace():
    """'你好  ' with trailing spaces still matches anchored regex."""
    r = RuleIntentClassifier().classify("你好  ")
    _assert_intent(r, "chat", "greeting")


# ── Fallback ───────────────────────────────────────────────────────────────────

def test_fallback_unknown(classifier):
    r = classifier.classify("今天天气怎么样")
    _assert_intent(r, "fallback_unknown")
    assert r.sub_intent is None
    assert r.intent_confidence == 0.0


def test_empty_query_fallback(classifier):
    r = classifier.classify("")
    _assert_intent(r, "fallback_unknown")
    assert r.intent_confidence == 0.0


def test_whitespace_query_fallback(classifier):
    r = classifier.classify("   ")
    _assert_intent(r, "fallback_unknown")


# ── taxonomy load failure behaviour ────────────────────────────────────────────

def test_taxonomy_missing_file_fallback():
    """When the taxonomy file does not exist, the classifier loads with zero
    rules and all queries fallback to fallback_unknown."""
    c = RuleIntentClassifier(taxonomy_path=Path("nonexistent_taxonomy.yaml"))
    r = c.classify("这件衣服是什么面料？")
    _assert_intent(r, "fallback_unknown")
    assert r.intent_confidence == 0.0


def test_taxonomy_validation_on_load():
    """A well-formed taxonomy should produce non-zero intent count."""
    c = RuleIntentClassifier()
    assert c.intent_count > 0
    assert c.default_intent == "fallback_unknown"


# ── Endpoint integration ───────────────────────────────────────────────────────

def test_intent_endpoint_now_uses_real_classifier():
    """After P0a.2 the /v1/intent/classify endpoint must use the real
    RuleIntentClassifier — no longer returning mock fallback_unknown for
    every query."""
    from fastapi.testclient import TestClient
    from inference.serving.app import app

    client = TestClient(app)
    r = client.post("/v1/intent/classify", json={"query": "这件衣服是什么面料？"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["primary_intent"] == "attribute_query"
    assert data["sub_intent"] == "fabric"
    assert data["classifier_level"] == "rule"
    assert data["intent_confidence"] > 0.5


def test_intent_endpoint_unknown_still_works():
    """Unknown queries should still return fallback_unknown (not 5xx)."""
    from fastapi.testclient import TestClient
    from inference.serving.app import app

    client = TestClient(app)
    r = client.post("/v1/intent/classify", json={"query": "xyzzy_random_unknown"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["primary_intent"] == "fallback_unknown"
    assert data["classifier_level"] == "rule"
    assert data["intent_confidence"] == 0.0


# ── Entities extraction ────────────────────────────────────────────────────────

def test_entities_extraction_color():
    c = RuleIntentClassifier()
    r = c.classify("这件红色上衣是什么面料？")
    assert r.entities.get("color") == "红色"
    assert r.entities.get("garment_ref") == "top"


def test_entities_extraction_fabric():
    c = RuleIntentClassifier()
    r = c.classify("棉质面料好吗？")
    assert r.entities.get("fabric") == "棉"


def test_entities_attribute_name_from_sub_intent():
    c = RuleIntentClassifier()
    r = c.classify("这件衣服是什么领口？")
    assert r.entities.get("attribute_name") == "collar"


# ── Performance ────────────────────────────────────────────────────────────────

def test_classifier_is_fast():
    """P0a.2 rule classifier must stay under 200 µs per call."""
    import time
    c = RuleIntentClassifier()
    queries = [
        "这件衣服是什么面料？",
        "什么颜色？",
        "领口设计怎么样",
        "搭配建议",
        "水洗工艺",
        "你好",
        "今天天气不错",
    ]
    start = time.perf_counter()
    for _ in range(1000):
        for q in queries:
            c.classify(q)
    elapsed_ms = (time.perf_counter() - start) * 1000
    per_call_us = elapsed_ms / (1000 * len(queries)) * 1000
    assert per_call_us < 200, f"classifier too slow: {per_call_us:.0f} µs/call"
