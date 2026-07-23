"""P0b.1 — ContentGenerationService tests."""

from __future__ import annotations

import json

import pytest

from inference.serving.content_generation_service import (
    ContentGenerationService,
    get_content_generation_service,
)


@pytest.fixture(scope="module")
def svc() -> ContentGenerationService:
    return ContentGenerationService()


# ── selling_points ─────────────────────────────────────────────────────────────


def test_selling_points_with_attrs(svc):
    r = svc.generate(content_type="selling_points", attributes={
        "fabric": {"value": "棉", "attribute_confidence": 0.9},
        "color": {"value": "白色"},
        "style": {"value": "通勤"},
    })
    assert isinstance(r.generated_content, list)
    assert len(r.generated_content) >= 2
    assert r.meta["llm_used"] is False
    assert len(r.sources) >= 2


def test_selling_points_primitive_attrs(svc):
    r = svc.generate(content_type="selling_points", attributes={
        "fabric": "棉", "color": "白色",
    })
    assert isinstance(r.generated_content, list)
    # Should not crash on primitive attrs.
    assert r.meta["attribute_count"] == 2


def test_selling_points_empty_attrs(svc):
    r = svc.generate(content_type="selling_points", attributes={})
    assert r.generated_content == []
    assert any(w.code == "content_input_missing" for w in r.warnings)


# ── title ──────────────────────────────────────────────────────────────────────


def test_title_with_attrs(svc):
    r = svc.generate(content_type="title", attributes={
        "fabric": {"value": "棉"},
        "color": {"value": "白色"},
        "style": {"value": "通勤"},
    }, garment_category="衬衫")
    assert isinstance(r.generated_content, str)
    assert "色" in r.generated_content or "棉" in r.generated_content


def test_title_max_length(svc):
    r = svc.generate(content_type="title", attributes={
        "fabric": {"value": "棉" * 100},
    }, max_length=20)
    assert len(r.generated_content) <= 20
    assert r.meta["max_length_applied"] is True


# ── short_description ──────────────────────────────────────────────────────────


def test_short_description(svc):
    r = svc.generate(content_type="short_description", attributes={
        "fabric": {"value": "棉"},
        "color": {"value": "白"},
    }, garment_category="T恤")
    assert isinstance(r.generated_content, str)
    assert "T恤" in r.generated_content or "商品" in r.generated_content


# ── detail_bullets ─────────────────────────────────────────────────────────────


def test_detail_bullets(svc):
    r = svc.generate(content_type="detail_bullets", attributes={
        "fabric": {"value": "麻"},
    })
    assert isinstance(r.generated_content, list)
    assert len(r.generated_content) >= 1
    assert "title" in r.generated_content[0]
    assert "text" in r.generated_content[0]


# ── Policy ─────────────────────────────────────────────────────────────────────


def test_policy_blocked_function(svc):
    r = svc.generate(content_type="selling_points", attributes={
        "fabric": "棉",
        "function": "抗菌",
    })
    assert len(r.blocked_claims) > 0
    assert r.blocked_claims[0]["field"] == "function"
    assert r.blocked_claims[0]["reason"] == "high_risk_attribute_key"
    assert any(w.code == "content_policy_blocked" for w in r.warnings)
    # Should still generate from safe attrs.
    assert len(r.generated_content) >= 1
    # Blocked claim must NOT expose the raw risky value.
    raw = str(r.blocked_claims)
    assert "抗菌" not in raw


def test_policy_blocked_claim(svc):
    r = svc.generate(content_type="selling_points", attributes={
        "fabric": {"value": "100%棉"},
    })
    assert r.blocked_claims
    assert r.blocked_claims[0]["field"] == "fabric"
    assert r.blocked_claims[0]["reason"] == "blocked_token"
    # Blocked claims must NOT echo raw "100%".
    raw = str(r.blocked_claims)
    assert "100%" not in raw
    # Generated content must NOT contain "100%".
    if isinstance(r.generated_content, list):
        for pt in r.generated_content:
            assert "100%" not in pt


# ── Sources ────────────────────────────────────────────────────────────────────


def test_sources_from_attributes(svc):
    r = svc.generate(content_type="selling_points", attributes={
        "fabric": {"value": "棉", "attribute_confidence": 0.88, "source": "request_raw"},
    })
    assert len(r.sources) >= 1
    assert r.sources[0]["type"] == "product_attribute"
    assert r.sources[0]["field"] == "fabric"


# ── JSON-safe ──────────────────────────────────────────────────────────────────


def test_to_dict_json_safe(svc):
    r = svc.generate(content_type="title", attributes={"fabric": "棉"})
    d = r.to_dict()
    json.dumps(d, ensure_ascii=False)


def test_singleton():
    a = get_content_generation_service()
    b = get_content_generation_service()
    assert a is b


def test_no_mutation(svc):
    attrs = {"fabric": "棉"}
    svc.generate(content_type="title", attributes=attrs)
    assert attrs == {"fabric": "棉"}


def test_unsupported_type(svc):
    r = svc.generate(content_type="unknown_xyz")
    assert any(w.code == "content_unsupported_type" for w in r.warnings)
