"""P0a.3 — Attribute Service tests (hardened).

Covers:
    - normalisation (AttributeValue / dict / primitive) with source defaults
    - field whitelist (unknown fields → ignored, no crash)
    - attribute_confidence boundary validation (out-of-range, string, non-numeric)
    - template config validation (missing core keys → RuntimeError)
    - WarningItem / SourceItem compatibility and AttributeAnswer.to_dict()
    - answer_confidence policy (never auto-set to 1.0)
    - sanitize_fabric_value for 7+ percentage variants
    - unknown attribute with/without same-name field
    - length fallback meta tracking
    - attribute missing / unavailable
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from inference.serving.attribute_service import (
    AttributeAnswer,
    AttributeService,
    get_attribute_service,
    normalize_attributes,
    sanitize_fabric_value,
)
from inference.serving.schemas import AttributeValue


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def svc() -> AttributeService:
    return AttributeService()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Normalisation — source defaults
# ═══════════════════════════════════════════════════════════════════════════════


def test_normalize_attributevalue_passthrough():
    av = AttributeValue(value="棉", attribute_confidence=0.9, source="merchant_input")
    result = normalize_attributes({"fabric": av})
    assert result["fabric"].value == "棉"
    assert result["fabric"].attribute_confidence == 0.9
    assert result["fabric"].source == "merchant_input"


def test_normalize_dict_defaults_source_to_request_raw():
    """dict without source → source='request_raw' (not None)."""
    result = normalize_attributes({
        "fabric": {"value": "纯棉", "attribute_confidence": 0.9}
    })
    assert result["fabric"].value == "纯棉"
    assert result["fabric"].source == "request_raw"


def test_normalize_dict_preserves_explicit_source():
    result = normalize_attributes({
        "fabric": {"value": "纯棉", "source": "merchant_input"}
    })
    assert result["fabric"].source == "merchant_input"


def test_normalize_primitive_str():
    result = normalize_attributes({"color": "白色"})
    assert result["color"].value == "白色"
    assert result["color"].attribute_confidence is None
    assert result["color"].source == "request_raw"


def test_normalize_primitive_int():
    result = normalize_attributes({"count": 42})
    assert result["count"].value == 42
    assert result["count"].source == "request_raw"


def test_normalize_empty():
    assert normalize_attributes({}) == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Field whitelist — unknown dict keys must not crash
# ═══════════════════════════════════════════════════════════════════════════════


def test_normalize_dict_ignores_unknown_fields():
    result = normalize_attributes({
        "fabric": {
            "value": "纯棉",
            "attribute_confidence": 0.92,
            "source": "merchant_input",
            "unexpected_field": "xxx",
        }
    })
    assert result["fabric"].value == "纯棉"
    # No exception raised — unknown field silently ignored.


def test_normalize_unknown_type_skipped():
    result = normalize_attributes({"bad": object()})
    assert "bad" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. attribute_confidence boundary validation
# ═══════════════════════════════════════════════════════════════════════════════


def test_confidence_out_of_range_high_clamped_to_none():
    result = normalize_attributes({
        "fabric": {"value": "x", "attribute_confidence": 1.2}
    })
    assert result["fabric"].attribute_confidence is None


def test_confidence_out_of_range_negative_clamped_to_none():
    result = normalize_attributes({
        "fabric": {"value": "x", "attribute_confidence": -0.1}
    })
    assert result["fabric"].attribute_confidence is None


def test_confidence_string_numeric():
    result = normalize_attributes({
        "fabric": {"value": "x", "attribute_confidence": "0.88"}
    })
    assert result["fabric"].attribute_confidence == 0.88


def test_confidence_string_non_numeric():
    result = normalize_attributes({
        "fabric": {"value": "x", "attribute_confidence": "high"}
    })
    assert result["fabric"].attribute_confidence is None


def test_confidence_none_ok():
    result = normalize_attributes({
        "fabric": {"value": "x", "attribute_confidence": None}
    })
    assert result["fabric"].attribute_confidence is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Template config validation
# ═══════════════════════════════════════════════════════════════════════════════


def test_missing_template_file_raises():
    with pytest.raises(RuntimeError):
        AttributeService(template_path=Path("nonexistent_templates.yaml"))


def test_template_missing_core_key_raises():
    """If fabric.default is missing the service MUST fail-fast."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        broken = {
            "version": "1.0.0",
            "thresholds": {"low_confidence": 0.6, "high_confidence": 0.8},
            "aliases": {},
            "attributes": {
                "fabric": {
                    "display_name": "面料",
                    "unavailable": "...",
                    # "default" intentionally missing
                }
            },
        }
        yaml.dump(broken, f, allow_unicode=True)
        tmp_path = f.name
    try:
        with pytest.raises(RuntimeError):
            AttributeService(template_path=Path(tmp_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AttributeAnswer.to_dict() compatibility
# ═══════════════════════════════════════════════════════════════════════════════


def test_to_dict_json_serializable():
    import json
    a = AttributeAnswer("测试", answer_confidence=0.9,
                        sources=[], warnings=[])
    d = a.to_dict()
    assert d["answer"] == "测试"
    assert d["answer_confidence"] == 0.9
    assert d["sources"] == []
    json.dumps(d)  # must not raise


def test_to_dict_with_sources_and_warnings(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="纯棉", attribute_confidence=0.92, source="merchant_input")
    })
    d = r.to_dict()
    assert len(d["sources"]) == 1
    assert d["sources"][0]["attribute_confidence"] == 0.92


# ═══════════════════════════════════════════════════════════════════════════════
# 6. answer_confidence policy
# ═══════════════════════════════════════════════════════════════════════════════


def test_answer_confidence_equals_attribute_confidence(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="x", attribute_confidence=0.75)
    })
    assert r.answer_confidence == 0.75


def test_answer_confidence_none_when_attr_confidence_none(svc):
    r = svc.answer_attribute("fabric", {"fabric": "棉"})
    assert r.answer_confidence is None


def test_answer_confidence_not_auto_set_to_1_for_manual_verified(svc):
    """manual_verified MUST NOT auto-set answer_confidence to 1.0."""
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="纯棉", attribute_confidence=None,
                                 source="manual_verified")
    })
    assert r.answer_confidence is None
    assert r.answer_confidence != 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. sanitize_fabric_value — all percentage variants
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("raw,expected", [
    ("100%纯棉", "纯棉"),
    ("100％纯棉", "纯棉"),
    ("100 % 纯棉", "纯棉"),
    ("百分百纯棉", "纯棉"),
    ("百分之百纯棉", "纯棉"),
    ("纯棉100%", "纯棉"),
    ("纯棉 100％", "纯棉"),
])
def test_sanitize_strips_percentage_variants(raw, expected):
    assert sanitize_fabric_value(raw, composition_verified=False) == expected


def test_sanitize_verified_true_preserves_pct():
    assert sanitize_fabric_value("100%纯棉", composition_verified=True) == "100%纯棉"


def test_sanitize_no_pct_unchanged():
    assert sanitize_fabric_value("棉麻混纺", composition_verified=False) == "棉麻混纺"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Unknown attribute
# ═══════════════════════════════════════════════════════════════════════════════


def test_unknown_attribute_with_value_uses_general(svc):
    """Unknown attr name + value present in norm → general template + warning."""
    r = svc.answer_attribute("fabric_type", {
        "fabric_type": AttributeValue(value="something")
    })
    assert r.answer is not None
    assert any(w.code == "unknown_attribute_type" for w in r.warnings)


def test_unknown_attribute_without_value_returns_unavailable(svc):
    """Unknown attr name + value NOT present → unavailable."""
    r = svc.answer_attribute("xyz_attr", {"color": "红色"})
    assert "暂未获取到" in r.answer
    assert any(w.code == "unknown_attribute_type" for w in r.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Length fallback meta
# ═══════════════════════════════════════════════════════════════════════════════


def test_length_fallback_meta(svc):
    r = svc.answer_attribute("length", {
        "skirt_length": AttributeValue(value="短裙", attribute_confidence=0.75)
    })
    assert r.meta["requested_attribute"] == "length"
    assert r.meta["resolved_attribute"] == "skirt_length"
    assert r.meta["resolved_by"] == "length_fallback"


def test_length_direct_no_fallback(svc):
    r = svc.answer_attribute("length", {
        "length": AttributeValue(value="中长款")
    })
    assert r.meta["resolved_attribute"] == "length"
    assert r.meta["resolved_by"] == "direct"


# ═══════════════════════════════════════════════════════════════════════════════
# Core functional tests (kept from original)
# ═══════════════════════════════════════════════════════════════════════════════


def test_fabric_merchant_input_high_confidence(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="纯棉", attribute_confidence=0.92, source="merchant_input")
    })
    assert "纯棉" in r.answer
    assert "标注为" in r.answer
    assert len(r.sources) == 1
    assert r.sources[0].attribute_confidence == 0.92
    assert r.sources[0].source == "merchant_input"
    assert len(r.warnings) == 0


def test_fabric_model_prediction_is_conservative(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="纯棉", attribute_confidence=0.90, source="model_prediction")
    })
    assert "可能为" in r.answer
    assert "仅凭图片无法准确确认" in r.answer


def test_fabric_100pct_composition_unverified(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="100%纯棉", attribute_confidence=0.88,
                                 source="merchant_input", composition_verified=False)
    })
    assert "100%" not in r.answer


def test_fabric_100pct_composition_verified(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="100%纯棉", attribute_confidence=0.95,
                                 source="manual_verified", composition_verified=True)
    })
    assert "100%纯棉" in r.answer


def test_color_query(svc):
    r = svc.answer_attribute("color", {
        "color": AttributeValue(value="白色", attribute_confidence=0.98, source="merchant_input")
    })
    assert "白色" in r.answer
    assert r.sources[0].field == "color"


def test_collar_alias_maps_to_collar_design(svc):
    r = svc.answer_attribute("collar", {
        "collar_design": AttributeValue(value="圆领", attribute_confidence=0.85)
    })
    assert "圆领" in r.answer
    assert r.sources[0].field == "collar_design"
    assert r.meta["requested_attribute"] == "collar"
    assert r.meta["resolved_attribute"] == "collar_design"
    assert r.meta["resolved_by"] == "direct"


def test_sleeve_alias_maps_to_sleeve_length(svc):
    r = svc.answer_attribute("sleeve", {
        "sleeve_length": AttributeValue(value="长袖", attribute_confidence=0.82)
    })
    assert "长袖" in r.answer
    assert r.sources[0].field == "sleeve_length"


def test_attribute_missing(svc):
    r = svc.answer_attribute("fabric", {"color": "红色"})
    assert "暂未获取到" in r.answer
    assert len(r.sources) == 0
    assert any(w.code == "attribute_unavailable" for w in r.warnings)


def test_low_confidence_returns_warning(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": AttributeValue(value="棉", attribute_confidence=0.45)
    })
    assert len(r.warnings) == 1
    assert r.warnings[0].code == "low_attribute_confidence"
    assert r.warnings[0].severity == "warn"


def test_medium_confidence_no_warning():
    svc2 = AttributeService()
    r = svc2.answer_attribute("fabric", {
        "fabric": AttributeValue(value="棉", attribute_confidence=0.65)
    })
    assert len(r.warnings) == 0


def test_garment_label_from_arg(svc):
    r = svc.answer_attribute("fabric", {"fabric": "棉"}, garment_category="上衣")
    assert "上衣" in r.answer


def test_garment_label_from_attributes(svc):
    r = svc.answer_attribute("fabric", {
        "fabric": "棉",
        "garment_category": "pants",
    })
    assert "pants" in r.answer


def test_length_fallback_skirt_length(svc):
    r = svc.answer_attribute("length", {
        "skirt_length": AttributeValue(value="短裙", attribute_confidence=0.75)
    })
    assert "短裙" in r.answer
    assert r.sources[0].field == "skirt_length"


def test_length_no_fallback(svc):
    r = svc.answer_attribute("length", {"color": "红色"})
    assert "暂未获取到" in r.answer
    assert len(r.sources) == 0


def test_attribute_answer_structure():
    a = AttributeAnswer("测试回答", answer_confidence=0.95)
    assert a.answer == "测试回答"
    assert a.answer_type == "attribute_query"
    assert a.answer_confidence == 0.95
    assert a.sources == []
    assert a.warnings == []
    assert a.used_tools == ["attribute_service"]


def test_singleton():
    a = get_attribute_service()
    b = get_attribute_service()
    assert a is b
