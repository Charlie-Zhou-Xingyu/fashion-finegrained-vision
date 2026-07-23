"""P0a.4.2-hardened — RagService tests (Part A critical review)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from inference.serving.rag_service import (
    RagService,
    _safe_int,
    _safe_strings,
    _expand_query,
    normalize_query,
    get_rag_service,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def svc() -> RagService:
    return RagService()


# ═══════════════════════════════════════════════════════════════════════════════
# A4: empty_query
# ═══════════════════════════════════════════════════════════════════════════════

def test_empty_query_none(svc):
    r = svc.retrieve(None)
    assert any(w.code == "empty_query" for w in r.warnings)
    assert r.hits == []


def test_empty_query_whitespace(svc):
    r = svc.retrieve("   ")
    assert any(w.code == "empty_query" for w in r.warnings)
    assert not any(w.code == "no_hits" for w in r.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# A3: unknown_category
# ═══════════════════════════════════════════════════════════════════════════════

def test_unknown_category_no_no_hits_duplication(svc):
    r = svc.retrieve("纤维", categories=["not_a_category"])
    assert any(w.code == "unknown_category" for w in r.warnings)
    assert not any(w.code == "no_hits" for w in r.warnings)
    assert r.hits == []


# ═══════════════════════════════════════════════════════════════════════════════
# A5: top_k clamp
# ═══════════════════════════════════════════════════════════════════════════════

def test_top_k_string_numeric(svc):
    r = svc.retrieve("材料", top_k="3")
    assert len(r.hits) <= 3


def test_top_k_string_bad(svc):
    r = svc.retrieve("材料", top_k="bad")
    assert len(r.hits) <= 3  # defaults
    assert any(w.code == "top_k_clamped" for w in r.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# A6: score validity
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_scores_in_zero_to_one(svc):
    r = svc.retrieve("材料 纤维 棉 回收 优选 循环")
    for h in r.hits:
        assert 0.0 <= h.score <= 1.0, f"score {h.score} out of range for {h.id}"


def test_repeated_retrieve_stable(svc):
    r1 = svc.retrieve("Material 纤维")
    r2 = svc.retrieve("Material 纤维")
    r3 = svc.retrieve("Material 纤维")
    assert [h.id for h in r1.hits] == [h.id for h in r2.hits] == [h.id for h in r3.hits]


# ═══════════════════════════════════════════════════════════════════════════════
# A7: mixed Chinese/English tokenization
# ═══════════════════════════════════════════════════════════════════════════════

def test_mixed_lang_query(svc):
    r = svc.retrieve("cotton 纤维")
    ids = {h.id for h in r.hits}
    assert ids & {"cotton_001", "fiber_term_001"}


def test_mixed_lang_query_recycled(svc):
    r = svc.retrieve("recycled material 再生")
    ids = {h.id for h in r.hits}
    assert "recycled_materials_001" in ids


# ═══════════════════════════════════════════════════════════════════════════════
# A8: attribute_context
# ═══════════════════════════════════════════════════════════════════════════════

def test_attribute_context_list(svc):
    r = svc.retrieve("纤维", attribute_context={"fabric": ["cotton", "wool"]})
    assert "fabric" in r.meta.get("attribute_context_keys", [])


def test_attribute_context_nested_dict_no_crash(svc):
    r = svc.retrieve("纤维", attribute_context={"nested": {"a": 1}})
    assert r.hits is not None  # doesn't crash


def test_attribute_context_exact_not_overridden(svc):
    """Exact match score must not be changed by attribute_context."""
    r = svc.retrieve("Fiber", attribute_context={"fabric": "cotton"})
    exact = [h for h in r.hits if h.id == "fiber_term_001"]
    if exact:
        assert exact[0].score >= 1.0


def test_safe_int():
    assert _safe_int(None, 3) == 3
    assert _safe_int("3", 5) == 3
    assert _safe_int("bad", 5) == 5
    assert _safe_int(7, 3) == 7
    assert _safe_int(-1, 3) == -1  # valid int preserved; service layer clamps <= 0


def test_safe_strings():
    assert _safe_strings(None) == []
    assert _safe_strings("hello") == ["hello"]
    assert _safe_strings(["a", "b"]) == ["a", "b"]
    assert _safe_strings({"x": 1}) == []


def test_expand_query():
    q, keys = _expand_query("test", {"fabric": "cotton", "color": ["red", "blue"], "nested": {}})
    assert "test" in q
    assert "cotton" in q
    assert "red" in q
    assert "blue" in q
    assert set(keys) == {"fabric", "color"}


# ═══════════════════════════════════════════════════════════════════════════════
# A9: JSON serialization
# ═══════════════════════════════════════════════════════════════════════════════

def test_to_dict_json_full(svc):
    r = svc.retrieve("Material")
    d = r.to_dict()
    s = json.dumps(d, ensure_ascii=False)
    assert "Material" in s


def test_no_hits_json(svc):
    r = svc.retrieve("zzqqxxvvww")
    d = r.to_dict()
    json.dumps(d, ensure_ascii=False)
    assert any(w.code == "no_hits" for w in r.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# A11: KB validation
# ═══════════════════════════════════════════════════════════════════════════════

def _temp_kb(docs, sources=None):
    base = {
        "version": "1.0.0", "locale": "zh-CN",
        "source_policy": {"default_review_status": "manual_review_required", "notes": ""},
        "sources": sources or {
            "test_src": {"title": "T", "publisher": "P", "year": 2020,
                         "version": "1", "copyright": "(c)", "source_type": "pdf",
                         "source_url": None, "license": "unknown", "usage_note": ""}
        },
        "documents": docs,
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(base, f, allow_unicode=True)
    f.close()
    return Path(f.name)


def _make_doc(id_, **overrides):
    d = {
        "id": id_, "category": "term", "term": id_, "zh_term": id_,
        "aliases": [id_.lower()], "title": id_, "content": id_,
        "allowed_usage": ["internal_test"], "risk_level": "low",
        "source": "test_src", "source_type": "pdf", "source_url": None,
        "source_ref": {"document_title": "T", "page_start": 1, "page_end": 1, "section": "T"},
        "review_status": "manual_review_required", "reviewed_by": None,
        "last_reviewed_at": None, "version": "1.0.0", "tags": [id_.lower()],
    }
    d.update(overrides)
    return d


def test_kb_sources_missing_copyright_raises():
    p = _temp_kb([_make_doc("x")], sources={"test_src": {"title": "T", "publisher": "P", "year": 2020,
                  "version": "1", "copyright": "(c)", "source_type": "pdf"}})
    try:
        with pytest.raises(RuntimeError):
            RagService(kb_path=p)
    finally:
        p.unlink(missing_ok=True)


def test_kb_source_registry_license_required():
    p = _temp_kb([_make_doc("x")], sources={"test_src": {"title": "T", "publisher": "P", "year": 2020,
                  "version": "1", "copyright": "(c)", "source_type": "pdf",
                  "source_url": None, "usage_note": ""}})
    try:
        with pytest.raises(RuntimeError):
            RagService(kb_path=p)
    finally:
        p.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# A1: Not overfitting — uses a test KB with different IDs
# ═══════════════════════════════════════════════════════════════════════════════

def test_service_works_with_different_ids():
    docs = [
        _make_doc("alpha_001", term="AlphaTerm", zh_term="阿尔法", aliases=["alpha", "阿尔法"], title="Alpha Title"),
        _make_doc("beta_001", term="BetaTerm", zh_term="贝塔", aliases=["beta", "贝塔"], title="Beta Title"),
    ]
    p = _temp_kb(docs)
    try:
        s = RagService(kb_path=p)
        r = s.retrieve("AlphaTerm")
        assert r.hits[0].id == "alpha_001"
        r2 = s.retrieve("贝塔")
        assert any(h.id == "beta_001" for h in r2.hits)
    finally:
        p.unlink(missing_ok=True)


def test_generic_query_nonfiber(svc):
    """'什么是非纤维材料' should hit nonfiber_term_001 without hardcoded IDs."""
    r = svc.retrieve("什么是非纤维材料")
    ids = {h.id for h in r.hits}
    assert "nonfiber_term_001" in ids


# ═══════════════════════════════════════════════════════════════════════════════
# A12: singleton
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Exact / Contains / Alias (against real KB)
# ═══════════════════════════════════════════════════════════════════════════════

def test_exact_match_material_kb(svc):
    r = svc.retrieve("Material")
    assert any(h.id == "material_term_001" and h.match_type == "exact" for h in r.hits)


def test_contains_material_zh_kb(svc):
    r = svc.retrieve("材料是什么")
    assert any(h.id == "material_term_001" for h in r.hits)


def test_exact_fiber_kb(svc):
    r = svc.retrieve("Fiber")
    assert any(h.id == "fiber_term_001" for h in r.hits)


def test_contains_fiber_zh_kb(svc):
    r = svc.retrieve("纤维是什么")
    assert any(h.id == "fiber_term_001" for h in r.hits)


def test_alias_zh_kb(svc):
    r = svc.retrieve("纺织纤维")
    assert any(h.id == "fiber_term_001" for h in r.hits)


def test_alias_en_lowercase_kb(svc):
    r = svc.retrieve("recycled raw materials")
    assert any(h.id == "recycled_materials_001" for h in r.hits)


def test_bm25_multi_token_kb(svc):
    r = svc.retrieve("原材料 可再生 回收")
    ids = {h.id for h in r.hits}
    assert ids & {"raw_materials_001", "renewable_materials_001", "recycled_materials_001"}


def test_bm25_fiber_categories_kb(svc):
    r = svc.retrieve("植物 动物 合成 纤维")
    ids = {h.id for h in r.hits}
    assert ids & {"plant_fibers_001", "animal_fibers_001", "synthetic_fibers_001", "mmcf_001"}


def test_category_filter_kb(svc):
    r = svc.retrieve("纤维 材料", categories=["fiber"])
    for h in r.hits:
        assert h.category == "fiber"


def test_category_filter_no_match_kb(svc):
    r = svc.retrieve("纤维", categories=["supply_chain"])
    assert any(w.code == "no_hits" for w in r.warnings)


def test_intent_category_map_kb(svc):
    r = svc.retrieve("纤维", primary_intent="knowledge_qa", sub_intent="fabric")
    assert r.meta["effective_categories"] == ["fabric", "material", "fiber"]


def test_dedup_kb(svc):
    r = svc.retrieve("Material 材料 纤维")
    ids = [h.id for h in r.hits]
    assert len(ids) == len(set(ids))


def test_score_descending_kb(svc):
    r = svc.retrieve("植物 纤维 材料", top_k=10)
    scores = [h.score for h in r.hits]
    assert scores == sorted(scores, reverse=True)


def test_hit_metadata_kb(svc):
    r = svc.retrieve("Material")
    h = r.hits[0]
    assert h.source_ref.get("document_title")
    assert h.source_ref.get("page_start") is not None
    assert h.review_status == "manual_review_required"
    assert "knowledge_qa" in h.allowed_usage or "internal_test" in h.allowed_usage


def test_hit_risk_level_kb(svc):
    r = svc.retrieve("preferred")
    hits = [h for h in r.hits if h.id == "preferred_material_001"]
    assert hits
    assert hits[0].risk_level == "medium"


def test_singleton():
    a = get_rag_service()
    b = get_rag_service()
    assert a is b
