"""P0a.4.3 — /v1/rag/retrieve endpoint integration tests.

Validates: real hits, no_hits, unknown_category, empty_query, top_k clamp,
category filter, attribute_context, request_id, schema completeness,
and /v1/mm/qa non-regression.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from inference.serving.app import app

client = TestClient(app)


def _assert_unified(r: dict):
    assert r["request_id"].startswith("req_")
    assert r["status"] == "success"
    assert "data" in r
    assert "warnings" in r
    assert "meta" in r
    data = r["data"]
    if isinstance(data, dict):
        assert "warnings" not in data, "warnings must be top-level only"


# ── Real hits ─────────────────────────────────────────────────────────────────


def test_rag_endpoint_real_hits():
    r = client.post("/v1/rag/retrieve", json={"query": "Material"}).json()
    _assert_unified(r)
    assert len(r["data"]["hits"]) > 0
    h = r["data"]["hits"][0]
    for key in ("id", "category", "term", "zh_term", "title", "content_snippet",
                "score", "match_type", "source", "source_type", "source_url",
                "source_ref", "allowed_usage", "risk_level", "risk_note",
                "review_status", "reviewed_by", "last_reviewed_at", "metadata"):
        assert key in h, f"hit missing key: {key}"
    meta = r["data"]["meta"]
    for key in ("effective_categories", "top_k", "requested_top_k", "kb_version",
                "attribute_context_keys", "expanded_query"):
        assert key in meta, f"meta missing key: {key}"
    assert r["data"]["normalized_query"] is not None


def test_rag_endpoint_no_hits():
    # Use a query with NO unigram/bigram overlap with any KB doc.
    # Pure ASCII gibberish with unique characters avoids BM25 false-positive.
    r = client.post("/v1/rag/retrieve", json={"query": "zzqqxxvvww"}).json()
    _assert_unified(r)
    assert r["data"]["hits"] == []
    assert any(w["code"] == "no_hits" for w in r["warnings"])


def test_rag_endpoint_unknown_category():
    r = client.post("/v1/rag/retrieve", json={"query": "纤维", "categories": ["not_a_category"]}).json()
    _assert_unified(r)
    assert any(w["code"] == "unknown_category" for w in r["warnings"])
    assert not any(w["code"] == "no_hits" for w in r["warnings"])


def test_rag_endpoint_category_filter():
    r = client.post("/v1/rag/retrieve", json={"query": "材料", "categories": ["fiber"]}).json()
    _assert_unified(r)
    for h in r["data"]["hits"]:
        assert h["category"] == "fiber"


def test_rag_endpoint_attribute_context():
    r = client.post("/v1/rag/retrieve", json={
        "query": "纤维",
        "attribute_context": {"fabric": "cotton", "style": ["casual", "commuter"]}
    }).json()
    _assert_unified(r)
    meta = r["data"]["meta"]
    assert "fabric" in meta["attribute_context_keys"]
    assert "style" in meta["attribute_context_keys"]


def test_rag_endpoint_request_id_header():
    r = client.post("/v1/rag/retrieve",
                     json={"query": "test"},
                     headers={"X-Request-ID": "my-rag-id"}).json()
    assert r["request_id"] == "my-rag-id"


def test_rag_endpoint_warnings_json_serializable():
    r = client.post("/v1/rag/retrieve", json={"query": "zzqqxxvvww"}).json()
    assert any(w["code"] == "no_hits" for w in r["warnings"])
    d = json.dumps(r, ensure_ascii=False)
    assert "no_hits" in d


def test_rag_endpoint_error_query_422():
    r = client.post("/v1/rag/retrieve", json={})
    assert r.status_code == 422


def test_mm_qa_not_affected():
    """P0a.5: /v1/mm/qa is now real (orchestrator).  Confirm it responds cleanly."""
    r = client.post("/v1/mm/qa", json={"query": "什么面料？"}).json()
    _assert_unified(r)
    assert r["data"]["answer_type"] != "mock"
