"""P0a.5 — /v1/mm/qa endpoint integration tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from inference.serving.app import app

client = TestClient(app)


def _assert_unified(r: dict):
    assert r["request_id"].startswith("req_")
    assert r["status"] == "success"
    assert r["data"] is not None
    data = r["data"]
    assert "warnings" not in data, "warnings must be top-level only"


def test_mm_qa_attribute_query():
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "attributes": {"fabric": {"value": "纯棉", "attribute_confidence": 0.92, "source": "merchant_input"}},
    }).json()
    _assert_unified(r)
    data = r["data"]
    assert data["answer_type"] == "attribute_answer"
    assert "纯棉" in data["answer"]
    assert data["intent_confidence"] is not None


def test_mm_qa_knowledge_query():
    r = client.post("/v1/mm/qa", json={"query": "纤维是什么"}).json()
    _assert_unified(r)
    data = r["data"]
    assert data["answer_type"] == "knowledge_answer"
    assert len(data["sources"]) > 0


def test_mm_qa_no_hits():
    """A truly unknown query that neither classifier nor RAG can handle."""
    r = client.post("/v1/mm/qa", json={"query": "zzqqxxvvww"}).json()
    _assert_unified(r)
    # Expect unsupported (classifier → fallback_unknown),
    # NOT knowledge_answer with no_hits (classifier must route to knowledge first).
    assert any(
        w["code"] in ("unsupported_intent", "no_hits")
        for w in r["warnings"]
    )


def test_mm_qa_empty_query():
    r = client.post("/v1/mm/qa", json={"query": " "}).json()
    _assert_unified(r)
    assert r["data"]["answer_type"] == "empty_query"


def test_mm_qa_request_id_header():
    r = client.post("/v1/mm/qa", json={"query": "test"},
                    headers={"X-Request-ID": "qa-123"}).json()
    assert r["request_id"] == "qa-123"


def test_mm_qa_warnings_top_level():
    r = client.post("/v1/mm/qa", json={"query": "这是什么？"}).json()
    assert "warnings" in r  # top-level
    data = r["data"]
    assert "warnings" not in data  # NOT in data


def test_mm_qa_response_json_safe():
    r = client.post("/v1/mm/qa", json={"query": "纤维是什么"}).json()
    json.dumps(r, ensure_ascii=False)


def test_missing_query_422():
    r = client.post("/v1/mm/qa", json={})
    assert r.status_code == 422


def test_rag_retrieve_still_works():
    """P0a.5 must not break /v1/rag/retrieve."""
    r = client.post("/v1/rag/retrieve", json={"query": "Material"}).json()
    assert len(r["data"]["hits"]) > 0


def test_health_still_works():
    r = client.get("/v1/health").json()
    assert r["data"]["ready"] is True


# ── P0a.6.1: image_bytes leak + regions + warnings top-level ──────────────────


def test_image_bytes_not_leaked():
    secret = "SECRET_BASE64_DO_NOT_LEAK_xyz"
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "image_bytes": secret,
    }).json()
    body = json.dumps(r, ensure_ascii=False)
    assert secret not in body


def test_regions_in_meta():
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "image_url": "http://example.com/img.jpg",
        "regions": ["collar", "sleeve"],
    }).json()
    meta = r["data"].get("meta", {})
    assert meta.get("requested_regions") == ["collar", "sleeve"]


def test_warnings_top_level_with_vision():
    r = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？",
        "image_url": "http://x.com/a.jpg",
    }).json()
    assert "warnings" in r  # top-level
    assert "warnings" not in r["data"]
    assert any(w["code"] == "vision_provider_mock" for w in r["warnings"])


def test_rag_no_vision_contamination():
    """P0a.6.1: /v1/rag/retrieve must NOT contain vision fields or warnings."""
    r = client.post("/v1/rag/retrieve", json={"query": "纤维是什么"}).json()
    assert "warnings" not in r["data"]
    for h in r["data"]["hits"]:
        assert "vision" not in h.get("match_type", "")

