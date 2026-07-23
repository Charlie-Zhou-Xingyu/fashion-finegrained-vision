"""P0a.1 FastAPI app smoke tests — endpoint contracts only."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference.serving.app import app

client = TestClient(app)


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _assert_unified(response_json: dict):
    """Every response must be a valid UnifiedResponse envelope."""
    assert "request_id" in response_json
    assert response_json["request_id"].startswith("req_")
    assert "status" in response_json
    assert "data" in response_json
    assert "elapsed_ms" in response_json
    assert "used_tools" in response_json
    assert "warnings" in response_json
    assert "meta" in response_json
    # warnings must be at top-level only — data should not carry its own warnings
    data = response_json["data"]
    if isinstance(data, dict):
        assert "warnings" not in data, (
            "warnings must only appear at top-level UnifiedResponse, not inside data"
        )


# ── /v1/health ───────────────────────────────────────────────────────────────────


def test_health_ok():
    r = client.get("/v1/health")
    assert r.status_code == 200
    _assert_unified(r.json())
    data = r.json()["data"]
    assert data["service"] == "fashion-vision-serving"
    assert data["ready"] is True
    assert "implemented_modules" in data
    assert "pending_modules" in data


# ── /v1/metrics ──────────────────────────────────────────────────────────────────


def test_metrics_ok():
    """P0a.9: /v1/metrics now returns UnifiedResponse JSON with in-process counters."""
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    data = r.json()["data"]
    assert "service" in data
    assert "requests_total" in data


# ── /v1/mm/qa ────────────────────────────────────────────────────────────────────


def test_qa_with_query_only():
    """P0a.5: /v1/mm/qa now returns real orchestrator answer."""
    r = client.post("/v1/mm/qa", json={"query": "这件衣服是什么面料？"})
    assert r.status_code == 200
    _assert_unified(r.json())
    data = r.json()["data"]
    assert data["answer_type"] != "mock"
    assert data["answer_type"] in ("attribute_answer", "knowledge_answer",
                                     "hybrid_answer", "unsupported", "empty_query")


def test_qa_missing_query_422():
    r = client.post("/v1/mm/qa", json={})
    assert r.status_code == 422


def test_qa_empty_query_422():
    r = client.post("/v1/mm/qa", json={"query": ""})
    assert r.status_code == 422


def test_qa_pure_knowledge_query():
    """Pure knowledge QA without product_id or image — now returns real answer."""
    r = client.post("/v1/mm/qa", json={"query": "纯棉和麻料哪个更适合夏天？"})
    assert r.status_code == 200
    _assert_unified(r.json())
    assert r.json()["data"]["answer_type"] != "mock"


def test_qa_visual_query_without_image():
    """P0a.5: visual-hint queries now route through orchestrator normally."""
    r = client.post("/v1/mm/qa", json={"query": "这件衣服的领口是什么设计？"})
    assert r.status_code == 200
    _assert_unified(r.json())
    # P0a.5 no longer returns 'clarification' — goes through intent→orchestrator


def test_qa_with_product_id_and_attributes():
    r = client.post("/v1/mm/qa", json={
        "query": "面料？",
        "product_id": "prod_123",
        "attributes": {"fabric": {"value": "棉质", "attribute_confidence": 0.9}},
    })
    assert r.status_code == 200
    _assert_unified(r.json())


# ── /v1/intent/classify ──────────────────────────────────────────────────────────


def test_intent_classify_uses_real_classifier():
    """P0a.2: the endpoint now calls the real RuleIntentClassifier."""
    r = client.post("/v1/intent/classify", json={"query": "这件适合搭配什么裤子？"})
    assert r.status_code == 200
    _assert_unified(r.json())
    data = r.json()["data"]
    assert data["primary_intent"] == "styling_advice"
    assert data["sub_intent"] == "match"
    assert data["classifier_level"] == "rule"
    assert data["intent_confidence"] > 0.5


def test_intent_classify_missing_query_422():
    r = client.post("/v1/intent/classify", json={})
    assert r.status_code == 422


# ── /v1/rag/retrieve ─────────────────────────────────────────────────────────────


def test_rag_retrieve_returns_real_hits():
    """P0a.4.3: endpoint now calls real RagService."""
    r = client.post("/v1/rag/retrieve", json={"query": "Material"})
    assert r.status_code == 200
    _assert_unified(r.json())
    data = r.json()["data"]
    assert len(data["hits"]) > 0
    assert data["hits"][0]["id"] == "material_term_001"


def test_rag_retrieve_missing_query_422():
    r = client.post("/v1/rag/retrieve", json={})
    assert r.status_code == 422


# ── /v1/merchant/content/generate ────────────────────────────────────────────────


def test_merchant_content_returns_real():
    """P0b.1: /v1/merchant/content/generate now calls real ContentGenerationService."""
    r = client.post("/v1/merchant/content/generate", json={})
    assert r.status_code == 200
    _assert_unified(r.json())
    data = r.json()["data"]
    assert "content_type" in data
    assert "generated_content" in data
    assert data["meta"]["llm_used"] is False


def test_merchant_content_with_attributes():
    r = client.post("/v1/merchant/content/generate", json={
        "product_id": "prod_123",
        "attributes": {"fabric": {"value": "棉质", "attribute_confidence": 0.92}},
    })
    assert r.status_code == 200
    _assert_unified(r.json())


# ── Warnings at top-level only ───────────────────────────────────────────────────


def test_warnings_not_nested_in_data():
    """Every endpoint must put warnings at UnifiedResponse.warnings, not inside data."""
    endpoints = [
        ("GET", "/v1/health"),
        ("POST", "/v1/mm/qa", {"query": "test"}),
        ("POST", "/v1/intent/classify", {"query": "test"}),
        ("POST", "/v1/rag/retrieve", {"query": "test"}),
        ("POST", "/v1/merchant/content/generate", {}),
    ]
    for method, path, *args in endpoints:
        body = args[0] if args else None
        if method == "GET":
            r = client.get(path)
        else:
            r = client.post(path, json=body)
        j = r.json()
        data = j.get("data", {})
        if isinstance(data, dict):
            assert "warnings" not in data, (
                f"{method} {path}: warnings found inside data — "
                f"they must only appear at top-level UnifiedResponse"
            )


# ── X-Request-ID header ──────────────────────────────────────────────────────────


def test_custom_request_id_via_header():
    r = client.post(
        "/v1/mm/qa",
        json={"query": "test"},
        headers={"X-Request-ID": "my-custom-id"},
    )
    assert r.json()["request_id"] == "my-custom-id"


# ── HTTP semantics: 404 / 422 must NOT be swallowed by global handler ───────────


def test_unknown_path_returns_404():
    """FastAPI's built-in routing returns 404 for unregistered paths."""
    r = client.get("/v1/nonexistent_path_xyz")
    assert r.status_code == 404


def test_validation_error_still_returns_422():
    """RequestValidationError must NOT be caught by our global Exception handler.
    FastAPI's built-in handler takes precedence and returns 422."""
    r = client.post("/v1/mm/qa", json={})
    assert r.status_code == 422
