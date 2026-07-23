"""P0a.8 — Golden contract tests.

Loads cases from ``tests/fixtures/serving_golden_cases.json`` and verifies
that the serving endpoints uphold their contracts: status, answer_type,
meta fields, warning codes, source types, forbidden keys, and no-leak.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference.serving.app import app

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "serving_golden_cases.json"


# ── Helper: load ───────────────────────────────────────────────────────────────

def _load_golden_cases() -> list[dict]:
    with open(_FIXTURE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Helper: recursive key check ────────────────────────────────────────────────

def _has_key_recursive(obj, key: str) -> bool:
    """Return True if *key* appears anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_has_key_recursive(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key_recursive(v, key) for v in obj)
    return False


# ── Helper: assertions ─────────────────────────────────────────────────────────

def _assert_status(resp_json: dict, expected: str):
    assert resp_json.get("status") == expected, f"expected status={expected}, got {resp_json.get('status')}"


def _assert_data_keys(resp_json: dict, expected_keys: list):
    data = resp_json.get("data", {})
    for key in expected_keys:
        assert key in data, f"data missing key: {key}"


def _assert_data_field(resp_json: dict, field: str, expected):
    data = resp_json.get("data", {})
    actual = data.get(field)
    assert actual == expected, f"data.{field}: expected {expected!r}, got {actual!r}"


def _assert_required_meta(resp_json: dict, required: dict):
    data = resp_json.get("data", {})
    meta = data.get("meta", {})
    for key, val in required.items():
        assert meta.get(key) == val, f"meta.{key}: expected {val!r}, got {meta.get(key)!r}"


def _assert_warning_codes(resp_json: dict, required: list, forbidden: list):
    warnings = resp_json.get("warnings", [])
    codes = [w["code"] for w in warnings]
    for code in required:
        assert code in codes, f"missing required warning: {code}. Got: {codes}"
    for code in forbidden:
        assert code not in codes, f"forbidden warning present: {code}. Got: {codes}"


def _assert_forbidden_substrings(resp_json: dict, forbidden: list):
    body = json.dumps(resp_json, ensure_ascii=False)
    for sub in forbidden:
        assert sub not in body, f"forbidden substring found in response: {sub!r}"


def _assert_data_no_warnings(resp_json: dict):
    data = resp_json.get("data", {})
    if isinstance(data, dict):
        assert "warnings" not in data, "warnings must NOT be inside data"


def _assert_required_source_types(resp_json: dict, required_types: list):
    sources = resp_json.get("data", {}).get("sources", [])
    types = {s.get("type") for s in sources}
    for t in required_types:
        assert t in types, f"missing source type '{t}'. Got: {types}"


def _assert_hit_required_keys(resp_json: dict, required_keys: list):
    hits = resp_json.get("data", {}).get("hits", [])
    for hit in hits:
        for key in required_keys:
            assert key in hit, f"hit missing key: {key}"


def _assert_forbidden_keys_recursive(resp_json: dict, forbidden: list):
    data = resp_json.get("data", {})
    for key in forbidden:
        assert not _has_key_recursive(data, key), f"forbidden key '{key}' found in data"


# ── Client ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return _load_golden_cases()


# ── Parametrized runner ────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", _load_golden_cases(), ids=lambda c: c["id"])
def test_serving_golden_contract(client, case):
    method = case["method"]
    endpoint = case["endpoint"]
    req = case["request"]
    expect = case.get("expect", {})

    # Send request.
    if method == "POST":
        r = client.post(endpoint, json=req)
    else:
        r = client.get(endpoint)
    resp_json = r.json()

    # Status.
    if "status" in expect:
        _assert_status(resp_json, expect["status"])

    # Data keys.
    if "data_keys" in expect:
        _assert_data_keys(resp_json, expect["data_keys"])

    # data.answer_type.
    if "data_answer_type" in expect:
        _assert_data_field(resp_json, "answer_type", expect["data_answer_type"])

    # data.primary_intent.
    if "data_primary_intent" in expect:
        _assert_data_field(resp_json, "primary_intent", expect["data_primary_intent"])

    if "data_sub_intent" in expect:
        _assert_data_field(resp_json, "sub_intent", expect["data_sub_intent"])

    if "data_classifier_level" in expect:
        _assert_data_field(resp_json, "classifier_level", expect["data_classifier_level"])

    # Meta.
    if "required_meta" in expect:
        _assert_required_meta(resp_json, expect["required_meta"])

    # Warning codes.
    _assert_warning_codes(
        resp_json,
        expect.get("required_warning_codes", []),
        expect.get("forbidden_warning_codes", []),
    )

    # Forbidden substrings.
    if "forbidden_response_substrings" in expect:
        _assert_forbidden_substrings(resp_json, expect["forbidden_response_substrings"])

    # data.warnings not allowed.
    if expect.get("forbidden_data_response_key"):
        _assert_data_no_warnings(resp_json)

    # Source types.
    if "required_source_types" in expect:
        _assert_required_source_types(resp_json, expect["required_source_types"])

    # Hit keys.
    if "hit_required_keys" in expect:
        _assert_hit_required_keys(resp_json, expect["hit_required_keys"])

    # Recursive forbidden keys.
    if "forbidden_keys_recursive" in expect:
        _assert_forbidden_keys_recursive(resp_json, expect["forbidden_keys_recursive"])
