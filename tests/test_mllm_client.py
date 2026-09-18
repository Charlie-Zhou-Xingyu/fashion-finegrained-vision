"""P2 — MLLMClient tests. Mac/CPU-safe: no model, no network; the real-mode
HTTP path is exercised against a fake ``requests`` module."""

from __future__ import annotations

import base64
import sys
import types

import pytest

from inference.llm import mllm_client as m


@pytest.fixture(autouse=True)
def _clean_env_and_singleton(monkeypatch):
    for k in ("MLLM_PROVIDER", "MLLM_REAL_ENABLED", "MLLM_ENDPOINT", "MLLM_MODEL",
              "MLLM_TIMEOUT_S", "MLLM_FAIL_OPEN"):
        monkeypatch.delenv(k, raising=False)
    m.reset_mllm_client()
    yield
    m.reset_mllm_client()


class _FakeResp:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _install_fake_requests(monkeypatch, *, health_status=200, chat_payload=None, capture=None):
    fake = types.ModuleType("requests")

    def get(url, timeout=None):
        return _FakeResp(health_status, {})

    def post(url, json=None, timeout=None):
        if capture is not None:
            capture.append({"url": url, "json": json})
        return _FakeResp(200, chat_payload or {})

    fake.get, fake.post = get, post
    monkeypatch.setitem(sys.modules, "requests", fake)


# ── default + config gating ──────────────────────────────────────────────


def test_default_is_mock_and_never_fabricates():
    client = m.get_mllm_client()
    assert isinstance(client, m.MockMLLMClient)
    r = client.chat(query="这件衣服是什么面料？", image_url="http://x/y.jpg")
    assert r.text is None
    assert [w.code for w in r.warnings] == ["mllm_provider_mock"]
    assert r.meta["provider"] == "mock" and r.meta["has_image"] is True


def test_real_requires_both_provider_and_enabled_flags(monkeypatch):
    monkeypatch.setenv("MLLM_PROVIDER", "real")  # real_enabled still false
    assert isinstance(m.get_mllm_client(), m.MockMLLMClient)

    m.reset_mllm_client()
    monkeypatch.setenv("MLLM_REAL_ENABLED", "true")
    monkeypatch.setenv("MLLM_ENDPOINT", "http://gpu:8082/")
    client = m.get_mllm_client()
    assert isinstance(client, m.HttpMLLMClient)
    assert client._endpoint == "http://gpu:8082"  # trailing slash stripped


# ── real-mode behaviour ──────────────────────────────────────────────────


def test_http_client_fails_open_to_mock_when_unavailable(monkeypatch):
    _install_fake_requests(monkeypatch, health_status=503)
    client = m.HttpMLLMClient(endpoint="http://gpu:8082", fail_open_to_mock=True)
    r = client.chat(query="q")
    assert r.text is None
    assert [w.code for w in r.warnings] == ["mllm_provider_unavailable"]
    assert r.used_tools == ["mock_mllm_client"]


def test_http_client_errors_when_unavailable_and_fail_closed(monkeypatch):
    _install_fake_requests(monkeypatch, health_status=503)
    client = m.HttpMLLMClient(endpoint="http://gpu:8082", fail_open_to_mock=False)
    r = client.chat(query="q")
    assert r.finish_reason == "error" and r.meta["error"] == "endpoint_unavailable"


def test_http_client_success_parses_openai_shape(monkeypatch):
    captured: list = []
    _install_fake_requests(
        monkeypatch,
        chat_payload={
            "choices": [{"message": {"content": "这是一件棉质外套。"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 42},
            "meta": {"backend": "trtllm"},
        },
        capture=captured,
    )
    client = m.HttpMLLMClient(endpoint="http://gpu:8082", model="qwen-vl-7b-chat")
    r = client.chat(query="这件外套是什么面料？", system_prompt="你是服饰专家。", max_new_tokens=64)

    assert r.text == "这是一件棉质外套。"
    assert r.finish_reason == "stop"
    assert r.used_tools == ["mllm_client_http"] and r.warnings == []
    assert r.meta["backend"] == "trtllm" and r.meta["usage"]["total_tokens"] == 42

    sent = captured[0]["json"]
    assert captured[0]["url"] == "http://gpu:8082/v1/chat/completions"
    assert sent["model"] == "qwen-vl-7b-chat" and sent["max_tokens"] == 64
    assert sent["messages"][0] == {"role": "system", "content": "你是服饰专家。"}
    assert sent["messages"][1]["content"] == [{"type": "text", "text": "这件外套是什么面料？"}]


def test_image_bytes_are_sent_as_base64_data_uri(monkeypatch):
    captured: list = []
    _install_fake_requests(
        monkeypatch,
        chat_payload={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        capture=captured,
    )
    raw = b"\xff\xd8\xff\xe0fakejpeg"
    m.HttpMLLMClient(endpoint="http://gpu:8082").chat(query="看图", image_bytes=raw)

    blocks = captured[0]["json"]["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "看图"}
    url = blocks[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw
