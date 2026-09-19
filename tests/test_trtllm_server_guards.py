"""trtllm_server request-hardening logic — pure parts, no GPU / tensorrt_llm needed."""

from __future__ import annotations

import json

import pytest

from inference.serving import trtllm_server as srv


def test_limits_derive_from_engine_and_never_exceed_it(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "build_config": {"max_batch_size": 96, "max_input_len": 1024, "max_output_len": 256, "max_num_tokens": 16384,
                         "plugin_config": {"use_paged_context_fmha": True}},
        "pretrained_config": {"quantization": {"quant_algo": "W4A16", "kv_cache_quant_algo": "INT8"}},
    }))
    meta = srv.read_engine_meta(str(tmp_path))
    assert meta["max_batch_size"] == 96 and meta["quant_algo"] == "W4A16" and meta["paged_context_fmha"] is True
    for k in ("MLLM_MAX_NEW_TOKENS", "MLLM_MAX_INPUT_TOKENS", "MLLM_MAX_QUEUE", "MLLM_REQUEST_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)
    lim = srv.resolve_limits(meta)
    assert lim == {"max_new_tokens": 256, "max_input_tokens": 1024, "max_image_bytes": 5 * 1024 * 1024,
                   "max_queue": 384, "request_timeout_s": 30.0}
    monkeypatch.setenv("MLLM_MAX_NEW_TOKENS", "9999")      # cannot exceed the engine
    monkeypatch.setenv("MLLM_MAX_INPUT_TOKENS", "512")      # can tighten
    monkeypatch.setenv("MLLM_MAX_QUEUE", "abc")             # garbage → default
    lim = srv.resolve_limits(meta)
    assert lim["max_new_tokens"] == 256 and lim["max_input_tokens"] == 512 and lim["max_queue"] == 384


def test_limits_without_engine_config_fall_back():
    assert srv.read_engine_meta("/nonexistent") == {}
    lim = srv.resolve_limits({})
    assert lim["max_new_tokens"] == 256 and lim["max_input_tokens"] == 1024 and lim["max_queue"] == 32


@pytest.mark.parametrize("req,cap,expected", [
    (100, 256, (100, None)), (5000, 256, (256, "clamped")), (0, 256, (1, "< 1")),
    (-3, 256, (1, "< 1")), ("x", 256, (256, "not an int")),
])
def test_clamp_max_tokens(req, cap, expected):
    val, warn = srv.clamp_max_tokens(req, cap)
    assert val == expected[0]
    assert (warn is None) if expected[1] is None else (expected[1] in warn)


def test_chatml_template():
    s = srv.build_chatml("这件外套是什么面料？", "你是服饰专家。")
    assert s == "<|im_start|>system\n你是服饰专家。<|im_end|>\n<|im_start|>user\n这件外套是什么面料？<|im_end|>\n<|im_start|>assistant\n"


def test_extract_system_text_and_image():
    msgs = [srv.ChatMessage(role="system", content="sys"),
            srv.ChatMessage(role="user", content=[{"type": "text", "text": "q"},
                                                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}])]
    system, text, url, img = srv._extract_text_and_image(msgs)
    assert system == "sys" and text == "q" and url is None and img == b"hello"


def test_latency_window_percentiles():
    w = srv.LatencyWindow(n=4)
    for x in (0.1, 0.2, 0.3, 0.4, 0.5):        # ring buffer drops 0.1
        w.add(x)
    s = w.summary()
    assert s["completed"] == 5 and s["n"] == 4 and s["p50_ms"] == 350.0 and s["p95_ms"] == 500.0
    assert s["rejected"] == {"413": 0, "429": 0, "504": 0, "500": 0}


def test_health_reports_not_loaded_without_engine():
    from fastapi.testclient import TestClient
    with TestClient(srv.app) as c:
        h = c.get("/health").json()
        assert h["status"] == "engine_not_loaded" and h["backend"] is None
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
