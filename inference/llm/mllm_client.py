"""
P2 — MLLMClient: multimodal LLM client for 3.2 (多模态大模型推理优化模块).

Distinct from ``llm_client.py`` (query-translation fallback ladder, <1KB
payloads, text-only). This module calls the full Qwen-VL-7B-Chat service
for §3.2.1 multimodal QA and §3.2.2 merchant content generation — image +
natural-language query in, natural-language answer out.

The 7B model is NEVER loaded in-process here. This is an HTTP client only.
Two things can sit behind the endpoint, selected entirely on the server side:
    - inference/engines/qwen_vl_trtllm_builder.py output, served by
      inference/serving/trtllm_server.py (TensorRT-LLM, GPU server only)
    - a plain HuggingFace ``transformers`` process (dev/CPU fallback)

Design mirrors inference/serving/vision_provider.py: default is ALWAYS mock
(provider=mock), real backend only activates when BOTH provider=real AND
real_enabled=true (config or env). Failures fail open to mock unless
fail_open_to_mock=false. Never fabricates an answer — mock mode returns an
explicit placeholder plus a warning, not a plausible-looking fake response.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference.serving.schemas import WarningItem, WarningSeverity

logger = logging.getLogger(__name__)

MOCK_CLIENT_VERSION = "0.1.0"

_MOCK_WARNING = WarningItem(
    code="mllm_provider_mock", scope="mllm",
    message="当前多模态大模型提供器为 mock，尚未接入真实 Qwen-VL 推理服务。",
    severity=WarningSeverity.info,
)
_UNAVAILABLE_WARNING = WarningItem(
    code="mllm_provider_unavailable", scope="mllm",
    message="多模态大模型服务不可达，已降级为 mock。",
    severity=WarningSeverity.warn,
)


# ── Result model ─────────────────────────────────────────────────────────────


@dataclass
class MLLMResult:
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    used_tools: List[str] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    latency_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "used_tools": list(self.used_tools),
            "warnings": [w.model_dump() for w in self.warnings],
            "latency_ms": self.latency_ms,
            "meta": dict(self.meta),
        }


# ── Abstract interface ──────────────────────────────────────────────────────


class MLLMClient:
    """Interface for multimodal QA / content-generation LLM calls."""

    def chat(
        self,
        *,
        query: str,
        image_url: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> MLLMResult:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


# ── Mock implementation ──────────────────────────────────────────────────────


class MockMLLMClient(MLLMClient):
    """Mock MLLM client — NEVER calls a real model. Never fabricates content."""

    def is_available(self) -> bool:
        return False

    def chat(
        self,
        *,
        query: str,
        image_url: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> MLLMResult:
        has_image = bool(image_url or image_bytes)
        return MLLMResult(
            text=None,
            finish_reason="mock",
            used_tools=["mock_mllm_client"],
            warnings=[_MOCK_WARNING],
            latency_ms=0.0,
            meta={
                "provider": "mock",
                "client_version": MOCK_CLIENT_VERSION,
                "has_image": has_image,
                "query_len": len(query or ""),
            },
        )


# ── Real implementation (OpenAI-compatible HTTP endpoint) ───────────────────


class HttpMLLMClient(MLLMClient):
    """Calls an OpenAI-chat-compatible ``/v1/chat/completions`` endpoint.

    Works against either backend behind the endpoint (trtllm_server.py on a
    GPU box, or a HF-transformers dev server) — both speak the same wire
    format, so this client does not need to know which one it is talking to.
    """

    def __init__(
        self,
        endpoint: str,
        model: str = "qwen-vl-7b-chat",
        timeout_s: float = 5.0,
        max_retries: int = 1,
        fail_open_to_mock: bool = True,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._fail_open_to_mock = fail_open_to_mock
        self._mock = MockMLLMClient()
        self._available: Optional[bool] = None
        self._available_checked_at: float = 0.0
        logger.info("[HttpMLLMClient] endpoint=%s model=%s", self._endpoint, self._model)

    def is_available(self) -> bool:
        """Health-check the endpoint. Cached for 30s to avoid hammering it."""
        now = time.monotonic()
        if self._available is not None and (now - self._available_checked_at) < 30.0:
            return self._available
        try:
            import requests  # lazy import — never required in mock mode

            resp = requests.get(f"{self._endpoint}/health", timeout=min(self._timeout, 2.0))
            self._available = resp.status_code == 200
        except Exception:  # noqa: BLE001 — any connectivity failure means unavailable
            self._available = False
        self._available_checked_at = now
        return self._available

    def chat(
        self,
        *,
        query: str,
        image_url: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> MLLMResult:
        t0 = time.perf_counter()

        if not self.is_available():
            if self._fail_open_to_mock:
                result = self._mock.chat(
                    query=query, image_url=image_url, image_bytes=image_bytes,
                    system_prompt=system_prompt, context=context,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                )
                result.warnings = [_UNAVAILABLE_WARNING]
                result.latency_ms = (time.perf_counter() - t0) * 1000
                return result
            return MLLMResult(
                text=None, finish_reason="error",
                warnings=[_UNAVAILABLE_WARNING],
                latency_ms=(time.perf_counter() - t0) * 1000,
                meta={"provider": "real", "error": "endpoint_unavailable"},
            )

        content: List[Dict[str, Any]] = [{"type": "text", "text": query}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif image_bytes:
            import base64

            b64 = base64.b64encode(image_bytes).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                import requests

                resp = requests.post(
                    f"{self._endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content")
                return MLLMResult(
                    text=text,
                    finish_reason=choice.get("finish_reason"),
                    used_tools=["mllm_client_http"],
                    warnings=[],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    meta={
                        "provider": "real",
                        "endpoint": self._endpoint,
                        "model": self._model,
                        "usage": data.get("usage", {}),
                        "backend": data.get("meta", {}).get("backend"),
                    },
                )
            except Exception as exc:  # noqa: BLE001 — retry loop, decide after
                last_exc = exc
                continue

        logger.warning("[HttpMLLMClient] request failed after %d attempt(s): %s",
                        self._max_retries + 1, last_exc)
        if self._fail_open_to_mock:
            result = self._mock.chat(
                query=query, image_url=image_url, image_bytes=image_bytes,
                system_prompt=system_prompt, context=context,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            result.warnings = [_UNAVAILABLE_WARNING]
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.meta["error"] = str(last_exc)
            return result
        return MLLMResult(
            text=None, finish_reason="error",
            warnings=[_UNAVAILABLE_WARNING],
            latency_ms=(time.perf_counter() - t0) * 1000,
            meta={"provider": "real", "error": str(last_exc)},
        )


# ── Lazy singleton with config/env-based selection ──────────────────────────

_client: Optional[MLLMClient] = None


def _resolve_mllm_settings() -> Dict[str, Any]:
    """Resolve MLLM client settings from serving_config.yaml + env overrides.

    Env vars win over YAML: MLLM_PROVIDER, MLLM_ENDPOINT, MLLM_MODEL,
    MLLM_TIMEOUT_S, MLLM_FAIL_OPEN. Any parse failure falls back to safe
    mock defaults (never raises) — same contract as vision_provider.py.
    """
    defaults: Dict[str, Any] = {
        "provider": "mock",
        "real_enabled": False,
        "endpoint": "http://localhost:8082",
        "model": "qwen-vl-7b-chat",
        "timeout_s": 5.0,
        "max_retries": 1,
        "fail_open_to_mock": True,
    }
    try:
        from inference.serving.deps import get_config

        mllm_cfg = get_config().get("mllm") or {}
        settings = dict(defaults)
        settings["provider"] = str(mllm_cfg.get("provider", "mock"))
        settings["real_enabled"] = bool(mllm_cfg.get("real_enabled", False))
        settings["endpoint"] = str(mllm_cfg.get("endpoint", defaults["endpoint"]))
        settings["model"] = str(mllm_cfg.get("model", defaults["model"]))
        settings["timeout_s"] = float(mllm_cfg.get("timeout_s", defaults["timeout_s"]))
        settings["max_retries"] = int(mllm_cfg.get("max_retries", defaults["max_retries"]))
        settings["fail_open_to_mock"] = bool(mllm_cfg.get("fail_open_to_mock", True))

        if os.getenv("MLLM_PROVIDER"):
            settings["provider"] = os.environ["MLLM_PROVIDER"].strip().lower()
        if os.getenv("MLLM_REAL_ENABLED") is not None:
            settings["real_enabled"] = os.environ["MLLM_REAL_ENABLED"].strip().lower() in ("1", "true", "yes")
        if os.getenv("MLLM_ENDPOINT"):
            settings["endpoint"] = os.environ["MLLM_ENDPOINT"]
        if os.getenv("MLLM_MODEL"):
            settings["model"] = os.environ["MLLM_MODEL"]
        if os.getenv("MLLM_TIMEOUT_S"):
            settings["timeout_s"] = float(os.environ["MLLM_TIMEOUT_S"])
        if os.getenv("MLLM_FAIL_OPEN") is not None:
            settings["fail_open_to_mock"] = os.environ["MLLM_FAIL_OPEN"].strip().lower() in ("1", "true", "yes")
        return settings
    except Exception:  # noqa: BLE001 — config problems must not break serving
        logger.warning("MLLM settings resolution failed — falling back to mock defaults.")
        return defaults


def _select_mllm_client() -> MLLMClient:
    settings = _resolve_mllm_settings()
    if settings["provider"] != "real":
        return MockMLLMClient()
    if not settings["real_enabled"]:
        logger.warning(
            "mllm_provider_real_disabled: MLLM_PROVIDER=real requested but "
            "real_enabled is false — using MockMLLMClient.")
        return MockMLLMClient()
    return HttpMLLMClient(
        endpoint=settings["endpoint"],
        model=settings["model"],
        timeout_s=settings["timeout_s"],
        max_retries=settings["max_retries"],
        fail_open_to_mock=settings["fail_open_to_mock"],
    )


def get_mllm_client() -> MLLMClient:
    """Return the process-wide MLLMClient singleton. Default is mock."""
    global _client
    if _client is None:
        _client = _select_mllm_client()
    return _client


def reset_mllm_client() -> None:
    """Reset the singleton — used by tests that toggle env/config."""
    global _client
    _client = None


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    client = MockMLLMClient()
    assert not client.is_available()
    result = client.chat(query="这件衣服是什么面料？")
    assert result.text is None
    assert result.warnings[0].code == "mllm_provider_mock"
    print("  mllm_client: mock skeleton OK (service not connected).")


if __name__ == "__main__":
    _demo()
