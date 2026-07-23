"""
External LLM service client — fourth/fifth level of the LLM preference ladder.

Calls an external translation service (separate GPU or remote API) for
queries that cannot be resolved by local vocab, synonyms, or embeddings.
Does NOT load a 7B model in-process.

Status: Skeleton. Service endpoint and protocol not yet defined.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class LLMClient:
    """Client for external LLM translation service.

    The LLM service (Qwen-VL-7B or similar) runs on a separate GPU or
    as a CPU-only quantized model. This client handles:
    - Connection pooling
    - Timeouts and retries
    - Graceful degradation when the service is unavailable
    - Response caching (same query → cached result)

    Usage::

        client = LLMClient(endpoint="http://localhost:8081/translate")
        result = client.translate("这个衣服的左边有什么装饰")
        # -> {"translation": "pocket", "confidence": 0.92}
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8081/translate",
        timeout_s: float = 2.0,
        max_retries: int = 2,
    ) -> None:
        self._endpoint = endpoint
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._available: Optional[bool] = None  # None = not yet checked
        print(f"[LLMClient] Endpoint: {endpoint} (not yet connected)")

    def is_available(self) -> bool:
        """Check if the LLM service is reachable. Cached for 30s."""
        # ponytail: implement health-check request to endpoint.
        # TODO Week 7: wire up actual HTTP health check.
        return False

    def translate(self, query: str) -> Optional[Dict[str, Any]]:
        """Translate a Chinese fashion query to canonical part name.

        Returns None if the service is unavailable or the query cannot
        be translated. Results are cached in-process after the first call.

        NOT YET IMPLEMENTED — returns None for all queries.
        """
        # Check cache first
        if query in self._cache:
            return self._cache[query]

        if not self.is_available():
            return None

        # ponytail: HTTP POST to self._endpoint with {"query": query}
        # TODO Week 7: implement actual HTTP request.
        return None

    def clear_cache(self) -> None:
        """Clear the translation cache (e.g., after vocab update)."""
        self._cache.clear()


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    client = LLMClient()
    assert not client.is_available()
    assert client.translate("口袋") is None
    print("  llm_client: skeleton OK (service not connected).")


if __name__ == "__main__":
    _demo()
