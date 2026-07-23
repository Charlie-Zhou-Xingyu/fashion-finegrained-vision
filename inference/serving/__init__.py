"""
FastAPI serving infrastructure.

**P0a (current):**
    - ``schemas.py`` — all request/response Pydantic v2 models
    - ``app.py``     — FastAPI application with ``/v1/health``, ``/v1/metrics``,
      and mock endpoints for ``/v1/mm/qa``, ``/v1/intent/classify``,
      ``/v1/rag/retrieve``, ``/v1/merchant/content/generate``
    - ``deps.py``    — ``ServiceState`` singleton (no models loaded)

**Pending (P0a.2 – P1):**
    - Real intent classifier (rule-based first, embedding later)
    - Attribute service (template-based fast path)
    - RAG service (exact/alias/BM25 first, dense/reranker later)
    - QA orchestrator
    - Content generation service
    - Cache service (Redis + memory)
    - Small LLM integration (P1)
    - MLLM client (P1 — separate service by design; never loaded in main process)

**Design constraints:**
    - MLLM is NEVER loaded in the main FastAPI process.
"""
