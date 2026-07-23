"""
LLM abstraction layer for query translation fallback.

Preference ladder (in-process first, separate service last):
    1. vocab_router.py        — Local vocabulary mapping (<1ms, covers 95%+ queries)
    2. synonym_dict.py        — Synonym dictionary (<1ms)
    3. embedding_retriever.py — Embedding-based retrieval (~5ms)
    4. llm_client.py          — External LLM service client (~100-940ms)

No 7B model is loaded in-process in the CV pipeline. The LLM client calls
an external service (separate GPU or CPU-only quantized model).

Status: Pre-implementation. No LLM integration exists yet.
"""
