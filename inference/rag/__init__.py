"""P2 — RAG enhancements for the serving layer (PRD §3.3).

Three independently gated pieces, all default-off so the deterministic
lexical ``RagService`` behaviour is unchanged unless explicitly enabled:

* ``settings``    — config + env resolution (``rag:`` block in serving_config).
* ``dense``       — BGE-M3 dense retrieval over the KB (FAISS exact index) and
                    BGE-reranker-v2-m3 cross-encoder reranking; fail-open.
* ``generation``  — grounded-prompt construction and citation checking for
                    MLLM answer generation (used by ``QaOrchestrator``).

Nothing here loads a model at import time.
"""
