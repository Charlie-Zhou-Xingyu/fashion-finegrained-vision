"""
Embedding-based retrieval — third level of the LLM preference ladder.

For queries that don't match the static vocabulary or synonym dictionary,
use lightweight text embeddings (e.g., sentence-transformers) to find
the closest known fashion term. In-process, ~5ms.

Status: Skeleton. Embedding model not yet selected.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class EmbeddingRetriever:
    """Semantic search over fashion part vocabulary using text embeddings.

    Covers the long-tail (~1.5% of queries) where the user's term is
    semantically close to but lexically different from known vocabulary.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # Lazy-load sentence-transformers
        self._embeddings = None  # Pre-computed embeddings for vocab terms
        self._terms: List[str] = []
        print(f"[EmbeddingRetriever] Not yet initialized (model={model_name})")

    def _ensure_loaded(self) -> None:
        """Lazy-load the embedding model. NOT YET IMPLEMENTED."""
        # ponytail: import sentence_transformers here if needed.
        # TODO: select model based on latency budget (~5ms target).
        raise NotImplementedError(
            "EmbeddingRetriever not yet implemented. "
            "Select lightweight embedding model first."
        )

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find the top-k closest vocabulary terms to the query.

        Returns list of (term, similarity_score) tuples.
        NOT YET IMPLEMENTED.
        """
        self._ensure_loaded()
        return []


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    retriever = EmbeddingRetriever()
    print("  embedding_retriever: skeleton OK (model not loaded).")


if __name__ == "__main__":
    _demo()
