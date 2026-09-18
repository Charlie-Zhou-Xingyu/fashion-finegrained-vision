"""RAG feature-flag resolution: ``rag:`` config block + environment overrides.

Precedence (highest first): environment variable → serving config → default.
Every feature is OFF by default; the lexical retriever always stays available
as the fallback, mirroring the ``vision:`` / ``mllm:`` provider pattern.

Environment variables
---------------------
RAG_DENSE_ENABLED, RAG_EMBEDDING_MODEL, RAG_INDEX_CACHE_DIR, RAG_DEVICE,
RAG_ALLOW_MODEL_DOWNLOAD (applies to both embedder and reranker),
RAG_RERANKER_ENABLED, RAG_RERANKER_MODEL,
RAG_GENERATION_ENABLED.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

_TRUE = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _cfg_bool(section: Dict[str, Any], key: str, default: bool) -> bool:
    val = section.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in _TRUE
    return default


def _cfg_int(section: Dict[str, Any], key: str, default: int) -> int:
    val = section.get(key, default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _cfg_float(section: Dict[str, Any], key: str, default: float) -> float:
    val = section.get(key, default)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f if f == f and f >= 0.0 else default   # reject NaN / negatives


@dataclass(frozen=True)
class DenseSettings:
    enabled: bool = False
    embedding_model: str = "BAAI/bge-m3"
    index_backend: str = "faiss"            # faiss (exact IndexFlatIP) — chroma not implemented
    index_cache_dir: str = "outputs/rag_index"
    allow_model_download: bool = False      # False → HF offline mode; missing weights = unavailable
    device: str = "cpu"
    fusion: str = "rrf"                     # reciprocal rank fusion (only option)
    rrf_k: int = 60
    candidate_multiplier: int = 3           # dense candidates = top_k * multiplier


@dataclass(frozen=True)
class RerankerSettings:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 10                         # rerank at most this many fused candidates
    min_score: float = 0.0                  # drop reranked hits below this sigmoid score (0 = off);
                                            # turns the reranker's absolute score into an evidence gate
    allow_model_download: bool = False
    device: str = "cpu"


@dataclass(frozen=True)
class GenerationSettings:
    enabled: bool = False
    max_context_hits: int = 3
    max_new_tokens: int = 256
    require_citation: bool = True           # answer without [n] citation → fall back to template


@dataclass(frozen=True)
class RagSettings:
    dense: DenseSettings
    reranker: RerankerSettings
    generation: GenerationSettings


def resolve_rag_settings(config: Optional[Dict[str, Any]] = None) -> RagSettings:
    """Build ``RagSettings`` from the ``rag:`` block of *config* plus env overrides."""
    rag_cfg = (config or {}).get("rag") or {}
    d = rag_cfg.get("dense") or {}
    r = rag_cfg.get("reranker") or {}
    g = rag_cfg.get("generation") or {}

    allow_download = _env_bool(
        "RAG_ALLOW_MODEL_DOWNLOAD",
        _cfg_bool(d, "allow_model_download", False) or _cfg_bool(r, "allow_model_download", False),
    )
    device = _env_str("RAG_DEVICE", str(d.get("device") or r.get("device") or "cpu"))

    dense = DenseSettings(
        enabled=_env_bool("RAG_DENSE_ENABLED", _cfg_bool(d, "enabled", False)),
        embedding_model=_env_str("RAG_EMBEDDING_MODEL", str(d.get("embedding_model") or "BAAI/bge-m3")),
        index_backend=str(d.get("index_backend") or "faiss").lower(),
        index_cache_dir=_env_str("RAG_INDEX_CACHE_DIR", str(d.get("index_cache_dir") or "outputs/rag_index")),
        allow_model_download=allow_download,
        device=device,
        fusion=str(d.get("fusion") or "rrf").lower(),
        rrf_k=max(1, _cfg_int(d, "rrf_k", 60)),
        candidate_multiplier=max(1, _cfg_int(d, "candidate_multiplier", 3)),
    )
    reranker = RerankerSettings(
        enabled=_env_bool("RAG_RERANKER_ENABLED", _cfg_bool(r, "enabled", False)),
        model=_env_str("RAG_RERANKER_MODEL", str(r.get("model") or "BAAI/bge-reranker-v2-m3")),
        top_n=max(1, _cfg_int(r, "top_n", 10)),
        min_score=_cfg_float(r, "min_score", 0.0),
        allow_model_download=allow_download,
        device=device,
    )
    generation = GenerationSettings(
        enabled=_env_bool("RAG_GENERATION_ENABLED", _cfg_bool(g, "enabled", False)),
        max_context_hits=max(1, _cfg_int(g, "max_context_hits", 3)),
        max_new_tokens=max(1, _cfg_int(g, "max_new_tokens", 256)),
        require_citation=_cfg_bool(g, "require_citation", True),
    )
    return RagSettings(dense=dense, reranker=reranker, generation=generation)


def load_serving_config() -> Dict[str, Any]:
    """Return the serving config dict, or ``{}`` if it cannot be loaded.

    Imported lazily so ``inference.rag`` stays importable without the serving
    package's side effects.
    """
    try:
        from inference.serving.deps import get_config
        return get_config()
    except Exception:  # pragma: no cover — defensive
        return {}
