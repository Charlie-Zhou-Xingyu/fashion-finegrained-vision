"""Dense retrieval (BGE-M3 + FAISS) and cross-encoder reranking (BGE-reranker).

Design notes
------------
* The KB is small (tens of documents), so the index is *exact* cosine
  similarity over L2-normalised vectors.  Two interchangeable backends:
  ``faiss`` (``IndexFlatIP``, the PRD's choice, scales to ANN variants) and
  ``numpy`` (a matmul — bit-identical results, zero extra dependencies).
  IVF/HNSW only pays off at ~1e5+ vectors; using them here would add recall
  loss for nothing.  Note: on macOS the pip ``faiss-cpu`` wheel vendors its
  own ``libomp.dylib`` next to torch's → "OMP: Error #15" / SIGSEGV once both
  have run; use ``numpy`` there (or conda-forge faiss, which shares torch's
  OpenMP).  ``KMP_DUPLICATE_LIB_OK=TRUE`` is an unsupported workaround.
* Category filtering is applied by the caller through ``allowed_ids`` and is
  enforced *after* scoring but *before* truncation, which for an exact index
  is equivalent to filtering before scoring (the invariant ``RagService``
  keeps for lexical search).
* Embeddings are cached under ``index_cache_dir`` keyed by a fingerprint of
  (model, kb_version, doc ids + texts), so a KB edit invalidates the cache.
* Model loading honours ``allow_model_download``: when False the HF hub is put
  in offline mode for the duration of the load, so a missing local snapshot
  raises ``DenseUnavailable`` instead of silently pulling ~2 GB at request
  time.  Failures are reported to the caller, which falls open to lexical.
* Nothing is loaded at import time; ``warmup()`` is called once by the
  builder so model-load cost lands at service start, not on the first query.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

from inference.rag.settings import DenseSettings, RerankerSettings

logger = logging.getLogger(__name__)


class DenseUnavailable(RuntimeError):
    """Raised when the dense stack cannot be brought up (missing lib / weights)."""


# ── Protocols (what RagService depends on; tests inject fakes) ─────────────────


class DenseRetriever(Protocol):
    name: str

    def search(
        self, query: str, *, allowed_ids: Optional[Set[str]] = None, top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return ``[(doc_id, cosine_score)]`` sorted descending, filtered to *allowed_ids*."""


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """Score ``[(doc_id, text)]`` pairs; return ``[(doc_id, score)]`` sorted descending."""


# ── Helpers ────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _hf_offline(enabled: bool) -> Iterator[None]:
    """Temporarily force Hugging Face hub / transformers offline mode."""
    if not enabled:
        yield
        return
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _fingerprint(model: str, kb_version: str, docs: Sequence[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(kb_version.encode("utf-8"))
    for doc_id, text in docs:
        h.update(b"\x00")
        h.update(doc_id.encode("utf-8"))
        h.update(b"\t")
        h.update(text.encode("utf-8"))
    return h.hexdigest()[:24]


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


class _NumpyFlatIP:
    """Exact inner-product index with faiss's ``search(q, k) -> (scores, idxs)`` shape."""

    def __init__(self, vecs: np.ndarray) -> None:
        self._vecs = vecs

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = queries @ self._vecs.T                       # (nq, n)
        k = min(k, scores.shape[1])
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, order, axis=1), order


def _build_index(backend: str, vecs: np.ndarray):
    if backend == "numpy":
        return _NumpyFlatIP(vecs)
    if backend == "faiss":
        try:
            import faiss
        except ImportError as exc:
            raise DenseUnavailable("faiss is not installed") from exc
        index = faiss.IndexFlatIP(int(vecs.shape[1]))
        index.add(vecs)
        return index
    raise DenseUnavailable(f"index_backend={backend!r} is not implemented (faiss | numpy)")


SUPPORTED_INDEX_BACKENDS = ("faiss", "numpy")


# ── BGE-M3 + FAISS ─────────────────────────────────────────────────────────────


class BgeM3FaissRetriever:
    name = "bge_m3_faiss"

    def __init__(self, docs: Sequence[Tuple[str, str]], settings: DenseSettings, kb_version: str) -> None:
        self._ids: List[str] = [d[0] for d in docs]
        self._texts: List[str] = [d[1] for d in docs]
        self._settings = settings
        self._kb_version = kb_version
        self._fingerprint = _fingerprint(settings.embedding_model, kb_version, docs)
        self._encoder = None
        self._index = None

    # -- lifecycle ---------------------------------------------------------

    def warmup(self) -> None:
        self._ensure_index()
        self._load_encoder()

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        s = self._settings
        with _hf_offline(not s.allow_model_download):
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise DenseUnavailable("FlagEmbedding is not installed") from exc
            try:
                self._encoder = BGEM3FlagModel(
                    s.embedding_model,
                    normalize_embeddings=True,
                    use_fp16=False,
                    devices=s.device,
                    return_dense=True, return_sparse=False, return_colbert_vecs=False,
                )
            except Exception as exc:  # weights missing offline, bad device, ...
                raise DenseUnavailable(
                    f"cannot load embedding model {s.embedding_model!r} "
                    f"(allow_model_download={s.allow_model_download}): {exc}"
                ) from exc

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load_encoder()
        out = self._encoder.encode(list(texts), batch_size=16, max_length=512)
        return _l2_normalize(np.asarray(out["dense_vecs"], dtype=np.float32))

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        cache_path = Path(self._settings.index_cache_dir) / f"{self._fingerprint}.npz"
        vecs: Optional[np.ndarray] = None
        if cache_path.exists():
            try:
                with np.load(cache_path) as z:
                    cached = z["vecs"]
                if cached.shape[0] == len(self._ids):
                    vecs = cached.astype(np.float32)
                    logger.info("dense index: loaded %d cached vectors from %s", len(vecs), cache_path)
            except Exception:
                logger.warning("dense index: cache %s unreadable, re-encoding", cache_path)
        if vecs is None:
            if not self._ids:
                raise DenseUnavailable("knowledge base has no documents to index")
            vecs = self._encode(self._texts)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, vecs=vecs)
            except OSError as exc:
                logger.warning("dense index: could not write cache %s: %s", cache_path, exc)

        self._index = _build_index(self._settings.index_backend, vecs)

    # -- query -------------------------------------------------------------

    def search(
        self, query: str, *, allowed_ids: Optional[Set[str]] = None, top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        self._ensure_index()
        q = self._encode([query])
        n = len(self._ids)
        scores, idxs = self._index.search(q, n)     # exact: rank every doc, filter after
        out: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            doc_id = self._ids[int(idx)]
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            out.append((doc_id, float(score)))
            if len(out) >= top_k:
                break
        return out


# ── BGE reranker (cross-encoder) ───────────────────────────────────────────────


class BgeReranker:
    name = "bge_reranker_v2_m3"

    def __init__(self, settings: RerankerSettings) -> None:
        self._settings = settings
        self._model = None

    def warmup(self) -> None:
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        s = self._settings
        with _hf_offline(not s.allow_model_download):
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise DenseUnavailable("FlagEmbedding is not installed") from exc
            try:
                # normalize=True → sigmoid, so scores are comparable probabilities in [0, 1].
                self._model = FlagReranker(s.model, use_fp16=False, normalize=True, devices=s.device)
            except Exception as exc:
                raise DenseUnavailable(
                    f"cannot load reranker {s.model!r} (allow_model_download={s.allow_model_download}): {exc}"
                ) from exc

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        self._load()
        scores = self._model.compute_score([(query, text) for _, text in candidates], normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]
        ranked = [(doc_id, float(s)) for (doc_id, _), s in zip(candidates, scores)]
        ranked.sort(key=lambda t: (-t[1], t[0]))
        return ranked


# ── Builders (fail-open: return (None, reason) instead of raising) ─────────────


def build_dense_retriever(
    docs: Sequence[Tuple[str, str]], settings: DenseSettings, kb_version: str,
) -> Tuple[Optional[DenseRetriever], Optional[str]]:
    if not settings.enabled:
        return None, None
    if settings.index_backend not in SUPPORTED_INDEX_BACKENDS:
        return None, (f"index_backend={settings.index_backend!r} is not implemented "
                      f"({' | '.join(SUPPORTED_INDEX_BACKENDS)})")
    if settings.fusion != "rrf":
        return None, f"fusion={settings.fusion!r} is not implemented (rrf only)"
    try:
        retriever = BgeM3FaissRetriever(docs, settings, kb_version)
        retriever.warmup()
    except DenseUnavailable as exc:
        logger.warning("dense retriever unavailable, falling back to lexical: %s", exc)
        return None, str(exc)
    logger.info("dense retriever ready: %s over %d docs (index=%s)",
                settings.embedding_model, len(docs), settings.index_backend)
    return retriever, None


def build_reranker(settings: RerankerSettings) -> Tuple[Optional[Reranker], Optional[str]]:
    if not settings.enabled:
        return None, None
    try:
        reranker = BgeReranker(settings)
        reranker.warmup()
    except DenseUnavailable as exc:
        logger.warning("reranker unavailable, skipping rerank: %s", exc)
        return None, str(exc)
    logger.info("reranker ready: %s", settings.model)
    return reranker, None
