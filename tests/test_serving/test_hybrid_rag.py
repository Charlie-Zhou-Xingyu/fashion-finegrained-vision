"""P2 — hybrid retrieval (dense fusion + rerank) tests.

Fakes stand in for BGE-M3 / the reranker so the suite never loads a model;
the real stack is exercised by ``inference/benchmarks/rag_hybrid_smoke.py``.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence, Set, Tuple

import pytest

import numpy as np

from inference.rag.dense import BgeM3FaissRetriever, DenseUnavailable, build_dense_retriever, build_reranker
from inference.rag.settings import DenseSettings, RagSettings, RerankerSettings, GenerationSettings, resolve_rag_settings
from inference.serving.rag_service import RagService


# ── Fakes ──────────────────────────────────────────────────────────────────────


class FakeDense:
    name = "fake_dense"

    def __init__(self, results: List[Tuple[str, float]], fail: bool = False) -> None:
        self.results = results
        self.fail = fail
        self.calls: List[dict] = []

    def search(self, query: str, *, allowed_ids: Optional[Set[str]] = None, top_k: int = 10):
        self.calls.append({"query": query, "allowed_ids": allowed_ids, "top_k": top_k})
        if self.fail:
            raise RuntimeError("boom")
        return [(i, s) for i, s in self.results if allowed_ids is None or i in allowed_ids][:top_k]


class FakeReranker:
    name = "fake_reranker"

    def __init__(self, scores: dict, fail: bool = False) -> None:
        self.scores = scores
        self.fail = fail

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]):
        if self.fail:
            raise RuntimeError("boom")
        ranked = [(i, self.scores.get(i, 0.0)) for i, _ in candidates]
        ranked.sort(key=lambda t: -t[1])
        return ranked


def _settings(dense: bool = True, rerank: bool = False, **dense_kw) -> RagSettings:
    """Only enable what a test injects a fake for — an enabled component with no
    injected fake makes RagService try to load the real model from cache."""
    return RagSettings(
        dense=DenseSettings(enabled=dense, **dense_kw),
        reranker=RerankerSettings(enabled=rerank, top_n=10),
        generation=GenerationSettings(),
    )


@pytest.fixture(scope="module")
def lexical() -> RagService:
    return RagService()


@pytest.fixture(scope="module")
def doc_ids(lexical) -> List[str]:
    return [d["id"] for d in lexical._docs]


# ── Default path is untouched ──────────────────────────────────────────────────


def test_default_is_lexical_only(lexical):
    r = lexical.retrieve("什么是纤维")
    assert r.used_tools == ["rag_service"]
    assert r.meta["retrieval"] == {"retrieval_mode": "lexical", "dense_used": False, "reranker_used": False}
    assert lexical._dense is None and lexical._reranker is None


# ── Dense fusion ───────────────────────────────────────────────────────────────


def test_dense_only_doc_is_added_with_dense_match_type(doc_ids):
    dense = FakeDense([(doc_ids[0], 0.81)])
    svc = RagService(rag_settings=_settings(), dense_retriever=dense)
    r = svc.retrieve("zzqqxx")            # no lexical hit possible
    assert [h.id for h in r.hits] == [doc_ids[0]]
    h = r.hits[0]
    assert h.match_type == "dense"
    assert 0.0 < h.score <= 1.0
    assert h.metadata["retrieval"]["dense_score"] == 0.81
    assert h.metadata["retrieval"]["lexical_score"] is None
    assert r.meta["retrieval"]["retrieval_mode"] == "hybrid_rrf"
    assert "dense_retriever:fake_dense" in r.used_tools


def test_rrf_prefers_doc_found_by_both_retrievers(lexical, doc_ids):
    both, dense_only = doc_ids[0], doc_ids[1]
    term = lexical._docs[0]["term"]      # exact lexical hit for doc 0
    dense = FakeDense([(dense_only, 0.95), (both, 0.90)])   # dense ranks the other doc first
    svc = RagService(rag_settings=_settings(), dense_retriever=dense)
    r = svc.retrieve(term, top_k=5)
    ids = [h.id for h in r.hits]
    assert ids.index(both) < ids.index(dense_only)
    top = r.hits[0]
    assert top.match_type == "exact"                      # lexical tier kept for priority
    # lexical rank 1 + dense rank 2, normalised by the rank-1-in-both maximum
    k = svc._rag_settings.dense.rrf_k
    assert top.score == round((1 / (k + 1) + 1 / (k + 2)) / (2 / (k + 1)), 4)
    assert top.metadata["retrieval"]["lexical_match_type"] == "exact"


def test_rrf_dense_only_rank_one_scores_half():
    # A doc seen by exactly one retriever at rank 1 gets half the maximum.
    svc = RagService(rag_settings=_settings(), dense_retriever=FakeDense([("material_term_001", 0.9)]))
    r = svc.retrieve("zzqqxx")
    assert r.hits[0].score == 0.5


def test_category_filter_is_applied_to_dense(lexical, doc_ids):
    dense = FakeDense([(i, 0.5) for i in doc_ids])
    svc = RagService(rag_settings=_settings(), dense_retriever=dense)
    r = svc.retrieve("zzqqxx", categories=["fiber"], top_k=10)
    allowed = dense.calls[0]["allowed_ids"]
    fiber_ids = {d["id"] for d in lexical._docs if d["category"] == "fiber"}
    assert allowed == fiber_ids
    assert {h.id for h in r.hits} <= fiber_ids


def test_dense_candidate_budget_is_top_k_times_multiplier(doc_ids):
    dense = FakeDense([])
    svc = RagService(rag_settings=_settings(candidate_multiplier=4), dense_retriever=dense)
    svc.retrieve("zzqqxx", top_k=2)
    assert dense.calls[0]["top_k"] == 8


def test_dense_failure_fails_open_to_lexical(lexical):
    term = lexical._docs[0]["term"]
    svc = RagService(rag_settings=_settings(), dense_retriever=FakeDense([], fail=True))
    r = svc.retrieve(term)
    assert r.hits and r.hits[0].id == lexical._docs[0]["id"]
    assert any(w.code == "dense_search_failed" for w in r.warnings)
    assert r.meta["retrieval"]["dense_used"] is False
    assert r.used_tools == ["rag_service"]


# ── Reranking ──────────────────────────────────────────────────────────────────


def test_reranker_reorders_head_and_replaces_scores(doc_ids):
    a, b, c = doc_ids[:3]
    dense = FakeDense([(a, 0.9), (b, 0.8), (c, 0.7)])
    rr = FakeReranker({a: 0.10, b: 0.95, c: 0.40})
    svc = RagService(rag_settings=_settings(rerank=True), dense_retriever=dense, reranker=rr)
    r = svc.retrieve("zzqqxx", top_k=3)
    assert [h.id for h in r.hits] == [b, c, a]
    assert [h.score for h in r.hits] == [0.95, 0.40, 0.10]
    assert r.hits[0].metadata["retrieval"]["pre_rerank_score"] > 0
    assert r.meta["retrieval"]["reranker_used"] is True
    assert "reranker:fake_reranker" in r.used_tools


def test_reranker_failure_keeps_order(doc_ids):
    a, b = doc_ids[:2]
    svc = RagService(rag_settings=_settings(rerank=True), dense_retriever=FakeDense([(a, 0.9), (b, 0.8)]),
                     reranker=FakeReranker({}, fail=True))
    r = svc.retrieve("zzqqxx", top_k=2)
    assert [h.id for h in r.hits] == [a, b]
    assert any(w.code == "rerank_failed" for w in r.warnings)
    assert r.meta["retrieval"]["reranker_used"] is False


def test_reranker_min_score_gates_evidence(doc_ids):
    a, b, c = doc_ids[:3]
    dense = FakeDense([(a, 0.9), (b, 0.8), (c, 0.7)])
    settings = RagSettings(dense=DenseSettings(enabled=True),
                           reranker=RerankerSettings(enabled=True, top_n=10, min_score=0.05),
                           generation=GenerationSettings())
    svc = RagService(rag_settings=settings, dense_retriever=dense, reranker=FakeReranker({a: 0.30, b: 0.04, c: 0.00}))
    r = svc.retrieve("zzqqxx", top_k=3)
    assert [h.id for h in r.hits] == [a]
    assert r.meta["retrieval"]["rerank_dropped_below_min_score"] == 2
    assert not any(w.code == "no_relevant_evidence" for w in r.warnings)

    svc = RagService(rag_settings=settings, dense_retriever=dense, reranker=FakeReranker({a: 0.01, b: 0.0, c: 0.0}))
    r = svc.retrieve("zzqqxx", top_k=3)
    assert r.hits == []
    codes = [w.code for w in r.warnings]
    assert "no_relevant_evidence" in codes and "no_hits" in codes


def test_min_score_setting_rejects_bad_values():
    assert resolve_rag_settings({"rag": {"reranker": {"min_score": "0.2"}}}).reranker.min_score == 0.2
    assert resolve_rag_settings({"rag": {"reranker": {"min_score": -1}}}).reranker.min_score == 0.0
    assert resolve_rag_settings({"rag": {"reranker": {"min_score": "abc"}}}).reranker.min_score == 0.0


def test_reranker_works_on_lexical_hits_without_dense(lexical, doc_ids):
    term = lexical._docs[0]["term"]
    rr = FakeReranker({doc_ids[0]: 0.33})
    svc = RagService(rag_settings=_settings(dense=False, rerank=True), reranker=rr)
    assert svc._dense is None
    r = svc.retrieve(term)
    assert r.hits[0].id == doc_ids[0] and r.hits[0].score == 0.33
    assert r.meta["retrieval"]["retrieval_mode"] == "lexical"
    assert r.meta["retrieval"]["reranker_used"] is True


# ── Builders / settings ────────────────────────────────────────────────────────


def test_builders_return_none_when_disabled():
    assert build_dense_retriever([("x", "y")], DenseSettings(enabled=False), "v") == (None, None)
    assert build_reranker(RerankerSettings(enabled=False)) == (None, None)


def test_builder_rejects_unimplemented_backend():
    r, err = build_dense_retriever([("x", "y")], DenseSettings(enabled=True, index_backend="chroma"), "v")
    assert r is None and "not implemented" in err


# ── Real index code path with a stub encoder (no model) ────────────────────────

_DOCS = [("a", "红色 连衣裙"), ("b", "蓝色 牛仔裤"), ("c", "羊毛 大衣")]
_VECS = {"红色 连衣裙": [1, 0, 0], "蓝色 牛仔裤": [0, 1, 0], "羊毛 大衣": [0.6, 0.8, 0],
         "q_red": [0.9, 0.1, 0], "q_mix": [0.5, 0.5, 0]}


def _stub_encoder(retriever: BgeM3FaissRetriever) -> None:
    retriever._encode = lambda texts: np.asarray([_VECS[t] for t in texts], dtype=np.float32) / np.linalg.norm(
        np.asarray([_VECS[t] for t in texts], dtype=np.float32), axis=1, keepdims=True)
    retriever._load_encoder = lambda: None


@pytest.mark.parametrize("backend", ["numpy", pytest.param("faiss", marks=pytest.mark.skipif(
    sys.platform == "darwin", reason="pip faiss-cpu + torch clash on libomp under macOS"))])
def test_index_backend_exact_cosine_and_filters(tmp_path, backend):
    r = BgeM3FaissRetriever(_DOCS, DenseSettings(enabled=True, index_backend=backend, index_cache_dir=str(tmp_path)), "v1")
    _stub_encoder(r)
    r.warmup()
    hits = r.search("q_red", top_k=3)
    assert [h[0] for h in hits] == ["a", "c", "b"]
    assert hits[0][1] == pytest.approx(0.9 / np.hypot(0.9, 0.1), abs=1e-5)   # exact cosine
    assert r.search("q_red", allowed_ids={"b", "c"}, top_k=1) == [("c", pytest.approx(hits[1][1], abs=1e-6))]
    assert r.search("q_mix", top_k=2)[0][0] == "c"                         # ties broken by score, not order


def test_index_cache_roundtrip_and_kb_invalidation(tmp_path):
    settings = DenseSettings(enabled=True, index_backend="numpy", index_cache_dir=str(tmp_path))
    r1 = BgeM3FaissRetriever(_DOCS, settings, "v1"); _stub_encoder(r1); r1.warmup()
    assert len(list(tmp_path.glob("*.npz"))) == 1
    r2 = BgeM3FaissRetriever(_DOCS, settings, "v1")
    r2._encode = lambda texts: (_ for _ in ()).throw(AssertionError("must hit cache"))
    r2._load_encoder = lambda: None
    r2._ensure_index()                                                       # served from cache
    r3 = BgeM3FaissRetriever(_DOCS, settings, "v2"); _stub_encoder(r3); r3.warmup()
    assert len(list(tmp_path.glob("*.npz"))) == 2                           # new kb_version → new file


def test_builder_fails_open_without_flagembedding(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)   # import → ImportError
    r, err = build_dense_retriever(
        [("x", "y")], DenseSettings(enabled=True, index_cache_dir=str(tmp_path)), "v",
    )
    assert r is None and "FlagEmbedding" in err


def test_service_surfaces_startup_unavailability(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    svc = RagService(rag_settings=_settings(rerank=True, index_cache_dir=str(tmp_path)))
    assert svc._dense is None and svc._reranker is None
    r = svc.retrieve("什么是纤维")
    codes = {w.code for w in r.warnings}
    assert {"dense_retriever_unavailable", "reranker_unavailable"} <= codes
    assert r.hits                                           # lexical still answers


def test_env_overrides_config(monkeypatch):
    cfg = {"rag": {"dense": {"enabled": False, "rrf_k": 10}, "generation": {"enabled": True}}}
    monkeypatch.setenv("RAG_DENSE_ENABLED", "true")
    monkeypatch.setenv("RAG_GENERATION_ENABLED", "0")
    monkeypatch.setenv("RAG_ALLOW_MODEL_DOWNLOAD", "yes")
    s = resolve_rag_settings(cfg)
    assert s.dense.enabled is True and s.dense.rrf_k == 10
    assert s.dense.allow_model_download is True and s.reranker.allow_model_download is True
    assert s.generation.enabled is False
    assert s.reranker.enabled is False


def test_defaults_are_all_off():
    s = resolve_rag_settings({})
    assert not s.dense.enabled and not s.reranker.enabled and not s.generation.enabled
    assert not s.dense.allow_model_download
