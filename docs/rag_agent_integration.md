# RAG + Agent integration (PRD §3.3 / §3.4) — engineering record

Companion to `docs/tensorrt_llm_integration.md` (§3.2).  Everything here runs
on CPU; nothing needs the GPU box.

## 1. What was built

| piece | file | default | what it adds |
|---|---|---|---|
| Feature flags | `inference/rag/settings.py` | all off | `rag:` config block + `RAG_*` env overrides, same precedence as `vision:`/`mllm:` |
| Dense retrieval | `inference/rag/dense.py` | off | BGE-M3 dense vectors over the KB, exact cosine index (`numpy` or `faiss`), embedding cache keyed by (model, kb_version, docs) |
| Reranking | `inference/rag/dense.py` | off | BGE-reranker-v2-m3 cross-encoder, sigmoid scores |
| Hybrid fusion | `inference/serving/rag_service.py` | off | lexical ⊕ dense by reciprocal rank fusion; reranker over the fused head; per-hit provenance in `metadata.retrieval` |
| Grounded generation | `inference/rag/generation.py`, `qa_orchestrator.py` | off | cited-answer prompt over the top-k hits, citation check, fallback to the template answer |
| Agent graph | `inference/agent/qa_graph.py` | n/a (library) | LangGraph state machine: classify → retrieve → generate → verify with a bounded grounding-retry loop and a node trace |

Tests: `tests/test_serving/test_hybrid_rag.py` (fusion, rerank, filters,
fail-open, index backends, cache), `test_grounded_generation.py`,
`test_qa_graph.py` — 45 tests, no model loaded.  The pre-existing 65
RagService/orchestrator tests pass unchanged, which is the point: with every
flag off the serving path is byte-identical to before.

## 2. Design decisions and why

**Mock-first, fail-open, per-request honesty.**  Each real component is
built once at service start; if it cannot load (library missing, weights not
cached, wrong backend) the service starts anyway in lexical mode and *every*
response carries a `dense_retriever_unavailable` / `reranker_unavailable`
warning.  A degraded service that looks healthy is worse than one that says
so.

**Reciprocal rank fusion, not score interpolation.**  The lexical retriever
emits tiered scores (exact 1.0 / alias 0.92 / title 0.85 / bm25 ≤ 0.75) and
the dense retriever emits cosines that cluster in 0.3–0.7 for BGE-M3.  These
are not on a common scale, so `α·lex + (1-α)·cos` would need a tuned α per
KB.  RRF (`Σ 1/(k+rank)`, k = 60) only uses ranks, needs no tuning, and has
the property we actually want: a doc found by *both* retrievers outranks one
found by only one at the same rank.  The fused score is normalised by the
best achievable value (rank 1 in both lists) so `score` stays in [0, 1] and
downstream consumers do not change.  The original lexical `match_type` is
kept for tie-break priority; dense-only docs get `match_type="dense"`.

**Exact index, two backends.**  The KB has 15 documents.  Any ANN index
(IVF, HNSW) would add recall loss for zero latency benefit below ~1e5
vectors, so both backends are exact cosine: `faiss.IndexFlatIP` (the PRD's
choice, and the one that grows into ANN later) or a numpy matmul that is
bit-identical.  The numpy backend exists because of a real pitfall, see §4.

**Category filter before truncation.**  The lexical path filters by
category *before* scoring (invariant A2 of the original service).  The dense
path passes the same allowed-id set into `search()`, which ranks every doc
and drops disallowed ids before taking top-k — for an exact index that is
equivalent, and it keeps one source of truth for what "allowed" means.

**Reranked set is always the returned set.**  `top_n` is raised to at least
`top_k`, otherwise a tail hit carrying a fused score in [0, 1] could outrank
a reranked head hit carrying a sigmoid score — mixing scales inside one
result list.

**Generation never runs without evidence, and never returns uncited text.**
The orchestrator calls the MLLM only if retrieval produced hits *and* the
client reports itself available.  The prompt numbers the hits, tells the
model to cite `[n]` and to reply with a fixed refusal phrase if the entries
do not cover the question.  `check_grounding` accepts a reply only if it
cites a valid index or contains the refusal phrase; `[7]` with three entries
does not count.  Anything else falls back to the template answer with a
`generation_ungrounded` warning.  `sources` are always the retrieval
sources, never something the model wrote.  With the default
`MockMLLMClient` (which reports unavailable) the knowledge route is
unchanged.

**Why a LangGraph graph on top of an orchestrator that already routes.**
The orchestrator is a single-pass dispatcher; it cannot cleanly express a
*bounded retry* (ungrounded → re-prompt once with a stricter instruction →
template) or expose *which node ran with what state*.  Those are the
runtime's job, so `qa_graph` does exactly that and nothing more: every node
delegates to the existing classifier / RagService / MLLM client, the
grounding policy is imported from the same module the orchestrator uses,
and a transport failure short-circuits to the template instead of burning
the retry.  `run_qa_graph()` returns the final state including `trace`
(e.g. `["classify","retrieve","generate","verify","generate","verify"]`),
`attempts`, `cited_ids` and structured warnings.  LangGraph is imported
lazily; the serving package does not require it.

## 3. Configuration

```yaml
rag:
  dense:     {enabled: false, embedding_model: BAAI/bge-m3, index_backend: faiss,
              index_cache_dir: outputs/rag_index, allow_model_download: false,
              device: cpu, fusion: rrf, rrf_k: 60, candidate_multiplier: 3}
  reranker:  {enabled: false, model: BAAI/bge-reranker-v2-m3, top_n: 10}
  generation:{enabled: false, max_context_hits: 3, max_new_tokens: 256, require_citation: true}
```
Env overrides: `RAG_DENSE_ENABLED`, `RAG_RERANKER_ENABLED`,
`RAG_GENERATION_ENABLED`, `RAG_ALLOW_MODEL_DOWNLOAD`, `RAG_EMBEDDING_MODEL`,
`RAG_RERANKER_MODEL`, `RAG_INDEX_CACHE_DIR`, `RAG_DEVICE`.
`embedding_model` may be a local directory — recommended for offline
deployments (see pitfall 2).

Full path in production: `RAG_DENSE_ENABLED=1 RAG_RERANKER_ENABLED=1
RAG_GENERATION_ENABLED=1 MLLM_PROVIDER=real MLLM_REAL_ENABLED=1
MLLM_ENDPOINT=http://<gpu-box>:8082` with `trtllm_server.py` serving the
Qwen-VL engines from §3.2.

## 4. Pitfalls hit (all reproducible)

1. **FlagEmbedding + offline mode demands a byte-complete hub snapshot.**
   With `HF_HUB_OFFLINE=1`, `BGEM3FlagModel("BAAI/bge-m3")` refuses to load
   if *any* repo file is missing from the cache — including `README.md`,
   `imgs/*` and the 2 GB `onnx/` export nobody uses.  A snapshot pulled with
   `allow_patterns` therefore "does not exist".  Fix: point
   `embedding_model` at the snapshot directory; FlagEmbedding then loads the
   directory directly.  (The reranker repo is small enough that a filtered
   snapshot happened to be complete.)
2. **`BAAI/bge-m3` ships `pytorch_model.bin`, not `model.safetensors`.**
   The 2.2 GB weight file has to be fetched explicitly; the safetensors
   filter fetched nothing.  `colbert_linear.pt` / `sparse_linear.pt` (2 MB)
   are also loaded by the M3 wrapper even when only dense vectors are used.
3. **pip `faiss-cpu` + torch on macOS → `OMP: Error #15` then SIGSEGV.**
   Both wheels vendor their own `libomp.dylib`
   (`torch/lib/libomp.dylib`, `faiss/.dylibs/libomp.dylib`); the process
   aborts the first time both runtimes have executed (torch forward, then
   `index.search`).  Exit code 139, nothing on stdout.  Fix used: the
   `numpy` index backend (identical results at this KB size); alternative:
   conda-forge faiss, which links torch's OpenMP.  `KMP_DUPLICATE_LIB_OK=TRUE`
   made a 4-dim toy example pass but the full smoke test **still
   segfaulted** with it, so it is not a workaround here.  Consequence: the
   faiss code path is covered by a unit test that is skipped on macOS
   (`test_index_backend_exact_cosine_and_filters[faiss]`) and has *not* been
   exercised with the real embedder on this machine — it is the same 6
   lines as numpy around `IndexFlatIP.search`, but say "untested on Mac,
   test on the Linux box" rather than "verified".
4. **An enabled component with no injected fake loads the real model in
   tests.**  Caught when a unit test took 30 s and printed
   "Loading weights 393/393": the test enabled the reranker in settings but
   did not pass a fake, so the service dutifully loaded BGE-reranker from
   cache.  Tests now enable exactly what they fake.

## 5. Real-model smoke test (CPU, Apple Silicon)

`inference/benchmarks/rag_hybrid_smoke.py` loads both real models and
prints top-3 ids for five query types under lexical / +dense / +rerank.

Run: `PYTHONPATH=. python inference/benchmarks/rag_hybrid_smoke.py
--embedding-model <bge-m3 snapshot dir> --index-backend numpy`, Apple
Silicon CPU, 2026-09-18.  Both models loaded in 10.6 s.  Verbatim:

```
Q: 什么是纤维
  lexical      0.8ms  fiber_term_001(exact,1.00), animal_fibers_001(title,0.89), mmcf_001(title,0.89)
  +dense     123.1ms  fiber_term_001(exact,1.00), mmcf_001(title,0.98), animal_fibers_001(title,0.95)
  +rerank   5138.2ms  fiber_term_001(exact,1.00), synthetic_fibers_001(title,0.63), mmcf_001(title,0.55)
Q: 衣服洗了会缩水是什么原因
  lexical      0.9ms  nonrenewable_materials_001(title,0.89), raw_materials_001(title,0.89), recycled_materials_001(title,0.89)
  +dense     194.9ms  raw_materials_001(title,0.95), nonrenewable_materials_001(title,0.50), circularity_001(dense,0.50)
  +rerank   1789.0ms  cotton_001(dense,0.00), synthetic_fibers_001(dense,0.00), material_term_001(dense,0.00)
Q: 哪种材质比较环保
  lexical      0.3ms  animal_fibers_001(title,0.89), conventional_material_001(title,0.89), material_term_001(title,0.89)
  +dense     136.9ms  animal_fibers_001(title,0.97), preferred_material_001(title,0.96), conventional_material_001(title,0.95)
  +rerank   1938.1ms  animal_fibers_001(title,0.14), renewable_materials_001(title,0.12), plant_fibers_001(title,0.11)
Q: 涤纶和聚酯纤维是一回事吗
  lexical      1.1ms  fiber_term_001(exact,1.00), animal_fibers_001(title,0.89), mmcf_001(title,0.89)
  +dense     135.4ms  fiber_term_001(exact,0.99), synthetic_fibers_001(title,0.96), nonfiber_term_001(title,0.96)
  +rerank   1796.5ms  synthetic_fibers_001(title,0.11), cotton_001(dense,0.05), fiber_term_001(exact,0.02)
Q: 夏天穿什么面料凉快
  lexical      0.4ms  animal_fibers_001(title,0.89), conventional_material_001(title,0.89), material_term_001(title,0.89)
  +dense     131.0ms  animal_fibers_001(title,0.97), plant_fibers_001(title,0.95), material_term_001(title,0.94)
  +rerank   1918.9ms  cotton_001(dense,0.06), fiber_term_001(dense,0.00), nonfiber_term_001(title,0.00)
```
(`+rerank` first call includes ~3 s of one-off warm-up; steady state
≈1.8–1.9 s for 10 pairs on CPU.)

What it shows — five queries, no labels, so read as evidence for design
choices, not as a metric:

* **The reranker corrects a lexical over-trigger.**  "涤纶和聚酯纤维是一回事吗"
  contains the substring "纤维", so the lexical *contains-term* rule scores
  the generic entry `fiber_term_001` as an exact hit (1.00).  Dense keeps it
  on top (RRF rewards "found by both").  The cross-encoder reads the
  question and puts `synthetic_fibers_001` (polyester *is* a synthetic
  fibre) first and demotes the generic entry to 0.02.  Same story for
  "夏天穿什么面料凉快" → `cotton_001`.  The lexical contains-rule is a
  known weakness worth fixing on its own (a one-character-class term should
  not be an exact hit inside a longer question).
* **Only the reranker knows when there is no evidence.**  For the
  shrinkage question the KB has nothing; lexical and dense still rank
  *something* at 0.89 / 0.95 because their scores are relative.  The
  reranker gives 0.00 across the board.  That is why `reranker.min_score`
  exists: it is the only score in the pipeline with an absolute meaning,
  and gating on it feeds straight into the generation policy ("no evidence
  → template/refusal, never a model answer").  Observed range on this KB:
  on-topic 0.11–0.63, off-KB 0.00–0.06.  The default stays 0.0 (off)
  because five queries are not a calibration set.
* **Cost.**  Dense adds ~120–190 ms on CPU, the reranker ~1.8 s for 10
  pairs — not serving-grade on CPU.  On the RTX 4090 both are expected in
  the tens of ms (BGE-reranker-v2-m3 is a 568 M-param XLM-R; unmeasured
  there, so treat as expectation).  Caching the query embedding and
  reranking `top_n=5` instead of 10 are the obvious levers if it matters.
* **Dense-only candidates surface at 0.50** (`circularity_001`): rank 1
  in one list is exactly half the fused maximum by construction.

## 6. What is and is not verified

Verified: hybrid + rerank code paths with the real BGE models on CPU; all
routing, fusion, filtering, fallback and grounding logic with fakes; the
orchestrator's knowledge route is unchanged with flags off.

Not verified: end-to-end generation with the real Qwen-VL TensorRT-LLM
endpoint (the pod is stopped; the client contract is exercised by
`tests/test_mllm_client.py` against a fake HTTP layer and was exercised live
in §3.2 with an image).  No retrieval-quality metric (recall@k) — the KB has
15 documents and no labelled query set, so any number would be theatre; the
smoke test is a qualitative check, read it as such.  Chroma backend: not
implemented (config value is rejected with a warning, not silently
ignored).
