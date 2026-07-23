# P0 Progress Summary & P1 Readiness

> Date: 2026-07-14 | Authors: Charlie Zhou + Claude Code

---

## 1. Executive Summary

P0 (P0a.1 ~ P0a.9 + P0b.1) is complete.  The system has:

- A deterministic serving skeleton with 6 endpoints
- Rule-based intent, template attribute QA, BM25-like knowledge retrieval
- Mock vision adapter contract ready for real pipeline swap-in
- Eval harness with 17 golden contracts and 11 benchmark cases
- Error handling, request tracing, and content safety policy
- Content generation deterministic skeleton (title / selling_points / description / bullets)

**Not yet connected**: real vision pipeline, LLM/MLLM, vector retrieval, large-scale KB, multi-turn dialogue.

**Current**: 290 tests, all pass. Benchmark: all 11 cases within budget on RTX 3090 dev machine.

---

## 2. P0 Phase Checklist

### P0a.1 — FastAPI + UnifiedResponse
- **Done**: `app.py`, `schemas.py`, `deps.py`, 6 endpoints, `WarningItem`, `SourceItem`, `disambiguated confidence`
- **Tests**: 43 (schemas + app basic)
- **Boundary**: mock endpoints, warnings top-level only

### P0a.2 — IntentClassifier
- **Done**: `intent_classifier.py`, `intent_taxonomy.yaml`, 17-class rule matching
- **Tests**: 31
- **Boundary**: rule only, no embedding/ONNX/LLM

### P0a.3 — AttributeService
- **Done**: `attribute_service.py`, `attribute_templates.yaml`, fabric sanitise, source routing, confidence policy
- **Tests**: 49
- **Boundary**: template only, no real vision attributes

### P0a.4 — RagService + KB
- **Done**: `rag_service.py`, `knowledge_base.yaml` (15 seeds), `retrieval_config.yaml`, exact/alias/BM25-like
- **Tests**: 60 (service + schema)
- **Boundary**: 15 seed entries, no FAISS/BGE/reranker, recall@5 not formally evaluated

### P0a.5 — QaOrchestrator
- **Done**: `qa_orchestrator.py`, `/v1/mm/qa` real dispatch
- **Tests**: 22
- **Boundary**: deterministic templates, no LLM, styling limited

### P0a.6 — VisionAttributeProvider Adapter
- **Done**: `vision_provider.py`, mock provider, contract doc
- **Tests**: 19
- **Boundary**: mock only, no real image processing

### P0a.6.1 — Vision Adapter Hardening
- **Done**: spy tests, image_bytes leak tests, no-mutation tests, regions semantics
- **Tests**: 247 → 247
- **Boundary**: contract frozen, spy/fake provider tests added

### P0a.7 — VisionContext Integration
- **Done**: `vision_context.py`, `build_vision_context`, effective_attributes merge
- **Tests**: 12 new
- **Boundary**: request > visual > unavailable priority fixed, no fake production attributes

### P0a.8 — Eval & Latency Harness
- **Done**: `serving_golden_cases.json` (17 cases), `test_golden_contracts.py`, `scripts/bench_serving.py`
- **Tests**: 271 total
- **Boundary**: benchmark report-only (not hard fail), no snapshot assertions

### P0a.9 — Error Handling, Observability, Request Tracing
- **Done**: `errors.py` (ServingError taxonomy), `logging_utils.py` (redaction), middleware (request_id + process_time + counters), `/v1/metrics` JSON
- **Boundary**: no external Prometheus dependency, in-process counters only

### P0b.1 — ContentGenerationService Skeleton
- **Done**: `content_policy.py`, `content_generation_service.py`, `/v1/merchant/content/generate` real endpoint
- **Tests**: 14 new
- **Boundary**: deterministic templates only, no LLM, blocked claims redacted (field/reason only), `llm_used=false`

---

## 3. Current Endpoint Matrix

| Endpoint | Input | Output | Deterministic | Real Model | Golden | Benchmark |
|---|---|---|---|---|---|---|
| GET /v1/health | — | service status, modules | ✅ | — | 0 | ✅ |
| GET /v1/metrics | — | in-process counters | ✅ | — | 0 | ✅ |
| POST /v1/intent/classify | query | primary_intent, sub_intent, confidence | ✅ | rule | 2 | ✅ |
| POST /v1/rag/retrieve | query, categories, top_k | hits with source_ref/review_status | ✅ | BM25 | 2 | ✅ |
| POST /v1/mm/qa | query, image_url, attributes, regions | answer, answer_type, sources | ✅ | MockVision | 7 | ✅ |
| POST /v1/merchant/content/generate | content_type, attributes | generated_content, blocked_claims | ✅ | template | 5 | ✅ |

---

## 4. Test & Eval Status

| Metric | Value |
|---|---|
| pytest total | 290 |
| golden cases | 17 |
| benchmark cases | 11 |
| benchmark mode | report-only (not hard fail) |
| benchmark result (RTX 3090) | all within budget |

---

## 5. Current Engineering Boundaries

- No real vision pipeline
- No LLM / MLLM
- No external generation API
- No Redis / FAISS / BGE
- No large-scale KB (15 seed entries)
- No real image parsing
- MockVisionAttributeProvider does NOT fabricate attributes / bbox / mask
- ContentGenerationService is deterministic template skeleton
- Content policy blocks 20+ tokens; blocked claims redacted to field/reason

---

## 6. P1 Readiness Evaluation

| Module | Status | Rationale |
|---|---|---|
| Serving Foundation | **Ready** | UnifiedResponse, request_id, errors, metrics, golden/benchmark, tests sufficient |
| IntentClassifier | **Partially Ready** | Stable rules but no formal eval set; needs 500-label eval |
| AttributeService | **Partially Ready** | Deterministic; needs more attribute schemas and real visual input samples |
| RagService | **Partially Ready** | BM25-like available but KB only 15 entries; needs 500+ KB + eval set + vector retrieval |
| QaOrchestrator | **Ready for P1 experiments** | Routes all intents; vision context integrated; needs LLM answer generator |
| Vision Provider / VisionContext | **Contract Ready, Model Not Ready** | Adapter contract frozen; real pipeline not connected |
| ContentGenerationService | **P0b skeleton ready** | Templates done, policy skeleton done; needs LLM for brand voice, A/B eval |

### P1 Entry Criteria

**Met**: stable endpoints, unified response, warnings/meta/sources, request tracing, golden contracts, benchmark baseline, mock vision contract, content safety policy.

**Not Met**: real vision pipeline, real image evaluation, large-scale KB, RAG eval set, LLM/MLLM generation evaluation, production deployment config, stress/throughput benchmark.

---

## 7. Recommended P1 Roadmap

| Phase | Goal | Deliverable |
|---|---|---|
| P1.0 | Eval set expansion | 500 intent labels, 200 RAG query-doc pairs, 100 content generation samples |
| P1.1 | RealVisionAttributeProvider experimental adapter | Connects existing 3.1 pipeline behind VisionAttributeProvider interface |
| P1.2 | KB expansion to 500+ | Add 500+ reviewed entries under semi-automated workflow |
| P1.3 | RAG quality eval + retrieval improvements | Vector search (FAISS/BGE), reranker, recall@5/MRR/NDCG |
| P1.4 | Content safety policy hardening | Extended policy rules, A/B evaluation framework, human review workflow |
| P1.5 | LLM AnswerGenerator (feature flag) | Optional LLM-based answer behind timeout/fallback/circuit_breaker |
| P2 | Production deployment, throughput benchmark | Docker, stress test, 60 QPS validation |

---

## 8. P1.0a — Eval Schema, Runner & Seed Eval Set (2026-07-15)

**Done** (see `eval/README.md` for full docs):

- `eval/schemas/eval_case_schema.json` + lightweight Python validator
  (id / task_type / input / expected / tags / difficulty / review_status /
  source_ref / notes; global id uniqueness; placeholder-only vision URIs;
  no-secret rules)
- 6 seed datasets, **136 cases/manifests total**:
  intent 34, attribute_qa 25, rag_retrieval 26, mm_qa 20,
  content_generation 21, vision_attribute 10 (manifest-only)
- `eval/scripts/run_serving_eval.py` — TestClient-based, report-only by
  default, `--fail-on-threshold` opt-in, per-request `X-Request-ID=eval_<id>`,
  20+ declarative checkers, image_bytes redaction in reports
- `eval/scripts/summarize_eval_report.py` — stdlib-only CI-log summary
- `tests/test_eval/` — 57 tests (schema/datasets, runner/checkers, summary)

**First run (2026-07-15, [measured])**: 136 total → 123 passed / 3 failed /
10 skipped; pass_rate 0.976; all task thresholds passed.
The 3 failures are intentional known-gap intent cases
(`attribute_query/style`, `fit_or_silhouette`, synonym comparison) kept as
`needs_review` to drive taxonomy expansion.

**Boundary**: no real vision, no image parsing/downloading, no LLM, no
FAISS/BGE/Redis, no KB expansion, no changes to serving business logic,
`UnifiedResponse` contract untouched, eval + benchmark both report-only.
Eval set is v0 seed — not a formal product metric.  P1.0b expands toward
500 intent / 200 RAG / 100 content samples.

---

## 9. P1.1 — RealVisionAttributeProvider Experimental Adapter (2026-07-15)

**Done** (see `docs/P1_real_vision_provider_adapter.md`):

- `inference/serving/real_vision_provider.py`: provider implementing the
  existing `extract()` contract; injectable backend seam; timeout wrapper;
  error mapping (`vision_provider_unavailable/error`, `vision_timeout`,
  `vision_input_too_large`, `vision_output_empty/schema_mismatch`,
  `vision_image_url_download_disabled`); output normalization +
  `ATTRIBUTE_KEY_MAP` (unmapped keys → meta); no-leak guarantees.
- Feature-flagged selection in `get_vision_provider()` — config
  (`vision:` section in serving_config.yaml) + env vars
  (`VISION_PROVIDER`, `VISION_REAL_ENABLED`, `VISION_TIMEOUT_MS`,
  `VISION_CHECKPOINT_ROOT`); **default mock**, `fail_open_to_mock`.
- `qa_orchestrator._build_result_meta`: whitelist passthrough of provider
  meta (vision_backend / vision_latency_ms / unmapped_attribute_keys / …).
- Eval runner `--enable-real-vision` plumbed; placeholder manifests still
  skipped (reason recorded).
- 37 new tests (`tests/test_serving/test_real_vision_provider.py`), all
  fake-backend based — no GPU/checkpoint/CI requirement.

**Deliberately NOT done**: real `GarmentPipeline.run_image` invocation —
`FashionVision31Backend` is probing-only.  Blocked → resolved in P1.2.

### P1.2 — Real 3.1.1 Instance Segmentation Wiring (2026-07-16)

**Done** (see `docs/P1_2_real_311_segmentation_wiring.md`):

- `FashionVision31SegmentationBackend` — REAL YOLO detection + SAM-HQ
  segmentation via `GarmentPipeline.run_image()` with
  `run_landmark_and_crops=False`.  temp-file pipeline, per-request cleanup,
  `mask_present`/`mask_ref` placeholders (never bitmap), dual-label output
  (coarse + fine class).  CPU by default, config/env overridable
  (`VISION_DEVICE=cpu|cuda|auto`).
- 11 default-safe tests + 2 optional real tests
  (`RUN_REAL_VISION_TESTS=1`) in
  `tests/test_serving/test_real_vision_segmentation.py`.
- Provider `meta`/`sources` passthrough; qa_orchestrator whitelist extended
  with `vision_backend_mode` / `num_garment_instances` /
  `mask_bitmap_returned`.
- **3.1.3 attribute classifiers NOT run** (the segmentation backend never
  touches them — the 8 missing `p2_*` checkpoints are irrelevant).
- `FashionVision31Backend` (P1.1 probing shell) preserved — real invocation
  is a different backend name / mode.

**Verification [measured]**: full repo 885 collected → 0 failed (all real-vision
tests gated); serving+eval 411 passed; eval threshold_passed=True (3 known
intent gaps unchanged); benchmark 11/11 within budget.

---

*End of document.*
