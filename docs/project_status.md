# Fashion Fine-Grained Vision — Project Status

> Last updated: 2026-07-15 | Phase: P0 + P1.0a/P1.0a.1 complete + P1.1 experimental vision adapter

## Quick Links

- [P0 Progress Summary & P1 Readiness](docs/P0_progress_summary_and_P1_readiness.md)
- [Vision Adapter Contract](docs/P0a_vision_adapter_contract.md)
- [Real Vision Provider Adapter (P1.1)](docs/P1_real_vision_provider_adapter.md)
- [Vision Context Integration](docs/P0a_vision_context_integration.md)
- [Eval & Latency Harness](docs/P0a_eval_and_latency_harness.md)
- [Quality Eval Framework (P1.0a)](../eval/README.md)
- [Architecture & Integration](docs/P0a_project_architecture_and_integration.md)
- [CLAUDE.md](../CLAUDE.md) — project rules

## Current State

- **306+37 serving tests** + **57 eval-framework tests** pass (400 total), **17 golden contracts**, **11 benchmark cases**; full repo: 874 collected → 872 passed, 2 env skips
- **P1.1 (2026-07-15)**: `RealVisionAttributeProvider` experimental adapter SHELL
  (`inference/serving/real_vision_provider.py`) — feature-flagged
  (`vision.provider=real` + `real_enabled=true` or `VISION_PROVIDER`/`VISION_REAL_ENABLED`),
  **default remains mock**; full contract (validation/timeout/error-mapping/
  normalization/no-leak) tested with fake backends; `FashionVision31Backend` is
  probing-only — real 3.1 invocation is a P1.2 item (8 attribute checkpoints
  missing + latency budget未批).  No image_url download.
- **P1.2 (2026-07-16)**: REAL 3.1.1 instance segmentation wiring —
  `FashionVision31SegmentationBackend` calls the ACTUAL
  `GarmentPipeline.run_image()` (YOLO detection + SAM-HQ segmentation ONLY;
  `run_landmark_and_crops=False`).  temp-file pipeline, per-request cleanup,
  mask_present/mask_ref placeholders (never bitmap), dual-label output.
  Default mock + CPU devices unchanged.  3.1.3 attribute classifiers NOT run.
  Sever: `backend: fashion_vision_3_1_segmentation` (or
  `fashion_vision_3_1` + `mode: segmentation_only`).
- **Full-repo test hygiene (P1.0a.1, 2026-07-15)**: `python -m pytest tests/ -q` is
  green on a plain dev machine **without any local model checkpoint**
  (837 collected → 835 passed, 2 environment skips).  The one previously failing
  test (`test_predict_from_json_missing_image_path_does_not_crash`) tested
  grouping logic but implicitly lazy-loaded a real checkpoint; it now patches
  `_get_tasks_for_class` like its sibling tests — no coverage lost, no skip
  needed, no new env var/marker.  Serving/eval tests always run.
- **P1.0a quality eval framework**: 6 seed datasets (136 cases/manifests), report-only
  runner (`eval/scripts/run_serving_eval.py`), summary script, per-task thresholds
  (opt-in `--fail-on-threshold`), `vision_attribute` manifest-only (skipped)
- First eval run (2026-07-15): pass_rate 0.976 (123/126 non-skipped); 3 known-gap
  intent failures documented in dataset notes (style / fit_or_silhouette / synonym comparison)
- `/v1/mm/qa` — deterministic QA dispatch (IntentClassifier → AttributeService / RagService)
- `/v1/rag/retrieve` — exact / alias / BM25-like knowledge retrieval
- `/v1/merchant/content/generate` — deterministic template content generation
- `/v1/intent/classify` — rule-based 17-class intent classifier
- Vision adapter contract stable, mock provider only, no real visual pipeline
- No LLM, MLLM, Redis, FAISS, BGE, or external API dependencies
- Request tracing middleware, structured error handling, content safety policy

## What's NOT Yet Done

- Real visual pipeline integration (MockVisionAttributeProvider only)
- Real-image vision eval (`vision_attribute` dataset is manifest-only, runner skips it)
- LLM/MLLM-based answer generation
- Large-scale knowledge base (15 seed entries only)
- Vector search / reranker
- Multi-turn dialogue
- Production deployment configuration
- Large-scale eval sets (P1.0b: 500 intent / 200 RAG / 100 content samples)
