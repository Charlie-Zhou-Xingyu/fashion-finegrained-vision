# P1.0a — Serving Quality Eval Framework

> Status: v0 seed eval set + report-only runner (2026-07-15).
> This is NOT a formal product metric — it is the quality-eval foundation
> on top of the P0 deterministic serving skeleton.

## 1. Goal

P0 already has three protection layers:

- **pytest regression** — unit/integration correctness
- **golden contracts** (`tests/fixtures/serving_golden_cases.json`) — API schema stability
- **latency benchmark** (`scripts/bench_serving.py`) — performance baseline, report-only

None of those answer *quality* questions: is the intent label right, did RAG
hit the right evidence, does the QA answer contain the required information,
does content generation respect the safety policy?  P1.0a adds:

- an auditable **eval case schema**
- **6 seed datasets** (>= 130 cases/manifests)
- a **report-only eval runner** with task-level metrics and case-level failure reports
- a CI-log-friendly **summary script**

## 2. Layout

```text
eval/
  schemas/eval_case_schema.json          # case schema (documented; validated by a light Python validator)
  datasets/
    intent_eval_v0.jsonl                 # 34 cases  → POST /v1/intent/classify
    attribute_qa_eval_v0.jsonl           # 25 cases  → POST /v1/mm/qa
    rag_retrieval_eval_v0.jsonl          # 26 cases  → POST /v1/rag/retrieve
    mm_qa_eval_v0.jsonl                  # 20 cases  → POST /v1/mm/qa
    content_generation_eval_v0.jsonl     # 21 cases  → POST /v1/merchant/content/generate
    vision_attribute_eval_manifest_v0.jsonl  # 10 manifests → SKIPPED (manifest-only)
  scripts/
    run_serving_eval.py                  # runner (TestClient, report-only)
    summarize_eval_report.py             # summary printer (stdlib only)
  reports/                               # generated reports (gitignored artifacts)
```

## 3. Eval case schema

Every JSONL line is one JSON object:

```json
{
  "id": "intent_001",
  "task_type": "intent | attribute_qa | rag_retrieval | mm_qa | content_generation | vision_attribute",
  "input": {},
  "expected": {},
  "tags": ["attribute_query/fabric"],
  "difficulty": "easy | medium | hard",
  "review_status": "seed | manual_reviewed | needs_review",
  "source_ref": {"dataset": "p1_0a_seed_manual", "grounded_by": "..."},
  "notes": "why this case exists / known gaps"
}
```

Rules enforced by the validator (no `jsonschema` dependency):

- all 9 fields required; `input`/`expected`/`source_ref` are objects, `tags` a list, `notes` a string
- `task_type` must match the dataset file
- case ids globally unique across all datasets
- no `SECRET_BASE64_DO_NOT_LEAK`, no raw image bytes, no token/secret fields
- `vision_attribute` `image_uri` must use `placeholder://`

## 4. How to add a case

1. Pick the right dataset file by task type.
2. Ground the `expected` values against real behavior (run the endpoint via
   TestClient) — do NOT invent KB ids or intent labels.
3. If the case documents a known gap (e.g. missing taxonomy rule), still write
   the *desired* ground-truth label, set `review_status: needs_review`, and
   explain in `notes`.  Report-only mode makes failures informative, not fatal.
4. Run `python -m pytest tests/test_eval -q` — schema/count/uniqueness tests must pass.

## 5. Running

```bash
python eval/scripts/run_serving_eval.py --task all --output eval/reports/serving_eval_report.json
python eval/scripts/summarize_eval_report.py eval/reports/serving_eval_report.json
```

- `--task` selects one of: `all`, `intent`, `attribute_qa`, `rag_retrieval`,
  `mm_qa`, `content_generation`, `vision_attribute`.
- Every request carries `X-Request-ID: eval_<case_id>`; the runner asserts the echo.
- **Default is report-only** — exit code 0 regardless of pass rate.
- `--fail-on-threshold` exits non-zero if any non-skipped task drops below:

```python
DEFAULT_THRESHOLDS = {
    "intent": 0.80,
    "attribute_qa": 0.80,
    "rag_retrieval": 0.70,
    "mm_qa": 0.70,
    "content_generation": 0.80,
}
```

## 6. vision_attribute is manifest-only

The 10 `vision_attribute` cases are future-facing manifests.  The runner
**skips** them (no HTTP call, no image loading/downloading), they do not
count toward pass/fail or thresholds, and the summary lists them as skipped.

P1.1 added a `--enable-real-vision` flag: it is plumbed through the runner
but placeholder:// manifests are **still skipped** (only the recorded skip
reason changes) — real vision eval requires local fixture images plus a
wired real backend (P1.2+; see `docs/P1_real_vision_provider_adapter.md`).
Real visual eval starts only after a real `VisionAttributeProvider` backend
is connected.

## 7. Boundaries (P1.0a)

- No real vision pipeline, no `tools/infer/`, no `src/fashion_vision/` changes
- No image parsing or downloading
- No LLM / MLLM / external generation API
- No Redis / FAISS / BGE / reranker
- No production KB expansion; no changes to RagService / ContentGenerationService logic
- Eval set is v0 seed — small, auditable, NOT a formal product metric
- P1.0b will expand toward 500 intent / 200 RAG / 100 content samples

## 8. Relationship to other harnesses

| Harness | Question answered | Hard fail? |
|---|---|---|
| pytest | is the code correct? | yes |
| golden contracts | is the API contract stable? | yes |
| latency benchmark | is it fast enough? | report-only |
| **quality eval (this)** | are the answers right? | report-only (opt-in threshold) |
