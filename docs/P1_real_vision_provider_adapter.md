# P1.1 — RealVisionAttributeProvider Experimental Adapter

> Status: experimental adapter SHELL (2026-07-15).
> Default provider is **still MockVisionAttributeProvider**.
> This does NOT represent production vision capability.

## 1. What P1.1 delivers

- `inference/serving/real_vision_provider.py` — `RealVisionAttributeProvider`
  implementing the exact existing serving contract
  (`VisionAttributeProvider.extract()` → `VisionAttributeResult`), plus:
  - injectable `backend_client` test seam
  - input validation (no-image / url-download-disabled / size guard)
  - timeout wrapper (ThreadPoolExecutor, Windows-safe)
  - full error mapping to structured warnings (never crashes `/v1/mm/qa`)
  - `normalize_vision_backend_output()` + `ATTRIBUTE_KEY_MAP`
  - no-leak guarantees (image_bytes never in result/log/exception text)
- Feature-flagged selection in `vision_provider.get_vision_provider()`
  (config + env vars), default mock, `fail_open_to_mock`.
- `FashionVision31Backend` — **probing shell only**: checks module
  availability (`find_spec`, nothing imported) and checkpoint file existence
  (filenames only, nothing loaded), then raises a structured
  `VisionProviderUnavailable`.  Real `GarmentPipeline.run_image` wiring is a
  **P1.2 item**, blocked on:
  - the 8 missing `outputs/p2_*/best.pt` attribute checkpoints
  - an approved per-request latency budget for YOLO+SAM-HQ stages
  - a temp-file invocation design (the 3.1 pipeline is file-based)

## 2. Feature flag

Default (`configs/serving_config.yaml`):

```yaml
vision:
  provider: mock            # mock | real
  real_enabled: false
  real_provider:
    backend: fashion_vision_3_1
    checkpoint_root: outputs/
    timeout_ms: 1500
    max_image_bytes: 5242880
    allow_image_url_download: false
    fail_open_to_mock: true
```

Env overrides (checked at provider selection time, not import time):

```text
VISION_PROVIDER=mock|real
VISION_REAL_ENABLED=true|false
VISION_TIMEOUT_MS=1500
VISION_CHECKPOINT_ROOT=outputs/
```

Real provider activates only when BOTH `provider=real` AND `real_enabled=true`.
`provider=real` without `real_enabled` logs `vision_provider_real_disabled`
and stays mock.  Any setup failure with `fail_open_to_mock=true` falls back
to mock.  Config parse failure always falls back to mock.

Note: `get_qa_orchestrator()` freezes the provider at first request — flip
flags before process start (or reset `_provider`/`_orchestrator` in tests
via monkeypatch).

## 3. Behavior matrix (RealVisionAttributeProvider.extract)

- request `provided_attributes` present → empty result, backend NOT called
  (request attrs are authoritative; same as mock)
- no image at all → `vision_input_missing` (info)
- `image_url` only → `vision_image_url_download_disabled` (info); **P1.1
  never downloads image_url**, regardless of config
- `image_bytes` > `max_image_bytes` → `vision_input_too_large` (warn)
- backend timeout → `vision_timeout` (warn); worker thread abandoned
  (`shutdown(wait=False)`) — cannot be force-killed, documented limitation
- backend `VisionProviderUnavailable` → `vision_provider_unavailable` (warn)
  + JSON-safe `meta.error_details`
- any other backend exception → `vision_provider_error` (warn); exception
  text is NOT propagated (could embed input payloads)
- success → normalized attributes; empty output → `vision_output_empty`
  (info); non-dict output → `vision_output_schema_mismatch` (warn)

Priority remains: **request attributes > visual attributes > unavailable**
(enforced by `vision_context.build_vision_context`, unchanged).

## 4. Attribute mapping

3.1 field/task names → serving keys (`ATTRIBUTE_KEY_MAP`):
`neckline_design/neck_design → neckline`, `lapel_design → collar_design`,
`class_name/coarse_class_name → garment_category`, native keys
(`color/fabric/style/fit_or_silhouette/length/sleeve_length/pattern/
collar_design/coat_length/dress_length/skirt_length/pant_length`) pass
through.  Unknown keys → `meta.unmapped_attribute_keys` (never dropped
silently).  Attribute structure:

```json
{"value": "white", "attribute_confidence": 0.82,
 "source": "vision_provider_real", "provider": "fashion_vision_3_1"}
```

Missing confidence stays `null` — never fabricated.  bboxes pass through
JSON-safe; mask bitmaps are replaced by `mask_present: true` (never full
arrays, never image data).

## 5. Response meta (via `/v1/mm/qa`)

`qa_orchestrator._build_result_meta` now whitelists these provider meta keys
into the response meta when present (backward compatible): `vision_backend`,
`vision_latency_ms`, `unmapped_attribute_keys`,
`vision_provider_real_enabled`, `error_code`.  Existing keys
(`vision_provider_used`, `vision_provider_name`, `visual_attributes_present`,
`provided_attributes_used`, …) unchanged.

## 6. Warning codes added in P1.1

`vision_provider_unavailable`, `vision_timeout`, `vision_input_too_large`,
`vision_output_empty`, `vision_output_schema_mismatch`,
`vision_image_url_download_disabled` (all scope `vision`);
`vision_provider_real_disabled` is logged at selection time (mock is
returned, which emits its own `vision_provider_mock`).  Warnings remain
top-level only in `UnifiedResponse`.

## 7. Eval

- `vision_attribute` eval manifests remain **skipped by default**.
- `eval/scripts/run_serving_eval.py --enable-real-vision` is plumbed but
  placeholder:// manifests are STILL skipped (recorded skip reason changes) —
  real vision eval requires local fixture images + a wired backend (P1.2+).
- Eval reports never contain image bytes.

## 8. Testing

`tests/test_serving/test_real_vision_provider.py` (37 tests) covers the full
contract with fake backends — CI needs no GPU, no checkpoints, no images.
Optional manual probing check:

```bash
python -c "from inference.serving.real_vision_provider import FashionVision31Backend; print(FashionVision31Backend().probe())"
```

## 9. P1.2 — Real 3.1.1 Segmentation Wiring (2026-07-16)

See `docs/P1_2_real_311_segmentation_wiring.md` for full details.
Real YOLO detection + SAM-HQ segmentation stages ARE connected via
`FashionVision31SegmentationBackend`.  The backend calls
`GarmentPipeline.run_image()` with `run_landmark_and_crops=False` so ONLY
stages 1+2 execute (3.1.3 attribute classifiers are NEVER run).  Default
remains mock + CPU device.  Dual-label output.

## 10. Boundaries

No image download, no network, no LLM/MLLM, no Redis/FAISS/BGE, no KB
change, no RagService/ContentGenerationService change, no UnifiedResponse
change, no model loading at import, no checkpoint requirement for tests,
`src/fashion_vision/` and `tools/infer/` untouched.
