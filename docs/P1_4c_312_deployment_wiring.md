# P1.4c — 3.1.2 Region Backend Deployment Wiring

> Date: 2026-07-21
> Status: Complete

---

## 1. Service Entrypoint

```
uvicorn inference.serving.app:app --reload
```

Framework: **FastAPI** (Uvicorn ASGI server)
Entry module: `inference/serving/app.py`
Config: `configs/serving_config.yaml` (loaded by `inference/serving/deps.py`)

---

## 2. Runtime Path for Region Queries

```
/v1/mm/qa
→ MultimodalQARequest parsing
→ QaOrchestrator.answer(query, image_bytes=...)
→ build_vision_context(vision_provider, image_bytes=...)
→ RuleIntentClassifier.classify(query)
→ region_* intent detected
→ _route_region_query(q, primary, ..., image_bytes=image_bytes)
→ if vc.localized_regions is empty:
    _try_region_backend(image_bytes, query, warnings, tools)
      → get_region_backend()        # singleton (config/env)
      → decode_image_bytes()        # base64/bytes → BGR numpy
      → extract_requested_region_part(query)  # Chinese → part_type
      → backend.locate_regions(image, query, requested_part)
        → FashionpediaRegionBackend
          → _ensure_loaded()        # lazy load YOLO model
          → _detect_part() per part
      → normalize_region_predictions()  # raw → LocalizedRegion dicts
→ intent-specific answer (location/existence/detail/count/attribute)
→ QAOrchestratorResult with safe localized_regions_summary + regions_used
```

---

## 3. Config / Env Behavior

### Default (safe, disabled):
```yaml
region_backend:
  backend: disabled
  enable_real: false
  model_path: models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt
  device: cpu
  confidence_threshold: 0.5
  timeout_ms: 5000
```

### Enable via environment:
```bash
export VISION_REGION_BACKEND=fashionpedia
export VISION_REGION_ENABLE_REAL=true
export VISION_REGION_DEVICE=cpu
export VISION_REGION_CONFIDENCE_THRESHOLD=0.5
```

### PowerShell:
```powershell
$env:VISION_REGION_BACKEND="fashionpedia"
$env:VISION_REGION_ENABLE_REAL="true"
$env:VISION_REGION_DEVICE="cpu"
$env:VISION_REGION_CONFIDENCE_THRESHOLD="0.5"
```

- Disabled → no model loaded, service starts instantly
- Enabled + model exists → lazy load on first region query
- Enabled + model missing → `region_backend_disabled` warning, safe fallback
- Service does not crash on backend failure

---

## 4. How to Verify

### Step 1: Start service with real backend

```powershell
conda activate fashion-demo2
$env:VISION_REGION_BACKEND="fashionpedia"
$env:VISION_REGION_ENABLE_REAL="true"
$env:VISION_REGION_DEVICE="cpu"
uvicorn inference.serving.app:app --host 127.0.0.1 --port 8000
```

### Step 2: Smoke test (backend-direct mode)

```powershell
conda activate fashion-demo2
$env:VISION_REGION_BACKEND="fashionpedia"
$env:VISION_REGION_ENABLE_REAL="true"
$env:VISION_REGION_DEVICE="cpu"
python scripts/smoke_test_312_region_backend.py `
  --image artifacts/p13_visual_qa/annotated/001640_annotated.jpg `
  --query "领口在哪里？" `
  --mode backend-direct
```

### Step 3: Smoke test (service-qa mode)

```powershell
python scripts/smoke_test_312_region_backend.py `
  --image artifacts/p13_visual_qa/annotated/001640_annotated.jpg `
  --query "领口在哪里？" `
  --mode service-qa `
  --url http://127.0.0.1:8000/v1/mm/qa
```

### Step 4: Curl test

```bash
curl -X POST http://127.0.0.1:8000/v1/mm/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "领口在哪里？", "image_bytes": "<base64>"}'
```

---

## 5. Model Dependency

| Item | Path |
|---|---|
| Fashionpedia YOLO | `models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt` |
| Status | EXISTS ✅ |

The model was trained in P7 (2026-07-02) and is available locally.

---

## 6. Safety Verification

Verified by automated tests that API responses never contain:
- mask_bitmap / mask_data
- crop_path / crop_data / crop_bytes
- temp_path / file_path / local_path
- tensor / checkpoint
- image_bytes (raw value)
- Traceback / RuntimeError / stack frames
- D:\\ / /tmp/ / \\temp paths

Only safe keys in `localized_regions_summary`:
`region_id, part_type, part_group, bbox, confidence, source, backend`

---

## 7. Sample Request/Response

### Request:
```json
{
  "query": "领口在哪里？",
  "image_bytes": "<base64-encoded-jpeg>"
}
```

### Response:
```json
{
  "request_id": "req_abc123",
  "status": "success",
  "data": {
    "answer": "检测到neckline区域，位置约为 bbox=[115, 162, 348, 281]，置信度 0.74。",
    "answer_type": "region_query_answer",
    "sources": [
      {
        "type": "localized_region",
        "id": "region_0",
        "label": "neckline",
        "confidence": 0.74
      }
    ],
    "meta": {
      "route": "region_query",
      "primary_intent": "region_location_query",
      "localized_regions_summary": [
        {
          "region_id": "region_0",
          "part_type": "neckline",
          "part_group": "collar_area",
          "bbox": [114.5, 161.8, 348.2, 281.4],
          "confidence": 0.74,
          "source": "fashion_vision_3_1_2",
          "backend": "fashionpedia_yolo"
        }
      ],
      "regions_used": ["region_0"]
    }
  }
}
```

---

## 8. Test Results

```
test_region_backend.py ........ 36 passed  (backend, normalization, integration, error handling, no-leak)
test_region_qa.py ............. 57 passed  (schema, mapper, intents, QA behaviors, no-leak, regression)
test_visual_instance_qa.py ... 25 passed  (P1.3 garment instance QA)
test_qa_orchestrator.py ...... 22 passed  (core orchestrator)
test_intent_classifier.py .... 32 passed  (intent classification)
test_schemas.py .............. 24 passed  (schema validation)
Total: 196 passed, 0 failed
```

---

## 9. Known Limitations

1. **Fashionpedia YOLO only** — full `locate_region()` (landmark + DINO) not wired
2. **No instance-region association** — regions are image-level
3. **No 3.1.3 attributes** — `region_attribute_query` returns placeholder
4. **Full-image inference** — YOLO runs on full image, not garment-cropped
5. **First call latency** — ~3.5s on CPU (includes YOLO model loading); subsequent ~0.2s
6. **CPU only by default** — GPU requires changing device config

---

## 10. Next Steps

1. Wire full `locate_region()` when garment instances available
2. Add `instance_id` to `LocalizedRegion` for instance-region association
3. Connect 3.1.3 attribute extraction for `region_attribute_query`
4. Add real evaluation cases with annotated images
