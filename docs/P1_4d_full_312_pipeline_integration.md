# P1.4d — Full 3.1.2 Pipeline Functional Integration and Visualization

> Date: 2026-07-21 | Status: Complete

---

## 1. Purpose

Connect the full existing 3.1.2 pipeline (Fashionpedia YOLO + Grounding DINO + routing/fallback/shape-priors) into `/v1/mm/qa` and provide offline visualization.

---

## 2. Backend Comparison

| Backend | What it does | Parts covered |
|---|---|---|
| `disabled` | No real detection (default) | None |
| `mock` | P1.4a mock regions for testing | Test-only |
| `fashionpedia` (P1.4c) | Fashionpedia YOLOv8s only | 14 FP parts |
| `full312` (P1.4d) | Fashionpedia YOLO + Grounding DINO + routing | FP + DINO parts (excl. fast-path) |

---

## 3. Existing 3.1.2 Entry Point

**Function**: `locate_region()` in `src/fashion_vision/localization/region_localization_router.py`

**Features used by Full312RegionBackend**:
- Fashionpedia YOLO priority path (with DINO fallback for neckline/cuff)
- Grounding DINO for non-Fashionpedia parts
- Per-part optimized prompts from `part_detection_config.py`
- Per-part box/text thresholds
- Shape prior filtering via `filter_by_shape_priors()`
- Anatomical zoom (DINO path)
- Class-aware multi-instance IoU NMS

**Intentionally NOT modified**: All 3.1.2 algorithm/model implementation code is unchanged.

---

## 4. Data Flow (full312)

```
/v1/mm/qa
  -> QaOrchestrator.answer()
  -> region_* intent classification
  -> _try_region_backend(query_all_parts=...)
  -> get_region_backend() [full312]
  -> Full312RegionBackend.locate_regions(image, requested_part)
     -> _make_full_image_instance(image)
     -> for each part:
        -> locate_region(query=part_prompt, instance=synthetic, ...)
           -> Fashionpedia YOLO (if in FP coverage)
           -> DINO fallback (neckline/cuff only)
           -> DINO open-vocab (non-FP parts)
        -> collect bbox + score + backend
     -> normalize_region_predictions()
  -> localized_regions
  -> QA answer
```

---

## 5. Config Flags

```yaml
region_backend:
  backend: disabled       # disabled | fashionpedia | full312
  enable_real: false
  device: cpu
```

Env:
```bash
VISION_REGION_BACKEND=full312
VISION_REGION_ENABLE_REAL=true
VISION_REGION_DEVICE=cpu   # or cuda
```

Default: `disabled`, `enable_real=false` — safe, no models loaded.

---

## 6. Supported Parts

**Full312 covers**: neckline, collar, lapel, pocket, zipper, sleeve, cuff, button, buckle, bow, sequin, hood, epaulette, fringe, ruffle, bead, applique, flower, ribbon, rivet, tassel, strap (22 parts)

**NOT covered** (fast-path, requires real garment instance): hem, waist, shoulder, leg_opening

---

## 7. How to Run

### Smoke test (full312, CPU):
```powershell
conda activate fashion-demo2
python scripts/smoke_test_312_region_backend.py `
  --image test.jpg --query "领口在哪里？" `
  --backend full312 --device cpu --mode backend-direct
```

### Smoke test (full312, CUDA):
```powershell
$env:VISION_REGION_DEVICE="cuda"
python scripts/smoke_test_312_region_backend.py `
  --image test.jpg --query "有没有口袋？" `
  --backend full312 --device cuda --mode backend-direct
```

### Visualization (dry-run, no models):
```powershell
python scripts/visualize_312_region_results.py `
  --dry-run --output-dir outputs/vis_312_test
```

### Visualization (full312, real):
```powershell
python scripts/visualize_312_region_results.py `
  --images-dir artifacts/p13_visual_qa/annotated `
  --output-dir outputs/vis_312_regions `
  --backend full312 --device cuda --samples-per-part 5
```

### Service:
```powershell
$env:VISION_REGION_BACKEND="full312"
$env:VISION_REGION_ENABLE_REAL="true"
$env:VISION_REGION_DEVICE="cuda"
uvicorn inference.serving.app:app --host 127.0.0.1 --port 8000
```

---

## 8. Output Examples

### /v1/mm/qa response (neckline):
```json
{
  "answer": "检测到neckline区域，位置约为 bbox=[115, 162, 348, 281]，置信度 0.74。",
  "answer_type": "region_query_answer",
  "meta": {
    "localized_regions_summary": [{
      "region_id": "region_0",
      "part_type": "neckline",
      "part_group": "collar_area",
      "bbox": [114.5, 161.8, 348.2, 281.4],
      "confidence": 0.74,
      "source": "fashion_vision_3_1_2",
      "backend": "fashionpedia_yolo"
    }]
  }
}
```

### Visualization output:
```
outputs/vis_312_regions/
  index.html          — HTML gallery
  summary.json        — per-part stats
  images/
    neckline_000.jpg
    pocket_000.jpg
    ...
```

---

## 9. Safety / No-Leak

Verified by automated tests that API responses never contain:
- mask_bitmap, crop_path, image_bytes, temp_path, local file paths
- raw tensors, checkpoint paths, stack traces

---

## 10. Test Results

```
test_region_backend.py ... 42 passed (incl. 7 Full312 tests)
test_region_qa.py ........ 57 passed
test_visual_instance_qa.py 25 passed
test_qa_orchestrator.py .. 22 passed
test_intent_classifier.py  32 passed
test_schemas.py .......... 24 passed
Total: 202 passed, 0 failed, 2 skipped
```

---

## 11. Current Limitations

- Fast-path parts (hem, waist, shoulder, leg_opening) require real garment instances
- DINO inference on CPU is slow (~20s per part)
- Full-image inference without mask gating (no 3.1.1 prerequisite)
- No instance-region association
- No 3.1.3 attribute extraction
- Accuracy/latency not evaluated

---

## 12. Next Steps

1. Instance-region association (P1.4c follow-up)
2. 3.1.3 attribute extraction
3. Formal accuracy/IoU evaluation
4. Performance optimization (GPU batching, etc.)
