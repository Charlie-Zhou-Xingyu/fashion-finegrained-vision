# P1.4b — Real 3.1.2 Region Backend Wiring

> Date: 2026-07-21
> Status: Complete

---

## 1. Why Real Backend Wiring

P1.4a added the `localized_regions` schema, intent classification, and QA routing — but used mock data only. P1.4b connects the real 3.1.2 models so `/v1/mm/qa` can answer region questions from actual visual evidence.

---

## 2. Data Flow

```
user image + user question
        ↓
intent classifier (RuleIntentClassifier)
        ↓
region_* intent detected
        ↓
requested_part extraction (region_query_mapper)
        ↓
P1.4b: RegionLocalizationBackend.locate_regions()
        ↓
FashionpediaPartDetector.detect() per part
        ↓
normalize_region_predictions() → LocalizedRegion dicts
        ↓
QA route uses localized_regions to answer
```

---

## 3. Backend Audit Result

| Item | Finding |
|---|---|
| Entry point | `locate_region()` in `region_localization_router.py` |
| Requirements | Garment instance + mask paths + SAM + DINO + Fashionpedia YOLO |
| Instance-level | Yes — needs 3.1.1 output first |
| Self-contained sub-component | `FashionpediaPartDetector` — only needs .pt file + numpy image |
| Output format | Dict with bbox_xyxy, score, label, mask (numpy array), debug |
| Mask risk | numpy arrays + temp file paths in raw output |
| Serving readiness | Raw output must be normalized; masks/paths must be stripped |

**Decision**: Wire `FashionpediaPartDetector` as the first real backend (self-contained, no instance dependency). Full `locate_region()` requires garment instances + SAM + DINO — skeleton provided.

---

## 4. Adapter Interface

```python
class RegionLocalizationBackend(ABC):
    def locate_regions(self, image, query=None, requested_part=None) -> list[dict]
    @property
    def backend_name(self) -> str
    @property
    def enabled(self) -> bool
```

Implementations:

| Class | Status | Description |
|---|---|---|
| `DisabledRegionLocalizationBackend` | **Default** | Always returns empty |
| `FashionpediaRegionBackend` | **Wired** | Wraps `FashionpediaPartDetector` |
| `FullRegionLocalizationBackend` | Skeleton | Wraps `locate_region()` — needs SAM+DINO+instances |

---

## 5. Config Flags

In `configs/serving_config.yaml`:

```yaml
region_backend:
  backend: disabled               # disabled | fashionpedia | full
  enable_real: false
  model_path: models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt
  device: cpu
  confidence_threshold: 0.5
  timeout_ms: 5000
```

Env overrides:
- `VISION_REGION_BACKEND` — backend name
- `VISION_REGION_ENABLE_REAL` — `true`/`1`/`yes` to enable
- `VISION_REGION_DEVICE` — `cpu` | `cuda`
- `VISION_REGION_CONFIDENCE_THRESHOLD` — float threshold

Default: `disabled`, `enable_real=false` — safe, no model loading.

---

## 6. Raw Output Normalization

`normalize_region_predictions()` converts raw Fashionpedia/DINO detections:

```python
# Raw (from FashionpediaPartDetector):
{"bbox_xyxy": [100, 40, 220, 110], "score": 0.82, "label": "neckline", "class_id": 6}

# Normalized (LocalizedRegion-compatible):
{
    "region_id": "region_0",
    "part_type": "neckline",
    "part_group": "collar_area",
    "bbox": [100.0, 40.0, 220.0, 110.0],
    "confidence": 0.82,
    "source": "fashion_vision_3_1_2",
    "backend": "fashionpedia_yolo",
    "mask_present": false,
    "mask_ref": null,
}
```

Rules:
- Invalid bboxes (x2<=x1, non-finite) silently dropped
- Unknown labels → `"unknown"`
- Missing confidence → `None`
- `mask_present` always `False`, `mask_ref` always `None`
- Raw mask/crop/tensor/path fields never pass through

---

## 7. Supported Part Labels

Fashionpedia YOLO covers: hood, collar, lapel, epaulette, sleeve, pocket, neckline, buckle, zipper, bow, fringe, ruffle, sequin, applique, bead, flower, ribbon, rivet, tassel.

Part label mapping in `region_backend.py` (`_FP_LABEL_TO_PART_TYPE`):
- Raw label → canonical `part_type`
- `part_type` → `part_group` via `_PART_TYPE_TO_GROUP`

Parts NOT in Fashionpedia (hem, waist, shoulder, leg_opening, button, pattern, etc.) require the full `locate_region()` pipeline — returned as empty by Fashionpedia backend.

---

## 8. Error Handling

| Condition | Warning Code | Behavior |
|---|---|---|
| Backend disabled | `region_backend_disabled` | Empty regions, info warning |
| Image decode failure | `region_backend_error` | Empty regions, warn warning |
| Backend exception | `region_backend_error` | Empty regions, warn warning |
| No detections | `region_backend_empty` | Empty regions, info warning |

All failures are graceful — empty `localized_regions`, structured warning, no crash.

---

## 9. Safety / No-Leak Policy

The following are verified by tests to never appear in API responses:
- Raw JPEG/PNG bytes
- Mask bitmaps
- Crop images
- Temp file paths (`/tmp/`, `D:\\`, `\\temp`)
- Local file paths
- Raw tensors
- Checkpoint paths
- `mask_ref` (actual paths)
- `mask_present` (in summaries)

---

## 10. How to Run

### Mock/default (always safe):
```bash
pytest tests/test_serving/test_region_backend.py -v
```

### With real Fashionpedia backend (requires model + GPU):
```bash
VISION_REGION_BACKEND=fashionpedia VISION_REGION_ENABLE_REAL=true \
VISION_REGION_DEVICE=cuda \
python -c "
from inference.serving.region_backend import get_region_backend
b = get_region_backend()
print(f'Backend: {b.backend_name}, enabled: {b.enabled}')
"
```

### Smoke test (requires model):
```bash
VISION_REGION_BACKEND=fashionpedia VISION_REGION_ENABLE_REAL=true \
python scripts/smoke_test_312_region_backend.py \
  --image path/to/image.jpg \
  --query "领口在哪里？"
```

---

## 11. Current Limitations

1. **Fashionpedia only** — full `locate_region()` not wired (needs garment instances + SAM + DINO)
2. **No landmark parts** — hem, waist, shoulder, leg_opening require fast-path pipeline
3. **No mask output** — `mask_present` always `False`
4. **No instance association** — regions are image-level (P1.4c)
5. **No 3.1.3 attributes** — `region_attribute_query` returns placeholder
6. **Full-image inference** — Fashionpedia YOLO runs on full image, not garment-cropped
7. **Disabled by default** — real model never loaded in default config

---

## 12. Files Changed

| File | Change |
|---|---|
| `inference/serving/region_backend.py` | **New** — backend interface + implementations + normalization + singleton |
| `configs/serving_config.yaml` | Added `region_backend` config section |
| `inference/serving/qa_orchestrator.py` | Added `_try_region_backend()`, wired into `_route_region_query()` |
| `tests/test_serving/test_region_backend.py` | **New** — 32 tests (backend, normalization, integration, no-leak) |
| `docs/P1_4b_real_region_backend.md` | **New** — This document |

---

## 13. Next Steps

1. **Full `locate_region()` wiring** (P1.4c) — requires garment instances to be available first
2. **Instance-to-region association** (P1.4c) — add `instance_id` to `LocalizedRegion`
3. **Mask output** — when SAM is wired, expose `mask_present=True` + safe `mask_ref`
4. **Add real evaluation cases** — run on annotated images with known regions
5. **Connect 3.1.3 attributes** — replace `region_attribute_query` placeholder
