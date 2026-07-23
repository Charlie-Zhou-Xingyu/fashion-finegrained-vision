# P1.4a — 3.1.2 Region Localization Audit

> Date: 2026-07-21
> Purpose: Audit existing 3.1.2 implementation before serving-layer integration.

---

## 1. Code Location

3.1.2 region localization code lives under:

```
src/fashion_vision/localization/
```

Key modules:

| File | Role |
|---|---|
| `region_localization_router.py` | Single entry point `locate_region()` — routes to fast-path (landmark) or open-vocab (Fashionpedia YOLO / Grounding DINO) |
| `intent_parser.py` | Parses NL query into structured `RegionIntent` (part, garment_ref, is_fast_path, is_zero_shot) |
| `open_vocab_prompt_map.py` | Maps part names to DINO text prompts (Chinese + English variants) |
| `grounding_dino_locator.py` | Grounding DINO wrapper — image → bbox predictions |
| `fashionpedia_part_detector.py` | Fashionpedia 19-class YOLOv8s part detector |
| `region_locator.py` | Fast-path landmark + geometry region cropping |
| `query_parser.py` | Legacy/deprecated query parser (similar to intent_parser) |
| `part_shape_priors.py` | Spatial shape priors for DINO result filtering |
| `spatial_constraint.py` | Directional/side selection for part detection |
| `anatomical_zoom.py` | Anatomical zoom for small-part localization |
| `garment_ref_filter.py` | Filters results by garment type match |
| `inner_garment_detector.py` | SAM-based inner garment detection for outerwear |
| `torso_prior.py` | Torso spatial priors |

---

## 2. How to Invoke

### Main entry: `locate_region()`

```python
from fashion_vision.localization.region_localization_router import locate_region

result = locate_region(
    query="领口",                    # NL query (Chinese or English)
    instance=instance_dict,          # Standardised instance record
    image=np_image,                  # BGR uint8 H×W×3 numpy array
    image_width=w, image_height=h,   # Image dimensions
    locator=dino_locator,            # GroundingDINOLocator (required for open-vocab)
    dino_threshold=0.3,
    prefer_pred_mask=True,
    sam_wrapper=sam,                 # SamHqWrapper for bbox→mask refinement
    fashionpedia_detector=fp_det,    # FashionpediaPartDetector (optional fast path)
)
```

### Requirements for invocation:
- **Real models required**: YOLO garment detector, SAM-HQ, Grounding DINO (or Fashionpedia YOLO)
- **Image format**: numpy array (BGR, uint8, H×W×3) — loaded from disk
- **Instance format**: Standardised instance dict with bbox, mask path, class info
- **GPU recommended**: DINO and SAM are heavy on CPU
- **File paths**: Instance masks are file-based (mask_path pointing to PNG on disk)

---

## 3. Input Modes

`locate_region()` accepts:
- **Image + text query**: YES — the primary mode. Query specifies the part; image+instance provide visual context.
- **Image only (all regions)**: NO — there is no "detect all regions" mode. Each call targets one part.
- **Text only**: NO — requires image + instance.

For serving integration, this means:
- Each user query about a specific part requires a full 3.1.2 pipeline call.
- "这件衣服有哪些细节？" (detail summary) would require multiple calls OR a different approach.

---

## 4. Raw Output Schema

`locate_region()` returns a dict with these keys:

```python
{
    "status": "ok" | "not_detected" | "error",
    "query": str,                    # Original query
    "part": str,                     # Resolved part name
    "intent": RegionIntent,          # Parsed intent object
    "path": str,                     # "fast" | "open_vocab" | "zero_shot"
    "backend": str,                  # "landmark_geometry" | "fp_yolo" | "grounding_dino" | "none"
    "bbox": [x1, y1, x2, y2] | None,
    "mask": np.ndarray | None,       # Binary mask (NOT serializable)
    "mask_path": str | None,         # Local file path (NOT safe for API)
    "confidence": float | None,
    "crop": np.ndarray | None,       # Cropped image region (NOT serializable)
    "crop_path": str | None,         # Local file path (NOT safe for API)
    "garment_ref_matched": bool,
    "warnings": list,
    "debug": dict,
}
```

### Critical observations for serving:
- **mask** is a numpy array — cannot be returned through API
- **mask_path**, **crop_path** are local file paths — must NOT leak
- **crop** is a numpy array — cannot be returned through API
- **confidence** may be None
- **bbox** is the safest field for API response

---

## 5. Output Level

- **Image-level**: YES — each `locate_region()` call returns one region for one part query
- **Instance-level**: YES — the result is tied to a specific garment instance (passed as input)
- **Global (all regions)**: NO — must call once per part per instance

---

## 6. Masks

- Masks ARE available from the 3.1.2 pipeline (numpy arrays)
- Masks are saved to disk as PNG files (mask_path)
- For serving: bbox-only is the correct initial approach
- Mask references can be added later (e.g., `mask_present: true`, `mask_ref: "region_0_mask"`)

---

## 7. Temp Files / Local Paths

- The 3.1.2 pipeline uses local file paths extensively:
  - Instance mask PNGs on disk
  - Crop images saved to disk
  - Debug visualizations saved to disk
- The serving layer MUST NOT return any of these paths
- The serving layer MUST NOT read and return image bytes

---

## 8. Existing Tests / Eval

- `tests/test_phase2_localization.py` — Phase 2 localization tests
- `tests/test_router_helpers.py` — Router helper tests
- `tests/test_fashionpedia_part_detector.py` — Fashionpedia detector tests
- `tools/eval/visualize_open_vocab_localization.py` — Visualization/eval script
- `scripts/validate_p13_visual_qa.py` — P1.3 visual QA validation

All tests require real models or mock model instances — not suitable for pure unit tests.

---

## 9. Integration Strategy for P1.4a

### What to do now:
1. Add `localized_regions` schema — bbox-only, no masks, no paths, no crops
2. Add Chinese query → part type mapping
3. Add intent classification for region queries
4. Mock provider returns pre-canned localized_regions
5. QA orchestrator routes region queries using mock regions
6. Confidence policy for safe fallback

### What to defer:
- Real 3.1.2 backend wiring (requires GPU, models, file I/O wrapper)
- Instance-to-region association
- Mask/crop serving
- Multi-part batch detection

### Approach:
The real 3.1.2 pipeline (`locate_region()`) is NOT wrapped in P1.4a. Instead:
- Schema + mock integration is fully implemented
- A `disabled_by_default` wrapper skeleton is added
- Documentation explains how to connect the real backend

---

## 10. Intentionally Out of Scope

- Instance-to-region association (no `instance_id` on regions yet)
- Full 3.1.3 attribute extraction
- Model training or fine-tuning
- LLM/VLM/LangGraph agent
- Real model inference in unit tests
- Returning mask bitmaps or crops through API
- Default real 3.1.2 execution
