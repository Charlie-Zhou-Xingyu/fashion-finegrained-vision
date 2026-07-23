# P1.4d — Full 3.1.2 Pipeline Audit

> Date: 2026-07-21

---

## 1. Full 3.1.2 Entry Point

**Function**: `locate_region()` in `src/fashion_vision/localization/region_localization_router.py`

## 2. Signature

```python
def locate_region(
    query: str,                                         # Chinese or English region query
    instance: Dict[str, Any],                           # Garment instance dict
    image: np.ndarray,                                  # BGR uint8 H×W×3
    image_width: int, image_height: int,
    locator: Optional[GroundingDINOLocator] = None,     # DINO (for open-vocab/zero-shot)
    dino_threshold: float = 0.3,
    prefer_pred_mask: bool = True,
    sam_wrapper: Optional[SamHqWrapper] = None,         # SAM (for bbox→mask)
    fashionpedia_detector: Optional[FashionpediaPartDetector] = None,
) -> Dict[str, Any]:
```

## 3. Instance Requirement

`locate_region()` is **instance-level** — needs a garment instance dict with:
- `bbox` / `det_bbox` / `bbox_xyxy` — for cropping
- `mask_path` (optional) — for garment mask gating
- `class_name` / `fine_class_name` — for garment_ref matching

**Serving workaround**: Create synthetic full-image instance:
```python
{"instance_id": "full_image", "bbox": [0, 0, w, h], "category": "unknown"}
```
Without mask path, garment_mask is None → no mask gating (works for full-image inference).

## 4. Routing Logic (inside locate_region)

| Path | Trigger | Backend |
|---|---|---|
| Fast-path (landmark) | `intent.is_fast_path` (hem, waist, shoulder, leg_opening) | `region_locator.py` — requires real instance + landmarks |
| Fashionpedia YOLO | `_is_fashionpedia_part(intent.part)` | `FashionpediaPartDetector.detect()` |
| Fashionpedia miss → DINO fallback | FP miss + neckline/cuff | Falls back to fast-path (landmark) |
| Fashionpedia miss → no fallback | FP miss + other FP-core parts | Returns `not_detected` (no DINO hallucination) |
| Open-vocab | Not in FP + locator available | `GroundingDINOLocator` with per-part prompts |
| Zero-shot | `intent.is_zero_shot` (not in PART_VOCAB) | DINO with raw query noun phrase |

## 5. Supported Parts

| Source | Count | Parts |
|---|---|---|
| PART_VOCAB | 25 | bag, belt, buckle, button, collar_stand, cuff, drawstring, epaulette, fringe, hem, lapel, leg_opening, neckline, pattern, placket, pocket, ruffle, shoes, shoulder, shoulder_seam, sleeve_seam, strap, tie_strap, waist, zipper |
| Fashionpedia (FP_CORE) | 13 | hood, collar, lapel, epaulette, sleeve, pocket, neckline, buckle, zipper, bow, fringe, ruffle, sequin |
| Fashionpedia (DECORATION) | 6 | applique, bead, flower, ribbon, rivet, tassel |
| Fast-path only | 4 | hem, waist, shoulder, leg_opening |

## 6. Does it include...

| Feature | Status |
|---|---|
| Fashionpedia YOLO | ✅ Yes — priority fast detector |
| DINO (Grounding DINO tiny) | ✅ Yes — open-vocab fallback |
| FP→DINO fallback | ✅ Partial — only neckline/cuff (by design; other FP-core parts get `not_detected`) |
| Shape priors / filtering | ✅ Yes — `filter_by_shape_priors()` |
| Spatial constraints (side/direction) | ✅ Yes — `select_side_detection()`, `select_direction_detection()` |
| Anatomical zoom | ✅ Yes — DINO path only |
| Mask gating | ✅ Yes — garment mask fills non-garment pixels |
| SAM box-prompt mask refinement | ✅ Yes — if `sam_wrapper` provided |
| Per-part thresholds | ✅ Yes — `part_detection_config.py` |

## 7. Unsafe Outputs

Raw `locate_region()` output contains:
- `mask`: numpy array (NOT serializable, MUST strip)
- `crop` / `crop_path`: internal path (MUST strip)
- `mask_path`: local file path (MUST strip)
- `debug`: debug metadata with internal paths (MUST strip for API)

**All must be normalized before serving.**

## 8. How P1.4c Differs from Full 3.1.2

| Aspect | P1.4c (FashionpediaRegionBackend) | Full 3.1.2 (locate_region) |
|---|---|---|
| Backends | Fashionpedia YOLO only | YOLO + DINO + landmark |
| DINO fallback | No | Yes (neckline/cuff only) |
| Shape priors | No | Yes |
| Anatomical zoom | No | Yes (DINO path) |
| Mask gating | No | Yes (if mask available) |
| Instance requirement | None (image-level) | Garment instance required |
| Prompt optimization | None | Per-part optimized prompts |
| Multi-instance NMS | None | Class-aware IoU thresholds |

## 9. Serving Adapter Strategy

Use `locate_region()` with synthetic full-image instance:
- Fashionpedia parts → FP YOLO runs, DINO fallback for neckline/cuff
- Non-FP parts → DINO with per-part prompts
- Fast-path parts (hem, waist, shoulder, leg_opening) → skip (requires real instance)
- No SAM → bbox-fill pseudo-mask

Normalize: strip mask/crop/path, keep bbox+score+backend label.
