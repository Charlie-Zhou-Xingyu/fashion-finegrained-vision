# P1.4e — 3.1.1 to 3.1.2 Instance Bridge Audit

## 3.1.1 garment_instances fields (from FashionVision31SegmentationBackend)

```python
{
    "instance_id": "inst_0",
    "category": "top",                    # PRD coarse class
    "fine_class_name": "short sleeve top", # 13-class fine name
    "bbox": [120.0, 80.0, 300.0, 350.0],  # xyxy absolute pixels
    "bbox_format": "xyxy_abs_pixels",
    "confidence": 0.93,
    "mask_present": True,
    "source": "fashion_vision_3_1",
    "stage": "segmentation",
    "mask_ref": "mask_inst_0",            # placeholder only — NOT a path
    "sam_score": 0.95,                    # optional
}
```

## locate_region() instance requirements

Extracted via `_instance_bbox()` and `resolve_instance_mask_path()`:

| Field | Required? | Used for |
|---|---|---|
| `bbox` / `det_bbox` / `bbox_xyxy` | Semi | Garment cropping (falls back to full image if missing) |
| `pred_mask_path` / `gt_mask_path` | No | Mask gating (returns None if missing → no gating) |
| `class_name` / `fine_class_name` / `category` | No | Garment_ref matching |
| `instance_id` | No | Result tracking |

Also needed by fast-path `locate_region_from_instance()`: instance mask (PNG on disk) + landmark data.

## Can mask_ref be resolved?

**No.** `mask_ref` is a deliberate placeholder ("mask_inst_0") — the real PNG path is lost after the 3.1.1 temp directory is cleaned up.

## Can bbox-only instance work?

**Yes.** `_load_garment_mask()` returns None when no `pred_mask_path`/`gt_mask_path` exists. The pipeline continues without mask gating. Fashionpedia YOLO and DINO both work on the garment-cropped image.

## Which parts need real instance geometry?

| Part type | Bbox-only instance | Full instance (mask+landmarks) |
|---|---|---|
| Fashionpedia parts (neckline, pocket, etc.) | ✅ Works | ✅ Works |
| DINO parts (button, sequin, etc.) | ✅ Works | ✅ Works (with mask gating) |
| Fast-path (hem, waist, shoulder, leg_opening) | ❌ Needs mask+landmarks | ✅ Works |

## Bridge strategy

1. Convert `garment_instance` dict to `locate_region()` instance dict
2. Map `instance_id`, `category`, `fine_class_name`, `bbox` directly
3. Set `pred_mask_path` only if mask_ref can be resolved (currently: never)
4. Fast-path parts pass through to `locate_region()` — they'll get `not_detected` without mask
5. Synthetic full-image instance remains as fallback when no garment instances exist
