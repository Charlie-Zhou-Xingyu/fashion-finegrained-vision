# P1.4a — Region Localization QA Integration

> Date: 2026-07-21
> Status: Complete

---

## 1. Product Goal

Enable `/v1/mm/qa` to answer local region questions from 3.1.2 visual evidence:

- "领口在哪里？" — region location
- "有没有口袋？" — region existence
- "这件衣服有哪些细节？" — detail summary
- "有几个口袋？" — region count
- "袖口是什么设计？" — region attribute (placeholder)

The system answers from `localized_regions` visual evidence only. Missing or low-confidence regions produce safe fallback messages — never fabricated answers.

---

## 2. Relationship to 3.1 Modules

| Module | Status | Evidence Layer |
|---|---|---|
| 3.1.1 | Integrated (P1.2) | `garment_instances` — whole-garment detection/segmentation |
| 3.1.2 | **Integrated (P1.4a)** | `localized_regions` — local part/region detection |
| 3.1.3 | Not integrated | `visual_attributes` — fine-grained attributes (future) |

No instance-to-region association is implemented yet. Each localized region is image-level.

---

## 3. Supported Region Intents

| Intent | Example | Behavior |
|---|---|---|
| `region_location_query` | "领口在哪里？" | Returns bbox + confidence of matching region |
| `region_existence_query` | "有没有口袋？" | Reports detected count or "cannot confirm" |
| `region_detail_query` | "有哪些细节？" | Summarizes all reliable regions |
| `region_count_query` | "有几个口袋？" | Counts reliable matching regions |
| `region_attribute_query` | "袖口是什么设计？" | Placeholder: says 3.1.3 not integrated |

---

## 4. `localized_regions` Schema

```json
{
  "region_id": "region_0",
  "part_type": "neckline",
  "part_group": "collar_area",
  "bbox": [100.0, 40.0, 220.0, 110.0],
  "confidence": 0.82,
  "source": "fashion_vision_3_1_2",
  "backend": "mock",
  "mask_present": false,
  "mask_ref": null
}
```

Bbox-only initially. Masks, crops, temp paths, and image bytes are forbidden in API responses.

---

## 5. Chinese Query Mapping

Deterministic mapping in `inference/serving/region_query_mapper.py`:

| Chinese | part_type | part_group |
|---|---|---|
| 领口 | neckline | collar_area |
| 领子 | collar | collar_area |
| 翻领 | lapel | collar_area |
| 袖口 | cuff | sleeve_area |
| 袖子 | sleeve | sleeve_area |
| 口袋 | pocket | pocket_area |
| 拉链 | zipper | closure |
| 扣子/纽扣 | button | closure |
| 下摆 | hem | hem_area |
| 肩部 | shoulder | shoulder_area |
| 腰部 | waist | waist_area |
| 蝴蝶结 | bow | decoration |
| 亮片 | sequin | decoration |
| 图案 | pattern | pattern_area |
| ... | ... | ... |

First match wins for overlapping keywords.

---

## 6. Confidence Policy

| Range | Classification | Behavior |
|---|---|---|
| confidence >= 0.5 | Reliable | Used as definitive evidence |
| 0.3 <= confidence < 0.5 | Low confidence | Warning `region_low_confidence`, no definitive claim |
| confidence < 0.3 | Unreliable | Ignored |
| confidence is None | Missing | Warning `region_confidence_missing`, treated as unreliable |

Constants defined in `qa_orchestrator.py`:
- `REGION_CONFIDENCE_THRESHOLD = 0.5`
- `REGION_LOW_CONFIDENCE_THRESHOLD = 0.3`

---

## 7. Response Examples

### Location present
```
Q: 领口在哪里？
A: 检测到neckline区域，位置约为 bbox=[100, 40, 220, 110]，置信度 0.82。
```

### Existence present
```
Q: 有没有口袋？
A: 当前检测到 1 个pocket区域，置信度 0.76。
```

### Detail summary
```
Q: 这件衣服有哪些细节？
A: 当前检测到的局部细节包括：领口、口袋、拉链、袖口。
```

### Count
```
Q: 有几个口袋？
A: 当前可靠检测到 2 个pocket区域。
```

### Attribute placeholder
```
Q: 袖口是什么设计？
A: 检测到cuff区域，但「cuff设计」需要接入 3.1.3 细粒度属性识别后才能回答。
Warning: region_attribute_not_integrated
```

### Missing
```
Q: 蝴蝶结在哪里？
A: 当前没有可靠定位到该局部区域（bow）。
```

---

## 8. Safety / No-Leak Policy

API responses are tested to never include:
- Mask bitmaps
- Crop images
- Image bytes
- Temp file paths
- Local file paths
- Raw tensors
- Checkpoint paths

`localized_regions_summary` in meta uses safe keys only:
`region_id, part_type, part_group, bbox, confidence, source, backend`

---

## 9. Current Limitations

1. **No real 3.1.2 backend** — Mock regions only. Real 3.1.2 (`locate_region()`) requires GPU, file I/O wrapper.
2. **No instance-region association** — `localized_regions` are image-level.
3. **No 3.1.3 attributes** — `region_attribute_query` returns placeholder.
4. **Bbox-only** — No mask/crop through API.
5. **Single-part queries** — Each query targets one part type. Detail summary lists all.

---

## 10. Next Steps

1. **Wire real 3.1.2 backend** — Wrap `locate_region()` in a backend adapter similar to `FashionVision31SegmentationBackend`.
2. **Add instance-to-region association** — Add `instance_id` to `LocalizedRegion` and filter regions by garment instance.
3. **Connect 3.1.3 attribute extraction** — Replace placeholder with real attribute answers.
4. **Add real evaluation cases** — Run on annotated images with known regions.
5. **Multi-part batch detection** — Support detecting all regions in one pipeline call.

---

## 11. Files Changed

| File | Change |
|---|---|
| `configs/intent_taxonomy.yaml` | Added 5 region_* intent rules |
| `inference/serving/schemas.py` | Added `LocalizedRegion`, `LocalizedRegionSummary` |
| `inference/serving/region_query_mapper.py` | **New** — Chinese→part_type mapping |
| `inference/serving/vision_provider.py` | Extended `MockVisionAttributeProvider` with `mock_regions` param |
| `inference/serving/vision_context.py` | Added `localized_regions` field to `VisionContext` |
| `inference/serving/qa_orchestrator.py` | Added region helpers, `_route_region_query`, 5 answer methods |
| `tests/test_serving/test_region_qa.py` | **New** — 57 tests |
| `docs/P1_4a_region_localization_audit.md` | **New** — 3.1.2 audit |
| `docs/P1_4a_region_localization_qa.md` | **New** — This document |
