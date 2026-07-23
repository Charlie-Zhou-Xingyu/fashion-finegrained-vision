# P1.3 — Visual Instance QA (3.1.1 garment_instances → 3.2.1 multimodal QA)

> Status: COMPLETE (2026-07-16)
> Depends on: P1.2 (real 3.1.1 segmentation wired via `FashionVision31SegmentationBackend`)

## 1. Problem

P1.2 made the serving system able to run real 3.1.1 detection + segmentation and
return `garment_instances` in the vision provider result.  But the QA layer
(`/v1/mm/qa`) never consumed them: only `len(garment_instances)` appeared in
response meta, and questions like "图里有几件衣服？" fell through to
`fallback_unknown` → "当前问题类型暂不支持".

P1.3 makes the already-wired 3.1.1 output answerable — nothing more.

## 2. What 3.1.1 can answer (and P1.3 exposes)

| Question type | sub_intent | Example | Answer source |
|---|---|---|---|
| Count | `count` | 图里有几件衣服？ | `len(garment_instances)` + per-category counts |
| Detection listing | `detection` | 图中检测到了什么？ | `category` + `fine_class_name` + `confidence` |
| Category existence | `existence` | 有没有上衣？ | `category` match against PRD 5 coarse classes |
| Location | `location` | 检测框在哪里？ | `bbox` (xyxy absolute pixels) |
| Segmentation existence | `segmentation` | 有没有分割结果？ | `mask_present` flags only |

## 3. What it can NOT answer (out of scope, by design)

- Fabric, color, craft, design detail, style, season, material — 3.1.3 attribute
  classification is NOT implemented; segmentation knows nothing about these.
- Language-guided part grounding ("领口在哪里") — that is 3.1.2, not wired into serving.
- Instance disambiguation ("左边那件") — no spatial reasoning.
- Anything requiring the mask bitmap — masks are never returned.

If instances are empty/missing the answer is always the safe string
"当前没有可用的服饰实例检测结果。" plus warning `vision_instances_unavailable`.
No values are ever fabricated.

## 4. Implementation

Minimal change set — no new architecture:

| File | Change |
|---|---|
| `configs/intent_taxonomy.yaml` | New primary intent `visual_instance_query` with 5 sub_intents (`segmentation`, `location`, `count`, `existence`, `detection`), placed after `chat`, ordered specific-first |
| `inference/serving/qa_orchestrator.py` | New route `_route_visual_instance` + helpers `_summarize_garment_instances` (safe-key whitelist) and `_build_garment_instance_sources` |
| `tests/test_serving/test_visual_instance_qa.py` | 25 tests (mocked provider, no YOLO/SAM) |

Not touched: `GarmentPipeline`, `tools/infer/`, provider defaults (mock remains
default), existing response fields, schemas, app.py.

## 5. Example request / response

Request:

```json
POST /v1/mm/qa
{
  "query": "图里有几件衣服？",
  "image_bytes": "<base64>"
}
```

Response (real provider enabled, 2 garments detected):

```json
{
  "status": "success",
  "data": {
    "answer": "检测到2件服饰：上衣1件、裤子1件。",
    "answer_type": "visual_instance_answer",
    "sources": [
      {"type": "garment_instance", "id": "inst_0", "value": "top",
       "attribute_confidence": 0.93,
       "metadata": {"fine_class_name": "short sleeve top",
                     "bbox": [120.0, 80.0, 300.0, 350.0], "mask_present": true}}
    ],
    "meta": {
      "route": "visual_instance_query",
      "primary_intent": "visual_instance_query",
      "sub_intent": "count",
      "garment_instance_count": 2,
      "num_garment_instances": 2,
      "garment_instances_summary": [
        {"instance_id": "inst_0", "category": "top",
         "fine_class_name": "short sleeve top",
         "bbox": [120.0, 80.0, 300.0, 350.0],
         "confidence": 0.93, "mask_present": true}
      ]
    }
  },
  "warnings": []
}
```

With mock provider (default) or no detections:

```json
{
  "data": {
    "answer": "当前没有可用的服饰实例检测结果。",
    "answer_type": "visual_instance_answer",
    "meta": {"garment_instances_summary": []}
  },
  "warnings": [{"code": "vision_instances_unavailable", "severity": "info"}]
}
```

## 6. Safety / no-leak rules

`garment_instances_summary` and `garment_instance` sources are built through a
whitelist (`_GARMENT_INSTANCE_SAFE_KEYS`): only `instance_id`, `category`,
`fine_class_name`, `bbox`, `confidence`, `mask_present` pass.  Guaranteed
absent, even if a backend adds fields later:

- mask bitmaps / `mask_ref` file semantics (mask_ref is dropped from summary)
- temp dir or any filesystem path
- `image_bytes` (only boolean `has_image_bytes` flags in meta)
- raw pipeline output

Covered by `test_response_no_temp_path`, `test_response_no_image_bytes`,
`test_response_no_mask_bitmap`, `test_garment_instances_summary_safe_keys`.

## 7. Limitations

- Intent rules are keyword/regex-based; paraphrases outside the rule set fall
  back to existing routes (no visual answer).
- `existence` relies on the classifier's `garment_ref` entity (5 coarse
  classes only); fine-class refs like "T恤" are not matched against instances.
- `location` returns the bbox of the first/first-matching instance only.
- Answers are deterministic templates in Chinese.
- Real vision remains opt-in (`VISION_PROVIDER=real` + `VISION_REAL_ENABLED=1`);
  the default mock provider always yields the safe "no instances" answer.
