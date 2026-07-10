# Text-guided Garment Region Demo Report

## 1. Objective

This report describes the current rule-based text-guided garment local region localization demo.

The goal is to support a prototype version of PRD 3.1.2:

> Given a garment image and a natural language query, locate the corresponding local garment region and output its mask, bounding box, crop, overlay visualization, and structured metadata.

The current implementation focuses on Chinese garment part queries and DeepFashion2-supported garment categories.

---

## 2. Input and Output

### 2.1 Input

The demo takes:

- One garment image.
- One natural language query.
- Optional target class.
- Optional target detection id.
- Optional component hint.
- Optional reused pipeline directory.

Example query:

```text
领口
左袖子
右袖子
下摆
裙摆
腰部
裤腿
```

Example command:

```bat
python -m tools.demo.query_region_online_demo ^
  --image D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg ^
  --query "腰部" ^
  --output-dir outputs\query_region_online_demo
```

---

### 2.2 Output

For each query, the demo outputs:

| Output | Description |
|---|---|
| `result.json` | Structured query result |
| `region_overlay.jpg` | Full-image visualization with selected region |
| `region_mask_full.png` | Full-image binary mask for selected region |
| `selected_image_crop.png` | Selected local image crop |
| `selected_mask_crop.png` | Selected local mask crop |
| `selected_masked_crop.png` | Selected masked local region |

---

## 3. Pipeline Overview

The demo uses the existing garment processing pipeline:

```text
Input image
    ↓
YOLO garment detection
    ↓
SAM-HQ garment segmentation
    ↓
Landmark prediction
    ↓
Semantic region crop generation
    ↓
Mask-aware region crop generation
    ↓
Chinese query parsing and candidate selection
    ↓
Selected region mask / bbox / overlay / JSON output
```

When `--reuse-pipeline-dir` is provided, the demo skips YOLO, SAM-HQ, landmark prediction, and crop generation, and only performs query parsing and region selection.

---

## 4. Supported Query Types

| Query | Parsed Region | Component | Class Constraint | Notes |
|---|---|---|---|---|
| 领口 | collar | None | None | General collar query |
| 袖子 | sleeve | None | None | General sleeve query |
| 左袖子 | sleeve | left_sleeve | None | Left sleeve only |
| 右袖子 | sleeve | right_sleeve | None | Right sleeve only |
| 下摆 | hem | None | None | Generic hem query |
| 裙摆 | hem | None | skirt | Skirt hem only |
| 连衣裙下摆 | hem | None | dress classes | Dress hem only |
| 腰部 | waist | None | None | Uses waist-specific priority |
| 裤腿 | pant_leg | None | None | Trousers/shorts leg if available |

---

## 5. Query Parsing Rules

The current parser is a lightweight rule-based Chinese parser.

It maps natural language expressions to target regions:

| Region | Aliases |
|---|---|
| collar | collar, neckline, neck, 领口, 衣领, 领子, 领部 |
| sleeve | sleeve, sleeves, 袖子, 衣袖, 袖口, 左袖, 右袖 |
| hem | hem, 下摆, 衣摆, 底边, 下边, 边缘 |
| waist | waist, 腰, 腰部, 腰线, 裤腰, 裙腰, 收腰, 腰围 |
| pant_leg | pant_leg, pants leg, trouser leg, 裤腿, 裤管, 腿部, 裤脚 |

Special query aliases are handled before generic region aliases:

| Special Query | Parsed Result |
|---|---|
| 裙摆 | region=hem, target_class=skirt |
| 裙子下摆 | region=hem, target_class=skirt |
| 连衣裙下摆 | region=hem, target_class=dress classes |
| 连衣裙裙摆 | region=hem, target_class=dress classes |

This is important because `裙摆` and `下摆` should not be treated as the same query.

---

## 6. Candidate Selection Rules

Candidate records are loaded from:

```text
pipeline/05_region_masked_crops/region_masked_crops.json
```

A candidate must satisfy:

1. `masked_success == true`
2. `region == target_region`
3. If `target_det_id` is provided, `det_id` must match.
4. If `target_class` is provided or inferred, `class_name` must match.
5. If `target_component` is provided, `component` must match.

---

## 7. Waist-specific Priority

Only `waist` uses garment-category priority.

The priority is:

```text
upper-body waist > dress waist > lower-body waist
```

This is implemented because a generic query such as `腰部` should normally refer to the waist area of the upper garment if both an upper garment and a skirt/pants are present.

Example:

If the candidates are:

| Class | Region |
|---|---|
| long sleeve top | waist |
| skirt | waist |

The selected candidate is:

```text
long sleeve top / waist
```

For all other regions, there is no upper/dress/lower category priority.

---

## 8. Separation between 下摆 and 裙摆

The demo explicitly separates:

```text
下摆
```

and:

```text
裙摆
```

Behavior:

| Query | Behavior |
|---|---|
| 下摆 | Selects a generic hem candidate without forcing skirt |
| 裙摆 | Selects hem candidate with class constraint `skirt` |
| 连衣裙下摆 | Selects hem candidate from dress classes |

This avoids incorrect behavior where `裙摆` might select an upper garment hem or `下摆` might be forced to skirt hem.

---

## 9. Example Result

Example image:

```text
D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg
```

Query:

```text
腰部
```

Selected result:

```json
{
  "query": "腰部",
  "target_region": "waist",
  "selection": {
    "rule": "deterministic_rule_based_selection",
    "reason": "waist query selected upper-body garment waist by default"
  },
  "selected": {
    "class_name": "long sleeve top",
    "det_id": 0,
    "region": "waist",
    "component": "waist"
  }
}
```

Candidate ranking:

| Rank | Class | Group | Region |
|---:|---|---|---|
| 1 | long sleeve top | upper | waist |
| 2 | skirt | lower | waist |

---

## 10. Batch Validation

A controlled 60-image validation was conducted.

Each image was queried with:

```text
领口
左袖子
右袖子
下摆
腰部
```

After deduplication by `image + query`, the final result contains 300 query rows.

| Metric | Value |
|---|---:|
| Sampled images | 60 |
| Query rows | 300 |
| Success | 276 |
| Failed | 24 |
| Valid response rate | 92.0% |

Per-query success rate:

| Query | Success Rate |
|---|---:|
| 腰部 | 100.0% |
| 领口 | 95.0% |
| 下摆 | 95.0% |
| 左袖子 | 85.0% |
| 右袖子 | 85.0% |

The valid response rate should not be interpreted as strict localization accuracy, because the selected regions were not manually annotated for correctness in this experiment.

---

## 11. Limitations

1. The current text parser is rule-based.
2. It only supports predefined garment regions.
3. It does not support arbitrary open-vocabulary visual grounding.
4. It depends on the quality of YOLO detection, SAM-HQ masks, and landmark prediction.
5. If a region crop is not generated by the upstream pipeline, the query cannot return that region.
6. The demo currently focuses on clothing categories from DeepFashion2 and does not support shoes, bags, or accessories.

---

## 12. Next Steps

1. Generate a batch result CSV for easier inspection.
2. Add more query aliases if needed.
3. Refactor the demo into reusable modules.
4. Connect selected local crops with FashionAI attribute classifiers.
5. Build a region-to-attribute demo for queries such as:

```text
这件衣服的袖子是什么长度？
这个领口是什么设计？
这条裙子的裙长是什么？
```
