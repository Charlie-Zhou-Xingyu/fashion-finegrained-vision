# Project Context

## Project Name

Fashion Fine-grained Vision

## PRD Goal

Build a multimodal fashion visual analysis system for e-commerce.

Main PRD modules:

- 3.1.1 Garment instance segmentation
- 3.1.2 Language-guided local region localization
- 3.1.3 Fine-grained attribute extraction
- 3.2 Multimodal QA
- 3.3 Agent/RAG

Current priority is 3.1.1 and 3.1.2.

---

## Current Implemented Pipeline

Current visual pipeline:

```text
image
-> YOLO garment detection
-> SAM-HQ garment segmentation
-> landmark prediction
-> semantic local region crop generation
-> mask-aware semantic region crop generation
```

A 500-image benchmark has been completed:

- 500 images
- 913 garment instances
- 3220 region crops
- 3212 successful mask-aware crops
- 420.13 ms/image
- SAM-HQ is the main runtime bottleneck

---

## Current Query-region Module

A rule-based Chinese query-to-region demo is implemented.

Supported queries:

- 领口
- 左袖子
- 右袖子
- 下摆
- 腰部
- 裙摆
- 裤腿

Batch60 result:

- 300 query runs
- 276 success
- 92.0 percent valid response rate

---

## Current Attribute Classification Progress

All 8 FashionAI attribute tasks have been trained and evaluated using the
multiview_v2_pipeline strategy with ResNet18 classifiers.

Completed tasks:

| Task | Test Macro-F1 |
|---|---:|
| collar_design | 0.764 |
| pant_length | 0.740 |
| lapel_design | 0.680 |
| neckline_design | 0.665 |
| coat_length | 0.618 |
| neck_design | 0.624 |
| sleeve_length | 0.612 |
| skirt_length | 0.593 |

3.1.3 is complete as a baseline.  Do not advance further now.

---

## Current Scope

In scope now:

- 3.1.1 garment detection / segmentation cleanup
- 3.1.2 language-guided local region localization cleanup
- PRD-facing 5-class output for 3.1.1
- Internal 13-class fine category preservation

Out of scope now:

- shoes
- bags
- accessories
- advancing 3.1.3
- runtime performance optimization as the first priority
- large refactor without audit
- model retraining unless explicitly requested

---

## Category Strategy

DeepFashion2 detector is trained on 13 classes.

For PRD 3.1.1 reporting and external output, map 13 classes to 5 DeepFashion2-supported PRD classes:

- top
- pants
- skirt
- outerwear
- dress

Internal modules must keep 13-class fine category for landmark schema and local region localization.

Do not collapse internal pipeline to 5 classes.

Recommended detection output fields:

```json
{
  "fine_class_id": 0,
  "fine_class_name": "short sleeve top",
  "coarse_class_id": 0,
  "coarse_class_name": "top"
}
```

---

## Current Next Task

Pre-retraining cleanup is complete.  The YOLO detection output now emits
dual-label fields for every detected instance:

```json
{
  "class_id": 0,
  "class_name": "short sleeve top",
  "fine_class_id": 0,
  "fine_class_name": "short sleeve top",
  "coarse_class_id": 0,
  "coarse_class_name": "top"
}
```

The immediate next step is YOLO balanced retraining:

```text
data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml
  -> yolo detect train
  -> eval_13cls_confusion_as_5cls.py
  -> 5-class PRD-facing 3.1.1 evaluation report
```

See `docs/yolo_balanced_training_plan.md` for exact commands.
