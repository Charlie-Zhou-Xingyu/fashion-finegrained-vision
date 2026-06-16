# Current Project Status

> Last updated: 2026-06-16

---

## 0. Executive Summary

All P1 (local region localization) and P2 (fine-grained attribute
classification) milestones are complete.

Current focus: code cleanup before YOLO balanced retraining for PRD 3.1.1.

---

## 1. PRD Module Status

| PRD Module | Description | Status |
|---|---|---|
| 3.1.1 Garment instance segmentation | YOLO 13-class detection + SAM-HQ segmentation | Prototype ready; balanced retraining planned |
| 3.1.2 Language-guided local region localization | Landmark + geometry region localization | P1 prototype complete (92.0% valid response on Batch60) |
| 3.1.3 Fine-grained attribute extraction | FashionAI ResNet18 classifiers | All 8 tasks trained; not advancing further now |
| 3.2 Multimodal QA | Natural language fashion QA | Not started |
| 3.3 Agent/RAG | Intent recognition, retrieval | Not started |

---

## 2. Completed Milestones

### P0: DeepFashion2 Annotation Benchmark

- Parsed 135,975 annotation files, 221,535 garment instances
- Full parsing time: 31.3 s (~4,342 files/s)

### P1: Local Region Localization Prototype

- YOLO (13-class) + SAM-HQ garment pipeline: functional and benchmarked
- ResNet18 landmark predictor: trained, 39 landmarks, 13 garment categories
- Semantic region crops: neckline, cuff, hem, waist, shoulder, leg_opening
- Mask-aware region crops: functional
- Rule-based Chinese query-to-region demo: 92.0% valid response on Batch60 (300 queries, 5 region types)

**500-image pipeline benchmark:**

| Stage | Time | % |
|---|---:|---:|
| YOLO detection | 12.1 s | 5.8% |
| SAM-HQ segmentation | 146.3 s | 69.6% |
| Landmark prediction | 27.2 s | 13.0% |
| Region crop generation | 3.4 s | 1.6% |
| Mask-aware crop generation | 21.1 s | 10.0% |
| **Total / 500 images** | **210.1 s** | — |
| **Average per image** | **420 ms** | — |

### P2: Fine-grained Attribute Classification

All 8 FashionAI attribute tasks trained with ResNet18, multiview_v2_pipeline strategy:

| Task | Val Macro-F1 | Test Macro-F1 |
|---|---:|---:|
| collar_design | 0.709 | 0.764 |
| pant_length | 0.688 | 0.740 |
| lapel_design | 0.636 | 0.680 |
| neckline_design | 0.679 | 0.665 |
| coat_length | 0.692 | 0.618 |
| neck_design | 0.681 | 0.624 |
| sleeve_length | 0.642 | 0.612 |
| skirt_length | 0.640 | 0.593 |

---

## 3. Current Pipeline Architecture

```
image
  → YOLO 13-class detection         tools/infer/predict_garments_yolo.py
  → SAM-HQ segmentation             tools/infer/segment_garments_samhq.py
  → Landmark prediction (ResNet18)  tools/infer/infer_landmarks_for_predictions.py
  → Region crop generation          tools/crop/crop_garment_regions_from_landmarks.py
  → Mask-aware region crops         tools/crop/apply_samhq_mask_to_region_crops.py
  → Chinese query → region match    tools/demo/query_region_online_demo.py
```

**YOLO detection output fields (after pre-retraining cleanup):**

```json
{
  "det_id": 0,
  "class_id": 0,
  "class_name": "short sleeve top",
  "fine_class_id": 0,
  "fine_class_name": "short sleeve top",
  "coarse_class_id": 0,
  "coarse_class_name": "top",
  "confidence": 0.87,
  "bbox_xyxy": [x1, y1, x2, y2],
  "bbox_xywh": [x, y, w, h],
  "bbox_format": "xyxy_abs_pixels",
  "image_width": 768,
  "image_height": 1024
}
```

Internal 13-class fields (`class_id`, `class_name`, `fine_class_id`,
`fine_class_name`) are preserved for downstream 3.1.2 landmark and region
localization.  PRD-facing 5-class output uses `coarse_class_id` and
`coarse_class_name`.

---

## 4. Current Category Mapping

**Source of truth:** `configs/category_mapping.yaml`

| DeepFashion2 (0-based) | Fine name | Coarse id | Coarse name |
|---:|---|---:|---|
| 0 | short sleeve top | 0 | top |
| 1 | long sleeve top | 0 | top |
| 2 | short sleeve outwear | 3 | outerwear |
| 3 | long sleeve outwear | 3 | outerwear |
| 4 | vest | 0 | top |
| 5 | sling | 0 | top |
| 6 | shorts | 1 | pants |
| 7 | trousers | 1 | pants |
| 8 | skirt | 2 | skirt |
| 9 | short sleeve dress | 4 | dress |
| 10 | long sleeve dress | 4 | dress |
| 11 | vest dress | 4 | dress |
| 12 | sling dress | 4 | dress |

---

## 5. Immediate Next Step

Run YOLO balanced retraining.

See `docs/plans/yolo_balanced_training_plan.md` for commands and evaluation plan.

---

## 6. Known Technical Debt (deferred to post-retraining)

| Issue | File(s) | Priority |
|---|---|---|
| Duplicate query parsing logic | `query_region_online_demo.py` vs `query_parser.py` | Medium |
| Two `region_visualizer.py` files | `localization/` and `visualization/` | Low |
| Utility helpers duplicated | `sanitize_filename_part`, `load_json` | Low |
| No end-to-end eval script for confusion matrix generation | `tools/eval/` | Medium |
| Script sprawl in `scripts/` (26 files) | `scripts/` | Low |

See `docs/architecture/codebase_cleanup_plan.md` for the full staged plan.

---

## 7. Out of Scope (Do Not Advance Now)

- Shoes, bags, accessories
- 3.1.3 further training or evaluation
- 3.2 Multimodal QA
- 3.3 Agent/RAG
- Runtime optimization (SAM-HQ replacement)
- Large-scale refactors before retraining
