# Current Project Status

> Last updated: 2026-07-02

---

## 0. Executive Summary

P1 (region localization), P2 (attribute training), P3 (attribute inference
infrastructure + Stage 6 pipeline integration), P4 (YOLO balanced retraining),
and P7 (Fashionpedia 19-class part detector baseline + balanced training) are complete.
**P7 balanced training DONE (2026-07-02):** model placed at
`models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt`.
Next: evaluate per-class mAP vs baseline, decide Fashionpedia → router integration trigger.

**3.1.2 Phases 0–4 COMPLETE (2026-06-23):** Full open-vocabulary localization pipeline
implemented and smoke-tested.  All Phase 1 deterministic fixes from the industrial
grounding plan are now **complete (2026-06-24)**, including mask-gated DINO detection,
shape-prior empty-list return, five-status vocabulary, long-tail Chinese vocabulary,
garment_ref mismatch flag, and debug visualization script.

- Phase 0+1: Unified `intent_parser.py`, 13-part vocab, `is_fast_path` routing.
- Phase 2: `grounding_dino_locator.py` — HF transformers 5.12.1,
  IDEA-Research/grounding-dino-tiny. Collar score 0.47 on test image.
  Key fix: text query requires trailing period (auto-appended in `detect()`).
- Phase 3: `spatial_constraint.py` — image-space left/right mask halving.
- Phase 4: `region_localization_router.py` — unified entry point; demo updated
  with open-vocab branch.

Current focus: Phase 2 evaluation baseline (threshold calibration) + SAM-HQ IoU
+ Fashionpedia balanced training execution.

---

## 1. PRD Module Status

| PRD Module | Description | Status |
|---|---|---|
| 3.1.1 Garment instance segmentation | YOLO 13-class detection + SAM-HQ segmentation | **Balanced model promoted** (2026-06-21); all 5 classes recall+precision improved |
| 3.1.2 Language-guided local region localization | Landmark + geometry + open-vocab (DINO) + Fashionpedia part detector | P1 prototype complete (92% on Batch60). Open-vocab upgrade complete (Phases 0–4). Industrial grounding **Phase 1 complete (2026-06-24)**. Fashionpedia 19-class part detector **baseline trained (2026-06-29)**, class-balanced training in progress — see Section 10. |
| 3.1.3 Fine-grained attribute extraction | FashionAI ResNet18 classifiers | Inference pipeline complete + Stage 6 integrated; accuracy gap diagnosed (data scarcity) |
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

### P2: Fine-grained Attribute Classification (training)

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

PRD target: macro-F1 ≥ 0.88.  Current best is 0.764 (collar_design).  Gap not yet closed.

### P4: YOLO Balanced Retraining Evaluation + Promotion (2026-06-21)

Retrained YOLOv8n on class-balanced DeepFashion2 (inverse-sqrt frequency, max repeat 5).
Evaluated on val2000 set; all 5 PRD classes improved in both recall and precision.

**5-class aggregated evaluation results:**

| Metric | Baseline | Balanced model | Change |
|---|---:|---:|---:|
| Foreground 5-class accuracy (excl. background) | 0.9274 | **0.9379** | +1.05% |
| Foreground correct detection rate (incl. missed) | 0.8925 | **0.9052** | +1.27% |

**Per-class comparison:**

| Class | Baseline Recall | Balanced Recall | Δ Recall | Baseline Precision | Balanced Precision | Δ Precision |
|---|---:|---:|---:|---:|---:|---:|
| top | 0.9202 | 0.9226 | +0.24% | 0.7941 | 0.8419 | +4.78% |
| pants | 0.9474 | 0.9551 | +0.77% | 0.8520 | 0.8943 | +4.23% |
| skirt | 0.8327 | 0.8612 | +2.85% | 0.7405 | 0.7961 | +5.56% |
| outerwear | 0.7933 | 0.8360 | +4.27% | 0.6279 | 0.6873 | +5.94% |
| dress | 0.8122 | 0.8377 | +2.55% | 0.7178 | 0.7588 | +4.10% |

No class regressed.  Largest gains on outerwear/skirt/dress — exactly the under-represented
classes targeted by balanced sampling.  Model promoted:
`D:\best.pt` → `models/detectors/yolov8n_deepfashion2_13cls_best.pt`

Evaluation artifacts: `outputs/yolo_eval_balanced/`, `outputs/yolo_eval_baseline/`

### P6: Industrial Grounding Phase 1 — Deterministic 3.1.2 Fixes (2026-06-24)

All six Phase 1 items from `docs/industrial_grounding_implementation_plan.md` are
complete.  No new models or external servers required.

| Item | Status | Key change |
|---|---|---|
| 1.1 Mask-gated DINO detection | ✅ | `_crop_image_and_mask()`: image + mask cropped from identical clamped box; INTER_NEAREST resize on mismatch; `crop_mask` passed to `detect_multi_prompt()` |
| 1.2 Shape priors: empty list on all-rejected | ✅ | `filter_by_shape_priors()` returns `[]`; callers emit `not_detected`; no fallback-best-candidate |
| 1.3 Status vocabulary (router) | ✅ | Router uses `not_detected` (with `reason`) and `success`; no `"failed"` in open-vocab path |
| 1.4 Long-tail Chinese vocab | ✅ | 6 new parts in `PART_VOCAB` + `_PART_TO_GROUNDING_TEXT`: `drawstring`, `tie_strap`, `ruffle`, `fringe`, `shoulder_seam`, `sleeve_seam` |
| 1.5 `garment_ref` mismatch flag | ✅ | `_flag_garment_ref_mismatch()` added to router; sets `instance["_garment_ref_mismatch"]=True` when `intent.garment_ref` doesn't match instance fine class; fast-path `setdefault` fixed |
| 1.6 Debug visualization script | ✅ | `scripts/visualize_localization_debug.py` — 6-panel montage: original, raw crop, mask-gated crop, all candidates, rejected+reasons, final result |

**New test file:** `tests/test_router_helpers.py` — 5 tests covering garment_ref
mismatch detection (mismatch flagged, instance dict mutated, correct match passes,
no-ref skips check, unknown class skips check).

**Test suite:** 373 passed, 2 skipped.

**Remaining Phase 1 gap (minor):** `STATUS_REASONS` closed enum and the three extra
statuses (`error`, `unsupported_query`, `uncertain`) not yet added to the router.
The fast-path legacy module (`region_locator.py`) still returns `"status": "failed"`.
These are Phase 1.3 items deferred to Phase 2 context (they require router changes
that are cleaner to do alongside threshold calibration).

### P7: Fashionpedia 19-Class Part Detector — Baseline + Balanced Training (2026-06-29 / 2026-07-02)

**P7a — Baseline** (2026-06-29): YOLOv8s on all 19 Fashionpedia apparel-part categories.
**P7b — Class-Balanced** (2026-07-02): Power-law image-level repeat (p=1.0, r=12)
training executed.  Model promoted to `models/detectors/`.

**19 part classes** (Fashionpedia IDs 27–45):

| # | Class | # | Class | # | Class |
|---|---:|---|--:|---|--:|---|
| 0 | hood | 7 | buckle | 14 | ribbon |
| 1 | collar | 8 | zipper | 15 | rivet |
| 2 | lapel | 9 | applique | 16 | ruffle |
| 3 | epaulette | 10 | bead | 17 | sequin |
| 4 | sleeve | 11 | bow | 18 | tassel |
| 5 | pocket | 12 | flower | | |
| 6 | neckline | 13 | fringe | | |

**Training data** (from Fashionpedia HuggingFace parquet):

| Split | Images | Images w/ parts | Part annotations |
|---|---:|---:|---:|
| Train | 45,623 | 44,898 | 170,341 |
| Val | 1,158 | 1,150 | 4,093 |

**Severe class imbalance (217:1):**

| Tier | Classes | Total annotations | % of total |
|---|---|---|---|
| Head (3 classes) | sleeve, neckline, pocket | 120,885 | 70.9% |
| Mid (8 classes) | collar, zipper, lapel, bead, rivet, applique, buckle, ruffle | 43,335 | 25.4% |
| Tail (8 classes) | flower, hood, sequin, epaulette, fringe, bow, tassel, ribbon | 6,121 | 3.6% |

**Baseline training results** (YOLOv8s, 100 epochs, no balancing):

- mAP50: ~0.45–0.50, mAP50-95: ~0.25–0.30
- Precision: ~0.52, Recall: ~0.44
- Tail classes (ribbon, tassel, bow, fringe): near-zero recall — confirmed data scarcity
- Confusion matrix shows heavy false-positive bleed from sleeve/neckline into other classes
- Model: `outputs/fashionpedia_19cls_yolov8s/fashionpedia_parts_19cls_yolov8s_best.pt`

**Balanced training results** (YOLOv8s, 100 epochs, p=1.0 r=12, epoch 93 best):

| Metric | Baseline (~epoch 93) | Balanced (p=1.0 r=12) | Δ |
|---|---:|---:|---|
| mAP50 | ~0.47 | 0.312 | -0.158 |
| mAP50-95 | ~0.28 | 0.199 | -0.081 |
| Precision | ~0.52 | 0.435 | -0.085 |
| Recall | ~0.44 | 0.289 | -0.151 |
| Max:min class ratio | 217:1 | 24:1 | 9× improvement |
| Train images (effective) | 32,065 | 43,291 | 1.35× expansion |

**Analysis:** mAP50 dropped ~0.16 as expected — the trade-off for class balancing.
Common classes (sleeve, neckline, pocket) lose precision as the model allocates
capacity to tail classes.  Per-class metrics awaited.

**FashionAI collar_design quick validation** (50 random images, 2026-07-02):

| Class | Detections (50 imgs) | Notes |
|---|---|---|
| sleeve | 108 | Expected — most garments have sleeves |
| pocket | 45 | Moderate |
| collar | 41 | Target class — detected on 82% of collar_design images |
| neckline | 27 | Frequent co-occurrence with collar |
| zipper | 18 | — |
| tail-6 (bow/ribbon/tassel/fringe/sequin/rivet) | 0 | No collar_design images contain these rare parts |

**Model:** `models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt`
**ONNX export:** `models/detectors/fashionpedia_yolov8s_19cls_balanced_best.onnx`
**Visualizations:** `outputs/fashionpedia_balanced_collar_viz/visualizations/` (50 annotated images)
**Script:** `scripts/visualize_fashionpedia_on_fashionai.py`

*Algorithm: Power-law image-level repeat with max-repeat cap.*

1. Count per-class annotation instances across all training images.
2. Set target = median(class_counts).
3. Per-class repeat factor: `repeat_factor[cls] = min(max_repeat, (target / count[cls]) ** power)`.
4. Per-image repeat: `image_repeat = ceil(max(repeat_factor[cls] for cls in image_classes))`.
5. Shuffle the repeated image list and write `train_balanced.txt`.

*Recommended parameters:* `power=1.0, max_repeat=12`

*Rationale:* p=1.0 gives inverse-frequency weighting — ribbon at 274 instances gets
12× repeats (capped), while sleeve at 59,448 gets factor 0.056 (ceil→1, no downsampling).
r=12 prevents extreme overfitting on the smallest classes.

*Parameter sweep results:*

| Power | Max Repeat | Expansion | Before max:min | After max:min |
|:-----:|:----------:|:---------:|:--------------:|:-------------:|
| 0.6 | 12 | 1.18× | 217.0 | 50.9 |
| 0.8 | 12 | 1.26× | 217.0 | 32.9 |
| **1.0** | **12** | **1.34×** | **217.0** | **23.0** |
| 1.0 | 20 | 1.34× | 217.0 | 22.6 |

p=1.0 r=12 chosen: 1.34× expansion is runtime-safe (88.3% of images repeat=1);
r=20 provides negligible improvement over r=12.

**Key limitation:** Rare classes (ribbon, bow) co-occur with common classes
(sleeve, neckline) in the same images, so image-level repeat cannot fully
decouple them.  Instance-level augmentation (copy-paste, mosaic bias) may be
needed as a follow-up if balanced training alone is insufficient.

**Scripts created:**

| Script | Purpose |
|---|---|
| `scripts/convert_fashionpedia_to_yolo.py` | Convert Fashionpedia parquet → YOLO-format dataset |
| `scripts/build_fashionpedia_balanced_train.py` | Power-law balanced train.txt + YAML + CSV reports |
| `scripts/benchmark_fashionpedia_yolo.py` | Inference runtime benchmark → `benchmark_runtime.csv` |

**Outputs per run** (in `E:/fashionpedia_yolo_19cls/`):

```
train_balanced_p1.0_r12.txt                           # 60,148 lines
fashionpedia_parts_balanced_p1.0_r12.yaml             # train → train_balanced_p1.0_r12.txt
balance_report.csv                                    # expansion_ratio, max:min, risk level
class_distribution_before_after.csv                   # per-class before/after counts
repeat_histogram.csv                                  # repeat=1/2/3/... distribution
```

**Relationship to DeepFashion2 YOLO (3.1.1):**

The DeepFashion2 13-class garment detector and Fashionpedia 19-class part detector
serve complementary roles and will coexist:

| | DeepFashion2 YOLO (13 cls) | Fashionpedia YOLO (19 cls) |
|---|---|---|
| **PRD module** | 3.1.1 garment segmentation | 3.1.2 part localization |
| **Detects** | Garment type (top/pants/skirt/outerwear/dress) | Apparel parts (collar/pocket/zipper/…) |
| **Granularity** | Whole-garment | Sub-garment part |
| **Downstream** | SAM-HQ mask → landmark → region crop | Part bbox → SAM-HQ part mask (bypass DINO) |
| **Input** | Full image | Garment crop (YOLO-crop mode) |

Fashionpedia YOLO will be integrated into the 3.1.2 open-vocab path as a
**priority fast detector**: if Fashionpedia has a confident bbox for the
queried part, use it directly; otherwise fall back to Grounding DINO.
Integration is deferred until balanced training results are evaluated.

**Next steps:**
1. ~~Convert full 19-class YOLO dataset~~ ✅ Done (2026-06-29)
2. ~~Run balanced training~~ ✅ Done (2026-07-02): p=1.0 r=12, epoch 93 best
3. Compare balanced vs baseline per-class mAP (pending — needs per-class validation script)
4. Evaluate whether tail-class improvement meets 3.1.2 accuracy thresholds
5. If yes: integrate Fashionpedia YOLO into `region_localization_router.py` open-vocab branch

---

### P5: Stage 6 Integration — Attribute Inference into GarmentPipeline (2026-06-21)

`tools/infer/garment_pipeline.py` extended with optional Stage 6.  No breaking changes.

**New `GarmentPipelineConfig` fields:**

| Field | Default | Description |
|---|---|---|
| `run_attribute_inference` | `False` | Enable Stage 6 |
| `attribute_device` | `"auto"` | Torch device for attribute models |
| `attribute_topk` | `3` | Top-k predictions per task |
| `attribute_inference_config` | `""` | Override `configs/attribute_inference.yaml` |
| `attribute_group_mapping_config` | `""` | Override `configs/attribute_group_mapping.yaml` |

**Stage 6 behaviour:**
- Reads `05_region_masked_crops/region_masked_crops.json` (falls back to stage 4 if absent)
- Runs `GarmentAttributePipeline.predict_from_json()` (lazy-loaded, cached)
- Writes `06_attributes/predictions.jsonl`
- Adds `timing["attribute_seconds"]` and `attributes_summary` to pipeline result

**Usage:**
```python
config = GarmentPipelineConfig(run_attribute_inference=True, attribute_device="auto")
result = GarmentPipeline(config).run_image("image.jpg", "outputs/run")
# result["attributes_summary"] → {"num_instances": N, "num_with_attributes": N, "num_errors": 0}
```

### P3: 3.1.3 Inference Infrastructure and PRD-Facing Flow (2026-06-17)

Config-driven attribute inference pipeline (Steps 1–10) and PRD-facing direct inference
interface (Steps 11–14) are both complete and smoke-tested.

**What was delivered:**

| Item | Details |
|---|---|
| Bug: `coat_length` class filter | Fixed `"outerwear"` → `"outwear"` in `attribute_inference.yaml` and `attribute_group_mapping.yaml` |
| Bug: `det_id` grouping | Fixed integer `det_id=0` falsy bug; key is now `{image_stem}__det{det_id}` (globally unique) |
| Bug: 4 wrong checkpoint paths | Fixed `collar_design`, `neck_design`, `lapel_design`, `coat_length` pointing to non-existent `multiview_v2_pipeline` dirs |
| `garment_attribute_pipeline.py` | End-to-end: `region_crops.json` → per-instance `{task: {label, score, topk}}` |
| `mask_attribute_pipeline.py` | PRD-facing: `image + binary mask + garment_category + component_type → attribute labels + confidence` |
| `scripts/run_attribute_from_mask_smoke.py` | CLI wrapper for the PRD-facing interface |
| `scripts/visualize_attribute_pipeline_output.py` | Contact-sheet visualizer for pipeline JSONL output |
| Tests | 283 passed, 2 skipped, 0 failed (7 test files: 47 new in `test_mask_attribute_pipeline.py`) |

**Smoke test result** (real assets, CPU, ~0.6 s):

```
Image : assets/random_train60/images/000004.jpg
Mask  : outputs/test_pipeline_smoke/02_samhq/masks/000004_det000_long sleeve top_mask.png
Input : garment_category=top, component_type=collar

neckline_design → Straight Neck  (0.403)
collar_design   → Invisible      (0.995)
neck_design     → Invisible      (0.467)
```

Artifacts: `outputs/smoke_test_attr_from_mask/` — `predictions.json`, `predictions.jsonl`,
`000004_collar_masked_crop.jpg`, `000004_collar_raw_crop.jpg`, `000004_collar_overlay.jpg`.

**Smoke test command:**

```bash
python scripts/run_attribute_from_mask_smoke.py \
    --image "assets/random_train60/images/000004.jpg" \
    --mask "outputs/test_pipeline_smoke/02_samhq/masks/000004_det000_long sleeve top_mask.png" \
    --garment-category top --component-type collar \
    --output-dir outputs/smoke_test_attr_from_mask --device cpu --topk 3
```

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

## 5. Work Plan (updated 2026-06-24)

### 5.0 3.1.2 Industrial Grounding — Phase 1 ✅ COMPLETE (2026-06-24)

All Phase 1 deterministic fixes shipped.  See P6 milestone above.

Next focus for 3.1.2 (Phase 2 of the industrial plan):

1. **Validation set creation** — annotate 50–100 images in Label Studio (5 primary
   parts: pocket, zipper, button_cluster, drawstring, collar). 2–4 hours annotation.
2. **Threshold calibration** — run `scripts/calibrate_part_thresholds.py` sweeping
   box_threshold 0.20→0.55; choose threshold where recall ≥ 0.50.
3. **DINO-tiny vs DINO-base comparison** — compare model quality on the validation
   set before deciding on fine-tuning (Phase 6 trigger condition).
4. **Left/right convention verification** — test whether `left_sleeve` is image-left
   or person-anatomical-left on real images.
5. **Fast-path regression on Batch60** — run `run_open_vocab_yolo_crop_test.py` and
   record per-query detected/low_conf/no_det counts as the numeric baseline.

### 5.1 Detection Direction (3.1.1)

- Compile complete experiment record for YOLOv8n balanced retraining (done above).
- Deep-dive analysis on outerwear and dress — the two weakest classes after balancing.
- Research data sources for shoes, bags, accessories, and pattern/print categories.
- Evaluate SAM-HQ segmentation IoU accuracy on a held-out sample set.

### 5.2 Attribute Inference Direction (3.1.3)

- Run `scripts/run_all_attribute_eval.py` on more diverse samples to verify pipeline stability.
- Prepare multi-sample test set for statistical attribute prediction evaluation.
- Investigate external attribute datasets beyond FashionAI:
  - **DeepFashion Attribute Prediction** (289K images, 1000 attributes — label mapping needed)
  - **iMaterialist Fashion 2019** (1M images, 228 fine-grained attributes — most relevant)
  - **Clothing Attributes Dataset** (Bossard et al., 26 attributes, small but clean)
- Map external dataset labels to the current 8 FashionAI task schema.
- Root cause confirmed: data scarcity (556–1647 train samples across 5–8 classes per task).
  Primary mitigation direction: FashionCLIP zero-shot / few-shot feature extraction.

### 5.3 Performance Optimization Direction

- Research lightweight SAM alternatives (current: 420 ms/image, PRD target: ≤ 50 ms):
  - **FastSAM** (YOLOv8-based, ~50× speedup)
  - **MobileSAM** (distilled ViT-Tiny encoder)
  - **EfficientSAM** (SAMI distillation)
  - **SAM2** (Meta, 2024 — improved speed/accuracy tradeoff)
- No replacement yet — research and benchmark only.
- **Runtime tracking now required** — every training run must record total wall time,
  per-epoch average, and balanced-list expansion factor.  Inference benchmark
  (single + batch throughput) also required.  See `scripts/benchmark_fashionpedia_yolo.py`.

### 5.4 Fashionpedia Part Detection (3.1.2 supplement)

- Baseline YOLOv8s 19-class trained (2026-06-29).  mAP50 ~0.45–0.50.  Tail classes
  near-zero recall due to 217:1 class imbalance.
- **Class-balanced training:**
  - Algorithm: power-law image repeat with max-repeat cap (see P7).
  - Chosen params: `power=1.0, max_repeat=12` — 1.34× expansion, 217→23× max:min.
  - Scripts: `convert_fashionpedia_to_yolo.py` (dataset), `build_fashionpedia_balanced_train.py`
    (sampling + YAML + CSVs), `benchmark_fashionpedia_yolo.py` (runtime).
  - **Awaiting server GPU run.**
- **Post-training evaluation:**
  - Per-class mAP comparison (baseline vs balanced).
  - Confusion matrix diff.
  - Tail-class recall threshold: aim for ≥ 0.15 (up from ~0.00).
- **Integration decision:** If balanced model meets per-part accuracy thresholds,
  wire Fashionpedia YOLO into `region_localization_router.py` as the priority
  fast detector (before DINO fallback) for the 19 supported part types.
- **If balanced training is insufficient:** Instance-level augmentation
  (copy-paste rare parts, mosaic with rare-class bias), or class-weighted loss.

---

## 6. Open Gaps

| Gap | Priority | Notes |
|---|---|---|
| **3.1.2 Fashionpedia balanced training** | **High** | Baseline done; balanced scripts ready; pending server GPU execution. See Section 10 (P7). |
| **3.1.2 threshold calibration** | **High** | Phase 1 complete. Phase 2 next: validation set annotation + threshold sweep. No annotation data yet. |
| **3.1.2 `STATUS_REASONS` + `error`/`unsupported_query`/`uncertain`** | Medium | Minor Phase 1.3 deferral; `region_locator.py` fast-path still returns `"failed"`. Deferred to Phase 2 context. |
| 3.1.3 accuracy (macro-F1 ≥ 0.88) | High | Root cause: data scarcity. Direction: iMaterialist + FashionCLIP |
| 3.1.1 latency (≤ 50 ms) | High | SAM-HQ is 293 ms/image. FastSAM/MobileSAM research pending |
| 3.1.3 latency (≤ 20 ms/task) | Medium | `benchmark_attribute_latency.py` not yet run on real hardware |
| Attribute coverage (8/14 groups) | Medium | Fabric and craftsmanship attributes absent |
| SAM-HQ IoU evaluation | Medium | No systematic IoU measurement yet |
| Left/right convention verify | Medium | `left_sleeve` label — image-left vs person-anatomical-left unverified |
| Shoes / bags / accessories | Low | No dataset; deferred |
| Stage 6 multi-sample eval | Low | Only smoke-tested on 1 image |

---

## 7. Known Technical Debt

| Issue | File(s) | Priority |
|---|---|---|
| ~~Duplicate query parsing logic~~ | ~~`query_region_online_demo.py` vs `query_parser.py`~~ | **FIXED (2026-06-23)** — merged into `intent_parser.py` |
| `region_locator.py` still returns `"failed"` status | Fast-path legacy module | Low — map to `not_detected`/`error` in Phase 2 context |
| `STATUS_REASONS` closed enum not yet in router | Phase 1.3 minor deferral | Low |
| Two `region_visualizer.py` files | `localization/` and `visualization/` | Low |
| Utility helpers duplicated | `sanitize_filename_part`, `load_json` | Low |
| Script sprawl in `scripts/` | `scripts/` (~35 files) | Low |
| Fashionpedia 19-class dataset not yet converted | `convert_fashionpedia_to_yolo.py` needs to run with 19 `--cats` | Medium — one-time conversion, ~30–60 min |

---

## 8. Out of Scope (Do Not Advance Now)

- Shoes, bags, accessories (no dataset)
- 3.2 Multimodal QA
- 3.3 Agent/RAG
- Large-scale SAM-HQ replacement (research only this week)
- Large-scale refactors

---

## 9. 3.1.2 Open-Vocabulary Upgrade — COMPLETE (2026-06-23)

### 9.0 What Was Built

The P1 prototype was a **closed-vocabulary fixed-part system with NLP front-end**
(6 region types, no side queries, no sub-components).  Phases 0–4 upgrade it to a
true language-guided open-vocabulary localizer.

### 9.1 System Architecture (3-layer pipeline — no existing code replaced)

```
Image
  ↓
[3.1.1] YOLO → SAM-HQ          ← unchanged; produces garment-level masks
  ↓ garment_mask per instance
[3.1.2 fast path]               ← unchanged; landmark + geometry for 6 standard regions
  Landmark → 6-class region       (no side query, no sub-components)
  ↓  OR  ↓
[3.1.2 open-vocab path — NEW]
  Intent Parser                 ← rule-based: {part, side, garment_ref}
  → Spatial constraint          ← geometric mask split for left/right
  → Grounding DINO (HF)         ← open-vocab detection; no training needed
  → SAM-HQ (reused)             ← bbox → precise part mask
  ↓ region_mask
[3.1.3] Attribute classifiers   ← unchanged; ResNet18 on region crop
```

**Key point:** SAM-HQ is reused at two levels: garment mask (Stage 2) and part
mask (Stage 6).  YOLO, SAM-HQ, and landmark code are NOT deprecated.

### 9.2 Environment

- **No Linux / rental server required** for inference.
- Grounding DINO via HuggingFace `transformers >= 4.40` — pure PyTorch, no mmcv.
- Confirmed compatible with current env: PyTorch 2.5.1 + CUDA 12.1 + SAM-HQ.
- Install: `pip install transformers>=4.40.0`
- Checkpoint: `IDEA-Research/grounding-dino-tiny` (HuggingFace Hub)

### 9.3 Data Requirements

| Component | DeepFashion2 needed? | Notes |
|---|---|---|
| YOLO garment detection | ✅ (done) | P4 model promoted |
| Landmark predictor | ✅ (done) | Fast path for 6 standard regions |
| Attribute classifiers | ✅ (in progress) | P2 done; accuracy gap remains |
| Grounding DINO | ❌ not needed | Uses own pretrained weights; zero-shot |

### 9.4 Phase Summary

| Phase | Status | Deliverable |
|---|---|---|
| **0** | ✅ DONE (2026-06-23) | `intent_parser.py` — unified `QueryIntent` dataclass, `parse_intent()` |
| **1** | ✅ DONE (2026-06-23) | 13-part PART_VOCAB, side-word extraction, `is_fast_path` routing |
| **2** | ✅ DONE (2026-06-23) | `grounding_dino_locator.py` — DINO via HF, multi-prompt, NMS, area filter |
| **3** | ✅ DONE (2026-06-23) | `spatial_constraint.py` — x-center reranking for left/right side queries |
| **4** | ✅ DONE (2026-06-23) | `region_localization_router.py` — unified entry point; demo updated |

### 9.5 Files Created / Modified

```
src/fashion_vision/localization/intent_parser.py           CREATED
    QueryIntent: {part, side, garment_ref, is_fast_path}
    Properties: crop_region, component, grounding_text
    PART_VOCAB: 13 parts (6 fast-path + 7 open-vocab)
    Global longest-match search (fixes "腰带" vs "腰" ambiguity)

src/fashion_vision/localization/grounding_dino_locator.py  CREATED
    GroundingDINOLocator — HF IDEA-Research/grounding-dino-tiny
    detect(): area filter (min_bbox_area_ratio=0.003), trailing period auto-appended
    detect_multi_prompt(): per-prompt DINO + greedy NMS (IoU > 0.5)
    Garment-context prompts: "clothing zipper", "clothing pocket", "clothing belt" etc.

src/fashion_vision/localization/spatial_constraint.py      CREATED
    select_side_detection(): reranks detections by bbox x-center for left/right

src/fashion_vision/localization/region_localization_router.py  CREATED
    locate_region(query, instance, image, ...) — single entry point
    fast path → locate_region_from_instance() (unchanged)
    open-vocab path → detect_multi_prompt() → select_side_detection()

src/fashion_vision/localization/open_vocab_prompt_map.py   MODIFIED
    Removed "waist belt" from belt prompts (triggered waist-region false positives)
    Reordered placket prompts: ["front placket", "shirt placket", "front opening of coat"]

src/fashion_vision/localization/query_parser.py            REWRITTEN (thin wrapper)
    parse_region_type() delegates to parse_intent().part
    is_supported_region() checks FAST_PATH_PARTS

tools/demo/query_region_online_demo.py                     MODIFIED
    Removed infer_region_from_query(), REGION_ALIASES etc.
    Added open-vocab branch using region_localization_router
    Fixed --reuse-pipeline-dir: added missing segmentation_json key

scripts/run_collar_visual_test.py                          CREATED
    Fast-path visual test: 100 images from collar_design_labels
    Output: outputs/visual_tests/collar_design_sample100/

scripts/run_open_vocab_visual_test.py                      CREATED
    Full-image DINO test: 50 images from coat_length_labels, 5 queries
    Two-level overlay: accept (coloured) + watermark (grey)
    Output: outputs/visual_tests/open_vocab_coat_length_sample50/

scripts/run_open_vocab_yolo_crop_test.py                   CREATED
    YOLO-crop mode: YOLO Stage1 → garment crops → DINO inside crops → map back
    Per-query config: prompts, accept_threshold, min/max_crop_area_ratio
    Status: detected / low_confidence / no_detection / error
    Output: outputs/visual_tests/open_vocab_coat_length_sample50_yolo_crop/
    HTML: 7-cell strip per image (original | yolo_boxes | 5 query overlays)
```

### 9.6 Bug Fixes

**Bug 1: `is_fast_path=False` for sided fast-path queries**
- "左边袖口" was routing to DINO instead of landmark pipeline
- Old: `is_fast = (part in FAST_PATH_PARTS) and (side is None)`
- Fix: `is_fast = part in FAST_PATH_PARTS`

**Bug 2: "腰带" parsed as "waist" instead of "belt"**
- First-match search: "腰" (1 char) matched before "腰带" (2 chars) was checked
- Fix: global longest-match across ALL parts in PART_VOCAB

**Bug 3: `--reuse-pipeline-dir` crash on open-vocab queries**
- Missing `"segmentation_json"` key in `pipeline_result["paths"]`
- Fix: added `"segmentation_json": str(pipeline_dir / "02_samhq" / "segmentation_results.json")`

### 9.7 Capability / Gap Analysis

**Can do now:**

| Capability | How |
|---|---|
| 领口 / 下摆 / 腰部 / 肩部 / 裤腿 | Fast path (landmark + geometry), stable |
| 袖口（含左边/右边修饰） | Fast path + `left_sleeve`/`right_sleeve` component filter |
| 拉链 / 口袋 / 扣子 / 腰带 / 门襟 | DINO open-vocab path (YOLO-crop mode recommended) |
| 左边/右边修饰 + 口袋/拉链等 | DINO + post-hoc x-center reranking |
| 碎花图案 / 印花 | DINO `fabric pattern` prompt |

**Cannot do now:**

| Gap | Root Cause | Status |
|---|---|---|
| **外套 vs 内搭区分** — mismatch flag | `intent.garment_ref` vs instance class | ✅ **DONE (Phase 1.5, 2026-06-24)** — `garment_ref_matched=False` in result when mismatch detected |
| **外套 vs 内搭区分** — instance selection | `filter_instances()` not wired into `locate_region()` | ⚠️ Flag set; full routing (passing only matched instances to DINO) is Phase 4 |
| 关系型查询 ("外套和内搭的接缝") | DINO has no cross-instance reasoning | Hard — Phase 4 |
| Left/right convention correctness (fast path) | `left_sleeve` label convention unverified (person-anatomical vs image-left?) | Medium — needs empirical test |
| High accuracy for placket / zipper / belt | DINO still noisy even with YOLO crop; thresholds not calibrated | Phase 2 (validation + threshold sweep) |
| Queries outside 18-part vocab | Must manually register in PART_VOCAB + PART_DETECTION_CONFIG | Low per part |

**Most impactful next step:** Phase 2 validation set creation (Label Studio annotation
of 50–100 images) and threshold calibration sweep.

### 9.8 Validation Targets

| Test | Sample | Target |
|---|---|---|
| T1: Fast-path regression | "领口", "下摆", "腰部" — Batch60 | ≥ 92% (no regression) |
| T2: Side queries | "左边袖口", "右肩" | ≥ 75% baseline |
| T3: Component queries | "拉链", "口袋", "扣子" | ≥ 60% baseline (measure from visual test output) |
| T4: Compound + garment_ref | "外套左边的口袋" | ≥ 50% (needs garment_ref fix first) |

---

## 10. Fashionpedia Part Detection — Methodology & Run-book

> **Detailed milestone:** See P7 in Section 2.
> **Related scripts:** `scripts/convert_fashionpedia_to_yolo.py`,
> `scripts/build_fashionpedia_balanced_train.py`,
> `scripts/benchmark_fashionpedia_yolo.py`.

### 10.1 Architecture Role

Fashionpedia YOLO is a **dedicated part detector** positioned as a priority fast
path within the 3.1.2 open-vocab localization branch.  It does NOT replace
DeepFashion2 YOLO (garment detection) or Grounding DINO (open-vocab fallback).

```
Image
  → [3.1.1] DeepFashion2 YOLO → SAM-HQ → garment mask
  → [3.1.2 open-vocab path]
      ├─ Fashionpedia YOLO  (priority fast: 19 known part types)  ← NEW
      └─ Grounding DINO      (fallback: open-vocab, any part)
  → SAM-HQ → part mask → [3.1.3] attribute classifier
```

### 10.2 Data Pipeline

```
E:/fashionpedia/data/*.parquet          ← HuggingFace download
    │  convert_fashionpedia_to_yolo.py
    ▼
E:/fashionpedia_yolo_19cls/
    ├── images/train/  (*.jpg, 44,898 files)
    ├── images/val/    (*.jpg,  1,150 files)
    ├── labels/train/  (*.txt, YOLO format)
    ├── labels/val/
    └── fashionpedia_parts.yaml          ← nc: 19, names: [...]
    │  build_fashionpedia_balanced_train.py
    ▼
    ├── train_balanced_p1.0_r12.txt     ← 60,148 lines (1.34×)
    ├── fashionpedia_parts_balanced_p1.0_r12.yaml
    ├── balance_report.csv
    ├── class_distribution_before_after.csv
    └── repeat_histogram.csv
```

### 10.3 Balanced Sampling Algorithm

**Purpose:** Mitigate 217:1 class imbalance (sleeve: 59,448 vs ribbon: 274).

**Method:** Power-law image-level repeat with max-repeat cap.  Images containing
rare classes are duplicated in the training list; common-class images are not
downsampled.

```
Input:
    power (p)        = 1.0    # 0=uniform, 1=inverse frequency
    max_repeat (r)   = 12     # cap to prevent overfitting on tiny classes
    seed             = 42     # deterministic shuffle

Algorithm:
    1. target := median(class_annotation_counts)
    2. For each class c:
         repeat_factor[c] := min(r, (target / count[c]) ** p)
    3. For each image i:
         image_repeat[i] := ceil(max(repeat_factor[c] for c in classes_in_image[i]))
    4. Write each image path image_repeat[i] times to train_balanced.txt
    5. Shuffle with seed
```

**Why p=1.0 r=12 (chosen over alternatives):**

| p | r | Expansion | After max:min | Notes |
|:--:|:--:|:---------:|:-------------:|-------|
| 0.6 | 12 | 1.18× | 50.9 | Too weak — tail boost insufficient |
| 0.8 | 12 | 1.26× | 32.9 | Moderate |
| **1.0** | **12** | **1.34×** | **23.0** | **Best trade-off** |
| 1.0 | 20 | 1.34× | 22.6 | r=20 adds no benefit (ribbon already capped at 12) |

**Known limitation:** Rare parts co-occur with common parts in the same garment
image (e.g., a "ribbon" almost always appears on an image that also has "sleeve"
or "neckline").  Image-level repeat therefore cannot fully isolate rare classes.
If balanced training results are insufficient, follow-ups include:
- Instance-level copy-paste augmentation for tail classes.
- Class-weighted loss (higher weight for rare classes).
- Mosaic augmentation with rare-class bias.

### 10.4 Run Commands

Steps 1–5 executed (2026-06-29 to 2026-07-02).  Final model at `models/detectors/`.

```bash
conda activate fashion-demo2

# Step 1: Convert 19-class YOLO dataset (one-time, ~30–60 min)
python scripts/convert_fashionpedia_to_yolo.py \
    --src E:/fashionpedia \
    --dst E:/fashionpedia_yolo_19cls \
    --cats hood collar lapel epaulette sleeve pocket neckline buckle zipper \
          applique bead bow flower fringe ribbon rivet ruffle sequin tassel

# Step 2: Dry-run preview (seconds)
python scripts/build_fashionpedia_balanced_train.py \
    --labels-dir E:/fashionpedia_yolo_19cls/labels/train \
    --images-dir E:/fashionpedia_yolo_19cls/images/train \
    --base-yaml E:/fashionpedia_yolo_19cls/fashionpedia_parts.yaml \
    --out-dir E:/fashionpedia_yolo_19cls \
    --power 1.0 --max-repeat 12 --dry-run

# Step 3: Generate balanced train list + YAML + CSVs
python scripts/build_fashionpedia_balanced_train.py \
    --labels-dir E:/fashionpedia_yolo_19cls/labels/train \
    --images-dir E:/fashionpedia_yolo_19cls/images/train \
    --base-yaml E:/fashionpedia_yolo_19cls/fashionpedia_parts.yaml \
    --out-dir E:/fashionpedia_yolo_19cls \
    --power 1.0 --max-repeat 12 --seed 42

# Step 4: Balanced training (server GPU) — DONE 2026-07-02
yolo detect train \
    data=E:/fashionpedia_yolo_19cls/fashionpedia_parts_balanced_p1.0_r12.yaml \
    model=yolov8s.pt \
    epochs=100 imgsz=640 batch=16 device=0 \
    project=outputs/fashionpedia_19cls_yolov8s_balanced \
    name=p1.0_r12

# Step 4b: Baseline training (control group) — DONE 2026-06-29
yolo detect train \
    data=E:/fashionpedia_yolo_19cls/fashionpedia_parts.yaml \
    model=yolov8s.pt \
    epochs=100 imgsz=640 batch=16 device=0 \
    project=outputs/fashionpedia_19cls_yolov8s_baseline \
    name=baseline

# Step 5: Quick validation on FashionAI test images
python scripts/visualize_fashionpedia_on_fashionai.py \
    --image-dir D:/Aliintern/fashion-ai-data/fashionai_attributes/round1_fashionAI_attributes_test_a/Images/collar_design_labels \
    --model models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt \
    --num-samples 50 --output-dir outputs/fashionpedia_balanced_collar_viz --device cuda
```

### 10.5 Evaluation Criteria

| Metric | Baseline (current) | Balanced target | Measurement |
|---|---|---|---|
| mAP50 (all 19 cls) | ~0.45–0.50 | ≥ 0.50 | `results.csv` from YOLO train output |
| Tail-class recall (ribbon, bow, tassel) | ~0.00 | ≥ 0.15 | Per-class AP from confusion matrix |
| Expansion ratio | 1.00× | 1.34× | `balance_report.csv` |
| Single-image inference | TBD | ≤ 10 ms (GPU) | `benchmark_runtime.csv` |
| Total train wall time | ~3–5 h (est.) | ~4–7 h (est.) | Record from server log |

### 10.6 Integration Trigger

Fashionpedia YOLO integration into the 3.1.2 pipeline (`region_localization_router.py`)
is gated on:

1. Balanced model tail-class recall ≥ 0.15 for ≥ 12 of 19 classes.
2. No regression on head classes (sleeve, neckline, pocket) vs baseline.
3. Inference latency ≤ 10 ms on target GPU.

Until all three conditions are met, the current DINO-only open-vocab path
remains the production path for 3.1.2 part queries.
