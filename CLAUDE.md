# Claude Code Instructions

## 1. High-level Rule

Do not modify code until explicitly asked.

The first task is project understanding and codebase audit only.

Before editing any file, always:

1. Explain what you plan to inspect.
2. Explain what files you plan to modify.
3. Wait for confirmation.

---

## 2. Token and File Reading Restrictions

This repository is connected to large datasets and generated outputs.

Do NOT recursively inspect or summarize the following directories unless explicitly requested:

```text
D:\Aliintern\fashion-ai-data
outputs/
runs/
wandb/
logs/
checkpoints/
models/
weights/
visualizations/
crops/
masks/
overlays/
```

Do NOT open:

- image files
- crop files
- mask files
- visualization files
- model weights
- ONNX/TensorRT files
- large JSONL files
- pickle files
- numpy array files

Allowed initial inspection:

- directory tree up to depth 2 or 3
- Python source files under tools/, scripts/, src/ if they exist
- config files under configs/
- markdown reports under docs/ or project root
- small JSON summary files only if explicitly requested

If a large output directory is important, inspect only:

- filenames
- directory names
- small summary JSON/MD files

Do not inspect every output artifact.

Before reading any file under outputs/, ask for permission and explain why.

---

## 3. Current Project Goal

This project is a fashion fine-grained visual analysis system aligned with the PRD.

Main PRD modules:

- 3.1.1 Garment instance segmentation
- 3.1.2 Language-guided local region localization
- 3.1.3 Fine-grained attribute extraction

Current priority:

- Continue optimizing 3.1.1 and 3.1.2.
- 3.1.3 progress can be summarized, but do not advance it for now.
- Shoes, bags, and accessories are not implemented yet and should not be developed now.
- Runtime performance is not the first priority at this stage.

---

## 4. Current Technical Strategy

DeepFashion2 has 13 garment classes.

The detector is trained as 13 classes and should keep 13-class outputs internally.

However, PRD 3.1.1 requires coarse garment categories.

Current DeepFashion2-supported PRD-facing classes are 5 classes:

```text
top
pants
skirt
outerwear
dress
```

Important rule:

- External 3.1.1 output and evaluation should use 5 classes.
- Internal pipeline should keep 13-class fine category because landmarks and local region localization depend on it.
- Do not replace internal 13-class logic with 5-class logic.
- Add dual-label fields when needed:
  - fine_class_id
  - fine_class_name
  - coarse_class_id
  - coarse_class_name

---

## 5. Category Mapping Policy

Use a config file for mapping.

Do not hard-code category mapping inside business logic.

DeepFashion2 13-class ids:

```text
0 short sleeve top
1 long sleeve top
2 short sleeve outwear
3 long sleeve outwear
4 vest
5 sling
6 shorts
7 trousers
8 skirt
9 short sleeve dress
10 long sleeve dress
11 vest dress
12 sling dress
```

13-to-5 mapping:

```yaml
0: 0   # short sleeve top -> top
1: 0   # long sleeve top -> top
2: 3   # short sleeve outwear -> outerwear
3: 3   # long sleeve outwear -> outerwear
4: 0   # vest -> top
5: 0   # sling -> top
6: 1   # shorts -> pants
7: 1   # trousers -> pants
8: 2   # skirt -> skirt
9: 4   # short sleeve dress -> dress
10: 4  # long sleeve dress -> dress
11: 4  # vest dress -> dress
12: 4  # sling dress -> dress
```

Coarse class ids:

```text
0 top
1 pants
2 skirt
3 outerwear
4 dress
```

---

## 6. Code Quality Requirements

All new code and refactoring must follow:

1. Modular design.
2. Use configuration files instead of hard-coded parameters.
3. Add exception handling for expected file/data errors.
4. Add docstrings for functions/classes.
5. Use type hints.
6. Avoid duplicated code.
7. Follow PEP8 naming and formatting.
8. Preserve backward compatibility.
9. Prefer small, testable functions.
10. Do not rewrite working pipeline unless explicitly requested.

---

## 7. Immediate First Task

The first task is NOT implementation.

First produce a project audit report:

1. What code modules currently exist?
2. What pipeline stages are already implemented?
3. Which scripts correspond to:
   - 3.1.1 garment detection / segmentation
   - 3.1.2 query-region localization
   - 3.1.3 attribute classification
4. What methods are used?
5. What progress has been achieved?
6. What duplicated or overlapping code exists?
7. Which output directories appear to be generated artifacts and can be ignored or archived?
8. What is the minimal next coding task to support 13-class internal / 5-class external output?

Do not modify files during this first task.

---

## 8. Project Memory — Current Engineering State

> Last updated: 2026-06-30

**Full details:** `docs/current_project_status.md`

### Completed milestones (P1–P7)

- **P1** (3.1.1 + 3.1.2): YOLO + SAM-HQ + landmark + region crop pipeline functional.
  Rule-based Chinese query-to-region demo: 92% valid response on Batch60.
- **P2** (3.1.3 training): All 8 FashionAI attribute tasks trained (ResNet18).
  Best test macro-F1: collar_design 0.764.  PRD target 0.88 — gap confirmed.
- **P3** (3.1.3 inference infra): Config-driven pipeline, PRD-facing interface, 283 tests passing.
- **P4** (YOLO balanced retraining): All 5 PRD classes improved recall + precision.
  Model promoted to `models/detectors/yolov8n_deepfashion2_13cls_best.pt`.
  Evaluation artifacts: `outputs/yolo_eval_balanced/`, `outputs/yolo_eval_baseline/`.
- **P5** (Stage 6 integration): `GarmentPipelineConfig.run_attribute_inference=True` wires
  Stage 6 into the end-to-end pipeline.  `tools/infer/garment_pipeline.py` modified.
- **P6** (3.1.2 industrial grounding Phase 1): All Phase 1 deterministic fixes complete.
  Key deliverables: `_crop_image_and_mask()` for mask-gated DINO; shape priors returns `[]`
  on all-rejected; `not_detected` status with reason; 18-part vocab (6 new long-tail terms);
  `_flag_garment_ref_mismatch()` in router; 6-panel debug viz script.
  New test file: `tests/test_router_helpers.py`.  **Total: 373 passed, 2 skipped.**
- **P7** (Fashionpedia 19-class part detector baseline + balanced, 2026-06-29 / 2026-07-02):
  Baseline YOLOv8s trained on all 19 Fashionpedia apparel-part categories: mAP50 ~0.47.
  Balanced training (p=1.0, r=12) executed 2026-07-02: mAP50 0.312, max:min ratio 217:1→24:1.
  Collar validation on 50 FashionAI images: 100% detection rate, 82% collar recall.
  Model: `models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt`.

### Active implementation plan

`docs/industrial_grounding_implementation_plan.md` is the current engineering guide.
- **Phase 1**: COMPLETE (2026-06-24) — see P6 above.
- **Phase 2**: NEXT — validation set annotation (Label Studio) + threshold calibration sweep.
  Requires 2–4 hours human annotation of 50–100 images.
- **Phase 3**: Qwen-VL-7B-Chat translation service (requires GPU server rental).
- **Phase 4**: Garment hierarchy + mask containment (after SAM-HQ mask quality verified).
- **Phase 5**: Composite anchor routing.
- **Phase 6**: DINO fine-tuning — BLOCKED until Phase 2 calibration results confirm need.

**Fashionpedia parallel track** (3.1.2 part detector supplement):
- **Fashionpedia P0** ✅: Baseline 19-class YOLOv8s trained (2026-06-29).
- **Fashionpedia P1** (NEXT): Execute balanced training (p=1.0, r=12) on server GPU.
- **Fashionpedia P2**: Evaluate per-class mAP; decide integration trigger.
- **Fashionpedia P3**: Wire into `region_localization_router.py` open-vocab branch
  as priority fast detector (before DINO fallback) — GATED on P2 results.
- See `docs/current_project_status.md` Section 10 for full run-book.

### Current accuracy gap root cause (confirmed)

3.1.3 tasks have 556–1647 train samples across 5–8 classes.
This is the primary cause of low macro-F1 (0.593–0.764).  Backbone size is not the
bottleneck — data quantity is.  Mitigation direction: iMaterialist Fashion 2019 +
FashionCLIP feature extraction.

### Runtime tracking (new requirement, 2026-06-30)

Every training run must record and report:
- Total wall-clock training time.
- Per-epoch average time.
- Balanced-list expansion factor (if applicable).
- Inference benchmark: single-image latency (mean/std/P50/P95) + batch throughput.
- Script: `scripts/benchmark_fashionpedia_yolo.py` → `benchmark_runtime.csv`.

### Safety Constraints (always active)

- Do not scan `outputs/`, `D:\Aliintern\fashion-ai-data`, or any model/weight/log directories.
- Do not run YOLO/SAM-HQ/training/full inference without explicit approval.
- Do not commit without explicit approval.
- Do not implement new features before reading the summary doc and confirming the plan.
