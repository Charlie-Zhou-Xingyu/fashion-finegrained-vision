# Codebase Cleanup Plan

> Created: 2026-06-16
> Status: Stage 1 complete; Stages 2–3 deferred to post-retraining.

---

## Stage 1 — Safe immediate changes (complete, no behavior change)

These changes are additive or documentation-only.  Merged before YOLO
balanced retraining.

| # | File | Change |
|---|---|---|
| 1 | `tools/infer/predict_garments_yolo.py` | Add dual-label fields (`fine_class_id`, `fine_class_name`, `coarse_class_id`, `coarse_class_name`) to every detection dict; load mapping from `configs/category_mapping.yaml` |
| 2 | `src/fashion_vision/schemas/instance_schema.py` | Add 4 optional dual-label params to `build_instance_record()`; backward-compatible |
| 3 | `src/fashion_vision/data/class_mapping.py` | Extend module docstring with scope/ID-convention warning; no functional change |
| 4 | `src/fashion_vision/localization/hybrid_region_policy.py` | Replace empty file with module docstring and `NotImplementedError` stub |
| 5 | `docs/current_project_status.md` | New document: current project state snapshot |
| 6 | `docs/codebase_cleanup_plan.md` | This file |
| 7 | `docs/yolo_balanced_training_plan.md` | New document: retraining commands and eval plan |
| 8 | `PROJECT_CONTEXT.md` | Update stale sections (P2 status, next task) |

---

## Stage 2 — Medium-risk refactors (post-retraining, require tests)

Do not attempt before retraining is complete and post-train evaluation is
confirmed working.

### 2a. Consolidate query parsing into canonical module

**Files:** `src/fashion_vision/localization/query_parser.py` (extend),
`tools/demo/query_region_online_demo.py` (simplify).

The demo maintains its own `REGION_ALIASES`, `SPECIAL_QUERY_ALIASES`, and
`COMPONENT_ALIASES` dicts plus a separate `infer_region_from_query()`
function.  These should be moved into `query_parser.py` and imported by
the demo.

**Prerequisite:** Run Batch60 regression test after change.
**Risk:** Medium — alias lists differ slightly; merging could change
behaviour for edge cases.

### 2b. Consolidate region_visualizer.py

**Files:** `src/fashion_vision/localization/region_visualizer.py`,
`src/fashion_vision/visualization/region_visualizer.py`.

Audit both files; merge into `visualization/region_visualizer.py` and
update imports.  `localization/` should not contain visualization logic.

**Prerequisite:** Full grep of importers before touching.
**Risk:** Low-medium — depends on which file is used where.

### 2c. Extract duplicated utility helpers

**Files:** `tools/infer/predict_garments_yolo.py`,
`tools/demo/query_region_online_demo.py`,
`tools/infer/garment_pipeline.py`.

`sanitize_filename_part()` is copy-pasted in at least two files.
`load_json()`/`save_json()` helpers are duplicated across pipeline files
rather than imported from `src/fashion_vision/utils/json_io.py`.

Move to canonical `src/fashion_vision/utils/` modules and update callers.

**Risk:** Low — mechanical change, but touches multiple files.

---

## Stage 3 — High-risk changes (postpone indefinitely)

These carry structural risk and are not required for retraining or PRD
delivery.  Revisit only after post-retraining evaluation is stable.

### 3a. Retire or archive `src/fashion_vision/data/class_mapping.py`

Currently used by `deepfashion2_parser.py` (1-based annotation IDs) and
tested in `tests/test_class_mapping.py`.  Any retirement requires:
- Full caller audit (`grep -r class_mapping`)
- Deciding what happens to the 8-class GT parsing path
- Updating or removing the existing unit tests

**Do not attempt before a full design decision on the annotation-parsing path.**

### 3b. Restructure `scripts/` directory

26 standalone scripts with overlapping concerns.  Reorganizing would
break `.bat` training commands that reference them by path.

### 3c. Archive `archive/old_experiments/`

OWL-ViT, SAM prompting, and local region baseline experiments are already
in `archive/`.  Moving or deleting them is low-value noise before
retraining.

### 3d. Build end-to-end eval script for confusion matrix generation

`tools/eval/eval_13cls_confusion_as_5cls.py` requires a pre-generated
14×14 confusion matrix JSON.  Building the script that runs YOLO `val`
and converts its output to the expected format is a new feature, not a
cleanup.  Required for post-retraining evaluation.

---

## Canonical modules to preserve

| Module | File | Role |
|---|---|---|
| Category mapping config | `configs/category_mapping.yaml` | Single source of truth for 13→5 mapping |
| Category mapping loader | `tools/eval/category_mapping.py` | Validated `CategoryMapping` dataclass |
| Confusion aggregation | `tools/eval/confusion_aggregation.py` | 14×14 → 5×5 aggregation |
| 5-class metrics | `tools/eval/detection_metrics.py` | Per-class precision/recall |
| Eval CLI | `tools/eval/eval_13cls_confusion_as_5cls.py` | PRD 3.1.1 evaluation entry point |
| Markdown reporter | `tools/eval/report_writer.py` | Eval report writer |
| Instance schema | `src/fashion_vision/schemas/instance_schema.py` | Canonical output record format |
| Query parser | `src/fashion_vision/localization/query_parser.py` | Canonical region-from-query logic |
| Landmark-region map | `src/fashion_vision/localization/landmark_region_map.py` | Hand-crafted per-class rules |
| Region locator | `src/fashion_vision/localization/region_locator.py` | Canonical region localization |
| Geometry fallback | `src/fashion_vision/localization/geometry.py` | Adaptive mask-based extraction |
| Garment pipeline | `tools/infer/garment_pipeline.py` | Pipeline orchestrator |

---

## Do-not-touch list (before and during retraining)

- `data/processed/deepfashion2_yolo_13cls/` — YOLO label set
- `data/processed/deepfashion2_yolo_13cls_val*/` — validation labels
- `tools/data/export_deepfashion2_to_yolo.py` — label conversion
- `tools/data/make_balanced_yolo_train_list.py` — balanced list builder
- `models/detectors/` — current YOLO weights
- `checkpoints/sam_hq/` — SAM-HQ weights
- `outputs/landmark_predictor_resnet18/best.pt` — landmark predictor
- `outputs/p2_*/best.pt` — all 3.1.3 attribute classifier checkpoints
- `tools/infer/segment_garments_samhq.py` — SAM-HQ stage
- `tools/infer/infer_landmarks_for_predictions.py` — landmark stage
- `src/fashion_vision/localization/landmark_region_map.py` — landmark rules
- All 3.1.3 scripts and `.bat` files
