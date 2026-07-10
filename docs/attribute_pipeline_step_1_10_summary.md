# PRD 3.1.3 Attribute Inference Pipeline — Steps 1–10 Engineering Summary

> Last updated: 2026-06-17
> Test status at time of writing: **283 passed, 2 skipped, 0 failed** (285 collected across 7 test files)
>
> Steps 1–10 were the original config-driven inference foundation.
> Steps 11–14 (2026-06-17) added bugfixes, a visualizer, and the PRD-facing `mask_attribute_pipeline.py` with a real smoke test.
> See **Section M** for the Steps 11–14 summary.

---

## A. Executive Summary

Steps 1–10 delivered a **config-driven, fully-tested, and extensible attribute inference
foundation** for PRD module 3.1.3 (fine-grained garment attribute extraction).

Before this work, the existing batch script (`scripts/run_p3_region_to_attribute_8tasks.py`)
contained a 90-line hardcoded task configuration dict with a latent bug in the
`coat_length` class filter, and all attribute inference logic was tightly coupled
to that single script.  There was no reusable library code, no unit tests, and no
integration path between the garment pipeline and the classifiers.

After Steps 1–10:

- A two-YAML config system drives all task routing (`attribute_inference.yaml`,
  `attribute_group_mapping.yaml`).
- A clean Python library (`src/fashion_vision/attributes/`) provides a tested,
  importable API for category gating, model loading, and end-to-end prediction.
- The batch script retains 100% backward compatibility: existing CLI calls are
  unchanged; it now reads from the same YAMLs.
- A standalone latency benchmark tool exists for PRD target verification.
- **197 tests are collected; 195 pass; 2 are skipped** (both skips are
  environment-conditional: CUDA auto-detect on a CPU-only machine).

---

## B. Current Architecture Overview

### End-to-End System (Stages 1–6)

```
Input image
    │
    ▼
[Stage 1] YOLO 13-class detector
    │   ↳ 01_yolo/detections.json
    │       fine_class_id/name (0–12), bbox, score
    ▼
[Stage 2] SAM-HQ segmentation
    │   ↳ 02_samhq/segmentation_results.json
    │       pred_mask_path, fine/coarse class fields
    ▼
[Stage 3] Landmark predictor (ResNet18, 39 landmarks)
    │   ↳ 03_landmarks/landmarks_results.json
    ▼
[Stage 4] Region crop generation (geometry-based)
    │   ↳ 04_region_crops/region_crops.json
    │       per-record: det_id, class_name, region, component,
    │                   expanded_crop_path, upper_crop_path
    ▼
[Stage 5] Mask-aware crop application
    │   ↳ 05_region_masked_crops/region_masked_crops.json
    ▼
[Stage 6] Attribute inference   ← NEW (Step 9)
        ↳ 06_attributes/predictions.jsonl
            per-instance: det_id, fine_class_name, coarse_class_name,
                          attributes: {task: {label, score, topk}}
```

Stages 1–5 are implemented by `tools/infer/garment_pipeline.py` (existing,
unchanged by Steps 1–10).  Stage 6 is the new `GarmentAttributePipeline`.

### 3.1.3 Attribute Inference Sub-System

```
region_crops.json
    │
    ▼
GarmentAttributePipeline.predict_from_json()
    │
    ├── Group records by det_id
    │
    ├── For each instance:
    │   ├── _infer_coarse_class(class_name)
    │   │     uses: attribute_group_mapping.yaml → coarse_class_to_fine_class_substrings
    │   │
    │   ├── category_gate.get_enabled_tasks(coarse_class_name)
    │   │     uses: attribute_group_mapping.yaml → coarse_class_to_tasks
    │   │
    │   ├── _get_tasks_for_class()  [lazy, cached per coarse class]
    │   │     calls: task_registry.load_tasks_for_class()
    │   │       → load_inference_config() from attribute_inference.yaml
    │   │       → load_task(): label_map → model → checkpoint → transform
    │   │
    │   └── For each enabled task:
    │       ├── region_filter + class_contains  → _select_crop_record()
    │       ├── crop_type                       → _get_crop_path()
    │       └── _run_inference(LoadedTask, crop_path)
    │             PIL.Image.open → transform → model → softmax → top-k
    │
    └── Output: list[{det_id, fine/coarse_class, attributes}]
```

### Backward-Compatible Batch Script

`scripts/run_p3_region_to_attribute_8tasks.py` retains its existing CLI surface.
Internally, `TASK_CONFIGS` is now built by `build_task_configs()` from the two
YAML files instead of a hardcoded dict.  All downstream subprocess calls to
`predict_region_attribute_batch.py` are identical.

---

## C. Step-by-Step Implementation Summary

### Step 1 — Bug Fix: `coat_length` class filter

| | |
|---|---|
| **Files modified** | `scripts/run_p3_region_to_attribute_8tasks.py` (one-word fix) |
| **Purpose** | Fix `class_contains="outerwear"` which never matched DeepFashion2 fine class names (`"short sleeve outwear"`, `"long sleeve outwear"`) |
| **Key behaviour** | Changed to `"outwear"`.  Without this fix, `coat_length` silently ran on all garment classes. |
| **Test coverage** | `test_run_p3_task_configs.py::test_coat_length_class_contains_outwear` (regression test) |

### Step 2 — Evaluation Target Config

| | |
|---|---|
| **Files created** | `configs/attribute_eval_targets.yaml` |
| **Purpose** | Formal record of PRD 3.1.3 accuracy targets (macro-F1 ≥ 0.88, best baseline: `collar_design` at 0.764) |
| **Key behaviour** | Reference only; no code reads this file yet. |
| **Test coverage** | None (config file, not business logic) |

### Step 3 — Attribute Group Mapping Config

| | |
|---|---|
| **Files created** | `configs/attribute_group_mapping.yaml` |
| **Purpose** | Single authoritative config for: coarse-class→task routing, task→region type, task→crop type, fine-class substring filters, component filters |
| **Key behaviour** | `coarse_class_to_tasks` determines which attribute tasks run per PRD class. `coarse_class_to_fine_class_substrings` enables backward-compatible record filtering where only `fine_class_name` is present. |
| **Test coverage** | `test_category_gate.py` (full structural + content tests) |

### Step 4 — `category_gate.py` and Package Init

| | |
|---|---|
| **Files created** | `src/fashion_vision/attributes/__init__.py`, `src/fashion_vision/attributes/category_gate.py` |
| **Purpose** | Parse and validate the group mapping YAML; expose a pure, importable API for task routing |
| **Key behaviour** | `load_attribute_group_mapping()` validates structural consistency on load. `get_enabled_tasks()` returns `[]` for unknown classes (never raises). `get_component_filter_for_task()` returns `None` for tasks without component filtering. |
| **Test coverage** | `tests/test_category_gate.py` — 29 tests covering all 6 public functions, all coarse classes, error paths, and the `"outwear"` regression |

### Step 5 — `crop_utils.py`

| | |
|---|---|
| **Files created** | `src/fashion_vision/utils/crop_utils.py` |
| **Purpose** | Canonical, deterministic region-crop function for live inference (not training) |
| **Key behaviour** | `crop_region_from_image(image_np, bbox_xyxy, mask, expand_ratio=0.15, target_size=224, background_fill)` — expands bbox, clamps to image bounds, applies optional mask fill, resizes to square PIL Image. Pre-expansion zero-area check raises before any clamping logic runs. |
| **Test coverage** | `tests/test_crop_utils.py` — 19 tests: output shape/mode, all 3 fill modes, edge clamping, 7 error paths including zero-area bbox |

### Step 6 — Config-Driven `run_p3` Refactor

Two sub-steps:

**Step 6a** — Extended `attribute_group_mapping.yaml` with `task_to_component_filter`; updated `category_gate.py` to load and expose this field.

**Step 6b** — Replaced hardcoded `TASK_CONFIGS` dict in `run_p3_region_to_attribute_8tasks.py` with `build_task_configs(inference_config, group_mapping_config)` reading from the two YAML files.

| | |
|---|---|
| **Files modified** | `configs/attribute_group_mapping.yaml`, `src/fashion_vision/attributes/category_gate.py`, `scripts/run_p3_region_to_attribute_8tasks.py` |
| **Purpose** | Eliminate 90-line hardcoded dict; make task configuration editable without touching Python source |
| **Key behaviour** | All 8 tasks, all 8 CLI arguments, all subprocess calls remain identical. New optional CLI args: `--inference-config`, `--group-mapping-config` for overriding YAML paths. |
| **Test coverage** | `tests/test_run_p3_task_configs.py` — 24 tests: all 8 tasks present, checkpoint paths, region values, crop types, component/class filters, error paths, `coat_length` regression |

### Step 7 — `attribute_inference.yaml`

| | |
|---|---|
| **Files created** | `configs/attribute_inference.yaml` |
| **Purpose** | Per-task model registry: checkpoint path, label map path, arch, img_size, region_filter, class_contains |
| **Key behaviour** | `neckline_design` uses the baseline checkpoint (`p2_neckline_design_resnet18_seed2`). All other 7 tasks use the `multiview_v2_pipeline` checkpoints. `num_classes` is intentionally absent — derived from the label map to avoid duplication. |
| **Test coverage** | Tested indirectly via `test_task_registry.py` and `test_run_p3_task_configs.py` |

### Step 8 — `task_registry.py`

| | |
|---|---|
| **Files created** | `src/fashion_vision/attributes/task_registry.py`, `tests/test_task_registry.py` |
| **Purpose** | Reusable library: parse inference config, load checkpoints (4 formats), parse label maps (5 formats), build inference transforms, instantiate models |
| **Key behaviour** | `load_checkpoint_state()` is public (reused by benchmark). `infer_num_classes_from_state()` inspects `fc.weight`, `classifier.weight`, `head.weight`. `load_task()` derives `num_classes` from `len(id_to_label)` — label map is authoritative. `load_tasks_for_class()` uses a local import to avoid circular imports. |
| **Test coverage** | `tests/test_task_registry.py` — 43 tests (11 config loading, 1 immutability, 9 transform, 7 label-map format, 3 file-based label map, 7 checkpoint loading, 5 class inference) |

### Step 9 — `garment_attribute_pipeline.py`

| | |
|---|---|
| **Files created** | `src/fashion_vision/attributes/garment_attribute_pipeline.py`, `tests/test_garment_attribute_pipeline.py` |
| **Purpose** | End-to-end bridge from 3.1.2 `region_crops.json` to 3.1.3 attribute predictions |
| **Key behaviour** | `GarmentAttributePipeline.predict_from_json()` groups records by `det_id`, infers coarse class from `class_name` via substring matching, runs only enabled tasks, selects the right crop type per task, skips missing files with a logged warning (never raises per-instance). Model loading is lazy and cached per coarse class. CLI: `--region-crops-json`, `--output-jsonl`, `--device`, `--topk`, `--max-instances`. |
| **Test coverage** | `tests/test_garment_attribute_pipeline.py` — 57 tests: all pure helpers (`_resolve_device`, `_infer_coarse_class`, `_get_crop_path`, `_select_crop_record`, `_run_inference`), config defaults, pipeline integration with mocked models, JSON grouping, error handling |

### Step 10 — `benchmark_attribute_latency.py`

| | |
|---|---|
| **Files created** | `tools/eval/benchmark_attribute_latency.py`, `tests/test_benchmark_attribute_latency.py` |
| **Purpose** | Standalone latency benchmark for a single attribute classifier; uses synthetic tensors only |
| **Key behaviour** | Reuses `load_checkpoint_state` and `infer_num_classes_from_state` from `task_registry`. CUDA synchronisation (`torch.cuda.synchronize()`) before/after each timed pass. Reports `latency_ms_per_batch` and `latency_ms_per_image` separately. `meets_prd_target = mean_per_image_ms <= 20.0`. Optional JSON output via `--out`. |
| **Test coverage** | `tests/test_benchmark_attribute_latency.py` — 30 tests: `_compute_stats` (13 tests), `_resolve_num_classes` (7 tests), `_resolve_device` (3 tests), PRD constant sanity (1 test) |

---

## D. Code Function Summary

### Configs

| File | Role |
|---|---|
| `configs/attribute_group_mapping.yaml` | Authoritative routing config: coarse_class→tasks, task→region, task→crop_type, fine-class substring filters, component filters. Read by `category_gate.py` and `run_p3`. |
| `configs/attribute_inference.yaml` | Per-task model registry: checkpoint paths, label map paths, arch, img_size, region_filter, class_contains. Read by `task_registry.py` and `run_p3`. |
| `configs/attribute_eval_targets.yaml` | PRD accuracy targets (reference only, no code reads it yet). |

### Library Modules (`src/fashion_vision/attributes/`)

| File | Role |
|---|---|
| `__init__.py` | Package marker. |
| `category_gate.py` | Loads and validates `attribute_group_mapping.yaml`. Exposes `get_enabled_tasks()`, `get_region_for_task()`, `get_crop_type_for_task()`, `get_fine_class_filter()`, `get_component_filter_for_task()`. |
| `task_registry.py` | Model loading library. `AttributeTaskConfig` (frozen dataclass), `LoadedTask` (live bundle). Public functions: `load_inference_config`, `load_checkpoint_state`, `infer_num_classes_from_state`, `build_inference_transform`, `load_id_to_label`, `load_task`, `load_tasks_for_class`. |
| `garment_attribute_pipeline.py` | End-to-end pipeline bridging 3.1.2→3.1.3. `GarmentAttributePipeline` class with lazy model cache. `predict_from_json()` reads region_crops.json, groups by `det_id`, calls `predict_instance()` for each. Pure helpers: `_infer_coarse_class`, `_get_crop_path`, `_select_crop_record`, `_run_inference`. CLI entry point. |

### Utility

| File | Role |
|---|---|
| `src/fashion_vision/utils/crop_utils.py` | `crop_region_from_image()` — deterministic crop from raw numpy image + bbox. Used when pre-saved crops are not available. `expand_ratio=0.15` matches multiview_v2 pipeline default. |

### Scripts / Tools

| File | Role |
|---|---|
| `scripts/run_p3_region_to_attribute_8tasks.py` | Batch runner: calls `predict_region_attribute_batch.py` via subprocess for each of the 8 tasks. Backward-compatible; now config-driven via `build_task_configs()`. |
| `tools/eval/benchmark_attribute_latency.py` | CLI latency benchmark. Synthetic tensors only. Reports per-batch and per-image stats, throughput, PRD target pass/fail. |

### Tests

| File | Tests | What it verifies |
|---|---|---|
| `tests/test_category_gate.py` | 29 | All 6 category_gate functions; all coarse classes; `"outwear"` regression; unknown-class safety |
| `tests/test_crop_utils.py` | 19 | Output shape/mode; 3 fill modes; edge clamping; 7 error paths |
| `tests/test_run_p3_task_configs.py` | 24 | `build_task_configs()` produces correct 8-task dict; checkpoint paths; region/crop/component/class filters; `coat_length` regression; error handling |
| `tests/test_task_registry.py` | 43 | Config parsing; 4 checkpoint formats; 5 label-map formats; transform pipeline; class inference |
| `tests/test_benchmark_attribute_latency.py` | 30 | `_compute_stats`; `_resolve_num_classes`; `_resolve_device`; PRD constant |
| `tests/test_garment_attribute_pipeline.py` | 57 | All pure helpers; pipeline integration with mocked models; JSON grouping; error handling |

---

## E. Runtime Flow

```
1. INPUT
   ├── region_crops.json (from garment_pipeline.py stage 4 or 5)
   │     Structure: {"crops": [{det_id, class_name, region, component,
   │                            success, expanded_crop_path, upper_crop_path, ...}]}
   └── (optional) attribute_inference.yaml + attribute_group_mapping.yaml overrides

2. GROUPING
   └── Records grouped by det_id → per-instance crop list

3. COARSE CLASS INFERENCE
   └── class_name (DeepFashion2 fine name, e.g. "long sleeve top")
         → _infer_coarse_class() via coarse_class_to_fine_class_substrings
         → coarse_class_name (e.g. "top")

4. TASK SELECTION
   └── get_enabled_tasks(coarse_class_name)
         ← coarse_class_to_tasks in attribute_group_mapping.yaml
         → ["neckline_design", "collar_design", "neck_design", "sleeve_length"]

5. MODEL LOADING  [lazy, cached per coarse class]
   └── load_tasks_for_class()
         → load_inference_config() → AttributeTaskConfig per task
         → load_task(): load_id_to_label → build_attribute_classifier → load_checkpoint_state
         → LoadedTask: {model (eval), id_to_label, transform}

6. CROP SELECTION  [per task]
   ├── region_filter from attribute_inference.yaml (e.g. "collar" or "all")
   ├── class_contains from attribute_inference.yaml (e.g. "outwear" for coat_length)
   ├── component_contains from attribute_group_mapping.yaml (e.g. "sleeve" for sleeve_length)
   └── _select_crop_record() → first matching successful crop record

7. CROP PATH RESOLUTION  [per task]
   └── _get_crop_path(record, crop_type)
         crop_type = task_to_crop_type[task]:
           "upper_crop"    → upper_crop_path → expanded_crop_path → image_crop_path
           "expanded_crop" → expanded_crop_path → image_crop_path → crop_path

8. INFERENCE  [per task]
   └── _run_inference(LoadedTask, crop_path, topk, device)
         PIL.Image.open → .convert("RGB")
         → LoadedTask.transform → unsqueeze(0) → model(x)
         → softmax → top-k → {label, score, topk: [{label, score}]}

9. OUTPUT
   └── predict_from_json() returns list[{det_id, fine_class_name, coarse_class_name,
                                          num_crops, attributes, error}]
       CLI writes one JSON record per line to --output-jsonl
```

---

## F. PRD Progress Summary

| PRD Requirement | Status | Evidence | Remaining Work |
|---|---|---|---|
| **3.1.1 Garment detection** (YOLO 13-class) | Done | YOLO + SAM-HQ pipeline benchmarked on 500 images | Balanced retraining planned (class imbalance) |
| **3.1.1 Coarse 5-class external output** | Partially Done | `instance_schema.py` has `fine/coarse_class_id/name` fields; `category_mapping.yaml` defines 13→5 mapping | Not yet applied at all detection pipeline outputs |
| **3.1.2 Segmentation / region crop generation** | Done | SAM-HQ segmentation + landmark + geometry region crops; 500-image benchmark; 92% valid response on Batch60 | SAM-HQ runtime bottleneck (69% of pipeline time) |
| **3.1.2 Language-guided region localization** | Partially Done | Rule-based Chinese query→region demo (7 query types) | Open-vocabulary grounding not implemented |
| **3.1.3 Attribute classification — model training** | Done | All 8 FashionAI tasks trained (ResNet18, multiview_v2_pipeline); best: collar_design 0.764 macro-F1 | PRD target 0.88 macro-F1 not yet reached; further training/architecture work needed |
| **3.1.3 Config-driven task routing** | Done | `attribute_group_mapping.yaml` + `attribute_inference.yaml` + `category_gate.py` | — |
| **3.1.3 Task model loading library** | Done | `task_registry.py`; 4 checkpoint formats; 5 label-map formats | — |
| **3.1.3 End-to-end attribute pipeline** | Done | `garment_attribute_pipeline.py`; reads region_crops.json; outputs per-instance attributes. PRD-facing interface `mask_attribute_pipeline.py` smoke-tested with real image+mask | — |
| **3.1.3 Backward-compatible batch runner** | Done | `run_p3_region_to_attribute_8tasks.py` refactored; CLI unchanged | — |
| **Category mapping 13→5** | Done (config) | `configs/category_mapping.yaml`; `attribute_group_mapping.yaml`; `coarse_class_to_fine_class_substrings` | Coarse class fields not uniformly applied throughout pipeline outputs |
| **Latency benchmark** | Done (tool) | `tools/eval/benchmark_attribute_latency.py`; synthetic tensors; PRD 20ms target check | Not yet run with real checkpoints on target hardware |
| **PRD 20ms latency target verified** | Not Done | Tool exists but no real checkpoint run performed | Run `benchmark_attribute_latency.py` with real checkpoints on target hardware |
| **Unit tests** | Done | 283 passed, 2 skipped, 0 failed; 7 test files; 285 tests | No integration tests with real images/checkpoints |
| **Documentation** | Done (this file) | `docs/attribute_pipeline_step_1_10_summary.md`; `docs/current_project_status.md` | — |
| **Real smoke test** | Done | `mask_attribute_pipeline.py` run on `000004.jpg` + SAM-HQ mask → 3 attribute predictions (neckline_design, collar_design, neck_design) | Broader multi-sample evaluation pending |
| **Production integration / API serving** | Not Started | — | Not in current scope |
| **3.2 Multimodal QA** | Not Started | — | Not in current scope |
| **3.3 Agent/RAG** | Not Started | — | Not in current scope |
| **Shoes / bags / accessories** | Not Started | — | Requires external data |

---

## G. Completed Work

The following items are fully implemented and passing tests as of 2026-06-16:

**Bug fixes:**
- `coat_length` class filter corrected from `"outerwear"` to `"outwear"` (perpetually embedded in both YAMLs and as a regression test)

**Configs (3 files):**
- `configs/attribute_group_mapping.yaml` — task routing, region types, crop types, fine-class substrings, component filters
- `configs/attribute_inference.yaml` — per-task checkpoint/label-map registry
- `configs/attribute_eval_targets.yaml` — PRD accuracy targets (reference)

**Library modules (4 files):**
- `src/fashion_vision/attributes/__init__.py`
- `src/fashion_vision/attributes/category_gate.py`
- `src/fashion_vision/attributes/task_registry.py`
- `src/fashion_vision/attributes/garment_attribute_pipeline.py`

**Utility (1 file):**
- `src/fashion_vision/utils/crop_utils.py`

**Refactored scripts (1 file):**
- `scripts/run_p3_region_to_attribute_8tasks.py` — now config-driven, backward-compatible

**Evaluation tools (1 file):**
- `tools/eval/benchmark_attribute_latency.py`

**Tests (6 files, 197 total):**
- `tests/test_category_gate.py` (29)
- `tests/test_crop_utils.py` (19)
- `tests/test_run_p3_task_configs.py` (24)
- `tests/test_task_registry.py` (43)
- `tests/test_benchmark_attribute_latency.py` (30)
- `tests/test_garment_attribute_pipeline.py` (57)

---

## H. Remaining Work / Next Steps

### Immediate (safe, low-risk)

1. **Real checkpoint smoke test** — run `garment_attribute_pipeline.py` on one
   approved sample `region_crops.json` with real checkpoints.  Verify output JSON
   schema and attribute predictions are sensible.

2. **Validate checkpoints and label maps** — confirm all 8 checkpoint `.pt` files
   and all 8 `label_map_*.json` files exist at the paths listed in
   `attribute_inference.yaml`.

3. **Run latency benchmark with real checkpoints** — for each of the 8 tasks:
   ```
   python tools/eval/benchmark_attribute_latency.py \
       --checkpoint outputs/p2_<task>_multiview_v2_pipeline_resnet18_seed2/best.pt \
       --num-classes <N> \
       --device auto \
       --runs 200
   ```
   Verify `meets_prd_target` for each task on the actual target hardware.

4. **Verify PRD 20ms latency target** — the target is `mean_per_image_ms <= 20.0`.
   CPU-only result is expected to exceed this; GPU result on the deployment device
   needs to be measured.

### Medium-term

5. **Integrate into garment_pipeline.py as Stage 6** — add optional attribute
   inference step after stage 5 (masked crops), gated by a config flag.  This
   would make `GarmentPipeline.run_source()` return per-instance attributes.

6. **Integration tests with tiny fixtures** — create minimal `region_crops.json`
   with synthetic crop images; run `predict_from_json()` end-to-end without real
   checkpoints (e.g. using a random-weight model saved to a temp file).

7. **Apply 13→5 coarse class fields uniformly** — ensure all pipeline output
   JSONs populate `coarse_class_id` and `coarse_class_name` using
   `configs/category_mapping.yaml`.

8. **Accuracy improvement toward PRD target (macro-F1 ≥ 0.88)** — current best
   baseline is `collar_design` at 0.764.  Possible directions: longer training,
   class-weighted loss, larger backbone, better crop strategy.

### Not In Scope (current phase)

- Production API serving / containerisation
- Frontend or consumer integration
- 3.2 Multimodal QA, 3.3 Agent/RAG
- Shoes, bags, accessories (no dataset coverage)
- Open-vocabulary grounding for 3.1.2

---

## I. Test Summary

### Overall status (as of 2026-06-17)

```
283 passed, 2 skipped, 0 failed  (285 collected)
7 test files
Runtime: ~7 seconds (CPU only, no model loading for most tests)
```

The 2 skips are both in the same condition:
`test_resolve_device_auto_cpu_without_cuda` — skipped when CUDA is actually
present.  Expected and correct.

### Test file breakdown

| File | Count | What is tested |
|---|---|---|
| `test_category_gate.py` | 29 | YAML loading, validation, all 6 public API functions, all 5 coarse classes, unknown-class safety, `"outwear"` regression |
| `test_crop_utils.py` | 19 | PIL output, target size, RGB mode, expand ratio, all 3 fill modes, 3-D mask, 7 error paths including zero-area bbox |
| `test_run_p3_task_configs.py` | 24 | `build_task_configs()` task names, checkpoint paths, region values, crop types, component filters, class filters, error handling, `coat_length` regression |
| `test_task_registry.py` | 43 | Config parsing (11), frozen dataclass (1), transform stages (9), 5 label-map formats (7), file-based label map (3), 4 checkpoint formats (7), class inference (5) |
| `test_benchmark_attribute_latency.py` | 30 | `_compute_stats` correctness (13), `_resolve_num_classes` with synthetic state dicts (7), `_resolve_device` (3), PRD constant (1), importlib loading (baseline) |
| `test_garment_attribute_pipeline.py` | 57 | `_resolve_device` (3), `_infer_coarse_class` all 12 cases (12), `_get_crop_path` all types + fallbacks (9), `_select_crop_record` all filters (10), `_run_inference` (4), config defaults (2), pipeline integration with mocked models (6), `predict_from_json` JSON grouping (9) — includes 3 new det_id grouping tests |
| `test_mask_attribute_pipeline.py` | 47 | `_mask_bbox_xyxy` (5), file loaders (8), `_normalize_garment_category` (9), `_get_region_component` (7), `_make_overlay` (4), `_build_synthetic_record` (4), `MaskAttributePipeline.predict` with mocked inference (10) |

### What is intentionally NOT tested (requires real assets)

- Loading real checkpoint files from `outputs/`
- Actual attribute predictions from real classifiers
- Full end-to-end `predict_from_json()` with real crop images
- Latency on real hardware (CPU or GPU)
- Real PRD target validation with actual models

---

## J. Operational Constraints / Safety Notes

The following constraints are project-wide and must be respected in all future
sessions:

1. **Do not scan datasets, outputs, images, masks, weights, logs, or checkpoints**
   without explicit approval.  Restricted directories:
   `D:\Aliintern\fashion-ai-data`, `outputs/`, `runs/`, `wandb/`, `logs/`,
   `checkpoints/`, `models/`, `weights/`, `visualizations/`, `crops/`, `masks/`,
   `overlays/`.

2. **Do not open** image files, crop files, mask files, model weights, ONNX/TRT
   files, large JSONL files, pickle files, or numpy array files.

3. **Do not run** YOLO inference, SAM-HQ segmentation, landmark inference, full
   pipeline inference, or attribute training without explicit approval.

4. **Do not run** `benchmark_attribute_latency.py` with real checkpoints without
   explicit approval and a specified checkpoint path.

5. **Do not modify conda dependencies** without explicit approval.

6. **Do not implement new features** without explicit user approval of a plan.

7. **Do not commit** without explicit user approval.

---

## K. Recommended Next Command Sequence

```bash
# 1. Verify current git state
git status --short
git diff --stat

# 2. Confirm all tests still pass
python -m pytest tests/test_category_gate.py tests/test_crop_utils.py \
    tests/test_run_p3_task_configs.py tests/test_task_registry.py \
    tests/test_benchmark_attribute_latency.py \
    tests/test_garment_attribute_pipeline.py -v

# 3. Syntax check all new source files
python -m py_compile \
    src/fashion_vision/attributes/category_gate.py \
    src/fashion_vision/attributes/task_registry.py \
    src/fashion_vision/attributes/garment_attribute_pipeline.py \
    src/fashion_vision/utils/crop_utils.py \
    tools/eval/benchmark_attribute_latency.py

# 4. Verify CLI help for both entry points
python src/fashion_vision/attributes/garment_attribute_pipeline.py --help
python tools/eval/benchmark_attribute_latency.py --help

# 5. Only after user approval:
git add configs/attribute_group_mapping.yaml \
        configs/attribute_inference.yaml \
        configs/attribute_eval_targets.yaml \
        src/fashion_vision/attributes/__init__.py \
        src/fashion_vision/attributes/category_gate.py \
        src/fashion_vision/attributes/task_registry.py \
        src/fashion_vision/attributes/garment_attribute_pipeline.py \
        src/fashion_vision/utils/crop_utils.py \
        scripts/run_p3_region_to_attribute_8tasks.py \
        tools/eval/benchmark_attribute_latency.py \
        tests/test_category_gate.py \
        tests/test_crop_utils.py \
        tests/test_run_p3_task_configs.py \
        tests/test_task_registry.py \
        tests/test_benchmark_attribute_latency.py \
        tests/test_garment_attribute_pipeline.py \
        CLAUDE.md \
        docs/attribute_pipeline_step_1_10_summary.md \
        docs/current_project_status.md
git commit -m "feat(3.1.3): add config-driven attribute inference pipeline (Steps 1-10)"
```

---

## L. Proposed Commit Message

```
feat(3.1.3): add config-driven attribute inference pipeline (Steps 1-10)

- Fix coat_length class_contains bug: "outerwear" → "outwear"
- Add attribute_group_mapping.yaml and attribute_inference.yaml
- Add category_gate.py: config-driven coarse-class → task routing
- Add task_registry.py: checkpoint loading, label-map parsing, model loading
- Add garment_attribute_pipeline.py: region_crops.json → per-instance attributes
- Add crop_utils.py: deterministic region crop from raw image + bbox
- Refactor run_p3_region_to_attribute_8tasks.py: config-driven, backward-compatible
- Add benchmark_attribute_latency.py: latency benchmark with PRD target check
- Add 197 unit tests across 6 files: 195 passed, 2 skipped, 0 failed
```

---

## M. Steps 11–14 — Bugfixes, Visualizer, and PRD-Facing Interface (2026-06-17)

This section documents the additional work done after Steps 1–10 to bring 3.1.3
from "code complete" to "smoke-tested with real product images and masks."

### Step 11 — Bugfixes

Three bugs found during smoke-test preparation:

**Bug A: 4 wrong checkpoint paths in `attribute_inference.yaml`**

Tasks `collar_design`, `neck_design`, `lapel_design`, and `coat_length` pointed to
`multiview_v2_pipeline` checkpoint directories that were never created — these tasks
only have baseline (`resnet18_seed2`) checkpoints.  Corrected paths:

```yaml
collar_design:  outputs/p2_collar_design_resnet18_seed2/best.pt
neck_design:    outputs/p2_neck_design_resnet18_seed2/best.pt
lapel_design:   outputs/p2_lapel_design_resnet18_seed2/best.pt
coat_length:    outputs/p2_coat_length_resnet18_seed2/best.pt
```

`neckline_design`, `sleeve_length`, `pant_length`, `skirt_length` retain valid
`multiview_v2_pipeline` checkpoints (unchanged).

**Bug B: `det_id=0` treated as falsy in instance grouping**

The original grouping key was built using:
```python
raw_det_id = str(crop.get("det_id") or crop.get("instance_id") or "")
```
Integer `0` is falsy in Python, so the first garment in every multi-garment image
got key `""` instead of `"img001__det0"`.

Fixed with an explicit `None` check:
```python
_det = crop.get("det_id")
if _det is None:
    _det = crop.get("instance_id")
raw_det_id = "" if _det is None else str(_det)
```

Added 3 regression tests to `test_garment_attribute_pipeline.py`.

**Bug C: `test_other_tasks_use_multiview_v2_pipeline_checkpoint` false assertion**

The existing test asserted all non-neckline tasks used `multiview_v2_pipeline`.
After the YAML fix this became wrong.  Replaced with two accurate tests:
- `test_multiview_v2_tasks_use_multiview_v2_pipeline_checkpoint` (3 tasks)
- `test_baseline_only_tasks_do_not_use_multiview_v2_checkpoint` (4 tasks)

### Step 12 — Pipeline Output Visualizer

**File:** `scripts/visualize_attribute_pipeline_output.py`

Renders per-instance attribute predictions (from `garment_attribute_pipeline.py`
JSONL output) as contact-sheet JPEG pages.  Key design choices:

- Uses same instance-key formula as the pipeline (`{image_stem}__det{det_id}`)
  to look up crop images from a `region_crops.json`.
- Resolves relative crop paths against `Path.cwd()` (must run from project root).
- Renders masked crops preferentially; falls back to raw/expanded crop.
- One tile per instance: thumbnail + task-name/label/score/top-k text.

### Step 13 — `mask_attribute_pipeline.py` (PRD-Facing Interface)

**File:** `src/fashion_vision/attributes/mask_attribute_pipeline.py`

Implements the PRD 3.1.3 input contract:

```
image + binary mask + garment_category + component_type
    → attribute labels + confidence scores
```

**API:**

```python
from fashion_vision.attributes.mask_attribute_pipeline import predict_attributes_from_mask

result = predict_attributes_from_mask(
    image_path="assets/random_train60/images/000004.jpg",
    mask_path="outputs/test_pipeline_smoke/02_samhq/masks/000004_det000_long sleeve top_mask.png",
    garment_category="top",   # accepts aliases: "upper", "coat", "trousers", fine class names
    component_type="collar",  # "neckline", "sleeve", "pant_leg", etc.
    output_dir="outputs/smoke_test_attr_from_mask",
    topk=3,
    device="cpu",
)
# result["attributes"] → {"neckline_design": {"label": "...", "score": 0.0, "topk": [...]}, ...}
```

**Saved artifacts** (per call): masked crop (background-filled), raw crop,
mask overlay, `predictions.json`, `predictions.jsonl`.

**Design:** Builds a synthetic crop record pointing all crop-type path keys
(`expanded_crop_path`, `upper_crop_path`, `masked_crop_path`, `crop_path`) to the
saved masked crop, then delegates to `GarmentAttributePipeline.predict_instance()`.
This reuses all task routing, model loading, and inference logic without duplication.

**CLI:**
```bash
python scripts/run_attribute_from_mask_smoke.py \
    --image PATH --mask PATH \
    --garment-category CATEGORY \
    --component-type COMPONENT \
    --output-dir DIR \
    [--device cpu|cuda|auto] [--topk N] [--background-fill mean|zero|keep]
```

**Pure helpers (all independently testable):**

| Helper | Purpose |
|---|---|
| `_mask_bbox_xyxy(mask)` | Tight bounding box from foreground pixels |
| `_load_image_rgb(path)` | Load image as (H,W,3) uint8 numpy array |
| `_load_binary_mask(path)` | Load mask as (H,W) bool array |
| `_make_overlay(image, mask)` | Semi-transparent coloured overlay |
| `_normalize_garment_category(cat, mapping)` | Alias + substring → coarse class |
| `_get_region_component(component_type)` | component → (region, component) fields |
| `_build_synthetic_record(...)` | Build synthetic crop record for `predict_instance` |

### Step 14 — `tests/test_mask_attribute_pipeline.py`

47 new unit tests.  All pass.  Full suite: **283 passed, 2 skipped, 0 failed**.

| Test class | Count | Coverage |
|---|---|---|
| `TestMaskBboxXyxy` | 5 | Centred mask, full mask, single pixel, empty mask raises, uint8 input |
| `TestFileLoaders` | 8 | Shape/dtype, foreground values, full/empty mask, missing file raises (both loaders) |
| `TestNormalizeGarmentCategory` | 9 | Direct names, aliases, fine class names, case-insensitive, unknown raises |
| `TestGetRegionComponent` | 7 | Known mappings (collar, neckline, sleeve, pant_leg, pant), unknown passthrough, case-insensitive |
| `TestMakeOverlay` | 4 | RGB PIL output, spatial dims, foreground tinting, background unchanged |
| `TestBuildSyntheticRecord` | 4 | All crop paths → masked crop, image_crop_path → raw crop, success=True, class_name preserved |
| `TestMaskAttributePipelinePredict` | 10 | Output schema, coarse class resolution, artifacts saved, bbox is 4 ints, attributes passthrough, empty mask raises, size mismatch raises, missing file raises, topk propagation, topk restored after call |

### Real Smoke Test Result

```
Command:
    python scripts/run_attribute_from_mask_smoke.py \
        --image "assets/random_train60/images/000004.jpg" \
        --mask "outputs/test_pipeline_smoke/02_samhq/masks/000004_det000_long sleeve top_mask.png" \
        --garment-category top --component-type collar \
        --output-dir outputs/smoke_test_attr_from_mask --device cpu --topk 3

Garment: long sleeve top (000004.jpg)
Task routing: top → [neckline_design, collar_design, neck_design, sleeve_length]
Collar crop matched: neckline_design, collar_design, neck_design
sleeve_length skipped: no sleeve component in synthetic record (correct)

Predictions:
    neckline_design → Straight Neck  (0.403)  [Invisible: 0.22, Square Neckline: 0.18]
    collar_design   → Invisible      (0.995)  [Puritan Collar: 0.00, Peter Pan: 0.00]
    neck_design     → Invisible      (0.467)  [Low Turtle Neck: 0.42, Turtle Neck: 0.07]

Artifacts saved to: outputs/smoke_test_attr_from_mask/
    000004_collar_masked_crop.jpg
    000004_collar_raw_crop.jpg
    000004_collar_overlay.jpg
    predictions.json
    predictions.jsonl

Wall time: ~0.6 s (CPU, models already loaded)
```

### Remaining 3.1.3 Gaps

| Gap | Notes |
|---|---|
| Accuracy target not validated | PRD requires macro-F1 ≥ 0.88; current best is 0.764 (collar_design) |
| Latency target not benchmarked | PRD requires ≤ 20 ms/image; `benchmark_attribute_latency.py` exists but not run with real checkpoints on target hardware |
| Attribute group coverage incomplete | 8 of 14 FashionAI attribute groups implemented; fabric and craftsmanship attributes absent |
| Multi-sample evaluation pending | Only one smoke-test image; no per-class confusion matrices |
| Stage 6 integration pending | `GarmentPipeline` does not yet invoke `GarmentAttributePipeline` as a pipeline stage |
