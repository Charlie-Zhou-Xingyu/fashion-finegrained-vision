# Final WSL/Linux Project Integration Plan (Conservative)

> Generated: 2026-07-27
> Revised: 2026-07-27 — conservative execution, minimal file moves
> Sources: `docs/d_drive_project_inventory.md`, `docs/linux_inference_inventory.md`
> Principle: D drive = 3.1/3.3 functional baseline. Linux = inference optimization enhancement.
> Strategy: **Keep original layout. Add Linux wrappers into the existing `inference/` namespace. No directory restructuring.**
> Status: **Plan only — no files have been copied, moved, or modified.**

---

## 0. Execution Rules Before Any Code Integration

These rules apply to every phase in §10. No exceptions without explicit approval.

### 0.1 Directory Layout

- **No directory restructuring.** `tools/infer/`, `tools/crop/`, `inference/serving/`, `configs/`, `tests/` stay at their original paths. Do not create `src/pipeline/` or `src/serving/` at this stage.
- Linux optimization wrappers go into the existing `inference/` namespace (`inference/wrappers/`). They are additive — no existing files are moved or renamed.

### 0.2 External References

- `external/d_drive_original/` and `external/linux_inference_original/` are **read-only snapshots.** No code in `src/`, `tools/`, `inference/`, `scripts/`, or `run_demo.py` may import from `external/`. The `PYTHONPATH` must not include `external/`.

### 0.3 Documentation and Reports

- Existing docs in `docs/` are preserved. Do not delete or overwrite `docs/d_drive_project_inventory.md` or `docs/linux_inference_inventory.md`.
- Existing benchmark reports in `results/benchmark/` are preserved. Do not regenerate or overwrite them during integration.

### 0.4 Exclusions

- **Generated/large artifacts are never copied.** See §7 for the full exclusion list. Key items: `outputs/`, `checkpoints/`, `models/`, `__pycache__/`, `.git/`, `*.engine`, `*.onnx`, `*.tar.gz`.

### 0.5 Git Audit Trail

- **Every phase must produce exactly one git commit.** The commit message format is prescribed in each phase step of §10.
- After each phase, run `git status` to confirm the working tree is clean (no untracked files outside the phase scope).
- After each phase, run `git diff --stat` against the previous commit to confirm only expected files changed.
- If a phase produces unexpected diffs, stop and resolve before proceeding to the next phase.
- No phase may squash or amend a previous phase's commit. Linear history only.

### 0.6 FP16 Policy

- FP16 is the **preferred default** for SAM inference because it is already validated (1.64x speedup, <2% IoU drift).
- FP16 wiring must be **non-invasive.** If enabling FP16 by default requires modifying more than the `run_demo.py` wrapper pass-through (e.g., changing internal pipeline signatures), revert to opt-in only (`--fp16` flag instead of `--no-fp16`). The first working demo is more important than the optimization default.
- The legacy `src/fashion_vision/models/sam_hq_wrapper.py` must remain untouched. FP16 flows through `inference/wrappers/sam_wrapper.py` only.

### 0.7 TensorRT Policy

- **TensorRT is benchmark and documentation only for this stage.**
- Do NOT wire TRT into `run_demo.py` default path. TRT wrappers (`sam_encoder_trt.py`, `sam_hq_trt_wrapper.py`) are copied into `inference/wrappers/` as reference code but are not imported by `run_demo.py` unless the user explicitly passes `--sam-backend tensorrt`.
- All TRT performance claims (4.02x encoder speedup, etc.) are presented as static benchmark reports in `results/benchmark/`. No live TRT inference is required for the demo.

### 0.8 MVP Priority

- **A working end-to-end demo is more important than architectural cleanliness.**
- If a choice arises between "clean import structure" and "runs today," pick "runs today." Document the tradeoff with a `# ponytail:` comment and move on.
- Schema normalization (`score`→`confidence`, `pred_mask_path`→`mask_path`, `part`→`part_type`) is **best-effort.** If the underlying pipeline output differs from the unified schema in §5, preserve the raw pipeline output under a `raw_pipeline_output` key rather than silently dropping fields or crashing.

---

## 1. Final Directory Structure

```
fashion_final_project/
├── src/
│   └── fashion_vision/            # D drive: core vision library (UNCHANGED)
│       ├── schemas/               # instance_schema.py
│       ├── models/                # sam_hq_wrapper.py (legacy, reference)
│       ├── data/                  # class_mapping.py, landmarks.py
│       ├── localization/          # ALL 3.1.2 modules (router, detectors, filters)
│       ├── attributes/            # ALL 3.1.3 modules (pipeline, task_registry, gates)
│       ├── landmarks/             # Landmark prediction
│       ├── utils/                 # Shared utilities
│       ├── visualization/         # Debug/result visualization
│       └── prompts/               # Box/point/text prompt utils
│
├── tools/                         # D drive: pipeline scripts (UNCHANGED)
│   ├── infer/
│   │   ├── garment_pipeline.py    # Primary orchestrator (stages 1-6)
│   │   ├── predict_garments_yolo.py
│   │   ├── segment_garments_samhq.py
│   │   ├── infer_landmarks_for_predictions.py
│   │   └── run_garment_pipeline.py  # Legacy subprocess orchestrator
│   └── crop/
│       ├── crop_garment_regions_from_landmarks.py
│       └── apply_samhq_mask_to_region_crops.py
│
├── inference/                     # Serving + Linux optimization layer
│   ├── serving/                   # D drive: FastAPI QA layer (UNCHANGED)
│   │   ├── app.py                 # FastAPI app (6 endpoints)
│   │   ├── schemas.py             # Pydantic v2 models
│   │   ├── qa_orchestrator.py     # QA dispatch
│   │   ├── intent_classifier.py   # Rule-based intent (17 types)
│   │   ├── attribute_service.py
│   │   ├── rag_service.py
│   │   ├── vision_provider.py
│   │   ├── vision_context.py
│   │   ├── subprocess_vision_provider.py
│   │   ├── region_backend.py
│   │   ├── region_query_mapper.py
│   │   ├── attribute_backend.py
│   │   ├── content_generation_service.py
│   │   ├── content_policy.py
│   │   ├── deps.py
│   │   ├── errors.py
│   │   └── logging_utils.py
│   │
│   ├── wrappers/                  # Linux: SAM/YOLO optimization (ADDED)
│   │   ├── __init__.py
│   │   ├── sam_wrapper.py         # FP16 + batched boxes (primary)
│   │   ├── sam_wrapper_factory.py # pytorch/tensorrt backend switch
│   │   ├── sam_encoder_trt.py     # TRT FP16 encoder
│   │   ├── sam_hq_trt_wrapper.py  # TRT wrapper with auto-fallback
│   │   └── yolo_wrapper.py        # YOLO wrapper (TRT NotImplemented)
│   │
│   ├── env_capture.py             # Linux: benchmark environment capture
│   └── latency_taxonomy.py        # Linux: latency taxonomy utility
│
├── configs/                       # D drive: canonical (UNCHANGED except paths)
│   ├── category_mapping.yaml
│   ├── attribute_inference.yaml
│   ├── attribute_group_mapping.yaml
│   ├── attribute_templates.yaml
│   ├── attribute_taxonomy.yaml
│   ├── attribute_eval_targets.yaml
│   ├── intent_taxonomy.yaml
│   ├── knowledge_base.yaml
│   ├── knowledge_schema.md
│   ├── retrieval_config.yaml
│   └── serving_config.yaml        # From Linux (unique, not in D drive)
│
├── scripts/                       # Curated demo/benchmark entry points
│   ├── run_qa_orchestrator.py     # D drive: CLI QA demo
│   ├── demo_collar_qa.py          # D drive: P0 collar golden path
│   ├── run_full_31x_pipeline.py   # Linux: batch CLI runner
│   ├── build_31x_product_demo.py  # Linux: HTML demo generator
│   └── benchmarks/                # Linux: key benchmarks
│       ├── bench_trt_vs_pytorch_sam.py
│       ├── benchmark_runner.py
│       └── validate_deepfashion2_trt_accuracy.py
│
├── tests/                         # D drive: canonical (373+ tests, UNCHANGED)
│
├── models/                        # Gitignored — model weights
│   ├── detectors/                 # YOLOv8n, Fashionpedia YOLOv8s
│   └── attributes/                # 8 ResNet18 attribute classifiers
│
├── checkpoints/                   # Gitignored — SAM-HQ checkpoint
│   └── sam_hq/
│
├── data/                          # D drive: minimal label maps
│   └── fashionai_attribute_index/
│
├── eval/                          # D drive: evaluation framework
│
├── results/                       # Linux: presentation-ready benchmark reports
│   └── benchmark/
│       ├── trt_encoder_bench_report_v2.md
│       ├── trt_encoder_validation_summary.md
│       ├── pipeline_benchmark_500_report.md
│       └── query_region_batch60_report.md
│
├── external/                      # Reference snapshots (NOT imported)
│   ├── d_drive_original/          # Everything not migrated from D drive
│   └── linux_inference_original/  # Everything not migrated from Linux
│
├── docs/                          # Documentation
│   ├── README.md                  # Final project README
│   ├── d_drive_project_inventory.md
│   ├── linux_inference_inventory.md
│   ├── integration_plan.md        # THIS file
│   └── architecture/
│
├── run_demo.py                    # NEW: root-level unified entry point (§6)
├── requirements.txt               # Merged + expanded + version-pinned
├── requirements-frozen.txt        # Exact pins for reproducibility (NEW)
├── pyproject.toml                 # Proper package setup (NEW)
├── .env.example                   # Linux/WSL2 paths (path-cleaned)
├── .gitignore
└── README.md                      # Project-level README
```

**Key difference from earlier plan**: `tools/infer/`, `tools/crop/`, `inference/serving/`, `configs/`, `tests/` stay in their original locations. No `src/pipeline/` or `src/serving/` directories are created. Linux wrappers slot into the existing `inference/` namespace. The only new file is root-level `run_demo.py`.

---

## 2. Modules from D Drive Project (3.1 / 3.3 Baseline)

These are the **canonical source** for functional correctness. All 373+ tests pass against these. **All stay at original paths — no file moves.**

### Category A — Core Pipeline (keep as-is)

| Source (D drive) | Destination | Notes |
|---|---|---|
| `src/fashion_vision/` (entire) | `src/fashion_vision/` | Core library. **Copy verbatim, no changes.** |
| `tools/infer/garment_pipeline.py` | `tools/infer/garment_pipeline.py` | Primary orchestrator. Keep at original path. |
| `tools/infer/predict_garments_yolo.py` | `tools/infer/predict_garments_yolo.py` | Stage 1: YOLO detection |
| `tools/infer/segment_garments_samhq.py` | `tools/infer/segment_garments_samhq.py` | Stage 2: SAM segmentation |
| `tools/infer/infer_landmarks_for_predictions.py` | `tools/infer/infer_landmarks_for_predictions.py` | Stage 3: Landmarks |
| `tools/infer/run_garment_pipeline.py` | `tools/infer/run_garment_pipeline.py` | Legacy subprocess orchestrator. Keep as reference. |
| `tools/crop/crop_garment_regions_from_landmarks.py` | `tools/crop/crop_garment_regions_from_landmarks.py` | Stage 4: Region crops |
| `tools/crop/apply_samhq_mask_to_region_crops.py` | `tools/crop/apply_samhq_mask_to_region_crops.py` | Stage 5: Mask-gated crops |
| `configs/` (all YAML) | `configs/` | All config files. **Path-clean only** (§8). |
| `tests/` (all) | `tests/` | Regression safety net. **Copy verbatim.** |
| `eval/` | `eval/` | Evaluation framework |
| `data/fashionai_attribute_index/` | `data/fashionai_attribute_index/` | Label maps |

### Category B — Serving / 3.3 QA Layer (keep as-is)

| Source (D drive) | Destination | Notes |
|---|---|---|
| `inference/serving/` (entire, all 18 files) | `inference/serving/` | FastAPI QA layer. **Copy verbatim, no changes.** |

### Category C — Demo Scripts

| Source (D drive) | Destination | Notes |
|---|---|---|
| `scripts/run_qa_orchestrator.py` | `scripts/run_qa_orchestrator.py` | CLI QA demo |
| `scripts/demo_collar_qa.py` | `scripts/demo_collar_qa.py` | P0 collar golden path |

### Category D — Top-Level Project Files

| Source (D drive) | Destination | Notes |
|---|---|---|
| `README.md` | `README.md` | Must update paths |
| `CLAUDE.md` | `CLAUDE.md` | Project rules |
| `.env.example` | `.env.example` | Must update paths (§8) |
| `.gitignore` | `.gitignore` | Already correct |
| `pytest.ini` | `pytest.ini` | Pytest config |

---

## 3. Modules from Linux Project (Inference Optimization)

Only the **optimization layer** migrates. The Linux project's `src/fashion_vision/`, `tools/`, `inference/serving/`, `tests/` are near-duplicates of D drive and are **NOT migrated** — D drive is canonical.

### Category A — Core Optimization Wrappers

All go into the existing `inference/` namespace — no `src/` prefix.

| Source (Linux) | Destination | Notes |
|---|---|---|
| `inference/wrappers/sam_wrapper.py` | `inference/wrappers/sam_wrapper.py` | FP16 autocast + batched boxes. **Primary SAM wrapper.** |
| `inference/wrappers/sam_wrapper_factory.py` | `inference/wrappers/sam_wrapper_factory.py` | Backend switch (pytorch/tensorrt) |
| `inference/wrappers/sam_encoder_trt.py` | `inference/wrappers/sam_encoder_trt.py` | TRT FP16 encoder |
| `inference/wrappers/sam_hq_trt_wrapper.py` | `inference/wrappers/sam_hq_trt_wrapper.py` | TRT + PyTorch decoder, auto-fallback |
| `inference/wrappers/yolo_wrapper.py` | `inference/wrappers/yolo_wrapper.py` | PyTorch path works. TRT path is `NotImplementedError`. Keep as reference. |
| `inference/env_capture.py` | `inference/env_capture.py` | Benchmark environment hygiene |
| `inference/latency_taxonomy.py` | `inference/latency_taxonomy.py` | Latency taxonomy utility |

### Category B — Benchmark Scripts

| Source (Linux) | Destination | Notes |
|---|---|---|
| `inference/benchmarks/bench_trt_vs_pytorch_sam.py` | `scripts/benchmarks/bench_trt_vs_pytorch_sam.py` | Main optimization benchmark |
| `inference/benchmarks/benchmark_runner.py` | `scripts/benchmarks/benchmark_runner.py` | Unified benchmark framework |
| `inference/benchmarks/validate_deepfashion2_trt_accuracy.py` | `scripts/benchmarks/validate_deepfashion2_trt_accuracy.py` | TRT accuracy validation |

### Category C — Presentation Results (key reports only)

| Source (Linux) | Destination | Notes |
|---|---|---|
| `outputs/tensorrt_spike/encoder/bench_report_v2.md` | `results/benchmark/trt_encoder_bench_report_v2.md` | TRT 4.02x speedup |
| `outputs/tensorrt_spike/encoder/final_encoder_validation_summary.md` | `results/benchmark/trt_encoder_validation_summary.md` | 150-image validation |
| `docs/reports/pipeline_benchmark_500_report.md` | `results/benchmark/pipeline_benchmark_500_report.md` | 420ms/image breakdown |
| `docs/reports/query_region_batch60_report.md` | `results/benchmark/query_region_batch60_report.md` | 92% valid rate |
| `outputs/tensorrt_spike/encoder/bench_report_v2.json` | `results/benchmark/bench_report_v2.json` | Raw TRT benchmark data |

### Category D — Batch/HTML Demo Scripts

| Source (Linux) | Destination | Notes |
|---|---|---|
| `scripts/run_full_31x_pipeline.py` | `scripts/run_full_31x_pipeline.py` | Batch CLI runner |
| `scripts/build_31x_product_demo.py` | `scripts/build_31x_product_demo.py` | HTML demo generator |

### Category E — `configs/serving_config.yaml`

| Source (Linux) | Destination | Notes |
|---|---|---|
| `configs/serving_config.yaml` | `configs/serving_config.yaml` | Unique to Linux. Feature flags for vision provider backends. |

### What Does NOT Migrate from Linux

See §7 for the full exclusion list.

---

## 4. Duplicate Module Resolution

### 4.1 Two SAM Wrappers

| Aspect | D drive (legacy) | Linux (optimized) |
|---|---|---|
| **File** | `src/fashion_vision/models/sam_hq_wrapper.py` | `inference/wrappers/sam_wrapper.py` |
| **Features** | Basic SAM-HQ load + predict | FP16 autocast, batched box prediction, lazy loading |
| **Used by** | `tools/infer/segment_garments_samhq.py` | `tools/infer/garment_pipeline.py` (Linux version) |
| **FP16** | No | Yes (1.64x speedup, <2% IoU drift) |
| **Batch boxes** | Sequential only | Yes (6.54x at N=5) |

**Resolution:**

1. **Primary wrapper**: `inference/wrappers/sam_wrapper.py` (Linux optimized).
   - FP16 becomes the **default** (`use_fp16=True`).
   - The D drive `garment_pipeline.py` will be the canonical orchestrator. When `run_demo.py` calls it, FP16 is enabled by default via the optimized wrapper.

2. **Legacy wrapper**: `src/fashion_vision/models/sam_hq_wrapper.py` (D drive).
   - **Keep as-is at original path.** Add a module-level docstring noting the optimized replacement.
   - `tools/infer/segment_garments_samhq.py` retains its existing import. The optimized path flows through `run_demo.py` → optimized wrapper, not through this file.

3. **TRT wrappers** (`sam_encoder_trt.py`, `sam_hq_trt_wrapper.py`, `sam_wrapper_factory.py`):
   - These are **Linux-only additions** with no D drive counterpart.
   - **Do NOT wire into `run_demo.py` by default.** The factory pattern is ready but requires a `.engine` file on the target GPU.
   - For the final presentation: FP16 (`use_fp16=True`) is the default optimization. TRT results are presented as **benchmark numbers**, not live demo integration.

### 4.2 Two Pipeline Entry Points

| Aspect | D drive (canonical) | Linux (legacy) |
|---|---|---|
| **File** | `tools/infer/garment_pipeline.py` | `tools/infer/run_garment_pipeline.py` |
| **Architecture** | Function-call API (`GarmentPipeline` class) | Subprocess-based (calls stages as separate processes) |
| **Stages** | 1-6 (YOLO→SAM→Landmarks→Crops→Mask→Attributes) | 1-5 only (no attributes) |
| **Lazy loading** | Yes | No |
| **Test coverage** | 373+ tests | Unknown |

**Resolution:**

1. **Canonical**: `tools/infer/garment_pipeline.py` (D drive function-call API). Kept at original path.
2. **Legacy**: `tools/infer/run_garment_pipeline.py` → keep at original path as reference. It stays in-tree but is not the primary entry point.
3. **Note**: The Linux project has `scripts/run_full_31x_pipeline.py` which is a **thin CLI wrapper** around `GarmentPipeline` — migrate this to `scripts/run_full_31x_pipeline.py`.

### 4.3 Inference/Serving vs scripts/run_qa_orchestrator.py

| Aspect | Serving QA | CLI QA |
|---|---|---|
| **File** | `inference/serving/qa_orchestrator.py` | `scripts/run_qa_orchestrator.py` |
| **Purpose** | FastAPI-integrated QA dispatch | Standalone CLI demo |
| **Input** | `VisionContext` + `MultimodalQARequest` | CLI args (`--image`, `--query`) |
| **Output** | `UnifiedResponse` (JSON over HTTP) | Console JSON + evidence crops |
| **Shared backend** | Same `IntentClassifier`, `AttributeService`, `RagService` | Same orchestrator internals |

**Resolution:**

1. **Both keep at original paths.** They serve different purposes: serving = API, CLI = demo.
2. **No import path changes needed** since both stay in their original locations.
3. `scripts/run_qa_orchestrator.py` already imports from `inference.serving.*` — this import stays valid.

### 4.4 Near-Duplicate Core Libraries

Both projects have `src/fashion_vision/` with substantially the same files.

**Resolution:**

1. **D drive `src/fashion_vision/` is canonical.** Copy it verbatim.
2. **needs manual review**: Run `diff -rq` between D drive and Linux `src/fashion_vision/` to catalog differences before copying. If any Linux-specific bug fixes exist, apply them manually to the canonical copy.
3. Do NOT copy Linux `src/fashion_vision/` — it would overwrite the canonical D drive copy.

### 4.5 Duplicate Config Files

Both projects have `configs/` with mostly identical files. Linux adds `serving_config.yaml`.

**Resolution:**

1. **D drive `configs/` is canonical** for all functional configs (category_mapping, attribute_*, intent_taxonomy, knowledge_base, retrieval).
2. **Linux `configs/serving_config.yaml`** is the only Linux-specific config — migrate it.
3. **Path-cleaned versions** of YAML files containing hardcoded paths supersede both — apply §8 cleanup.
4. **needs manual review**: Diff D drive vs Linux `configs/` to confirm only path differences.

---

## 5. Unified JSON Schema

### 5.1 `garment_instances`

Array of per-garment detection+segmentation results. Produced by Stage 1-2 of the pipeline.

```json
{
  "garment_instances": [
    {
      "instance_id": "img001_garment_0",
      "category": "top",
      "fine_class_id": 3,
      "fine_class_name": "short sleeve top",
      "coarse_class_id": 0,
      "coarse_class_name": "top",
      "category_zh": "上装",
      "bbox": [120, 80, 340, 450],
      "bbox_format": "xyxy",
      "confidence": 0.95,
      "mask_present": true,
      "mask_path": "outputs/demo/masks/img001_garment_0_mask.png",
      "segmentor": "sam_hq_vit_b",
      "source_type": "yolo_detection"
    }
  ]
}
```

**Field consolidation notes:**

| Field | Source module | Resolution |
|---|---|---|
| `score` (3.1.1) vs `confidence` (3.3) | D drive §5.5 | **Use `confidence`** everywhere. More intuitive for presentation. Add `confidence` to `build_instance_record()` output; keep `score` as an alias in the Pydantic model for backward compat. |
| `category` | D drive §2.2 | The 5-class PRD label (`top`, `bottom`, `outerwear`, `full_body`, `accessory`). Kept as primary. |
| `fine_class_name` / `coarse_class_name` | D drive §2.2 | Both kept. 13-class fine, 5-class coarse. Dual-label is a key feature. |
| `pred_mask_path` vs `mask_path` | D drive §2.2 | **Rename to `mask_path`** in unified schema. Shorter, consistent with `mask_present`. **needs manual review** — `pred_mask_path` is embedded in `build_instance_record()`. |
| `status` | D drive §2.2 | Kept. Signals segmentation success/failure per instance. |
| `segmentor` | D drive §2.2 | Kept. Identifies which segmentation model was used. |

### 5.2 `localized_regions`

Array of per-query region localization results. Produced by 3.1.2 router.

```json
{
  "localized_regions": [
    {
      "region_id": "img001_garment_0_collar",
      "instance_id": "img001_garment_0",
      "part_type": "collar",
      "part_group": "neck",
      "garment_ref": "top",
      "bbox": [180, 75, 280, 120],
      "bbox_format": "xyxy",
      "confidence": 0.72,
      "backend": "fast_path",
      "status": "success",
      "all_detections": [
        {"bbox": [180, 75, 280, 120], "confidence": 0.72},
        {"bbox": [175, 80, 285, 125], "confidence": 0.31}
      ],
      "mask_present": false,
      "mask_path": null
    }
  ]
}
```

| Field | Source module | Resolution |
|---|---|---|
| `part` (router) vs `part_type` (serving) | D drive §5.5 | **Use `part_type`** as canonical. `part` is an alias in Pydantic. |
| `score` (router) vs `confidence` (serving) | D drive §5.5 | **Use `confidence`**. See §5.1. |
| `all_bboxes` / `all_scores` | D drive §2.4 | **Rename to `all_detections`** — array of `{bbox, confidence}` objects. |
| `garment_ref_matched` | D drive §2.4 | **Drop.** Redundant with `instance_id` + `garment_ref`. |
| `debug` | D drive §2.4 | **Keep but optional.** Useful for presentation debugging. |

### 5.3 `attribute_results`

Per-instance attribute inference results. Produced by 3.1.3 pipeline.

```json
{
  "attribute_results": [
    {
      "instance_id": "img001_garment_0",
      "region_id": "img001_garment_0_collar",
      "image_id": "img001",
      "attributes": {
        "collar_design": {
          "value": "翻领",
          "value_en": "turn-down collar",
          "confidence": 0.88,
          "topk": [
            {"value": "翻领", "confidence": 0.88},
            {"value": "立领", "confidence": 0.07},
            {"value": "圆领", "confidence": 0.03}
          ]
        }
      },
      "error": null
    }
  ]
}
```

| Field | Source module | Resolution |
|---|---|---|
| `attribute_confidence` vs `confidence` | D drive §5.5 | **Use `confidence`** within each attribute dict. |
| `topk` format | D drive §2.5 | **Standardize** as `[{"value": ..., "confidence": ...}]`. **needs manual review** of actual output format. |
| `value_en` | NEW | Add English translation for bilingual presentation. **needs manual review** — may require mapping table. |
| `error` | D drive §2.5 | Kept. Per-instance error field for partial failures. |

### 5.4 `qa`

QA response envelope. Produced by 3.3 orchestrator.

```json
{
  "qa": {
    "query": "领口是什么设计？",
    "answer": "领口设计为翻领（置信度 0.88）。",
    "answer_type": "attribute_answer",
    "answer_confidence": 0.88,
    "intent": {
      "primary": "attribute_query",
      "sub": "collar_design",
      "matched_pattern": "领口.*设计",
      "confidence": 1.0
    },
    "used_tools": ["intent_classifier", "attribute_service", "template_answer"],
    "evidence": {
      "instance_id": "img001_garment_0",
      "region_id": "img001_garment_0_collar",
      "attribute_task": "collar_design",
      "attribute_value": "翻领",
      "attribute_confidence": 0.88
    },
    "warnings": []
  }
}
```

| Field | Source module | Resolution |
|---|---|---|
| `primary_intent` / `sub_intent` | D drive §3.1 | **Rename to `primary` / `sub`**. |
| `evidence` | NEW | Structured evidence linking the answer back to pipeline results. **Optional** — omit for knowledge-base answers. |
| `answer_confidence` | D drive §3.1 | Kept. |
| `warnings` | D drive §3.1 | Kept. |

### 5.5 Unified Demo Output (Top-Level)

The complete `run_demo.py` output merges all four schemas:

```json
{
  "image_id": "img001",
  "image_path": "inputs/demo/img001.jpg",
  "pipeline": {
    "timing": {
      "yolo_seconds": 0.12,
      "sam_hq_seconds": 0.35,
      "landmarks_seconds": 0.08,
      "region_crops_seconds": 0.05,
      "attribute_seconds": 0.22,
      "total_seconds": 0.82
    },
    "garment_instances": [...]
  },
  "localization": {
    "query": "领口是什么设计？",
    "localized_regions": [...]
  },
  "attributes": {
    "attribute_results": [...]
  },
  "qa": { ... }
}
```

---

## 6. `run_demo.py` Design

### 6.1 Purpose

Single entry point at the **project root** that runs the complete pipeline and produces a structured result. Replaces the need to run multiple scripts in sequence.

### 6.2 Architecture

```
run_demo.py                     # Root-level, imports from original locations
├── Argument parsing (argparse)
├── tools/infer/garment_pipeline.py    # GarmentPipeline.run_image() (stages 1-6)
├── [if --query] inference/serving/intent_classifier.py   # IntentClassifier
├── [if --query] src/fashion_vision/localization/         # 3.1.2 router
├── [if --query] inference/serving/qa_orchestrator.py     # QaOrchestrator.answer()
└── Output formatting (JSON/text)
```

All imports use the original module paths — no import rewrites needed since we kept the original layout:

```python
# run_demo.py imports (illustrative)
from tools.infer.garment_pipeline import GarmentPipeline
from inference.serving.qa_orchestrator import QaOrchestrator
from inference.serving.intent_classifier import IntentClassifier
from inference.wrappers.sam_wrapper import SamHqWrapper  # optimized, FP16 default
```

### 6.3 CLI Interface

```
python run_demo.py \
    --image <path>                          # Input image (required)
    --query "领口是什么设计？"               # Optional: NL query for QA
    --output-dir outputs/demo/              # Output directory (default)
    --mode pipeline|qa|full                 # pipeline=3.1 only, qa=3.3 only, full=both (default: full)
    --yolo-weights models/detectors/...     # Override default
    --sam-checkpoint checkpoints/sam_hq/... # Override default
    --sam-backend pytorch|tensorrt          # SAM backend (default: pytorch)
    --no-fp16                               # Disable FP16 (default: FP16 enabled)
    --no-attributes                         # Skip attribute inference
    --no-visualization                      # Skip debug visualization
    --format json|text                      # Output format (default: json)
```

### 6.4 Implementation Notes

- `run_demo.py` is a **thin script** (~100-150 lines). It calls `GarmentPipeline`, not reimplements it.
- FP16 is **enabled by default** (`--no-fp16` to disable). The optimized `SamHqWrapper` from `inference/wrappers/sam_wrapper.py` is used.
- `--sam-backend tensorrt` requires an `.engine` file. If missing, auto-fallback to pytorch (the TRT wrapper already handles this). **TRT is NOT the default.**
- All paths are resolved relative to the script's location at project root via `pathlib.Path(__file__).resolve().parent`.
- No hardcoded `D:/Aliintern/` paths — all defaults use project-relative paths or env vars.
- The `--query` path is optional — when omitted, output contains pipeline/loc/attr but no `qa` section.

### 6.5 What `run_demo.py` Does NOT Do

- Does NOT load models eagerly — delegates to `GarmentPipeline` which uses lazy loading.
- Does NOT start a server — that's `inference/serving/app.py` via uvicorn.
- Does NOT generate HTML — that's `scripts/build_31x_product_demo.py`.
- Does NOT run benchmarks — that's `scripts/benchmarks/`.
- Does NOT train models — training scripts are in `external/` reference only.
- Does NOT wire TensorRT into the default path — TRT is benchmark/documentation-only.

### 6.6 MVP Implementation Scope

The first working version of `run_demo.py` must meet these requirements. Everything else is deferred.

#### Required (MVP)

| # | Requirement | Detail |
|---|---|---|
| 1 | Parse CLI args | `--image` (required), `--query` (optional), `--output-dir` (default `outputs/demo/`), `--mode` (`pipeline|qa|full`, default `full`), `--no-attributes` (flag), `--format` (`json|text`, default `json`) |
| 2 | Call `GarmentPipeline` | Import from `tools.infer.garment_pipeline` and call `run_image()` with the input image path |
| 3 | Save `result.json` | Write the unified output to `<output-dir>/result.json`. Create `output-dir` if it does not exist |
| 4 | Best-effort field normalization | Map `score`→`confidence`, `pred_mask_path`→`mask_path`, `part`→`part_type` in the output. If a field is absent in the raw output, skip it — do not crash |
| 5 | QA graceful degradation | If `--query` is provided but the QA path fails (missing import, model not loaded, etc.), catch the exception, save the error as a `warnings` entry in the output, and continue. Do not crash |
| 6 | Preserve raw output | Always include a `raw_pipeline_output` key containing the unmodified `GarmentPipeline.run_image()` return value. This ensures no data is lost if the normalized schema differs from the actual pipeline output |

#### Explicitly NOT in MVP

| # | Non-requirement | Why |
|---|---|---|
| 1 | TensorRT live inference | Benchmark/docs only. Requires `.engine` file on target GPU |
| 2 | Server startup | `inference/serving/app.py` is run separately via uvicorn. `run_demo.py` is CLI-only |
| 3 | HTML generation | `scripts/build_31x_product_demo.py` handles this independently |
| 4 | Model training | Training scripts live in `external/` reference only |
| 5 | Evaluation/metrics | Eval framework lives in `eval/`. Not wired into demo |
| 6 | Internal refactors | No changes to `tools/infer/`, `inference/serving/`, or `src/fashion_vision/` internals. `run_demo.py` is a consumer, not a refactor |
| 7 | `--sam-backend`, `--yolo-weights`, `--sam-checkpoint`, `--no-fp16`, `--no-visualization` CLI flags | Defer to v2. MVP only needs the 6 flags listed above. The optimized `SamHqWrapper` import can be attempted and gracefully degraded if unavailable |
| 8 | Schema-enforced output validation | The unified schema in §5 is a target, not a hard contract for MVP. Use `raw_pipeline_output` as the safety net |

#### MVP Success Criterion

```
PYTHONPATH=. python run_demo.py --image inputs/demo/img001.jpg --output-dir outputs/demo/ --format json
```

This command must:
1. Complete without an unhandled exception
2. Produce `outputs/demo/result.json`
3. The JSON must contain at least `image_id`, `image_path`, `pipeline` (with `garment_instances`), and `raw_pipeline_output`

Optional QA smoke test:
```
PYTHONPATH=. python run_demo.py --image inputs/demo/img001.jpg --query "领口是什么设计？" --output-dir outputs/demo/
```
Must complete without an unhandled exception (QA failure as a warning is acceptable).

---

## 7. What NOT to Migrate

### 7.1 From D Drive — Excluded Entirely

| Item | Reason |
|---|---|
| `outputs/` (all) | Generated artifacts. Regenerate from pipeline. |
| `runs/` (all) | Training logs. Reference only. |
| `calibration_v*_results/` | Threshold calibration outputs. |
| `archive/` | Old experiments. |
| `artifacts/` | P13 visual QA artifacts. Reference only. |
| `final_fashionpedia_*/` | Exported build artifacts. |
| `*.tar.gz` (root) | Archived training outputs (~200MB). |
| `*.pt` (root) | YOLO base weights. Already in `models/detectors/`. |
| `*.txt` debug dumps (root) | `output_crops.txt`, `output_jsons.txt`, etc. |
| `*.bat` files | Windows scripts. Convert to `.sh` if needed. |
| `{time.time()-t0`, `powershell`, `skills-lock.json` | Stale system artifacts. |
| `data/coat_annotation_batch1/` | 50+ raw images. Reference only. |
| `datasets/` | Dataset building cache. |
| `assets/` | Test/eval images. Reference only. |

### 7.2 From D Drive — Archived to `external/d_drive_original/`

| Item | Reason |
|---|---|
| `tools/train/` | Training scripts. Done, not needed for demo. |
| `tools/eval/` | Evaluation scripts. Reference only. |
| `tools/data/` | Data preparation. Reference only. |
| `tools/analysis/` | Debugging tools. Reference only. |
| `tools/demo/query_region_online_demo.py` | Superseded by newer demos. |
| `tools/demo/summarize_query_region_batch.py` | Superseded. |
| `scripts/analyze_*.py`, `scripts/compare_*.py`, `scripts/calibrate_*.py`, `scripts/filter_*.py`, `scripts/inspect_*.py` | One-off analysis scripts. |
| `scripts/build_*_train.py`, `scripts/convert_*.py`, `scripts/make_*.py`, `scripts/rebuild_*.py` | Training-related. |
| `scripts/eval_*.py`, `scripts/run_*_eval*.py`, `scripts/summarize_*.py` | Evaluation scripts. |
| `scripts/run_*_visual_test.py`, `scripts/visualize_*.py` | Debug visualization scripts. |
| `scripts/build_31x_product_demo.py` (D drive version) | Superseded by Linux version. |
| `docs/周报*.md` | Weekly internship reports. |
| `docs/phase2_engineering_review.md` | Phase report. |
| `docs/industrial_grounding_implementation_plan.md` | Planning doc. |
| `docs/P0a_*.md`, `docs/P1_*.md` | Planning docs. |
| `docs/architecture/` | Architecture reference. |
| `PROJECT_CONTEXT.md` | Info is in README. |

### 7.3 From Linux — Excluded Entirely (Large/Generated)

| Item | Reason |
|---|---|
| `checkpoints/` (362 MB) | Model weights — re-download or symlink. |
| `models/` (22 MB) | Model weights — re-download or symlink. |
| `outputs/` (742 MB, all subdirs) | Generated results, TRT engines, intermediate artifacts. Extract key reports only. |
| `third_party/sam-hq/` (153 MB) | Clone separately per README. Already gitignored. |
| `.local/trt_python/` (~200 MB) | TensorRT wheel. Install via pip, don't copy. |
| `**/__pycache__/`, `**/*.pyc` | Bytecode cache. |
| `.git/` | Start fresh or selective clone. |
| `.claude/` | Claude Code session data. |
| `*.engine`, `*.onnx` (in outputs/) | Compiled TRT/ONNX. Rebuild on target GPU. |
| `outputs/tensorrt_spike/pipeline/` | Intermediate experiment. |
| `outputs/tensorrt_spike/yolo/` | Intermediate experiment. |
| `outputs/benchmarks/fp16_outlier_debug/` | Debug artifacts. |
| `outputs/benchmarks/legacy_vs_optimized_sam*/` | Intermediate benchmark runs. |
| `outputs/benchmarks/yolo_det_*/` | Intermediate benchmark runs. |
| `outputs/full_31x_demo/` | Demo output — regenerate. |
| `outputs/visualizations/` | Visualization output — regenerate. |

### 7.4 From Linux — Archived to `external/linux_inference_original/`

| Item | Reason |
|---|---|
| `src/fashion_vision/` (Linux version) | Duplicate of D drive canonical. Archived for diff reference. |
| `tools/` (all) | Duplicate of D drive. Archived for diff reference. |
| `inference/serving/` (all 18 files) | Duplicate of D drive canonical. Archived for diff reference. |
| `inference/benchmarks/` (remaining 12 `bench_*.py`) | Individual benchmarks. `benchmark_runner.py` is the preferred entry. |
| `inference/pipelines/` | Experimental fast-path wrappers. |
| `inference/export/export_sam_encoder.py` | Already executed. Keep as build reference. |
| `inference/llm/` (4 files) | LLM stubs. Not wired. |
| `inference/optimized/segment_garments_sam_optimized.py` | Alternative SAM variant. |
| `inference/engines/registry.json` | Empty TRT registry placeholder. |
| `scripts/` (remaining 50+ scripts, excluding migrated ones) | Duplicates of D drive + one-off benchmarks. |
| `tests/` (Linux version) | Duplicate of D drive canonical. Diff first for any Linux-only tests. |
| `docs/` (all except benchmark reports) | Planning docs, runbooks, retrospectives. |
| `configs/` (Linux versions, except `serving_config.yaml`) | Duplicates of D drive. Archived for diff reference. |
| `scripts/run_attribute_from_mask_smoke.py` | Thin test wrapper (23 lines). |
| `scripts/bench_serving.py` | CI-only serving benchmark. |
| `scripts/benchmark_312_timing.py` | 3.1.2 timing benchmark. |
| `scripts/benchmark_deepfashion2_gt.py` | GT processing benchmark. |
| `scripts/benchmark_fashionpedia_yolo.py` | Fashionpedia YOLO benchmark. |

---

## 8. Path Cleanup Plan for `D:/Aliintern` Hardcoded Paths

### 8.1 Files Requiring Mandatory Fix

| File | Current Path | Replacement |
|---|---|---|
| `configs/dataset/deepfashion2.yaml` | `root: D:/Aliintern/fashion-ai-data/deepfashion2` | `root: ${DEEPFASHION2_ROOT}` or relative `./data/deepfashion2` |
| `configs/model/sam_hq.yaml` | `checkpoint: D:/Aliintern/fashion-ai-models/sam_hq/sam_hq_vit_b.pth` | `checkpoint: ${SAM_HQ_CHECKPOINT}` or relative `./checkpoints/sam_hq/sam_hq_vit_b.pth` |
| `configs/inference/sam_box_prompt.yaml` | Two paths: dataset root + checkpoint | Same as above — env var substitution |

### 8.2 Files Requiring Optional Fix (docstrings/examples only)

| File | Location | Action |
|---|---|---|
| `scripts/run_full_31x_pipeline.py` | Line 13, docstring example | Replace with WSL2 path or `./inputs/` |
| `scripts/run_qa_orchestrator.py` | Line 19, docstring example | Replace with `./inputs/demo/` |

### 8.3 Path Resolution Strategy

**Principle: env vars for external data, project-relative for bundled assets.**

```
# .env.example (updated for WSL2/Linux)
DEEPFASHION2_ROOT=/home/charlie/datasets/deepfashion2
SAM_HQ_CHECKPOINT=./checkpoints/sam_hq/sam_hq_vit_b.pth
SAM_HQ_REPO=./third_party/sam-hq
FASHION_AI_MODELS=/home/charlie/models/fashion-ai
```

YAML configs use `${VAR}` syntax resolved at load time by the existing config loader. If the config loader does not support env var substitution, add it to `src/fashion_vision/utils/` before applying path cleanup.

**Python source files**: Already use `pathlib.Path(__file__).resolve().parent` — no changes needed for most files. Only the YAML configs and `.env.example` need path fixes.

### 8.4 `.env.example` Rewrite

Replace all Windows paths with WSL2/Linux equivalents:

```bash
# Before (D drive)
DEEPFASHION2_ROOT=D:/Aliintern/fashion-ai-data/deepfashion2
SAM_HQ_CHECKPOINT=D:/Aliintern/fashion-ai-models/sam_hq/sam_hq_vit_b.pth

# After (WSL2/Linux)
DEEPFASHION2_ROOT=/home/charlie/datasets/deepfashion2
SAM_HQ_CHECKPOINT=./checkpoints/sam_hq/sam_hq_vit_b.pth
SAM_HQ_REPO=./third_party/sam-hq
FASHION_AI_MODELS=./models
```

---

## 9. Dependency Cleanup Plan

### 9.1 Current State

Both projects share the same incomplete `requirements.txt` (~13 packages). Missing runtime dependencies:
- `ultralytics` (YOLO)
- `segment-anything` (SAM)
- `transformers` (Grounding DINO)
- `groundingdino` (open-vocab detection)
- `huggingface_hub` (model download)
- `scikit-learn` (BM25 token overlap in RAG)
- `onnx`, `onnxruntime-gpu`, `onnxsim` (ONNX export, optional)
- `tensorrt` (optional, Linux-only, extracted wheel)

### 9.2 Tiered Requirements Strategy

**`requirements.txt`** — Runtime requirements for the demo pipeline:

```
# Core ML
torch>=2.0.0,<3.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
segment-anything>=1.0
transformers>=4.30.0
groundingdino>=0.1.0
huggingface_hub>=0.16.0

# Data processing
numpy>=1.24,<2.0
opencv-python>=4.8.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
pycocotools>=2.0.6
scikit-learn>=1.3.0
scipy>=1.10.0

# Serving (for inference/serving/)
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
httpx>=0.24.0

# Optional: ONNX export
# onnx>=1.14.0
# onnxruntime-gpu>=1.15.0
# onnxsim>=0.4.0

# Optional: TensorRT (install from .local/ wheel or pip)
# tensorrt>=10.0.0
```

**`requirements-frozen.txt`** — Exact pins after first successful install. Generated with `pip freeze > requirements-frozen.txt`.

### 9.3 Version Pin Strategy

| Package | Pin? | Reason |
|---|---|---|
| `torch` | `>=2.0,<3.0` | CUDA-dependent, let pip resolve |
| `ultralytics` | `>=8.0,<9.0` | API stability |
| `numpy` | `>=1.24,<2.0` | `numpy<2` constraint is explicit and required |
| `transformers` | `>=4.30,<5.0` | API stability |
| `fastapi` | `>=0.100,<1.0` | Stable API |
| `pydantic` | `>=2.0,<3.0` | v2 is used (UnifiedResponse, model_validate) |
| Everything else | `>=` lower bound | Let pip resolve compatible versions |

### 9.4 Setup Process

```bash
# 1. Create environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch (CUDA-specific, do first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install everything else
pip install -r requirements.txt

# 4. Clone third-party repos
git clone https://github.com/SysCV/sam-hq.git third_party/sam-hq
# needs manual review: record exact commit hash

# 5. Download model weights (documented in README table)
# models/detectors/yolov8n_deepfashion2_13cls_best.pt
# checkpoints/sam_hq/sam_hq_vit_b.pth
# models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt
# models/attributes/*.pth (8 files)
# models/grounding_dino_tiny/ (HuggingFace download)

# 6. Verify imports
PYTHONPATH=. python -c "from tools.infer.garment_pipeline import GarmentPipeline; print('OK')"
PYTHONPATH=. python -c "from inference.serving.app import app; print('OK')"
PYTHONPATH=. python -c "from inference.wrappers.sam_wrapper import SamHqWrapper; print('OK')"
pytest tests/ -q
```

---

## 10. Git-Safe Execution Order

### 10.0 Pre-Migration (NO file changes)

- [ ] **Step 0a**: Read both inventory reports completely. Done.
- [ ] **Step 0b**: Run `diff -rq` between D drive and Linux `src/fashion_vision/` to catalog actual differences. **needs manual execution.**
- [ ] **Step 0c**: Run `diff -rq` between D drive and Linux `configs/` to confirm only path differences. **needs manual execution.**
- [ ] **Step 0d**: Run `diff -rq` between D drive and Linux `tests/` to find any Linux-only tests. **needs manual execution.**
- [ ] **Step 0e**: Verify which commit of `third_party/sam-hq/` is cloned. Record hash. **needs manual execution.**
- [ ] **Step 0f**: Verify all 8 attribute classifier checkpoints exist at `models/attributes/`. **needs manual execution.**

### 10.A Recommended Copy Strategy

Use these command templates as a starting point. Paths must be adjusted to match the actual source locations on your machine.

#### 10.A.1 Reference Snapshots (Phase 0)

Archive excluded D drive items into `external/d_drive_original/`:

```bash
# D drive source root (adjust to actual path)
D_SRC="/mnt/d/Aliintern/fashion-ai"

# Create archive directory
mkdir -p external/d_drive_original

# Archive training tools
rsync -av --progress "$D_SRC/tools/train/" external/d_drive_original/tools/train/
rsync -av --progress "$D_SRC/tools/eval/" external/d_drive_original/tools/eval/
rsync -av --progress "$D_SRC/tools/data/" external/d_drive_original/tools/data/
rsync -av --progress "$D_SRC/tools/analysis/" external/d_drive_original/tools/analysis/

# Archive one-off scripts (illustrative — adjust globs per §7.2)
rsync -av --progress --include='analyze_*.py' --include='compare_*.py' \
    --include='calibrate_*.py' --include='filter_*.py' --include='inspect_*.py' \
    --include='build_*_train.py' --include='convert_*.py' --include='make_*.py' \
    --include='rebuild_*.py' --include='eval_*.py' --include='run_*_eval*.py' \
    --include='summarize_*.py' --include='run_*_visual_test.py' \
    --include='visualize_*.py' --exclude='*' \
    "$D_SRC/scripts/" external/d_drive_original/scripts/

# Archive old docs
rsync -av --progress \
    --include='周报*.md' --include='phase2_engineering_review.md' \
    --include='industrial_grounding_implementation_plan.md' \
    --include='P0a_*.md' --include='P1_*.md' --include='architecture/' \
    --exclude='*' \
    "$D_SRC/docs/" external/d_drive_original/docs/
```

Archive excluded Linux items into `external/linux_inference_original/`:

```bash
# Linux source root (adjust to actual path)
L_SRC="/home/charlie/fashion_final_project.linux"

mkdir -p external/linux_inference_original

# Archive Linux duplicates of D drive code (diff reference)
rsync -av --progress "$L_SRC/src/fashion_vision/" external/linux_inference_original/src/fashion_vision/
rsync -av --progress "$L_SRC/tools/" external/linux_inference_original/tools/
rsync -av --progress "$L_SRC/inference/serving/" external/linux_inference_original/inference/serving/
rsync -av --progress "$L_SRC/inference/benchmarks/" external/linux_inference_original/inference/benchmarks/
rsync -av --progress "$L_SRC/inference/pipelines/" external/linux_inference_original/inference/pipelines/
rsync -av --progress "$L_SRC/inference/export/" external/linux_inference_original/inference/export/
rsync -av --progress "$L_SRC/inference/llm/" external/linux_inference_original/inference/llm/
rsync -av --progress "$L_SRC/inference/optimized/" external/linux_inference_original/inference/optimized/
rsync -av --progress "$L_SRC/inference/engines/" external/linux_inference_original/inference/engines/
rsync -av --progress "$L_SRC/tests/" external/linux_inference_original/tests/
rsync -av --progress "$L_SRC/docs/" external/linux_inference_original/docs/
rsync -av --progress "$L_SRC/configs/" external/linux_inference_original/configs/

# Archive remaining Linux scripts (excluding the ones being migrated)
rsync -av --progress --exclude='run_full_31x_pipeline.py' \
    --exclude='build_31x_product_demo.py' \
    "$L_SRC/scripts/" external/linux_inference_original/scripts/
```

#### 10.A.2 D Drive Baseline Copy (Phase 1)

```bash
D_SRC="/mnt/d/Aliintern/fashion-ai"
DEST="."   # project root (fashion_final_project/)

# Core library (verbatim)
rsync -av --progress "$D_SRC/src/fashion_vision/" src/fashion_vision/

# Pipeline scripts (verbatim)
rsync -av --progress "$D_SRC/tools/infer/" tools/infer/
rsync -av --progress "$D_SRC/tools/crop/" tools/crop/

# Serving layer (verbatim)
rsync -av --progress "$D_SRC/inference/serving/" inference/serving/

# Configs (verbatim — path-clean happens in Phase 3)
rsync -av --progress "$D_SRC/configs/" configs/

# Tests, eval, label data (verbatim)
rsync -av --progress "$D_SRC/tests/" tests/
rsync -av --progress "$D_SRC/eval/" eval/
rsync -av --progress "$D_SRC/data/fashionai_attribute_index/" data/fashionai_attribute_index/

# Curated demo scripts
cp "$D_SRC/scripts/run_qa_orchestrator.py" scripts/
cp "$D_SRC/scripts/demo_collar_qa.py" scripts/

# Top-level project files
cp "$D_SRC/README.md" .
cp "$D_SRC/CLAUDE.md" .
cp "$D_SRC/.env.example" .
cp "$D_SRC/.gitignore" .
cp "$D_SRC/pytest.ini" .
```

**Important**: All copies are verbatim. No edits at this stage. The `.env.example` and YAML configs still contain `D:/Aliintern` paths — those are fixed in Phase 3.

#### 10.A.3 Linux Optimization Copy (Phase 2)

```bash
L_SRC="/home/charlie/fashion_final_project.linux"
DEST="."

# Optimization wrappers
mkdir -p inference/wrappers
cp "$L_SRC/inference/wrappers/sam_wrapper.py" inference/wrappers/
cp "$L_SRC/inference/wrappers/sam_wrapper_factory.py" inference/wrappers/
cp "$L_SRC/inference/wrappers/sam_encoder_trt.py" inference/wrappers/
cp "$L_SRC/inference/wrappers/sam_hq_trt_wrapper.py" inference/wrappers/
cp "$L_SRC/inference/wrappers/yolo_wrapper.py" inference/wrappers/

# Create __init__.py for the wrappers package
cat > inference/wrappers/__init__.py << 'EOF'
# Linux inference optimization wrappers.
# Primary: sam_wrapper.py (FP16 default).
# Reference: sam_encoder_trt.py, sam_hq_trt_wrapper.py (TRT, benchmark only).
# Reference: yolo_wrapper.py (TRT path NotImplementedError).
EOF

# Benchmark utilities
cp "$L_SRC/inference/env_capture.py" inference/
cp "$L_SRC/inference/latency_taxonomy.py" inference/

# Unique config
cp "$L_SRC/configs/serving_config.yaml" configs/

# Demo scripts
cp "$L_SRC/scripts/run_full_31x_pipeline.py" scripts/
cp "$L_SRC/scripts/build_31x_product_demo.py" scripts/

# Benchmark scripts
mkdir -p scripts/benchmarks
cp "$L_SRC/inference/benchmarks/bench_trt_vs_pytorch_sam.py" scripts/benchmarks/
cp "$L_SRC/inference/benchmarks/benchmark_runner.py" scripts/benchmarks/
cp "$L_SRC/inference/benchmarks/validate_deepfashion2_trt_accuracy.py" scripts/benchmarks/
```

**Important**: No D drive files are overwritten by this phase. All destinations are either new files under `inference/wrappers/` or new files under `scripts/` that don't exist in the D drive baseline.

#### 10.A.4 Benchmark Report Copy (Phase 5)

```bash
L_SRC="/home/charlie/fashion_final_project.linux"

mkdir -p results/benchmark

cp "$L_SRC/outputs/tensorrt_spike/encoder/bench_report_v2.md" results/benchmark/trt_encoder_bench_report_v2.md
cp "$L_SRC/outputs/tensorrt_spike/encoder/final_encoder_validation_summary.md" results/benchmark/trt_encoder_validation_summary.md
cp "$L_SRC/docs/reports/pipeline_benchmark_500_report.md" results/benchmark/pipeline_benchmark_500_report.md
cp "$L_SRC/docs/reports/query_region_batch60_report.md" results/benchmark/query_region_batch60_report.md
cp "$L_SRC/outputs/tensorrt_spike/encoder/bench_report_v2.json" results/benchmark/bench_report_v2.json
```

#### 10.A.5 General Notes

- **Always use `rsync -av` for directories** — it preserves timestamps and shows progress. `cp -r` is an acceptable fallback if `rsync` is unavailable.
- **Always verify after each copy**: `ls <dest>` to confirm files landed.
- **If a source path does not exist**, stop and resolve. Do not skip silently — a missing source means the inventory or path assumption is wrong.
- **The `$D_SRC` and `$L_SRC` variables are placeholders.** Replace with actual paths before running. Verify with `ls "$D_SRC/src/fashion_vision"` and `ls "$L_SRC/inference/wrappers"` first.

### 10.1 Phase 0: Reference Snapshots (archive only)

Purpose: Preserve everything not being migrated as read-only snapshots. No business code is modified.

- [ ] **Step 0.1a**: Create `external/d_drive_original/` with archive listing of excluded D drive items (§7.2).
- [ ] **Step 0.1b**: Create `external/linux_inference_original/` with archive listing of excluded Linux items (§7.4).
- [ ] **Step 0.1c**: Copy archived files into their respective `external/` directories (or create manifest files pointing to original locations if space is a concern).
- [ ] **Step 0.1d**: Git commit: `"docs: reference archive manifests"`.

### 10.2 Phase 1: Copy D Drive Baseline (read-only copy, no changes)

Purpose: Establish the functional baseline in its original layout.

- [ ] **Step 1a**: Copy `src/fashion_vision/` from D drive → `src/fashion_vision/` (verbatim).
- [ ] **Step 1b**: Copy `tools/infer/` from D drive → `tools/infer/` (verbatim).
- [ ] **Step 1c**: Copy `tools/crop/` from D drive → `tools/crop/` (verbatim).
- [ ] **Step 1d**: Copy `inference/serving/` from D drive → `inference/serving/` (verbatim, all 18 files).
- [ ] **Step 1e**: Copy `configs/` from D drive → `configs/` (verbatim, path-clean later in Phase 3).
- [ ] **Step 1f**: Copy `tests/` from D drive → `tests/` (verbatim).
- [ ] **Step 1g**: Copy `eval/` from D drive → `eval/` (verbatim).
- [ ] **Step 1h**: Copy `data/fashionai_attribute_index/` from D drive → `data/fashionai_attribute_index/`.
- [ ] **Step 1i**: Copy `scripts/run_qa_orchestrator.py` and `scripts/demo_collar_qa.py` from D drive → `scripts/`.
- [ ] **Step 1j**: Copy top-level files from D drive: `README.md`, `CLAUDE.md`, `.env.example`, `.gitignore`, `pytest.ini`.
- [ ] **Step 1k**: Git commit: `"feat: D drive baseline — core library, pipeline, serving, tests, configs"`.

### 10.3 Phase 2: Add Linux Optimization Modules

Purpose: Add the Linux inference wrappers into the `inference/` namespace. No existing files are overwritten.

- [ ] **Step 2a**: Create `inference/wrappers/__init__.py`.
- [ ] **Step 2b**: Copy Linux `inference/wrappers/sam_wrapper.py` → `inference/wrappers/sam_wrapper.py`.
- [ ] **Step 2c**: Copy Linux `inference/wrappers/sam_wrapper_factory.py` → `inference/wrappers/sam_wrapper_factory.py`.
- [ ] **Step 2d**: Copy Linux `inference/wrappers/sam_encoder_trt.py` → `inference/wrappers/sam_encoder_trt.py`.
- [ ] **Step 2e**: Copy Linux `inference/wrappers/sam_hq_trt_wrapper.py` → `inference/wrappers/sam_hq_trt_wrapper.py`.
- [ ] **Step 2f**: Copy Linux `inference/wrappers/yolo_wrapper.py` → `inference/wrappers/yolo_wrapper.py`.
- [ ] **Step 2g**: Copy Linux `inference/env_capture.py` → `inference/env_capture.py`.
- [ ] **Step 2h**: Copy Linux `inference/latency_taxonomy.py` → `inference/latency_taxonomy.py`.
- [ ] **Step 2i**: Copy Linux `configs/serving_config.yaml` → `configs/serving_config.yaml`.
- [ ] **Step 2j**: Copy Linux demo scripts: `scripts/run_full_31x_pipeline.py`, `scripts/build_31x_product_demo.py`.
- [ ] **Step 2k**: Create `scripts/benchmarks/` and copy 3 benchmark scripts (§3, Category B).
- [ ] **Step 2l**: Git commit: `"feat: Linux inference optimization — wrappers, env_capture, latency_taxonomy, demos"`.

### 10.4 Phase 3: Path Cleanup

Purpose: Fix hardcoded `D:/Aliintern` paths in configs and `.env.example`. No code logic changes.

- [ ] **Step 3a**: Path-clean `configs/dataset/deepfashion2.yaml` — replace `D:/Aliintern/fashion-ai-data/deepfashion2` with `${DEEPFASHION2_ROOT}`.
- [ ] **Step 3b**: Path-clean `configs/model/sam_hq.yaml` — replace Windows checkpoint path with `${SAM_HQ_CHECKPOINT}`.
- [ ] **Step 3c**: Path-clean `configs/inference/sam_box_prompt.yaml` — replace Windows paths with env var references.
- [ ] **Step 3d**: Rewrite `.env.example` with WSL2/Linux paths (§8.4).
- [ ] **Step 3e**: Optional: fix docstring paths in `scripts/run_full_31x_pipeline.py` and `scripts/run_qa_orchestrator.py` if they contain `D:/Aliintern`.
- [ ] **Step 3f**: Verify: `grep -r "D:/Aliintern" configs/ .env.example` must return ZERO hits.
- [ ] **Step 3g**: Git commit: `"fix: path-clean configs and .env.example for WSL2/Linux"`.

### 10.5 Phase 4: Create `run_demo.py` (new file only)

Purpose: Single root-level entry point. Imports from original locations. FP16 default, no TRT wiring.

- [ ] **Step 4a**: Create `run_demo.py` at project root per §6 design.
- [ ] **Step 4b**: Implement argument parsing, pipeline orchestration (imports from `tools.infer.garment_pipeline`, `inference.serving.qa_orchestrator`, `inference.wrappers.sam_wrapper`).
- [ ] **Step 4c**: Wire FP16 as default (`use_fp16=True`). TRT only via `--sam-backend tensorrt` opt-in.
- [ ] **Step 4d**: Git commit: `"feat: unified run_demo.py entry point with FP16 default"`.

### 10.6 Phase 5: Copy Benchmark Reports and Docs

Purpose: Presentation artifacts. No code.

- [ ] **Step 5a**: Copy key benchmark reports from Linux → `results/benchmark/` (§3, Category C).
- [ ] **Step 5b**: Copy both inventory reports to `docs/` (already done).
- [ ] **Step 5c**: Create `docs/architecture/` with a one-page architecture overview.
- [ ] **Step 5d**: Create `pyproject.toml` with package metadata.
- [ ] **Step 5e**: Create merged `requirements.txt` (§9).
- [ ] **Step 5f**: Create `requirements-frozen.txt` placeholder.
- [ ] **Step 5g**: Git commit: `"docs: benchmark reports, architecture overview, project metadata"`.

### 10.7 Phase 6: Import Checks and Tests

Purpose: Verify everything imports and existing tests still pass.

- [ ] **Step 6a**: Verify core import: `PYTHONPATH=. python -c "from tools.infer.garment_pipeline import GarmentPipeline; print('OK')"`.
- [ ] **Step 6b**: Verify serving import: `PYTHONPATH=. python -c "from inference.serving.app import app; print('OK')"`.
- [ ] **Step 6c**: Verify optimized wrapper import: `PYTHONPATH=. python -c "from inference.wrappers.sam_wrapper import SamHqWrapper; print('OK')"`.
- [ ] **Step 6d**: Verify `run_demo.py` import: `PYTHONPATH=. python -c "import run_demo; print('OK')"`.
- [ ] **Step 6e**: Run test suite: `pytest tests/ -q --tb=short`.
- [ ] **Step 6f**: Fix any import errors or test failures discovered.
- [ ] **Step 6g**: Git commit: `"fix: import verification and test fixes"`.

---

## 11. Manual Review Checklist

### 11.1 Pre-Migration Verification

- [ ] **R1**: Diff D drive `src/fashion_vision/` vs Linux `src/fashion_vision/`. Are there any Linux-specific bug fixes worth keeping? Or are they byte-identical?
- [ ] **R2**: Do the 8 attribute classifier checkpoints exist at `models/attributes/`? Verify paths: `collar_design`, `lapel_design`, `neckline_design`, `neck_design`, `sleeve_length`, `coat_length`, `pant_length`, `skirt_length`.
- [ ] **R3**: What is the exact commit hash of `third_party/sam-hq/` in the Linux project? Document for reproducible setup.
- [ ] **R4**: Does `groundingdino` install from pip or must it be loaded from local `models/grounding_dino_tiny/`? Verify import mechanism.
- [ ] **R5**: Are there any Linux-only test files in Linux `tests/` that don't exist in D drive `tests/`? If so, migrate them during Phase 2.

### 11.2 Schema Alignment

- [ ] **R6**: Verify actual `build_instance_record()` output format — does it use `score` or `confidence`? Does it have `pred_mask_path` or `mask_path`?
- [ ] **R7**: Verify actual `locate_region()` output format — does it return `part` or `part_type`? Is `all_bboxes` + `all_scores` parallel arrays or a list of objects?
- [ ] **R8**: Verify actual attribute pipeline output format — is `topk` a list of `{value: confidence}` dicts with variable keys, or a list of `{value, confidence}` objects?
- [ ] **R9**: Verify `GarmentPipeline.run_image()` return type. Is it a dict? A dataclass? Does it include timing info?

### 11.3 Import Path Audit

- [ ] **R10**: Verify that `tools/infer/garment_pipeline.py` and `inference/serving/` internal imports work with `PYTHONPATH=.` from project root.
- [ ] **R11**: Verify that `inference/wrappers/sam_wrapper.py` can be imported standalone (it likely imports from `src/fashion_vision/` — confirm this path resolves).
- [ ] **R12**: Verify `pytest.ini` `pythonpath` setting — confirm `./src` and `./tools` are covered.

### 11.4 Runtime Verification

- [ ] **R13**: Can `GarmentPipeline` be imported cleanly? `python -c "from tools.infer.garment_pipeline import GarmentPipeline"`
- [ ] **R14**: Can `SamHqWrapper` (optimized) be imported? `python -c "from inference.wrappers.sam_wrapper import SamHqWrapper"`
- [ ] **R15**: Can the FastAPI app be imported? `python -c "from inference.serving.app import app"`
- [ ] **R16**: Can `run_demo.py` be imported? `python -c "import run_demo"`
- [ ] **R17**: Run a single-image pipeline test with `--no-attributes --no-visualization` to verify the core path.
- [ ] **R18**: Run `pytest tests/ -q --tb=short` and confirm all 373+ tests still pass (or document which fail and why).

### 11.5 Path Cleanup Verification

- [ ] **R19**: `grep -r "D:/Aliintern" configs/ .env.example` must return ZERO hits.
- [ ] **R20**: `grep -r "D:/Aliintern" src/ tools/ inference/ scripts/` — any hits in source files (excluding docstrings) must be fixed or documented.

### 11.6 Presentation Readiness

- [ ] **R21**: Can `scripts/build_31x_product_demo.py` run from the migrated project and produce a self-contained HTML report?
- [ ] **R22**: Are all key benchmark numbers (TRT 4.02x, FP16 1.64x, batch 6.54x, IoU 0.945) present in `results/benchmark/` with source attribution?
- [ ] **R23**: Does `run_demo.py` produce valid JSON matching the unified schema (§5)?
- [ ] **R24**: Can the FastAPI server start and respond to `/v1/health` with mock vision backend?

---

## Appendix A: Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Keep original layout | Yes — `tools/infer/`, `tools/crop/`, `inference/serving/`, `configs/` unchanged | Zero import breakage. Zero risk of path rewrite bugs. Internship closing deadline. |
| Primary SAM wrapper | Linux `inference/wrappers/sam_wrapper.py` (FP16) | 1.64x speedup, batched boxes, production-ready. Slots into existing `inference/` namespace. |
| FP16 as default | Yes, `use_fp16=True` | <2% IoU drift is acceptable; FP32 available as fallback. |
| TRT in demo pipeline | **Not wired.** Benchmark/docs only. | Validated but requires `.engine` file on target GPU. Present as static benchmark numbers. |
| Pipeline orchestrator | D drive `tools/infer/garment_pipeline.py` | Function-call API, lazy loading, 373+ tests, all 6 stages. At original path. |
| Serving canonical | D drive `inference/serving/` | Feature-complete, test-covered, deterministic. At original path. |
| Config canonical | D drive `configs/` (path-cleaned) | Functional baseline; add Linux `serving_config.yaml`. |
| Test canonical | D drive `tests/` | 373+ passing; merge any Linux-only tests after diff. |
| Entry point | One new root-level `run_demo.py` | Thin wrapper. Imports from original locations. Does not replace `run_full_31x_pipeline.py` or `app.py`. |
| No `src/pipeline/` or `src/serving/` | Deferred | Architectural cleanup is real but too risky for internship closing. Do in a follow-up branch. |
| `score` vs `confidence` | **`confidence`** everywhere | More intuitive for presentation. Alias `score` for backward compat. |
| `part` vs `part_type` | **`part_type`** | More descriptive. Alias `part` for backward compat. |
| `pred_mask_path` | **`mask_path`** | Shorter. Consistent with `mask_present`. |
| Reference snapshots | `external/d_drive_original/` and `external/linux_inference_original/` | Read-only archives. Not imported at runtime. |

## Appendix B: Files Touched Per Phase

| Phase | Files Created/Modified | Risk |
|---|---|---|
| Phase 0: Reference snapshots | `external/` (new, archive only) | None |
| Phase 1: D drive baseline | `src/`, `tools/`, `inference/serving/`, `configs/`, `tests/`, `eval/`, `data/`, `scripts/`, top-level files (verbatim copy) | None |
| Phase 2: Linux optimization | `inference/wrappers/`, `inference/env_capture.py`, `inference/latency_taxonomy.py`, `configs/serving_config.yaml`, `scripts/run_full_31x_pipeline.py`, `scripts/build_31x_product_demo.py`, `scripts/benchmarks/` (new files only) | None |
| Phase 3: Path cleanup | `configs/dataset/deepfashion2.yaml`, `configs/model/sam_hq.yaml`, `configs/inference/sam_box_prompt.yaml`, `.env.example` (path strings only) | Low |
| Phase 4: run_demo.py | `run_demo.py` (new file only) | Low |
| Phase 5: Benchmark reports | `results/benchmark/`, `docs/architecture/`, `pyproject.toml`, `requirements.txt`, `requirements-frozen.txt` (new + doc files) | None |
| Phase 6: Verification | Import checks + test run (no file changes unless fixes needed) | Low |

**No existing file is moved or renamed across the entire plan.** The only edits are path-string replacements in 3 YAML files and `.env.example`. The only new business-code files are under `inference/wrappers/` and root-level `run_demo.py`.

## Appendix C: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Linux `src/fashion_vision/` has important bug fixes not in D drive | Low | Medium | Diff before copying (R1). Apply any fixes manually to canonical D drive copy. |
| `inference/wrappers/sam_wrapper.py` has import dependencies needing `PYTHONPATH` adjustment | Medium | Low | Verify import in Phase 6. Add `PYTHONPATH` entry if needed. |
| `configs/` YAML env var substitution not supported by config loader | Medium | Medium | Check before path cleanup (R19). Fallback: use relative paths + document working-directory requirement. |
| Attribute checkpoints missing or misnamed | Medium | Medium | Verify before migration (R2). Attribute pipeline is optional (`--no-attributes`). |
| `groundingdino` import mechanism unclear | Low | High | Verify before migration (R4). Fallback: document as optional, demo works without it. |
| Linux-only tests lost if we only copy D drive `tests/` | Low | Medium | Diff `tests/` before copying (R5). Merge any unique Linux tests during Phase 2. |
| `build_31x_product_demo.py` hardcodes paths to pipeline output | Medium | Low | Review before migration (R21). Fix paths if needed. |
| TRT `.engine` file not rebuildable on target GPU | Medium | Low | TRT not wired into demo. Benchmark results are static reports in `results/benchmark/`. |
| `run_demo.py` import path clashes with existing namespace | Low | Low | Root-level script, imports via `PYTHONPATH=.`. Verify in Phase 6. |
