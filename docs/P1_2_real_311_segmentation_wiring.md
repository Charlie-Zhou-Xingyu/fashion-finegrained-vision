# P1.2 — Real 3.1.1 Instance Segmentation Wiring

> Status: experimental REAL 3.1.1 wiring (2026-07-16).
> YOLO detection + SAM-HQ segmentation stages ARE connected.
> Default provider is **still MockVisionAttributeProvider**.
> Real vision is **disabled by default**.
> This does NOT represent production vision capability.

## 1. What P1.2 delivers

- `FashionVision31SegmentationBackend` (`real_vision_provider.py`) — calls the
  REAL `GarmentPipeline` with `run_landmark_and_crops=False` (YOLO detection
  + SAM-HQ segmentation ONLY; stages 3-6 are always skipped).
  `probe()` checks YOLO+SAM checkpoints via filename existence and
  module `find_spec` — **nothing heavy is loaded at probe or constructor time**.
  Heavy imports (`torch`, `ultralytics`, `cv2`) happen only inside `predict()`,
  never at import.
- Per-request temp-file pipeline: image_bytes → `TemporaryDirectory` →
  `GarmentPipeline.run_image()` → parse `01_yolo/detections.json` +
  `02_samhq/segmentation_results.json` → normalize → cleanup in `finally`
  block (removable via `cleanup_temp_files=false` for debugging).
- Output: `garment_instances` list (per detection: `instance_id`, `category`
  / `fine_class_name`, `bbox`, `confidence`, `mask_present`, `mask_ref`
  placeholder — never mask bitmap or temp path), `sources`, `meta`.
- Feature-flag unchanged from P1.1: real provider requires both
  `vision.provider=real` AND `vision.real_enabled=true` (or env equivalents).
  Default remains mock.
- P1.1 `FashionVision31Backend` (probing-only shell) remains unchanged and
  is still the default for `backend: fashion_vision_3_1` without `mode`.

## 2. 3.1.1 execution details

- **Pipeline** — `tools/infer/garment_pipeline.py::GarmentPipeline.run_image()`
  was NOT modified.  `GarmentPipelineConfig(run_landmark_and_crops=False)`
  already exists and skips stages 3/4/5.
- **Checkpoints required** — `models/detectors/yolov8n_deepfashion2_13cls_best.pt`
  and `checkpoints/sam_hq/sam_hq_vit_b.pth` (both present locally).
  Landmark (`outputs/landmark_predictor_resnet18/best.pt`) and all 8
  `outputs/p2_*/best.pt` attribute checkpoints are **NOT required** and
  never loaded.
- **3.1.3 attribute classifiers** — NEVER run (no approach change).
- **CPU support** — YOLO and SAM both support CPU via device string.
  Device defaults to `cpu` for safety; server experiments can set
  `yolo_device=cuda`/`sam_device=cuda` (or `auto`) via config/env.
  Note: SAM on CPU is slow (~5-30s per image depending on resolution);
  10s default timeout may need adjustment for CPU experiments.
- **Env override**: `VISION_BACKEND_MODE=segmentation_only`,
  `VISION_DEVICE=cpu|cuda|auto` (sets both stage devices at once).

## 3. Input/output contract

- **Input**: `image_bytes` (raw bytes or base64 string; validated + decoded inside backend).
  `image_url` is NEVER downloaded — the provider blocks url-only input.
- **Output garment_instance shape**:
  ```json
  {
    "instance_id": "inst_0",
    "category": "top",
    "fine_class_name": "short sleeve top",
    "bbox": [10.0, 20.0, 100.0, 200.0],
    "bbox_format": "xyxy_abs_pixels",
    "confidence": 0.91,
    "mask_present": true,
    "mask_ref": "mask_inst_0",
    "source": "fashion_vision_3_1",
    "stage": "segmentation"
  }
  ```
- **Dual-label rule**: `category` uses the coarse (PRD 5-class) label;
  `fine_class_name` preserves the internal 13-class label.
- **Mask policy**: `mask_present` is a bool flag.  `mask_ref` is a
  non-sensitive placeholder (e.g. `mask_inst_0`) — never a file path and
  never a bitmap/array.  SAM `sam_score` is included when available.
- **Bbox validation**: instances with malformed bbox (not 4 numeric values)
  are dropped (counted in `meta.invalid_bbox_count`).

## 4. Response metadata (via `/v1/mm/qa`)

Keys surfaced in `response.data.meta` when the real provider is active:
`vision_backend`, `vision_backend_mode`, `vision_latency_ms`,
`num_garment_instances`, `mask_bitmap_returned`, `vision_provider_used`,
`vision_provider_name`, `visual_attributes_present`, `provided_attributes_used`,
`requested_regions`.

## 5. Testing

`tests/test_serving/test_real_vision_segmentation.py`:
- 11 default-safe tests (always run, no model/GPU/checkpoint).
- 2 optional tests gated by `RUN_REAL_VISION_TESTS=1` (real YOLO+SAM on one
  local fixture image).
- Default smoke test uses an injectable `pipeline_runner` callable — real
  model path is only exercised by the optional tests.

Run:
```bash
# default (safe)
python -m pytest tests/test_serving/test_real_vision_segmentation.py -q

# optional real
RUN_REAL_VISION_TESTS=1 python -m pytest tests/test_serving/test_real_vision_segmentation.py -q
```

## 6. Boundaries

No image_url download, no network, no LLM/MLLM, no Redis/FAISS/BGE, no KB
change, no RagService/ContentGenerationService change, no UnifiedResponse
change, no model training/download/commit, no mask bitmap, no temp path
leak, no 3.1.3 attribute classifiers, default mock provider unchanged,
real vision disabled by default.  This is an experimental local 3.1.1
wiring — NOT production-grade.  P2 would add model caching (current YOLO
and SAM reload every request), GPU optimization, production temp-file
strategy, and real-vision eval with labeled fixtures.
