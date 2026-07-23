# Fashion Fine-Grained Vision

End-to-end fashion visual analysis pipeline: garment detection & segmentation → language-guided local region localization → fine-grained attribute extraction → QA with RAG knowledge enhancement.

Aligned with PRD 3.1 (3.1.1 / 3.1.2 / 3.1.3) and 3.3 (Agent + RAG).

---

## What this project does

Given a fashion image and an optional natural-language query, the pipeline outputs:

1. **Garment instances** — bbox, mask, and category (13 fine classes internally, 5 coarse classes for PRD output)
2. **Local region localization** — find any garment part (collar, zipper, pocket, ruffle, sequin…) by Chinese or English query
3. **Attribute classification** — collar design, sleeve length, coat length, etc. (8 FashionAI tasks)
4. **Natural-language QA** — template-based answers with visual evidence (collar QA golden path)
5. **RAG knowledge enhancement** — intent classification (15+ types) + structured knowledge retrieval

```
Image + Query ("领口是什么设计？")
    │
    ▼
YOLOv8n garment detection (13-class)
    │
    ▼
SAM-HQ instance segmentation
    │
    ├──► Landmark prediction (39 keypoints)
    │        │
    │        ▼
    │    Fast-path localization (collar/sleeve/hem/waist/pant_leg)
    │
    ├──► Fashionpedia YOLOv8s part detector (zipper/pocket/button/lapel/…)
    │
    └──► Grounding DINO open-vocab (ruffle/sequin/fringe/…)
             │
             ▼
         Unified router → bbox + mask for requested part
             │
             ▼
         Attribute classifiers (ResNet18 × 8 tasks)
             │
             ▼
         QaOrchestrator → template answer + evidence crops
```

---

## Features

### 3.1.1 Garment instance segmentation
- YOLOv8n 13-class detector with class-balanced retraining
- SAM-HQ vit_b instance segmentation
- Dual-label output: fine (13-class) for internal use, coarse (5-class: top/pants/skirt/outerwear/dress) for PRD
- 39-point landmark prediction per garment instance
- **FP16 autocast** — 1.64x SAM speedup with <2% mask IoU drift

### 3.1.2 Language-guided local region localization
- **3-backend routing**: fast-path (landmark + geometry) → Fashionpedia YOLO → Grounding DINO
- **26-part bilingual vocabulary**: Chinese + English queries (领口/collar, 拉链/zipper, 荷叶边/ruffle…)
- **30+ per-part detection configs**: individually tuned DINO prompts, thresholds, and shape priors
- **Mask-gated inference**: non-garment pixels filled to suppress background
- **Anatomical zoom**: sub-region 2× magnification for small parts
- **Shape prior filtering**: area ratio, aspect ratio, vertical/horizontal band constraints
- **Spatial constraints**: left/right side selection, upper/lower direction filtering
- **Inner garment detection**: SAM multimask for finding inner layers under outerwear

### 3.1.3 Fine-grained attribute extraction
- 8 FashionAI tasks: collar_design, lapel_design, neckline_design, neck_design, sleeve_length, coat_length, pant_length, skirt_length
- Config-driven pipeline (`configs/attribute_inference.yaml`)
- Category gating: only relevant tasks run per garment type
- ResNet18 classifiers with multi-view crop augmentation
- **P0 Collar QA golden path** — end-to-end from query to NL answer with evidence crops (see `scripts/demo_collar_qa.py`)

### 3.3 Agent & RAG (P0a)
- **Rule-based intent classifier** — 15+ intent types, YAML config driven (`configs/intent_taxonomy.yaml`)
- **Structured knowledge base** — 1200+ fabrics, 300+ crafts, 200+ terms (`configs/knowledge_base.yaml`)
- **Multi-layer retrieval** — exact match → alias match → title match → BM25 (`configs/retrieval_config.yaml`)
- **QaOrchestrator** — unified dispatch: attribute_query / knowledge_qa / region_query / visual_instance
- **Serving layer** — FastAPI skeleton with mock + real vision provider adapter (`inference/serving/app.py`)

### Inference optimization (ongoing)
- **SAM-HQ FP16 autocast** — 1.64x speedup measured (275ms → 168ms), see `docs/sam_hq_optimization_findings.md`
- **Batched box prediction** — `SamHqWrapper.predict_boxes(N,4)` for multi-garment images
- **Call pattern analysis** — confirmed pipeline uses 1× set_image + N× predict (correct pattern)
- **Full optimization plan** — `docs/inference_optimization_plan_v3.md`

---

## Repository Structure

```
fashion-finegrained-vision/
├── configs/
│   ├── category_mapping.yaml           # 13→5 class mapping
│   ├── attribute_inference.yaml        # attribute task registry (8 tasks)
│   ├── attribute_group_mapping.yaml    # category→task→region mapping
│   ├── attribute_templates.yaml        # NL answer templates (3.3)
│   ├── attribute_eval_targets.yaml     # eval target config
│   ├── attribute_taxonomy.yaml         # attribute class taxonomy
│   ├── intent_taxonomy.yaml            # 15+ intent types (3.3.1)
│   ├── knowledge_base.yaml             # 1200+ knowledge entries (3.3.2)
│   ├── knowledge_schema.md             # KB schema documentation
│   ├── retrieval_config.yaml           # RAG retrieval config (3.3.2)
│   └── serving_config.yaml            # serving layer feature flags
│
├── src/fashion_vision/
│   ├── data/class_mapping.py           # dual-label class mapping
│   ├── models/sam_hq_wrapper.py        # SAM-HQ wrapper
│   ├── schemas/instance_schema.py      # instance data schema
│   ├── attributes/                     # 3.1.3: attribute inference
│   │   ├── garment_attribute_pipeline.py
│   │   ├── mask_attribute_pipeline.py
│   │   ├── category_gate.py
│   │   └── task_registry.py
│   ├── localization/                   # 3.1.2: local region localization
│   │   ├── region_localization_router.py   # unified routing entry point
│   │   ├── intent_parser.py               # NL query → structured intent
│   │   ├── fashionpedia_part_detector.py   # Fashionpedia YOLOv8s (19-class)
│   │   ├── grounding_dino_locator.py       # Grounding DINO open-vocab
│   │   ├── part_detection_config.py        # 30+ per-part configs
│   │   ├── part_shape_priors.py            # geometric prior filters
│   │   ├── anatomical_zoom.py              # sub-region magnification
│   │   ├── spatial_constraint.py           # left/right, upper/lower
│   │   ├── bbox_mask_refiner.py            # SAM box→mask refinement
│   │   ├── garment_ref_filter.py           # garment type match check
│   │   ├── inner_garment_detector.py       # inner layer detection
│   │   └── viz_utils.py                    # debug visualization
│   └── utils/crop_utils.py
│
├── tools/
│   ├── infer/                          # inference entry points
│   │   ├── garment_pipeline.py         # end-to-end pipeline (stages 1-6)
│   │   └── predict_garments_yolo.py    # YOLO inference
│   ├── demo/                           # interactive demos
│   ├── eval/                           # evaluation toolset
│   ├── train/                          # training scripts
│   └── data/                           # data preparation
│
├── inference/                           # optimization workspace (v3)
│   ├── benchmarks/                      # SAM FP16, batch boxes, call pattern
│   │   ├── bench_sam_fp16.py
│   │   ├── bench_sam_batch_boxes.py
│   │   ├── bench_sam_call_pattern.py
│   │   └── visualize_sam_fp16_drift.py
│   ├── wrappers/
│   │   └── sam_wrapper.py              # SamHqWrapper (autocast + batched predict)
│   ├── optimized/
│   │   └── segment_garments_sam_optimized.py
│   ├── serving/                        # 3.3 serving layer (FastAPI + orchestrator)
│   │   ├── app.py                      # FastAPI app
│   │   ├── qa_orchestrator.py          # unified QA dispatch
│   │   ├── intent_classifier.py        # rule-based intent classifier
│   │   ├── attribute_service.py        # attribute lookup + template answer
│   │   ├── rag_service.py              # knowledge retrieval
│   │   ├── region_backend.py           # 3.1.2 backend (Fashionpedia + full)
│   │   ├── region_query_mapper.py      # Chinese query → part type
│   │   ├── subprocess_vision_provider.py  # P0 subprocess pipeline provider
│   │   ├── real_vision_provider.py     # experimental 3.1 adapter
│   │   ├── vision_provider.py          # VisionAttributeProvider interface
│   │   ├── vision_context.py           # attribute merge logic
│   │   └── schemas.py                  # Pydantic models
│   ├── pipelines/                      # optimized fast-path implementations
│   ├── engines/                        # TensorRT engine registry
│   ├── llm/                            # embedding retriever, vocab router
│   ├── export/                         # ONNX export utilities
│   └── deployment/                     # Docker / deployment configs
│
├── scripts/                            # analysis, demo, batch scripts
│   ├── demo_collar_qa.py               # P0 collar QA demo (3 images)
│   ├── run_qa_orchestrator.py          # CLI: fast/slow path QA
│   ├── run_full_31x_pipeline.py        # batch pipeline runner
│   ├── build_31x_product_demo.py       # HTML demo generator
│   └── ...
│
├── tests/                              # 373+ tests (serving, eval, pipeline)
│   ├── test_serving/                   # orchestrator, intent, RAG, vision
│   └── test_eval/                      # evaluation framework
│
├── docs/                               # engineering plans and reports
│   ├── inference_optimization_plan_v3.md       # comprehensive optimization plan
│   ├── sam_hq_optimization_findings.md          # SAM-HQ benchmark results
│   ├── project_status.md                        # current project status
│   └── ...
│
├── CLAUDE.md
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Charlie-Zhou-Xingyu/fashion-finegrained-vision.git
cd fashion-finegrained-vision
pip install -r requirements.txt
```

Key dependencies: `torch`, `ultralytics`, `transformers`, `opencv-python`, `numpy`, `pillow`, `matplotlib`, `pytest`.

### SAM-HQ (separate clone)

SAM-HQ is gitignored and must be cloned separately:

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/SysCV/sam-hq.git
cd ..
```

### WSL2 / Linux setup

For CUDA optimization work on WSL2:

```bash
cd ~/projects
git clone https://github.com/Charlie-Zhou-Xingyu/fashion-finegrained-vision.git
cd fashion-finegrained-vision
git checkout -b optimize/samhq-trt

# Clone SAM-HQ
mkdir -p third_party && cd third_party
git clone https://github.com/SysCV/sam-hq.git && cd ..

# Create conda environment
conda create -n samhq-trt python=3.10 -y && conda activate samhq-trt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install opencv-python segment-anything ultralytics onnx onnxsim onnxruntime-gpu

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python inference/benchmarks/bench_sam_fp16.py --num-images 3 --n-bench 5
```

---

## Required Model Weights

Model weights are **not included**. Prepare these files locally:

| Model | Path |
|---|---|
| YOLO garment detector | `models/detectors/yolov8n_deepfashion2_13cls_best.pt` |
| SAM-HQ vit_b | `checkpoints/sam_hq/sam_hq_vit_b.pth` |
| Landmark predictor | `outputs/landmark_predictor_resnet18/best.pt` |
| Fashionpedia part detector | `models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt` |
| Attribute classifiers (×8) | `outputs/p2_*_resnet18_*/best.pt` |

Grounding DINO-tiny is downloaded automatically from HuggingFace Hub on first use.

---

## Quick Start

### SAM-HQ optimization benchmarks

```bash
# FP32 vs FP16 benchmark
python inference/benchmarks/bench_sam_fp16.py --num-images 5 --n-bench 10

# One-by-one vs batched predict
python inference/benchmarks/bench_sam_batch_boxes.py --n-bench 10

# Call pattern: naive vs one-set-image vs batched
python inference/benchmarks/bench_sam_call_pattern.py --n-bench 10
```

### P0 Collar QA demo

```bash
# Fast path (pre-processed images)
python scripts/run_qa_orchestrator.py --image-id 000088 \
    --query "这件衣服的领口是什么设计？"

# Slow path (new image — runs full pipeline from scratch)
python scripts/run_qa_orchestrator.py \
    --image D:/path/to/image.jpg --slow \
    --query "这件衣服的领口是什么设计？"
```

### Full garment pipeline (3.1.1)

```bash
python tools/infer/garment_pipeline.py \
  --source assets/examples/ \
  --output-dir outputs/demo \
  --yolo-weights models/detectors/yolov8n_deepfashion2_13cls_best.pt \
  --sam-checkpoint checkpoints/sam_hq/sam_hq_vit_b.pth \
  --sam-model-type vit_b \
  --landmark-checkpoint outputs/landmark_predictor_resnet18/best.pt
```

### With attribute inference (3.1.1 + 3.1.3)

```python
from tools.infer.garment_pipeline import GarmentPipelineConfig, run_pipeline

config = GarmentPipelineConfig(
    yolo_weights="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
    sam_checkpoint="checkpoints/sam_hq/sam_hq_vit_b.pth",
    run_attribute_inference=True,
)
results = run_pipeline(config, image_paths=["image.jpg"])
```

### Region localization (3.1.2)

```python
from fashion_vision.localization.region_localization_router import locate_region
from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
from fashion_vision.localization.fashionpedia_part_detector import FashionpediaPartDetector

dino = GroundingDINOLocator()
fp = FashionpediaPartDetector("models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt")

result = locate_region(
    query="外套的拉链",
    instance=garment_instance_dict,
    image=image_bgr,
    locator=dino,
    fashionpedia_detector=fp,
)
```

---

## Testing

```bash
# Windows
set PYTHONPATH=%CD%\src
pytest -q

# Linux / macOS
PYTHONPATH=./src pytest -q
```

---

## Current Status

| Module | Status | Key metrics |
|---|---|---|
| 3.1.1 Garment segmentation | ✅ Complete | 35/35 images end-to-end pass, 5-class, 13→5 dual labels |
| 3.1.2 Region localization | ✅ Core complete | 5 landmark regions, Fashionpedia+DINO routing, 26-part vocab |
| 3.1.3 Attribute extraction | ✅ Baseline complete | 8 tasks, top3 output, category-gated |
| 3.3 Agent & RAG | ⏳ P0a complete | Intent classifier (15+ types), seed KB, QaOrchestrator skeleton |
| P0 Collar QA | ✅ Golden path | Fast + slow path, NL answer + evidence crops (3 images tested) |
| SAM-HQ FP16 optimization | ✅ Benchmarked | 1.64x speedup, <2% IoU drift, RTX 4060 |
| Tests | 373+ passed | |

### SAM-HQ benchmark results (RTX 4060 Laptop GPU)

| Mode | set_image | predict (2 boxes) | Total |
|------|-----------|-------------------|-------|
| FP32 | 266ms | 9ms | 275ms |
| **FP16 autocast** | **159ms** | 9ms | **168ms** |
| **Speedup** | **1.68x** | — | **1.64x** |

See `docs/sam_hq_optimization_findings.md` for full benchmark results, call pattern analysis, and drift visualizations.

### Primary bottleneck
SAM-HQ vit_b image encoder at ~266ms (FP32) / ~159ms (FP16). Full optimization plan at `docs/inference_optimization_plan_v3.md`.

### Known gaps
- Attribute F1 0.59-0.76 vs PRD 0.88 — root cause: training data scarcity (556-1,647 samples/task)
- Fabric/material attributes not yet trained — waiting for iMaterialist Fashion 2019 data
- Pocket/shoulder/pattern/decoration regions — Fashionpedia model trained but not wired into serving layer
- MLLM (Qwen-VL 7B) not integrated — server-side deployment planned
- Knowledge base entries pending domain expert review

---

## License

TODO.
