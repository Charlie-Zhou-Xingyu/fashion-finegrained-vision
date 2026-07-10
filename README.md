# Fashion Fine-Grained Vision

End-to-end fashion visual analysis pipeline: garment detection & segmentation → language-guided local region localization → fine-grained attribute extraction.

Aligned with PRD 3.1 (3.1.1 / 3.1.2 / 3.1.3).

---

## What this project does

Given a fashion image and an optional natural-language query, the pipeline outputs:

1. **Garment instances** — bbox, mask, and category (13 fine classes internally, 5 coarse classes for PRD output)
2. **Local region localization** — find any garment part (collar, zipper, pocket, ruffle, sequin…) by Chinese or English query
3. **Attribute classification** — collar design, sleeve length, coat length, etc. (8 FashionAI tasks)

```
Image + Query ("左边袖口")
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
    │    Fast-path localization (hem/waist/shoulder/leg_opening)
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
```

---

## Features

### 3.1.1 Garment instance segmentation
- YOLOv8n 13-class detector with class-balanced retraining
- SAM-HQ vit_b instance segmentation
- Dual-label output: fine (13-class) for internal use, coarse (5-class: top/pants/skirt/outerwear/dress) for PRD
- 39-point landmark prediction per garment instance

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

---

## Repository Structure

```
fashion-finegrained-vision/
├── configs/
│   ├── category_mapping.yaml           # 13→5 class mapping
│   ├── attribute_inference.yaml        # attribute task registry (8 tasks)
│   ├── attribute_group_mapping.yaml    # category→task→region mapping
│   ├── attribute_eval_targets.yaml     # eval target config
│   └── attribute_taxonomy.yaml         # attribute class taxonomy
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
├── scripts/                            # analysis, calibration, viz scripts
├── tests/                              # 373+ tests
├── docs/                               # engineering plans and reports
│   ├── project_retrospective.md        # full project review
│   ├── industrial_grounding_implementation_plan.md
│   └── ...
│
├── CLAUDE.md
├── PROJECT_CONTEXT.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/Charlie-Zhou-Xingyu/fashion-finegrained-vision.git
cd fashion-finegrained-vision
pip install -r requirements.txt
```

Key dependencies: `torch`, `ultralytics`, `transformers`, `opencv-python`, `numpy`, `pillow`, `matplotlib`, `pytest`.

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
# result["status"] → "success" | "not_detected"
# result["bbox"]  → [x1, y1, x2, y2]
# result["mask"]  → binary mask array
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
| 3.1.1 Garment segmentation | ✅ Complete | 5-class accuracy 0.938, 420ms/image |
| 3.1.2 Region localization | ✅ Core complete | 92% fast-path, 60.6% collar/neckline/lapel, 26-part vocab |
| 3.1.3 Attribute extraction | ✅ Baseline complete | Best macro-F1 0.764 (collar_design), PRD target 0.88 |
| Tests | 373+ passed, 2 skipped | |

### Primary bottleneck
SAM-HQ vit_b at 293ms/image (69.6% of pipeline runtime). Optimization plan at `docs/inference_optimization_plan.md`.

### Known accuracy gaps
- Attribute F1 0.12-0.29 below PRD target — root cause: training data scarcity (556-1,647 samples/task)
- DINO-tiny hits capability ceiling on very small parts (rivets, studs)

---

## License

TODO.
