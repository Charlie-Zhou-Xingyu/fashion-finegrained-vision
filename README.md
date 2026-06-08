# Fashion Fine-grained Vision

This repository implements the fine-grained visual foundation module for fashion product understanding.

## 1. Project Scope

The final system follows the PRD requirement and targets 8 fashion categories:

| English Key | Chinese Name |
|---|---|
| top | 上衣 |
| pants | 裤子 |
| skirt | 裙子 |
| outwear | 外套 |
| dress | 连衣裙 |
| shoes | 鞋子 |
| bag | 包包 |
| accessory | 配饰 |

The first engineering milestone focuses on SAM-HQ based instance mask generation on DeepFashion2.

## 2. Stage 1: SAM-HQ Box Prompt Baseline

Current stage:

```text
DeepFashion2 image
        ↓
Read annotation
        ↓
Extract garment bbox + category + ground-truth mask
        ↓
Use bbox as SAM-HQ box prompt
        ↓
Generate predicted mask
        ↓
Compute IoU between predicted mask and ground-truth mask
        ↓
Save prediction JSON
        ↓
Save visualization image
        ↓
Save metrics summary and report
```

This stage uses DeepFashion2 ground-truth bounding boxes as SAM-HQ box prompts to evaluate the mask generation capability of SAM-HQ.

The system interface is designed for 8 target categories required by the PRD. DeepFashion2 provides official clothing categories that can be mapped to part of the target taxonomy, including top, pants, skirt, outwear, and dress.

For categories that are not sufficiently covered by DeepFashion2, such as shoes, bag, and accessory, the project will introduce supplementary datasets or custom annotations:

- Shoes: UT Zappos, Fashionpedia, OpenImages, LVIS, or custom Label Studio annotations.
- Bag: Fashionpedia, ModaNet, OpenImages, LVIS, COCO, or custom Label Studio annotations.
- Accessory: Fashionpedia, ModaNet, OpenImages, LVIS, or custom Label Studio annotations.

For the DeepFashion2 category `vest`, this project maps it to `outwear` by default, because vest may represent an outer garment such as a zippered vest or sleeveless jacket in fashion e-commerce scenarios. If more fine-grained attributes are available later, vest can be further divided into inner-wear vest and outer-wear vest.

## 3. Directory Structure

```text
fashion-finegrained-vision/
├── configs/
│   ├── dataset/
│   ├── model/
│   └── inference/
├── src/
│   └── fashion_vision/
│       ├── data/
│       ├── prompts/
│       ├── models/
│       ├── inference/
│       ├── evaluation/
│       ├── visualization/
│       └── utils/
├── tools/
├── tests/
├── docs/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. External Data and Model Paths

Large datasets and model weights are stored outside this repository.

Recommended layout:

```text
D:/Aliintern/
├── fashion-ai-data/
│   ├── deepfashion2/
│   └── fashionai_attributes/
├── fashion-ai-models/
│   └── sam_hq/
│       └── sam_hq_vit_b.pth
└── fashion-finegrained-vision/
```

## 5. Installation

```bash
pip install -r requirements.txt
```

SAM-HQ should be installed or added to `PYTHONPATH` separately.

## 6. Run Stage 1 Baseline

```bash
python tools/run_sam_deepfashion2_box_prompt.py --config configs/inference/sam_box_prompt.yaml
```

Expected outputs:

```text
outputs/sam_hq_deepfashion2/
├── predictions/
├── visualizations/
├── metrics/
└── logs/
```

## 7. Engineering Principles

This project follows the code review requirements:

1. Modular design.
2. Configuration-driven parameters.
3. Robust exception handling.
4. Clear comments and docstrings.
5. Type hints.
6. Reusable functions and classes.
7. PEP8 coding style.
