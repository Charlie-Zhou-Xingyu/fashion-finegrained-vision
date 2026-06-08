# Experiment Log

## Experiment 001: SAM-HQ DeepFashion2 Box Prompt Baseline

### Date

To be filled.

### Goal

Evaluate SAM-HQ mask generation quality on DeepFashion2 using ground-truth bounding boxes as box prompts.

### Dataset

- Dataset: DeepFashion2
- Split: validation
- Data root: `D:/Aliintern/fashion-ai-data/deepfashion2`

### Model

- Model: SAM-HQ
- Variant: ViT-B
- Checkpoint: `D:/Aliintern/fashion-ai-models/sam_hq/sam_hq_vit_b.pth`

### Method

```text
DeepFashion2 image
        ↓
GT bbox as box prompt
        ↓
SAM-HQ
        ↓
Predicted mask
        ↓
IoU with GT mask
```

### Metrics

- Mean IoU
- Median IoU
- IoU@0.50
- IoU@0.75
- IoU@0.85
- Mean latency

### Result

To be filled after running the experiment.

### Notes

Current stage evaluates segmentation mask generation ability. End-to-end automatic instance segmentation will be implemented by integrating automatic detection or open-vocabulary grounding modules.

### Dataset Extension Plan

DeepFashion2 is used for the first-stage quantitative evaluation of garment mask generation.

For categories not sufficiently covered by DeepFashion2:

- Shoes will be supplemented by UT Zappos or custom Label Studio annotations.
- Bags can be supplemented by Fashionpedia, ModaNet, COCO, OpenImages, LVIS, or custom Label Studio annotations.
- Accessories can be supplemented by Fashionpedia, ModaNet, OpenImages, LVIS, or custom Label Studio annotations.

The `vest` category in DeepFashion2 is mapped to `outwear` by default in this project, following the business rule that zippered vests or sleeveless jackets are treated as outerwear.
