# Fashion Fine-Grained Vision

A fine-grained garment understanding pipeline for fashion images.

This repository provides an end-to-end vision pipeline for garment detection, instance segmentation, landmark inference, semantic region cropping, and mask-aware local crop refinement.

The current focus is to automatically extract clean local garment regions such as collars, sleeves, hems, waists, and pant legs from fashion images.

---

## Features

- **YOLO-based garment detection**
  - Supports DeepFashion2-style garment categories.
  - Current pipeline is configured for a 13-class detector.

- **SAM-HQ garment instance segmentation**
  - Uses detected garment bounding boxes as prompts.
  - Produces instance masks, overlays, and segmentation metadata.

- **Landmark-based garment structure prediction**
  - Predicts garment landmarks from segmented garment instances.
  - Supports landmark schema attachment and mask-quality filtering.

- **Semantic garment region cropping**
  - Extracts local garment regions based on predicted landmarks.
  - Supported region types include:
    - `collar`
    - `sleeve`
    - `hem`
    - `waist`
    - `pant_leg`

- **SAM-HQ mask-aware crop refinement**
  - Applies SAM-HQ masks to local region crops.
  - Removes background noise from local garment parts.
  - Supports white, black, gray, or transparent background.

- **One-click inference pipeline**
  - A single command runs the full pipeline from input images to masked local crops.

---

## Pipeline Overview

```text
Input image / image directory
        |
        v
01 YOLO garment detection
        |
        v
02 SAM-HQ instance segmentation
        |
        v
03 Garment landmark inference
        |
        v
04 Landmark-based semantic region crops
        |
        v
05 SAM-HQ mask-aware region crops
```

Main entry:

```text
tools/infer/run_garment_pipeline.py
```

---

## Repository Structure

```text
fashion-finegrained-vision
├── configs
│   ├── dataset
│   ├── inference
│   └── model
├── docs
│   ├── 3_1_1_sam_instance_segmentation_plan.md
│   └── experiment_log.md
├── src
│   └── fashion_vision
│       ├── data
│       ├── evaluation
│       ├── landmarks
│       ├── localization
│       ├── models
│       ├── prompts
│       ├── schemas
│       ├── utils
│       └── visualization
├── tests
├── tools
│   ├── analysis
│   ├── crop
│   ├── data
│   ├── eval
│   ├── infer
│   ├── train
│   └── visualize
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Large files such as datasets, checkpoints, generated outputs, and model weights are intentionally excluded from Git.

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/fashion-finegrained-vision.git
cd fashion-finegrained-vision
```

### 2. Create environment

Example with Conda:

```bash
conda create -n fashion-demo python=3.10 -y
conda activate fashion-demo
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Depending on your CUDA / PyTorch setup, you may need to install PyTorch manually from the official website:

```bash
https://pytorch.org/get-started/locally/
```

### 4. Install optional dependencies

This project uses YOLO, SAM-HQ, and PyTorch-based landmark models.  
Make sure the required packages for your local environment are installed.

Typical additional packages may include:

```bash
pip install ultralytics opencv-python pillow numpy tqdm matplotlib pytest
```

---

## Required Model Weights

Model weights are **not included** in this repository.

Please prepare the following files locally.

### YOLO garment detector

Expected path:

```text
models/detectors/yolov8n_deepfashion2_13cls_best.pt
```

### SAM-HQ checkpoint

For ViT-B:

```text
checkpoints/sam_hq/sam_hq_vit_b.pth
```

For ViT-H:

```text
checkpoints/sam_hq/sam_hq_vit_h.pth
```

### Landmark predictor checkpoint

Current default path:

```text
outputs/landmark_predictor_resnet18/best.pt
```

Recommended future path:

```text
models/landmarks/resnet18_deepfashion2_best.pt
```

If you move the landmark checkpoint to another path, specify it with:

```bash
--landmark-checkpoint path/to/checkpoint.pt
```

---

## SAM-HQ Setup

This repository expects SAM-HQ to be available locally.

Recommended location:

```text
third_party/sam-hq/
```

This directory is ignored by Git because SAM-HQ is an external dependency.

Please clone or install SAM-HQ according to its official instructions.

---

## Quick Start

### Run the full one-click pipeline

Windows CMD example:

```bat
python tools\infer\run_garment_pipeline.py ^
  --source assets\examples\sample_images ^
  --output-dir outputs\demo ^
  --yolo-weights models\detectors\yolov8n_deepfashion2_13cls_best.pt ^
  --sam-checkpoint checkpoints\sam_hq\sam_hq_vit_b.pth ^
  --sam-model-type vit_b ^
  --landmark-checkpoint outputs\landmark_predictor_resnet18\best.pt ^
  --save-landmark-visualizations ^
  --draw-landmark-index
```

Linux / macOS example:

```bash
python tools/infer/run_garment_pipeline.py \
  --source assets/examples/sample_images \
  --output-dir outputs/demo \
  --yolo-weights models/detectors/yolov8n_deepfashion2_13cls_best.pt \
  --sam-checkpoint checkpoints/sam_hq/sam_hq_vit_b.pth \
  --sam-model-type vit_b \
  --landmark-checkpoint outputs/landmark_predictor_resnet18/best.pt \
  --save-landmark-visualizations \
  --draw-landmark-index
```

---

## Output Structure

A successful run creates the following directory structure:

```text
outputs/demo
├── 01_yolo
│   ├── detections.json
│   └── visualizations
├── 02_samhq
│   ├── segmentation_results.json
│   ├── masks
│   └── overlays
├── 03_landmarks
│   ├── landmarks_results.json
│   ├── summary.json
│   └── visualizations
├── 04_region_crops
│   ├── region_crops.json
│   └── crops
├── 05_region_masked_crops
│   ├── region_masked_crops.json
│   ├── image_crops
│   ├── mask_crops
│   └── masked_crops
└── pipeline_summary.json
```

---

## Pipeline Arguments

Important arguments for the one-click pipeline:

```text
--source                      Input image path or image directory.
--output-dir                  Root output directory.

--yolo-weights                YOLO detector weights.
--imgsz                       YOLO image size.
--conf                        YOLO confidence threshold.
--iou                         YOLO NMS IoU threshold.
--yolo-device                 YOLO device.

--sam-checkpoint              SAM-HQ checkpoint path.
--sam-model-type              SAM-HQ model type: vit_b, vit_l, or vit_h.
--sam-device                  SAM-HQ device.

--landmark-checkpoint         Landmark model checkpoint.
--landmark-model              Landmark backbone name.
--landmark-image-size         Landmark input image size.
--landmark-device             Landmark inference device.

--region-crop-regions         Region types to crop.
--region-pad-ratio            Padding ratio for region crop.
--region-min-points           Minimum reliable landmarks for region crop.

--masked-crop-background      Background color: white, black, or gray.
--masked-crop-transparent     Save transparent masked crops.
--min-mask-area-ratio         Minimum SAM mask area ratio inside region crop.
```

Skip flags for reusing existing intermediate results:

```text
--skip-yolo
--skip-sam
--skip-landmarks
--skip-region-crops
--skip-masked-crops
```

---

## Example: Random 60 Images from DeepFashion2 Train

If your DeepFashion2 train images are located at:

```text
D:\Aliintern\fashion-ai-data\deepfashion2\train\image
```

Randomly sample 60 images:

```bat
powershell -Command "New-Item -ItemType Directory -Force assets\random_train60\images | Out-Null; Get-ChildItem 'D:\Aliintern\fashion-ai-data\deepfashion2\train\image' -File | Where-Object {$_.Extension -match 'jpg|jpeg|png|bmp|webp'} | Get-Random -Count 60 | Copy-Item -Destination assets\random_train60\images"
```

Run the pipeline:

```bat
python tools\infer\run_garment_pipeline.py ^
  --source assets\random_train60\images ^
  --output-dir outputs\pipeline_random_train60 ^
  --yolo-weights models\detectors\yolov8n_deepfashion2_13cls_best.pt ^
  --sam-checkpoint checkpoints\sam_hq\sam_hq_vit_b.pth ^
  --sam-model-type vit_b ^
  --landmark-checkpoint outputs\landmark_predictor_resnet18\best.pt ^
  --save-landmark-visualizations ^
  --draw-landmark-index
```

Generated sampled images and outputs are ignored by Git.

---

## Data Preparation Tools

### Export DeepFashion2 to YOLO format

```bash
python tools/data/export_deepfashion2_to_yolo.py
```

### Build landmark dataset

```bash
python tools/data/build_deepfashion2_landmark_dataset.py
```

### Sample balanced evaluation images

```bash
python tools/data/sample_balanced_eval_images.py
```

---

## Training

### Train landmark predictor

```bash
python tools/train/train_landmark_predictor.py
```

The trained checkpoints are written to `outputs/` by default and are ignored by Git.

---

## Analysis and Visualization

### Summarize region crops

```bash
python tools/analysis/summarize_region_crops.py \
  --json outputs/demo/04_region_crops/region_crops.json
```

### Visualize landmark predictions

```bash
python tools/visualize/visualize_landmark_predictions.py
```

### Visualize region crop debug results

```bash
python tools/visualize/visualize_region_crops_debug.py
```

---

## Testing

Run all tests:

```bash
pytest -q
```

If the package cannot be found, set `PYTHONPATH` first.

Windows CMD:

```bat
set PYTHONPATH=%CD%\src
pytest -q
```

Linux / macOS:

```bash
export PYTHONPATH=$PWD/src
pytest -q
```

---

## Current Status

Completed:

- YOLO garment detection pipeline
- SAM-HQ instance segmentation pipeline
- Landmark inference pipeline
- Landmark schema and mask-quality integration
- Semantic region crop generation
- SAM-HQ mask-aware region crop refinement
- One-click 01-05 pipeline
- Tested on:
  - Small 10-image one-click run
  - Random 60-image DeepFashion2 train sample

Current main output:

```text
Input image(s) -> clean masked local garment crops
```

---

## Roadmap

Planned next steps:

- **06 Region embeddings**
  - Extract embeddings from masked local crops using CLIP / ResNet / other visual encoders.

- **07 Region-level retrieval**
  - Build similarity search over garment local regions.

- **08 Interactive demo**
  - Provide a simple UI for uploading fashion images and viewing parsed regions.

- **09 Text-guided local retrieval**
  - Query local garment parts with text, e.g. "striped collar", "wide sleeve", "denim hem".

---

## Notes

This repository does not include:

- DeepFashion2 dataset files
- SAM-HQ model weights
- YOLO trained weights
- Landmark trained weights
- Generated outputs
- Experiment runs

Please prepare the required datasets and model checkpoints locally.

---

## License

TODO.

Please add an appropriate license before public release if this repository is intended to be open source.
