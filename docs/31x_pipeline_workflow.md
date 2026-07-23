# 3.1.x Full Pipeline — Technical Workflow Documentation

> **Last updated**: 2026-07-22  
> **Scope**: YOLO → SAM-HQ → Landmark → Region Crops → Masked Crops → Attribute Inference  
> **Models loaded**: ~7 models (YOLO 22MB + SAM 362MB + Landmark 135MB + 2-4 attrs 520MB) ≈ 5-6GB VRAM peak

---

## 1. Architecture Overview

```
                          INPUT IMAGE
                              │
 ┌────────────────────────────┼────────────────────────────┐
 │  Stage 1: YOLOv8n (13 cls) │  tools/infer/predict_garments_yolo.py
 │  ──────────────────────────│───────────────────────────
 │  • 640×640 input            │
 │  • conf=0.25, iou=0.7       │
 │  • 13 DeepFashion2 classes  │
 │  Output: detections.json    │
 └────────────────────────────┼────────────────────────────┘
                              │  [{instance_id, class_name, bbox, confidence}, ...]
                              ▼
 ┌────────────────────────────┼────────────────────────────┐
 │  Stage 2: SAM-HQ ViT-B     │  tools/infer/segment_garments_samhq.py
 │  ──────────────────────────│───────────────────────────
 │  • Box-prompt segmentation  │
 │  • Mask RLE + polygon out   │
 │  • Mask quality scoring     │
 │  Output: segmentation_results.json + overlays/ + masks/
 └────────────────────────────┼────────────────────────────┘
                              │  [{mask_rle, mask_area, mask_quality}, ...]
                              ▼
 ┌────────────────────────────┼────────────────────────────┐
 │  Stage 3: Landmark ResNet18│  tools/infer/infer_landmarks_for_predictions.py
 │  ──────────────────────────│───────────────────────────
 │  • 256×256 crop input       │
 │  • 39 landmarks × 2 coords  │
 │  • Mask-quality gate        │
 │  Output: landmarks_results.json
 └────────────────────────────┼────────────────────────────┘
                              │  [{landmarks: [(x,y,v),...], schema: {...}}]
                              ▼
 ┌────────────────────────────┼────────────────────────────┐
 │  Stage 4: Region Crops     │  tools/crop/crop_garment_regions_from_landmarks.py
 │  ──────────────────────────│───────────────────────────
 │  • 5 regions: collar,       │
 │    sleeve, hem, waist,      │
 │    pant_leg                 │
 │  • Category-aware filtering  │
 │  • Multi-point → bbox + pad │
 │  Output: region_crops.json + crops/
 └────────────────────────────┼────────────────────────────┘
                              │  [{det_id, region, component, crop_paths}, ...]
                              ▼
 ┌────────────────────────────┼────────────────────────────┐
 │  Stage 5: Masked Crops     │  tools/crop/apply_samhq_mask_to_region_crops.py
 │  ──────────────────────────│───────────────────────────
 │  • SAM mask → isolate garment│
 │  • White/black/gray bg      │
 │  • min_mask_area_ratio=0.005│
 │  Output: region_masked_crops.json + masked_crops/
 └────────────────────────────┼────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────┼────────────────────────────┐
 │  Stage 6: Attribute Inf.   │  src/fashion_vision/attributes/
 │  ──────────────────────────│───────────────────────────
 │  • 8 ResNet18 classifiers   │
 │  • Config-driven routing    │  garment_attribute_pipeline.py
 │  • 5 coarse → 8 task map    │  task_registry.py
 │  • Lazy model loading       │  category_gate.py
 │  Output: predictions.jsonl  │
 └────────────────────────────┼────────────────────────────┘
```

---

## 2. Stage 1 — YOLO Garment Detection

**File**: `tools/infer/predict_garments_yolo.py`  
**Model**: YOLOv8n trained on 13 DeepFashion2 classes  
**Checkpoint**: `models/detectors/yolov8n_deepfashion2_13cls_best.pt` (22 MB)

### Implementation

```python
# tools/infer/predict_garments_yolo.py (simplified)
from ultralytics import YOLO

model = YOLO(args.weights)                    # load YOLOv8n
results = model.predict(
    source=source,
    imgsz=640, conf=0.25, iou=0.7,
    device=args.device,
)
# Post-process: normalize bboxes, assign class names, save detections.json
```

### 13-class → 5-class mapping

The detector uses 13 fine classes internally. Coarse mapping happens downstream:

| Fine class ID | Fine class name | Coarse class |
|:---|:---|---|
| 0 | short sleeve top | top |
| 1 | long sleeve top | top |
| 2 | short sleeve outwear | outerwear |
| 3 | long sleeve outwear | outerwear |
| 4 | vest | top |
| 5 | sling | top |
| 6 | shorts | pants |
| 7 | trousers | pants |
| 8 | skirt | skirt |
| 9 | short sleeve dress | dress |
| 10 | long sleeve dress | dress |
| 11 | vest dress | dress |
| 12 | sling dress | dress |

Config: `configs/category_mapping.yaml`

### Output: `detections.json`

```json
{
  "task": "predict_garments_yolo",
  "images": [{
    "image": "000001.jpg",
    "detections": [
      {
        "instance_id": 0,
        "class_id": 1,
        "class_name": "long sleeve top",
        "bbox": [204.0, 189.0, 293.0, 414.0],
        "confidence": 0.94
      }
    ]
  }]
}
```

---

## 3. Stage 2 — SAM-HQ Instance Segmentation

**File**: `tools/infer/segment_garments_samhq.py`  
**Model**: SAM-HQ ViT-B  
**Checkpoint**: `checkpoints/sam_hq/sam_hq_vit_b.pth` (362 MB)

### Implementation

```python
# tools/infer/segment_garments_samhq.py (simplified)
from third_party.sam_hq.segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)
predictor = SamPredictor(sam)

for detection in detections:
    bbox = detection["bbox"]                     # xyxy from YOLO
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=bbox,
        multimask_output=False,                   # single best mask
    )
    # Convert mask to RLE + polygon for storage efficiency
```

SAM-HQ uses "High-Quality output tokens" added during fine-tuning to produce sharper masks than vanilla SAM.

### Output: `segmentation_results.json`

```json
{
  "task": "segment_garments_samhq",
  "images": [{
    "image": "000001.jpg",
    "instances": [{
      "instance_id": 0,
      "class_name": "long sleeve top",
      "mask_rle": {"counts": "...", "size": [768, 512]},
      "mask_area": 45231,
      "mask_quality": 0.92,
      "overlay_path": "02_samhq/overlays/000001_det0.jpg",
      "mask_save_path": "02_samhq/masks/000001_det0.npy"
    }]
  }]
}
```

---

## 4. Stage 3 — Garment Landmark Prediction

**File**: `tools/infer/infer_landmarks_for_predictions.py`  
**Model**: ResNet18 regressor (78 outputs = 39 landmarks × 2)  
**Checkpoint**: `outputs/landmark_predictor_resnet18/best.pt` (135 MB)

### Architecture

```
ResNet18 (pretrained on ImageNet)
  └── fc: Linear(512, 78)   # replaces the 1000-class head with 39 (x,y) pairs
```

Input: 256×256 garment image crop around the SAM mask bounding box, padded by 5%.  
Output: 39 landmark points. Each landmark has `(x, y, visibility)`. Visibility codes: `0` = invisible, `1` = occluded, `2` = visible.

### Key landmarks by garment category

| Category | Key landmarks (by index) |
|---|---|
| Tops/dresses | 0-13: collar points, 14-21: sleeve endpoints, 22-29: hem line, 30-37: waist |
| Pants | 0-5: waist line, 6-17: leg opening (left + right), 18-21: crotch |
| Skirts | 0-7: waist line, 8-19: hem line |

### Implementation

```python
# tools/infer/infer_landmarks_for_predictions.py → load_landmark_model()
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features, max_landmarks * 2)  # 512 → 78
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

### Output: `landmarks_results.json`

```json
{
  "num_instances_processed": 1,
  "instances": [{
    "instance_id": 0,
    "class_name": "long sleeve top",
    "landmarks": [[247, 260, 2], [210, 195, 2], ...],
    "landmark_schema": {
      "landmark_names": ["top_center", "left_collar", ...],
      "point_groups": {"collar": [0,1,2,3], "sleeve": [14,15,16,17], ...}
    }
  }]
}
```

---

## 5. Stage 4 — Semantic Region Cropping

**File**: `tools/crop/crop_garment_regions_from_landmarks.py`

### How it works

For each garment instance and each applicable region type:

1. **Look up landmark group** for the region (e.g., "collar" → landmarks [0,1,2,3])
2. **Filter by mask quality**: landmarks outside the SAM mask by > max_outside_distance (5px) are excluded
3. **Construct bounding box**:
   - If ≥ min_points (2) landmarks remain: compute axis-aligned bbox from landmark coords
   - Else: fallback to using full garment bbox (if fallback=True)
4. **Pad the bbox** by pad_ratio (0.35 = 35% expansion on each side)
5. **Crop and save** the image region

### Category-aware region filtering

Not all region types are applicable to all garment categories:

| Category | Enabled regions |
|---|---|
| short sleeve top, long sleeve top | collar, sleeve, hem, waist |
| short/long sleeve outwear | collar, sleeve, hem, waist |
| vest, sling | collar, hem, waist |
| shorts, trousers | waist, pant_leg |
| skirt | waist, hem |
| short/long sleeve dress | collar, sleeve, hem, waist |
| vest dress, sling dress | collar, hem, waist |

### Output: `region_crops.json`

```json
{
  "crops": [
    {
      "det_id": "000001_det0",
      "image_path": "000001.jpg",
      "class_name": "long sleeve top",
      "region": "collar",
      "component": "front_collar",
      "success": true,
      "crop_path": "04_region_crops/crops/000001_det0_collar_collar.jpg",
      "expanded_crop_path": "04_region_crops/crops/000001_det0_collar_collar.jpg"
    },
    ...
  ]
}
```

The `expanded_crop_path` is a wider version of the crop (larger pad ratio), used for length-related tasks (sleeve, coat, pant, skirt). The `crop_path` is the standard crop.

---

## 6. Stage 5 — Mask-Aware Region Cropping

**File**: `tools/crop/apply_samhq_mask_to_region_crops.py`

Applies the SAM-HQ instance mask to each region crop, producing a clean garment-only image with the background filled white:

```python
# For each crop record + its source image:
mask = load_mask(segmentation_json, det_id)
crop = load_image(crop_path)
masked = crop * mask + (1 - mask) * background_color  # white=255
save(masked, masked_crop_path)
```

Config:
- `background`: white (255,255,255), black (0,0,0), gray (128,128,128), or transparent PNG
- `min_mask_area_ratio`: 0.005 — crops with <0.5% mask coverage are rejected

### Output: `region_masked_crops.json`

```json
{
  "crops": [
    {
      "det_id": "000001_det0",
      "region": "collar",
      "masked_success": true,
      "masked_crop_path": "05_region_masked_crops/masked_crops/000001_det0_collar.jpg",
      "mask_fill_ratio": 0.87
    },
    ...
  ]
}
```

---

## 7. Stage 6 — Fine-Grained Attribute Inference

### Architecture

This is the most complex stage. It involves multiple modules:

```
configs/attribute_inference.yaml          — 8 tasks × checkpoint paths
configs/attribute_group_mapping.yaml      — 5 coarse → task routing
data/fashionai_attribute_index/label_map_*.json  — 8 label maps
models/attributes/p2_*_best.pt            — 8 trained ResNet18 weights
models/attribute_classifier.py            — Model builder
src/fashion_vision/attributes/
  category_gate.py                         — Config loader + task router
  task_registry.py                         — Checkpoint loader + model builder
  garment_attribute_pipeline.py            — Full pipeline orchestrator
```

### 7a. Task registry (`task_registry.py`)

Handles all model loading:

```python
# Load inference config
configs = load_inference_config("configs/attribute_inference.yaml")
# → {"neckline_design": AttributeTaskConfig(checkpoint=..., label_map=..., arch="resnet18", img_size=224), ...}

# Load one task
loaded = load_task(configs["sleeve_length"], device)
# → LoadedTask(model=ResNet18(fc=8), id_to_label={0:"Sleeveless",1:"Cup Sleeves",...}, transform=Compose)
```

Key design decisions:
- `num_classes` is derived from `len(id_to_label)` — never hardcoded
- Supports 4 checkpoint wrapper formats: `model_state_dict`, `state_dict`, `model`, bare dict
- Strips `module.` DataParallel prefix automatically
- Supports 5 label-map JSON formats: `id_to_label`, `idx_to_label`, `classes` list, plain dict, `label_to_id`
- Deterministic inference transform: `Resize(224) → ToTensor → Normalize(ImageNet)`

### 7b. Category gate (`category_gate.py`)

Config-driven task routing:

```python
mapping = load_attribute_group_mapping("configs/attribute_group_mapping.yaml")
tasks = get_enabled_tasks("top", mapping)
# → ["neckline_design", "collar_design", "neck_design", "sleeve_length"]
```

The mapping:

| Coarse class | Enabled tasks |
|---|---|
| top | neckline_design, collar_design, neck_design, sleeve_length |
| outerwear | lapel_design, coat_length, sleeve_length |
| pants | pant_length |
| skirt | skirt_length |
| dress | neckline_design, skirt_length, sleeve_length |

### 7c. Crop selection logic

For each task, the pipeline selects the best crop record using 4 filters:

| Filter | Source | Example |
|---|---|---|
| `region_filter` | `attribute_inference.yaml` per task | `"collar"` for neck tasks, `"all"` for length tasks |
| `class_contains` | `attribute_inference.yaml` per task | `"outwear"` for coat_length (only match outwear, not dresses) |
| `component_contains` | `attribute_group_mapping.yaml` | `"sleeve"` for sleeve_length, `"pant"` for pant_length |
| `crop_type` | `attribute_group_mapping.yaml` | `"upper_crop"` for collar/neck → falls back to expanded_crop → image_crop → crop |

The fallback chain in `_get_crop_path()`:

```
expanded_crop: expanded_crop_path → image_crop_path → crop_path
upper_crop:    upper_crop_path → expanded_crop_path → image_crop_path → crop_path
masked_crop:   masked_crop_path
image_crop:    image_crop_path → crop_path
```

### 7d. Model builder (`models/attribute_classifier.py`)

```python
def build_attribute_classifier(arch="resnet18", num_classes=8, pretrained=False):
    model = models.resnet18(weights=None)      # ImageNet weights loaded, then overwritten
    model.fc = nn.Linear(512, num_classes)      # Replace 1000-class head
    return model
```

Note: `pretrained=False` is used at inference because the checkpoint already contains the complete state dict, including ImageNet-pretrained backbone weights.

### 7e. 8 attribute tasks

| Task | Classes | Labels (sample) | Best macro-F1 |
|---|---|---|---|
| neckline_design | 10 | Invisible, Strapless Neck, Deep V, Round, V, Square, Off Shoulder... | 0.716 |
| collar_design | 5 | Invisible, Shirt Collar, Peter Pan, Puritan Collar, Rib Collar | 0.764 |
| neck_design | 5 | Invisible, Turtle Neck, Ruffle Semi-High, Low Turtle Neck, Draped | 0.687 |
| lapel_design | 5 | Invisible, Notched, Collarless, Shawl Collar, Plus Size Shawl | 0.661 |
| sleeve_length | 8 | Sleeveless, Cup Sleeves, Short Sleeves, Elbow, 3/4, Wrist, Long, Extra Long | 0.653 |
| coat_length | 8 | Invisible, High Waist, Regular, Long, Micro, Knee, Midi, Ankle&Floor | 0.628 |
| pant_length | 6 | Invisible, Short Pant, Mid Length, 3/4 Length, Cropped, Full Length | 0.593 |
| skirt_length | 6 | Invisible, Short Length, Knee Length, Midi, Ankle, Floor Length | 0.675 |

Training details: ResNet18, seed=2, ~556-1647 train samples per task. Low data quantity is the primary cause of sub-0.88 macro-F1.

---

## 8. Configuration Reference

### `configs/attribute_inference.yaml`

```yaml
version: "1.0"
tasks:
  sleeve_length:
    checkpoint: models/attributes/p2_sleeve_length_multiview_v2_pipeline_resnet18_seed2/best.pt
    label_map:  data/fashionai_attribute_index/label_map_sleeve_length.json
    arch: resnet18
    img_size: 224
    region_filter: all          # accept crops from any region
    # class_contains: not set   # applies to all garment classes

  coat_length:
    checkpoint: models/attributes/p2_coat_length_resnet18_seed2/best.pt
    label_map:  data/fashionai_attribute_index/label_map_coat_length.json
    arch: resnet18
    img_size: 224
    region_filter: all
    class_contains: outwear      # only match "short/long sleeve outwear"

  neckline_design:
    checkpoint: models/attributes/p2_neckline_design_resnet18_seed2/best.pt
    label_map:  data/fashionai_attribute_index/label_map_neckline_design.json
    arch: resnet18
    img_size: 224
    region_filter: collar        # only accept collar region crops
```

### `GarmentPipelineConfig` defaults

```python
GarmentPipelineConfig(
    yolo_weights="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
    sam_checkpoint="checkpoints/sam_hq/sam_hq_vit_b.pth",
    sam_model_type="vit_b",
    landmark_checkpoint="outputs/landmark_predictor_resnet18/best.pt",
    landmark_model="resnet18",
    landmark_image_size=256,
    landmark_max_landmarks=39,
    landmark_pad_ratio=0.05,

    yolo_imgsz=640, yolo_conf=0.25, yolo_iou=0.7,
    yolo_device="0", sam_device="cuda", landmark_device="cuda",

    region_crop_regions=("collar","sleeve","hem","waist","pant_leg"),
    use_category_regions=True,
    region_max_outside_distance=5.0,
    region_min_points=2,
    region_pad_ratio=0.35,
    region_single_point_box_ratio=0.18,
    region_fallback=True,

    masked_crop_background="white",
    masked_crop_transparent=False,
    min_mask_area_ratio=0.005,

    run_landmark_and_crops=True,    # Set False for fast-path
    run_attribute_inference=False,  # Set True to enable Stage 6
    attribute_device="auto",
    attribute_topk=3,
)
```

---

## 9. Checkpoint Format Reference

| Model | Format | Size | Wrapper key |
|---|---|---|---|
| YOLOv8n | ultralytics `.pt` | 22 MB | Native ultralytics |
| SAM-HQ | PyTorch `.pth` | 362 MB | Bare state dict |
| Landmark ResNet18 | PyTorch `.pt` | 135 MB | `"model_state_dict"` |
| Attribute ResNet18 ×8 | PyTorch `.pt` | ~128 MB each | `"model_state_dict"` |

`task_registry.load_checkpoint_state()` handles all four wrapper formats:
```python
if "model_state_dict" in ckpt:     return ckpt["model_state_dict"]
elif "state_dict" in ckpt:         return ckpt["state_dict"]
elif "model" in ckpt:              return ckpt["model"]
else:                              return ckpt  # bare state dict
```

---

## 10. How to Run

### Full pipeline with attribute inference

```bash
conda activate fashion-demo2

# Batch mode (35 images with diverse categories):
python scripts/run_full_31x_pipeline.py \
    --image-dir D:/Aliintern/fashion-ai-data/deepfashion2/validation/image \
    --output-dir outputs/full_31x_demo \
    --num-images 35 --seed 42

# Single image:
python scripts/run_full_31x_pipeline.py \
    --single-image path/to/image.jpg \
    --output-dir outputs/single_test
```

### Generate HTML product demo

```bash
python scripts/build_31x_product_demo.py \
    --pipeline-output-dir outputs/full_31x_demo \
    --output-html outputs/full_31x_demo/product_demo.html
```

### Programmatic usage

```python
from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig

config = GarmentPipelineConfig(
    run_attribute_inference=True,
    attribute_device="cuda",
)
pipeline = GarmentPipeline(config)
result = pipeline.run_source("path/to/images", "outputs/demo")
```

### Run only attribute inference (stages 1-5 already done)

```bash
python src/fashion_vision/attributes/garment_attribute_pipeline.py \
    --region-crops-json outputs/demo/04_region_crops/region_crops.json \
    --output-jsonl outputs/attributes.jsonl \
    --device cuda
```

### Environment

- **Conda env**: `fashion-demo2` (Python 3.10, PyTorch 2.5.1+cu121)
- **GPU**: CUDA-capable (RTX 3090 or similar, ≥6GB VRAM)
- **Checkpoints**: all under `models/`, `checkpoints/`, and `outputs/`
