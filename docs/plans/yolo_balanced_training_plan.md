# YOLO Balanced Retraining Plan

> Created: 2026-06-16
> Target: YOLOv8 13-class garment detector on class-balanced DeepFashion2

---

## 0. Goal

Retrain the YOLOv8 garment detector on a class-balanced subset of
DeepFashion2 to improve detection recall for under-represented categories
(skirt, sling, vest dress, sling dress).

The retrained model remains 13-class internally.  PRD-facing 5-class output
is produced at inference time by the mapping in `configs/category_mapping.yaml`.

---

## 1. Current Training Assets

### Dataset

| Path | Description |
|---|---|
| `data/processed/deepfashion2_yolo_13cls/` | Full 13-class YOLO dataset (images + labels) |
| `data/processed/deepfashion2_yolo_13cls_val2000/` | 2000-image validation subset |
| `data/processed/deepfashion2_yolo_13cls_val10000/` | 10000-image validation subset |

### Training YAMLs (already generated)

| YAML | Train split | Notes |
|---|---|---|
| `data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls.yaml` | `images/train` (all, unbalanced) | Baseline unbalanced config |
| `data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml` | `train_balanced.txt` | Standard balanced list (inverse-sqrt frequency, max repeat 5) |
| `data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced_30k.yaml` | `train_balanced_30k.txt` | 30k-cap balanced list |

Use `deepfashion2_13cls_balanced.yaml` for the primary retraining run.

### Current model

`models/detectors/yolov8n_deepfashion2_13cls_best.pt` — YOLOv8n, trained
on unbalanced split.

---

## 2. Regenerating the Balanced List (if needed)

The balanced list files (`train_balanced.txt`, `train_balanced_30k.txt`) were
previously generated and already exist.  Regenerate only if the dataset or
balance parameters change.

```bash
python tools/data/make_balanced_yolo_train_list.py \
  --dataset-root data/processed/deepfashion2_yolo_13cls \
  --yaml data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls.yaml \
  --split train \
  --balance-power 0.5 \
  --max-repeat 5 \
  --output-list-name train_balanced.txt \
  --output-yaml-name deepfashion2_13cls_balanced.yaml
```

Outputs:
- `data/processed/deepfashion2_yolo_13cls/train_balanced.txt` — balanced image list
- `data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml` — training YAML
- `data/processed/deepfashion2_yolo_13cls/balance_report.json` — per-class stats
- `data/processed/deepfashion2_yolo_13cls/balance_report.md` — markdown summary

---

## 3. Training Command

```bash
yolo detect train \
  data=data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=yolov8n_df2_13cls_balanced \
  project=runs/detect \
  device=0
```

**Notes:**
- `model=yolov8n.pt` uses the ImageNet-pretrained YOLOv8n backbone (not the
  current best checkpoint).  This is intentional for a clean balanced
  retraining run.
- To fine-tune from the existing checkpoint instead, replace with:
  `model=models/detectors/yolov8n_deepfashion2_13cls_best.pt`
- `batch=16` — adjust down to 8 if GPU memory is insufficient.
- Output will be saved to `runs/detect/yolov8n_df2_13cls_balanced/`.

---

## 4. Post-training Evaluation

### Step 4a: Run YOLO val to get confusion matrix

```bash
yolo detect val \
  data=data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml \
  model=runs/detect/yolov8n_df2_13cls_balanced/weights/best.pt \
  imgsz=640 \
  device=0 \
  save_json=True
```

YOLO saves a `confusion_matrix.json` in the run directory (format:
`{"names": [...], "matrix": [[...], ...]}`).

### Step 4b: Aggregate 13-class matrix to 5-class PRD metrics

```bash
python tools/eval/eval_13cls_confusion_as_5cls.py \
  --matrix-json runs/detect/yolov8n_df2_13cls_balanced/confusion_matrix.json \
  --mapping-yaml configs/category_mapping.yaml \
  --out-dir runs/detect/yolov8n_df2_13cls_balanced/eval_5cls
```

Outputs:
- `eval_13cls_as_5cls_metrics.json` — structured metrics
- `eval_13cls_as_5cls_report.md` — human-readable report with 5×5 confusion matrix

### Step 4c: Run detection inference with dual-label output

```bash
python tools/infer/predict_garments_yolo.py \
  --weights runs/detect/yolov8n_df2_13cls_balanced/weights/best.pt \
  --source assets/eval_images_13cls_balanced/images \
  --output-dir outputs/pipeline_balanced_eval \
  --save-vis
```

Each detection record in `detections.json` will now contain both
`fine_class_id`/`fine_class_name` (13-class internal) and
`coarse_class_id`/`coarse_class_name` (5-class PRD-facing).

---

## 5. Acceptance Criteria

Compare the new balanced model against the current baseline
(`yolov8n_deepfashion2_13cls_best.pt`) on the 5-class aggregated metrics:

| Metric | Baseline target | Notes |
|---|---|---|
| Foreground 5-class accuracy | ≥ baseline | Measured on confusion matrix diagonal |
| Per-class recall (skirt, sling, vest dress, sling dress) | ≥ baseline | Primary motivation for balanced retraining |
| Foreground correct detection rate | Monitor | May decrease if background precision drops |

Do not replace the current model weights unless the new model meets or exceeds
the baseline on the 5-class evaluation.

---

## 6. Files Not to Modify During Retraining

- `data/processed/deepfashion2_yolo_13cls/images/` — source images
- `data/processed/deepfashion2_yolo_13cls/labels/` — source labels
- `configs/category_mapping.yaml` — category mapping
- `src/fashion_vision/localization/landmark_region_map.py` — landmark rules
- Any `outputs/p2_*/` or `outputs/p3_*/` — 3.1.3 checkpoints
