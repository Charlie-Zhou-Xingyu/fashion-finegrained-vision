# Fashionpedia YOLO v2 Retrain — Server Runbook

> Date: 2026-07-09
> v1 weights preserved at: `models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt`
> v1 params: p=1.0 r=12 → mAP50 0.312, max:min 24:1
> v2 params: p=0.6 r=6  → expected mAP50 0.38–0.42, max:min ~50:1

---

## Step 1: Generate balanced train list

```bash
cd /root/fashionpedia_train/fashionpedia_yolo_19cls

python /root/build_fashionpedia_balanced_train.py \
    --labels-dir /root/fashionpedia_train/fashionpedia_yolo_19cls/labels/train \
    --images-dir /root/fashionpedia_train/fashionpedia_yolo_19cls/images/train \
    --base-yaml /root/fashionpedia_train/fashionpedia_yolo_19cls/fashionpedia_parts.yaml \
    --out-dir /root/fashionpedia_train/fashionpedia_yolo_19cls \
    --power 0.6 --max-repeat 6 --seed 42
```

This creates:
- `train_balanced_p0.6_r6.txt`
- `fashionpedia_parts_balanced_p0.6_r6.yaml`
- `balance_report.csv`
- `class_distribution_before_after.csv`
- `repeat_histogram.csv`

Check the report: expansion should be ~1.18× (vs 1.34× for v1).

## Step 2: Train

```bash
yolo detect train \
    data=/root/fashionpedia_train/fashionpedia_yolo_19cls/fashionpedia_parts_balanced_p0.6_r6.yaml \
    model=yolov8s.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=/root/fashionpedia_train/outputs \
    name=fp_v2_p0.6_r6
```

Expected: ~4 hours on single GPU.

## Step 3: Copy best model back

After training completes, the best weights are at:
`/root/fashionpedia_train/outputs/fp_v2_p0.6_r6/weights/best.pt`

On your local machine:
```bash
scp user@server:/root/fashionpedia_train/outputs/fp_v2_p0.6_r6/weights/best.pt \
    D:/Aliintern/fashion-finegrained-vision/models/detectors/fashionpedia_yolov8s_19cls_balanced_v2_best.pt
```

## Step 4: Compare v1 vs v2 (local)

```bash
conda activate fashion-demo2
cd D:/Aliintern/fashion-finegrained-vision

# Update FP_MODEL in eval_validation_v2.py to point to v2 (temporarily)
# Then run eval:
python scripts/eval_validation_v2.py --device cuda --max-images 200

# Or run a focused comparison on FP-core parts only:
python scripts/eval_validation_v2.py --device cuda --skip-dino --max-images 300
```

Compare `data/validation/eval_v2/metrics.json` against the v1 baseline.

## Decision Gate

| If v2 mAP50... | Then... |
|---|---|
| ≥ 0.38 AND no FP part regresses >2% | Switch to v2, update all model paths |
| < 0.35 | Discard v2, keep v1 |
| 0.35–0.38 | Human review per-part metrics before deciding |

## Rollback

To revert to v1:
```python
# In eval_validation_v2.py and other files:
FP_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
```
