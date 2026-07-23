# 3.1.3 FashionAI Attribute Classifier 重新训练指南

> checkpoint误删，需在服务器上重新训练8个ResNet18属性分类器

---

## 需要拷贝到服务器的文件

```bash
# 最小集合 — 只需要这4个目录/文件：
fashion-finegrained-vision/
  tools/train/train_attribute_classifier.py      # 训练脚本
  datasets/fashionai_attribute_dataset.py         # Dataset类
  models/attribute_classifier.py                  # 模型架构
  data/fashionai_attribute_index/                 # 全部数据+标签(40个文件)
  configs/attribute_inference.yaml                # 推理配置
```

打包命令（在项目根目录）：
```bash
tar -czf fashionai_313_training.tar.gz \
  tools/train/train_attribute_classifier.py \
  datasets/fashionai_attribute_dataset.py \
  models/attribute_classifier.py \
  data/fashionai_attribute_index/ \
  configs/attribute_inference.yaml
```

文件大小：`data/fashionai_attribute_index/` 约200-300MB（含裁剪图像JSONL路径引用），打包后整体约100-200MB。

---

## 环境要求

```bash
pip install torch torchvision scikit-learn numpy pillow pyyaml tqdm
```

CUDA推荐，CPU也可训练（只是慢一些）。

---

## 8个任务训练命令

**公共参数**: `--arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 --seed 42`

### Task 1: neckline_design（领型，10类）
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/neckline_design_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/neckline_design_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_neckline_design.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_neckline_design_resnet18_seed2
```

### Task 2: collar_design（领子设计，5类）
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/collar_design_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/collar_design_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_collar_design.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_collar_design_resnet18_seed2
```

### Task 3: neck_design（颈部设计，5类）
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/neck_design_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/neck_design_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_neck_design.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_neck_design_resnet18_seed2
```

### Task 4: lapel_design（翻领设计，5类）
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/lapel_design_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/lapel_design_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_lapel_design.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_lapel_design_resnet18_seed2
```

### Task 5: sleeve_length（袖长，8类）— multiview_v2_pipeline
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/sleeve_length_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/sleeve_length_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_sleeve_length.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_sleeve_length_multiview_v2_pipeline_resnet18_seed2
```

### Task 6: coat_length（外套长度，5类）
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/coat_length_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/coat_length_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_coat_length.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_coat_length_resnet18_seed2
```

### Task 7: pant_length（裤长，6类）— multiview_v2_pipeline
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/pant_length_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/pant_length_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_pant_length.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_pant_length_multiview_v2_pipeline_resnet18_seed2
```

### Task 8: skirt_length（裙长，5类）— multiview_v2_pipeline
```bash
python tools/train/train_attribute_classifier.py \
  --train-jsonl data/fashionai_attribute_index/skirt_length_train.jsonl \
  --val-jsonl data/fashionai_attribute_index/skirt_length_val.jsonl \
  --label-map data/fashionai_attribute_index/label_map_skirt_length.json \
  --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
  --seed 42 --output-dir outputs/p2_skirt_length_multiview_v2_pipeline_resnet18_seed2
```

---

## 一键批量训练脚本（Linux服务器）

```bash
#!/bin/bash
# 在服务器项目根目录执行
set -e

TASKS=(
  "neckline_design:neckline_design_resnet18_seed2"
  "collar_design:collar_design_resnet18_seed2"
  "neck_design:neck_design_resnet18_seed2"
  "lapel_design:lapel_design_resnet18_seed2"
  "sleeve_length:sleeve_length_multiview_v2_pipeline_resnet18_seed2"
  "coat_length:coat_length_resnet18_seed2"
  "pant_length:pant_length_multiview_v2_pipeline_resnet18_seed2"
  "skirt_length:skirt_length_multiview_v2_pipeline_resnet18_seed2"
)

for entry in "${TASKS[@]}"; do
  TASK="${entry%%:*}"
  OUTDIR="outputs/p2_${entry##*:}"

  echo "============================================================"
  echo "Training: $TASK -> $OUTDIR"
  echo "============================================================"

  python tools/train/train_attribute_classifier.py \
    --train-jsonl "data/fashionai_attribute_index/${TASK}_train.jsonl" \
    --val-jsonl "data/fashionai_attribute_index/${TASK}_val.jsonl" \
    --label-map "data/fashionai_attribute_index/label_map_${TASK}.json" \
    --arch resnet18 --epochs 20 --batch-size 32 --img-size 224 --lr 0.0003 \
    --seed 42 \
    --output-dir "$OUTDIR"

  echo "Done: $TASK checkpoint -> $OUTDIR/best.pt"
done

echo ""
echo "All 8 tasks done. Copy outputs/p2_*/best.pt back to project."
```

---

## 训练完成后

把服务器上8个 `outputs/p2_*/best.pt` 拷贝回本项目的对应路径：

```
outputs/p2_neckline_design_resnet18_seed2/best.pt
outputs/p2_collar_design_resnet18_seed2/best.pt
outputs/p2_neck_design_resnet18_seed2/best.pt
outputs/p2_lapel_design_resnet18_seed2/best.pt
outputs/p2_sleeve_length_multiview_v2_pipeline_resnet18_seed2/best.pt
outputs/p2_coat_length_resnet18_seed2/best.pt
outputs/p2_pant_length_multiview_v2_pipeline_resnet18_seed2/best.pt
outputs/p2_skirt_length_multiview_v2_pipeline_resnet18_seed2/best.pt
```

拷贝后执行验证：
```powershell
conda activate fashion-demo2
$env:VISION_ATTR_BACKEND="fashionai"
$env:VISION_ATTR_ENABLE_REAL="true"
$env:VISION_ATTR_DEVICE="cuda"
python -c "
from inference.serving.attribute_backend import FashionAttributeBackend
b = FashionAttributeBackend(device='cuda')
print(f'Loaded: {b._ensure_loaded()}, tasks: {len(b._tasks)}')
"
```

预期输出: `Loaded: True, tasks: 8`

---

## 各任务数据量参考

| Task | 类别数 | Train | Val | 之前Test F1 |
|---|---|---|---|---|
| neckline_design | 10 | 1647 | ~400 | 0.665 |
| collar_design | 5 | ~1200 | ~300 | 0.764 |
| neck_design | 5 | ~1200 | ~300 | 0.624 |
| lapel_design | 5 | ~1200 | ~300 | 0.680 |
| sleeve_length | 8 | 1355 | ~350 | 0.612 |
| coat_length | 5 | ~1000 | ~250 | 0.618 |
| pant_length | 6 | ~1000 | ~250 | 0.740 |
| skirt_length | 5 | ~800 | ~200 | 0.593 |

每个任务GPU训练约5-15分钟（ResNet18，20 epochs），8个任务总计约1-2小时。
