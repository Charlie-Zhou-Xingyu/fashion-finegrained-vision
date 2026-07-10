# 3.1.2 工业级推进冲刺计划 v2.0

> 撰写：2026-06-25  
> 依据：Phase 2工程审视报告 + 今日网络调研  
> 状态：**执行版，按天分解，含备案**  
> 关联文档：`docs/industrial_grounding_implementation_plan.md`（全量计划），`docs/phase2_engineering_review.md`（Phase 2审视）

---

## 一、当前局面判断（执行前必读）

### 根本瓶颈（已确认）

DINO-tiny（48M参数）零样本场景下无法检测拉链/扣子/口袋：
- Calibration v2（100张coat图，有shape priors）：pocket P=0.073，zipper P=0.066，button P=0.118
- 可视化直接证实：DINO-tiny输出是整件衣服框，不是实际部件
- 这是一个**模型能力问题**，不是工程配置问题

### 评估基础设施有4处Bug影响结论可信度

在做任何模型决策前，必须先修这4个Bug：

| Bug | 影响 | 修复位置 |
|---|---|---|
| 坐标系错误：calibration在全图空间跑，shape priors用全图bbox | zipper recall被低估（1.8宽高比在全图过严） | `scripts/calibrate_part_thresholds.py` |
| garment_mask=None：calibration从未测试mask gating效果 | FP数字被高估 | 同上 |
| 标注不一致：button单颗/整排混用，拉链标注不一致 | 召回率计算不准确 | Label Studio人工操作 |
| 多实例缺陷：router只返回kept[0] | 双口袋场景TP被漏算 | `region_localization_router.py` |

### 关键新发现（今日网络调研）

**Fashionpedia数据集（HuggingFace）是Phase 6的现成训练数据：**
- 46,781张图片，342,182个bbox标注
- 直接包含：`zipper`、`pocket`、`collar`、`lapel`、`fringe`、`ruffle` 的标注
- 一行代码下载，Pascal VOC格式，转换成本低
- 这意味着Phase 6 fine-tuning不需要从零标注数据

**GroundingDINO LoRA微调实践经验（learnopencv + Asad-Ismail/Grounding-Dino-FineTuning）：**
- lr=1e-5，AdamW，batch_size=4-8
- 早停在epoch 5-10最优，epoch 45后过拟合明显
- LoRA rank=32，<2%参数量，显存友好
- 冻结视觉backbone前几层效果更好

### 战略执行顺序

```
[Day 0] 标注质量修复（人工，30min）
    ↓
[Day 1] 修复评估基础设施 → Calibration v3（正确坐标系）
    ↓
[Day 2] DINO-base对比实验 → 决策门
    ↙                    ↘
[Day 3A]                 [Day 3B]
切换DINO-base             触发Phase 6 → Fashionpedia数据准备
完成配置更新                    ↓
运行回归测试             [Day 4] LoRA微调DINO-base
    ↓                          ↓
[Day 5] Phase 4（mask containment）+ Phase 3（Qwen翻译，有GPU时）
```

---

## 二、Day 0：执行前准备（今天，30分钟人工）

### D0.1 Label Studio标注质量修复

打开Label Studio，进入 coat_annotation_batch1 项目：

1. **删除2个placket标注**（filter by label=placket，确认删除）
2. **统一拉链标注规则**：今后无论开合，统一标整条拉链外接矩形（不要标2条框）
3. **统一button标注规则**：改为button_cluster（整排外接矩形），不标单颗
4. **重新导出**：Export → COCO JSON → 保存为 `data/validation/coat_annotation_batch1_coco_v2.json`

> 如果Label Studio本地数据库损坏或无法启动，直接用Python脚本修改原JSON中的annotations数组，删除category_id对应placket的条目。

### D0.2 确认YOLO模型路径

```bash
# 确认模型存在
ls models/detectors/yolov8n_deepfashion2_13cls_best.pt
```

---

## 三、Day 1（2026-06-25）：修复评估基础设施 + Calibration v3

**目标：得到在正确坐标系下的Calibration v3数字，用于Day 2决策**

### D1.1 对100张coat图运行YOLO → 生成garment detections

```bash
python tools/infer/predict_garments_yolo.py \
  --source data/validation/coat_images/ \
  --model models/detectors/yolov8n_deepfashion2_13cls_best.pt \
  --output data/validation/coat_yolo_detections.json \
  --conf 0.25
```

如果 `predict_garments_yolo.py` 不支持JSON输出，直接在脚本里加保存逻辑：

```python
# 在predict_garments_yolo.py中添加 --save-json 选项
# 输出格式：{image_filename: [{bbox_xyxy, conf, cls_id, cls_name}]}
import json
detections = {}
for result in results:
    fname = Path(result.path).name
    detections[fname] = [
        {"bbox_xyxy": box.xyxy[0].tolist(), "conf": float(box.conf), "cls_id": int(box.cls)}
        for box in result.boxes
    ]
json.dump(detections, open("coat_yolo_detections.json", "w"), indent=2)
```

### D1.2 修复calibration脚本：加入garment crop坐标系

**文件：** `scripts/calibrate_part_thresholds.py`

**修复点1：接受garment detections参数**

```python
# 在argparse中添加
parser.add_argument("--garment-detections", type=str, default=None,
                    help="JSON: {image_file: [{bbox_xyxy, conf, cls_id}]}")
```

**修复点2：在garment crop坐标系内运行DINO（核心修复）**

```python
def run_detection_in_garment_crop(
    image: np.ndarray,
    garment_bbox: List[int],  # [x1, y1, x2, y2] in full image coords
    prompts: List[str],
    locator,
    pad_px: int = 8,
) -> List[dict]:
    """
    Run DINO on the garment crop, then remap detections to full-image coords.
    This fixes the coordinate space bug from calibration v1/v2.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = garment_bbox
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w, x2 + pad_px)
    y2 = min(h, y2 + pad_px)
    crop = image[y1:y2, x1:x2]

    # Run DINO on the crop
    raw_detections = locator.detect_multi_prompt(crop, prompts)

    # Remap crop-space coords back to full-image space
    for det in raw_detections:
        bx1, by1, bx2, by2 = det["bbox_xyxy"]
        det["bbox_xyxy"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]

    return raw_detections
```

**修复点3：GT坐标已在全图空间，无需转换；shape priors参照传garment_bbox（crop bbox）**

```python
# 在主评估循环中
garment_bbox_for_priors = [x1_crop, y1_crop, x2_crop, y2_crop]  # 同crop坐标
# filter_by_shape_priors(detections, part, garment_bbox=garment_bbox_for_priors)
```

**修复点4：添加garment_mask支持（可选，如果有SAM-HQ masks）**

```python
# ponytail: 先不接入，mask gating是可选的，calibration v3目的是验证坐标系修复效果
# garment_mask=None for now, add in calibration v4 if masks are available
```

### D1.3 修复router多实例返回

**文件：** `src/fashion_vision/localization/region_localization_router.py`

找到 `locate_region()` 最终返回kept[0]的位置，改为返回列表：

```python
# 原始（错误）：
if kept:
    return {**kept[0], "status": "success"}

# 修复后：
if kept:
    # ponytail: return all kept detections, caller picks
    return {
        "status": "success",
        "detections": kept,       # 全部通过shape priors的框
        "bbox": kept[0]["bbox_xyxy"],   # 兼容旧接口：保留最高分
        "score": kept[0]["score"],
        "n_detections": len(kept),
    }
```

注意：这是API变更，需要同步更新demo脚本和visualize脚本。

**更新 `tests/test_phase2_localization.py`**：在返回值断言中加上 `result["n_detections"]` 检查。

### D1.4 运行Calibration v3

```bash
python scripts/calibrate_part_thresholds.py \
  --images data/validation/coat_images/ \
  --annotations data/validation/coat_annotation_batch1_coco_v2.json \
  --garment-detections data/validation/coat_yolo_detections.json \
  --model IDEA-Research/grounding-dino-tiny \
  --parts pocket,zipper,button_cluster,collar \
  --threshold-sweep 0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
  --output calibration_v3_results/ \
  --model-tag dino-tiny-v3
```

**预期结果：** 与v2相比，zipper recall应该提高（坐标系修正），pocket和button可能有轻微改善。但DINO-tiny的根本能力问题仍存在，期望数字仍低于目标。

---

## 四、Day 2（2026-06-26）：DINO-base对比 + 决策门

**目标：得到决策数字，做出是否进入Phase 6的判断**

### D2.1 下载DINO-base并测试

```python
# 测试DINO-base能否加载（需要约6GB显存）
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to("cuda")
print(f"DINO-base loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

如果显存不足：
```bash
# INT8量化（需要bitsandbytes）
pip install bitsandbytes
# 在AutoModelForZeroShotObjectDetection.from_pretrained中加load_in_8bit=True
```

### D2.2 运行DINO-base Calibration（与v3完全相同的参数）

```bash
python scripts/calibrate_part_thresholds.py \
  --images data/validation/coat_images/ \
  --annotations data/validation/coat_annotation_batch1_coco_v2.json \
  --garment-detections data/validation/coat_yolo_detections.json \
  --model IDEA-Research/grounding-dino-base \
  --parts pocket,zipper,button_cluster,collar \
  --threshold-sweep 0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
  --output calibration_v3_dino_base_results/ \
  --model-tag dino-base-v3
```

### D2.3 决策矩阵（用Calibration v3数字填入）

| 场景 | DINO-base最佳P（目标部件） | 执行路径 |
|---|---|---|
| A：达标 | pocket≥0.65 AND zipper≥0.60 AND button≥0.55 | → Day 3A：切换DINO-base，完成 |
| B：显著提升但未达标 | 相比DINO-tiny提升≥15ppt | → Day 3B：Phase 6 LoRA微调DINO-base |
| C：轻微提升 | 提升5-15ppt | → Day 3B：Phase 6，同时考虑数据增强策略 |
| D：几乎无提升 | 提升<5ppt | → Day 3B：Phase 6，审查prompt设计是否有误 |

**注意：场景A概率较低（约20%），但必须先测。跳过这步直接fine-tuning是浪费GPU时间。**

---

## 五、Day 3A：DINO-base达标分支（估计概率20%）

如果D2.3判断为场景A，执行以下操作：

### A.1 更新模型配置

**文件：** `configs/attribute_inference.yaml` 或相应的localization配置

```yaml
localization:
  grounding_dino:
    model_name: "IDEA-Research/grounding-dino-base"  # 从 grounding-dino-tiny 改
    device: "cuda"
```

### A.2 更新PART_DETECTION_CONFIG的阈值

根据Calibration v3 DINO-base的结果，更新每个part的 `box_threshold` 和 `text_threshold`：

```python
# 在 part_detection_config.py 中，用实测最优阈值替换估计值
# 每个部件写注释：# calibrated on coat_annotation_batch1_coco_v2, 2026-06-26
```

### A.3 回归测试

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
# 确认：373+ passed, 0 failed
```

### A.4 在5张新图上手动验证

用 `scripts/visualize_localization_debug.py` 在之前未见过的图片上运行，确认效果。

---

## 六、Day 3B：触发Phase 6分支（估计概率80%）

如果DINO-base未达标，进入Phase 6 fine-tuning。关键优势：**Fashionpedia数据集已有现成标注**。

### B.1 下载Fashionpedia数据集

```python
# 脚本：scripts/download_fashionpedia.py
from datasets import load_dataset
import json
from pathlib import Path

print("下载Fashionpedia（约3.5GB）...")
ds = load_dataset("detection-datasets/fashionpedia", split="train")
# ds包含 image, objects字段
# objects: {bbox: [[x1,y1,x2,y2],...], category: [int,...], area: [...]}

# 保存为COCO JSON（方便后续处理）
output_dir = Path("data/fashionpedia")
output_dir.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(output_dir / "raw"))
print(f"已保存到 {output_dir}/raw")
```

### B.2 Fashionpedia类别映射

Fashionpedia的46个类别中与我们相关的：

| Fashionpedia类别名 | 我们的part名 | 预估正样本数 |
|---|---|---|
| zipper | zipper | ~8,000 |
| pocket | pocket | ~12,000 |
| collar | collar | ~20,000 |
| lapel | placket（近似） | ~6,000 |
| fringe | fringe | ~3,000 |
| ruffle | ruffle | ~5,000 |
| bow | tie_strap | ~2,000 |
| **无** | **button_cluster** | **0（需自有数据）** |

> **button_cluster的数据缺口：** Fashionpedia没有button bbox标注。当前100张coat标注中有button数据，但样本量不足。方案：Phase 6先不训练button，用DINO-base的最优threshold作为interim结果，后续单独收集button数据。

### B.3 数据格式转换（Fashionpedia → GroundingDINO）

**新建脚本：** `scripts/convert_fashionpedia_to_grounding_dino.py`

```python
"""
Convert Fashionpedia HuggingFace dataset to GroundingDINO COCO training format.

GroundingDINO training requires:
- COCO JSON format annotations
- Text captions per image (category names concatenated with ' . ')
- Bounding boxes in [x, y, w, h] format (COCO standard)
"""
import json
from pathlib import Path
from datasets import load_from_disk
from PIL import Image
import io

# Fashionpedia category IDs → our part names (hand-verified mapping)
FASHIONPEDIA_TO_PART = {
    # category_id (0-indexed): part_name
    # Run: ds.features["objects"]["category"].feature.names to get full list
    # Then map manually:
    "zipper": "zipper",
    "pocket": "pocket",
    "collar": "collar",
    "lapel": "placket",
    "fringe": "fringe",
    "ruffle": "ruffle",
    "bow": "tie_strap",
}

TARGET_PARTS = list(FASHIONPEDIA_TO_PART.values())
MAX_SAMPLES_PER_PART = 3000  # 防止数据不平衡

def convert(dataset_path: str, output_dir: str, max_per_part: int = MAX_SAMPLES_PER_PART):
    ds = load_from_disk(dataset_path)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": []}
    
    # 先确认类别ID（需要在实际下载后填写）
    # category_names = ds.features["objects"]["category"].feature.names
    # 本脚本需要在实际数据下载后根据真实category_names填写FASHIONPEDIA_ID_MAP
    
    part_counts = {p: 0 for p in TARGET_PARTS}
    ann_id = 1
    
    for img_id, sample in enumerate(ds):
        # 按part控制样本数
        sample_parts = []
        for cat_id, bbox in zip(sample["objects"]["category"], sample["objects"]["bbox"]):
            cat_name = ds.features["objects"]["category"].feature.names[cat_id]
            if cat_name in FASHIONPEDIA_TO_PART:
                part = FASHIONPEDIA_TO_PART[cat_name]
                if part_counts[part] < max_per_part:
                    sample_parts.append((part, bbox, cat_id))
        
        if not sample_parts:
            continue
        
        # 保存图片
        img = sample["image"]
        img_filename = f"fashionpedia_{img_id:06d}.jpg"
        img.save(str(images_dir / img_filename))
        
        coco["images"].append({
            "id": img_id, "file_name": img_filename,
            "width": img.width, "height": img.height
        })
        
        for part, bbox, cat_id in sample_parts:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": TARGET_PARTS.index(part),
                "bbox": [x1, y1, w, h],  # COCO format
                "area": w * h,
                "iscrowd": 0,
            })
            part_counts[part] += 1
            ann_id += 1
    
    coco["categories"] = [
        {"id": i, "name": p} for i, p in enumerate(TARGET_PARTS)
    ]
    
    json.dump(coco, open(str(output_dir / "fashionpedia_parts_train.json"), "w"), indent=2)
    print("Part counts:", part_counts)
    print(f"Total annotations: {ann_id - 1}")

if __name__ == "__main__":
    convert("data/fashionpedia/raw", "data/fashionpedia/coco_format")
```

### B.4 混合训练数据集

```python
# scripts/merge_training_datasets.py
# 合并 Fashionpedia COCO JSON + 自有100张coat标注

import json

def merge_coco(coco_a_path, coco_b_path, output_path, b_weight=20):
    """
    Merge two COCO JSONs. b_weight: how many times to repeat dataset B
    (use for upsampling small own-data to match Fashionpedia scale)
    """
    a = json.load(open(coco_a_path))
    b = json.load(open(coco_b_path))
    
    # 重新分配ID
    offset_img = max(x["id"] for x in a["images"]) + 1
    offset_ann = max(x["id"] for x in a["annotations"]) + 1
    
    merged_images = list(a["images"])
    merged_anns = list(a["annotations"])
    
    for repeat in range(b_weight):
        for img in b["images"]:
            merged_images.append({**img, "id": img["id"] + offset_img + repeat * 10000})
        for ann in b["annotations"]:
            merged_anns.append({
                **ann,
                "id": ann["id"] + offset_ann + repeat * 100000,
                "image_id": ann["image_id"] + offset_img + repeat * 10000,
            })
    
    merged = {**a, "images": merged_images, "annotations": merged_anns}
    json.dump(merged, open(output_path, "w"), indent=2)
    print(f"Merged: {len(merged_images)} images, {len(merged_anns)} annotations")

# 用法：
# merge_coco(
#     "data/fashionpedia/coco_format/fashionpedia_parts_train.json",
#     "data/validation/coat_annotation_batch1_coco_v2.json",
#     "data/training/merged_finetune_train.json",
#     b_weight=20  # 自有数据过采样20倍，补偿数量差距
# )
```

---

## 七、Day 4（2026-06-28）：Phase 6 LoRA微调

**前提：** Day 3B的数据准备完成

### D4.1 微调配置

**新建：** `configs/dino_finetune.yaml`

```yaml
# GroundingDINO LoRA fine-tuning configuration
model:
  name: "IDEA-Research/grounding-dino-base"
  # ponytail: fine-tune base not tiny; larger model generalizes better on small datasets

training:
  lora_rank: 32              # <2% parameters
  learning_rate: 1.0e-5
  optimizer: "AdamW"
  weight_decay: 0.01
  batch_size: 4              # adjust by VRAM
  max_epochs: 30
  early_stop_patience: 5    # stop if val precision doesn't improve for 5 epochs
  val_interval: 2
  freeze_layers: 12         # freeze first 12 vision backbone layers

inference:
  nms_threshold: 0.3
  # part-specific thresholds will be calibrated post-training (calibration v4)

data:
  train: "data/training/merged_finetune_train.json"
  val: "data/validation/coat_annotation_batch1_coco_v2.json"
  train_images: "data/fashionpedia/coco_format/images/"
  val_images: "data/validation/coat_images/"
```

### D4.2 训练监控要点

根据learnopencv调研：
- **epoch 5-10**：通常是最优点，精度提升最快
- **epoch 45后**：过拟合开始，box confidence反而下降
- 监控指标：validation precision（不是training loss）
- 每2个epoch在val set上跑calibration，记录P和R

### D4.3 运行微调

```bash
# 使用 Asad-Ismail/Grounding-Dino-FineTuning 或官方GroundingDINO训练脚本
python scripts/finetune_grounding_dino.py \
  --config configs/dino_finetune.yaml \
  --checkpoint-dir checkpoints/dino_base_finetune_v1/
```

### D4.4 Calibration v4（微调后评估）

```bash
python scripts/calibrate_part_thresholds.py \
  --images data/validation/coat_images/ \
  --annotations data/validation/coat_annotation_batch1_coco_v2.json \
  --garment-detections data/validation/coat_yolo_detections.json \
  --model checkpoints/dino_base_finetune_v1/best_checkpoint.pth \
  --model-type custom  \
  --parts pocket,zipper,button_cluster,collar \
  --output calibration_v4_results/ \
  --model-tag dino-base-finetuned-v1
```

---

## 八、Day 5（2026-06-29）：Phase 4 + Phase 3

### 优先级判断（按实际情况选择）

| 可用资源 | 优先执行 |
|---|---|
| 有多件服装叠穿图片可以看SAM-HQ mask | Phase 4 mask_containment.py |
| 有GPU服务器（AutoDL等）| Phase 3 Qwen-VL翻译服务 |
| 两者都有 | Phase 4（纯CPU，不需要服务器） |
| 两者都没有 | 补充button标注数据 / 完善测试 |

### Phase 4：mask_containment.py（先目视检查）

**前提（D16检查清单）：** 先对20张多件服装叠穿图片检查SAM-HQ masks质量：
- 每件衣服是否有独立mask
- mask边界是否干净
- 外套mask是否明显比内搭大

如果masks不合格，跳过Phase 4，在result中使用 `_hierarchy_status = "fallback_area_sort"` 并记录。

**如果masks合格，实现 `src/fashion_vision/localization/mask_containment.py`**（代码规格见主计划Section 4.2）。

### Phase 3：Qwen-VL-7B-Chat翻译服务

**按主计划Section 3.3执行（代码规格已完整，直接实现）**

服务器推荐：AutoDL RTX 3090（约¥3/小时），估计总用时<5小时（初始化+测试），成本<¥15。

部署验证checklist（D11-D13）：
- [ ] 10个长尾中文时装词汇测试：肩缝、荷叶边、流苏、绑带等
- [ ] 超时测试：关闭服务器，确认 `source="fallback_literal"` 
- [ ] 缓存测试：同一词汇第二次调用不触发HTTP请求

---

## 九、风险应对矩阵

| 风险 | 概率 | 影响 | 应对方案 |
|---|---|---|---|
| DINO-base VRAM不足（<8GB） | 中 | 阻塞Day2 | 用INT8量化（load_in_8bit=True）；或申请GPU服务器 |
| Fashionpedia无button标注 | **确定** | 中 | button暂用DINO-base最优threshold；后续单独收集button图 |
| LoRA微调后精度仍<0.55 | 低-中 | 高 | 增大训练数据（自有标注扩展至300张）；调整LoRA rank；数据增强 |
| SAM-HQ masks质量差（多件叠穿） | 中 | 中 | Phase 4推迟；garment_ref只用class prior过滤 |
| GPU服务器贵/不可用 | 中 | 低 | Phase 3推迟；扩展本地词汇表覆盖90%以上常见长尾词 |
| calibration v3坐标转换有bug | 中 | 高 | 单图可视化验证（D2检查：GT框与DINO输出框应在同一坐标系） |
| Fashionpedia category names与预期不符 | 低 | 中 | 实际下载后 `print(ds.features["objects"]["category"].feature.names)` 确认 |
| LoRA epoch 45后过拟合 | 中 | 中 | 用validation-based早停，不要训练到完整100 epochs |

### 关键备案：Phase 6不依赖GPU服务器的替代路径

如果没有GPU服务器，Phase 6微调无法进行，但可以：
1. 扩大本地PART_VOCAB：从Fashionpedia标注文本中提取更多中英文synonym，通过prompt engineering改善DINO-base零样本能力
2. 改变检测策略：对拉链/扣子等小部件，使用**两阶段策略**（先检测服装类型，再用固定landmark位置估算部件位置），而非依赖DINO开放词汇检测
3. 使用FashionCLIP特征：将part detection转为相似度检索（检索Fashionpedia中的部件模板），无需fine-tuning

---

## 十、成功标准

### 5天结束时最低可接受结果（Must Have）

- [ ] Calibration v3在正确坐标系下完成，有新的P/R数字
- [ ] DINO-base vs DINO-tiny对比完成，决策已记录
- [ ] 多实例修复已合并

### 理想结果（Should Have）

- [ ] DINO-base在pocket/zipper上P≥0.60，R≥0.45
- [ ] 或：Fashionpedia数据准备完成，LoRA微调已启动
- [ ] Phase 4 mask_containment.py完成（如果masks合格）

### 超额结果（Nice to Have）

- [ ] Phase 6微调完成，P≥0.65在2+个部件
- [ ] Phase 3 Qwen翻译服务部署并测试通过
- [ ] Phase 5复合锚点路由初版实现

---

## 十一、每日结束时需要记录的内容

每天工作结束，请在 `docs/phase2_engineering_review.md` 底部追加一个日志条目：

```markdown
## 日志：YYYY-MM-DD

### 完成
- 

### 数字（Calibration结果）
| 模型 | Part | 最佳P | 最佳R | 阈值 |
|---|---|---|---|---|

### 发现的问题
- 

### 明天计划
- 
```

---

## 十二、参考资料

调研来源（2026-06-25）：

- **Fashionpedia数据集（HuggingFace）**：`detection-datasets/fashionpedia` — 包含zipper/pocket/fringe/ruffle标注，45k图，3.5GB，可一行代码下载
- **Fashionpedia官网**：[fashionpedia.github.io](https://fashionpedia.github.io/home/) — 本体定义，27主类 + 19部件 + 294细粒度属性
- **GroundingDINO LoRA微调实践**：[learnopencv.com/fine-tuning-grounding-dino](https://learnopencv.com/fine-tuning-grounding-dino/) — 关键发现：epoch 5-10早停，epoch 45后过拟合
- **Asad-Ismail/Grounding-Dino-FineTuning**：[github.com/Asad-Ismail](https://github.com/Asad-Ismail/Grounding-Dino-FineTuning) — 可用微调脚本，LoRA rank=32，lr=1e-5，AdamW
- **官方IDEA-Research/GroundingDINO**：[github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — 原始实现，ECCV 2024
- **FashionFail论文**（2024）：addressing failure cases in fashion object detection — 与本项目类似的失败案例分析
- **NCBI：Efficient Fine Tuning for Fashion Object Detection** — adapter模块方案，Garment40K数据集，GroundingDINO adapter

---

*计划版本：v2.0 | 执行周期：2026-06-25 至 2026-06-29 | 下次更新：Phase 6决策后*
