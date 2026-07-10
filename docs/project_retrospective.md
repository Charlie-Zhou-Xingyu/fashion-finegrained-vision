# Fashion Fine-grained Vision — 项目完整复盘报告

> Date: 2026-07-10 | Author: Charlie Zhou
> Scope: PRD 3.1 全部子模块 (3.1.1 / 3.1.2 / 3.1.3)
> Purpose: 答辩用完整技术回顾

---

## 一、项目背景与 PRD 对齐

### 1.1 项目定位

构建面向时尚电商的多模态视觉分析系统。输入一张时尚图片和自然语言查询（如"这件外套的拉链在哪里"），系统输出：(1) 每件服装的检测框和分割 mask，(2) 用户查询指向的局部区域坐标，(3) 该服装的细粒度属性标签（领型、袖长、衣长等）。

### 1.2 PRD 3.1 模块划分

| 模块 | 描述 | 输入 | 输出 | 当前状态 |
|---|---|---|---|---|
| **3.1.1** | 服装实例分割 | 图片 | 每件服装的 bbox + mask + 类别 | 已完成 |
| **3.1.2** | 语言引导的局部区域定位 | 图片 + NL查询 | 目标区域的 bbox + mask | 核心完成，持续优化 |
| **3.1.3** | 细粒度属性提取 | 服装 crop | 属性标签 (领型/袖长/...) | 基线完成 |

### 1.3 明确不做的事情

- 鞋子、包、配饰（PRD 3.1 未覆盖）
- 3.2 多模态 QA、3.3 Agent/RAG（后续阶段）
- 推理速度优化作为第一优先级（功能正确性优先）

---

## 二、3.1.1 服装实例分割

### 2.1 数据：DeepFashion2

**数据集规模：** 训练集 135,975 张图片，221,535 个服装实例。验证集 (val2000) 2,000 张。每张图片可能包含 1-6 件服装。

**13 类原生类别 (DeepFashion2 标注体系)：**

```
 0: short sleeve top       1: long sleeve top
 2: short sleeve outwear   3: long sleeve outwear
 4: vest                   5: sling
 6: shorts                 7: trousers
 8: skirt                  9: short sleeve dress
10: long sleeve dress     11: vest dress
12: sling dress
```

**类别分布不均：** short sleeve top 占比约 30%，sling dress 占比 <2%。这是后续需要平衡重训练的原因。

### 2.2 双标签设计：为什么需要 13→5 映射

**这是一个关键的架构决策。** PRD 3.1.1 要求对外输出 5 个粗粒度类别（top/pants/skirt/outerwear/dress），但内部模块不能只用 5 类。原因：

1. **关键点预测依赖精细类别：** short sleeve top 和 long sleeve top 的关键点分布不同（袖子长度差异），如果合并为 "top"，关键点模型无法区分短袖/长袖的袖口位置
2. **局部区域定位需要精细类别：** vest 无袖、sling 单肩，这些结构差异影响 landmark 裁剪逻辑
3. **向后兼容：** 下游评估脚本和 crop 记录都按 13 类组织

**设计方案：每个检测实例同时携带两份标签：**

```python
# src/fashion_vision/schemas/instance_schema.py 和 predict_garments_yolo.py
{
    "class_id": 0,                # 原始 YOLO 输出
    "class_name": "short sleeve top",
    "fine_class_id": 0,           # 13 类
    "fine_class_name": "short sleeve top",
    "coarse_class_id": 0,         # 5 类 (PRD facing)
    "coarse_class_name": "top",
}
```

**映射规则在配置文件 `configs/category_mapping.yaml` 中管理，不在代码中硬编码：**

```yaml
# 13-to-5 mapping
0: 0    # short sleeve top -> top
1: 0    # long sleeve top -> top
2: 3    # short sleeve outwear -> outerwear
3: 3    # long sleeve outwear -> outerwear
4: 0    # vest -> top
5: 0    # sling -> top
6: 1    # shorts -> pants
7: 1    # trousers -> pants
8: 2    # skirt -> skirt
9: 4    # short sleeve dress -> dress
10: 4   # long sleeve dress -> dress
11: 4   # vest dress -> dress
12: 4   # sling dress -> dress
```

**为什么 vest 和 sling 归为 top 而不是 outerwear？** vest 和 sling 在 structural 语义上更接近 base layer（内层打底），而非 outer layer（外套）。这是 PRD 定义的映射方式。

### 2.3 方法链

```
图像 (BGR, HxWx3)
    |
    v
YOLOv8n 检测 (640x640 输入, conf>=0.25, NMS IoU 0.7)
    |  输出: N 个 [x1,y1,x2,y2, conf, class_id]
    |  class_id in [0,12], 双标签扩展为 fine + coarse
    |
    v
SAM-HQ vit_b 分割 (1024x1024 输入)
    |  输入: 原图 + 每个 YOLO bbox 作为 box prompt
    |  输出: N 个 binary mask (与原图同尺寸)
    |
    v
ResNet18 关键点预测 (256x256 输入)
    |  输入: 按 bbox 裁剪的 garment crop
    |  输出: 39 个关键点 (x,y) 坐标
    |
    v
区域裁剪 (基于关键点的几何规则)
    |  neckline/collar -> 领口矩形
    |  cuff/sleeve -> 袖口矩形 (左/右)
    |  hem -> 下摆矩形, waist -> 腰部矩形
    |  pant_leg -> 裤腿矩形, shoulder -> 肩部矩形
    |
    v
Mask-aware 区域裁剪 (SAM mask 与 几何区域 取交集)
    |  输出: 仅包含服装像素的裁剪区域 (背景置为白色/透明)
```

### 2.4 YOLOv8n 检测器详解

**模型规格：** YOLOv8n (Ultralytics)，~3M 参数，训练分辨率 640x640。

**平衡重训练 (P4)：**
- **问题：** DeepFashion2 类别极度不平衡 (short sleeve top 占 30%，sling dress <2%)
- **策略：** Inverse-sqrt frequency 权重采样，最大 repeat 5 次。少数类（sling dress, vest dress）每 epoch 被采样 5 次，多数类（short sleeve top）每 epoch 被采样 1 次
- **结果：** 所有 5 个 PRD 类 recall + precision 均提升，5 类 foreground accuracy 从 0.9274 -> 0.9379

**为什么用 YOLOv8n 而不是更大的模型？** 3M 参数，在 RTX 3090 上约 24ms。检测任务本身不是 pipeline 瓶颈（占 5.8% 总耗时）。更大的模型会增加延迟但 mAP 提升有限。

**效果验证方式：** `tools/eval/eval_13cls_confusion_as_5cls.py` — 先用 13 类 confusion matrix 评估，再通过配置映射聚合为 5 类，保留细粒度错误分析能力。

### 2.5 SAM-HQ 分割详解

**为什么选择 SAM-HQ 而不是原始 SAM？** SAM-HQ 在 mask 边界的精细度优于原始 SAM。服装边缘（袖口、领口、下摆）需要高质量 mask，因为后续 mask-gated DINO 推理直接将非服装像素填充，mask 边界不准确会导致零件被误裁剪。

**性能数据：** 编码器约 250ms（主要瓶颈，ViT-B 在 1024x1024 上做全分辨率 forward），解码器约 15ms/box x 平均 2.5 个 instance = 38ms，总计约 293ms。

**编码器为什么这么慢？** ViT-B 约 90M 参数，1024x1024 输入 -> patch size 16 -> 64x64 = 4096 个 patch tokens。12 层 transformer，每层做 self-attention on 4096 tokens。总计算量约 200 GFLOPs。RTX 3090 虽然在 FP16 tensor core 上有约 71 TFLOPS，但 ViT 的 attention 计算是 memory-bound（attention map 的读写量远超计算量），实际利用率不到 50%。

### 2.6 性能瓶颈的工程含义

**Amdahl 定律分析：**
- SAM-HQ 占 69.6%，优化它（293 -> 18ms）节省 275ms
- YOLO 占比 5.8%，优化它（24 -> 5ms）节省 19ms
- 即使把 YOLO 降到 0ms，也只节省 5.8%
- **结论：必须先换 SAM 模型架构，再做其他优化才有意义**

### 2.7 500 张图片 pipeline benchmark

| 阶段 | 耗时/500张 | 单张耗时 | 占比 |
|---|---|---|---|
| YOLO 检测 | 12.1s | 24.2ms | 5.8% |
| SAM-HQ 分割 | 146.3s | 292.6ms | 69.6% |
| 关键点预测 | 27.2s | 54.4ms | 13.0% |
| 区域裁剪 | 3.4s | 6.8ms | 1.6% |
| Mask 裁剪 | 21.1s | 42.2ms | 10.0% |
| **总计** | **210.1s** | **420.2ms** | - |

---

## 三、3.1.2 语言引导的局部区域定位

### 3.1 问题定义

**输入：** 一张时尚图片 + 自然语言查询（如"左边袖口"、"外套的拉链"、"胸前的口袋"）

**输出：** 目标区域的 bbox + mask，或 "not_detected" 状态

**难度层次：**
1. **简单（快路径）：** 6 个结构性部件（领口/袖口/下摆/腰部/肩部/裤腿），通过关键点+几何规则定位
2. **中等（Fashionpedia YOLO）：** 13 个高频零件（拉链/纽扣/口袋/翻领/肩章/蝴蝶结等），用专门训练的检测器
3. **困难（DINO 开放词汇）：** 任意开放词汇零件（荷叶边/流苏/铆钉/亮片等），用零样本视觉语言模型

### 3.2 架构演进过程

**阶段 0（P1 原型）：** 仅快路径，规则驱动，92% Batch60 有效响应。但只能处理 6 种结构性部件。

**阶段 1（Phase 0-1 开放词汇升级）：** 引入 Grounding DINO + 统一路由器。支持任意中英文查询。但发现以下问题：
- DINO 运行在整张图上，背景干扰严重
- 中文查询直接送入英文 DINO，完全无法匹配
- 没有 mask gating，DINO 看到相邻服装的零件
- 没有阈值调优，误检率极高

**阶段 2（Phase 2 工业级打磨）：** 逐零件配置阈值 + mask-gated + 解剖学缩放 + 形状先验。DINO 精度大幅提升。

**阶段 3（Phase 3 Fashionpedia 引入）：** 发现 DINO 在已标注零件上不如专门训练的 YOLO。引入 19 类 Fashionpedia 检测器作为优先路径，DINO 仅作为不受 FP 覆盖的零件的 fallback。

### 3.3 整体架构（当前版本）

```
用户中文查询 (如 "外套的拉链", "内搭的领口", "胸前的口袋")
    |
    v
+-----------------------------------------------------------+
|  intent_parser.parse_intent(query)                        |
|  输入: "外套的拉链"                                         |
|  输出: QueryIntent { part="zipper", garment_ref="outerwear",|
|          side=None, is_fast_path=False }                   |
+-----------------------------------------------------------+
    |
    v
+-----------------------------------------------------------+
|  region_localization_router.locate_region()               |
|  路由决策树:                                                |
|                                                           |
|  1. garment_ref mismatch 检查                              |
|     -> 查询"外套"但 instance 是 dress -> 标记 mismatch      |
|                                                           |
|  2. 内搭检测 (仅 outerwear + garment_ref="inner")          |
|     -> SAM multimask 找内层衣服轮廓                        |
|                                                           |
|  3. is_fast_path? (hem/waist/shoulder/leg_opening)         |
|     -> YES: 关键点 + 几何规则 -> 直接返回                   |
|                                                           |
|  4. part in Fashionpedia? (13 类 FP 核心零件)              |
|     -> YES: Fashionpedia YOLO 检测                        |
|         |- 命中 -> 早返回 (DINO 永不调用)                   |
|         |- 未命中 + neckline/cuff -> fast-path 回退        |
|         |- 未命中 + 其他 -> "not_detected"                 |
|                                                           |
|  5. part NOT in FP -> Grounding DINO                      |
|     |- 解剖学缩放 (anatomical_zoom)                        |
|     |- mask-gated 多 prompt 检测                          |
|     |- 空间约束 (left/right, upper/lower)                  |
|     |- 形状先验过滤                                        |
|     |- SAM box->mask 精修                                 |
+-----------------------------------------------------------+
```

### 3.4 意图解析详解 (intent_parser.py)

**为什么需要自己写解析器而不是用 LLM？**

1. **延迟：** LLM 翻译需要 900ms+，本地规则解析 <0.1ms
2. **可靠性：** LLM 输出格式不可控，可能返回错误 JSON
3. **覆盖：** 95%+ 用户查询落在已知词汇表内，不需要 LLM
4. **可调试：** 规则匹配的每个决策可追溯，LLM 是黑盒

**解析策略（按优先级排序）：**

```python
def parse_intent(query: str) -> QueryIntent:
    # 1. 提取侧边信息（最长匹配，"左边"优先于"左"）
    # 2. 提取方向信息（胸前->front_upper, 上方->upper）
    # 3. 提取服装指代（连衣裙->dress, 外套->outerwear, 内搭->inner）
    # 4. 匹配空间锚点（"X附近"->提取X作为spatial_anchor）
    # 5. 匹配特殊复合词（"连衣裙下摆"->hem+dress，"裙摆"->hem+skirt）
    # 6. 匹配通用零件词汇（最长匹配，"腰带"优先于"腰"）
    # 7. 零样本回退（未知词 -> 提取名词短语 -> DINO zero-shot）
```

**解决的具体歧义案例：**

| 查询 | 错误匹配 | 正确匹配 | 机制 |
|---|---|---|---|
| "腰带" | waist (腰) | belt | 最长匹配: "腰带"(2字) > "腰"(1字) |
| "连衣裙下摆" | hem (通用) | hem+dress | 特殊复合词优先于通用 hem |
| "裙摆" | hem (通用) | hem+skirt | garment_ref 隐式推导 |
| "左边袖口" | 无 side | cuff + left | side words 独立提取 |
| "胸前的口袋" | pocket | pocket + front_upper | direction 独立提取 |
| "帽兜上的拉链" | zipper | zipper + spatial_anchor | 正则匹配 |

**QueryIntent 数据结构：**

```python
@dataclass
class QueryIntent:
    raw_query: str              # 原始查询
    part: Optional[str]         # 规范零件名
    side: Optional[str]         # "left" / "right"
    garment_ref: Optional[str]  # "outerwear" / "skirt" / "dress" / "pants" / "inner"
    direction: Optional[str]    # "upper" / "lower" / "front_upper" / "back"
    spatial_anchor: Optional[str]  # 空间锚点
    is_fast_path: bool          # -> 关键点+几何管线
    is_zero_shot: bool          # -> 零样本 DINO fallback
```

**词汇覆盖（26 类零件，中英双语）：**

| 类别 | 中文关键词 | 英文关键词 |
|---|---|---|
| neckline | 衣领/领口/领子/脖颈 | neckline/collar/neck |
| cuff | 袖口/衣袖/袖子 | sleeve cuff/cuff/sleeve |
| hem | 下摆/衣摆/底边 | hem/bottom |
| waist | 腰围/腰线/腰部/裤腰 | waistline/waist |
| shoulder | 肩部/肩膀/肩线 | shoulder |
| leg_opening | 裤管/腿部/裤脚/裤腿 | leg opening/pant leg |
| zipper | 拉链/拉锁 | zipper/zip |
| button | 纽扣/扣子 | button |
| pocket | 口袋/衣兜 | pocket |
| placket | 门襟 | placket |
| pattern | 碎花/花纹/图案/印花 | pattern/print |
| belt | 腰带/皮带 | belt |
| lapel | 翻领/驳领/西装领 | lapel |
| epaulette | 肩章/肩袢 | epaulette |
| buckle | 扣环/皮带扣 | buckle |
| drawstring | 抽绳/收绳 | drawstring |
| tie_strap | 绑带/系带 | tie strap |
| ruffle | 荷叶边/波浪边 | ruffle |
| fringe | 流苏/穗子 | fringe |
| strap | 吊带/肩带 | strap |
| bag | 包/包包/手提包 | bag |
| shoes | 鞋子/鞋 | shoes |
| shoulder_seam | 肩缝/肩线缝合 | shoulder seam |
| sleeve_seam | 袖缝/袖子缝合 | sleeve seam |

### 3.5 为什么引入 Fashionpedia 检测器

**核心洞察：** Grounding DINO 是零样本模型，没有在服装零件上专门训练。在 Fashionpedia 已标注的 19 类零件上，一个专门训练的 YOLOv8s 的检测精度和速度都优于 DINO。

**Fashionpedia 数据集：** 46,000 张图片，19 类服装零件标注。

**训练过程：**
1. 数据集转换：COCO JSON -> YOLO txt 格式
2. 基线训练：YOLOv8s，标准配置，mAP50 约 0.47
3. 平衡重训练：p=1.0, repeat=12。平衡后 mAP50 0.312（下降了，因为增加了难例的权重），但 max:min 类比例从 217:1 -> 24:1，少数类 recall 显著提升

**在 router 中的位置：**
- **优先于 DINO：** FP 覆盖的 13 类零件直接用 YOLO
- **早返回机制：** FP 命中 -> 立即返回，DINO 不被调用
- **FP 未命中处理：** neckline/cuff -> fallback 到快路径（关键点+几何），不信任 DINO。其他 FP 零件 -> 返回 "not_detected"，不尝试 DINO。因为如果专门训练的 YOLO 检测器在 garment crop 上都找不到这个零件，那这个服装大概率确实没有这个零件；让 DINO 去找只会产生幻觉

**验证结果（50 张 FashionAI 图片）：** 100% 检测率（50/50 张至少检测到 1 个 collar），82% collar recall

### 3.6 Grounding DINO 详解

**模型：** `IDEA-Research/grounding-dino-tiny`，Vision Backbone: Swin-Tiny (~28M params)，Text Encoder: BERT-base，Cross-attention: vision-language fusion。

**集成挑战与解决方案：**

1. **Trailing period 问题：** HF GDINO 的 text processor 期望 prompt 以句号结尾。不带句号时 score calibration 不稳定。"zipper." 得分 0.45 而 "zipper" 得分 0.12。

```python
# grounding_dino_locator.py
if not text_query.rstrip().endswith("."):
    text_query = text_query.rstrip() + "."
```

2. **单阈值 API：** HF 的 post_process 只接受一个 threshold 参数，不像原始 GDINO 那样支持独立的 box_threshold 和 text_threshold。配置中保留两个阈值字段供未来使用。

3. **ONNX 导出困难：** `GroundingDinoForObjectDetection` 的 forward 包含动态控制流（post_process 根据检测数量做条件分支），torch.onnx.export 基于 tracing，无法处理。

### 3.7 Mask-gated 推理

**问题：** DINO 在 garment crop 上运行时，crop 中可能包含背景、手臂、相邻服装。这些区域的纹理与目标零件相似，导致误检。

**解决方案：** 在送入 DINO 前，将非服装像素填充为灰色 (128, 128, 128)：

```python
@staticmethod
def mask_to_garment(image, garment_mask, fill_mode="grey", dilation_px=0):
    out = image.copy()
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...)
        mask = cv2.dilate(mask, kernel)
    out[mask == 0] = 128  # 灰色填充
    return out
```

**为什么是灰色 (128) 而不是黑色 (0)？** 黑色会引入强边缘（garment 边界处从服装颜色跳变到纯黑），DINO 可能把这条假边缘误检为目标。灰色是 ImageNet 归一化后的"中性"像素值，视觉 backbone 对它没有响应。

**Mask 膨胀 (dilation_px)：** 对于小零件（zipper, button），它们可能正好位于服装边缘（拉链在门襟边缘，纽扣在袖口边缘）。如果 mask 太硬，零件的一部分像素被填充，DINO 检测框就不完整。button=3px, zipper=5px。

### 3.8 多 Prompt 合并机制

**为什么需要多个 prompt？** 单 prompt 的 recall 有限。"zipper" 可能漏检某些风格的拉链。描述性 prompt 能覆盖更多视觉表达：

```python
"zipper": {
    "prompts": [
        "a vertical metal zipper line on the front of a jacket",
        "a central zipper closure running down a coat",
        "a long thin zipper on clothing",
        "metal zipper teeth on a garment front",
    ],
}
```

**合并策略：** 每个 prompt 独立调用 detect() -> 所有检测结果合并到一个列表 -> 按 score 排序 -> NMS 去重。

### 3.9 关于 Soft-NMS 和 Class-aware IoU 的诚实评估

**这个方向被证明效果有限，不建议作为亮点展示。**

**原始设想：** 传统 greedy NMS 的 IoU 阈值全局统一（如 0.5），当两个同类别检测框的 IoU > 0.5 时，低分框被直接删除。对于服装零件场景，一件衣服上可能有多个同类型零件（两个口袋、一排纽扣、多个铆钉），greedy NMS 会把相邻的合法检测全删掉。

**尝试的改进：**
1. **Class-aware IoU：** 为多实例零件设置更低 IoU 阈值（pocket=0.30, button=0.20, rivet=0.15, sequin=0.10）
2. **Soft-NMS：** 用高斯惩罚替代硬删除 `score *= exp(-IoU^2 / sigma)`

**实际表现不佳的原因：**
- 多 prompt 机制已经产生了大量候选框（5 个 prompt x 每 prompt 5-10 个候选 = 25-50 个候选），低 IoU 阈值让几乎所有候选都存活，最终输出 10+ 个检测框，其中大部分是假阳性
- Soft-NMS 的衰减参数 sigma 极难调优——太小则和 greedy NMS 没区别，太大则假阳性泛滥
- 本质问题：DINO 在服装零件上的 score calibration 不够好（零样本模型的固有问题），过滤应该发生在 score threshold 和 shape priors 层面，而不是 NMS 层面

**教训：** 优化 pipeline 中最早出错的那一步，而不是在后续步骤修补。真正有效的过滤是形状先验（面积比/宽高比/位置 band）——这些基于服装结构知识，比基于 IoU 的纯几何去重大幅更可靠。

### 3.10 形状先验过滤 (part_shape_priors.py)

**这是实际证明最有效的误检过滤手段。**

**原理：** 服装零件在 garment 坐标系中有可预测的位置和形状。

**六类约束：**

| 约束 | 机制 | 例子 |
|---|---|---|
| min/max_area_ratio | bbox 面积 / garment bbox 面积 | button <= 10% |
| min/max_aspect_ratio_h_over_w | 高/宽比 | zipper >= 1.2（竖长）|
| min/max_aspect_ratio_w_over_h | 宽/高比 | belt >= 2.0（横长）|
| y_band | bbox 中心 Y / garment 高度 | collar in [0, 0.30]（上部30%）|
| x_band | bbox 中心 X / garment 宽度 | epaulette in [0, 0.25] |
| prefer_center_x | 倾向于 X 中心对称 | zipper, placket |
| mask_dilation_px | mask 膨胀半径 | button=3px, zipper=5px |

**关键迭代经验：**

1. **button 的 prefer_center_x 被移除。** 最初认为纽扣在前门襟中央，但实际上袖口纽扣和侧边纽扣大幅偏离中心。prefer_center_x=True 导致大量合法纽扣被拒绝。

2. **button 的 max_area_ratio 从 0.06 放宽到 0.10。** shape_priors 一度太严，杀死了所有 button 检测。

3. **返回空列表 = 信号，非错误。** 当所有候选被形状先验拒绝时，返回 []（而非 fallback 到未过滤的结果）。下游应返回 not_detected。

### 3.11 布料装饰类零件的特殊处理

**问题：** 亮片 (sequin)、流苏 (fringe)、荷叶边 (ruffle) 等装饰零件与结构性零件（拉链、纽扣）有本质区别：没有固定形状约束，没有固定位置，视觉特征复杂。

**sequin 的 prompt 迭代过程：**

| 迭代 | Prompt | 正检率 | 问题 |
|---|---|---|---|
| v1 | "sequin on clothing" | ~27% | 过度泛化，把任何反光布料当 sequin |
| v2 | "sequin decoration on garment" | ~35% | 仍然很多 FP |
| v3 (当前) | "shiny iridescent sequin discs sewn onto an evening dress" | **77%** | "iridescent/glittering/reflecting light" 有效过滤普通反光 |

**教训：** 对 DINO 来说，prompt engineering 不是"多说几个同义词"，而是"精确描述视觉特征"。"iridescent"（虹彩的）这个形容词比 "decoration"（装饰）效果好得多，因为它对应了可区分的视觉信号。

### 3.12 解剖学缩放 (anatomical_zoom.py)

**问题：** 小零件（纽扣 5x5px、铆钉 3x3px）在 640x640 的全图上只占几十个像素。DINO 的 Swin-Tiny backbone 下采样 32x，在 feature map 上这些零件只有 1-2 个 grid cell，根本无法检测。

**策略：** 根据零件类型裁剪 garment bbox 的子区域并放大。neckline/collar -> garment bbox 上部 60%，2x 放大。cuff -> 左侧或右侧 50%，2x 放大。hem -> 下部 50%，2x 放大。Fashionpedia YOLO 路径不使用解剖学缩放（避免分布偏移），仅做 garment bbox crop + 8px padding。

### 3.13 空间约束 (spatial_constraint.py)

- **Side (左右):** 按 bbox x 坐标重新排序（viewer perspective），若过滤后为空则返回原列表
- **Direction (上下):** upper=上半40%, lower=下半40%, front_upper=上半50%（胸前区域代理）, back=不支持（单张正面图无法判断）

### 3.14 内搭检测 (inner_garment_detector.py)

**场景：** 用户查询"内搭的领口"，实际图片上模特穿了外套+内搭。YOLO 可能只检测到外套（遮挡导致内搭不可见）。

**两阶段策略：**
1. **主策略：领口区域的几何互补分析。** 在 outerwear neckline ROI 内，寻找 outerwear mask 外但在 bbox 内的区域。通过 Canny 边缘检测 + 连通组件分析 + SAM 多 mask 候选，用得分函数筛选最佳内搭轮廓
2. **回退策略：SAM multimask 全 bbox 扫描。** 找与外套 mask 互补度最高的 SAM mask

**为什么只在用户明确查询"内搭"时触发？** SAM multimask 推理额外增加约 200ms。不全局运行。

### 3.15 当前 3.1.2 指标

| 指标 | 数值 | 评估方式 |
|---|---|---|
| 快路径有效响应率 (Batch60) | 92.0% | 300 查询 x 5 区域类型 |
| Fashionpedia collar 检测率 | 100% (50/50) | 50张 FashionAI 图片 |
| Fashionpedia collar recall | 82% | 标注 collar 中检出比例 |
| neckline/collar/lapel 合并 @IoU<=0.3 | 60.6% | 导师要求标准 |
| sequin DINO 正检率 | 77% | Prompt v3 优化后 |
| 测试覆盖 | 373+ passed, 2 skipped | pytest |

**IoU<=0.3 标准的含义：** 导师 07-09 评估设定检测框与 GT 框 IoU<=0.3 即算命中（而非通常的 0.5）。理由是服装零件标注本身带有主观性（同一领口的标注边界因人而异），0.3 的标准对应"大致定位正确"而非"精确对齐"。

### 3.16 DINO-base 评估实验 (2026-07-10)

**背景：** 导师 07-09 评估指出 DINO-tiny 在鞋子、铆钉等小零件上存在能力天花板，提出了切换 DINO-base 的可能性。DINO-base 使用 Swin-Base backbone (~100M params) 替代 Swin-Tiny (~28M)，理论上更强的视觉 backbone 应该提供更好的零样本检测精度。

**实验设计：** 在 eval_v2 验证集上做 Tiny vs Base 裸 DINO 对比（仅 garment crop，无 mask-gating/解剖学缩放/形状先验，控制变量），15 个零件 x 15 张/零件，评估 IoU>0.3 hit rate。

**实验环境：** NVIDIA RTX 4060 Laptop (8GB)，fashion-demo2 conda 环境，torch 2.5.1。脚本：`scripts/compare_dino_tiny_vs_base.py`。

**模型规格对比：**

| | DINO-tiny | DINO-base |
|---|---|---|
| 参数量 | 172M | 232M |
| 模型大小 (磁盘) | ~200MB | 1.8GB |
| GPU 显存占用 | 660 MB | 891 MB |
| 单次 detect 延迟 (4060) | 319 ms | 409 ms |
| 多 prompt detect (4 prompts) | 1257 ms | 1652 ms |

**精度对比（IoU>0.3 hit rate, 15 samples/part）：**

| 零件 | Tiny | Base | Delta | 判断 |
|---|---|---|---|---|
| collar | **66.7%** | 60.0% | **-6.7%** | Tiny 赢 |
| neckline | **20.0%** | 13.3% | -6.7% | Tiny 赢 |
| lapel | **20.0%** | 13.3% | -6.7% | Tiny 赢 |
| zipper | **46.7%** | 26.7% | **-20.0%** | Tiny 赢 |
| pocket | **46.7%** | 40.0% | -6.7% | Tiny 赢 |
| button | 20.0% | **40.0%** | +20.0% | Base 赢 |
| epaulette | 13.3% | **33.3%** | +20.0% | Base 赢 |
| fringe | 80.0% | **93.3%** | +13.3% | Base 赢 |
| bag | 46.7% | **53.3%** | +6.7% | Base 赢 |
| hood | 53.3% | **60.0%** | +6.7% | Base 赢 |
| rivet | 20.0% | 20.0% | 0 | 持平 |
| sequin | 80.0% | 80.0% | 0 | 持平 |
| shoes | 26.7% | 26.7% | 0 | 持平 |
| ruffle | 60.0% | 60.0% | 0 | 持平 |
| bow | 80.0% | 80.0% | 0 | 持平 |
| buckle | 66.7% | 66.7% | 0 | 持平 |
| **collar+neckline+lapel 合并** | **35.6%** | 28.9% | **-6.7%** | Tiny 赢 |

**关键发现：**

1. **核心结构零件退化：** collar/neckline/lapel/zipper/pocket 这 5 个用户查询频率最高的零件，Base 精度全面低于 Tiny。zipper 下降 20 个百分点最为显著。
2. **collar+neckline+lapel 合并：** Tiny 35.6% vs Base 28.9%，Base 反而退步 6.7 个百分点。
3. **小零件天花板未突破：** 原本期望 base 能改善的 shoes (26.7%) 和 rivet (20%) 完全持平——更强的 backbone 没有转化为更好的零样本检测能力。
4. **Base 的收益在 Fashionpedia 已覆盖的零件：** button (+20%)、epaulette (+20%) 是 Base 改善最大的两个零件，但这两个恰好是 Fashionpedia YOLO 的核心覆盖零件。如果 pipeline 已经有 FP YOLO 优先路径，DINO-base 的这些收益毫无意义——因为 FP YOLO 比 DINO-base 更快更准。
5. **延迟增加 31%：** Base 推理比 Tiny 慢 31%（409ms vs 319ms），显存多占 231MB。在 8GB 笔记本 GPU 上，891MB vs 660MB 的差距意味着 Base 在多模型共存场景下更危险。

**如果重来：** 关于"应该用 DINO-base"的假设被实验推翻。Base 在核心零件上反向退化，在小零件上无法突破天花板。如果后续需要做 DINO 微调，Base 作为起点仍有价值（更大的参数空间）；但作为零样本推理引擎，Tiny 更优。

**最终决策：保持 DINO-tiny 作为 3.1.2 的 DINO 推理引擎。** Base 的 button/epaulette 改善由 Fashionpedia YOLO 覆盖，无需 DINO 介入。

---

## 四、3.1.3 细粒度属性提取

### 4.1 问题定义

**输入：** 服装的 region crop（领口区域/袖子区域/全身 expanded crop）

**输出：** 细粒度属性标签（如 collar_design 的 5 类：翻领/立领/圆领/V领/无领）

### 4.2 数据困境

| 任务 | 类别数 | 训练样本 | PRD 目标 | 实际 Macro-F1 | 差距 |
|---|---|---|---|---|---|
| collar_design | 5 | ~1,647 | 0.88 | 0.764 | -0.116 |
| pant_length | 5 | ~1,200 | 0.88 | 0.740 | -0.140 |
| lapel_design | 5 | ~1,200 | 0.88 | 0.680 | -0.200 |
| neckline_design | 8 | ~1,600 | 0.88 | 0.665 | -0.215 |
| coat_length | 5 | ~1,200 | 0.88 | 0.618 | -0.262 |
| neck_design | 5 | ~800 | 0.88 | 0.624 | -0.256 |
| sleeve_length | 6 | ~1,300 | 0.88 | 0.612 | -0.268 |
| skirt_length | 5 | ~556 | 0.88 | 0.593 | -0.287 |

**根因分析：**
- 这是典型的**数据稀缺问题**而非模型容量问题。ResNet18（11M 参数）在 ImageNet 上能达到 70%+ Top-1，在 1600 个样本上训练 5-8 分类不应该只得到 0.6-0.76 F1
- 真正的问题是：(a) 样本数 556-1647 对于细粒度分类太少，(b) 类内方差大（同一类"翻领"在不同角度/光照/遮挡下差异巨大），(c) 类间差异小（"中长款"和"长款"的视觉边界模糊）
- **结论：需要更多数据，而非更大模型。**

### 4.3 训练方法

**模型：** ResNet18 (torchvision)，ImageNet 预训练，替换 FC 层。

**多视角增强策略 (multiview_v2_pipeline)：** 对每个训练样本生成 2-3 个不同视角的 crop（collar 区域 + upper body + 全身 expanded）。训练时随机选择一个视角，推理时用所有视角的 ensemble。

**注意：** collar_design/neck_design/lapel_design/coat_length 只训练了 baseline (seed2)，未跑 multiview_v2（训练时间限制）。这在 attribute_inference.yaml 中有明确标注。

### 4.4 推理 Pipeline

**配置驱动设计：** `configs/attribute_inference.yaml` 定义每个任务的 checkpoint、label_map、region_filter、class_contains。

**类别门控 (category_gate.py)：** 不是每件衣服都跑所有 8 个任务：
- top -> neckline_design, collar_design, neck_design, sleeve_length
- outerwear -> lapel_design, coat_length, sleeve_length
- pants -> pant_length
- skirt -> skirt_length
- dress -> neckline_design, skirt_length, sleeve_length

**为什么 pants 只跑 pant_length？** pant_length 分类器训练时只见过裤子图片，给它看裙子会导致不可预测的输出。类别门控确保每个分类器只在自己领域内工作。

### 4.5 与 3.1.2 的 Stage 6 集成

`GarmentPipelineConfig.run_attribute_inference=True` 将属性推理作为 pipeline 的第 6 个阶段接入端到端流程。属性分类器使用 3.1.2 的 region crop 作为输入。

---

## 五、失败经验与教训

### 5.1 Soft-NMS / Class-aware IoU

**尝试：** 为多实例零件（口袋、纽扣、铆钉）设计低 IoU 阈值 + 高斯衰减 Soft-NMS。

**为什么失败：** 多 prompt 合并后候选框太多，低 IoU 让假阳性也存活。问题不在于"NMS 太激进"，而在于 DINO 产生的候选质量本身就不高。在 score threshold 和形状先验层面过滤远比在 NMS 层面微调有效。

**教训：** 优化 pipeline 中最早出错的那一步，而不是在后续步骤修补。

### 5.2 button prefer_center_x

**尝试：** 假设纽扣在衣服中央（前门襟），设置 prefer_center_x=True。

**为什么失败：** 袖口纽扣、肩部纽扣、侧边纽扣大量被拒绝。

**教训：** 几何先验必须基于对服装结构的具体理解，而非简单假设。先验条件需要与服装类型联动（外套的纽扣 vs 衬衫的纽扣位置不同），但当前实现没有这个粒度。

### 5.3 裤子 hem 与 skirt hem 的混淆

**问题：** "下摆"查询对应 hem，但 hem 在裤子上是裤脚（leg opening），在裙子上是裙摆（skirt hem）。它们的关键点位置和区域裁剪逻辑完全不同。

**解决：** 通过 garment_ref 区分——"裙摆" -> skirt -> hem；"裤腿" -> pants -> leg_opening。但 bare "下摆" 仍无法区分，需依赖 instance 的类别信息做二次判断。

### 5.4 Fashionpedia collar vs neckline 的混淆

Fashionpedia 有独立的 collar (class 1) 和 neckline (class 6)。但 collar = 领子织物（折叠的面料），neckline = 领口开口（皮肤可见的边缘），在视觉上高度重叠。当前的合并评估策略 (neckline/collar/lapel 统一) 是对这个模糊性的工程妥协。

---

## 六、关键技术决策汇总

| # | 决策 | 理由 | 备选方案 | 如果重来 |
|---|---|---|---|---|
| 1 | 内部 13 类 + 对外 5 类双标签 | 关键点和 landmark 依赖细分类别 | 全部改为 5 类 | 仍选此方案 |
| 2 | SAM-HQ 而非原始 SAM | 服装边缘需要高质量 mask | MobileSAM/原始SAM | 应该直接用 MobileSAM |
| 3 | Fashionpedia YOLO 优先于 DINO | 专门训练的检测器精度+速度均优于零样本 | 仅用 DINO | 仍选此方案 |
| 4 | 配置驱动而非硬编码 | 30+零件配置如果硬编码无法维护 | 硬编码在 router 中 | 仍选此方案 |
| 5 | DINO tiny 而非 base | 2026-07-10 A/B 实验: base 核心零件退化 -6.7%, 延迟 +31% | DINO-base | 实验推翻假设，保持 tiny |
| 6 | 本地词汇表 + LLM fallback | 95% 查询在词汇表内，LLM 太慢 | 全部走 LLM 翻译 | 仍选此方案 |
| 7 | ResNet18 而非更大 backbone | 数据量是瓶颈，不是模型容量 | ResNet50/ViT | 仍选此方案 |

---

## 七、数据与模型全景

### 7.1 使用的数据集

| 数据集 | 用途 | 规模 | 标注类型 |
|---|---|---|---|
| DeepFashion2 | 3.1.1 服装检测/分割训练 | 135,975 图片，221,535 实例 | bbox + mask + 13类 + landmarks |
| FashionAI | 3.1.3 属性分类训练 | 8 类属性，每类 556-1,647 样本 | 属性标签 |
| Fashionpedia | 3.1.2 零件检测器训练 | 46K 图片 | bbox + 19 类零件 |

### 7.2 模型清单

| 模型 | 参数 | 用途 | 显存 | 延迟 |
|---|---|---|---|---|
| YOLOv8n | ~3M | 3.1.1 13类检测 | ~200MB | 24ms |
| SAM-HQ vit_b | ~90M | 3.1.1 分割 | ~1.5GB | 293ms |
| ResNet18 关键点 | ~11M | 3.1.1 39关键点 | ~50MB | 54ms |
| Fashionpedia YOLOv8s | ~11M | 3.1.2 19类零件检测 | ~200MB | 10ms |
| Grounding DINO-tiny | ~28M | 3.1.2 开放词汇检测 | ~400MB | 175ms |
| ResNet18 x8 | ~11M each | 3.1.3 属性分类 | ~400MB total | 5-8ms each |

**显存总计：** 约 18GB / 24GB（RTX 3090）

---

## 八、代码组织

```
src/fashion_vision/
├── data/class_mapping.py              # 13<->5 类别映射（配置驱动）
├── models/sam_hq_wrapper.py           # SAM-HQ 封装
├── schemas/instance_schema.py         # 双标签实例 schema
├── attributes/                        # 3.1.3: 属性推理
│   ├── garment_attribute_pipeline.py
│   ├── mask_attribute_pipeline.py
│   └── category_gate.py
├── localization/                      # 3.1.2: 局部定位
│   ├── region_localization_router.py  # [核心] 唯一入口, 743行
│   ├── intent_parser.py              # [核心] 意图解析, 385行
│   ├── fashionpedia_part_detector.py  # FP YOLO 检测器
│   ├── grounding_dino_locator.py     # GDINO + mask-gated + NMS
│   ├── part_detection_config.py      # [核心] 30+零件独立配置, 500行
│   ├── part_shape_priors.py          # 几何先验过滤
│   ├── anatomical_zoom.py            # 解剖学子区域放大
│   ├── spatial_constraint.py         # 左右/上下空间约束
│   ├── bbox_mask_refiner.py          # SAM box->mask 精修
│   ├── garment_ref_filter.py         # 服装类型匹配检查
│   ├── inner_garment_detector.py     # 内搭检测 (SAM-based)
│   ├── inner_boundary_refiner.py     # 内搭边界精修
│   ├── inner_mask_cleaner.py         # 内搭 mask 清洗
│   ├── torso_prior.py                # 躯干先验
│   └── viz_utils.py                  # 6面板 debug 可视化
└── utils/crop_utils.py               # 裁剪工具

configs/
├── category_mapping.yaml              # 13->5 映射
├── attribute_inference.yaml           # 8任务注册
├── attribute_group_mapping.yaml       # 类别->任务->区域映射
└── attribute_eval_targets.yaml        # 评估目标

tests/                                 # 373+ tests
```

---

## 九、当前不足与改进方向

### 高优先级

| 问题 | 根因 | 方案 |
|---|---|---|
| 属性 F1 差 0.12-0.29 | 训练数据不足 (556-1647/任务) | iMaterialist + FashionCLIP |
| SAM 293ms 瓶颈 | ViT-B 计算量大 | -> MobileSAM (~18ms) |
| DINO tiny 小零件天花板 | 零样本模型容量限制 | DINO-base 或微调 |

### 中优先级

| 问题 | 根因 | 方案 |
|---|---|---|
| FP 平衡训练 mAP 0.312 | 增加难例权重导致多数类下降 | 可接受 tradeoff |
| 内搭检测未大规模验证 | 缺少标注数据 | Phase 2 人工标注 |
| 60 QPS 单卡不可行 | SAM 编码器不能跨图 batch | 双卡或接受 50 QPS |

### 低优先级

| 问题 | 根因 | 方案 |
|---|---|---|
| 肩缝/袖缝 DINO 精度低 | DINO 缺乏缝线训练数据 | 标记为 low confidence |
| Qwen-VL 未部署 | 需要 GPU 服务器 | Phase 3 |
| 部分 checkpoint 未跑 multiview | 训练时间限制 | 可补训练 |

---

## 十、答辩要点速查

**为什么用 YOLOv8n 而不是更大的检测器：** 3M 参数，检测不是瓶颈（5.8% runtime），更大模型收益递减。

**为什么保留 13 类而不是全改为 5 类：** 关键点预测依赖精细类别区分短袖/长袖/马甲/吊带，5 类合并后 keypoint 模型无法工作。

**为什么 Mask-gated 用灰色 (128) 填充：** 黑色/白色产生强边缘，DINO 会把假边缘当目标。灰色是 ImageNet 归一化中性值，不触发 backbone 响应。

**为什么 Fashionpedia YOLO 优先于 DINO：** 专门训练的检测器在已知零件上精度+速度双优。早返回设计让 FP 覆盖的 13 类从不触发 DINO 推理。

**为什么 Soft-NMS 效果不好：** 多 prompt 产生大量候选，低 IoU 让假阳性也存活。真正有效的过滤是形状先验（基于服装结构知识）。

**为什么属性分类精度不够：** 数据量（556-1647 样本/任务）是瓶颈，不是模型容量。ResNet18 够了，需要更多标注数据或外部数据引入。

**DINO 文本提示为什么加句号：** HF GDINO 的 processor 以句号作为句子结束信号。不带句号时 score calibration 不稳定。

**为什么 prompt engineering 不是加同义词：** DINO 不是 LLM，不能"理解"同义关系。需要描述视觉特征（iridescent/glittering），而非语义概念（decoration/ornament）。

**为什么不用 DINO-base 替代 DINO-tiny：** 2026-07-10 A/B 实验证明 base 在核心结构零件（collar/neckline/lapel/zipper）上精度退化（-6.7% 合并），小零件天花板未突破（shoes/rivet 持平），延迟增加 31%，显存多占 231MB。Base 改善的 button/epaulette 已被 Fashionpedia YOLO 覆盖。

**neckline/cuff FP 未命中为什么走 fast-path 而非 DINO：** neckline 和 cuff 有可靠的关键点几何规则，DINO 的零样本检测反而不如专门训练的 YOLO+规则稳定。FP 未命中时已经说明"这个零件不存在或太小"，让 DINO 去找只会产生假阳性。

**vest 为什么归 top 而非 outerwear：** PRD 定义。vest 在结构语义上是 base layer（内层），不是 outer layer（外套）。

---

*End of retrospective.*
