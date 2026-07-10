# 3.1.2 Phase 2 工程审视报告

> 撰写时间：2026-06-24  
> 作者：CharlieZhou + Claude  
> 状态：Phase 2 进行中，本文为阶段性审视

---

## 一、整体项目视角

### PRD三大模块现状速览

| 模块 | 描述 | 当前状态 | 核心问题 |
|---|---|---|---|
| **3.1.1** 服装实例分割 | YOLO 13类检测 + SAM-HQ分割 | ✅ P4完成，模型已晋升 | SAM-HQ延迟420ms，PRD目标≤50ms |
| **3.1.2** 语言引导局部定位 | 地标快速路径 + DINO开放词汇 | 🔄 Phase 2进行中 | DINO-tiny能力不足，阈值未校准 |
| **3.1.3** 细粒度属性提取 | FashionAI ResNet18分类器 | ⚠️ 基础架构完成，精度gap未解决 | 训练数据太少（556-1647样本/任务） |

**结论：三个模块都有未解决的核心瓶颈，3.1.2是当前工作重心。**

---

## 二、3.1.2 实现历程工程化回顾

### 2.1 架构演进

```
初始状态（P1原型）
  ├── 6个固定区域 + 规则NLP
  ├── 92% Batch60命中率
  └── 无法处理: 拉链/口袋/扣子/门襟等小部件

Phase 0-4（开放词汇升级，2026-06-23完成）
  ├── intent_parser.py — 中文→结构化意图，18个部件，最长匹配
  ├── grounding_dino_locator.py — DINO via HuggingFace，multi-prompt+NMS
  ├── spatial_constraint.py — 左/右侧筛选
  └── region_localization_router.py — 统一入口，fast-path/open-vocab路由

Phase 1（工业化确定性修复，2026-06-24完成）
  ├── 1.1 mask-gated DINO — _crop_image_and_mask()，同一crop box
  ├── 1.2 shape priors返回[]而非fallback候选
  ├── 1.3 状态词汇统一（not_detected/success/error等）
  ├── 1.4 长尾中文词汇（肩缝/袖缝/荷叶边/流苏/绑带/抽绳）
  ├── 1.5 garment_ref不匹配标记
  └── 1.6 6面板debug可视化脚本

Phase 2（阈值标定，进行中）
  ├── 标注：100张外套图 via Label Studio ✅
  ├── Calibration v1（60张，无shape priors）✅ 结果极差
  ├── Calibration v2（100张，加shape priors）✅ 结果仍差
  └── 可视化分析 ✅ 发现若干工程bug
```

### 2.2 当前精度数字（Calibration v2）

| Part | 最佳P | 最佳R | 最佳阈值 | 评级 |
|---|---|---|---|---|
| collar | 0.444 | 0.117 | 0.50 | ⚠️ 勉强（但走fast-path，DINO是fallback） |
| pocket | 0.073 | 0.152 | 0.50 | ❌ 不可用 |
| button | 0.118 | 0.400 | 0.40 | ❌ 不可用 |
| zipper | 0.066 | 0.400 | 0.25 | ❌ 不可用 |
| placket | 0.000 | 0.000 | — | ❌ 完全失败 |

**PRD目标：Precision ≥ 0.65，Recall ≥ 0.50。所有小部件均未达标。**

---

## 三、已发现的局限和漏洞

### 3.1 DINO模型能力问题（根本瓶颈）

**现象：** 可视化显示DINO-tiny对拉链、扣子的检测结果是整个服装框，不是实际部件。

**根因：** DINO-tiny（48M参数）的预训练数据中服装小部件样本不足，无法在零样本场景下定位1-3cm的拉链齿/扣子。

**影响：** 即使shape priors、mask gating全部正确，召回率也接近0。

**对策：** 按Phase 2.4计划，测试DINO-base（150M参数）。如果仍不够，触发Phase 6 fine-tuning。

---

### 3.2 多实例检测缺陷（设计漏洞）

**现象：** 一件衣服有两个口袋，系统只返回置信度最高的一个框。

**根因：** router和可视化脚本都只取`kept[0]`（第一个/最高分的检测结果）。

**代码位置：**
- `region_localization_router.py` — 返回单个result
- `visualize_localization_debug.py` — Panel 6只画`kept[0]`

**影响范围：** pocket（常见双口袋）、button_cluster（多排扣）。

**修复方向：** 返回所有通过NMS+shape prior的框列表，让调用方决定。需要API变更。

---

### 3.3 Calibration脚本的mask未接入（评估失真）

**现象：** `calibrate_part_thresholds.py`第173行调用`detect_multi_prompt(..., garment_mask=None)`。

**根因：** calibration脚本是独立写的，没有复用router的mask加载逻辑，而且calibration跑时也没有SAM-HQ masks可用。

**影响：** calibration量的是无mask的DINO性能，与真实pipeline（有mask gating）的性能有差距。FP数字被高估。

**现状：** Phase 1.1修复了router，但calibration脚本仍未修复，且缺乏SAM-HQ mask输入路径。

---

### 3.4 garment_bbox=None导致shape priors部分失效

**现象：** 可视化时未传`--garment-bbox`，面积比检查（max_area_ratio）被跳过，大框无法被过滤。

**根因：** `filter_by_shape_priors`在`garment_area is None`时跳过所有需要garment参照的检查。

**已修复：** `visualize_localization_debug.py`已改为默认使用图片尺寸`[0,0,w,h]`作为garment_bbox。

**仍需修复：** calibration脚本在无garment_detections时同样存在此问题（虽然用了`[0,0,crop_w,crop_h]`，但crop=全图时效果相同）。

---

### 3.5 Calibration坐标空间错误

**现象：** calibration v2跑时没有使用`--garment-detections`，DINO在512×512全图上跑，shape priors的garment_bbox是`[0,0,512,512]`。

**影响：** shape priors本来是为"服装占满crop"的坐标系设计的。在全图空间里，zipper的`min_aspect_ratio_h_over_w: 1.8`会拒绝很多实际正确但bbox偏宽的检测框，导致recall被低估。

**对策：** 需要先对100张coat图跑YOLO得到garment detections，再加`--garment-detections`重跑calibration v3。

---

### 3.6 标注质量问题

| 问题 | 影响 | 处理方式 |
|---|---|---|
| button一次标单颗，一次标整排 | button的recall评估不一致 | 统一用button_cluster（整排），旧标注丢弃 |
| drawstring用松紧带标注 | drawstring评估无效 | 已排除出v2评估 |
| 开合拉链标注不一致（1条vs2条框） | zipper recall被低估 | 统一标整条拉链外接矩形 |
| coat_annotation_batch1残留2个placket | calibration出现无效的placket指标 | 下次导出前在Label Studio删除 |

---

### 3.7 Shape Prior参数未经实证校准

**现象：** `PART_DETECTION_CONFIG`里的所有阈值（box_threshold、max_area_ratio、min_aspect_ratio等）都是人工估计值。

**已知问题：**
- zipper的`min_aspect_ratio_h_over_w: 1.8`在全图空间过严，导致真实拉链框被reject
- placket的`min_aspect_ratio_h_over_w: 1.5`导致calibration v2中placket完全失败（0 TP）
- button的面积约束（8%）在全图空间因garment_bbox=None而未生效

**对策：** 需要在正确坐标系（YOLO garment crop）下重新跑calibration，再根据结果调整。

---

### 3.8 中文显示bug（次要）

**现象：** debug可视化Panel标题中的中文字符显示为"??????"。

**根因：** OpenCV的`cv2.putText`不支持中文字体。

**影响：** 仅为显示问题，不影响实际检测（DINO接收的是英文prompts）。

**修复方向：** 使用PIL绘制文字再转OpenCV格式，或用拼音/英文替代panel标题。

---

## 四、Phase 2当前状态与剩余工作

### 已完成
- [x] D6：100张外套图标注（pocket/zipper/button_cluster/collar）
- [x] Calibration v1（旧数据，无shape priors，结论：不可用）
- [x] Calibration v2（新数据，有shape priors，结论：DINO-tiny不足）
- [x] 可视化分析，发现garment_bbox/mask/多实例等工程bug

### 待完成
- [ ] **D7决策：** 基于calibration v2数字，确认是否进入DINO-base对比
- [ ] **修复标注质量问题：** 删除2个placket，统一拉链标注方式
- [ ] **Calibration v3：** 先跑YOLO得到garment detections，再重跑calibration
- [ ] **D10：** 跑DINO-base对比，决定是否换模型或进入Phase 6
- [ ] **多实例检测修复：** router返回全部kept框

---

## 五、3.1.2各Phase评价

| Phase | 评价 | 遗留问题 |
|---|---|---|
| Phase 0-4（开放词汇升级） | ✅ 架构完整，代码质量可，快速路径无回归 | left/right convention未验证 |
| Phase 1（工业化修复） | ✅ 6项修复全部落地，373测试通过 | calibration脚本未同步mask修复 |
| Phase 2（阈值标定） | 🔄 数据已准备，但发现DINO-tiny根本能力不足 | 需先修坐标系问题再重跑 |
| Phase 3（Qwen翻译服务） | ⏸ 未启动，需GPU服务器 | 依赖Phase 2结论决定优先级 |
| Phase 4（服装层次结构） | ⏸ 未启动，依赖SAM-HQ mask质量验证 | — |
| Phase 5（复合锚点查询） | ⏸ 未启动 | — |
| Phase 6（DINO微调） | ⏸ 未启动，需Phase 2结论触发 | 标注数据已积累（100张coat） |

---

## 六、最重要的结论

1. **DINO-tiny在无fine-tuning情况下无法可靠检测拉链/扣子/口袋。** 这是当前3.1.2开放词汇路径的根本瓶颈。

2. **Shape priors在工程上是有效的**，但需要正确的garment_bbox参照才能发挥面积过滤功能。

3. **Mask gating代码已实现，但在calibration和独立可视化中均未接入SAM-HQ mask。** 这意味着我们从未真正测试过mask gating对精度的提升效果。

4. **标注数据质量存在多处不一致，影响calibration数字的可信度。** 需要在重跑前修正。

5. **下一个关键决策点：DINO-base vs fine-tuning。** 这个决策需要calibration v3（正确坐标系）的数字才能做出。
