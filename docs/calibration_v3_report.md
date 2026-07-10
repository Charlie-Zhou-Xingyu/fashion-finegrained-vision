# 3.1.2 Calibration v3 评估报告

> 时间：2026-06-25  
> 作者：CharlieZhou  
> 数据：100张外套图，标注部件：collar / pocket / zipper / button_cluster  
> 模型：GroundingDINO-tiny（IDEA-Research/grounding-dino-tiny）  
> threshold：0.35（sweep范围 0.20–0.55）

---

## 一、评估条件（v3 相比 v2 的改进）

| 条件 | v2 | v3 |
|---|---|---|
| DINO 运行范围 | 整张 512×512 图 | YOLO garment crop |
| SAM mask gating | 无（garment_mask=None） | 有（SAM-HQ vit_b 生成的服装 mask） |
| shape prior 坐标系 | 全图空间（错误） | garment crop 空间（正确） |
| zipper min_aspect_ratio | 1.8（过严） | 1.2（放宽，允许宽框） |

v3 的评估条件与真实 pipeline 一致，v2 的数字因坐标系错误不可信。

---

## 二、各部件结果分析

### 2.1 拉链（zipper）— 完全失败，根本能力问题

**现象：** Gallery 中几乎全部为 FN，`passed shape: 0`，`best IoU with GT: 0.000`。DINO 在服装图上对"garment zipper"、"front zipper on jacket"等 prompt 基本没有响应。

**根因分析：**
GroundingDINO 的预训练数据中"zipper"对应的视觉样本以**产品特写图**（背包拉链、夹克拉链局部特写）为主，缺乏"拉链作为服装整体一部分"的训练样本。因此在服装全身图中，DINO 无法将"zipper"概念定位到衣服中轴那条细线。

这不是阈值问题，也不是 prompt 问题，是**预训练数据分布与本场景的根本性错配**。

**结论：** GroundingDINO 不适合在服装全身图中检测拉链。需要专用检测器（Fashionpedia 标注 + YOLOv8 fine-tune）。

---

### 2.2 扣子（button）— 无法有效评估，标注覆盖不足

**现象：** Gallery 中大量 NO-GT（GT boxes: 0）。DINO 确实产生了得分在 0.38–0.53 的候选框（橙色），但因为没有对应 GT 无法判断是否 TP。

**根因分析：**
- 标注策略：button_cluster 只在部分有明显排扣的图上标注，非全图覆盖
- DINO 对"button"概念有一定响应，但容易将装饰纽扣、印花、饰物等误检为 button（FP 概率高）

**结论：** 当前标注不足以评估 button 精度。需要：① 统一标注策略（有扣的图全部标，无扣的图明确标记为 negative）；② 扩大标注量后再评估。DINO 对 button 的能力存疑，同样建议专用检测器。

---

### 2.3 口袋（pocket）— 部分有效，FP 过高

**现象：** 有少量 TP（IoU 0.5–0.85），说明 DINO 确实具备一定的 pocket 定位能力。但 FP 数量多：
- 在没有口袋的服装上也会框出"疑似口袋"区域（如腰部分割线、装饰边）
- 在有口袋的图上经常只检测到两个口袋中的一个

**数字（t=0.35）：** Precision 低（大量 FP），Recall 中等（约 0.3–0.4 估计），F1 不达 PRD 目标（≥0.65/0.50）。

**根因：**
- DINO 对 pocket 语义理解比 zipper 强（因为 pocket 在预训练中样本更丰富）
- 但"口袋轮廓"和"服装接缝/拼接线"在视觉上相似，导致 FP
- 当前 shape prior（`max_area_ratio: 0.25`）过滤了部分大框误检，但细小误检仍漏过
- **隐蔽性问题（重要）：** 服装口袋常与主体面料同色同纹，缺乏明显的颜色或纹理边界。Gallery 中可见大量案例：口袋仅靠缝线或极浅的压线与衣身区分，人眼在缩略图下也难以辨认。GroundingDINO 依赖视觉-语义对齐，对这类"低对比度结构性部件"的定位能力天然较弱——模型看到的是一片均匀面料，语言 prompt 无法引导它找到肉眼都不易察觉的边界。这一问题在纯色外套、羽绒服等品类上尤为突出，也解释了为何 FN 率居高不下、即便降低阈值召回率也提升有限。

**结论：** DINO 对 pocket 的能力处于"勉强可用"边界。Precision 需要通过更严格的 shape prior 或专用检测器来提升。近期可以用 Fashionpedia 的 pocket 标注 fine-tune YOLO，作为 dedicated path 接入 router。

---

### 2.4 领子（collar）— 评估方法有偏差，结论待定

**现象：** Gallery 中 collar 的 DINO 结果不理想，但存在评估方法问题。

**评估偏差：** calibration 脚本绕过了 `region_localization_router.py`，直接用 DINO 对 collar 进行检测。但在真实 pipeline 中，collar 走的是 **fast path（landmark + 几何规则）**，不经过 DINO。calibration v3 量的是"DINO检collar"的性能，不是系统实际使用的路径的性能。

**真实情况：** collar 的 landmark 几何路径在 P1 阶段的 Batch60 测试中有 92% 有效响应率。

**结论：** collar 的 calibration v3 数字参考意义有限。如需评估 collar 定位精度，应单独测试 fast path（landmark 路径），与 DINO 无关。

---

## 三、总结对比表

| 部件 | DINO 能力 | 根本原因 | 建议路径 |
|---|---|---|---|
| zipper | ❌ 完全失败 | 预训练无服装上下文拉链样本 | 专用 YOLOv8（Fashionpedia 标注） |
| button | ❓ 无法评估 | 标注覆盖不足 | 先补标注，再训练专用检测器 |
| pocket | ⚠️ 部分有效 | FP 过高，precision 不达标 | 短期调 shape prior；中期专用检测器 |
| collar | — 评估无效 | calibration 用错路径（DINO vs landmark） | 单独测 landmark fast path |

---

## 四、架构层面结论

当前问题的核心不是参数调整，而是**用开放词汇模型（GroundingDINO）承担了它不擅长的封闭集小目标检测任务**。

建议与带教老师讨论的方向：

1. **短期（1-2周）**
   - collar：评估 landmark fast path 精度，不依赖 DINO
   - pocket：收紧 shape prior 降低 FP，接受当前精度作为 baseline
   - zipper/button：明确不再依赖 DINO，转为专用检测器立项

2. **中期（1个月）**
   - Fashionpedia（19个garment parts）中**有** zipper、pocket 的 segmentation 标注，可直接用于专用检测器训练
   - Fashionpedia **没有** button 类别（只有 buckle），button 检测器需另找数据源（候选：自行标注少量样本、ModaNet，或降级为不做 button 检测）
   - 将 Fashionpedia zipper / pocket 标注转换为 YOLO 格式，训练 YOLOv8n 专用检测器
   - 接入 router 作为 dedicated path（router 中 `_DEDICATED_PARTS` 已预留接口）
   - GroundingDINO 保留为长尾开放描述的 fallback（ruffle / fringe / bow 等，这些在 Fashionpedia 里有标注但数量少，DINO 反而更合适）

3. **待确认事项**
   - button 标注策略：有扣的图全部标？还是只标扣子可见且清晰的图？
   - zipper 标注：开合状态的拉链是否统一标为一个整体外接矩形？
   - 是否申请 Fashionpedia 数据集访问权限（公开数据集，可直接下载）

---

## 五、下一步行动建议

| 优先级 | 行动 | 负责 | 前置条件 |
|---|---|---|---|
| P0 | 与带教老师确认中期方向：是否启动专用检测器 | CharlieZhou | 本报告 |
| P0 | 单独评估 collar landmark fast path 精度 | CharlieZhou | — |
| P1 | 补充 button 标注（统一策略，覆盖全量） | CharlieZhou | 带教确认 |
| P1 | 调研 Fashionpedia 数据集可用性 | CharlieZhou | — |
| P2 | 训练 YOLOv8 服装部件专用检测器 | CharlieZhou | 数据准备完成 |
