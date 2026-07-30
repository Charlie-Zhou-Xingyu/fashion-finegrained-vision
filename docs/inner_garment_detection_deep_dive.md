# PRD 3.1.2 内搭（Inner Garment）检测子系统 — 实现深度解析

> 生成时间：2026-07-30  
> 对应代码：`src/fashion_vision/localization/inner_garment_detector.py` 及配套模块

---

## 一、系统定位与触发条件

### 1.1 问题场景

当用户查询指向"内搭"（如"外套里面的衣服是什么领型"）且 YOLO 检测到的实例为 **outerwear（外套）** 时，由于外套的遮挡，YOLO 通常无法直接检测到被外套覆盖的内部衣物。内搭子系统通过 **SAM 多掩码 + 几何先验** 在外套边界框内搜索内层衣物的可见轮廓。

### 1.2 触发条件

在 `region_localization_router.py:97–140` 中，内搭检测的触发条件为 **三个条件同时满足**：

| 条件 | 来源 | 说明 |
|---|---|---|
| `garment_ref == "inner"` | `intent_parser.py` 解析用户查询 | "内搭"、"里面" 等关键词映射到 `"inner"` |
| `coarse_class_name == "outerwear"` | YOLO 13 类检测 → 5 类映射 | 只有外套实例才会触发内搭检测 |
| `sam_wrapper is not None` | 管线传入 | SAM-HQ 必须已加载 |

触发后，如果内搭检测成功，会用内搭的 mask/bbox **替换** 外套实例的对应字段，后续 DINO 部件定位将基于内搭区域进行。

### 1.3 优雅降级

如果内搭检测失败（找不到合格候选），管线**不回退到整个外套**，而是在结果中标记 `garment_ref_matched=False`，下游可以据此判断查询未匹配到对应衣物类型。

---

## 二、模块架构

内搭子系统由以下 **4 个模块** 组成，形成一条完整的处理流水线：

```
inner_garment_detector.py  ←── 主控：领口补集分析 + SAM 全框回退
    │
    ├── torso_prior.py              ←── 躯干先验遮罩（代理 / 关键点）
    ├── inner_boundary_refiner.py   ←── 边界精修（边缘 + 颜色 + 纹理剖面）
    └── inner_mask_cleaner.py       ←── 伪影清理（上角残留 + 侧条 + 孤立分量）
```

所有模块均位于 `src/fashion_vision/localization/` 下。

---

## 三、核心策略：领口补集几何分析（PRIMARY）

这是内搭检测的**主路径**，不依赖任何训练数据，完全基于服装的几何和视觉先验。

### 3.1 总体流程

```
输入: 全图 BGR, 外套 bbox, 外套 mask, SAM-HQ wrapper
     │
 [1] 构建领口 ROI               ←── 外套 bbox 上部的中心区域 (18%-82% x, 3%-58% y)
     │
 [2] 构建补集搜索遮罩           ←── 外套 bbox ∩ 领口 ROI ∩ ¬外套 mask
     │
 [3] 形态学清理                 ←── CLOSE + OPEN (3×3 椭圆核)
     │
 [4] 三源候选生成               ←── CC分析 | Canny边缘 | SAM颈区多掩码
     │
 [5] 统一评分 + 过滤            ←── 13 维加权评分 + 多层阈值拒绝
     │
 [6] 种子向下扩展               ←── 前开区域连通分量合并
     │
 [7] 边界精修                   ←── 水平边缘剖面 + 垂直颜色纹理扫描
     │
 [8] 伪影清理                   ←── 软开襟走廊 + 上角抑制 + 侧条移除
     │
输出: 内搭 mask + bbox（伪实例 dict）
```

### 3.2 步骤详解

#### [1] 领口 ROI 构建 (`_construct_neckline_roi`)

在外套 bbox 内划定领口可能出现的区域：

```python
NECKLINE_RULES = {
    "roi_x_lo": 0.18,   # 左边界在外套宽的 18% 处
    "roi_x_hi": 0.82,   # 右边界在外套宽的 82% 处
    "roi_y_lo": 0.03,   # 上边界在外套高的 3% 处
    "roi_y_hi": 0.58,   # 下边界在外套高的 58% 处
}
```

这个区域覆盖了从领口到胸口的垂直范围，水平方向排除了外套边缘（通常是袖子区域）。

#### [2] 补集搜索遮罩 (`_build_complement_search_mask`)

关键思想：**内搭 = 外套 bbox 内、领口 ROI 内、但不属于外套 mask 的像素**。

```
search_mask = bbox_mask ∩ neckline_mask ∩ ¬outer_mask
```

外套 mask 通常在前开区域有空洞（露出内搭），这些空洞就是候选区域。

#### [3] 形态学清理

对搜索遮罩进行 CLOSE → OPEN 处理，弥合小裂缝同时去除噪点。

#### [4] 三源候选生成

每个候选是一个 `{mask, bbox_xyxy, area, centroid, source}` 字典，来自三种互补来源：

**4a. 连通分量分析 (`_extract_cc_candidates`)**
- 在清理后的搜索遮罩上运行 `cv2.connectedComponentsWithStats`
- 过滤掉面积 < 20px 的噪声分量
- 最可靠的来源，因为搜索遮罩本身已经是"非常可能的内搭区域"

**4b. Canny 边缘轮廓 (`_extract_canny_candidates`)**
- 在领口 ROI 内运行 Canny 边缘检测（阈值 40/120）
- 形态学 CLOSE (5×5 椭圆核) 弥合小间隙
- 提取闭合轮廓作为候选
- 适用于内搭和外衣颜色相近、补集分析漏检的情况

**4c. SAM 领口多掩码 (`_extract_sam_candidates`)**
- 对领口 ROI 图像调用 `sam_wrapper.predict_all_masks()`
- SAM 可能从领口区域分割出内搭的可见部分
- 这是最重但最灵活的来源，依赖 SAM 的分割质量

#### [5] 统一评分 (`_score_candidate`)

每个候选通过 **13 维加权评分函数** 进行评估。评分公式如下：

```
score = w_inside_bbox     * inside_bbox_ratio      (1.0)
      + w_outside_outer   * outside_outer_ratio    (1.5)
      + w_neckline_overlap * neckline_overlap      (1.5)
      + w_center_score    * center_score           (1.8)
      + w_opening_core    * opening_core_overlap   (2.0)  ← 最高权重
      + w_upper_position  * upper_position_score   (1.0)
      + w_solidity        * solidity               (1.0)
      + w_area_ratio      * (1 - |area_ratio - 0.08|)  (0.5)
      - penalty_side_edge * side_edge_penalty      (2.0)
      + w_torso_overlap   * torso_overlap          (2.0)  ← 最高权重
```

**各指标含义：**

| 指标 | 含义 | 期望 |
|---|---|---|
| `inside_bbox_ratio` | 候选像素在外套 bbox 内的比例 | ≥ 0.75 |
| `outside_outer_ratio` | 候选像素在外套 mask **之外** 的比例 | ≥ 0.45（内搭不应该在外套 mask 内） |
| `neckline_overlap` | 候选像素在领口 ROI 内的比例 | ≥ 0.60 |
| `opening_core_overlap` | 候选像素在前开核心区的比例 | ≥ 0.35 |
| `center_score` | 候选质心与外套中线的距离（归一化） | 越靠近中心越高 |
| `upper_position_score` | 候选质心在外套上部的程度 | 越靠上越高（内搭领口在上部） |
| `area_ratio_bbox` | 候选面积 / 外套 bbox 面积 | 0.006–0.25 |
| `solidity` | 候选轮廓面积 / 凸包面积 | 越高越紧凑 |
| `side_edge_penalty` | 候选是否触碰外套左右边缘 | 触碰 = 扣 2.0 分 |
| `torso_overlap` | 候选与躯干先验的重叠比例 | ≥ 0.35（关键点躯干）或 ≥ 0.25（代理躯干） |

**多层阈值拒绝机制：**

候选必须通过以下所有硬阈值检查才能被标记为 `passed`：

| 检查项 | 阈值 | 含义 |
|---|---|---|
| `inside_bbox_ratio` | ≥ 0.75 | 必须在 bbox 内 |
| `outside_outer_ratio` | ≥ 0.45 | 必须在外套 mask 外 |
| `neckline_overlap` | ≥ 0.60 | 必须在领口 ROI 内 |
| `area_ratio_bbox` | 0.006–0.25 | 面积不能太小或太大 |
| `rel_cx` | 0.30–0.70 | 质心必须在中心 40% 宽度内 |
| `opening_core_overlap` | ≥ 0.35 | 必须与前开核心区有足够重叠 |
| `bbox_w_ratio` | ≥ 0.08 | bbox 宽度不能太窄 |
| `bbox_h_ratio` | ≥ 0.10 | bbox 高度不能太矮 |
| `score` | ≥ 3.0 | 加权总分必须达标 |
| `torso_overlap` (可选) | ≥ 0.35/0.25 | 必须在躯干区域内 |

所有未通过的候选都会记录详细的拒绝原因（如 `off_center rel_cx=0.892 not in [0.30,0.70]`），便于调试。

#### [6] 种子向下扩展 (`_extend_inner_mask_downward`)

领口区域检测到的内搭通常只是上衣领口的一小段（种子遮罩）。这个步骤将种子沿前开区域向下扩展，恢复内搭的完整可见轮廓。

**扩展逻辑：**
1. 在前开 ROI（外套宽的 22%–78%，高的 8%–85%）内计算补集
2. 与躯干先验取交集，排除外套边缘区域
3. 提取连通分量
4. 满足以下条件的分量被合并到种子：
   - x 方向与种子 bbox 重叠 ≥ 20%
   - 垂直间距 ≤ 外套高的 10%（即紧邻种子下方）
   - 质心在外套中心 22%–78% 范围内
   - 面积 ≥ 外套 bbox 面积的 0.2%

#### [7] 边界精修 (`inner_boundary_refiner.py`)

对扩展后的内搭 mask/bbox 进行水平和垂直两个方向的独立精修。

**7a. 水平精修 — 复合剖面峰值搜索**

在内搭 bbox 两侧 ±10% 外套宽的搜索带内：

1. 提取三种剖面（沿 y 轴求和并归一化）：
   - **边缘剖面** (`_edge_profile_x`)：Canny 边缘检测（阈值 30/100）
   - **颜色梯度剖面** (`_color_grad_profile_x`)：Lab a*/b* 通道的 x 方向梯度
   - **纹理剖面** (`_texture_profile_x`)：Laplacian 响应的绝对值

2. 加权融合为复合剖面：
   ```
   composite = 0.40 × edge + 0.35 × color_grad + 0.25 × texture
   ```

3. 3 像素滑动平均平滑

4. 从中线向左搜索最强峰值 → 左边界；向右搜索第一个峰值 → 右边界

5. 约束边界不超出前开 ROI 和外套 bbox

**7b. 垂直精修 — 逐行颜色纹理扫描**

从内搭 bbox 下边界开始，向下逐行（步长 2px）扫描：

1. 每行提取前开 ROI 内、非外套 mask 的像素
2. 计算该行 Lab 中位数与种子颜色的 CIE76 ΔE 色差
3. 计算该行 Laplacian 响应的标准差（纹理变化）
4. 如果色差 > 35.0 或纹理变化 > 2.0 → 计入变化条纹计数
5. 连续 5 行变化 → 停止扩展（遇到颜色/纹理的显著变化，即内搭-外衣边界）
6. 最大向下扩展不超过外套高的 15%

**安全门 (`_check_boundary_refine_safety`)：**

精修结果必须通过安全检查，否则回退到精修前的原始结果：

| 检查项 | 范围 | 说明 |
|---|---|---|
| `area_ratio` | 0.45–2.80 | 精修后面积 / 精修前面积 |
| `bbox_area_ratio` | 0.45–3.00 | 精修后 bbox 面积 / 精修前 bbox 面积 |
| `center_shift_ratio` | ≤ 0.18 | 质心位移 / 外套宽度 |
| `torso_overlap` (可选) | ≥ 精修前 - 0.20 | 躯干重叠不能大幅下降 |

#### [8] 伪影清理 (`inner_mask_cleaner.py`)

边界精修后，mask 可能包含矩形 ROI 裁剪残留的直角伪影。清理流水线分 6 步：

**8a. 输入归一化 (`_bin`)**
- 将任意格式的 mask 转为 uint8 0/255

**8b. 软开襟走廊 (`_build_soft_opening_corridor`)**

用**梯形**替代矩形 ROI：

```
       顶部窄 (34% 外套宽)
       ╱              ╲
      ╱                ╲
     ╱                  ╲
    底部宽 (58% 外套宽)
```

- 梯形顶部对应领口（窄），底部对应胸口（宽）
- 自然裁剪掉矩形 ROI 的上角直角残留
- 与 `opening_roi` 和 `torso_mask` 取交集进一步约束

**8c. 上角色彩一致性抑制 (`_suppress_upper_corners`)**

针对上角直角残留（矩形 ROI 裁剪的典型伪影）：

1. 定义上角区域：内搭 bbox 上部 30% × 左右各 25%
2. 提取种子 Lab 颜色中位数（来自 seed_mask 或内搭 mask 中心 40% 区域）
3. 对每个上角像素，以下条件**全部不满足**则移除：
   - 与主连通分量 8-邻接连通
   - 与 seed_mask 8-邻接连通
   - 与种子颜色的 CIE76 ΔE ≤ 38.0

**8d. 侧条移除 (`_remove_side_strips`)**

检测并移除三种侧边伪影：

| 规则 | 条件 | 说明 |
|---|---|---|
| 极小分量 | 面积 < 内搭面积的 3% | 噪声级别 |
| 高窄侧条 | 宽度 < 内搭宽的 12% ∧ 高度 > 内搭高的 35% ∧ rel_cx 在边缘 25% | ROI 裁剪边缘残留 |
| 低躯干重叠侧分量 | 在左右 15% 边缘 ∧ 躯干重叠 < 15% | 外套边缘误检 |

**8e. 主分量保留 + 辅助分量智能保留 (`_preserve_main_and_aux`)**

1. 最大连通分量 = 主体（无条件保留）
2. 辅助分量保留条件（满足任一即可）：
   - 与主体的 x 方向重叠 > 35%
   - 与 seed_mask 有重叠像素（≥ 5px）
   - 面积 > 内搭面积的 10% ∧ 质心在中心 50%（rel_cx ∈ [0.25, 0.75]）
3. 不满足条件的孤立分量被移除

**8f. 形态学平滑 (`_morphology_smooth`)**
- OPEN (3×3 椭圆核)：去除小噪点
- CLOSE (3×3 椭圆核)：填充小孔洞

**8g. 面积比安全门**

```
area_ratio = cleaned_area / original_area
if area_ratio < 0.45 → 拒绝清理，回退到清理前的 mask
```

防止过度清理导致内搭主体丢失。

---

## 四、躯干先验 (`torso_prior.py`)

躯干先验用于过滤掉非躯干区域的候选（如外套边缘、背景等）。

### 4.1 代理躯干 (`build_proxy_torso_prior`)

当姿态关键点不可用时，从外套 bbox 构建代理躯干：

```
tx1 = 外套 x1 + 外套宽 × 0.18
tx2 = 外套 x1 + 外套宽 × 0.82
ty1 = 外套 y1 + 外套高 × 0.03
ty2 = 外套 y1 + 外套高 × 0.88
```

即外套 bbox 水平缩小 18%–82%（排除袖子区域），垂直保持 3%–88%。

### 4.2 关键点躯干 (`build_torso_prior_from_keypoints`)

当有 `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip` 四个关键点时：

1. 用四个关键点构建凸包
2. 9×9 椭圆核膨胀 × 2，扩展至衣物范围
3. 如果关键点缺失或无效 → 回退到代理躯干

### 4.3 在评分中的使用

- 代理躯干（source="proxy"）：`torso_min_overlap = 0.25`（较宽松，因为代理躯干精度有限）
- 关键点躯干（source="keypoints"）：`torso_min_overlap = 0.35`（较严格）

---

## 五、回退策略：SAM 全框多掩码（FALLBACK）

当领口补集分支没有找到合格候选时，启动回退策略 (`_detect_fallback_full_bbox`)：

### 5.1 流程

1. 在外套 bbox 内缩 4px（排除边缘像素）
2. 对裁剪区域调用 `sam_wrapper.predict_all_masks()`
3. 对每个 SAM 候选进行过滤：

| 检查项 | 阈值 | 说明 |
|---|---|---|
| `containment` | ≥ 0.80 | 候选在外套 mask 内的比例（应为内层衣物） |
| `area_ratio` | 0.01–0.50 | 候选面积 / 外套面积 |
| `solidity` | ≥ 0.65 | 凸包比 |
| `touches_edge` | = False | 不触碰裁剪区域边缘 |
| `is_whole_garment` | = False | 不是整个外套（面积 < 外套的 85%） |

4. 选得分最高的候选

### 5.2 与主路径的关系

- 主路径（领口补集）优先，因为它利用了服装的几何先验
- 回退策略是通用方案，但精确度不如主路径（可能检测到非内搭的 SAM 前景区域）
- 日志会明确标记使用的是哪种策略

---

## 六、管线集成流程图

```
用户查询 "外套的内搭是什么领型"
         │
         ▼
   intent_parser.py
   parse_intent("外套的内搭是什么领型")
   → garment_ref = "inner", body_part = "collar"
         │
         ▼
   garment_ref_filter.py
   filter_instances_by_garment_ref("inner")
   → 按 mask_area 升序排列（内搭通常面积最小）
         │
         ▼
   region_localization_router.py  ←── 关键入口
   locate_region(image, query, instance, sam_wrapper)
         │
         ├── garment_ref == "inner"?
         │   └── YES
         │       └── coarse_class_name == "outerwear"?
         │           └── YES
         │               └── sam_wrapper is not None?
         │                   └── YES → 触发内搭检测
         │                       │
         │                       ▼
         │               detect_inner_garment_from_sam()
         │                       │
         │                       ├── PRIMARY: detect_inner_by_neckline_rules()
         │                       │   ├── 领口 ROI + 补集搜索遮罩
         │                       │   ├── 三源候选生成
         │                       │   ├── 13 维加权评分
         │                       │   ├── 向下扩展
         │                       │   ├── 边界精修 (inner_boundary_refiner)
         │                       │   └── 伪影清理 (inner_mask_cleaner)
         │                       │       │
         │                       │       ├── SUCCESS → 返回内搭 mask + bbox
         │                       │       └── FAIL
         │                       │           │
         │                       │           └── FALLBACK: _detect_fallback_full_bbox()
         │                       │               └── SAM 全 bbox 多掩码
         │                       │
         │                       ├── SUCCESS → 用内搭 mask/bbox 替换外套的
         │                       │              instance._inner_garment_detected = True
         │                       │              instance._garment_ref_mismatch = False
         │                       │
         │                       └── FAIL → garment_ref_matched = False
         │                                   继续使用外套实例（降级）
         │
         ▼
   DINO part localization 基于内搭（或外套）mask 进行部件定位
```

---

## 七、关键设计决策

### 7.1 为什么不用 YOLO 直接检测内搭？

YOLO 在 DeepFashion2 上训练的 13 类检测器没有"内搭"类别。外套遮挡下，内部衣物的大部分不可见，检测器训练信号不足。SAM 基于视觉提示的分割能力更适合从外套开襟区域"挖出"内搭的可见部分。

### 7.2 为什么用几何先验而非训练？

- 服装的前开结构是普适的几何规律，不需要标注数据
- 领口区域在外套上部中心，这个位置几乎不变
- 外层衣物 mask 由 SAM-HQ 提供，质量可靠，补集分析有效

### 7.3 为什么不使用皮肤检测？

早期版本曾使用皮肤检测来辅助定位领口，但存在问题：
- 不同肤色需要不同的肤色模型
- 高领/围巾等会遮挡皮肤
- 引入皮肤检测增加了复杂度但未显著提升准确率
- 当前版本已完全移除皮肤检测逻辑

### 7.4 为什么需要多层安全门？

每一步精修（边界精修、伪影清理）都可能引入新的错误。安全门确保精修结果不会比原始结果更差：
- 边界精修安全门：检查面积比、质心位移
- 伪影清理安全门：检查面积比 ≥ 45%
- SAM 精修安全门：检查 IoU、面积比、质心位移

不合格的精修会被自动回退，确保系统的保守性。

---

## 八、配置与可调参数

所有阈值和权重都集中在 `NECKLINE_RULES`、`OPENING_EXTENSION_RULES`、`FALLBACK_RULES`、`DEFAULT_REFINE_RULES` 和 `DEFAULT_CLEANUP_RULES` 字典中，便于校准：

| 配置组 | 位置 | 参数数量 | 用途 |
|---|---|---|---|
| `NECKLINE_RULES` | `inner_garment_detector.py` | ~25 | 领口 ROI、评分权重、候选阈值 |
| `OPENING_EXTENSION_RULES` | `inner_garment_detector.py` | ~7 | 向下扩展的 ROI 和过滤参数 |
| `FALLBACK_RULES` | `inner_garment_detector.py` | ~5 | SAM 全框回退的过滤参数 |
| `DEFAULT_REFINE_RULES` | `inner_boundary_refiner.py` | ~12 | 水平/垂直精修参数 |
| `DEFAULT_CLEANUP_RULES` | `inner_mask_cleaner.py` | ~15 | 伪影清理的多级参数 |

---

## 九、调试可视化

`viz_utils.py` 中的 `draw_inner_garment_debug()` 函数提供 7 层调试可视化：

| 图层 | 颜色 | 内容 |
|---|---|---|
| 躯干遮罩 | 品红（半透明） | 躯干先验区域 |
| 躯干 bbox | 品红（实线） | 躯干边界 |
| 前开 ROI | 紫色（实线） | 前开区域边界 |
| 精修前 bbox | 橙色（虚线） | 边界精修前的内搭 bbox |
| 内搭 mask | 青色（半透明） | 内搭分割结果 |
| 精修后 bbox | 蓝色（实线） | 边界精修后的内搭 bbox |
| 清理前 bbox | 绿色（虚线） | 伪影清理前的内搭 bbox |
| 评分字幕 | 白色文本 | 候选分数、来源、各项指标 |

---

## 十、测试覆盖

`tests/test_inner_garment_detector.py`（862 行）覆盖以下测试类：

| 测试类 | 覆盖内容 |
|---|---|
| `TestNecklineRules` | 阈值存在性、权重正值、无皮肤检测残留 |
| `TestOpeningCore` | 核心区在 bbox 内 |
| `TestComplementMask` | 补集搜索遮罩正确检测空洞 |
| `TestScoring` | 候选评分通过/拒绝、离心拒绝、面积拒绝 |
| `TestOpeningROI` | 前开 ROI 构建正确使用配置 |
| `TestExtension` | 向下扩展合并下方分量、拒绝离心分量 |
| `TestTorsoPrior` | 代理躯干、关键点躯干、回退逻辑 |
| `TestTorsoScoring` | 躯干重叠计算、低重叠拒绝 |
| `TestBoundaryRefiner` | 精修边界约束、调试剖面输出 |
| `TestSAMRefineSafety` | 低 IoU 拒绝、高 IoU 接受、面积衰减拒绝 |
| `TestBoundaryRefineSafetyGate` | 面积比、bbox 比、质心位移、躯干重叠下降 |
| `TestProxyTorsoThreshold` | 代理/关键点两种躯干的阈值差异 |
| `TestVizRobustness` | 可视化函数对缺失字段/空遮罩的鲁棒性 |
| `TestSoftOpeningCorridor` | 梯形走廊顶部窄底部宽 |
| `TestCleanupCornerRemoval` | 上角直角残留移除、主体保留 |
| `TestSideStripRemoval` | 高窄侧条检测和移除 |
| `TestCleanupAreaSafetyGate` | 低面积比拒绝 |
| `TestCleanupDebugFields` | 调试字典字段完整性 |
| `TestDetectorArtifactCleanupField` | 检测器输出包含 artifact_cleanup 字段 |
| `TestCleanupVizRobustness` | 清理后的可视化鲁棒性 |

---

## 十一、已知限制与后续方向

1. **仅支持外套类别的内搭检测。** 裙子、连衣裙的内部层叠暂未实现（PRD 优先级限制）。
2. **躯干先验目前只用代理模式。** 姿态关键点数据尚未对接（需要姿态估计模型的集成）。
3. **阈值基于启发式经验设定。** Phase 2 的标注校准扫参（见 `docs/industrial_grounding_implementation_plan.md`）可系统化优化阈值。
4. **极端遮挡场景可能失败。** 如果外套完全拉链闭合（无可见开襟区域），领口补集分析没有搜索信号，此时依赖 SAM 回退策略。
5. **SAM 颈区多掩码开销较大。** 如果能确认前两个候选来源（CC + Canny）在大多数场景够用，可考虑默认跳过 SAM 候选生成以加速。
