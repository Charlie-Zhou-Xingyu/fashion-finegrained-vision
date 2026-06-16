# 3.1.1 服饰实例分割功能开发计划

## 1. PRD 要求

### 功能描述

对输入图片中的服饰进行精准实例分割。

### 输入

RGB 格式的商品图片，支持任意尺寸。

### 输出

每个服饰实例的：

- 分割掩码
- 边界框
- 类别标签

### 支持类别

系统目标类别按照 PRD 设计为 8 类：

| 类别 Key | 中文类别 |
|---|---|
| top | 上衣 |
| pants | 裤子 |
| skirt | 裙子 |
| outwear | 外套 |
| dress | 连衣裙 |
| shoes | 鞋子 |
| bag | 包包 |
| accessory | 配饰 |

### 性能要求

- 单张图片分割时间 ≤ 50ms
- 分割 IoU ≥ 0.85

## 2. 第一阶段目标

第一阶段构建 SAM-HQ 在 DeepFashion2 上的 box prompt baseline。

流程如下：

```text
DeepFashion2 image
        ↓
读取 annotation
        ↓
提取 garment bbox + category + gt mask
        ↓
bbox 输入 SAM-HQ
        ↓
得到 pred mask
        ↓
计算 pred mask 和 gt mask 的 IoU
        ↓
保存结果 JSON
        ↓
保存可视化图
```

## 3. 阶段说明

当前阶段使用 DeepFashion2 ground-truth bbox 作为 SAM-HQ 的 box prompt，
验证 SAM-HQ 在服饰实例 mask 生成上的能力。

后续将用自动检测器或语言定位模块生成 prompt，实现端到端实例分割。

## 4. 类别体系说明

系统接口按照 PRD 设计为 8 大类：

| 类别 Key | 中文类别 |
|---|---|
| top | 上衣 |
| pants | 裤子 |
| skirt | 裙子 |
| outwear | 外套 |
| dress | 连衣裙 |
| shoes | 鞋子 |
| bag | 包包 |
| accessory | 配饰 |

DeepFashion2 官方标注类别可以映射到其中的服装主体类别，包括：

- 上衣
- 裤子
- 裙子
- 外套
- 连衣裙

其中，DeepFashion2 的 `vest` 类别在当前项目中默认映射为 `outwear / 外套`。原因是电商服饰场景中 vest 可能表示外穿马甲、拉链马甲或无袖外套。后续如果具备更细粒度的属性标注，可以进一步区分内穿背心和外穿马甲。

对于 DeepFashion2 未充分覆盖的鞋子、包包、配饰，项目将采用以下方式补齐：

1. 使用 Label Studio 或 CVAT 对电商商品图进行自建标注；
2. 引入补充数据集：
   - 鞋子：UT Zappos、Fashionpedia、OpenImages、LVIS；
   - 包包：Fashionpedia、ModaNet、OpenImages、LVIS、COCO；
   - 配饰：Fashionpedia、ModaNet、OpenImages、LVIS；
3. 使用 GroundingDINO + SAM-HQ 的开放词汇检测分割方案，实现 8 类功能原型。


## 5. 第一阶段输出结构

```text
outputs/sam_hq_deepfashion2/
├── predictions/
│   ├── 000001.json
│   └── 000002.json
├── visualizations/
│   ├── 000001_vis.jpg
│   └── 000002_vis.jpg
├── metrics/
│   ├── iou_summary.json
│   └── iou_report.md
└── logs/
    └── run.log
```

## 6. 后续阶段

### 第二阶段：自动检测 + SAM-HQ

```text
Image
  ↓
Detector / GroundingDINO / YOLO / Mask2Former
  ↓
bbox + category
  ↓
SAM-HQ
  ↓
mask
```

### 第三阶段：语言引导局部区域定位

```text
Image + text prompt
        ↓
GroundingDINO / CLIP region proposal
        ↓
box prompt
        ↓
SAM-HQ
        ↓
target region mask
```
