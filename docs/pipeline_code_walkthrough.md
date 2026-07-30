# Pipeline Code Walkthrough

> 生成日期: 2026-07-27
> 项目: fashion_final_project
> 用途: 代码走读，理解完整 pipeline 实现细节
> 语言: 中文

---

## 0. 整体 Pipeline 概述

### 0.1 高层流程

```
                          图片输入
                             ↓
                     3.1.1 服饰检测/分割
                     输出: garment_instances
                     字段: category, bbox, confidence, mask_path, mask_present
                     ├── 视觉实例问答: "有几件衣服？" "有没有上衣？"
                     ↓
                     3.1.2 局部区域定位
                     输出: localized_regions
                     字段: part_type, bbox, confidence
                     ↓
                     3.1.3 属性识别
                     输入: region crop（从 3.1.2 输出的 bbox 裁剪）
                     输出: attribute value, confidence
                     ↓
                     区域属性问答: "领子是什么设计？"
```

### 0.2 两套调用路径

项目提供**两套独立的调用路径**，实现相同的 3.1.x Pipeline，但目标场景不同：

| 维度 | 路径 A: 函数调用 | 路径 B: 子进程 |
|---|---|---|
| 入口 | `tools/infer/garment_pipeline.py` | `tools/infer/run_garment_pipeline.py` |
| 架构 | `GarmentPipeline` 类，直接 import 调用 | 子进程 subprocess，读 JSON 文件 |
| Stage 覆盖 | 1-6 全阶段 | 1-5（无属性推理） |
| 谁在用 | `run_demo.py` 直接调用 | `SubprocessVisionProvider`（serving 层） |
| FP16 | 使用 legacy SAM wrapper（无 FP16） | 同上（子进程调用同一函数） |
| 延迟加载 | ✓（YOLO/SAM 每次重新加载） | ✓ |

### 0.3 核心数据流

```
run_demo.py
  └─→ GarmentPipeline.run_image(image_path, output_dir)
        ├─ Stage 1: YOLO 检测 → detections.json
        ├─ Stage 2: SAM-HQ 分割 → segmentation_results.json
        ├─ Stage 3: 关键点预测 → landmarks_results.json
        ├─ Stage 4: 区域裁剪 → region_crops.json
        ├─ Stage 5: 遮罩裁剪 → region_masked_crops.json
        └─ Stage 6: 属性推理 → predictions.jsonl

inference/serving/qa_orchestrator.py
  └─→ QaOrchestrator.answer(query, image, ...)
        ├─ IntentClassifier.classify(query) → primary_intent + sub_intent
        ├─ [visual_instance_query] → 读取 garment_instances
        ├─ [region_*] → 读取 localized_regions
        ├─ [attribute_query] → AttributeService → template answer
        └─ [knowledge_qa] → RagService → BM25 retrieval + template answer
```

---

## 1. 入口点（Entry Points）

### 1.1 `run_demo.py` — 统一 CLI 入口

| 属性 | 说明 |
|---|---|
| **文件** | `run_demo.py`（项目根目录） |
| **用途** | 单张图片端到端推理 + 可选的 QA |
| **何时使用** | 本地演示、快速验证、调试 |
| **导入** | `tools.infer.garment_pipeline.GarmentPipeline`、`inference.serving.qa_orchestrator.QaOrchestrator` |
| **输出** | `outputs/demo/result.json`（统一 JSON） |
| **必需依赖** | torch, ultralytics, segment-anything, numpy, opencv |
| **CLI 参数** | `--image`（必需）、`--query`（可选）、`--output-dir`、`--mode`（pipeline/qa/full）、`--no-attributes`、`--no-visualization`、`--format`（json/text） |
| **FP16** | 未接入（使用 legacy `SamHqWrapper`） |
| **TRT** | 未接入（`--sam-backend` 标记已推迟） |

### 1.2 `tools/infer/garment_pipeline.py` — 主 Pipeline 类

| 属性 | 说明 |
|---|---|
| **文件** | `tools/infer/garment_pipeline.py` |
| **用途** | 函数调用式 6 阶段 garment pipeline |
| **何时使用** | 被 `run_demo.py` 直接 import；其他脚本也可复用 |
| **核心类** | `GarmentPipeline`、`GarmentPipelineConfig` |
| **输入** | 图片路径（单张或目录） |
| **输出** | dict（含 timing、paths、config、summary） |
| **模型加载** | 延迟加载（landmark model 和 attribute pipeline 缓存；YOLO/SAM 每次在 stage 函数内重新加载） |
| **是否必需** | 是 — `run_demo.py` 依赖它 |

### 1.3 `scripts/run_qa_orchestrator.py` — CLI QA 演示

| 属性 | 说明 |
|---|---|
| **文件** | `scripts/run_qa_orchestrator.py` |
| **用途** | 单独演示 QA orchestration 链路 |
| **何时使用** | 快速测试 IntentClassifier → AttributeService → Template Answer |
| **模式** | 快速路径（读取已有 pipeline 输出）或慢速路径（子进程重跑 pipeline） |
| **导入** | `SubprocessVisionProvider`、`RuleIntentClassifier`、`AttributeService`、`QaOrchestrator` |
| **输出** | 控制台打印 + 可选 `result.json` |

### 1.4 `inference/serving/app.py` — FastAPI 服务

| 属性 | 说明 |
|---|---|
| **文件** | `inference/serving/app.py` |
| **用途** | 生产级 FastAPI serving 层 |
| **何时使用** | 部署为 HTTP API |
| **端点** | `GET /v1/health`、`GET /v1/metrics`、`POST /v1/mm/qa`、`POST /v1/intent/classify`、`POST /v1/rag/retrieve`、`POST /v1/merchant/content/generate` |
| **启动命令** | `uvicorn inference.serving.app:app` |
| **是否必需** | 否 — demo 可只用 `run_demo.py` |

### 1.5 `scripts/run_full_31x_pipeline.py` — 批量 CLI Runner

| 属性 | 说明 |
|---|---|
| **文件** | `scripts/run_full_31x_pipeline.py`（来自 Linux 项目） |
| **用途** | 批量跑多张图片的 pipeline |
| **何时使用** | 批量评估、预处理数据集 |
| **是否必需** | 否 — 用于批量处理 |

### 1.6 `scripts/build_31x_product_demo.py` — HTML 报告生成器

| 属性 | 说明 |
|---|---|
| **文件** | `scripts/build_31x_product_demo.py`（来自 Linux 项目） |
| **用途** | 从 pipeline 输出生成自包含 HTML 产品 demo 报告 |
| **输入** | pipeline 输出目录（`outputs/full_31x_demo/`） |
| **输出** | 单个 HTML 文件（base64 内嵌图片） |
| **是否必需** | 否 — 用于生成演示报告 |

---

## 2. 3.1.1 服饰检测/分割

### 2.1 代码位置映射

| 概念 | 代码位置 | 类/函数 | 输入 | 输出 | 备注 |
|---|---|---|---|---|---|
| Pipeline 编排 | `tools/infer/garment_pipeline.py:204` | `GarmentPipeline.run_source()` | image_path, output_dir | dict（含 timing, paths） | 调用 Stages 1-6 |
| YOLO 检测 | `tools/infer/predict_garments_yolo.py:275` | `run_inference(args)` | argparse.Namespace | 写入 `detections.json` | 无返回值 |
| SAM-HQ 分割 | `tools/infer/segment_garments_samhq.py:221` | `run(args)` | argparse.Namespace | 写入 `segmentation_results.json` | 无返回值 |
| 类别映射 | `configs/category_mapping.yaml` | — | 13-class YOLO ID | 5-class PRD ID + 中英文名 | 静态 YAML |
| 实例 Schema | `src/fashion_vision/schemas/instance_schema.py:31` | `build_instance_record()` | 字段参数 | `Dict[str, Any]` | 函数式，非 class |
| Legacy SAM 封装 | `src/fashion_vision/models/sam_hq_wrapper.py:41` | `SamHqWrapper` | checkpoint, model_type, device | `{mask, score, sam_latency_ms}` | 无 FP16，单 box |
| Linux SAM 封装 | `inference/wrappers/sam_wrapper.py:32` | `SamHqWrapper` | checkpoint, model_type, device, use_fp16=True | `(masks, scores, logits)` | FP16 + batched boxes |

### 2.2 图片进入 Pipeline 的方式

```
GarmentPipeline.run_image(image_path, output_dir)
  └─→ run_source(source=image_path, output_dir=output_dir, max_images=1)
        ├─ Stage 1: run_yolo_inference(args) → detections.json
        ├─ Stage 2: run_samhq_segmentation(args) → segmentation_results.json
        ├─ Stage 3: run_segmentation_json_mode(args) → landmarks_results.json
        ├─ Stage 4: run_region_crop(args) → region_crops.json
        ├─ Stage 5: run_apply_samhq_mask(args) → region_masked_crops.json
        └─ Stage 6: attr_pipeline.predict_from_json(crops_input) → predictions.jsonl
```

GarmentPipeline 不直接返回 garment_instances 列表。它返回的是**各阶段输出文件路径**的 dict。实际的 garment_instances 存在 `segmentation_results.json` 中，由后续的 QA 层（`SubprocessVisionProvider`）读取。

### 2.3 YOLO 检测细节

**文件:** `tools/infer/predict_garments_yolo.py`

- 模型: `ultralytics.YOLO` 加载 `models/detectors/yolov8n_deepfashion2_13cls_best.pt`
- 13 类别映射从 `configs/category_mapping.yaml` 加载（`deepfashion2_13cls` key）
- 每个检测记录字段：
  - `class_id`（0-12，YOLO class index）
  - `class_name`（如 `short sleeve top`）
  - `fine_class_id` / `fine_class_name`（同 class_id / class_name）
  - `coarse_class_id` / `coarse_class_name`（5 类 PRD 映射）
  - **`confidence`**（浮点数，YOLO 置信度）
  - `bbox_xyxy`（`[x1, y1, x2, y2]`，绝对像素坐标）
  - `bbox_xywh`、`bbox_format`

**关键字段名: `confidence`（不是 `score`）**

### 2.4 SAM-HQ 分割细节

**文件:** `tools/infer/segment_garments_samhq.py`

- 模型: 通过 `import_samhq()` 加载（先尝试 `segment_anything`，再尝试 `segment_anything_hq`）
- 使用 legacy `SamHqWrapper` 的底层 SAM Predictor（非 FP16）
- 每个 segment 记录字段：
  - `det_id`、`class_id`、`class_name`
  - `confidence`（来自 YOLO，保持不变）
  - `bbox_xyxy`（来自 YOLO）
  - **`mask_path`**（PNG 文件路径，如 `02_samhq/masks/img001_det000_short_sleeve_top_mask.png`）
  - `mask_area`（像素数）
  - `sam_score`（SAM 内部 IoU 预测分数，可为 None）
  - `image_width`、`image_height`

**关键字段名: `mask_path`（不是 `pred_mask_path`），`confidence`（不是 `score`）**

### 2.5 garment_instances 的构造

`garment_instances` **不是**由 `GarmentPipeline` 直接返回的。它由 `SubprocessVisionProvider._read_existing_output()` 从 `detections.json` 读出来构造。每个 instance 包含：

```python
{
    "instance_id": "img001_det000",
    "category": "top",                    # 5-class PRD
    "fine_class_name": "short sleeve top", # 13-class DeepFashion2
    "confidence": 0.95,                   # YOLO 置信度
    "bbox_xyxy": [120, 80, 340, 450],
    "mask_present": True,                 # SAM 分割是否成功
}
```

### 2.6 字段命名不一致（重要）

| 概念 | detections.json / segmentation_results.json | instance_schema.py `build_instance_record` |
|---|---|---|
| 分割遮罩路径 | **`mask_path`** | **`pred_mask_path`** |
| 检测置信度 | **`confidence`** | **`score`** |
| SAM 内部分数 | **`sam_score`** | — |
| 边界框 | **`bbox_xyxy`** | **`bbox`** |

`instance_schema.py` 使用 `pred_mask_path` 和 `score`，但实际 pipeline JSON 输出使用 `mask_path` 和 `confidence`。`run_demo.py` 目前**不做字段归一化**（`normalize_pipeline_output` 只转发 `output_paths` 和 `timing`）。

### 2.7 Linux 优化 SAM Wrapper 是否已接入

**未接入。** `GarmentPipeline.run_source()` 的 Stage 2 调用 `tools/infer/segment_garments_samhq.py:run()`，它使用自己的 `import_samhq()` 函数加载 SAM（不经过 `inference/wrappers/sam_wrapper.py`）。

`inference/wrappers/sam_wrapper.py` 的功能（FP16 + batched boxes）在当前 pipeline 中**不可用**，除非修改 `segment_garments_samhq.py` 使其使用优化 wrapper。

### 2.8 所需模型权重

| 模型 | 路径 | 用途 |
|---|---|---|
| YOLOv8n | `models/detectors/yolov8n_deepfashion2_13cls_best.pt` | 13 类服饰检测 |
| SAM-HQ ViT-B | `checkpoints/sam_hq/sam_hq_vit_b.pth` | 实例分割 mask 生成 |
| SAM-HQ 代码库 | `third_party/sam-hq/`（git clone） | SAM 模型定义 |

---

### 2.9 当前检测/分割成果数据

> **数据来源**: 周报 + YOLO balanced retraining evaluation (`outputs/yolo_eval_balanced/`)  
> **验证规模**: 32,153 张 DeepFusion2 验证集图片  
> **训练策略**: Hard-Mining 难例挖掘 + 均衡化采样 (bp10_r8_hard, Balance Power 1.0, Max Repeat 8, 分辨率 800, AMP 关闭)  
> **最终采用模型**: `models/detectors/yolov8n_deepfashion2_13cls_best.pt` (内部 13 类, 对外映射为 5 粗类)

#### 5 大类 mAP

| 大类 | Instances | mAP50 | mAP50-95 |
|---|---:|---:|---:|
| Bottoms / 裤子 | 13,753 | 95.8% | 78.8% |
| Tops / 上衣 | 20,957 | 91.6% | 76.1% |
| Skirts / 裙子 | 6,522 | 90.5% | 79.5% |
| Outwear / 外套 | 2,153 | 83.9% | 75.4% |
| Dresses / 连衣裙 | 9,105 | 81.5% | 75.1% |

#### 分割 IoU (SAM-HQ ViT-B)

- **Overall IoU**: 0.84
- **验证样本数**: n = 467

| IoU 区间 | 数量 | 占比 |
|---|---:|---:|
| [0.0, 0.2) | 4 | 0.9% |
| [0.2, 0.3) | 4 | 0.9% |
| [0.3, 0.4) | 10 | 2.1% |
| [0.4, 0.5) | 12 | 2.6% |
| [0.5, 0.6) | 34 | 7.3% |
| [0.6, 0.7) | 43 | 9.2% |
| [0.7, 0.8) | 92 | 19.7% |
| [0.8, 0.9) | 165 | 35.3% |
| [0.9, 1.0) | 103 | 22.1% |

#### 500 图 Pipeline 稳定性

> 来源: `results/benchmark/pipeline_benchmark_500_report.md`

| 指标 | 数值 |
|---|---:|
| 输入图片 | 500 |
| 成功处理 | 500 (100%) |
| 处理实例 | 913 (平均 1.83/图) |
| 区域裁剪 | 3,220 (100%) |
| Mask裁剪 | 3,212 (99.75%) |
| 端到端延迟 | 420.13 ms/图 |
| 吞吐量 | 2.38 图/s |

#### Pipeline 阶段耗时分解

| 阶段 | 平均耗时/图 | 占比 |
|---|---:|---:|
| YOLO 检测 | 24.27 ms | 5.78% |
| SAM-HQ 分割 | 292.51 ms | 69.62% |
| Landmark 预测 | 54.40 ms | 12.95% |
| 区域裁剪 | 6.80 ms | 1.62% |
| Mask裁剪 | 42.14 ms | 10.03% |

#### 结论

- **裤子、上衣、裙子** mAP50 均 ≥ 90%，表现优秀
- **外套、连衣裙** mAP50 相对较低 (81-84%)，受长尾分布影响
- **分割 IoU 集中在 0.8-1.0 区间 (57.4%)**，说明主体 mask 质量整体较稳定
- **SAM-HQ 是最大瓶颈**，占总耗时 ~70%，是推理优化的首要目标
- 仍未覆盖: 鞋子、包包、配饰、图案/印花、局部装饰元素等类别

---

## 3. 视觉实例问答

### 3.1 问题类型

以下问题走 `visual_instance_query` 路由：

| sub_intent | 示例问题 | 处理方式 |
|---|---|---|
| `count` | "有几件衣服？"、"图里检测到几件？" | 计数 `garment_instances`，按类别分组 |
| `existence` | "有没有上衣？"、"有没有外套？" | 匹配 `entities["garment_ref"]` 与 instance category |
| `detection` | "检测到了什么？"、"图里有什么？" | 列举所有 instance 的 `fine_class_name` |
| `location` | "检测框在哪里？" | 报第一个 instance 的 `bbox` |
| `segmentation` | "有没有分割结果？" | 统计 `mask_present` 数量 |

### 3.2 意图分类

**文件:** `inference/serving/intent_classifier.py`

`RuleIntentClassifier.classify(query)` 是**纯规则引擎**：
1. 先匹配关键词（`keywords` 列表），置信度 = 0.95
2. 再匹配正则（`patterns` 列表），置信度 = 0.80
3. 规则从上到下评估，**第一个匹配就返回**
4. 无匹配返回 `fallback_unknown`（置信度 = 0.0）

延迟 < 200 微秒。不依赖任何 ML 模型、embedding、LLM。

### 3.3 答案生成

**文件:** `inference/serving/qa_orchestrator.py:500` — `_route_visual_instance()`

**所有答案都是确定性的模板生成，不使用 LLM。**

`_route_visual_instance` 从 `VisionContext.garment_instances` 读取数据，按 `sub_intent` dispatch：

```python
if sub == "count":
    # 统计 "上衣 2件, 裤子 1件"
elif sub == "detection":
    # 列举 "检测到: 短袖上衣 (置信度 0.95), 长裤 (置信度 0.88)"
elif sub == "existence":
    # 检查 "上衣" in [inst["category"] for inst in instances]
```

- **使用字段**: `category`、`fine_class_name`、`confidence`、`bbox_xyxy`、`mask_path`
- **不依赖 LLM**: 所有回答由 Python f-string 模板拼装
- **如果无 garment**: 返回"未检测到服饰"

### 3.4 数据流

```
QaOrchestrator.answer(query="有几件衣服？")
  └─→ IntentClassifier.classify(query)
        → primary_intent="visual_instance_query", sub_intent="count"
  └─→ build_vision_context(vision_provider, ...)
        → VisionContext.garment_instances = [...]  # 来自 vision provider
  └─→ _route_visual_instance(q, primary, "count", entities, ...)
        → 遍历 vc.garment_instances
        → 按 category 分组计数
        → 模板生成: "检测到上衣 2 件、裤子 1 件。"
```

---

## 4. 3.1.2 局部区域定位

### 4.1 localized_regions 的含义

`localized_regions` 是从每个 garment instance 中定位出的**局部服饰区域**列表。每个 region 描述一个具体的服饰部位（如领口、袖口、下摆）的空间位置。

### 4.2 路由架构

**文件:** `src/fashion_vision/localization/region_localization_router.py:51`

`locate_region(query, instance, image, ...)` 是统一入口，按三路路由：

| 条件 | 后端 | result.backend | 使用场景 |
|---|---|---|---|
| `intent.is_fast_path == True` | `locate_region_from_instance()` (landmark+geometry) | `"fast_path"` | 结构性部位: neckline, cuff, hem, waist, shoulder, leg_opening |
| Fast path = False + 在 Fashionpedia 覆盖范围 | Fashionpedia YOLO 检测器 | `"fashionpedia_yolo"` | zipper, pocket, button, buckle, bow, ruffle 等 |
| Fast path = False + 不在 Fashionpedia 范围 | Grounding DINO 开放词汇检测 | `"open_vocab_grounding_dino"` | 不在 FP 19 类中的部位 |
| Zero-shot（未知部位） | Grounding DINO zero-shot | `"zero_shot_grounding_dino"` | 无法匹配任何已知 part 词汇 |

**特殊降级逻辑**（line 240-243）：`neckline` 和 `cuff` 在 Fashionpedia 类别中，但如果 YOLO 返回空，会 fallback 到 fast path（landmark 方法）。

### 4.3 Fast Path（Landmark + Geometry）

**文件:** `src/fashion_vision/localization/region_locator.py`

流程：
1. `parse_region_type(query)` → 规范 part name（如 "领口" → `"neckline"`）
2. 检查 `SUPPORTED_REGIONS_BY_CATEGORY[garment_category]`
3. 加载 instance mask
4. **优先**: `locate_region_by_landmarks()` 使用预测的关键点定位
5. **降级**: 几何方法（`locate_neckline_mask_adaptive()`、`locate_cuff_mask_adaptive()` 等）

**关键点支持 39 个 landmark**（文件: `tools/infer/infer_landmarks_for_predictions.py`），来自 DeepFashion2 landmark schema。

### 4.4 Fashionpedia YOLO 检测器

**文件:** `src/fashion_vision/localization/fashionpedia_part_detector.py`

- 模型: YOLOv8s 加载 `models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt`
- 19 类部位: hood, collar, lapel, epaulette, sleeve, pocket, neckline, buckle, zipper, applique, bead, bow, flower, ribbon, rivet, fringe, ruffle, sequin, tassel
- 别名: `"cuff"` 映射到 sleeve class
- 输出: `[{"bbox_xyxy": [...], "score": 0.85, "label": "zipper"}, ...]`

### 4.5 Grounding DINO 开放词汇定位

**文件:** `src/fashion_vision/localization/grounding_dino_locator.py`

- 模型: `IDEA-Research/grounding-dino-tiny`（HuggingFace transformers 加载）
- 将 garment mask 外的像素涂灰后，发送文本 prompt 给 DINO
- 支持单 prompt 和多 prompt（multi-instance parts 如口袋、扣子）
- 输出: `[{"bbox_xyxy": [...], "score": 0.72, "label": "clothing pocket", "prompt": "clothing pocket"}, ...]`

### 4.6 部位查询映射

**文件:** `src/fashion_vision/localization/intent_parser.py`

`parse_intent(query)` 解析中文查询，提取：
- `part`（如 `"neckline"`、`"cuff"`、`"zipper"`）
- `side`（`"left"` / `"right"`）
- `garment_ref`（`"outerwear"` / `"top"` / `"pants"` / `"skirt"` / `"dress"` / `"inner"`）
- `direction`（`"upper"` / `"lower"` / `"front_upper"` / `"back"`）
- `spatial_anchor`（从 "X附近" / "X上的" 提取）

**`PART_VOCAB`** 覆盖约 26 个部位，每个有中英文关键词。`FAST_PATH_PARTS = {"hem", "waist", "leg_opening", "shoulder"}`（neckline 和 cuff 已迁移到 Fashionpedia-first）。

### 4.7 区域裁剪工具

**文件:** `tools/crop/crop_garment_regions_from_landmarks.py`

从 `landmarks_results.json` 读入关键点，按 `CATEGORY_TO_REGIONS` 映射裁剪每个部位区域。

输出字段（`region_crops.json` 每条记录）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `region` | str | 逻辑区域: `"collar"`, `"sleeve"`, `"hem"`, `"waist"`, `"pant_leg"` |
| `component` | str | 具体构件: `"left_sleeve"`, `"right_sleeve"`, `"collar"` |
| `bbox_xyxy` | list | 裁剪框 |
| `crop_path` | str | 裁剪图片路径 |
| `source` | str | `"landmark_region"` 或 `"bbox_fallback"` |
| `fallback` | bool | 是否使用降级 bbox |

**文件:** `tools/crop/apply_samhq_mask_to_region_crops.py`

对每个 region crop 应用 SAM-HQ mask，输出三种图片：`image_crop`、`mask_crop`、`masked_crop`。

### 4.8 字段命名转换（关键）

| 层 | bbox 字段 | score 字段 | part 字段 |
|---|---|---|---|
| Router 原始输出 | `bbox`（top-1）、`all_bboxes`（list） | `score` | `part` |
| Fashionpedia / DINO 检测 | `bbox_xyxy` | `score` | `label` |
| Serving 层输出（normalize） | `bbox`（归一化 float list） | **`confidence`** | **`part_type`** |

**在 serving 层（`region_backend.py:normalize_region_predictions()`）**：
- `raw["score"]` → `"confidence"`
- `raw["label"]` → 通过 `_FP_LABEL_TO_PART_TYPE` 映射 → `"part_type"`
- `raw["bbox_xyxy"]` → `"bbox"`

### 4.9 路由器输出结构

**成功时的完整字段:**
```python
{
    "query": "领口在哪儿",
    "part": "neckline",          # ← 注意: "part" 不是 "part_type"
    "garment_ref": None,
    "backend": "fast_path",
    "status": "success",
    "bbox": [180, 75, 280, 120],
    "score": 0.82,              # ← 注意: "score" 不是 "confidence"
    "all_bboxes": [[180, 75, 280, 120], ...],
    "all_scores": [0.82, 0.31], # ← 注意: 独立数组，不是 list of dict
    "mask": np.ndarray,
    "mask_source": "landmark_crop",
    "debug": {...}
}
```

### 4.10 关键配置

| 配置 | 文件 | 关键 key |
|---|---|---|
| Region backend 开关 | `configs/serving_config.yaml` | `region_backend.backend: disabled`（默认关闭） |
| Fashionpedia 模型路径 | `configs/serving_config.yaml` | `region_backend.model_path` |
| 置信度阈值 | `configs/serving_config.yaml` | `region_backend.confidence_threshold: 0.5` |
| Intent taxonomy | `configs/intent_taxonomy.yaml` | `region_*` 规则（6 种 region intent） |
| Region 查询映射 | `inference/serving/region_query_mapper.py` | `_CHINESE_TO_PART`（76 条映射） |

---

### 4.11 当前局部区域定位评估结果

> **数据来源**: 周报 (`周报_312.pdf`) + Label Studio 标注验证集  
> **最终采用方案**: Fast Path (Landmark + Geometry) + Fashionpedia YOLO **v1** + GroundingDINO **tiny** 三路路由

#### 验证集规模

- **标注工具**: Label Studio
- **零件类别**: 23 类服装零件
- **图片数**: 1,127 张
- **标注框数**: 1,189 个

#### 标注规则

- **neckline vs collar 区分**: 开口露皮肤 → neckline; 布料实体 → collar
- 删除"图中看不见但凭常识推断存在"的样本
- button 统一标为 button_cluster
- collar + neckline + lapel 分开标注，评估时支持合并计算

#### 评估规则

**Containment 规则**: 若预测框完全落在 GT 框内部 (`pred ⊆ GT`)，则计为命中。该规则主要解决装饰类小零件 GT 框偏大、预测框偏紧导致传统 IoU 低估的问题。

Containment 对部分小零件的影响:

| 零件 | 无 containment | 有 containment |
|---|---:|---:|
| applique | 0% | 93.8% |
| bead | 0% | 84.8% |
| ribbon | 0% | 97.4% |
| flower | 0% | 60.0% |
| rivet | 0% | 20.5% |

#### 整体 Baseline (23 零件)

| IoU 阈值 | Overall Accuracy | 最佳零件 | 最差零件 |
|---|---:|---|---|
| > 0.01 | 66.7% | ribbon 97.4% | ruffle 34.9% |
| > 0.15 | 57.8% | sleeve 88.2% | ruffle 32.6% |
| > 0.30 | 51.9% | sleeve 88.2% | ruffle 25.6% |

#### collar + neckline + lapel 合并评估

| IoU 阈值 | Accuracy | 标注数 |
|---|---:|---:|
| > 0.01 | 67.6% | 142 |
| > 0.15 | 65.5% | 142 |
| > 0.30 | 60.6% | 142 |

#### 当前 23 零件精度总览 (已知部分)

| 零件 | 标注数 | @IoU>0.01 | @IoU>0.15 | @IoU>0.30 | 后端 | 备注 |
|---|---:|---:|---:|---:|---|---|
| ribbon | 39 | 97.4% | 79.5% | 64.1% | DINO | 小装饰 |
| applique | 48 | 93.8% | 52.1% | 50.0% | DINO | 小装饰, containment关键 |
| tassel | 55 | 90.9% | 63.6% | 40.0% | DINO | — |
| sequin | 48 | 89.6% | 77.1% | 77.1% | DINO | 本周期最大改善 |
| sleeve | 85 | 89.4% | 88.2% | 88.2% | FP YOLO | 已稳定 |
| bead | 50 | 86.0% | 50.0% | 46.0% | DINO | 小装饰, containment关键 |
| lapel | 42 | 81.0% | 78.6% | 71.4% | FP YOLO | — |
| bag | 48 | 79.2% | 68.8% | 52.1% | DINO | — |
| neckline | 53 | 71.7% | 67.9% | 60.4% | FP + DINO | 动态路由 |
| epaulette | 68 | 69.1% | 69.1% | 66.2% | FP YOLO | — |
| hood | 45 | 68.9% | 66.7% | 60.0% | FP YOLO | — |
| strap | 80 | 65.0% | 55.0% | 48.8% | DINO | — |

> ⚠️ 完整 23 类精度表需进一步从 `周报_312.pdf` 补全。以上 12 类为已知部分。剩余 11 类 (shoes, pocket, collar, buckle, fringe, zipper, ruffle, flower, rivet, button, bow) 待补充。

#### Fashionpedia YOLO v1 vs v2 对比

> 最终决策: **保持 v1** (`models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt`)

| 设置 | v1 | v2 |
|---|---|---|
| Balance Power | 1.0 (极端) | 0.6 |
| Max Repeat | 12 | 6 |
| 扩张比 | 1.0× | 1.18× |
| mAP50 | ~0.312 | — |

在 12 个 FP-core 零件 (562 张验证图) 上的对比:

| Part | v1 Recall | v2 Recall | Δ Recall | v1 Precision | v2 Precision | Δ Precision |
|---|---:|---:|---:|---:|---:|---:|
| bow | 0.4800 | 0.3800 | **-0.1000** | 0.6486 | 0.6129 | -0.0357 |
| buckle | 0.4468 | 0.4681 | +0.0213 | 1.0000 | 1.0000 | 0.0000 |
| collar | 0.2766 | 0.2766 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| epaulette | 0.4200 | 0.4200 | 0.0000 | 0.4565 | 0.5250 | +0.0685 |
| fringe | 0.5000 | 0.4545 | **-0.0455** | 0.9565 | 0.9524 | -0.0041 |
| hood | 0.4889 | 0.4444 | **-0.0445** | 1.0000 | 1.0000 | 0.0000 |
| lapel | 0.5238 | 0.5000 | **-0.0238** | 1.0000 | 0.9545 | -0.0455 |
| neckline | 0.5800 | 0.5800 | 0.0000 | 0.9355 | 0.9355 | 0.0000 |
| pocket | 0.5200 | 0.4800 | **-0.0400** | 0.4333 | 0.4364 | +0.0031 |
| ruffle | 0.3953 | 0.3721 | **-0.0232** | 0.6538 | 0.5517 | -0.1021 |
| sleeve | 0.7400 | 0.7800 | **+0.0400** | 0.5286 | 0.5270 | -0.0016 |
| zipper | 0.3864 | 0.3864 | 0.0000 | 0.5000 | 0.3953 | -0.1047 |
| **Overall** | **0.4822** | **0.4644** | **-0.0178** | — | — | — |

**结论**: v2 整体 recall 下降 1.78pp, 6 个零件退化, 仅 sleeve 明显提升. **最终保持 v1**, 以 tail part recall 为优先.

#### DINO tiny vs base

> 最终决策: **保持 DINO-tiny** (`IDEA-Research/grounding-dino-tiny`, 本地路径 `models/grounding_dino_tiny/`)

| 指标 | DINO-tiny | DINO-base |
|---|---:|---:|
| 参数量 | 172M | 232M |
| 磁盘占用 | ~200MB | 1.8GB |
| GPU 显存 | 660MB | 891MB |
| 单 prompt 延迟 | 319ms | 409ms |
| 延迟变化 | — | +31% |

精度对比 (IoU > 0.3, 15 samples/part):

| 零件 | Tiny | Base | Delta | 结论 |
|---|---:|---:|---:|---|
| collar | 66.7% | 60.0% | -6.7% | Tiny 赢 |
| neckline | 20.0% | 13.3% | -6.7% | Tiny 赢 |
| zipper | 46.7% | 26.7% | -20.0% | Tiny 赢 |
| pocket | 46.7% | 40.0% | -6.7% | Tiny 赢 |
| button | 20.0% | 40.0% | +20.0% | Base 赢 |
| epaulette | 13.3% | 33.3% | +20.0% | Base 赢 |
| shoes | 26.7% | 26.7% | 0 | 持平 |
| rivet | 20.0% | 20.0% | 0 | 持平 |

**结论**: Base 在 button/epaulette 上提升, 但这些收益落在已有 FP YOLO 覆盖或非核心优势上. Base 对 collar/neckline/zipper/pocket 等**高频核心零件退化**, 延迟 +31%, 显存 +231MB. **最终保持 DINO-tiny** 作为 zero-shot fallback.

#### sequin 后端动态切换

- FP YOLO 对 sequin 召回约 27%
- DINO 裸召回约 90%
- **切换后 eval 召回从 27% 提升到 77.1%** (IoU > 0.3)
- 这是该周期**最大单点提升**
- 证明了后端动态选择策略的有效性

#### 3.1.2 Batch60 在线查询验证

> 来源: `results/benchmark/query_region_batch60_report.md`

| 指标 | 数值 |
|---|---:|
| 查询次数 | 300 (60 图 × 5 query) |
| 成功 | 276 |
| 失败 | 24 |
| 有效响应率 | **92.0%** |

| Query | 成功率 |
|---|---:|
| 腰部 | 100.0% |
| 下摆 | 95.0% |
| 领口 | 95.0% |
| 左袖子 | 85.0% |
| 右袖子 | 85.0% |

> ⚠️ 注意: 92% 是有效响应率 (valid response rate), 不是严格定位精度. 该 batch 未做人工 correctness 标注.

---

## 5. 3.1.3 属性识别

### 5.1 代码位置映射

| 概念 | 代码位置 | 类/函数 | 说明 |
|---|---|---|---|
| 属性 Pipeline | `src/fashion_vision/attributes/garment_attribute_pipeline.py` | `GarmentAttributePipeline` | 主要入口，从 crop JSON 推理 |
| 任务注册表 | `src/fashion_vision/attributes/task_registry.py` | `load_inference_config()`、`load_task()` | 从 YAML 加载 checkpoint/label_map |
| 类别门控 | `src/fashion_vision/attributes/category_gate.py` | `AttributeGroupMapping`、`get_enabled_tasks()` | 服饰类别 → 属性任务映射 |
| Serving 属性服务 | `inference/serving/attribute_service.py` | `AttributeService` | 模板回答生成 |
| Serving 属性后端 | `inference/serving/attribute_backend.py` | `FashionAttributeBackend` | 加载 classifier、执行推理 |

### 5.2 Region Crop 如何进入属性分类器

```
region_masked_crops.json (Stage 5 输出)
  └─→ GarmentAttributePipeline.predict_from_json(crops_path)
        ├─ 按 {image_stem}__det{det_id} 分组 crop 记录
        ├─ 对每个 instance，确定 coarse_class_name
        ├─ 查询 category_gate.get_enabled_tasks(coarse_class_name)
        ├─ 对每个 task，选择对应的 crop（upper_crop / expanded_crop）
        └─ 对每个 crop，运行 ResNet18 分类器
             → 输出 {"label": "翻领", "score": 0.88, "topk": [...]}
```

### 5.3 8 个属性任务

| 任务 | 中文名 | 区域 | 模型路径 | 类别数 | 当前 F1 |
|---|---|---|---|---|---|
| `neckline_design` | 领口设计 | collar | `models/attributes/p2_neckline_design_resnet18_seed2/best.pt` | 10 | 0.665 |
| `collar_design` | 领型设计 | collar | `models/attributes/p2_collar_design_resnet18_seed2/best.pt` | 5 | 0.764 |
| `neck_design` | 颈部设计 | collar | `models/attributes/p2_neck_design_resnet18_seed2/best.pt` | 5 | 0.624 |
| `lapel_design` | 翻领设计 | collar | `models/attributes/p2_lapel_design_resnet18_seed2/best.pt` | 5 | 0.680 |
| `sleeve_length` | 袖长 | cuff | `models/attributes/p2_sleeve_length_multiview_v2_pipeline_resnet18_seed2/best.pt` | 9 | 0.612 |
| `coat_length` | 衣长 | hem | `models/attributes/p2_coat_length_resnet18_seed2/best.pt` | 8 | 0.618 |
| `pant_length` | 裤长 | leg_opening | `models/attributes/p2_pant_length_multiview_v2_pipeline_resnet18_seed2/best.pt` | 6 | 0.740 |
| `skirt_length` | 裙长 | hem | `models/attributes/p2_skirt_length_multiview_v2_pipeline_resnet18_seed2/best.pt` | 6 | 0.593 |

所有模型目标 F1 = 0.88，当前没有任务达标。最佳是 `collar_design`（0.764）。

### 5.4 类别到属性映射

来自 `configs/attribute_group_mapping.yaml`：

| 服饰类别 | 启用的属性任务 |
|---|---|
| `top`（上衣） | neckline_design, collar_design, neck_design, sleeve_length |
| `outerwear`（外套） | lapel_design, coat_length, sleeve_length |
| `pants`（裤子） | pant_length |
| `skirt`（裙子） | skirt_length |
| `dress`（连衣裙） | neckline_design, skirt_length, sleeve_length |

### 5.5 各层输出字段对照

| 层 | label 字段 | confidence 字段 | topk 格式 |
|---|---|---|---|
| `GarmentAttributePipeline._run_inference()` | `"label"` | `"score"` | `[{"label": "...", "score": 0.87}, ...]` |
| `FashionAttributeBackend._normalize_attribute_result()` | `"value"` | `"attribute_confidence"` | 透传（截断到 top 3） |
| `AttributeService.answer_attribute()` | `.answer` | `.answer_confidence` | — |

### 5.6 检查点缺失时的行为

`FashionAttributeBackend._ensure_loaded()` 对每个任务检查 checkpoint 是否存在。缺失的路径记录在 `self._missing_checkpoints` 列表中。

**注意:** Serving 后端（`attribute_backend.py`）硬编码的 checkpoint 路径使用 `outputs/` 前缀（如 `outputs/p2_collar_design_resnet18_seed2/best.pt`），而 `configs/attribute_inference.yaml` 使用 `models/attributes/` 前缀。**以 YAML 配置为准**（`GarmentAttributePipeline` 通过 task_registry 加载正确的 YAML 路径）。

### 5.7 属性 Pipeline 输出示例

```python
{
    "det_id": "img001_det000",
    "fine_class_name": "short sleeve top",
    "coarse_class_name": "top",
    "num_crops": 4,
    "attributes": {
        "neckline_design": {
            "label": "Round Neckline",
            "score": 0.87,
            "topk": [
                {"label": "Round Neckline", "score": 0.87},
                {"label": "V Neckline", "score": 0.09},
                {"label": "Square Neckline", "score": 0.03}
            ]
        },
        "collar_design": {
            "label": "Rib Collar",
            "score": 0.76,
            "topk": [...]
        }
    },
    "error": None
}
```

---

### 5.8 当前属性识别成果与限制

#### 8 个属性任务当前 F1

| 任务 | 中文名 | 类别数 | 当前 F1 | vs PRD 目标 (0.88) |
|---|---|---|---|---|
| `collar_design` | 领型设计 | 5 | **0.764** | -0.116 |
| `pant_length` | 裤长 | 6 | **0.740** | -0.140 |
| `lapel_design` | 翻领设计 | 5 | 0.680 | -0.200 |
| `neckline_design` | 领口设计 | 10 | 0.665 | -0.215 |
| `neck_design` | 颈部设计 | 5 | 0.624 | -0.256 |
| `coat_length` | 衣长 | 8 | 0.618 | -0.262 |
| `sleeve_length` | 袖长 | 9 | 0.612 | -0.268 |
| `skirt_length` | 裙长 | 6 | 0.593 | -0.287 |

#### 结论

- **最优任务**: `collar_design`, F1 = 0.764
- **次优任务**: `pant_length`, F1 = 0.740
- **所有任务均未达到 PRD 目标 F1 = 0.88** (差距 0.116 ~ 0.287)
- **主要瓶颈**: FashionAI 数据覆盖有限 (每任务 556-1,647 训练样本), 模型参数量 (ResNet18) 不是瓶颈
- **候选数据集**:
  - DeepFashion Attribute Prediction
  - iMaterialist Fashion 2019 FGVC6 (~23GB, 下载/使用尚未完成)
- **已集成能力**:
  - 按服饰粗类别自动路由 (top→领型+领口+袖长, pants→裤长, dress→领口+裙长+袖长 等)
  - Mask-aware crop (SAM-HQ mask 背景擦除, 减少衣架/皮肤/其他衣物干扰)
  - Lazy Loading + 显存缓存 (`task_registry.py`)
  - 20 图批量验证: 100% 通过率 (WSL samhq-trt, 2026-07-28)

---

## 6. 区域属性问答

### 6.1 问题类型

以下问题走 `region_attribute_query` 或 `attribute_query/collar` 路由：

| 意图 | 示例问题 | 处理方式 |
|---|---|---|
| `region_attribute_query` | "领口是什么设计？"、"袖口是什么样式？" | region backend 定位 → attribute backend 分类 |
| `attribute_query/collar` | "这件衣服的领子是什么设计？" | attribute_service 直接回答 |
| `attribute_query/sleeve` | "袖子是什么长度？" | 同上 |
| `attribute_query/length` | "衣长是多少？" | 同上（通过 length fallback 链） |

### 6.2 查询 → 属性答案的完整链路

```
"领口是什么设计？"
  └─→ IntentClassifier.classify()
        → primary_intent="region_attribute_query", sub_intent=None
        → matched_pattern: ".*(领口).*(什么设计).*"
  └─→ QaOrchestrator._route_region_query()
        ├─ 从 vc.localized_regions 提取 region
        ├─ 确定 part_type="neckline"
        └─→ attribute_backend.extract_attributes(image, bbox, tasks=["neckline_design", "collar_design", ...])
              ├─ 从 bbox 裁剪图片区域 (BGR→RGB)
              ├─ 运行 ResNet18 分类器
              ├─ 输出: {"value": "V Neckline", "attribute_confidence": 0.88, "topk": [...]}
              └─→ 生成模板答案: "领口设计为V Neckline（置信度0.88）。"
```

**或走简化的 attribute_query 路由：**

```
"这件衣服的领子是什么设计？"
  └─→ IntentClassifier.classify()
        → primary_intent="attribute_query", sub_intent="collar"
  └─→ QaOrchestrator._route_attribute()
        ├─ attr_name = "collar_design" (从 alias 解析)
        └─→ AttributeService.answer_attribute("collar_design", attrs)
              ├─ 从 attrs 字典查找 "collar_design"
              ├─ 如果不存在: 返回 "暂无领型设计信息。"
              ├─ 如果存在: 模板渲染 → "领型设计为翻领（置信度0.88）。"
```

### 6.3 答案生成方式

**所有答案都是模板生成，不使用 LLM。** 模板来自 `configs/attribute_templates.yaml`。

置信度分为三档：
- **高置信度**（≥ 0.8）: 直接陈述
- **中置信度**（≥ 0.6）: 加 "可能" 修饰
- **低置信度**（< 0.6）: 加 "检测到但不完全确定" 修饰
- **不可用**: 返回 "暂无XXX信息"

### 6.4 证据返回

`QaOrchestrator.answer()` 的返回包含 `sources` 列表，每个 source 包含：
- `type`、`field`、`value`、`confidence`/`attribute_confidence`
- 对于 region attribute: 还包含 `region_id`、`instance_id`

### 6.5 降级和警告

| 场景 | 行为 |
|---|---|
| 无 region backend | `serving_config.yaml` 默认 `region_backend.backend: disabled` → region 查询返回"不支持" |
| 无 attribute backend | `serving_config.yaml` 默认 `attribute_backend.backend: disabled` → 属性查询走 attribute_service 的字典查找路径 |
| region 定位失败 | `_route_region_query` 返回 `"not_detected"` 状态，answer 说明该部位未找到 |
| 属性分低置信度 | 模板添加置信度说明，仍返回最佳猜测 |
| 模型文件缺失 | `FashionAttributeBackend._missing_checkpoints` 记录，`_ensure_loaded` 失败则返回空 |

---

## 7. 数据 Schema 和字段归一化

### 7.1 目标 vs 实际字段对照

| 期望字段（统一 Schema） | 实际代码中的字段 | 在哪里产生 | 是否需要归一化 |
|---|---|---|---|
| `confidence` | YOLO/SAM JSON: **`confidence`** | `predict_garments_yolo.py:391` | **不需要**（已是 `confidence`） |
| `confidence` | Router 原始输出: **`score`** | `region_localization_router.py:391` | **需要**（serving 层 `normalize_region_predictions` 已做） |
| `confidence` | Attribute pipeline: **`score`** | `garment_attribute_pipeline.py` | **需要**（serving 层 `_normalize_attribute_result` 已做） |
| `mask_path` | SAM JSON: **`mask_path`** | `segment_garments_samhq.py:355` | **不需要**（已一致） |
| `mask_path` | instance_schema: **`pred_mask_path`** | `instance_schema.py:97` | 需注意（此 schema 用于不同场景） |
| `mask_present` | Serving 层: 从 `mask_path` 存在性推导 | `subprocess_vision_provider.py` | **不需要**（动态推导） |
| `part_type` | Serving 输出: **`part_type`** | `region_backend.py:normalize_region_predictions()` | **不需要**（serving 层已归一化） |
| `part_type` | Router 原始输出: **`part`** | `region_localization_router.py` | **需要**（serving 层已做映射） |
| `bbox` | YOLO/SAM: **`bbox_xyxy`** | 多处 | **需要**（serving 层已做转换） |
| `topk` | Attribute pipeline: `[{"label": ..., "score": ...}]` | `garment_attribute_pipeline.py` | **不需要**（格式已标准化） |
| `attribute_confidence` | Serving 层输出: **`attribute_confidence`** | `attribute_backend.py:269` | **不需要**（已在输出层统一） |

### 7.2 `run_demo.py` 的归一化状态

**当前 `run_demo.py:normalize_pipeline_output()` 不做字段归一化。** 它只转发 `output_paths` 和 `timing`。`garment_instances` 的字段归一化被推迟（`# ponytail: Full normalization off a re-read of the segmentation JSON is deferred.`）。

QA 结果通过 `asdict(qa_result)` 直接序列化，保留所有原始字段名（包括 `answer_confidence`、`sources` 等）。

### 7.3 需要人工确认的字段差异

1. **`score` vs `confidence`**: Router 原始输出用 `score`，serving 层已做转换。但 `run_demo.py` 不经过 serving 层的 normalizer，直接读 `raw_pipeline_output` → 保留 `score`。
2. **`part` vs `part_type`**: Router 用 `part`，serving 用 `part_type`。`run_demo.py` 不调用 router，所以不影响。
3. **`pred_mask_path` vs `mask_path`**: `instance_schema.py` 用 `pred_mask_path`，但实际 pipeline JSON 用 `mask_path`。两个 schema 并存，互不冲突（`instance_schema.py` 的调用者是 DeepFashion2 GT 评估，不是 `run_demo.py`）。
4. **`all_bboxes` + `all_scores`**: Router 使用两个独立数组，而非 `[{"bbox": ..., "score": ...}]` 对象列表。serving 层将其拆分为独立的 `LocalizedRegion` 记录。

---

## 8. 配置文件和模型权重

### 8.1 所有配置文件清单

| 配置文件 | 用途 | 加载代码 | 重要 key | 是否已路径清理 |
|---|---|---|---|---|
| `configs/category_mapping.yaml` | 13 类→5 类映射 | `predict_garments_yolo.py:285` | `deepfashion2_13cls`, `prd_5cls`, `map_13_to_5` | N/A（无硬编码路径） |
| `configs/model/sam_hq.yaml` | SAM-HQ 模型路径 | `segment_garments_samhq.py` | `model.checkpoint` | ✓（`${SAM_HQ_CHECKPOINT}`） |
| `configs/dataset/deepfashion2.yaml` | 数据集路径 | `sam_box_prompt.yaml` evaluation | `dataset.root` | ✓（`${DEEPFASHION2_ROOT}`） |
| `configs/inference/sam_box_prompt.yaml` | SAM box prompt 推理配置 | 评估脚本 | `dataset.root`, `model.checkpoint` | ✓（两个都已替换） |
| `configs/attribute_inference.yaml` | 8 个属性任务配置 | `task_registry.py:load_inference_config()` | `tasks.<name>.checkpoint`, `label_map`, `region_filter` | N/A（相对路径） |
| `configs/attribute_group_mapping.yaml` | 类别→属性任务映射 | `category_gate.py` | `coarse_class_to_tasks`, `task_to_region` | N/A |
| `configs/attribute_taxonomy.yaml` | 属性标签词汇表 | `attribute_backend.py` | 每个 task 的 class labels | N/A |
| `configs/attribute_templates.yaml` | 属性回答模板 | `attribute_service.py` | `thresholds`, `aliases`, 每个 attr 的模板 | N/A |
| `configs/attribute_eval_targets.yaml` | 属性评估目标 | 评估脚本 | `prd_target: 0.88`，每个 task 的 baseline | N/A |
| `configs/intent_taxonomy.yaml` | 意图分类规则 | `intent_classifier.py` | `intents` 列表，`default_intent` | N/A |
| `configs/knowledge_base.yaml` | 知识库（面料/工艺/风格） | `rag_service.py` | seed documents | N/A |
| `configs/knowledge_schema.md` | 知识库 Schema 文档 | 文档参考 | 字段定义 | N/A |
| `configs/retrieval_config.yaml` | RAG 检索配置 | `rag_service.py` | 分数阈值、BM25 tokenization、类别 boosts | N/A |
| `configs/serving_config.yaml` | 服务层 feature flags | `deps.py`、多个 `_resolve_*_settings()` | `vision.provider`, `region_backend`, `attribute_backend` | N/A |

### 8.2 环境变量

`.env.example`（已路径清理）:

```bash
DEEPFASHION2_ROOT=/home/charlie/datasets/deepfashion2
SAM_HQ_CHECKPOINT=./checkpoints/sam_hq/sam_hq_vit_b.pth
SAM_HQ_REPO=./third_party/sam-hq
FASHION_AI_MODELS=./models
```

**需要人工确认**: YAML 中的 `${DEEPFASHION2_ROOT}` 和 `${SAM_HQ_CHECKPOINT}` 是否被 config loader 支持。如果不支持，需要手动编辑 YAML 或添加 env var 替换代码。

### 8.3 所需模型文件清单

| 模型 | 路径 | 大小 | 必需? | 用途 |
|---|---|---|---|---|
| YOLOv8n 检测器 | `models/detectors/yolov8n_deepfashion2_13cls_best.pt` | ~6MB | **是** | Stage 1: 13 类服饰检测 |
| SAM-HQ ViT-B | `checkpoints/sam_hq/sam_hq_vit_b.pth` | ~360MB | **是** | Stage 2: 实例分割 |
| 关键点预测器 | `outputs/landmark_predictor_resnet18/best.pt` | ~45MB | 半必需 | Stage 3: 39 点关键点（跳过则无 fast-path 定位） |
| Fashionpedia YOLO | `models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt` | ~22MB | 可选 | 3.1.2 Fashionpedia 部位检测 |
| Grounding DINO Tiny | `IDEA-Research/grounding-dino-tiny`（HF 下载） | ~700MB | 可选 | 3.1.2 开放词汇定位 |
| 属性分类器 ×8 | `models/attributes/p2_*_resnet18_*/best.pt` | ~350MB | 可选 | Stage 6: 8 个 ResNet18 分类器 |
| SAM-HQ 代码库 | `third_party/sam-hq/`（git clone） | ~150MB | **是** | SAM 模型定义 |

**注意**: `attribute_backend.py`（serving 层）硬编码的 checkpoint 路径使用 `outputs/` 前缀，与 `attribute_inference.yaml` 的 `models/attributes/` 前缀不同。以 YAML 为准。

---

## 9. Linux 优化层

### 9.1 优化模块清单

| 文件 | 功能 | FP16 | Batched Boxes | TRT | 状态 |
|---|---|---|---|---|---|
| `inference/wrappers/sam_wrapper.py` | SAM-HQ 封装（主 wrapper） | **✓**（默认 `use_fp16=True`） | **✓**（`predict_boxes(N,4)`） | ✗ | **可用但未接入 pipeline** |
| `inference/wrappers/sam_wrapper_factory.py` | 后端切换（pytorch/tensorrt） | — | — | 间接 | 未接入 |
| `inference/wrappers/sam_encoder_trt.py` | TRT FP16 图像编码器 | ✓（TRT FP16） | ✗ | **✓** | **仅 benchmark 参考** |
| `inference/wrappers/sam_hq_trt_wrapper.py` | TRT 编码器 + PyTorch 解码器 | ✓ | ✗ | **✓**（含自动降级） | **仅 benchmark 参考** |
| `inference/wrappers/yolo_wrapper.py` | YOLO 封装 | ✗ | ✗ | TRT = `NotImplementedError` | **仅参考** |
| `inference/env_capture.py` | Benchmark 环境采集（GPU/CPU/OS） | — | — | — | 仅 benchmark |
| `inference/latency_taxonomy.py` | 延迟分层工具 | — | — | — | 仅 benchmark |

### 9.2 FP16 状态

`inference/wrappers/sam_wrapper.py` 的 `SamHqWrapper` 支持 FP16（`use_fp16=True` 默认），通过 `torch.amp.autocast("cuda")` 实现。

**但 `GarmentPipeline.run_source()` 的 Stage 2 不经过这个 wrapper。** 它调用 `segment_garments_samhq.py:run()`，后者使用自己的 `import_samhq()` 加载标准 SAM（无 FP16）。

要启用 FP16，需要修改 `segment_garments_samhq.py` 或 `GarmentPipeline.run_source()` 使其使用优化 wrapper。

### 9.3 Batched Boxes 状态

`inference/wrappers/sam_wrapper.py` 的 `predict_boxes()` 接受 `(N, 4)` 输入并一次推理。benchmark 显示 N=5 时 6.54× 加速。

**同样未接入 pipeline 默认路径。**

### 9.4 TensorRT 状态

TRT 封装（`sam_encoder_trt.py`、`sam_hq_trt_wrapper.py`、`sam_wrapper_factory.py`）:
- **已验证**: 编码器 4.02× 加速（benchmark 报告见 `results/benchmark/`）
- **未接入**: `run_demo.py` 不导入 TRT 模块
- **依赖**: `.engine` 文件（需在目标 GPU 上重新编译）
- **`--sam-backend tensorrt` 标志**: 推迟到 v2

### 9.5 Benchmark 报告位置

| 报告 | 位置 | 内容 |
|---|---|---|
| TRT Encoder Benchmark | `results/benchmark/trt_encoder_bench_report_v2.md` | 4.02× 编码器加速 |
| TRT Validation | `results/benchmark/trt_encoder_validation_summary.md` | 150 张图片精度验证 |
| Pipeline 500 Benchmark | `results/benchmark/pipeline_benchmark_500_report.md` | 420ms/image 分解 |
| Query Region Batch60 | `results/benchmark/query_region_batch60_report.md` | 92% 有效率 |
| Raw TRT Data | `results/benchmark/bench_report_v2.json` | TRT benchmark 原始数据 |

---

### 9.6 SAM-HQ TensorRT / FP16 优化验证结果

> **数据来源**: `results/benchmark/trt_encoder_bench_report_v2.md` + `results/benchmark/trt_encoder_validation_summary.md`  
> **日期**: 2026-07-24  
> **状态**: ⚠️ 已验证, **未默认接入 pipeline** (需 feature flag + shadow testing)

#### 优化范围

- **仅 SAM-HQ image encoder** (prompt encoder 和 mask decoder 仍为 PyTorch)
- YOLO、Fashionpedia、DINO、属性分类器尚未做 TensorRT/ONNX 优化
- 当前生产 pipeline 默认未启用 TRT

#### Benchmark 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| TensorRT | 10.8.0.43 |
| torch | 2.5.1+cu121 |
| 注意 | 部署到不同 GPU 需重建 TensorRT engine 并重新验证 |

#### Encoder-Only Benchmark

| 后端 | 平均延迟 | vs PT FP32 | vs PT FP16 |
|---|---|---|:---:|
| PyTorch FP32 | 309.8 ms | 1.00× | — |
| PyTorch FP16 | 168.9 ms | 1.84× | 1.00× |
| **TensorRT FP16** | **77.1 ms** | **4.02×** | **2.19×** |

#### SAM Wrapper 端到端 Benchmark

| 阶段 | PyTorch FP16 | TensorRT FP16 | 加速比 |
|---|---:|---:|---:|
| set_image | 171.0 ms | 81.3 ms | 2.10× |
| predict_boxes | 8.6 ms | 8.6 ms | 1.00× |
| **总计** | **179.7 ms** | **90.0 ms** | **2.00×** |

> `predict_boxes` 基本不变 (decoder 仍为 PyTorch), 主要收益来自 `set_image`.

#### Pipeline 级 SAM Backend Benchmark

| 阶段 | PyTorch FP16 | TensorRT FP16 | 加速比 |
|---|---:|---:|---:|
| set_image | 258.7 ms | 133.0 ms | 1.94× |
| predict_boxes | 16.6 ms | 15.1 ms | 1.10× |
| **总计** | **275.3 ms** | **148.1 ms** | **1.86×** |

#### 精度验证 (DeepFusion2 150 图 / 234 框)

| 指标 | 值 |
|---|---:|
| TRT fallback | 0 次 |
| Mean IoU | 0.905 |
| **Median IoU** | **0.964** |
| P95 IoU | 0.9996 |
| Median pixel disagreement | 0.13% |
| Mean area ratio (TRT/PT) | 1.0055 |
| masks ≥ 1,000 px median IoU | 0.972 |
| masks ≥ 20,000 px median IoU | 0.978 |

> **说明**: Mean IoU 被小 mask / near-empty case 的 low-IoU tail 拉低. 人工查看可视化 panel 后未发现明显分割失败. PyTorch FP16 是 reference, 不是 ground truth.

#### FP16 / Batched Boxes 状态

| 特性 | 代码位置 | 状态 |
|---|---|---|
| FP16 autocast | `inference/wrappers/sam_wrapper.py` (`use_fp16=True`) | ✅ 可用 (wrapper级) |
| Batched boxes | 同上 (`predict_boxes(N,4)`) | ✅ 可用 (wrapper级) |
| N=5 加速 | — | ~6.54× (来源: benchmark 记录, 待确认出处) |
| 接入 pipeline | `garment_pipeline.py` Stage 2 | ❌ **未接入** (pipeline 用 legacy `segment_garments_samhq.py`) |

#### 当前结论

- TensorRT FP16 SAM encoder **已通过 benchmark + 精度验证**, 可作为 feature-flagged 候选后端
- **不建议**无条件设为默认路径 — 需 pipeline-level validation + shadow testing
- 除非下游 garment metrics 回归, 否则不需要进一步做像素级 encoder parity 优化
- 考虑 dynamic-shape export 消除 padding 开销

---

## 10. `run_demo.py` 当前行为

### 10.1 CLI 参数

```
--image        必需。输入图片路径
--query        可选。自然语言 QA 查询
--output-dir   输出目录（默认 outputs/demo）
--mode         pipeline | qa | full（默认 full）
--no-attributes  跳过 Stage 6 属性推理
--no-visualization  跳过可视化输出（未实现）
--format       json | text（默认 json）
```

### 10.2 导入策略

**所有重量级 import 都是延迟的（函数内 import）**，因此 `import run_demo` 本身不需要 torch。

- `run_pipeline()` 内: `from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig`
- `run_qa()` 内: `from inference.serving.intent_classifier import IntentClassifier` 等
- 模块顶层只 import stdlib（argparse, json, pathlib 等）

### 10.3 Pipeline 调用

```python
config = GarmentPipelineConfig()
config.run_attribute_inference = not args.no_attributes
pipeline = GarmentPipeline(config)
result = pipeline.run_image(image_path=image_path, output_dir=output_dir)
```

使用默认模型路径（`GarmentPipelineConfig` 的默认值）。不传 `--yolo-weights` / `--sam-checkpoint`（推迟到 v2）。

### 10.4 QA 调用

```python
intent = IntentClassifier()
attr_svc = AttributeService()
rag = RagService()
orchestrator = QaOrchestrator(
    intent_classifier=intent,
    attribute_service=attr_svc,
    rag_service=rag,
    vision_provider=None,  # ← mock! 不调用真实 vision pipeline
)
qa_result = orchestrator.answer(query=query)
```

**关键限制**: `vision_provider=None` 意味着 QA 层只能使用 `attributes` 字典中已有的数据，不能从图片中提取新的视觉属性。这意味着 region 查询会因为没有 `localized_regions` 数据而返回"当前问题类型暂不支持"。

### 10.5 输出 JSON 结构

```json
{
    "image_id": "img001",
    "image_path": "/abs/path/to/img001.jpg",
    "mode": "full",
    "timestamp": "2026-07-27T...",
    "warnings": [],
    "errors": [],
    "pipeline": {
        "timing": { "yolo_seconds": 0.12, ... },
        "output_paths": { "detections_json": "...", ... }
    },
    "raw_pipeline_output": { ... },
    "qa": { ... }
}
```

### 10.6 降级行为

| 失败场景 | 行为 |
|---|---|
| Pipeline import 失败（无 torch） | `result.errors` + `result.warnings` 各加一条，不崩溃 |
| Pipeline 运行时异常 | 同上 + `pipeline_error_trace` |
| QA import 失败 | `result.warnings` 加一条，QA 结果为 None |
| `--mode qa` 但无 `--query` | `result.warnings` 加一条 |

### 10.7 当前能做什么 / 不能做什么

**能做的:**
- 运行完整的 Stage 1-6 pipeline（如有所有模型权重）
- 输出 pipeline timing 和各阶段 JSON 文件路径
- 在 `--mode pipeline` 下生成基本的 `result.json`

**不能做的（MVP 限制）:**
- 不调用真实的 vision QA（`vision_provider=None`）
- 不做 `garment_instances` 字段归一化
- 不做 FP16 推理
- 不做 TensorRT 推理
- 不支持 `--yolo-weights` / `--sam-checkpoint` / `--sam-backend` 覆盖
- 不控制可视化输出（`--no-visualization` 标记存在但 `GarmentPipelineConfig` 已有 `save_yolo_vis` 等配置）

---

### 10.8 多模态服务层接入成果

> **数据来源**: 周报 2026-07-16 + 2026-07-24 + 服务层代码验证  
> **状态**: ✅ FastAPI 服务底座就绪, ⚠️ vision_provider 待完整接入

#### 已完成模块

| 模块 | 状态 | 说明 |
|---|---|---|
| FastAPI 服务底座 | ✅ | `inference/serving/app.py` |
| 统一响应结构 `UnifiedResponse` | ✅ | 标准化 JSON 响应 |
| 规则意图识别 `IntentClassifier` | ✅ | 中文规则引擎, 3/3 路由正确 (WSL验证) |
| QA 编排 `QaOrchestrator` | ✅ | 意图→路由→服务调用 |
| 属性问答 `AttributeService` | ✅ | 模板+属性值回答 |
| RAG 检索 `RagService` | ✅ | BM25 + 知识库检索 |
| 商品文案生成 `ContentGenerationService` | ✅ | — |
| VisionProvider 插槽 | ⚠️ | 接口定义就绪, adapter 未完整实现 |
| Error / Metrics | ✅ | — |
| Golden contract 测试 | ✅ | `eval/reports/serving_eval_report.json` |

#### 服务接口

| Endpoint | 功能 | 状态 |
|---|---|---|
| `GET /v1/health` | 健康检查 | ✅ |
| `GET /v1/metrics` | 基础指标 | ✅ |
| `POST /v1/intent/classify` | 意图识别 | ✅ |
| `POST /v1/rag/retrieve` | 知识检索 | ✅ |
| `POST /v1/mm/qa` | 多模态问答 | ⚠️ vision_provider 待接入 |
| `POST /v1/merchant/content/generate` | 商家文案生成 | ✅ |

#### 3.1 能力接入状态 (`/v1/mm/qa`)

| PRD 模块 | 接入方式 | 状态 |
|---|---|---|
| 3.1.1 检测/分割 | YOLO + SAM-HQ (子进程) | ✅ 路径 B |
| 3.1.2 局部区域定位 | Fashionpedia / GroundingDINO | ✅ 封装就绪 |
| 3.1.3 属性识别 | FashionAI 8 任务 ResNet18 | ✅ 封装就绪 |
| 意图识别 | `RuleIntentClassifier` | ✅ |
| 属性问答 | `AttributeService` | ✅ |
| RAG | `RagService` | ✅ |

#### 视觉实例问答支持的问题类型

基于 `garment_instances`:

| 问题类型 | 示例 | 回答依据 |
|---|---|---|
| 数量 | 图里有几件衣服？ | `garment_instances` 数量 |
| 类别 | 图中检测到了什么？ | `category` / `fine_class_name` |
| 存在性 | 有没有上衣？ | 是否存在对应类别 |
| 位置 | 检测框在哪里？ | `bbox` |
| 分割 | 有没有分割结果？ | `mask_present` |

#### 安全字段过滤

服务层输出 `garment_instances` 时仅保留安全字段:
- `instance_id`, `category`, `fine_class_name`, `bbox`, `confidence`, `mask_present`, `mask_ref`
- **不返回**: 原始 mask 位图, 临时文件路径, `image_bytes`

#### Mock / Real Provider 机制

- 真实视觉默认关闭, 需双开关: `provider = real` + `real_enabled = true`
- 任一缺失则回落 mock
- 真实视觉失败时返回结构化 warning (非崩溃)

#### WSL 验证结果 (2026-07-28)

- **意图分类**: 3/3 查询正确路由 ("有几件衣服？"→visual_instance_query, "有没有上衣？"→visual_instance_query, "领子是什么设计？"→attribute_query)
- **回答生成**: 当前使用 smoke wrapper fallback (确定性规则), 因为 `vision_provider` adapter 接口与 `build_vision_context` 不完全匹配
- **WSL 20 图验证**: 核心流水线 100% 通过, QA 环节 fallback 可用

---

## 11. 需要人工确认的检查清单

### 11.1 优先打开的文件

1. `tools/infer/garment_pipeline.py:373-400` — `run_source()` 返回 dict 结构
2. `tools/infer/segment_garments_samhq.py:345-374` — segmentation JSON 字段
3. `src/fashion_vision/schemas/instance_schema.py:31-109` — `build_instance_record()` 字段
4. `src/fashion_vision/localization/region_localization_router.py:51-62` — `locate_region()` 签名
5. `inference/serving/qa_orchestrator.py:333-441` — `answer()` 完整路由
6. `inference/serving/intent_classifier.py` — 规则引擎实现
7. `configs/intent_taxonomy.yaml` — 所有 intent 规则

### 11.2 需要阅读的关键函数/类

- `GarmentPipeline.run_source()` — pipeline 全流程
- `RuleIntentClassifier.classify()` — 意图分类
- `QaOrchestrator._route_visual_instance()` — 视觉实例 QA
- `QaOrchestrator._route_region_query()` — 区域查询 QA
- `QaOrchestrator._route_attribute()` — 属性查询 QA
- `GarmentAttributePipeline.predict_from_json()` — 属性推理
- `FashionAttributeBackend.extract_attributes()` — serving 层属性推理
- `SubprocessVisionProvider.extract_from_path()` — 子进程 pipeline 调用

### 11.3 需要验证的配置

1. `configs/serving_config.yaml` — `vision.provider` 是否设为 `"real"`
2. `configs/serving_config.yaml` — `region_backend.enable_real` 是否设为 `true`
3. `configs/serving_config.yaml` — `attribute_backend.enable_real` 是否设为 `true`
4. `.env.example` 中的路径是否与实际 WSL 环境匹配
5. YAML `${VAR}` 替换是否被 config loader 支持

### 11.4 模型路径检查

```bash
ls models/detectors/yolov8n_deepfashion2_13cls_best.pt
ls checkpoints/sam_hq/sam_hq_vit_b.pth
ls outputs/landmark_predictor_resnet18/best.pt
ls models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt  # 可选
ls models/attributes/p2_collar_design_resnet18_seed2/best.pt        # 8个中的一个
```

### 11.5 Import 检查命令

```bash
# 核心 pipeline import（需要 torch）
PYTHONPATH=. python -c "from tools.infer.garment_pipeline import GarmentPipeline; print('OK')"

# Serving import（无 torch 依赖，但需要 pydantic, fastapi）
PYTHONPATH=. python -c "from inference.serving.intent_classifier import RuleIntentClassifier; print('OK')"

# 优化 wrapper import（需要 torch）
PYTHONPATH=. python -c "from inference.wrappers.sam_wrapper import SamHqWrapper; print('OK')"

# run_demo import（无 torch 依赖）
PYTHONPATH=. python -c "import run_demo; print('OK')"
```

### 11.6 烟雾测试命令

```bash
# 仅 pipeline（需要所有模型权重）
PYTHONPATH=. python run_demo.py --image inputs/demo/img001.jpg --output-dir outputs/demo/

# Pipeline + QA（需要 serving 依赖）
PYTHONPATH=. python run_demo.py --image inputs/demo/img001.jpg --query "领口是什么设计？" --output-dir outputs/demo/

# 仅 QA（需要 serving 依赖，不需要 torch）
PYTHONPATH=. python run_demo.py --image inputs/demo/img001.jpg --query "有几件衣服？" --mode qa
```

### 11.7 预期的失败情况

| 命令 | 预期失败原因 |
|---|---|
| `import GarmentPipeline` | `ModuleNotFoundError: torch`（torch 未安装） |
| `run_demo.py --mode full` | `GarmentPipeline` import 失败（无 torch） |
| `run_demo.py --mode qa` | `IntentClassifier` import 可能需要 pydantic/yaml |
| `import SamHqWrapper` (optimized) | `ModuleNotFoundError: torch`（torch 未安装） |
| QA `vision_provider=None` | Region 查询返回 "当前问题类型暂不支持" |

---

## 12. 已知缺口 / TODO

### 12.1 依赖未安装

- torch, torchvision, ultralytics, segment-anything, transformers, groundingdino
- fastapi, uvicorn, pydantic
- 所有 8 个属性分类器的模型文件

### 12.2 模型权重未链接

- `models/detectors/yolov8n_deepfashion2_13cls_best.pt` — 需下载
- `checkpoints/sam_hq/sam_hq_vit_b.pth` — 需下载
- `outputs/landmark_predictor_resnet18/best.pt` — 需从 D 盘或训练产物的 outputs/ 复制
- `models/attributes/p2_*_resnet18_*/best.pt` ×8 — 需下载
- `third_party/sam-hq/` — 需 `git clone`

### 12.3 FP16 未接入 `run_demo.py`

- 优化 `SamHqWrapper`（`inference/wrappers/sam_wrapper.py`）存在但 `GarmentPipeline` 不用它
- 需要在 `segment_garments_samhq.py` 或 `garment_pipeline.py` 中切换 wrapper

### 12.4 Env Var 替换支持未知

- YAML 中的 `${DEEPFASHION2_ROOT}` / `${SAM_HQ_CHECKPOINT}` 需要 config loader 支持
- 如不支持，需添加 env var 展开代码或使用相对路径

### 12.5 TensorRT 仅 Benchmark

- TRT 封装（`sam_encoder_trt.py`、`sam_hq_trt_wrapper.py`）已验证但未接入 demo
- 需要 `.engine` 文件（GPU 特定）
- `--sam-backend tensorrt` 标志推迟到 v2

### 12.6 完整 pytest 未运行

- 373+ 测试存在于 `tests/` 目录
- 需要安装所有依赖后才能运行
- D drive 上这些测试是通过的

### 12.7 Schema 归一化不完整

- `run_demo.py:normalize_pipeline_output()` 仅转发 `output_paths`，不做字段归一化
- `score` → `confidence`、`pred_mask_path` → `mask_path` 等映射在 `run_demo.py` 中未实现
- Serving 层的 normalizer（`region_backend.py:normalize_region_predictions()` 和 `attribute_backend.py:_normalize_attribute_result()`）已做归一化，但 `run_demo.py` 不经过它们

### 12.8 Serving 层后端默认关闭

- `configs/serving_config.yaml` 中 `region_backend.backend: disabled`、`attribute_backend.backend: disabled`
- QA 层当前只能做 template/字典查找回答，不能做真实的 vision 推理
- 需设置 `enable_real: true` 并加载对应模型

### 12.9 `pyproject.toml` 未创建

- 项目元数据和依赖声明仅在 `requirements.txt` 中
- 正式的 Python 包配置推迟

---

## 13. 当前最佳成果数据总览

> **数据截止**: 2026-07-28  
> **原则**: 同一指标优先选时间更晚、验证规模更大、最终决策采用的版本

### 总览表

| 模块 | 指标 | 当前最佳结果 | 验证规模 | 最终采用方案 | 备注 |
|---|---|---|---|---|---|
| 3.1.1 | Bottoms mAP50 | **95.8%** | 32,153 图 / 13,753 实例 | YOLOv8n 13cls + balanced retrain | bp10_r8_hard |
| 3.1.1 | Tops mAP50 | **91.6%** | 32,153 图 / 20,957 实例 | 同上 | — |
| 3.1.1 | Skirts mAP50 | **90.5%** | 32,153 图 / 6,522 实例 | 同上 | — |
| 3.1.1 | Outwear mAP50 | **83.9%** | 32,153 图 / 2,153 实例 | 同上 | 长尾类, 样本少 |
| 3.1.1 | Dresses mAP50 | **81.5%** | 32,153 图 / 9,105 实例 | 同上 | — |
| 3.1.1 | Overall IoU (分割) | **0.84** | 467 样本 | SAM-HQ ViT-B | 57.4% 在 [0.8, 1.0) |
| 3.1.1 | Pipeline 稳定性 | **500/500 (100%)** | 500 图 / 913 实例 | 全流水线 | 3,220 区域裁剪 |
| 3.1.1 | 端到端延迟 | **420 ms/图** | 500 图 benchmark | — | SAM-HQ 占 70% |
| 3.1.2 | Overall @IoU>0.30 | **51.9%** | 1,127 图 / 1,189 框 / 23 类 | 三路路由 (Fast+FP+DINO-tiny) | — |
| 3.1.2 | sleeve @IoU>0.30 | **88.2%** | 85 标注 | FP YOLO | 最稳定零件 |
| 3.1.2 | sequin @IoU>0.30 | **77.1%** | 48 标注 | DINO (后端切换后) | 从 27% 提升 |
| 3.1.2 | collar+neckline+lapel @IoU>0.30 | **60.6%** | 142 标注 | FP + DINO | 合并评估 |
| 3.1.2 | 在线查询有效响应率 | **92.0%** | 300 queries (60图×5) | Fast Path | 非严格精度 |
| 3.1.2 | Fashionpedia v1 Overall Recall | **48.2%** | 12 parts / 562 图 | v1 (p=1.0, r=12) | v2 退化 1.78pp, 保留 v1 |
| 3.1.2 | DINO-tiny 高频零件 Recall | **46-67%** (collar/pocket/zipper) | 15 samples/part | DINO-tiny (172M) | base 在核心零件退化 |
| 3.1.3 | best F1 (collar_design) | **0.764** | FashionAI 测试集 | ResNet18 | PRD 目标 0.88, 差 0.116 |
| 3.1.3 | pant_length F1 | **0.740** | FashionAI 测试集 | ResNet18 (multiview_v2) | 次优任务 |
| 3.1.3 | 8-task avg F1 | **~0.662** | — | — | 数据量不足 (556-1647/任务) |
| 推理优化 | SAM encoder TRT speedup | **2.19×** vs PT FP16 | 单图 benchmark | TRT FP16 encoder | 仅 encoder, 未接入 pipeline |
| 推理优化 | Pipeline SAM stage speedup | **1.86×** | 单图 benchmark | 同上 | — |
| 推理优化 | TRT vs PT median IoU | **0.964** | 150 图 / 234 框 | 同上 | 无 fallback 事件 |
| 推理优化 | FP16 SAM wrapper speedup | **1.83×** vs FP32 | wrapper benchmark | FP16 autocast | 未接入默认 pipeline |
| 服务层 | `/v1/mm/qa` 接入状态 | **3.1.1+3.1.2+3.1.3 已封装** | — | FastAPI + QaOrchestrator | vision_provider 待完整接入 |
| 服务层 | 意图分类准确率 | **3/3 路由正确** | 3 queries (WSL 验证) | RuleIntentClassifier | 中文规则引擎 |
| 服务层 | WSL 20 图全流程 | **20/20 (100%)** | 20 图 batch | 全流水线 | YOLO+SAM+Landmark+Crop+Attr |

### 关键结论

1. **3.1.1 检测/分割**: 裤子/上衣/裙子 mAP50 ≥ 90%, 外套/连衣裙相对较低 (~82%), 分割 IoU 集中在 0.8-1.0
2. **3.1.2 局部定位**: 三路路由方案可行, sleeve 最稳定 (88.2%), sequin 经后端切换提升最大 (+50pp). Fashionpedia v1 优于 v2, DINO-tiny 优于 base (在核心零件上)
3. **3.1.3 属性识别**: 8 任务均未达 PRD 目标 0.88, 瓶颈是数据量而非模型. iMaterialist Fashion 2019 是候选扩充数据集
4. **推理优化**: TensorRT FP16 已验证 (encoder 2.19×, pipeline 1.86×), FP16 wrapper 已就绪, 均**未接入默认 pipeline**. 需要 feature flag + shadow testing
5. **服务层**: FastAPI 底座就绪, QA orchestrator 路由正确, vision_provider adapter 是最后一个 wiring gap

---

> **文档结束** — 所有文件路径、类名、函数名、字段名均来自实际代码验证。成果数据来自周报、benchmark 报告和 WSL 运行时验证。


## 14. 真实 VisionProvider MVP 接入

> **日期**: 2026-07-28 | **状态**: ✅ MVP 接入完成, QA 层可调用真实 GarmentPipeline 消费归一化视觉输出

### 14.1 接入背景

**之前的状态:**
- 视觉 Pipeline (`GarmentPipeline`) 可离线跑通 (3.1.1 检测/分割 + 3.1.2 区域裁剪 + 3.1.3 属性推理)
- QA 服务框架已就绪 (`QaOrchestrator`, `IntentClassifier`, `AttributeService`, `RagService`)
- `build_vision_context()` 已支持调用 `vision_provider.extract()` 并读取 `VisionAttributeResult`
- 但 `run_demo.py` 中 `QaOrchestrator(vision_provider=None)`, 缺少将真实 Pipeline 输出归一化并注入 QA 层的适配器

**本次接入目标:**
将 `GarmentPipeline.run_image()` 的输出统一归一化为 QA 服务可用的 `VisionAttributeResult`, 使 `QaOrchestrator` 可以基于真实视觉结果回答数量、存在性、检测结果和属性问题。

### 14.2 新增/修改文件

| 文件 | 类型 | 作用 |
|---|---|---|
| `inference/serving/pipeline_vision_provider.py` | **新增** | 真实 VisionProvider MVP, 调用 GarmentPipeline 并归一化输出 |
| `scripts/smoke_test_real_vision_provider.py` | **新增** | 端到端 smoke test: 真实 provider → QaOrchestrator → answer |
| `docs/pipeline_code_walkthrough.md` | 修改 | 新增本节 (Section 14) |

**未修改的文件** (无需改动, 现有接口已兼容):
- `inference/serving/qa_orchestrator.py` — `build_vision_context()` 已支持 `vp.extract()`
- `inference/serving/vision_context.py` — 已正确读取 `vr.garment_instances`, `vr.attributes`
- `inference/serving/vision_provider.py` — 接口已定义, `PipelineVisionProvider` 实现该接口
- `configs/serving_config.yaml` — 配置保留, 可通过 `vision.provider=real` + `mode=full_pipeline` 启用

### 14.3 接入后的数据流

```text
QaOrchestrator.answer(query="有几件衣服？", image="inputs/demo/test_garment.jpg")
  ↓
build_vision_context(vision_provider=pipeline_provider, image=image_path)
  ↓
PipelineVisionProvider.extract(image=image_path)
  ↓
GarmentPipeline.run_image(image_path, output_dir)    ← 全 6 阶段
  ↓
读取输出文件
  ├─ 02_samhq/segmentation_results.json → 合并 01_yolo/detections.json → garment_instances
  │   字段: instance_id, category, fine_class_name, confidence, bbox, mask_present
  ├─ 04_region_crops/region_crops.json → localized_regions
  │   字段: region_id, part_type, component, bbox, crop_path, masked_crop_path
  └─ 06_attributes/predictions.jsonl → attributes (flat by task_name)
      字段: {task_name: {value, attribute_confidence, topk}}
  ↓
返回 VisionAttributeResult
  ├─ .garment_instances = [...]
  ├─ .regions = [...]         (localized_regions)
  ├─ .attributes = {...}      (flat)
  └─ .meta.attributes_by_instance = {...}
  ↓
build_vision_context 构造 VisionContext
  ├─ vc.garment_instances  ← vr.garment_instances
  ├─ vc.localized_regions  ← vr.regions
  └─ vc.effective_attributes ← vr.attributes
  ↓
QaOrchestrator 路由
  ├─ visual_instance_query → _route_visual_instance(vc.garment_instances) → "检测到2件服饰：上衣1件、裤子1件。"
  └─ attribute_query → _route_attribute(effective_attrs) → AttributeService.answer_attribute() → "这件商品采用Invisible设计。"
```

### 14.4 字段归一化说明

| 原始字段 | 归一化字段 | 说明 |
|---|---|---|
| `detections[].bbox_xyxy` | `garment_instances[].bbox` | 统一 bbox 表达 |
| `detections[].confidence` | `garment_instances[].confidence` | 检测置信度 |
| `segments[].mask_path` | `garment_instances[].mask_present` / `mask_ref` | mask 是否存在 + 路径引用 |
| `detections[].coarse_class_name` | `garment_instances[].category` | PRD 5粗类 |
| `crops[].region` | `localized_regions[].part_type` | 局部区域类型 |
| `crops[].bbox_xyxy` | `localized_regions[].bbox` | 区域 bbox |
| `predictions[].attributes[task].label` | `attributes[task].value` / `.label` | 属性值 (双字段兼容) |
| `predictions[].attributes[task].score` | `attributes[task].attribute_confidence` / `.confidence` | 属性置信度 (双字段兼容) |

### 14.5 当前支持的问题类型

| 问题类型 | 示例 | 数据来源 | 状态 |
|---|---|---|---|
| 数量问答 | 有几件衣服？ | `garment_instances` | ✅ 真实 orchestrator |
| 存在性问答 | 有没有上衣？ | `garment_instances[].category` | ✅ 真实 orchestrator |
| 检测问答 | 图里检测到了什么？ | `fine_class_name` | ✅ 真实 orchestrator |
| 分割问答 | 有没有分割结果？ | `mask_present` | ✅ 真实 orchestrator |
| 属性问答 | 领型是什么？ | `attributes[].collar_design` | ✅ 真实 orchestrator + 真实模型 |
| 属性问答 | 领子是什么设计？ | — | ⚠️ 路由到 region_attribute_query (需扩展 intent taxonomy) |

### 14.6 运行方式

```bash
conda activate samhq-trt
cd ~/fashion_final_project

# 视觉实例问答
PYTHONPATH=".:third_party/sam-hq:src" python scripts/smoke_test_real_vision_provider.py \
  --image inputs/demo/test_garment.jpg \
  --query "有几件衣服？"

# 属性问答
PYTHONPATH=".:third_party/sam-hq:src" python scripts/smoke_test_real_vision_provider.py \
  --image inputs/demo/test_garment.jpg \
  --query "领型是什么？"

# 存在性问答
PYTHONPATH=".:third_party/sam-hq:src" python scripts/smoke_test_real_vision_provider.py \
  --image inputs/demo/test_garment.jpg \
  --query "有没有上衣？"
```

### 14.7 验证结果 (WSL samhq-trt, 2026-07-28)

| 验证项 | 结果 | 详情 |
|---|---|---|
| Pipeline 调用 | ✅ 成功 | 全6阶段 (YOLO+SAM+Landmark+Crop+Mask+Attr) |
| garment_instances 读取 | ✅ 2个实例 | short sleeve top (top), trousers (pants) |
| localized_regions 读取 | ✅ 7个区域 | collar/sleeve×2/hem/waist×2/pant_leg |
| attributes 读取 | ✅ 5个任务 | neckline/collar/neck/sleeve/pant |
| "有几件衣服？" | ✅ 成功 | "检测到2件服饰：上衣1件、裤子1件。" |
| "有没有上衣？" | ✅ 成功 | "检测到上衣（检测到1件）。" |
| "领型是什么？" | ✅ 成功 | "这件商品采用Invisible设计。" (conf=0.864) |
| QA 路径 | ✅ 真实 orchestrator | 非 smoke wrapper, 非 mock provider |

### 14.8 当前限制

- **每次 QA 调用重新运行 Pipeline**: `build_vision_context` 会重新调用 `vp.extract()`, 目前无缓存 (MVP 阶段可接受, 后续可加输出目录复用检测)
- **"领子是什么设计？" 路由到 region_attribute_query**: 规则意图分类器将 "领子" 识别为区域词, 需要扩展 intent taxonomy 将 region+attribute 组合查询路由到 attribute_query
- **未默认启用 TensorRT / FP16**: PipelineVisionProvider 使用默认 PyTorch 后端
- **`/v1/mm/qa` 未自动切换**: 需要在 serving config 中设置 `vision.provider=real` 和 `vision.real_provider.mode=full_pipeline`
- **未做生产级优化**: 无显存管理、无请求队列、无并发控制

### 14.9 PPT 可用成果描述

> 完成真实 VisionProvider MVP 接入:
> - 新增 `PipelineVisionProvider` (`inference/serving/pipeline_vision_provider.py`), 将离线 `GarmentPipeline` 接入 QA 服务层
> - 自动读取 YOLO 检测、SAM-HQ 分割、Landmark 定位、区域裁剪和属性推理的全部输出
> - 统一归一化为 `garment_instances` / `localized_regions` / `attributes`, 符合 `VisionAttributeResult` 接口
> - `QaOrchestrator` 成功基于真实视觉结果回答数量 ("2件服饰：上衣1件、裤子1件")、存在性 ("检测到上衣") 和属性 ("Invisible设计", conf=0.864)
> - 3/3 查询通过真实 orchestrator 链路 (非 mock, 非 fallback), 为后续 FastAPI 服务化和生产部署打下基础
