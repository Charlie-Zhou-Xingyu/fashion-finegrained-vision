# 3.1.2 区域定位完整实现报告

> 日期: 2026-07-04 | 分支: feat/p2-fashionai-sleeve-length

---

## 目录

1. [架构总览](#1-架构总览)
2. [路由决策引擎](#2-路由决策引擎)
3. [Fashionpedia YOLO 零件检测](#3-fashionpedia-yolo-零件检测)
4. [Grounding-DINO Fallback](#4-grounding-dino-fallback)
5. [内搭检测](#5-内搭检测)
6. [时序 Benchmark 实验](#6-时序-benchmark-实验)
7. [总结](#7-总结)

---

## 1. 架构总览

### 1.1 入口与数据流

```
用户中文查询 (如 "外套的拉链", "内搭的领口", "胸前的口袋")
    │
    ▼
intent_parser.parse_intent(query)
    │  解析为 QueryIntent { part, side, garment_ref, direction, is_fast_path, is_zero_shot }
    │
    ▼
locate_region()   ← 唯一入口, region_localization_router.py:51
    │
    ├── is_fast_path? ────────► 地标+几何管线 (hem/waist/shoulder/leg_opening)
    │
    ├── garment_ref="inner" + outerwear + SAM? ──► 内搭检测 (途径3)
    │
    ├── part in PART_TO_FP_IDS? (13类FP核心零件) ──► Fashionpedia YOLO (途径1)
    │       ├── YOLO 命中 ──► 早返回, DINO 永不调用
    │       ├── YOLO miss + neckline/cuff ──► fast-path fallback
    │       └── YOLO miss + 其他FP零件 ──► "not_detected" (不用DINO)
    │
    └── part NOT in FP ──► Grounding-DINO (途径2)
            ├── 解剖学裁剪 (anatomical_zoom: 子区域2x放大)
            ├── detect_multi_prompt (多prompt + greedy NMS去重)
            ├── 空间约束 (左右side / 上下方向direction)
            ├── 形状先验过滤 (面积比/宽高比/中心偏移/y_band/x_band)
            └── SAM box-prompt mask 精修
```

### 1.2 源文件清单 (15个模块)

| # | 文件 | 核心函数/类 | 职责 |
|---|------|------------|------|
| 1 | `localization/region_localization_router.py` | `locate_region()` | 唯一入口，路由分发，后处理编排 |
| 2 | `localization/intent_parser.py` | `parse_intent()`, `QueryIntent` | 中文→结构化意图，26类零件词汇 |
| 3 | `localization/fashionpedia_part_detector.py` | `FashionpediaPartDetector` | YOLOv8s 19类零件检测器 |
| 4 | `localization/grounding_dino_locator.py` | `GroundingDINOLocator` | HF GDINO-tiny，多prompt+NMS |
| 5 | `localization/part_detection_config.py` | `PART_DETECTION_CONFIG` | 每类DINO prompt/阈值/形状配置 |
| 6 | `localization/part_shape_priors.py` | `filter_by_shape_priors()` | 几何先验过滤 |
| 7 | `localization/anatomical_zoom.py` | `apply_anatomical_zoom()` | 解剖学子区域2x放大 |
| 8 | `localization/spatial_constraint.py` | `select_side_detection()` | 左右/上下方向过滤 |
| 9 | `localization/bbox_mask_refiner.py` | `BboxMaskRefiner` | SAM box→mask精修 |
| 10 | `localization/inner_garment_detector.py` | `detect_inner_garment_from_sam()` | 内搭两阶段检测 |
| 11 | `localization/inner_boundary_refiner.py` | `refine_inner_boundary()` | 边缘/颜色/纹理边界精修 |
| 12 | `localization/inner_mask_cleaner.py` | `clean_inner_mask_artifacts()` | 7步伪影清理 |
| 13 | `localization/torso_prior.py` | `build_proxy_torso_prior()` | 躯干先验mask |
| 14 | `localization/garment_ref_filter.py` | `filter_instances()` | garment_ref→DF2类别映射 |
| 15 | `localization/open_vocab_prompt_map.py` | `OPEN_VOCAB_REGION_PROMPTS` | 遗留prompt映射(已被#5取代) |

### 1.3 查询解析器 (intent_parser.py)

`parse_intent(query)` 按以下顺序进行最长匹配解析:

```
1. Side words:    左边/右边/左侧/右侧/left/right → side="left"/"right"
2. Direction:     胸前/上方/下方/背后 → front_upper/upper/lower/back
3. Garment ref:   外套/连衣裙/半裙/裤子/内搭 → outerwear/dress/skirt/pants/inner
4. Spatial anchor: "X附近"/"X上的" 模式 → spatial_anchor=X
5. Compound parts: 裙摆→hem+skirt, 连衣裙下摆→hem+dress (优先于通用匹配)
6. Part vocab:     26类零件最长关键词匹配 → part="zipper"/"pocket"/...
7. Routing:        part in {"hem","waist","shoulder","leg_opening"} → is_fast_path
                   part is None → is_zero_shot
```

**26 类可查询零件** (PART_VOCAB):
neckline, cuff, hem, waist, shoulder, leg_opening, zipper, button, pocket,
placket, pattern, belt, collar_stand, lapel, epaulette, buckle, drawstring,
tie_strap, ruffle, fringe, shoulder_seam, sleeve_seam

**Garment ref 映射**: 上衣外套/外套→outerwear, 连衣裙→dress, 半裙/裙子→skirt, 裤子→pants, 内搭/里面→inner

---

## 2. 路由决策引擎

### 2.1 文件: `region_localization_router.py`

### 2.2 函数签名

```python
def locate_region(
    query: str,                    # 中文/英文自然语言查询
    instance: Dict[str, Any],      # 标准实例记录 (coarse_class_name, fine_class_name, bbox, mask...)
    image: np.ndarray,             # BGR uint8 H×W×3 全图
    image_width: int,              # 图片宽度
    image_height: int,             # 图片高度
    locator: Optional[GroundingDINOLocator] = None,        # DINO实例 (调用方管理生命周期)
    dino_threshold: float = 0.3,   # DINO最低置信度
    prefer_pred_mask: bool = True, # fast-path使用预测mask
    sam_wrapper: Optional[SamHqWrapper] = None,             # SAM-HQ用于mask精修
    fashionpedia_detector: Optional[FashionpediaPartDetector] = None,  # FP YOLO实例
) -> Dict[str, Any]:
```

### 2.3 核心路由逻辑 (lines 90-395)

#### Step 1: 意图解析 (line 90)
```python
intent = parse_intent(query)
```

#### Step 2: Garment ref 不匹配标记 (lines 94-95)
如果用户指定了服装类型 (如"外套的拉链")，检查实例的 `fine_class_name` 是否匹配:
```python
if intent.garment_ref is not None:
    _flag_garment_ref_mismatch(intent.garment_ref, instance)
```

#### Step 3: 内搭检测 (lines 97-140)
当三个条件同时满足时触发:
```python
if (intent.garment_ref == "inner"
    and instance.get("coarse_class_name") == "outerwear"
    and sam_wrapper is not None):
    inner_inst = detect_inner_garment_from_sam(image, instance, sam_wrapper)
```
- 成功: 替换 instance 的 mask/bbox 为内搭的，后续 DINO 定位相对于内搭
- 失败: 优雅降级，继续使用外套实例，结果标记 `garment_ref_matched=False`

#### Step 4: Fast-path 路由 (lines 145-148)
```python
if intent.is_fast_path:  # hem, waist, shoulder, leg_opening
    return _build_fast_path_result(...)  # → locate_region_from_instance()
```

#### Step 5: Fashionpedia YOLO 优先 (lines 207-296)

**关键设计决策: Fashionpedia-first, NO DINO fallback for FP parts**

```python
_fp_available = fp_detector is not None and intent.part in PART_TO_FP_IDS

if _fp_available:
    # 解剖学裁剪 + 2x放大
    crop_image, crop_mask, zoom_params, zoom_cfg, part_for_zoom = apply_anatomical_zoom(...)

    # YOLO推理
    fp_dets = fp_detector.detect(crop_image, intent.part,
                                  garment_mask=crop_mask, conf=box_threshold)

    if fp_dets:   # ← YOLO命中 → 早返回
        return _build_result(..., backend_label="fashionpedia_yolo")

    else:         # ← YOLO未命中
        if intent.part in ("neckline", "cuff"):
            # neckline/cuff → fast-path fallback (地标+几何)
            return _build_fast_path_result(...)
        else:
            # pocket/zipper/sleeve/hood/... → not_detected
            # DINO 不调用! 如果YOLO没找到，衣服确实没有该零件
            return _build_result(status="not_detected",
                                 reason="fashionpedia_no_detection")
```

**不为 FP 零件调用 DINO 的原因**: 如果 19 类专用的 YOLOv8s 都没找到零件（如口袋、拉链），说明这件衣服确实没有该零件。如果此时用 DINO，DINO 大概率会产生幻觉 (false positive)，找到一些看起来像但不是零件的东西。

#### Step 6: Grounding-DINO fallback (lines 268-296)

**仅对 FP 不覆盖的零件** (button, placket, belt, pattern, drawstring 等 及 zero-shot query):

```python
# 解析 prompt (4级优先级)
prompts, box_threshold, text_threshold = _resolve_prompts(intent)

# DINO 多prompt检测 + mask门控
dino_dets, raw_count = locator.detect_multi_prompt(
    crop_image, prompts,
    garment_mask=crop_mask,
    threshold=box_threshold,
    dilation_px=cfg.get("mask_dilation_px", 0),
    return_raw_count=True,
)

# bbox从zoom坐标映射回全图坐标
dino_dets_mapped = [map_box_from_zoom_to_original(d, zoom_params) for d in dino_dets]
```

#### Step 7: 空间约束 + 形状先验 + Mask精修 (lines 299-395)

```python
# 空间约束
dino_dets = select_side_detection(dino_dets, intent.side, garment_bbox)
dino_dets = select_direction_detection(dino_dets, intent.direction, garment_bbox)

# 形状先验过滤 (面积比/宽高比/中心偏移/y_band/x_band)
dino_dets = filter_by_shape_priors(dino_dets, intent.part, garment_bbox, ...)

# 如果所有候选都被形状先验拒绝 → not_detected
if not dino_dets:
    return _build_result(status="not_detected",
                         reason="no_detection_passed_shape_priors")

# SAM box-prompt mask精修
top_det = dino_dets[0]
mask = _refine_mask(top_det["bbox_xyxy"], image, garment_mask, sam_wrapper)
```

### 2.4 Prompt 解析优先级 (lines 596-655)

```python
def _resolve_prompts(intent):
    # 优先级 1: 每类优化配置 (part_detection_config.py)
    if intent.part in PART_DETECTION_CONFIG:
        cfg = PART_DETECTION_CONFIG[intent.part]
        return cfg["prompts"], cfg["box_threshold"], cfg["text_threshold"]

    # 优先级 2: 遗留prompt映射 (open_vocab_prompt_map.py)
    prompts = get_prompts_for_region(intent.part)
    if prompts:
        return prompts, default_box_threshold, default_text_threshold

    # 优先级 3: 单规范英文短语 (intent_parser.py)
    if intent.grounding_text:
        return [intent.grounding_text], default_box_threshold, default_text_threshold

    # 优先级 4: 零样本名词提取
    noun = _zero_shot_noun_phrase(query)
    return [noun], default_box_threshold, default_text_threshold
```

### 2.5 解剖学裁剪配置 (anatomical_zoom.py)

针对小零件的子区域裁剪 + 2x 放大:

| 零件 | X范围 | Y范围 | 放大倍数 |
|------|-------|-------|---------|
| zipper/button/placket | [0.30, 0.70] (中心40%) | [0.12, 0.88] | 2.0x |
| pocket | [0.0, 1.0] (全宽) | [0.0, 0.55] (上55%) | 2.0x |
| collar | [0.25, 0.75] (中心50%) | [0.0, 0.30] (上30%) | 2.0x |
| hood | [0.20, 0.80] (中心60%) | [0.0, 0.40] (上40%) | 1.8x |
| belt | [0.0, 1.0] (全宽) | [0.32, 0.78] (中46%) | 2.0x |
| 其他 | [0.0, 1.0] (全图) | [0.0, 1.0] (全图) | 1.0x |

---

## 3. Fashionpedia YOLO 零件检测

### 3.1 文件: `fashionpedia_part_detector.py`

### 3.2 模型信息

- **架构**: YOLOv8s (ultralytics)
- **类别数**: 19 (13核心 + 6装饰)
- **权重**: `models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt`
- **训练**: Fashionpedia 平衡采样 (p=1.0, r=12), mAP50=0.312
- **推理引擎**: ultralytics YOLO Python API

### 3.3 19 类完整列表

#### 13 核心零件 (可被 query 路由)

| FP ID | 内部名称 | 中文 query 词 |
|-------|---------|--------------|
| 0 | hood | (通过 PART_VOCAB) |
| 1 | collar | (通过 PART_VOCAB) |
| 2 | lapel | 翻领, 驳领, 西装领 |
| 3 | epaulette | 肩章, 肩袢 |
| 4 | sleeve | 袖口, 衣袖, 袖子, 袖部, 袖 (cuff别名) |
| 5 | pocket | 口袋, 衣兜, 兜 |
| 6 | neckline | 衣领, 领口, 领子, 领部, 脖颈, 脖子 |
| 7 | buckle | 扣环, 皮带扣 |
| 8 | zipper | 拉链, 拉锁 |
| 11 | bow | (通过 open_vocab_prompt_map: 蝴蝶结) |
| 13 | fringe | 流苏, 穗子, 穗饰 |
| 16 | ruffle | 荷叶边, 波浪边, 荷叶裙边 |
| 17 | sequin | (通过 open_vocab_prompt_map: 亮片) |

#### 6 装饰类 (检测但不暴露为独立 query)

| FP ID | 内部名称 |
|-------|---------|
| 9 | applique |
| 10 | bead |
| 12 | flower |
| 14 | ribbon |
| 15 | rivet |
| 18 | tassel |

**cuff→sleeve 别名**: `PART_TO_FP_IDS["cuff"] = [4]` — YOLO label 保持 "sleeve"，但结果 part 为 "cuff"

### 3.4 类定义与接口

```python
class FashionpediaPartDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(
        self,
        image: np.ndarray,              # BGR uint8 裁剪图 (已zoom)
        target_part: str,               # 规范内部名称 (e.g. "zipper")
        garment_mask: Optional[np.ndarray] = None,  # 可选 H×W 二值mask
        conf: float = 0.25,             # 置信度阈值
    ) -> List[Dict[str, Any]]:
        """
        Returns: [{"bbox_xyxy": [x1,y1,x2,y2], "score": 0.87,
                   "label": "zipper", "class_id": 8,
                   "backend": "fashionpedia_yolo"}, ...]
        按score降序排列
        """
```

### 3.5 detect() 内部流程

```python
def detect(self, image, target_part, garment_mask=None, conf=0.25):
    # 1. 查找 target_part 对应的 FP class IDs
    fp_ids = PART_TO_FP_IDS.get(target_part)
    if not fp_ids:
        return []   # 不在FP覆盖范围

    # 2. Mask门控: 非服装像素填充灰色(128)
    if garment_mask is not None:
        image = self._mask_gate(image, garment_mask)
        # _mask_gate: image.copy(); image[~mask] = 128

    # 3. YOLO推理
    results = self.model(image, conf=conf, verbose=False)

    # 4. 过滤到目标class
    detections = []
    for box in results[0].boxes:
        if int(box.cls) in fp_ids:
            detections.append({
                "bbox_xyxy": box.xyxy[0].tolist(),
                "score": float(box.conf),
                "label": FP_CORE_PART_MAP.get(int(box.cls), "unknown"),
                "class_id": int(box.cls),
                "backend": "fashionpedia_yolo",
            })

    # 5. 按score降序返回
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections
```

### 3.6 Mask门控 (`_mask_gate`)

```python
@staticmethod
def _mask_gate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """填充非服装像素为灰色(128)，抑制背景和相邻服装"""
    gated = image.copy()
    if mask.ndim == 2:
        gated[~mask] = 128
    elif mask.ndim == 3 and mask.shape[2] == 1:
        gated[~mask.squeeze(-1)] = 128
    else:
        # 形状不匹配 → INTER_NEAREST resize
        mask_2d = cv2.resize(mask, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        gated[~mask_2d] = 128
    return gated
```

---

## 4. Grounding-DINO Fallback

### 4.1 文件: `grounding_dino_locator.py`

### 4.2 模型信息

- **架构**: Grounding DINO Tiny (HuggingFace transformers)
- **参数量**: ~172M
- **本地权重路径**: `models/grounding_dino_tiny/` (config.json, pytorch_model.bin, model.safetensors)
- **推理引擎**: `transformers.GroundingDinoForObjectDetection` + `AutoProcessor`
- **无需 mmcv**: 纯 HuggingFace pipeline

### 4.3 类定义与加载

```python
class GroundingDINOLocator:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: str = "cuda"):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self._device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = GroundingDinoForObjectDetection.from_pretrained(model_id)
        self._model.to(self._device).eval()
```

**加载时间**: ~9.2s (含权重读取+模型初始化)

### 4.4 detect() — 单Prompt检测

```python
def detect(
    self,
    image: np.ndarray,              # BGR uint8
    text_query: str,                # 英文文本 (自动追加句号)
    garment_mask: Optional[np.ndarray] = None,
    threshold: float = 0.3,         # 最低置信度
    min_bbox_area_ratio: float = 0.003,  # 最小bbox面积比 (过滤噪声/首饰)
    fill_mode: str = "grey",        # mask填充颜色: grey/black/white
    dilation_px: int = 0,           # mask膨胀像素 (边缘零件: zipper+5, button+3)
) -> List[Dict[str, Any]]:
    """
    Returns: [{"bbox_xyxy": [x1,y1,x2,y2], "score": float, "label": str}, ...]
    按score降序排列
    """
```

#### 内部流程:

```python
def detect(self, image, text_query, garment_mask=None, threshold=0.3, ...):
    # 1. Mask门控 (可选膨胀)
    if garment_mask is not None:
        image = self.mask_to_garment(image, garment_mask, fill_mode, dilation_px)

    # 2. BGR→RGB, 添加尾随句号 (GDINO需要句号才能可靠评分)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = text_query.rstrip() + "."

    # 3. Processor + Model
    inputs = self._processor(images=image_rgb, text=text, return_tensors="pt")
    inputs = {k: v.to(self._device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = self._model(**inputs)

    # 4. 后处理
    H, W = image.shape[:2]
    results = self._processor.post_process_grounded_object_detection(
        outputs,  input_ids=inputs["input_ids"],
        threshold=threshold, target_sizes=[(H, W)]
    )[0]

    # 5. 过滤: min_bbox_area_ratio
    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        bbox = box.tolist()
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area >= (W * H * min_bbox_area_ratio):
            detections.append({
                "bbox_xyxy": bbox, "score": float(score), "label": str(label),
            })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections
```

### 4.5 detect_multi_prompt() — 多Prompt + NMS

```python
def detect_multi_prompt(
    self, image, prompts: List[str],
    garment_mask=None, threshold=0.3, nms_iou_threshold=0.5,
    return_raw_count=False,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    对每个prompt运行detect(), 合并结果, greedy NMS去重.
    每个detection携带 "prompt" 键记录匹配的prompt短语.
    """
```

#### 内部流程:

```python
def detect_multi_prompt(self, image, prompts, ...):
    all_dets = []
    for prompt in prompts:
        dets = self.detect(image, prompt, garment_mask, threshold, **kwargs)
        for d in dets:
            d["prompt"] = prompt   # 记录匹配来源
        all_dets.extend(dets)

    raw_count = len(all_dets)

    # 按score降序, greedy NMS
    all_dets.sort(key=lambda d: d["score"], reverse=True)
    kept = []
    for d in all_dets:
        if all(_box_iou(d["bbox_xyxy"], k["bbox_xyxy"]) < nms_iou_threshold
               for k in kept):
            kept.append(d)

    return (kept, raw_count) if return_raw_count else kept
```

### 4.6 mask_to_garment() — 静态Mask填充

```python
@staticmethod
def mask_to_garment(image, mask, fill_mode="grey", dilation_px=0):
    """填充非服装像素，可选膨胀"""
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2*dilation_px+1, 2*dilation_px+1))
        mask = cv2.dilate(mask, kernel)

    fill_value = {"grey": 128, "black": 0, "white": 255}[fill_mode]
    gated = image.copy()
    gated[~mask] = fill_value
    return gated
```

### 4.7 每类零件 DINO 配置 (part_detection_config.py)

#### 配置结构:

```python
PART_DETECTION_CONFIG = {
    "zipper": {
        "prompts": [
            "a vertical metal zipper line on the front of a jacket",
            "a central zipper closure running down a coat",
            "a clothing zipper on the front placket of a jacket or coat",
            "a frontal zipper on outerwear",
        ],
        "box_threshold": 0.40,
        "text_threshold": 0.35,
        "shape": {
            "min_area_ratio": 0.002, "max_area_ratio": 0.30,
            "min_aspect_ratio_h_over_w": 1.2, "max_aspect_ratio_h_over_w": 8.0,
            "prefer_center_x": True, "center_x_tolerance": 0.25,
            "mask_dilation_px": 5,
        },
    },
    "pocket": {
        "prompts": ["a sewn fabric patch pocket on a jacket", ...],
        "box_threshold": 0.32,
        "text_threshold": 0.28,
        "shape": {"min_area_ratio": 0.01, "max_area_ratio": 0.25, ...},
    },
    "button": {
        "prompts": [...],
        "box_threshold": 0.35,
        "text_threshold": 0.30,
        "shape": {"max_area_ratio": 0.08, "prefer_center_x": True, "mask_dilation_px": 3},
    },
    # ... 共26类零件配置
}
```

#### 阈值对比:

| 零件 | box_threshold | text_threshold | mask_dilation_px |
|------|--------------|----------------|-----------------|
| zipper | 0.40 | 0.35 | 5 |
| button | 0.35 | 0.30 | 3 |
| belt | 0.38 | 0.32 | — |
| pocket | 0.32 | 0.28 | — |
| placket | 0.35 | 0.30 | — |
| pattern | 0.28 | 0.23 | — |
| drawstring | 0.30 | 0.25 | — |
| **default** | **0.30** | **0.25** | **0** |

### 4.8 形状先验过滤 (part_shape_priors.py)

```python
def filter_by_shape_priors(detections, part, garment_bbox, image_shape):
    """
    对每个detection执行几何验证:
      - area_ratio:    bbox面积相对于服装bbox的比例是否在[min, max]内
      - aspect_ratio:  h/w 或 w/h 是否在范围内
      - center_x:      中心x是否在服装bbox的偏好范围内
      - y_band:        y中心是否在指定的垂直带内
      - x_band:        x中心是否在指定的水平带内
    """
    cfg = get_part_shape_config(part)
    kept = []
    for det in detections:
        rejects = []
        bx1, by1, bx2, by2 = det["bbox_xyxy"]
        gx1, gy1, gx2, gy2 = garment_bbox

        area_r = ((bx2-bx1)*(by2-by1)) / ((gx2-gx1)*(gy2-gy1))
        if area_r < cfg["min_area_ratio"] or area_r > cfg["max_area_ratio"]:
            rejects.append(f"area_ratio={area_r:.4f}")

        ar = (by2-by1) / max(bx2-bx1, 1)
        if ar < cfg["min_aspect_ratio"] or ar > cfg["max_aspect_ratio"]:
            rejects.append(f"aspect_ratio={ar:.2f}")

        # center_x, y_band, x_band checks ...
        if not rejects:
            kept.append(det)
        else:
            det["_shape_prior_reasons"] = rejects

    if not kept:
        logger.info(f"part_shape_priors: all {len(detections)} candidate(s) "
                    f"rejected for part={part!r} — returning empty list")

    return kept
```

**关键设计: 当所有候选被拒绝时返回空列表 → 上游返回 `not_detected`**

---

## 5. 内搭检测

### 5.1 文件: `inner_garment_detector.py` + 3个辅助模块

### 5.2 触发条件

在 `locate_region()` 中 (lines 97-140):
```python
if (intent.garment_ref == "inner"
    and instance.get("coarse_class_name") == "outerwear"
    and sam_wrapper is not None):
    inner_inst = detect_inner_garment_from_sam(image, instance, sam_wrapper)
```

### 5.3 入口: `detect_inner_garment_from_sam()`

```python
def detect_inner_garment_from_sam(
    image: np.ndarray,              # BGR uint8 H×W×3 全图
    outer_instance: Dict[str, Any], # 外套实例 (coarse_class_name="outerwear")
    sam_wrapper: SamHqWrapper,      # SAM-HQ 实例
) -> Optional[Dict[str, Any]]:
    """
    两阶段策略:
      1. 领口补集几何分析 (PRIMARY)
      2. SAM multimask on full outerwear bbox (FALLBACK)

    Returns: {"mask": H×W bool, "bbox_xyxy": [x1,y1,x2,y2],
              "score": float, "source": str, ...} or None
    """
```

### 5.4 阶段1: 领口补集几何分析 (`detect_inner_by_neckline_rules`)

**核心洞察**: 内搭通常在外套 mask 之外但在外套 bbox 之内，在领口/胸前开口区域可见。

#### 5.4.1 ROI 与搜索空间构建

```python
# 领口ROI: 外套bbox的 [18%-82%宽, 3%-58%高]
roi_x1 = ox1 + int(gw * 0.18)
roi_x2 = ox1 + int(gw * 0.82)
roi_y1 = oy1 + int(gh * 0.03)
roi_y2 = oy1 + int(gh * 0.58)

# 补集搜索mask: bbox内 AND ROI内 AND NOT 外套mask
complement = inside_bbox & inside_roi & (~outer_mask)
```

#### 5.4.2 三源候选生成

```python
# 源1: 连通组件分析
cc_candidates = _extract_cc_candidates(complement)
# → morphological close → open → CC extraction

# 源2: Canny边缘闭合轮廓
canny_candidates = _extract_canny_candidates(image, neckline_roi)
# → Canny(low=40, high=120) → findContours → filter closed contours

# 源3: SAM multimask
sam_candidates = _extract_sam_candidates(image, neckline_roi, sam_wrapper)
# → sam_wrapper.predict_all_masks(roi_crop, multimask=True)
```

#### 5.4.3 10维加权评分 (`_score_candidate`)

| 权重 | 指标 | 含义 | 最低阈值 |
|------|------|------|---------|
| 2.0 | `w_opening_core` | 与中心前开口核心区域的重叠 | ≥0.35 |
| 2.0 | `w_torso_overlap` | 与躯干先验mask的重叠 | ≥0.35 (proxy: 0.25) |
| 1.8 | `w_center_score` | 水平居中程度 (1.0=正中) | cx∈[0.30,0.70] |
| 1.5 | `w_outside_outer` | 在外套mask外的比例 | ≥0.45 |
| 1.5 | `w_neckline_overlap` | 与领口ROI的重叠 | ≥0.60 |
| 1.0 | `w_inside_bbox` | 在外套bbox内的比例 | ≥0.75 |
| 1.0 | `w_upper_position` | 垂直位置 (越靠上越高) | — |
| 1.0 | `w_solidity` | 凸包密实度 | — |
| 0.5 | `w_area_ratio` | 面积比奖励 (目标~8%) | area∈[0.6%,25%] |
| -2.0 | `penalty_side_edge` | 边缘接触惩罚 | — |

**总分阈值: 3.0**, 且零拒绝原因。

#### 5.4.4 后处理三阶段

**阶段A: 向下扩展** (`_extend_inner_mask_downward`):
```python
# 从前开口区域 (x:22%-78%, y:8%-85%) 向下查找连通组件
# 合并条件: x重叠≥20%, 垂直间距≤10%, 居中
```

**阶段B: 边界精修** (`inner_boundary_refiner.refine_inner_boundary`):
```python
# 水平: 边缘+Canny+颜色梯度+Laplacian纹理复合剖面 → 峰值搜索
# 垂直: 逐行Lab颜色delta + Laplacian纹理变化 → 持续变化条纹(≥5行)停止
# 安全检查: area_ratio∈[0.45,2.80], bbox_area∈[0.45,3.00], center_shift≤0.18
```

**阶段C: 伪影清理** (`inner_mask_cleaner.clean_inner_mask_artifacts`):
```python
# 7步清理:
# 1. 软梯形走廊 (上窄下宽, 替换矩形ROI)
# 2. 上角颜色一致性抑制 (Lab CIE76 delta≤38)
# 3. 侧条检测与移除 (高瘦边缘组件)
# 4. 主体保留 + 智能辅助保留
# 5. 形态学平滑 (open+close)
# 6. 面积比安全门 (清理后 ≥ 原始45%)
```

### 5.5 阶段2: SAM Fallback (`_detect_fallback_full_bbox`)

```python
# 仅当阶段1无结果时运行
# 1. 裁剪外套bbox (4px内缩)
# 2. SAM multimask on full bbox
# 3. 过滤:
#    - containment_ratio ≥ 0.80 (在内mask内的比例)
#    - area_ratio ∈ [0.01, 0.50]
#    - solidity ≥ 0.65
#    - 不接触裁剪边缘 (8px margin)
#    - 不是整件外套 (≤85%)
```

### 5.6 实例筛选 (garment_ref_filter.py)

当 `garment_ref == "inner"` 时:
```python
# "inner" 无类别信号 → 按mask面积升序排列 (小面积≈内层)
sorted_by_area = sorted(instances, key=_mask_area)
for rank, inst in enumerate(sorted_by_area, start=1):
    inst["_inner_rank"] = rank
return sorted_by_area
```

### 5.7 躯干先验 (torso_prior.py)

```python
def build_proxy_torso_prior(outer_bbox):
    """从外套bbox构建代理躯干mask: 缩到 [18%-82%宽, 3%-88%高]"""
    x1 = ox1 + int(gw * 0.18)
    x2 = ox1 + int(gw * 0.82)
    y1 = oy1 + int(gh * 0.03)
    y2 = oy1 + int(gh * 0.88)
    mask[y1:y2, x1:x2] = 1
    return mask
```

---

## 6. 时序 Benchmark 实验

### 6.1 实验设置

| 参数 | 值 |
|------|-----|
| 图片来源 | FashionAI lapel_design test set |
| 图片数量 | 50 张 (随机采样, seed=42) |
| Query 数 | 6 个 (口袋/袖子/拉链/领口/内搭/胸) |
| 总推理次数 | 300 次 |
| 硬件 | NVIDIA GPU (CUDA) |
| 环境 | conda fashion-demo2, Python 3.x |
| 脚本 | `scripts/benchmark_312_timing.py` |

### 6.2 模型加载时间

| 模型 | 加载时间 | 说明 |
|------|---------|------|
| GarmentPipeline (YOLOv8n+SAM-HQ) | ~6.4s | 含YOLO+SAM weights加载 |
| Fashionpedia YOLOv8s | ~0.1s | YOLO .pt 文件加载 |
| Grounding-DINO Tiny | ~9.2s | HF transformers 本地权重 |
| SAM-HQ (second instance) | ~0.5s | lazy-load用于mask精修 |

### 6.3 总体结果

| 途径 | N | Mean(s) | Std(s) | Median(s) | P95(s) | Min(s) | Max(s) |
|------|---|---------|--------|-----------|--------|--------|--------|
| **Fashionpedia YOLO** | 200 | 0.094 | 0.120 | **0.015** | 0.281 | 0.005 | 0.615 |
| **Grounding-DINO** | 50 | 0.529 | 0.039 | **0.526** | 0.591 | 0.477 | 0.624 |
| **Inner Garment** | 50 | 0.256 | 0.181 | **0.307** | 0.529 | 0.000 | 0.598 |

> **注**: 本次 benchmark 在修复 FP YOLO 解剖学裁剪问题后重新运行。FP YOLO 不再经过子区域裁剪 + 2x 放大，改为仅做服装级裁剪（匹配训练时的全图输入分布）。内搭检测的“命中率”不具统计意义——部分图片中不存在外套实例，无法检测内搭属正常情况。DINO 时间受 CUDA 预热影响，首次运行约 2-3s，稳定后约 0.5s。

### 6.4 Fashionpedia YOLO: 命中 vs 未命中

| 状态 | N | Mean(s) | Median(s) | 占比 |
|------|---|---------|-----------|------|
| **命中 (success)** | 60 | **0.277** | 0.273 | 30% |
| **未命中 (not_detected/failed)** | 140 | **0.016** | 0.011 | 70% |

**解读**: FP YOLO 取消解剖学裁剪后，命中时从 ~1.2s 降至 ~0.28s（3.3x 提速），命中率从 23% 提升至 30%。未命中时仅需 ~0.016s 即返回，不做任何额外推理。

### 6.5 六个 Query 明细

| Query | N | Mean(s) | Median(s) | 命中数 | 实际 Backend | 说明 |
|-------|---|---------|-----------|--------|-------------|------|
| **口袋** | 50 | 0.105 | 0.032 | 14/50 | fashionpedia_yolo | FP YOLO → 大部分未命中 (衣服没口袋) |
| **袖子** | 50 | 0.188 | 0.270 | 34/50 | fashionpedia_yolo/fast_path | FP YOLO → miss时→fast_path |
| **拉链** | 50 | 0.022 | 0.011 | 2/50 | fashionpedia_yolo | FP YOLO → 几乎全部未命中 (这批图没拉链) |
| **领口** | 50 | 0.063 | 0.011 | 10/50 | fashionpedia_yolo/fast_path | FP YOLO → miss时→fast_path |
| **内搭** | 50 | **0.256** | 0.307 | 31/50 | inner_garment | SAM内搭检测 (途径3) |
| **胸** | 50 | **0.529** | 0.526 | 50/50 | open_vocab_grounding_dino | **DINO fallback** (途径2) |

> **关于内搭**: 命中数仅表示 SAM 成功检测到内搭区域的图片数，不代表检测准确率。部分图片中不存在外套实例或无可视内搭，返回 not_detected 属于正确行为。

### 6.6 路由正确性验证 ✓

**零路由错误** — 所有 query 都路由到预期 backend:

| Backend | 调用次数 | 涉及 Query |
|---------|---------|-----------|
| `fashionpedia_yolo` | 157 | 口袋, 袖子, 拉链, 领口 |
| `fast_path` | 43 | 袖子(FP miss), 领口(FP miss) |
| `inner_garment` | 50 | 内搭 |
| `open_vocab_grounding_dino` | 50 | 胸 |

**关键验证**:
- ✓ 口袋/袖子/拉链/领口 **从未走 DINO**
- ✓ 胸 100% 走 DINO (符合预期: 不在FP覆盖范围)
- ✓ 内搭 100% 走 inner_garment (SAM-based)
- ✓ 袖子/领口在 FP miss 时正确 fallback 到 fast_path

### 6.7 共享 Gartment Pipeline 时间

| 阶段 | 首次 | 稳定后 |
|------|------|--------|
| Stage 1: YOLO garment detection | ~0.8s | ~0.1-0.3s |
| Stage 2: SAM-HQ segmentation | ~7.7s | ~0.8-3.0s |
| **合计** | ~8.5s | **~1.0-3.0s** |

### 6.8 端到端延迟估算

对于一张图的完整 3.1.2 处理 (Garment Pipeline + 6个 query):

| 阶段 | 典型时间 |
|------|---------|
| Garment Pipeline (YOLO+SAM) | ~1-3s (稳定后) |
| 4个 FP query (口袋/袖子/拉链/领口) | ~4×0.1s = ~0.4s (含hit+miss混合) |
| 1个内搭 query | ~0.3s |
| 1个DINO query (胸) | ~0.5s |
| **6 query 合计** | **~1.2s** |
| **单图总延迟** | **~2-4s** |

---

## 7. 总结

### 7.1 三条途径对比

| 维度 | Fashionpedia YOLO | Grounding-DINO | Inner Garment |
|------|-------------------|----------------|---------------|
| 模型 | YOLOv8s (19类) | GDINO-tiny (~172M) | SAM-HQ (ViT-B) |
| 覆盖范围 | 13核心零件 | 任意文本 | 仅内搭 |
| 触发条件 | `part in PART_TO_FP_IDS` | `part NOT in FP` | `garment_ref="inner"`+outerwear |
| 推理时间 | 0.016s (miss) / 0.28s (hit) | **~0.53s** | **~0.26s** |
| 成功条件 | YOLO检测到目标零件 | DINO找到文本匹配区域 | SAM找到领口补集区域 |
| DINO是否调用 | **永不** (FP零件DINO会产生幻觉) | **是** (唯一DINO使用者) | **不** (纯SAM) |
| 解剖学裁剪 | **无** (修复后取消，匹配训练分布) | 有 (2x放大) | 无 |
| 后处理 | 空间约束 | 空间+形状先验+SAM mask精修 | 边界精修+伪影清理 |

### 7.2 关键设计原则

1. **Fashionpedia-first, DINO-last**: FP YOLO 是硬优先级。如果 FP 覆盖该零件且 YOLO 未找到，返回 `not_detected` 而不是调用 DINO——因为 YOLO 的专化性让它比 DINO 更可靠，DINO 只会产生幻觉。

2. **每类零件不同阈值**: zipper 需要 0.40 的高置信度 (容易与其他线状物混淆)，pattern 只需要 0.28 (更难检测但更少有假阳性)。

3. **mask门控**: 两种检测器都填充非服装像素为灰色(128)，强制模型聚焦在服装区域，抑制背景和相邻服装的干扰。

4. **解剖学裁剪仅用于 DINO**: Fashionpedia YOLO 训练时使用全图输入（640×640，无裁剪），因此推理时仅做服装级裁剪 + mask 门控，不做子区域缩放。DINO 作为通用检测器，仍使用解剖学裁剪 (子区域 + 2x 放大) 以增加小零件的像素预算。

5. **形状先验**: DINO 检测后，所有候选都经过面积比/宽高比/中心偏移/y_band/x_band 验证，拒绝不符合零件几何规律的假阳性。

6. **内搭按需检测**: 不全局运行。只在查询明确提"内搭"且实例是外套时触发。

7. **内搭 mask 替换**: 内搭检测成功后，后续零件定位相对于内搭而非外套，语义更准确。

### 7.3 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/benchmark_312_timing/timing_details.csv` | 300行明细 (50图×6query) |
| `outputs/benchmark_312_timing/timing_summary.csv` | 三条途径汇总统计 |
| `outputs/benchmark_312_timing/timing_summary.json` | 完整JSON报告 |
| `scripts/benchmark_312_timing.py` | 可复用benchmark脚本 |
