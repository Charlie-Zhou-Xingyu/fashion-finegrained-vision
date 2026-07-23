# SAM-HQ Optimization Findings

> 2026-07-23 | RTX 4060 Laptop GPU (7GB VRAM) | SAM-HQ ViT-B

---

## 1. Measured Results

### FP32 vs FP16 autocast (5 images, 10 iterations each)

| Metric | FP32 | FP16 autocast | Speedup |
|--------|------|--------------|---------|
| set_image mean | 266ms | 159ms | **1.68x** |
| predict mean (1-2 boxes) | 9ms | 9ms | ~1x |
| **Total mean** | **275ms** | **168ms** | **1.64x** |
| p95 total | 285ms | 173ms | |
| Mask IoU range | — | 0.981–0.996 | <2% drift |

### Call pattern benchmark (1/2/5 boxes)

| N boxes | FP32 naive (per-box set_image) | FP32 one-set loop | FP32 one-set batch | FP16 naive | FP16 one-set loop | **FP16 one-set batch** | Speedup vs naive |
|---------|------|------|------|------|------|------|------|
| 1 | 304ms | 303ms | 303ms | 200ms | 199ms | **198ms** | 1.53x |
| 2 | 575ms | 311ms | 308ms | 360ms | 203ms | **198ms** | 2.90x |
| 5 | 1,368ms | 337ms | 324ms | 865ms | 227ms | **209ms** | 6.54x |

---

## 2. Main Bottleneck

**`set_image()` (SAM image encoder, ViT-B, 1024x1024) is the bottleneck — 95% of SAM time.**

- `predict()` (mask decoder): ~7-10ms per box — negligible
- `set_image()` (image encoder): ~266ms FP32, ~159ms FP16 — the thing to optimize
- Batched `predict_boxes()` saves on mask decoder for 5+ boxes (36ms -> 20ms FP16), but even then set_image dominates at 189ms

---

## 3. Does Current Pipeline Repeat set_image?

**NO.** Static analysis of `tools/infer/segment_garments_samhq.py:270-317`:

```python
for image_record in images:              # line 270 — per IMAGE
    predictor.set_image(image_rgb)       # line 282 — ONCE per image
    for det in detections:               # line 289 — per GARMENT
        predictor.predict(box=box_np)    # line 306 — ONCE per garment
```

The pipeline already uses the correct pattern: **1 set_image + N predicts**.

However, the call pattern benchmark shows what WOULD happen if it didn't — a 5-garment image would take 1,368ms instead of 337ms (4x slower in FP32, 6.5x slower vs FP16 batched).

---

## 4. FP16 Mask Drift Assessment

Visualization output: `outputs/benchmarks/sam_fp16_vis/`

| Image | IoU | fp32_only (px) | fp16_only (px) | diff_ratio |
|-------|-----|---------------|---------------|------------|
| 000002 | 0.9823 | 3 | 544 | 1.77% |
| 000003 | 0.9814 | 59 | 781 | 1.86% |
| 000004 | 0.9939 | 151 | 6 | 0.61% |
| 000005 | 0.9811 | 907 | 33 | 1.89% |

**Conclusion:** FP16 introduces <2% pixel-level mask drift. The drift is bidirectional — sometimes FP16 produces slightly larger masks, sometimes slightly smaller. For garment segmentation (region crops for collar/sleeve/hem), this is acceptable. The drift is concentrated at mask boundaries where the exact edge matters least for downstream region crops.

---

## 5. Recommended Integration

### Yes, integrate FP16 autocast into the pipeline.

**Method:** Replace `SamPredictor` with `SamHqWrapper(use_fp16=True)` in the SAM segmentation stage.

**Expected impact on a 2-garment image:**
- Current (FP32, one set_image + 2 predicts): ~311ms SAM
- Optimized (FP16, one set_image + batched predict): ~198ms SAM
- **SAM speedup: 1.57x**
- **End-to-end pipeline: 1,240ms → ~1,127ms (1.10x)**

The end-to-end speedup is modest (10%) because SAM is 66% of the pipeline and the other 34% (YOLO + Landmark + Attribute) is unchanged.

### How to integrate

```python
# Current (segment_garments_samhq.py:254-255):
sam = sam_model_registry[args.model_type](checkpoint=str(sam_checkpoint))
sam.to(device=device)
predictor = SamPredictor(sam)

# Optimized replacement:
from inference.wrappers.sam_wrapper import SamHqWrapper
wrapper = SamHqWrapper(checkpoint, model_type=args.model_type,
                       device=device, use_fp16=True)
# Then use wrapper.set_image() / wrapper.predict_boxes()
```

### Risk

- **Mask quality:** <2% pixel drift at boundaries. Acceptable for garment segmentation.
- **FP16 numerical stability:** No NaN/Inf observed in 50+ benchmark iterations across 5 images.
- **CPU fallback:** `SamHqWrapper` auto-disables autocast on CPU.

---

## 6. Next Optimization Steps (Without Changing Model)

Priority order:

| Priority | Optimization | Expected Impact | Effort | Risk |
|----------|-------------|----------------|--------|------|
| P0 | **FP16 autocast** (this round) | 1.64x SAM, 1.10x pipeline | Done | Low |
| P0 | **Batched predict_boxes** (this round) | Minor for 1-2 boxes, 2.3x predict for 5 boxes | Done | None |
| P1 | **Image encoder ONNX/TensorRT** | 266ms → ~80ms (3.3x encoder) | 2-3 days | Medium (SAM dynamic shapes) |
| P1 | **Input resolution A/B (1024/768/640)** | set_image scales ~O(n^2) with resolution | 1 day | Mask quality may degrade |
| P2 | **Image embedding cache** (reuse for multi-query) | 2nd+ query: skip set_image entirely | 1-2 days | Added memory |
| P2 | **torch.compile()** on image encoder | 10-20% on set_image | 1 day | SAM compatibility unknown |
| P3 | **Persistent model server** | Eliminates subprocess cold-start (18s) | 2-3 days | GPU memory management |
| P3 | **YOLO feature sharing** (reuse YOLO features for SAM box prompts) | Theoretical only | Research | Architecture change |

---

## 7. Benchmark Artifacts

| Artifact | Path |
|----------|------|
| FP16 benchmark JSON | `outputs/benchmarks/sam_fp16_20260723_115011.json` |
| Batch boxes benchmark JSON | `outputs/benchmarks/sam_batch_boxes_20260723_115225.json` |
| Call pattern benchmark JSON | `outputs/benchmarks/sam_call_pattern_20260723_120441.json` |
| Static analysis report | `outputs/benchmarks/sam_call_static_analysis.md` |
| Drift visualization | `outputs/benchmarks/sam_fp16_vis/` (4 cases, 12 PNGs) |
| Optimized segmentation path | `inference/optimized/segment_garments_sam_optimized.py` |
| Findings doc | `docs/sam_hq_optimization_findings.md` |
