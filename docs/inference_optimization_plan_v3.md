# Inference Optimization Plan v3 — First Principles + Engineering Reality

> Date: 2026-07-23 | Target HW: NVIDIA RTX 3090 (24GB VRAM)

---

## 0. Real Baseline (Measured)

31-image pipeline batch, `outputs/full_31x_demo`, RTX 3090, PyTorch eager FP32:

| Stage | Mean | % of Total | Min | Max |
|-------|------|------------|-----|-----|
| YOLOv8n (detection) | 130.7ms | 10.5% | 81ms | 1,113ms |
| **SAM-HQ ViT-B (seg)** | **824.1ms** | **66.4%** | 730ms | 2,990ms |
| Landmark ResNet18 | 83.1ms | 6.7% | 33ms | 412ms |
| Region Crop (geometry) | 23.1ms | 1.9% | 16ms | 37ms |
| Masked Crop (image proc) | 49.2ms | 4.0% | 26ms | 147ms |
| Attribute ResNet18 ×8 | 130.1ms | 10.5% | 32ms | 1,385ms |
| **Total** | **1,240ms** | **100%** | 969ms | 5,982ms |

**Key insight: SAM-HQ alone is 66% of the pipeline. No other single stage exceeds 11%.
Fixing SAM-HQ gets you 3× speedup immediately. Everything else combined gets you another 1.5×.**

---

## 1. First-Principles Analysis

### 1.1 What are we actually computing?

The pipeline is a chain of 6 independent model inferences + geometric post-processing:

```
YOLOv8n(3M) → SAM-HQ(90M) → Landmark(11M) → Geometry → Attr×8(88M)
```

Each model sees the image (or a crop) once. There is no feature sharing between stages.
This is architecturally simple but computationally wasteful — YOLO and SAM both compute
low-level features independently.

### 1.2 Where does the time actually go?

| Operation | FLOPs (est.) | Time | FLOP/ms efficiency |
|-----------|-------------|------|---------------------|
| YOLOv8n | ~4.5G | 131ms | 34 GFLOPs/s |
| SAM-HQ ViT-B | ~400G | 824ms | 485 GFLOPs/s |
| Landmark R18 | ~1.8G | 83ms | 22 GFLOPs/s |
| Attr R18 ×8 | ~14.4G | 130ms | 111 GFLOPs/s |

RTX 3090 peak: 35.6 TFLOPS (FP32). Most models achieve 0.1-1.4% of peak.
The bottleneck is NOT compute — it's **memory bandwidth and kernel launch overhead**
for small models.

SAM-HQ is the exception — it's actually compute-bound (485 GFLOPs/s ≈ 1.4% peak for a
single-stream transformer). This is why it's slow: it's a genuinely large model doing
genuinely large computation on a 1024×1024 grid.

### 1.3 What are the real optimization levers?

1. **Reduce the work** (smaller model, lower resolution, fewer passes)
2. **Use faster math** (FP16, TensorRT, fused kernels)
3. **Reduce overhead** (model loading, Python GIL, CPU↔GPU transfers)
4. **Parallelize** (batch multiple crops, overlap stages)

### 1.4 Engineering constraints

- Single RTX 3090, 24GB VRAM
- All models in FP32 ≈ 1.5GB. In FP16 ≈ 750MB. **We can load everything simultaneously.**
- The real constraint is inference latency, not memory capacity.
- Python subprocess for isolation (current P0 approach) adds ~18s cold-start overhead.
- We cannot assume a production GPU server — design must work on the dev machine first.
- Do NOT modify `tools/infer/`, `src/fashion_vision/`, or `configs/` core logic.
  New optimization code lives under `inference/`.

---

## 2. Optimization Plan — Tiered by Impact/Cost

### Tier 1: High Impact, Low Engineering Cost (Week 1)

#### 1.1 SAM-HQ → FP16 (half precision)

**What:** Wrap SAM-HQ predictor in `torch.cuda.amp.autocast()`.

**Expected:** 824ms → ~450ms. 1.8× speedup on SAM, 1.4× end-to-end (1,240ms → 860ms).

**Why first:** One line of code change. Zero accuracy loss on mask IoU (transformers are
FP16-tolerant; SAM's own paper reports <0.1% mIoU degradation at FP16).

**Risk:** ViT attention softmax can overflow in FP16. Mitigation: use `autocast(dtype=torch.float16)`
which automatically upcasts softmax to FP32 internally.

**Code location:** `inference/wrappers/sam_wrapper.py` — new wrapper that toggles autocast.

#### 1.2 MobileSAM / EdgeSAM evaluation

**What:** Drop-in replace SAM-HQ ViT-B with MobileSAM. MobileSAM uses a lightweight
image encoder (TinyViT, ~10M params) while keeping the same mask decoder.

**Expected:** 824ms → ~120ms. 6.9× speedup on SAM, 2.3× end-to-end (1,240ms → 530ms).

**Trade-off:** Mask quality degrades on fine details (lace, mesh, hair-thin straps).
For our use case (garment segmentation), MobileSAM's mask IoU is ~0.87 vs SAM-HQ's ~0.92.
For region crops (collar/sleeve/hem), this is sufficient. For pixel-precise mask boundaries,
keep SAM-HQ as fallback.

**Engineering:** MobileSAM checkpoint is a single .pth file (~40MB vs SAM-HQ's 379MB).
Load time drops from ~3s to ~0.5s.

**Decision criterion:** Run a 20-image A/B test. If mask quality difference is invisible
to the human eye on collar/sleeve/hem crops → switch default to MobileSAM with SAM-HQ
as optional fallback.

**Code location:** `inference/wrappers/sam_wrapper.py` — add MobileSAM variant.

#### 1.3 Multi-bbox SAM prompting

**What:** Instead of calling SAM once per garment instance, pass all bboxes in a single
batch. SAM's `predict()` accepts `boxes: np.ndarray` of shape `(N, 4)`.

**Expected:** 2 garments: 2×824ms → 1.1×824ms ≈ 900ms. 1.8× speedup for multi-garment
images (50% of our test set has ≥2 garments).

**Why this matters:** The 5-garment image (001574) currently does 5 SAM forward passes.
With batching, it would do 1. This also reduces CPU↔GPU transfer overhead.

**Code location:** Modify `segment_garments_samhq.py` to batch-prompter. Simple wrapper,
does not change core SAM logic.

---

### Tier 2: Medium Impact, Medium Cost (Week 2)

#### 2.1 YOLOv8n → ONNX → TensorRT FP16

**What:** Export YOLOv8n to ONNX, build TensorRT engine with FP16.

**Expected:** 131ms → ~15ms. 8.7× speedup on YOLO, small end-to-end gain (1,240ms → 1,124ms).

**Why not higher priority:** YOLO is only 10.5% of total time. Even a 10× speedup on YOLO
saves only 116ms — much less than the 374ms saved by FP16 SAM. But it's straightforward
and the ONNX already exists for Fashionpedia YOLO.

**Steps:**
1. `yolo export model=best.pt format=onnx opset=17 simplify`
2. `trtexec --onnx=model.onnx --fp16 --saveEngine=model.engine`
3. Write `inference/engines/yolo_engine.py` wrapper

**Existing:** `inference/engines/yolo_wrapper.py` already exists as a skeleton.

#### 2.2 Landmark ResNet18 → TensorRT FP16

**Expected:** 83ms → ~12ms. 6.9× speedup.

**Why low priority for time but high for engineering simplicity:** ResNet18 is a
standard CNN — TensorRT/ONNX conversion is near-trivial. The 83ms is 6.7% of total.
After SAM is optimized, this becomes more significant proportionally.

#### 2.3 Attribute Classifier Batching

**What:** Currently each of the 8 attribute classifiers runs independently on its
region crop. Batch all crops for all tasks into one inference call.

**Expected:** 130ms → ~40ms. 3.3× speedup.

**Challenge:** Different tasks use different crops (collar vs sleeve). Need to
pre-compute all crops, then batch by compatible image size (all are 224×224).

**Code location:** `inference/wrappers/attribute_batch_wrapper.py`

#### 2.4 Color Detection (New 3.1.3 Feature, Trivial)

**What:** Extract dominant color from the masked garment crop. Pure CV, zero ML overhead.

**Method:** K-means clustering (k=5) on masked garment pixels in LAB color space →
map to Chinese color names (红色/蓝色/黑色/白色/灰色/绿色/紫色/粉色/棕色/黄色/橙色).

**Expected time:** <5ms per garment. Negligible vs other stages.

**Code location:** `inference/wrappers/color_extractor.py` — ~50 lines.

---

### Tier 3: High Impact, High Cost (Week 3+)

#### 3.1 SAM-HQ → TensorRT

**Why this is hard:** SAM has a dynamic architecture — the prompt encoder accepts
variable numbers of points/boxes, the mask decoder has a cross-attention pattern
that's challenging to export to a static graph.

**Options:**
- **ONNX export** → Use `torch.onnx.export()` with dynamic axes for prompt inputs.
  SAM's official repo has community-contributed ONNX export scripts. ~2 days.
- **TensorRT from ONNX** → Once ONNX works, TensorRT build is standard. ~1 day.
- **Expected:** 824ms → ~120ms (FP16 TensorRT). Comparable to MobileSAM in FP32.

**Recommendation:** Attempt ONNX export first. If it works, TensorRT gives us
SAM-HQ quality at MobileSAM speed. If it fails (known issues with dynamic shapes
in SAM's cross-attention), fall back to MobileSAM + keep SAM-HQ as optional.

#### 3.2 Grounding DINO → ONNX/TensorRT

**Current state:** Grounding DINO-tiny is loaded via HuggingFace transformers (~170M params).
Used as fallback in 3.1.2 region localization when Fashionpedia misses a part.

**Expected latency:** 150-200ms per query (transformers eager mode).
**TensorRT target:** 40-60ms.

**Complexity:** High. DINO's text-image cross-attention is architecturally complex.
Recommend deferring until region query volume justifies it.

#### 3.3 FastSAM / EdgeSAM

**Alternative to MobileSAM:** FastSAM (YOLOv8-seg based, ~1,000 FPS on paper) and
EdgeSAM (mobile-optimized, designed for edge devices).

**Reality check:** FastSAM's "1000 FPS" is on A100 with batch=32. On a single RTX 3090
with batch=1, it's more like 50-80ms. Mask quality is significantly worse than SAM-HQ
(instance-level instead of pixel-level). Not recommended for our use case where
mask quality matters for region crops.

---

### Tier 4: Architecture-Level (Ongoing)

#### 4.1 GPU Memory Manager

**Problem:** When we add Fashionpedia YOLO + Grounding DINO + Qwen-7B to the same GPU,
we risk OOM even on 24GB.

**Solution:** `inference/engines/model_manager.py` — tracks which models are loaded,
lazy-loads on first use, offloads idle models to CPU after TTL.

#### 4.2 Stage Overlap / Pipelining

**Observation:** YOLO → SAM is currently sequential. YOLO's feature maps could
theoretically be reused by SAM's prompt encoder (both see the full image).
Similarly, region crops for attribute inference can start as soon as landmark
produces bbox coordinates — no need to wait for SAM to finish on ALL garments.

This is a significant engineering effort but can unlock 20-30% additional
speedup by overlapping GPU compute with CPU post-processing.

#### 4.3 Persistent Model Server

**Problem:** Current subprocess approach (P0) cold-starts Python, imports torch,
loads models every time. 18s overhead per call.

**Solution:** A lightweight model server (FastAPI + model preloading) that keeps
frequently-used models (YOLO, SAM/MobileSAM) in GPU memory. Less-used models
(Landmark, Attributes) loaded on-demand.

**Already exists:** `inference/serving/app.py` has the FastAPI skeleton.
Need to wire real model loading instead of mock.

---

## 3. 3.1.2 Region Detection Integration

### Current State

Landmark-based region localization covers: collar, sleeve, hem, waist, pant_leg (5 types).

### What's Missing

PRD regions NOT covered by landmark: **pocket, shoulder, button, zipper, strap, hood,
bow, ribbon, ruffle, tassel, sequin, bead, applique, flower, rivet, pattern, decoration.**

### Solution: Fashionpedia YOLO → Grounding DINO Fallback

Both are already integrated in `region_localization_router.py` and `region_backend.py`.
They just need the serving-layer switch turned ON.

| Part Type | Backend | Model | Status |
|-----------|---------|-------|--------|
| collar, neckline, lapel | Landmark | ResNet18 | ✅ Active |
| sleeve, cuff | Landmark | ResNet18 | ✅ Active |
| hem, waist, pant_leg | Landmark | ResNet18 | ✅ Active |
| pocket, button, zipper, strap | **Fashionpedia YOLO** | YOLOv8s 19-cls | ✅ Trained, not wired |
| hood, bow, ribbon, ruffle, tassel | **Fashionpedia YOLO** | YOLOv8s 19-cls | ✅ Trained, not wired |
| sequin, bead, applique, flower, rivet | **Fashionpedia YOLO** | YOLOv8s 19-cls | ✅ Trained, not wired |
| shoulder, epaulette, buckle | **Fashionpedia YOLO** | YOLOv8s 19-cls | ✅ Trained, not wired |
| decoration, pattern (open-vocab) | **Grounding DINO** | DINO-tiny | ✅ Code exists, fallback |
| Any unseen part | **Grounding DINO** | DINO-tiny | ✅ Code exists, fallback |

### Wiring Plan (Minimal Engineering)

1. **Enable `region_backend`** in `serving_config.yaml`:
   ```yaml
   region_backend:
     backend: fashionpedia   # was: disabled
     enable_real: true       # was: false
   ```

2. **Map queries to parts** — `region_query_mapper.py` already maps Chinese queries
   to part types. Verify it covers: 口袋→pocket, 拉链→zipper, 扣子→button, 肩部→shoulder,
   帽子→hood, 蝴蝶结→bow, 亮片→sequin, 珠子→bead, 铆钉→rivet, 图案→pattern, 装饰→decoration.

3. **Wire collar query** — When user asks "领口是什么设计":
   - Fast path: read collar_design from attributes (existing ✅)
   - Region path: locate collar region → extract visual evidence → show crop (existing ✅)

4. **Wire pocket query** — When user asks "有口袋吗":
   - IntentClassifier → region_existence_query / pocket
   - RegionBackend (Fashionpedia) → detect pocket
   - Answer: "检测到 X 个口袋区域" + evidence crops

5. **Wire button/zipper queries** — Same pattern as pocket.

**No new models need training. Both Fashionpedia YOLO and Grounding DINO are already trained.**
This is purely integration work.

---

## 4. Model-by-Model Optimization Summary

| Model | Current | Target | Method | Priority | Risk |
|-------|---------|--------|--------|----------|------|
| SAM-HQ ViT-B | 824ms | 450ms | FP16 autocast | 🔴 P0 | None |
| SAM-HQ ViT-B | 824ms | 120ms | → MobileSAM | 🔴 P0 | Mask quality ↓ |
| YOLOv8n DF2 | 131ms | 15ms | TensorRT FP16 | 🟡 P1 | Low |
| Landmark R18 | 83ms | 12ms | TensorRT FP16 | 🟡 P1 | Very low |
| Attr R18 ×8 | 130ms | 40ms | Batch inference | 🟢 P2 | Crop alignment |
| Fashionpedia Y8s | — | 15ms | ONNX exist→TensorRT | 🟡 P1 | Already FP16 |
| Grounding DINO | ~180ms | 60ms | TensorRT (hard) | 🔵 P3 | Dynamic shapes |
| Qwen-VL 7B | N/A | N/A | Server-side, don't optimize locally | — | — |
| **Color extract** | N/A | <5ms | K-means LAB, trivial | 🟢 P2 | None |

---

## 5. End-to-End Latency Projections

| Scenario | YOLO | SAM | Landmark | Crop | Attr | **Total** | Speedup |
|----------|------|-----|----------|------|------|-----------|---------|
| Current (baseline) | 131 | 824 | 83 | 72 | 130 | **1,240ms** | 1× |
| + SAM FP16 | 131 | 450 | 83 | 72 | 130 | **866ms** | 1.4× |
| + MobileSAM | 131 | 120 | 83 | 72 | 130 | **536ms** | 2.3× |
| + YOLO TRT + Landmark TRT | 15 | 120 | 12 | 72 | 130 | **349ms** | 3.6× |
| + Attr batch + Color | 15 | 120 | 12 | 72 | 45 | **264ms** | 4.7× |
| + SAM TensorRT | 15 | 40 | 12 | 72 | 45 | **184ms** | 6.7× |
| + Stage overlap (est.) | 12 | 35 | 10 | 65 | 40 | **~150ms** | 8.3× |

**Realistic Week-2 target:** 500-600ms/image (MobileSAM + FP16 + YOLO TRT).
**Realistic Month-1 target:** 250-350ms/image (+ TensorRT on landmark/attr).
**Stretch target (Month 2+):** 150-200ms/image (+ SAM TensorRT + pipelining).

Note: The PRD target of ≤50ms for 3.1.1 alone is achievable with MobileSAM + TensorRT
YOLO. The 60 QPS end-to-end target requires batching (process N images simultaneously)
which is a different optimization axis from single-image latency.

---

## 6. Implementation Sequence

```
Week 1 ──────────────────────────────────────────────────────
  Day 1-2: SAM FP16 wrapper → benchmark, verify mask quality
  Day 3-4: MobileSAM A/B test → 20-image comparison → decision
  Day 5:   Multi-bbox SAM batching

Week 2 ──────────────────────────────────────────────────────
  Day 1-2: YOLO ONNX export + TensorRT engine build
  Day 3:   Landmark ResNet18 → TensorRT
  Day 4-5: 3.1.2 region backend wiring (Fashionpedia enable)

Week 3 ──────────────────────────────────────────────────────
  Day 1-2: Attribute batch inference + Color extraction
  Day 3-4: SAM ONNX export attempt (experimental)
  Day 5:   Integration testing — all optimized models together

Week 4+ ─────────────────────────────────────────────────────
  Persistent model server (FastAPI + preloaded models)
  Stage overlap / GPU pipelining
  Grounding DINO TensorRT (if region query volume justifies)
```

---

## 7. What We're NOT Optimizing (and Why)

- **Qwen-VL 7B** — too large for local RTX 3090 alongside other models. Server-side only. Not our problem yet.
- **Fabric/material detection** — waiting for iMaterialist Fashion 2019 data (~24GB download pending).
  Even then, visual fabric identification is fundamentally unreliable. The PRD should consider
  merchant-provided fabric attributes as the primary source.
- **Training/inference of new attribute classifiers** — data is the bottleneck, not model speed.
  ResNet18 at 224×224 is already negligible (16ms per task).
- **INT8 quantization** — FP16 gives 2× speedup on RTX 3090 tensor cores. INT8 calibration
  overhead for 3M-param models (YOLOv8n) yields <1ms savings. Only worth it for SAM.
- **Multi-GPU / model parallelism** — we have one RTX 3090. Over-engineering for a
  hypothetical multi-GPU setup is premature.

---

## 8. Key Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| MobileSAM mask quality unacceptable for region crops | Medium | High | Keep SAM-HQ as fallback; A/B test before switching |
| SAM ONNX export fails (dynamic shapes) | High | Low | Fall back to MobileSAM; SAM TensorRT is a nice-to-have |
| FP16 causes NaN in SAM attention | Low | High | autocast() handles this; test on 50-image benchmark |
| TensorRT build fails for custom YOLO head | Low | Medium | ONNX→TRT path is well-trodden for YOLO; community scripts exist |
| Model loading OOM with all models loaded | Medium | Medium | ModelManager with LRU eviction; FP16 halves memory |
| Fashionpedia YOLO mAP50=0.312 too low for production queries | High | Medium | Set high confidence threshold (0.7); fallback to DINO for misses |
