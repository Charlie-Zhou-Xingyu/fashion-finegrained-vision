# 3.1.2 Region Localization Accuracy Improvement Plan v2

> Date: 2026-07-09
> Author: CharlieZhou + Claude
> Context: eval_v2 overall 51.9%@IoU>0.3 (936 images, 1180 annotations, 23 parts).
> Mentor feedback incorporated (2026-07-09).
> DINO-base NOT available locally — deferred until model is downloaded.

---

## 0. Current Accuracy Baseline

| IoU | Overall | Best Part | Worst Part |
|-----|---------|-----------|------------|
| >0.01 | 66.7% | ribbon 97.4% | ruffle 34.9% |
| >0.15 | 57.8% | sleeve 88.2% | ruffle 32.6% |
| >0.30 | 51.9% | sleeve 88.2% | ruffle 25.6% |

**Key findings from mentor revision (2026-07-09):**
- DINO prompt optimization ceiling reached — 17 variants tested, no breakthrough.
- **button: DINO raw recall 70%, eval accuracy 46%** → shape_priors are killing 24% of valid detections.
- **rivet/shoes: DINO-tiny capacity bottleneck** — confirmed by query A/B testing.
- Collar+neckline+lapel merged: 60.6%@IoU>0.3 (142 annotations).

---

## 1. Approved Tasks Summary

| # | Task | Axis | Status | Effort |
|---|------|------|--------|--------|
| T1 | Shape prior calibration from eval_v2 data | 4 | ✅ Approved | 2h dev |
| T2 | Soft NMS + class-aware IoU thresholds | 4 | ✅ Approved | 1h dev |
| T3 | Wire all_bboxes into pipeline output | 4 | ✅ Approved | 1h dev |
| T4 | Multi-scale TTA with config toggle (default OFF) | 2 | ✅ Approved with constraints | 3h dev |
| T5 | Higher DINO input resolution (configurable) | 2 | ✅ Approved with constraints | 1h dev |
| T6 | Model ensemble (FP YOLO+DINO overlap → avg+intersection) | 1 | ✅ Approved | 2h dev |
| T7 | WBF merge for DINO-only parts (replace greedy NMS) | 1 | ✅ Approved | 1h dev |
| T8 | FP YOLO re-train with p=0.6 r=6 (keep old weights) | 1 | ✅ Approved | 4h GPU + 1h dev |
| T9 | Expand Label Studio annotations for weak parts | 3 | ✅ Approved | 6h human |
| — | Pipeline architecture changes (Axis 5) | 5 | ❌ Deferred | — |
| — | Two-stage coarse-to-fine | 5 | ❌ Deferred | — |
| — | Confidence calibration (Platt scaling) | 5 | ❌ Deferred | — |
| — | Uncertainty-aware routing | 5 | ❌ Deferred | — |

---

## 2. Task Details

### T1: Shape Prior Calibration from eval_v2 Data [P0]

**Why:** button DINO raw recall 70% → eval accuracy 46% = shape_priors killing 24% of valid detections. All shape prior thresholds are author estimates. Must replace with data-driven thresholds.

**Files:**
- NEW: `scripts/calibrate_shape_priors_from_eval.py`
- MODIFY: `src/fashion_vision/localization/part_detection_config.py` (update `shape` entries from calibrated values)
- NEW: `configs/shape_priors_calibrated_v1.yaml` (calibrated config, human-reviewable)

**Key logic:**
```python
# For each part, from eval_v2 per_result.jsonl:
# 1. Filter TP detections (best_iou > 0.01)
# 2. Compute area_ratio, aspect_ratio_hw, aspect_ratio_wh, center_offset
# 3. Set thresholds at μ ± 2σ (or [P5, P95] for bounded values)
# 4. Output YAML with old vs new thresholds side by side
# 5. Flag parts where μ ± 2σ would be WIDER than current — these need manual review
```

**Verification:** Run `scripts/eval_validation_v2.py` with calibrated config. Expected: button accuracy improves (46% → 55%+). No part should regress >2%.

**Rollback:** `part_detection_config.py` preserves old thresholds under `_v1_backup` key in the config dict for 30 days. Manual revert path: `git checkout` the config file.

**Dependencies:** eval_v2 per_result.jsonl (exists at `data/validation/eval_v2/per_result.jsonl`).

---

### T2: Soft NMS + Class-Aware IoU Thresholds [P0]

**Why:** Greedy NMS at IoU 0.5 kills adjacent valid detections (e.g., two pockets side by side, multiple buttons in a row). Soft NMS decays scores instead of removing boxes.

**Files:**
- MODIFY: `src/fashion_vision/localization/grounding_dino_locator.py` (`_nms()` method)
- MODIFY: `src/fashion_vision/localization/fashionpedia_part_detector.py` (add post-NMS for multi-instance parts)

**Key changes:**
```python
# grounding_dino_locator.py — replace _nms() with soft_nms()
@staticmethod
def _soft_nms(detections: list[dict], iou_threshold: float = 0.5,
              sigma: float = 0.5, score_threshold: float = 0.001) -> list[dict]:
    """Soft-NMS: decay scores by Gaussian penalty instead of removing boxes."""
    # Algorithm:
    # for each box in sorted order:
    #   for each remaining lower-score box:
    #     iou = compute_iou(box, lower_box)
    #     lower_box.score *= exp(-iou^2 / sigma)
    #   if lower_box.score < score_threshold: remove
    # return boxes with score > 0 (sorted)

# Class-aware IoU thresholds for multi-instance parts:
MULTI_INSTANCE_IOU = {
    "pocket": 0.30,   # two pockets at same y-level → keep both
    "button": 0.20,   # buttons in a row → keep all
    "rivet": 0.15,    # many rivets close together
    "bead": 0.15,
    "sequin": 0.10,   # sequins cluster densely
}
# Default: 0.50 (unchanged)
```

**Verification:** Run eval_v2 with `--soft-nms` flag. Compare per-part accuracy — pocket/button should improve (two detections → both can match separate GT boxes).

**Rollback:** Keep old `_nms()` as `_greedy_nms()` (rename, don't delete). Router selects via `nms_mode` parameter: `"greedy"` (default) or `"soft"`. A 1-line change reverts.

**Dependencies:** None.

---

### T3: Wire all_bboxes into Pipeline Output [P0]

**Why:** Router already stores `all_bboxes` in result dict, but downstream consumers (garment_pipeline, visualization) only use `bbox_xyxy` (top-1). Multi-instance parts (pockets, buttons) are under-reported.

**Files:**
- MODIFY: `src/fashion_vision/localization/region_localization_router.py` (already returns `all_bboxes` — verify)
- MODIFY: `tools/infer/garment_pipeline.py` (consume `all_bboxes` for region crop generation)
- MODIFY: `scripts/eval_validation_v2.py` (use all pred_bboxes, not just top-1, for hit calculation)

**Key change in eval_validation_v2.py:**
```python
# Current: only pred_bboxes = [top_1]
# New: pred_bboxes = all detections returned by backend
# Hit check: ANY pred bbox is a hit → annotation is a hit
```

**Verification:** eval_v2 pocket/button accuracy should increase (second pocket/button now matches its own GT annotation).

**Rollback:** Trivial — the old `bbox_xyxy` field is preserved. Downstream code reads `all_bboxes` only when available.

**Dependencies:** T2 (soft NMS produces more kept boxes → more boxes in all_bboxes).

---

### T4: Multi-Scale TTA with Config Toggle [P1]

**Why:** Small parts (buttons ~2-5% of garment area) have insufficient pixel budget. Multi-scale inference gives more pixels to resolve fine detail — confirmed by research: +1.8 AP from scale TTA alone.

**Constraints (mentor feedback):**
- Default: TTA **OFF**.
- Per-part whitelist: only small parts (button, rivet, zipper, buckle, drawstring).
- Configurable time budget: max `tta_timeout_ms` (default 500ms).
- Latency logging: log per-request TTA time vs baseline in debug dict.

**Files:**
- MODIFY: `src/fashion_vision/localization/grounding_dino_locator.py` (new `detect_multiscale()` method)
- MODIFY: `src/fashion_vision/localization/region_localization_router.py` (TTA routing logic)
- MODIFY: `configs/attribute_inference.yaml` (new TTA config section)

**Config additions:**
```yaml
# configs/attribute_inference.yaml — new section
localization:
  tta:
    enable_multiscale_tta: false        # master toggle
    tta_parts:                          # per-part whitelist
      - button
      - rivet
      - zipper
      - buckle
      - drawstring
    scales: [1.0, 2.0, 3.0]            # crop scale factors
    merge_method: "wbf"                 # wbf | nms
    timeout_ms: 500                     # max additional latency per request
    log_latency: true                   # write tta_time_ms to debug dict
  dino_input_resolution:
    enable_hires: false                 # force-resize to target_w × target_h
    hires_parts:                        # per-part whitelist
      - button
      - rivet
    target_size: [800, 800]             # target (w, h)
```

**Implementation:**
```python
# grounding_dino_locator.py
def detect_multiscale(self, image, prompts, garment_mask=None,
                      scales=[1.0, 2.0, 3.0], merge="wbf", **kwargs):
    """Run detect() at multiple crop scales, merge results with WBF."""
    all_dets = []
    for scale in scales:
        if scale == 1.0:
            dets = self.detect(image, prompts[0], garment_mask, **kwargs)
        else:
            # Center-crop to 1/scale, resize up to original → zoomed view
            h, w = image.shape[:2]
            ch, cw = int(h / scale), int(w / scale)
            cy1, cx1 = (h - ch) // 2, (w - cw) // 2
            zoomed = cv2.resize(image[cy1:cy1+ch, cx1:cx1+cw], (w, h))
            dets = self.detect(zoomed, prompts[0],
                              garment_mask[cy1:cy1+ch, cx1:cx1+cw] if garment_mask is not None else None,
                              **kwargs)
            # Remap bboxes back to original coordinates
            for d in dets:
                d["bbox_xyxy"] = [
                    d["bbox_xyxy"][0] / scale + cx1,
                    d["bbox_xyxy"][1] / scale + cy1,
                    d["bbox_xyxy"][2] / scale + cx1,
                    d["bbox_xyxy"][3] / scale + cy1,
                ]
        all_dets.extend(dets)
    return self._wbf_merge(all_dets) if merge == "wbf" else self._nms(all_dets)
```

**Verification:**
1. Unit test: synthetic 200×200 image, button at known location, verify multi-scale detects it with correct coordinates.
2. eval_v2 run with `--tta` — button/rivet/zipper accuracy should improve 5-10%.
3. Latency log: check `debug.tta_time_ms` < `timeout_ms`.

**Rollback:** Master toggle `enable_multiscale_tta: false` completely disables TTA path. Router skips `detect_multiscale()` and calls `detect_multi_prompt()` directly.

**Dependencies:** None (works with existing DINO-tiny).

---

### T5: Higher DINO Input Resolution [P1]

**Why:** DINO-tiny default processor resizes to 800×800 max. Small parts in a 300×400 garment crop have ~50×50 pixel DINO input — insufficient for feature extraction. Force-resize to 800×800 gives small parts more pixel budget.

**Files:**
- MODIFY: `src/fashion_vision/localization/grounding_dino_locator.py` (`detect()` method)
- MODIFY: `configs/attribute_inference.yaml` (uses same config block as T4)

**Implementation:**
```python
# In detect(), before processor call:
if enable_hires and part in hires_parts:
    # Resize image to target_size before processor
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    # Record scale factors for bbox remapping
    scale_x = target_size[0] / original_w
    scale_y = target_size[1] / original_h
    # After detection, remap bboxes back:
    d["bbox_xyxy"] = [x1/scale_x, y1/scale_y, x2/scale_x, y2/scale_y]
```

**Verification:** eval_v2 with `--hires` flag. Expected improvement on button/rivet: +5-8%.

**Rollback:** Config toggle `enable_hires: false` (default) skips the resize entirely.

**Dependencies:** None.

---

### T6: Model Ensemble — FP YOLO + DINO Overlap [P1]

**Why:** When both FP YOLO and DINO run on the same part (currently FP runs first, DINO only on miss), FP's false negatives could be recovered by DINO, and FP's false positives suppressed by DINO disagreement.

**New strategy:** For parts in Fashionpedia core coverage:
1. Run FP YOLO (fast, ~16ms miss).
2. Run DINO in parallel (or serial with timeout).
3. If both detect → average bboxes, keep only the intersection region.
4. If only one detects → use that one, mark as single-model.
5. If neither detects → not_detected.

**Files:**
- MODIFY: `src/fashion_vision/localization/region_localization_router.py` (FP YOLO + DINO parallel path)

**Key changes:**
```python
# Current: FP YOLO first → if hit, early return (DINO skipped)
# New: FP YOLO + DINO both run for ensemble parts.
ENSEMBLE_PARTS = {"zipper", "pocket", "button"}  # parts where both models are useful

if part in ENSEMBLE_PARTS and locator is not None:
    fp_dets = fashionpedia_detector.detect(...)
    dino_dets = locator.detect_multi_prompt(...)
    merged = _ensemble_boxes(fp_dets, dino_dets, method="intersection")
    # intersection: for each FP box, find best-matching DINO box.
    # If IoU > 0.3 → average the two boxes, use intersection region.
    # If no match → keep original box with lowered confidence.
```

**Verification:** eval_v2 with ensemble enabled. Expected: zipper/pocket/button improve 5-10% (DINO provides complementary signal).

**Rollback:** 1-line change: set `ENSEMBLE_PARTS = set()` to disable.

**Dependencies:** T2 (soft NMS ensures both backends produce more reasonable boxes before merging).

---

### T7: WBF Merge for DINO-Only Parts [P1]

**Why:** Current `detect_multi_prompt()` runs 3-5 prompts independently, then greedy NMS merges. WBF (Weighted Box Fusion) is proven better for small/overlapping objects — it computes a weighted average of all boxes instead of discarding.

**Files:**
- MODIFY: `src/fashion_vision/localization/grounding_dino_locator.py` (replace `_nms()` call in `detect_multi_prompt()` with WBF option)
- NEW: `src/fashion_vision/localization/box_fusion.py` (WBF implementation, ~60 lines)

**Implementation:**
```python
# box_fusion.py — self-contained WBF implementation
def weighted_box_fusion(boxes: list[dict], iou_thr: float = 0.55) -> list[dict]:
    """
    Merge overlapping boxes by weighted average of coordinates.
    Weight = score.  Returns fused boxes sorted by confidence.
    
    Algorithm:
    1. Sort boxes by score desc.
    2. For each box, find all boxes with IoU > iou_thr → form a cluster.
    3. For each cluster: fused_x = sum(score_i * x_i) / sum(scores)
    4. Fused confidence = mean of cluster scores × min(1, cluster_size / expected_size)
    """
```

**Verification:** eval_v2 DINO parts (shoes, button, bag) — expected +3-5%.

**Rollback:** `merge_method` parameter in `detect_multi_prompt()`: `"wbf"` (new) vs `"nms"` (default). 1-line revert.

**Dependencies:** None (independent of other changes).

---

### T8: Fashionpedia YOLO Re-Train with Milder Balancing [P1]

**Why:** Current p=1.0 r=12 balancing traded 34% mAP50 (0.47→0.312) for class balance. Milder balancing (p=0.6 r=6) recovers more head-class performance while still improving tail.

**Files:**
- MODIFY: `scripts/build_fashionpedia_balanced_train.py` (run with new params, not code change)
- NEW: `models/detectors/fashionpedia_yolov8s_19cls_balanced_v2_best.pt` (new weights)
- PRESERVE: `models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt` → `models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt` (rename, keep)

**Commands:**
```bash
# Step 1: Preserve old weights
cp models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt \
   models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt

# Step 2: Generate milder balanced train list
python scripts/build_fashionpedia_balanced_train.py \
    --labels-dir E:/fashionpedia_yolo_19cls/labels/train \
    --images-dir E:/fashionpedia_yolo_19cls/images/train \
    --base-yaml E:/fashionpedia_yolo_19cls/fashionpedia_parts.yaml \
    --out-dir E:/fashionpedia_yolo_19cls \
    --power 0.6 --max-repeat 6 --seed 42

# Step 3: Train
yolo detect train \
    data=E:/fashionpedia_yolo_19cls/fashionpedia_parts_balanced_p0.6_r6.yaml \
    model=yolov8s.pt \
    epochs=100 imgsz=640 batch=16 device=0 \
    project=outputs/fashionpedia_19cls_yolov8s_balanced_v2 \
    name=p0.6_r6

# Step 4: Promote best.pt
cp outputs/fashionpedia_19cls_yolov8s_balanced_v2/p0.6_r6/weights/best.pt \
   models/detectors/fashionpedia_yolov8s_19cls_balanced_v2_best.pt

# Step 5: Switch FashionpediaPartDetector to v2 weights
# Change model_path in eval_validation_v2.py or router init
```

**Expected results:**
| Metric | v1 (p=1.0 r=12) | v2 target (p=0.6 r=6) |
|--------|------------------|----------------------|
| mAP50 | 0.312 | 0.38–0.42 |
| mAP50-95 | 0.199 | 0.24–0.28 |
| Max:min ratio | 24:1 | 50:1 |
| Expansion | 1.34× | 1.18× |
| Train time | ~5h | ~4h |

**Verification:** Run eval_v2 with v2 weights. Compare per-part FP-backed accuracy (collar, sleeve, pocket, zipper, etc.) vs v1 baseline.

**Rollback:** Set `FP_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt"` in eval script. Router uses v1 until v2 is confirmed better on all FP-backed parts.

**Dependencies:** Server GPU availability.

---

### T9: Expand Label Studio Annotations for Weak Parts [P3]

**Why:** Weak parts (ruffle 26%, shoes 29%, rivet 33%, buckle 34%, flower 36%) have 39-55 annotations each. More data enables better evaluation and potential fine-tuning.

**Target:**
| Part | Current | Target | Priority |
|------|---------|--------|----------|
| ruffle | 43 | 100 | High |
| rivet | 39 | 100 | High |
| buckle | 47 | 100 | Medium |
| flower | 47 | 100 | Medium |
| shoes | 78 | 100 | Low (already decent number) |
| tassel | 55 | 100 | Medium |

**Annotation guidelines (from mentor):**
- neckline = skin-exposed opening; collar = fabric entity
- Pocket: include pocket opening + visible fabric, not just opening
- Zipper: tall narrow box along zipper teeth, mark open vs closed state
- Button: annotate button_cluster (row), not individual buttons
- Drawstring: mark the visible hanging portion
- Annotate negative images (part absent) — required for FP rate measurement

**Verification:** Re-run eval_v2 after annotation. Expected: per-part metrics become more reliable (n=100+ per class → statistically meaningful).

**Dependencies:** Human annotation time (6 hours estimated).

---

## 3. Time-Budget Guardrails for TTA (T4)

```
┌─ TTA Decision Flow ─────────────────────────────────────────┐
│                                                              │
│  locate_region(query, instance, image, ...)                  │
│    │                                                         │
│    ├─ enable_multiscale_tta == false?  ──► skip TTA          │
│    │                                                         │
│    ├─ intent.part NOT in tta_parts?    ──► skip TTA          │
│    │                                                         │
│    ├─ start_tta = time.perf_counter()                        │
│    │                                                         │
│    ├─ run detect_multiscale(scales, timeout_ms)              │
│    │     ├─ scale 1.0 → ~150ms (baseline)                    │
│    │     ├─ scale 2.0 → ~150ms                               │
│    │     ├─ scale 3.0 → ~150ms (or until timeout)            │
│    │     └─ if elapsed > timeout_ms: return results so far   │
│    │                                                         │
│    ├─ tta_time_ms = time.perf_counter() - start_tta          │
│    │                                                         │
│    └─ debug["tta_time_ms"] = tta_time_ms                     │
│        debug["tta_scales_completed"] = [1.0, 2.0]  (if 3.0   │
│             timed out)                                       │
│                                                              │
│  TTA latency logging (per request):                          │
│    {                                                         │
│      "tta_enabled": true,                                    │
│      "tta_scales_requested": 3,                              │
│      "tta_scales_completed": 2,                              │
│      "tta_time_ms": 387,                                     │
│      "tta_timeout_ms": 500,                                  │
│      "tta_timed_out": true                                   │
│    }                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Prioritized Execution Plan

### Phase 0: Immediate (today, no GPU required)

| Order | Task | Effort | Blocks |
|-------|------|--------|--------|
| 0.1 | T1: Shape prior calibration | 2h | Nothing |
| 0.2 | T2: Soft NMS | 1h | Nothing |
| 0.3 | T3: Wire all_bboxes | 1h | T2 (more boxes → more impact) |

**Phase 0 checkpoint:** Run `eval_validation_v2.py --soft-nms`. Expected: +5-8% overall (mostly from button/pocket T1+T2 improvements).

### Phase 1: Short-term (this week, some GPU)

| Order | Task | Effort | Blocks |
|-------|------|--------|--------|
| 1.1 | T6: Model ensemble (FP+DINO overlap) | 2h | Phase 0 |
| 1.2 | T7: WBF merge for DINO parts | 1h | Nothing |
| 1.3 | T4: Multi-scale TTA (behind toggle, default OFF) | 3h | Nothing |
| 1.4 | T5: Higher DINO resolution (behind toggle, default OFF) | 1h | Nothing |
| 1.5 | T8: FP YOLO re-train p=0.6 r=6 | 1h dev + 4h GPU | Phase 0 (need baseline) |

**Phase 1 checkpoint:** eval_v2 with T4+T5 enabled. Expected: +8-12% on small parts.

### Phase 2: Mid-term (next 1-2 weeks)

| Order | Task | Effort | Blocks |
|-------|------|--------|--------|
| 2.1 | T9: Expand annotations | 6h human | Phase 1 (need to know what to annotate) |
| 2.2 | Evaluate DINO-base (if downloaded) | 2h GPU | T9 (need more annotations to evaluate properly) |
| 2.3 | Fine-tune decision gate: DINO-tiny vs DINO-base vs FP YOLO per part | 2h | 2.1 + 2.2 |

**Phase 2 checkpoint:** Decision on DINO-base adoption + fine-tuning scope.

---

## 5. Timeline

```
Week 1 (Jul 10-14):
  Mon-Tue: Phase 0 (T1, T2, T3) — 4h dev
  Wed:     Phase 0 checkpoint — eval_v2 baseline
  Thu-Fri: Phase 1 start (T6, T7, T4, T5) — 7h dev

Week 2 (Jul 17-21):
  Mon:     Phase 1 continued — T8 training launch (overnight GPU)
  Tue-Wed: T8 training analysis + Phase 1 checkpoint — eval_v2
  Thu-Fri: Phase 2 start (T9 annotation)

Week 3 (Jul 24-28):
  Mon-Wed: T9 annotation completion
  Thu:     DINO-base evaluation (if available)
  Fri:     Phase 2 checkpoint — per-part accuracy ceiling analysis
```

---

## 6. Milestone Tracking Table

| Milestone | eval_v2 Overall@IoU>0.3 | Key Weak Parts | Date Target |
|-----------|--------------------------|----------------|-------------|
| **Current baseline** | **51.9%** | ruffle 26%, shoes 29%, rivet 33% | 2026-07-09 |
| Phase 0 complete (T1+T2+T3) | **57-60%** | button ~55%, pocket ~55% | 2026-07-11 |
| Phase 1 complete (T4-T8) | **63-68%** | button ~60%, zipper ~55%, rivet ~45% | 2026-07-21 |
| Phase 2 complete (T9+DINO-base eval) | **68-73%** | ruffle ~40%, shoes ~45%, rivet ~50% | 2026-07-28 |
| With DINO-base + fine-tuning (future) | **75-82%** | All parts >50% | TBD |

**PRD target (≥92%) is reachable only after:**
1. DINO-base (or stronger model) replaces DINO-tiny
2. In-domain fine-tuning on 200+ annotations per weak part
3. This is Phase 3+ work, not in current sprint

---

## 7. Files Changed Summary

| File | Change | Tasks |
|------|--------|-------|
| `src/fashion_vision/localization/part_detection_config.py` | Calibrated shape thresholds | T1 |
| `scripts/calibrate_shape_priors_from_eval.py` | **NEW** — extract TP distributions | T1 |
| `configs/shape_priors_calibrated_v1.yaml` | **NEW** — calibrated config | T1 |
| `src/fashion_vision/localization/grounding_dino_locator.py` | `_soft_nms()`, `_wbf_merge()`, `detect_multiscale()`, hires resize | T2, T4, T5, T7 |
| `src/fashion_vision/localization/box_fusion.py` | **NEW** — WBF implementation | T7 |
| `src/fashion_vision/localization/region_localization_router.py` | Ensemble routing, TTA toggle, all_bboxes | T3, T4, T6 |
| `tools/infer/garment_pipeline.py` | Consume all_bboxes | T3 |
| `configs/attribute_inference.yaml` | TTA config block | T4, T5 |
| `scripts/eval_validation_v2.py` | `--soft-nms`, `--tta`, `--hires`, `--ensemble` flags | verification |
| `scripts/build_fashionpedia_balanced_train.py` | Run with p=0.6 r=6 (not code change) | T8 |
| `models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt` | Rename from current (preserve) | T8 |
| `models/detectors/fashionpedia_yolov8s_19cls_balanced_v2_best.pt` | **NEW** — v2 weights | T8 |

---

## 8. Verification Checklist

- [ ] Phase 0: `python scripts/eval_validation_v2.py --soft-nms` → metrics.json diff vs baseline
- [ ] Phase 0: `python scripts/calibrate_shape_priors_from_eval.py` → YAML output reviewed by human
- [ ] Phase 1: `python scripts/eval_validation_v2.py --tta --hires` → confirm TTA toggle works, latency logged
- [ ] Phase 1: `python scripts/eval_validation_v2.py --ensemble` → confirm no regression on FP parts
- [ ] Phase 1: Compare FP YOLO v1 vs v2 per-part accuracy — no FP part should regress >2%
- [ ] Phase 2: Label Studio export → `data/validation/project-11-*/result.json` → re-run eval_v2
- [ ] All phases: Record metrics.json before/after each change. Never deploy without measured improvement.
- [ ] All phases: `pytest tests/` — no existing test regression

---

*End of plan v2. Based on mentor feedback (2026-07-09) + eval_v2 data + first-principles analysis.*
