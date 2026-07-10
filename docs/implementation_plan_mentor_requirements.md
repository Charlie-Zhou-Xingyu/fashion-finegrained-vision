# Implementation Plan: Mentor Requirements for 3.1.2 Localization
> Written: 2026-06-23  
> Status: **PLAN — no code written; awaiting approval**

---

## H. Verification (Self-Check — Read This First)

Before reading the detailed plan, here are the ten yes/no checks from the requirements, answered against what follows:

| Check | Answer |
|---|---|
| Plan avoids sending every Chinese query to Qwen 7B? | **YES** — local synonym table first; Qwen only for unmatched |
| Plan applies garment masks, not just bbox crops? | **YES** — Phase 1 fix wires `garment_mask` into `detect_multi_prompt()` |
| Plan returns "not detected" for low-confidence/rejected? | **YES** — Phase 1 removes the fallback-best-candidate behavior |
| Plan avoids pretending hierarchy is already implemented? | **YES** — Sec A clearly marks mask containment as NOT IMPLEMENTED |
| Plan includes human annotation and visual inspection steps? | **YES** — Sec D has explicit checklist |
| Plan explains how to deploy Qwen 7B on a server? | **YES** — Phase 3 covers vLLM + FastAPI step by step |
| Plan avoids multi-stage accumulated error for composite anchors? | **YES** — one-step first; two-step only with explicit coordinate-mapping |
| Plan includes tests and visualization? | **YES** — Sec F and per-phase visualization requirements |
| Plan separates immediate deterministic fixes from later fine-tuning? | **YES** — Phase 1 is deterministic; Phase 6 is fine-tuning only-if-needed |
| Plan identifies assumptions and risks? | **YES** — risks listed per phase and in Sec G |

---

## A. Repository Audit

### A.1 Intent Parsing

**File:** `src/fashion_vision/localization/intent_parser.py`

**Status: Partially supports requirements.**

What works:
- `parse_intent()` returns a structured `QueryIntent` dataclass with `part`, `side`, `garment_ref`, `direction`, `spatial_anchor`, `is_fast_path`, `is_zero_shot`.
- Chinese-to-part mapping via `PART_VOCAB` covers: 领口, 袖口, 下摆, 腰部, 裤腿, 拉链, 口袋, 扣子, 门襟, 腰带, 碎花/图案, 帽兜/领座.
- Longest-match search prevents 腰带/腰 ambiguity.
- Side words (左边/右边), garment refs (外套/裙子), direction words (前胸/上方) all parsed.
- `spatial_anchor` field parsed from "X附近/X上的" patterns.

What is **missing / wrong**:
1. **Long-tail fashion terms not in PART_VOCAB:** 肩缝 (shoulder seam), 袖缝 (sleeve seam), 荷叶边 (ruffle), 流苏 (fringe), 绑带 (tie strap), 抽绳 (drawstring — present in `part_detection_config` but NOT in `PART_VOCAB`). Queries for these hit `is_zero_shot=True` and get passed as raw Chinese text to DINO, which is English-only.
2. **No Qwen fallback.** `_zero_shot_noun_phrase()` only strips structural prefixes — it performs NO translation. Zero-shot Chinese queries are sent verbatim to DINO where they will fail silently.
3. **`spatial_anchor` is parsed but never routed.** The router does not implement the two-step anchor-crop strategy for queries like "口袋上的扣子".

### A.2 Part Vocabulary / Synonym Table

**File:** `src/fashion_vision/localization/intent_parser.py` (`PART_VOCAB`), `src/fashion_vision/localization/open_vocab_prompt_map.py` (`REGION_ALIASES`)

**Status: Partially supports requirements.**

- `PART_VOCAB` covers ~13 parts with Chinese synonyms.
- `open_vocab_prompt_map.py::REGION_ALIASES` adds Chinese→canonical mappings for aliases but is a separate code path (used by `is_open_vocab_region()` / `get_prompts_for_region()`, not by `parse_intent()`).
- **Gap:** The two synonym systems (`PART_VOCAB` in `intent_parser` and `REGION_ALIASES` in `open_vocab_prompt_map`) are not synchronized. A term missing from `PART_VOCAB` that exists in `REGION_ALIASES` will not be found by `parse_intent()`.
- **Gap:** No long-tail terms (荷叶边, 流苏, 绑带, 抽绳, 肩缝, 袖缝).
- **Gap:** No fallback translation layer.

### A.3 GroundingDINO / DINO Calls

**File:** `src/fashion_vision/localization/grounding_dino_locator.py`

**Status: Partially supports requirements.**

What works:
- `mask_to_garment()` grays out non-garment pixels to 128 — the mechanism exists.
- `detect()` accepts a `garment_mask` parameter.
- Multi-prompt with NMS, area filter, trailing period, sorted by score.

**Critical gap:**
- In `region_localization_router.py` (line 137): `detect_multi_prompt()` is called with `garment_mask=None` even though `garment_mask` was loaded just above (line 115).
- The router crops to the instance bbox (`_crop_to_instance`) and then passes `garment_mask=None` to DINO.
- **Consequence:** DINO runs on a plain bbox crop, not a mask-gated image. Background pixels, adjacent garments, arms, and hair inside the bbox are all visible to DINO. This directly violates the Layer A requirement.
- The garment mask IS loaded in `_load_garment_mask()` and IS used later in `_refine_mask()` (SAM intersection) — but never in the DINO detection step.

### A.4 Region Localization Router

**File:** `src/fashion_vision/localization/region_localization_router.py`

**Status: Partially supports requirements.**

What works:
- Unified fast-path / open-vocab routing.
- Crop-to-instance reduces false positives from other garments in the image.
- Per-part config takes priority over legacy prompt map.
- Shape priors filter applied after spatial constraints.
- `"status": "failed"` returned when DINO finds nothing.
- Debug metadata (`debug` dict) included in result.

**Gaps:**
1. `garment_mask=None` passed to DINO (see A.3 above).
2. `garment_ref` resolved in `_resolve_prompts` label but `filter_instances()` is NOT called inside `locate_region()`. The `garment_ref_filter.py` module exists but is not wired into the main router. The only place it is called is in the demo script (`query_region_online_demo.py`) as an optional step. If a caller uses `locate_region()` directly without the demo wrapper, `garment_ref` filtering is silently skipped.
3. `spatial_anchor` is present in `QueryIntent` but the router has no two-step logic for it — it is included in the result dict but never acted on.

### A.5 Garment Instance Masks

**Source:** SAM-HQ stage (Stage 2 of `GarmentPipeline`).  
**Output:** PNG mask files in `outputs/<run>/02_samhq/masks/`.  
**Schema field:** `mask_path` / `pred_mask_path` on instance records.

**Status: Masks exist and are loaded.**

- `region_locator.py::load_binary_mask()` loads from PNG.
- `resolve_instance_mask_path()` finds the mask file from the instance record.
- `bbox_mask_refiner.py::refine()` intersects SAM part-mask with garment mask.

**Gaps:**
- No systematic IoU evaluation of SAM-HQ mask quality on held-out images (known open gap from `current_project_status.md`).
- No "mask unavailable" signal propagated to the user — the router silently runs without mask if the path is missing.
- Mask quality is unverified. Low-quality masks (from occluded or partially visible garments) will produce misleading mask-gated crops.

### A.6 Part Detection Configuration

**File:** `src/fashion_vision/localization/part_detection_config.py`

**Status: Substantially implemented.**

- Per-part prompts, box/text thresholds, shape config for 22 parts.
- `drawstring` is registered here but NOT in `PART_VOCAB` (see A.1).
- Thresholds are **designed** values, not empirically calibrated. No validation dataset exists yet.

### A.7 Shape Prior Filtering

**File:** `src/fashion_vision/localization/part_shape_priors.py`

**Status: Substantially implemented but with a critical behavior conflict.**

What works:
- Area ratio, aspect ratio (h/w and w/h), center-x proximity, y_band, x_band checks.
- Reasons logged per rejection.
- `_shape_prior_status` and `_shape_prior_reasons` written to each detection dict.

**Critical conflict with requirements:**
- When **all candidates fail shape priors**, the function returns the highest-scoring rejected candidate with `_shape_prior_status = "fallback_best_candidate_after_all_rejected"`.
- The router sees a non-empty detection list and returns `"status": "success"` with a box that was geometrically implausible.
- **The mentor requirement is explicit:** "Implausible detections should be discarded. Low-confidence detections should return 'not detected'."
- **The current tests assert this wrong behavior.** `test_zipper_rejects_wide_box`, `test_belt_rejects_vertical_box`, etc. all assert `"fallback_best_candidate_after_all_rejected"` as the expected status.

### A.8 Garment Reference Filtering

**File:** `src/fashion_vision/localization/garment_ref_filter.py`

**Status: Module implemented but NOT wired into the main localization path.**

- `filter_instances()` exists and handles outerwear/top/pants/skirt/dress/inner filtering correctly.
- `inner` uses mask area sort (smallest = inner) — this is a heuristic, NOT mask containment.
- **Not called inside `locate_region()`.** Only used in the demo script.

**Gaps for mentor Requirement 3:**
- Inner/outer determined by mask area alone, not by geometric containment (mask A inside mask B).
- No mask containment ratio (`area_of_A_inside_B / area_of_A`) computed anywhere.
- No ambiguity signal — if area sort is inconclusive, no uncertainty is reported.

### A.9 Visualization / Debug Output

**Scripts:**
- `scripts/run_open_vocab_visual_test.py` — full image DINO overlay (accept/watermark).
- `scripts/run_open_vocab_yolo_crop_test.py` — per-query 7-cell HTML strip.
- `scripts/run_collar_visual_test.py` — fast-path collar overlay.

**Status: Partially supports requirements.**

What exists:
- Accept/reject overlays with watermark for rejected detections.
- HTML strip showing YOLO boxes and per-query overlays.

**Gaps:**
- No visualization of: garment mask, masked crop (what DINO actually sees), shape-prior rejection reasons per box.
- No per-detection confidence score displayed on overlay.
- No "not detected" banner when the router returns `"status": "failed"`.
- No visualization that shows the inner/outer containment decision.

### A.10 Tests

**File:** `tests/test_phase2_localization.py`

**Status: Good unit tests for config and shape priors; missing tests for several requirements.**

What is tested:
- `part_detection_config` helpers.
- Shape prior filter logic and fallback behavior (but asserting the wrong fallback).
- Intent parsing: garment_ref, zero-shot, direction.
- NMS and IoU utilities.
- Debug dict structure.

**Missing tests:**
- Chinese synonym matching for long-tail terms (荷叶边, 流苏, etc.).
- "All rejected → not detected" behavior (test currently asserts the opposite).
- Mask-gated vs. non-mask-gated DINO call (API check at minimum).
- Qwen 7B service interaction (mock/stub required).
- Mask containment ratio computation.
- `filter_instances()` called from `locate_region()`.
- Composite anchor two-step routing.

---

## B. Gap Analysis Against the Three Mentor Requirements

### Requirement 1: Chinese-to-English Query Handling

| Item | Status |
|---|---|
| Local synonym table for common fashion terms | Partially exists (PART_VOCAB); long-tail terms missing |
| 肩缝, 袖缝, 荷叶边, 流苏, 绑带, 抽绳 in vocab | **NOT IN VOCAB** |
| drawstring in part_detection_config | Exists there but not in PART_VOCAB |
| Qwen 7B fallback for unmatched queries | **NOT IMPLEMENTED** |
| Forced structured JSON output from LLM | **NOT IMPLEMENTED** |
| Timeout + deterministic fallback | **NOT IMPLEMENTED** |
| Caching for repeated queries | **NOT IMPLEMENTED** |
| Server deployment plan | **NOT IMPLEMENTED** |
| Currently zero-shot Chinese text is passed to English-only DINO | **BUG** |

**Files to modify:** `intent_parser.py`, `region_localization_router.py`  
**New files needed:** `translation_service.py`, server-side `serve_qwen.py`

### Requirement 2: Reducing False Positives for Small Components

| Item | Status |
|---|---|
| Garment mask loaded in router | ✅ Loaded |
| Garment mask PASSED to DINO | **NOT PASSED (bug at line 137 of router)** |
| `mask_to_garment()` method exists | ✅ Exists and works |
| Shape priors for button (circular / small area) | Partially — area_ratio and center_x, but no circularity check |
| Shape priors for zipper (elongated) | ✅ h/w ≥ 1.8 |
| Shape priors for drawstring | Exists in config but `drawstring` not in PART_VOCAB |
| Max area ≤ 5% for small parts | Button: max 8%, which is reasonable; others vary |
| "Not detected" when all fail shape priors | **NOT IMPLEMENTED — fallback returns best rejected** |
| Fallback behavior test asserts wrong outcome | **TESTS NEED TO BE UPDATED** |
| Visualization of mask-gated crop | **NOT IMPLEMENTED** |
| Visualization of rejected boxes with reasons | Partially — watermark overlay exists but no reason text |
| Low-confidence → "not detected" | **NOT IMPLEMENTED** |

**Files to modify:** `region_localization_router.py` (line 137), `part_shape_priors.py` (remove fallback), `tests/test_phase2_localization.py`

### Requirement 3: Garment Hierarchy and Composite Anchors

| Item | Status |
|---|---|
| Garment instance masks exist (SAM-HQ) | ✅ |
| Inner/outer by mask containment (area_A_in_B / area_A) | **NOT IMPLEMENTED** |
| Inner/outer by mask area heuristic only | Exists in `garment_ref_filter.py` but unreliable |
| Seam detection (boundary band between masks) | **NOT IMPLEMENTED** |
| `garment_ref` filter wired into `locate_region()` | **NOT WIRED — demo-only** |
| Composite anchor parsing (`spatial_anchor` field) | Parsed but NOT acted on |
| One-step DINO for compound queries | No routing logic |
| Two-step anchor-crop → small-part search | **NOT IMPLEMENTED** |
| Ambiguity / uncertainty signal for inner/outer | **NOT IMPLEMENTED** |
| Visualization of containment decision | **NOT IMPLEMENTED** |

**Files to modify:** `region_localization_router.py`, `garment_ref_filter.py`  
**New files needed:** `mask_containment.py`, `composite_anchor_router.py`

---

## C. Step-by-Step Implementation Roadmap

### Phase 1: Immediate Deterministic Fixes (No New Dependencies)

**Goal:** Fix the three highest-risk bugs. These changes are safe, testable, and do not require any new models or servers.

#### 1.1 Wire garment mask into DINO detection

**File:** `src/fashion_vision/localization/region_localization_router.py`  
**Change:** Pass `garment_mask` to `detect_multi_prompt()`. But there is a subtlety: when we crop to the instance bbox (`_crop_to_instance`), the full-image mask must be cropped to the same window before passing to DINO.

```
# Current (wrong):
detections, n_before_nms = locator.detect_multi_prompt(
    crop_image, prompts, garment_mask=None, ...
)

# Correct:
crop_mask = _crop_mask_to_instance(garment_mask, inst_bbox, pad_px=8)
detections, n_before_nms = locator.detect_multi_prompt(
    crop_image, prompts, garment_mask=crop_mask, ...
)
```

New helper `_crop_mask_to_instance()` mirrors `_crop_to_instance()` but for the mask array. If `garment_mask` is None, `crop_mask` is also None (no change in behavior for that case).

**Risk:** Low. The `mask_to_garment()` method is already tested. The new helper is a trivial slice.  
**Test:** Unit test: given a synthetic image + mask + bbox, verify that pixels outside the mask are 128 in the cropped image passed to DINO.

#### 1.2 Change shape priors fallback: "not detected" instead of best rejected

**File:** `src/fashion_vision/localization/part_shape_priors.py`  
**Change:** Remove the fallback-best-candidate block. When `kept` is empty, return an empty list. Add an `allow_fallback` parameter (default `False`) so callers that explicitly want the old behavior can opt in.

The router, upon receiving an empty list from `filter_by_shape_priors`, will hit the existing `if not detections:` check and return `"status": "failed", "reason": "no_detection_passed_shape_priors"`.

```python
# Change the end of filter_by_shape_priors:
if not kept:
    logger.warning(
        "part_shape_priors: all %d candidates rejected for part=%r — returning empty (not detected)",
        len(rejected), part,
    )
    return []   # ← changed from [best_fallback]
return kept
```

**Tests to update:** `test_zipper_rejects_wide_box`, `test_belt_rejects_vertical_box`, `test_button_rejects_box_exceeding_max_area`, `test_button_rejects_off_center_box`, `test_fallback_returns_highest_score` — all must be changed to assert empty list (or `"status": "failed"` via router). Add new test: `test_all_rejected_returns_not_detected`.

**Risk:** Medium. This changes observable behavior. The visual test scripts and the demo will now return "failed" for cases that previously returned a bad box. This is the correct behavior but callers must handle `"status": "failed"`.

#### 1.3 Add long-tail Chinese terms to PART_VOCAB

**File:** `src/fashion_vision/localization/intent_parser.py`  
**Change:** Add entries to `PART_VOCAB` and `_PART_TO_GROUNDING_TEXT`, and register in `PART_DETECTION_CONFIG`:

| Chinese term | Canonical part | Grounding text |
|---|---|---|
| 肩缝, 肩线缝合 | `shoulder_seam` | `shoulder seam` |
| 袖缝, 袖子缝合 | `sleeve_seam` | `sleeve seam` |
| 荷叶边, 波浪边 | `ruffle` | `ruffle trim on clothing` |
| 流苏, 穗子 | `fringe` | `fringe on clothing` |
| 绑带, 系带 | `tie_strap` | `tie strap on clothing` |
| 抽绳, 收绳 | `drawstring` | (already in PART_DETECTION_CONFIG — just add to PART_VOCAB) |

**Risk:** Low. Additive change. New parts go to DINO open-vocab path.  
**Test:** Parameterized test for each new Chinese term confirming correct `part` and `grounding_text`.

#### 1.4 Wire garment_ref filter into locate_region()

**File:** `src/fashion_vision/localization/region_localization_router.py`  
**Change:** Call `filter_instances()` before dispatching to DINO. This is pure routing logic.

```python
from fashion_vision.localization.garment_ref_filter import filter_instances

# Before resolving prompts, filter the instance list if garment_ref is set.
# (locate_region currently operates on a single instance, so this is prep for multi-instance API)
```

**Note:** The current `locate_region()` API takes a single `instance` dict, not a list. The `filter_instances()` call makes sense at the caller level (demo, pipeline). For now, we can add a `garment_ref_mismatch` flag to the result when `intent.garment_ref` is set but the instance's class does not match.

Immediate safe change: add a check inside `locate_region()` that if `intent.garment_ref` is set and the instance class does not match, set `_garment_ref_mismatch=True` on the instance before calling DINO. Log a warning. Return the result with `garment_ref_matched=False`.

Full wiring (when multiple instances are passed) is in Phase 4.

**Risk:** Low. Only adds a flag. Doesn't break existing behavior.

#### 1.5 Add visualization of mask-gated crop

**File:** `scripts/run_open_vocab_visual_test.py` or a new `scripts/visualize_masked_dino_crop.py`  
**Change:** For a given image + instance + query, produce a side-by-side output:
1. Original image with garment bbox
2. Bbox crop (what was used before)
3. Mask-gated crop (what is now passed to DINO)
4. DINO output on masked crop (accepted boxes = green, rejected = red + reason label)
5. Final result bbox on original image

This is the primary human-verification artifact for Phase 1.

**Prerequisite for human inspection:** Before trusting Phase 1 fixes, the user must visually inspect 20–30 sample outputs of this visualization.

---

### Phase 2: Evaluation and Threshold Calibration

**Goal:** Establish a numeric baseline for the fixed system before any further changes.

#### 2.1 Create a small validation set

**Task:** Collect 50–100 images with known ground-truth locations for at least 5 part types: pocket, zipper, button, collar, belt.

**How to collect:**
- Take images from `assets/random_train60/` and from the DeepFashion2 val set.
- For each image, record manually: does the garment have a pocket/zipper/button? If yes, approximate bounding box (you can mark with any image viewer).
- Target: 15–20 images per part type, mix of positive (part present) and negative (part absent or not visible).

**Annotation format:** A JSON file per image:
```json
{
  "image": "images/000004.jpg",
  "garment_det_id": 0,
  "parts": {
    "pocket": {"present": true, "bbox_xyxy": [120, 200, 180, 260]},
    "zipper": {"present": false},
    "button": {"present": true, "bbox_xyxy": [220, 100, 260, 380]}
  }
}
```

You do NOT need Label Studio for this validation set. A simple JSON file per image is sufficient. Use any image viewer to read pixel coordinates.

**Human action required:** You must mark the ground-truth bboxes. Expect 2–4 hours for 50 images covering 5 parts.

#### 2.2 Add debug visualization with explicit rejection reasons

**New script:** `scripts/visualize_localization_debug.py`  
**Output per image:** A single canvas showing:
1. Garment mask overlay (semi-transparent green)
2. Mask-gated DINO input crop
3. All DINO candidates before shape filter (grey boxes with score labels)
4. Candidates after shape filter:
   - Green box = passed, with score
   - Red box = rejected, with reason (e.g. "h/w=0.3 < 1.8")
5. Final selected box (blue, thick)
6. "NOT DETECTED" banner in red if `status == "failed"`
7. Confidence score at top

**This visualization must exist before any threshold calibration.** You cannot choose thresholds without seeing what the system is doing.

#### 2.3 Calibrate confidence thresholds empirically

**Process:**
1. Run the localization pipeline on the validation set with the current thresholds.
2. For each part type, record: TP, FP, FN at current threshold.
3. Sweep threshold from 0.20 to 0.50 in steps of 0.05.
4. Plot precision-recall curve per part.
5. Choose the threshold that gives precision ≥ 0.70 (not just recall).

**Do NOT guess thresholds.** The current values (button: 0.35, zipper: 0.40, pocket: 0.32) are designed, not measured. They may be wrong.

**Human action required:** Run the sweep script, look at the overlays at each threshold level, and make a judgment call. Plan for 1–2 hours per part type.

**Key decision:** What is acceptable precision-recall tradeoff for this use case? If the product shows "not detected" when a part is present but barely visible, is that acceptable? You need to decide this with the product team or your mentor.

#### 2.4 Define and measure per-part metrics

For each part type, measure on the validation set:

| Metric | Definition |
|---|---|
| FP rate | Detections returned when part is absent / total absent images |
| FN rate | "not detected" when part is present / total present images |
| Localization error | IoU of predicted bbox vs. ground truth (when TP) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Manual acceptance rate | Human reviewer sees the output and judges it "usable" |

**Target before Phase 3:** Precision ≥ 0.65 for pocket, zipper, button. This is a conservative baseline given zero fine-tuning.

---

### Phase 3: Qwen 7B Fallback Translation Service

**Goal:** For Chinese queries that do not match any entry in the local synonym table, call a Qwen-VL-7B-Chat service to produce a structured English grounding prompt.

#### 3.1 Local synonym table (already in Phase 1, extended here)

The local lookup in `intent_parser.py::parse_intent()` is the first and only path for known terms. No network call. This covers > 95% of expected queries.

`is_zero_shot=True` indicates the query is NOT in the vocab and should go to the translation service.

#### 3.2 Translation service client

**New file:** `src/fashion_vision/localization/translation_service.py`

**Design:**
```python
class TranslationService:
    def translate(self, query: str) -> TranslationResult:
        # 1. Check local cache (dict or sqlite)
        # 2. Call Qwen server if not cached
        # 3. Parse JSON response
        # 4. On timeout/invalid JSON → deterministic fallback
```

**`TranslationResult` dataclass:**
```python
@dataclass
class TranslationResult:
    original_query: str
    english_phrase: str        # e.g. "shoulder seam"
    grounding_text: str        # same or enriched, e.g. "shoulder seam on jacket"
    source: str                # "local_vocab" | "qwen_llm" | "fallback_literal"
    confidence: float          # 1.0 for local_vocab, model-reported for LLM, 0.0 for fallback
    raw_llm_response: Optional[str]  # for debugging
```

**Prompt to Qwen** (strict JSON format):
```
You are a fashion image analysis assistant. The user has a Chinese query about a garment part.
Translate it into a short English phrase suitable for an object detection model.

Rules:
- Output ONLY valid JSON, nothing else.
- Format: {"english_phrase": "...", "grounding_text": "... on clothing"}
- The grounding_text should be 3-6 words, garment-context.
- Examples: 肩缝 -> {"english_phrase": "shoulder seam", "grounding_text": "shoulder seam on garment"}
- If you are uncertain, still output a valid JSON with your best guess.

Query: {query}
```

**Timeout:** 3 seconds. If the server does not respond in 3 s, fall back to `_zero_shot_noun_phrase()` (literal stripping) and set `source="fallback_literal"`.

**Cache:** Simple Python dict in memory during a session. For persistent cache, write to `~/.cache/fashion_vision_translations.json`. Key = query string (after stripping whitespace/lowercase).

**Integration into `locate_region()`:**
```python
if intent.is_zero_shot:
    translation = translation_service.translate(intent.raw_query)
    prompts = [translation.grounding_text]
    # If source == "fallback_literal", add low_confidence flag to result
```

#### 3.3 Server deployment plan for Qwen-VL-7B-Chat

**Why a server:** Qwen-7B requires ~14GB VRAM for full precision or ~8GB for INT8. A consumer GPU (RTX 3080/4080) may not have enough VRAM alongside the existing DINO and SAM models. A rented GPU server avoids this constraint.

**Recommended stack:**

**Step 1: Rent a GPU server**

Options (in order of simplicity):
- **AutoDL** (autodl.com) — Chinese provider, easy setup, ~¥2–5/hour for A100/V100. Best option for this project given China context.
- **RunPod** — global, similar pricing.
- **Vast.ai** — cheapest, but less reliable.

Select: **RTX 3090 (24GB) or A100 (40GB)**. INT8 Qwen-7B needs ~8GB; full Qwen-7B needs ~14GB.

**Step 2: Set up the server**

SSH into the server. Then:

```bash
# Install vLLM (fastest inference for 7B models)
pip install vllm

# Download Qwen-VL-7B-Chat (or Qwen-7B-Chat if VL not needed for text-only translation)
# For translation only, Qwen-7B-Chat (text-only) is sufficient and lighter
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-7B-Chat')"
```

**Step 3: Start the vLLM serving endpoint**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-7B-Chat \
    --dtype auto \
    --max-model-len 2048 \
    --port 8000 \
    --host 0.0.0.0
```

This starts an OpenAI-compatible REST API on port 8000.

**Step 4: Expose via ngrok or SSH tunnel**

From your local machine:
```bash
ssh -L 8000:localhost:8000 user@<server_ip>
```

Or on the server:
```bash
pip install pyngrok
ngrok http 8000  # get public URL like https://xxxx.ngrok.io
```

**Step 5: Test the endpoint locally**

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen-7B-Chat", "prompt": "...", "max_tokens": 100}'
```

**Step 6: Configure the client**

In `translation_service.py`, set `QWEN_SERVER_URL` from an environment variable:
```python
QWEN_SERVER_URL = os.environ.get("QWEN_SERVER_URL", "http://localhost:8000")
```

Run locally:
```bash
export QWEN_SERVER_URL=http://localhost:8000  # when using SSH tunnel
```

**Request/response schema:**

Request (POST `/v1/completions`):
```json
{
  "model": "Qwen/Qwen-7B-Chat",
  "prompt": "...",
  "max_tokens": 80,
  "temperature": 0.1,
  "stop": ["\n"]
}
```

Response:
```json
{
  "choices": [{"text": "{\"english_phrase\": \"ruffle trim\", \"grounding_text\": \"ruffle trim on clothing\"}"}]
}
```

The client calls `json.loads(response["choices"][0]["text"])`.

**Timeout behavior:**
```python
import requests
try:
    resp = requests.post(url, json=payload, timeout=3.0)
    data = json.loads(resp.json()["choices"][0]["text"])
    return TranslationResult(english_phrase=data["english_phrase"], source="qwen_llm", ...)
except (requests.Timeout, KeyError, json.JSONDecodeError, requests.RequestException):
    return TranslationResult(english_phrase=fallback_noun, source="fallback_literal", confidence=0.0)
```

**Security:** Do NOT expose port 8000 directly to the internet without authentication. Use the SSH tunnel approach for development. For production, add an API key header.

**Logging:** Log every LLM call with: `query`, `response`, `latency_ms`, `source`. This helps audit LLM usage and catch unexpected output.

**Cost estimate:** At ¥3/hour and ~0.1s per call, 1000 LLM calls = ~0.028 hours = ~¥0.08. Caching means most repeated queries are free.

#### 3.4 Integration and fallback chain

```
query
  ↓
parse_intent() in PART_VOCAB?
  YES → grounding_text from _PART_TO_GROUNDING_TEXT
  NO  →  is_zero_shot=True
         ↓
         check local cache (dict)
         HIT  → cached english_phrase
         MISS → call Qwen (3s timeout)
                OK  → parse JSON, add to cache
                FAIL → _zero_shot_noun_phrase() + low_confidence flag
                ↓
         DINO with english_phrase
         result.source = "qwen_llm" | "fallback_literal"
         if source == "fallback_literal": result["translation_warning"] = True
```

---

### Phase 4: Garment Hierarchy and Mask Containment

**Goal:** Replace the mask-area heuristic for inner/outer with a proper geometric mask containment computation.

#### 4.1 Confirm mask source and quality first

**Before writing any containment code:**
1. Run the Phase 2 debug visualization on 20 images with 2+ garment instances (e.g., outerwear over top).
2. Visually verify: do the SAM-HQ masks actually separate the outer and inner garment correctly?
3. If masks are poor quality (significant overlap, wrong boundaries), the containment math will produce wrong answers. Fix mask quality first or the hierarchy result is meaningless.

**Human action required:** Visually inspect 20 multi-garment images. Expect 1 hour.

#### 4.2 Implement mask containment

**New file:** `src/fashion_vision/localization/mask_containment.py`

```python
@dataclass
class ContainmentResult:
    a_in_b_ratio: float   # area of (A ∩ B) / area(A)
    b_in_a_ratio: float   # area of (A ∩ B) / area(B)
    iou: float
    relationship: str     # "a_inside_b" | "b_inside_a" | "partial" | "disjoint" | "ambiguous"
    confidence: float     # how confident is the relationship label

def compute_containment(mask_a: np.ndarray, mask_b: np.ndarray) -> ContainmentResult:
    ...
```

**Thresholds (initial; must be calibrated empirically):**
- `a_in_b_ratio > 0.75` AND `b_in_a_ratio < 0.60` → `a_inside_b`
- `b_in_a_ratio > 0.75` AND `a_in_b_ratio < 0.60` → `b_inside_a`
- Both `a_in_b_ratio < 0.30` AND `b_in_a_ratio < 0.30` → `disjoint`
- Otherwise → `ambiguous`
- `confidence = max(a_in_b_ratio, b_in_a_ratio)` (simple proxy)

**Important:** These threshold values (0.75, 0.60) are initial guesses. They MUST be calibrated on real images with visual inspection. Do not treat them as fixed.

**Ambiguous case handling:** Return `"ambiguous"` and a `confidence` score. The caller should NOT force inner/outer when relationship is ambiguous. Log and surface "uncertain" to the user.

#### 4.3 Update garment_ref_filter.py for inner/outer

**File:** `src/fashion_vision/localization/garment_ref_filter.py`  
**Change:** For `garment_ref == "inner"`, if masks are available, use `compute_containment()` instead of mask area sort:

```python
if garment_ref == "inner":
    if all masks available:
        # compute pairwise containment
        # the garment most "contained" by others = inner
        # if ambiguous → return all with uncertainty flag
    else:
        # fall back to current area sort (with warning)
```

**Seam detection:** For queries like "外套和内搭的接缝", compute the boundary band between two adjacent/overlapping masks:
```python
def compute_seam_bbox(mask_a, mask_b) -> Optional[list]:
    # dilate each mask by ~5px, find intersection of dilated regions
    # return bounding box of the intersection as the "seam region"
```

#### 4.4 Tests with synthetic masks

Unit tests using synthetic masks (no real images needed):
- Test 1: Small square mask A entirely inside large square mask B → `a_inside_b`, high confidence
- Test 2: Two non-overlapping squares → `disjoint`
- Test 3: 50% overlap squares → `ambiguous`
- Test 4: Nearly contained (70% overlap) → check threshold sensitivity

**Human action required:** After implementing, inspect the containment results on 10–20 real multi-garment images. Check whether the computed relationship matches your visual judgment. Adjust thresholds if needed.

#### 4.5 Visualization for inner/outer decisions

**Add to debug visualization:**
1. Display both garment masks in different colors (red vs. blue semi-transparent).
2. Overlay with text: `a_in_b=0.82, b_in_a=0.23 → A inside B (confidence=0.82)`.
3. For `ambiguous`: show "AMBIGUOUS" banner with ratios.

This is the only way to verify the containment logic is working correctly.

---

### Phase 5: Composite Anchor Handling

**Goal:** Handle queries like "外套左边的口袋上的扣子" (button on the left pocket of the outerwear).

#### 5.1 One-step strategy (try first)

For compound queries where `spatial_anchor` is parsed (e.g., "口袋上的扣子" → `part=button`, `spatial_anchor=pocket`):

1. Concatenate into a single DINO prompt: `"button on clothing pocket"` or `"clothing button near pocket"`.
2. Apply garment mask.
3. Apply shape priors for the target part (button).
4. If DINO returns a high-confidence (> threshold) detection → done.

**When to use one-step:** `part` is a small component (button, zipper, drawstring), `spatial_anchor` is a medium component (pocket, placket). One-step works when DINO's pretraining has seen this composite visual concept.

#### 5.2 Two-step strategy (fallback)

If one-step fails (no detection above threshold):

**Step A — Detect the anchor region:**
1. Resolve prompts for `spatial_anchor` (e.g., "clothing pocket").
2. Run DINO on the garment-masked crop.
3. Apply shape priors for pocket.
4. If anchor not found → return `"status": "failed", "reason": "anchor_not_found"`.

**Step B — Search for the target within the anchor:**
1. Crop to anchor bbox with 20% padding.
2. Resize the crop to 640×640 (DINO's expected input).
3. Run DINO with prompts for the target part (e.g., "clothing button").
4. Apply shape priors with garment_bbox = anchor_bbox (relative).
5. Map bbox coordinates back to full-image space: `box[0] += anchor_x1`, etc.
6. If target not found within anchor → return `"status": "failed", "reason": "target_not_found_in_anchor"`.

**Coordinate remapping:** This is error-prone. The crop resize introduces scale; the offset adds a translation. Write and test this carefully:
```python
scale_x = anchor_w / 640
scale_y = anchor_h / 640
full_x1 = anchor_x1 + det_x1_in_crop * scale_x
full_y1 = anchor_y1 + det_y1_in_crop * scale_y
# etc.
```

Add a unit test: given a known bbox in a known crop, verify the remapped coordinates.

#### 5.3 Routing decision

```
if spatial_anchor is not None:
    try one-step with composite prompt
    if result.status == "success":
        return result
    # else fall through to two-step
    if user / config prefers two-step:
        return two_step_anchor_search(...)
    else:
        return one_step_result  # which is "failed"
```

#### 5.4 Failure modes for composite anchors

| Failure | Status returned |
|---|---|
| Anchor not detected | `failed`, `reason: anchor_not_found` |
| Target not found in anchor | `failed`, `reason: target_not_found_in_anchor` |
| Anchor found but below threshold | `failed`, `reason: anchor_low_confidence` |
| Coordinate mapping overflow (bbox outside image) | `failed`, `reason: coordinate_mapping_error` |

**Never return a result for a composite query where either the anchor or the target was not reliably found.**

---

### Phase 6: Fine-Tuning Plan (Only If Needed)

**Trigger condition:** After Phase 2 evaluation, if precision for any small part (pocket, zipper, button) is below 0.50 despite all engineering fixes, fine-tuning is justified. Do NOT fine-tune before evaluating the engineering baseline.

#### 6.1 Which model to fine-tune

Fine-tune **Grounding DINO base** (IDEA-Research/grounding-dino-base), not tiny. The base model is larger and more accurate; fine-tuning it on a small dataset is feasible.

Do NOT fine-tune SAM-HQ — its input is a confirmed bbox, and the localization errors are upstream.

#### 6.2 Data to collect

| Part | Min samples | Annotation needed |
|---|---|---|
| Pocket | 200 (100 positive + 100 negative) | Bounding box per pocket |
| Zipper | 150 (80+ / 70-) | Bounding box per zipper |
| Button | 200 (but buttons are tiny; may need 300) | Bounding box per button OR per button cluster |
| Drawstring | 100 (60+ / 40-) | Bounding box |
| Shoulder seam | 100 | Bounding box for the seam band |

Images: pull from DeepFashion2 training set (you already have access) or download from iMaterialist.

#### 6.3 How to use Label Studio

**Installation:**
```bash
pip install label-studio
label-studio start
# Opens at http://localhost:8080
```

**Project setup:**
1. Create a new project: "Garment Small Parts - DINO Fine-tuning".
2. Upload images (or connect to a local folder).
3. Label config (XML for Label Studio):
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="pocket" background="#FFA39E"/>
    <Label value="zipper" background="#D4380D"/>
    <Label value="button" background="#FFC069"/>
    <Label value="drawstring" background="#AD6800"/>
    <Label value="shoulder_seam" background="#096DD9"/>
  </RectangleLabels>
</View>
```

4. Annotate: draw tight bounding boxes around each part.

**Tips for annotation quality:**
- For buttons: draw a box around the entire button cluster (row of buttons), not individual buttons.
- For zippers: draw a tall narrow box along the zipper teeth.
- For pockets: include the full pocket opening + a few pixels of fabric.
- For negative images (part absent): still annotate the image, leave it with no boxes.

**Export format:** COCO JSON. Label Studio → Export → COCO JSON. This is directly usable for DINO fine-tuning.

**Train/val/test split:**
- 70% train, 15% val, 15% test.
- Split by image, not by annotation.
- Make sure there is no overlap between splits.

**Avoid leakage:** If an image appears in multiple crops (e.g., the same DeepFashion2 image used to create two garment crops), keep all crops from the same source image in the same split.

#### 6.4 When to stop fine-tuning

- Validation precision ≥ 0.70 AND recall ≥ 0.60 for each target part.
- OR the validation curve plateaus for 3 consecutive epochs.
- After fine-tuning, visually inspect 20 predictions on the held-out test set. If the model is clearly fitting training distribution but failing on test, it is overfitting — add more training data.

---

## D. Human-in-the-Loop Checklist

This is the definitive list of things **you must do by eye or by hand**. Nothing here can be automated away.

### Before Phase 1 (One-Time Setup)
- [ ] **D1.** Run `scripts/run_open_vocab_yolo_crop_test.py` on 10–15 images with known garment parts. Inspect the 7-cell HTML strip. Record which queries work, which fail. This is your baseline before any fixes.

### After Phase 1 (Verify Fixes)
- [ ] **D2.** Run the new mask-gated visualization (Phase 1.5) on 20–30 images. Visually confirm: the masked crop (what DINO sees) does not contain background, hair, or adjacent garments.
- [ ] **D3.** Inspect at least 10 "not detected" cases after the fallback-removal fix. Decide: is the "not detected" result acceptable (part truly absent) or a false negative (part present but missed)?
- [ ] **D4.** For each new Chinese term added (Phase 1.3), run a query and verify the correct part is found.

### Phase 2 (Calibration)
- [ ] **D5.** Annotate the 50–100-image validation set with ground-truth part bboxes. (2–4 hours.)
- [ ] **D6.** Run the threshold sweep and inspect the per-part precision-recall curves. Choose thresholds based on visual inspection + numeric metrics.
- [ ] **D7.** Inspect 20 visualization outputs at the chosen threshold. Count manually: "how many of these results would I accept as correct?" Target ≥ 65%.
- [ ] **D8.** Decide acceptable precision-recall tradeoff with mentor or product owner before locking thresholds.

### Phase 3 (Qwen Translation)
- [ ] **D9.** Test the Qwen server with 10 long-tail Chinese queries. Inspect the JSON responses. Are they reasonable English phrases? Are they garment-context aware?
- [ ] **D10.** Test the timeout/fallback path by deliberately stopping the server and submitting a query.
- [ ] **D11.** Check the cache: submit the same query twice. Verify the second call does not hit the server (inspect logs).

### Phase 4 (Hierarchy)
- [ ] **D12.** Inspect SAM-HQ masks on 20 multi-garment images before writing any containment code. Confirm masks are sufficiently clean to support containment math.
- [ ] **D13.** After implementing containment, visually verify 15–20 multi-garment examples. Does the system correctly identify inner vs. outer? Adjust thresholds if needed.
- [ ] **D14.** Inspect "ambiguous" cases. Decide: is the uncertainty signal useful, or does it block too many legitimate queries?

### Phase 5 (Composite Anchors)
- [ ] **D15.** Test "口袋上的扣子" (button on pocket) one-step path on 10 images. Inspect overlays. Does DINO find the button in the pocket region?
- [ ] **D16.** For two-step, verify coordinate remapping: inspect the anchor crop and the final bbox on the original image. Do they align?
- [ ] **D17.** Collect 5–10 compound query examples that represent realistic user queries. Test each one. Record pass/fail.

### Phase 6 (Fine-Tuning, If Triggered)
- [ ] **D18.** Annotate 150–300 bounding boxes per part type in Label Studio. (Expect 4–8 hours total.)
- [ ] **D19.** After training, inspect 20 predictions on the test set. Are they better than the engineering baseline?
- [ ] **D20.** If overfitting is visible (test performance << val performance), add more training data before concluding fine-tuning is done.

---

## E. Code Modification Plan

| File | Change | Risk | Test | Immediate? |
|---|---|---|---|---|
| `src/fashion_vision/localization/region_localization_router.py` | Add `_crop_mask_to_instance()` helper; pass cropped garment mask to `detect_multi_prompt()` | Low | Unit test: mask pixels outside bbox are 128 in crop | Yes |
| `src/fashion_vision/localization/part_shape_priors.py` | Remove fallback-best-candidate; return `[]` when all rejected | Medium | Update 5 existing tests; add "all rejected → empty" test | Yes |
| `src/fashion_vision/localization/intent_parser.py` | Add 6 new long-tail part entries to PART_VOCAB and `_PART_TO_GROUNDING_TEXT` | Low | Parameterized test per new term | Yes |
| `src/fashion_vision/localization/part_detection_config.py` | Add shape configs for `shoulder_seam`, `sleeve_seam`, `ruffle`, `fringe`, `tie_strap` | Low | Test: get_part_prompts for new parts returns non-empty list | Yes |
| `tests/test_phase2_localization.py` | Update tests that assert wrong fallback behavior; add new "not_detected" tests | Low | Run pytest | Yes |
| `scripts/visualize_localization_debug.py` | New script: side-by-side debug visualization | Low | Run manually on 1 image | Yes |
| `src/fashion_vision/localization/translation_service.py` | New file: TranslationService with Qwen client, cache, timeout, fallback | Medium | Unit tests with mocked HTTP; integration test with real server | Phase 3 |
| `src/fashion_vision/localization/mask_containment.py` | New file: compute_containment(), seam bbox | Low | Synthetic mask tests | Phase 4 |
| `src/fashion_vision/localization/garment_ref_filter.py` | Replace area-sort for "inner" with containment logic | Medium | Tests with synthetic and real masks | Phase 4 |
| `src/fashion_vision/localization/region_localization_router.py` | Add two-step composite anchor routing; coordinate remapping | High | Unit test: bbox remapping math; integration test | Phase 5 |

---

## F. Test Plan

### F.1 Unit Tests (No GPU)

| Test | File | Assertion |
|---|---|---|
| New long-tail Chinese terms parse correctly | `test_intent_parser_extended.py` | `parse_intent("荷叶边").part == "ruffle"` |
| Zero-shot query still routes to DINO | same | `parse_intent("帽檐装饰").is_zero_shot == True` |
| All shape priors rejected → empty list | `test_phase2_localization.py` | `filter_by_shape_priors([wide_box], "zipper", garment) == []` |
| Mask crop helper: pixels outside mask become 128 | `test_router_helpers.py` | pixel value check |
| Mask crop helper: offset is correctly computed | same | offset matches bbox |
| TranslationService: cached query does not call HTTP | `test_translation_service.py` | mock HTTP called 0 times on second call |
| TranslationService: timeout returns fallback literal | same | `result.source == "fallback_literal"` |
| TranslationService: invalid JSON returns fallback literal | same | same |
| TranslationService: valid JSON parsed correctly | same | `result.english_phrase == "shoulder seam"` |
| ContainmentResult: A inside B | `test_mask_containment.py` | synthetic masks; `relationship == "a_inside_b"` |
| ContainmentResult: disjoint | same | `relationship == "disjoint"` |
| ContainmentResult: ambiguous | same | `relationship == "ambiguous"` |
| Coordinate remapping: known bbox in known crop | `test_composite_anchor.py` | pixel-exact assertion |

### F.2 Tests That Must Be Updated

These currently test the wrong behavior and will fail after Phase 1:

| Test | Current assertion | New assertion |
|---|---|---|
| `test_zipper_rejects_wide_box` | `"fallback_best_candidate_after_all_rejected"` | empty list |
| `test_belt_rejects_vertical_box` | `"fallback_best_candidate_after_all_rejected"` | empty list |
| `test_button_rejects_box_exceeding_max_area` | `"fallback_best_candidate_after_all_rejected"` | empty list |
| `test_button_rejects_off_center_box` | `"fallback_best_candidate_after_all_rejected"` | empty list |
| `test_fallback_returns_highest_score` | 1 result with score=0.8 | empty list |

### F.3 Integration Tests

| Test | Description |
|---|---|
| Router: garment mask passed to DINO | Run with synthetic image+mask; inspect `detect_multi_prompt` call args |
| Router: "failed" status when all shape priors rejected | End-to-end with known bad detection |
| Router: `garment_ref_matched=False` when class mismatches | Instance class != query garment_ref |
| Composite anchor: one-step finds button in pocket | Requires GPU; marked xfail without |

### F.4 Visualization Tests

| Test | How to verify |
|---|---|
| Masked DINO crop has correct grey pixels | Load output PNG, check pixel values at known non-garment location |
| Rejected box shown in red on debug overlay | Load output PNG, check pixel color at rejected bbox center |
| "NOT DETECTED" banner present on failed result | String check in HTML output |

---

## G. Failure Modes and Fallback Behavior

| Failure Mode | System Response | Log Level |
|---|---|---|
| No garment mask available | Run DINO without mask; add `"mask_source": "none"` to result | WARNING |
| Garment mask file exists but corrupted/unreadable | Same as above; log exception | WARNING |
| DINO returns no boxes (above threshold) | `"status": "failed", "reason": "no_detection_above_threshold"` | INFO |
| DINO returns boxes but all fail shape priors | `"status": "failed", "reason": "no_detection_passed_shape_priors"` | INFO |
| All spatial constraint filters empty result | Fall back to unfiltered; log | WARNING |
| Qwen 7B timeout | Use `_zero_shot_noun_phrase()` fallback; set `"translation_warning": True` | WARNING |
| Qwen 7B invalid JSON | Same fallback as timeout | WARNING |
| Qwen 7B server unreachable | Same fallback; log server URL and error | ERROR |
| Composite anchor not found | `"status": "failed", "reason": "anchor_not_found"` | INFO |
| Target not found in anchor region | `"status": "failed", "reason": "target_not_found_in_anchor"` | INFO |
| Coordinate remapping produces out-of-bounds bbox | Clamp to image bounds; log as anomaly | WARNING |
| Inner/outer masks ambiguous (containment inconclusive) | Return `"relationship": "ambiguous"`, surface to caller | INFO |
| SAM-HQ box prompt fails or empty mask | Bbox-fill pseudo-mask with `"mask_source": "bbox_fill"` | WARNING |
| Image read fails | Raise `FileNotFoundError` to caller — do not silently return empty | ERROR |

**General principle:** The system must never silently return a bad result. Every `"status": "failed"` or degraded path must include a `"reason"` field and a log entry at appropriate level.

---

## Risks and Assumptions

### Risks

| Risk | Severity | Mitigation |
|---|---|---|
| SAM-HQ mask quality is poor for occluded garments | High | Phase 4.1 human inspection before mask containment code |
| DINO thresholds in part_detection_config are wrong (designed, not measured) | High | Phase 2 empirical calibration |
| Qwen 7B server may be slow (> 3s per call) | Medium | Timeout + cache + fallback |
| Mask-gated DINO may reduce recall (masking out valid part pixels near boundary) | Medium | Measure FN rate before and after mask gating in Phase 2 |
| Composite anchor two-step accumulates error | Medium | Only use two-step when one-step fails; validate coordinate remapping |
| Left/right convention (person-anatomical vs image-left) is unverified | Medium | Phase 1 follow-up: empirical test on left/right queries (from current_project_status open gap) |
| Qwen 7B translations may be semantically imprecise for fashion domain | Low | Check 10 sample outputs manually (D9); add to local vocab if pattern emerges |

### Assumptions

1. SAM-HQ masks are the source of truth for garment masks. If they are wrong, mask gating and hierarchy both fail.
2. DINO tiny has enough capacity for garment sub-parts after mask gating. If recall drops significantly, upgrade to DINO base.
3. The Qwen 7B server is accessible from the local project machine via the SSH tunnel or environment variable.
4. Left/right in `select_side_detection()` is image-left, not person-anatomical-left. This is NOT verified. (See open gap in current_project_status.md.)

---

## What Can Be Implemented Immediately vs. Later

### Immediately (no new dependencies, no new models, no server):
- Phase 1.1: Wire garment mask into DINO (1-line fix + helper)
- Phase 1.2: Remove shape priors fallback
- Phase 1.3: Add long-tail Chinese terms to PART_VOCAB
- Phase 1.4: Add garment_ref mismatch flag in locate_region()
- Phase 1.5: Debug visualization script
- Update tests

### Requires more work but no server:
- Phase 2: Validation set (requires your annotation time)
- Phase 4: mask_containment.py (pure numpy, no new dependencies)

### Requires server deployment:
- Phase 3: Qwen 7B translation service

### Requires annotation + labeled data:
- Phase 2 calibration: small validation set (your time, no tools other than an image viewer)
- Phase 6: fine-tuning dataset (Label Studio + 4–8 hours annotation)

### Should NOT be done yet:
- Phase 6 fine-tuning — wait for Phase 2 evaluation results
- Phase 5 composite anchors — lower priority than getting the basics right first
- Any changes to the fast-path landmark pipeline — risk of regression, no clear benefit

---

*End of plan. Awaiting approval before any code is written.*
