# Industrial Grounding Implementation Plan — 3.1.2 Localization
> Written: 2026-06-23 | Revised: 2026-07-03  
> Status: **Phase 1 COMPLETE; Phase 4 (inner garment detection) IMPLEMENTED — see §4.6**  
> Supersedes: `docs/implementation_plan_mentor_requirements.md`

---

## Revision Log

| Rev | Date | Changes |
|---|---|---|
| v1.0 | 2026-06-23 | Initial plan from repo audit |
| v1.1 | 2026-06-23 | 13 corrections applied (see revision notes inline) |
| v1.2 | 2026-07-03 | Added §4.6: Inner garment detection subsystem documentation (detector, refiner, cleaner, torso prior, 150-image validation results) |

---

## H. Verification (Self-Check — Read This First)

| Check | Answer |
|---|---|
| Plan avoids sending every Chinese query to Qwen 7B? | **YES** — local synonym table first; Qwen-VL-7B-Chat only for unmatched queries |
| Plan applies garment masks, not just bbox crops? | **YES** — Phase 1.1 wires mask crop into `detect_multi_prompt()` with rigorous crop-box reuse |
| Plan returns "not detected" (not "failed") for low-confidence/rejected? | **YES** — Phase 1.2 removes fallback; Sec G defines 5-status API |
| Plan avoids pretending hierarchy is already implemented? | **YES** — Sec A.8 is explicit; Phase 4 starts with human mask inspection |
| Plan includes human annotation and visual inspection steps? | **YES** — Sec D is a numbered checklist |
| Plan explains how to deploy Qwen-VL-7B-Chat on a server? | **YES** — Phase 3.3 covers primary (vLLM) and fallback (transformers + FastAPI) |
| Plan avoids multi-stage accumulated error for composite anchors? | **YES** — one-step uses prompt ensemble; two-step has explicit coordinate validation |
| Plan includes tests and visualization? | **YES** — Sec F and Sec G with debug directory schema |
| Plan separates deterministic fixes from fine-tuning? | **YES** — Phase 1 is immediate; Phase 6 is deferred and requires separate design doc |
| Plan identifies assumptions and risks? | **YES** — Sec G and per-phase risk notes |

---

## A. Repository Audit

### A.1 Intent Parsing

**File:** `src/fashion_vision/localization/intent_parser.py`  
**Status: Partially supports requirements.**

What works:
- `parse_intent()` returns `QueryIntent` with `part`, `side`, `garment_ref`, `direction`, `spatial_anchor`, `is_fast_path`, `is_zero_shot`.
- `PART_VOCAB` covers ~13 parts with Chinese synonyms including: 领口, 袖口, 下摆, 腰部, 裤腿, 拉链, 口袋, 扣子, 门襟, 腰带, 碎花/图案, 帽兜, 领座.
- Longest-match search prevents 腰带/腰 ambiguity. Side words, garment refs, direction words all parsed.
- `spatial_anchor` parsed from "X附近/X上的" regex patterns.

What is **missing / wrong**:
1. **Long-tail fashion terms absent from PART_VOCAB:** 肩缝 (shoulder seam), 袖缝 (sleeve seam), 荷叶边 (ruffle), 流苏 (fringe), 绑带 (tie strap), 抽绳 (drawstring — in `PART_DETECTION_CONFIG` but NOT in `PART_VOCAB`). These hit `is_zero_shot=True` and are currently passed verbatim (Chinese text) to the English-only DINO model.
2. **No Qwen translation fallback.** `_zero_shot_noun_phrase()` strips structural prefixes but does NOT translate. Zero-shot Chinese queries silently fail in DINO.
3. **`spatial_anchor` is parsed but never acted on.** The router has no two-step composite logic.

### A.2 Part Vocabulary / Synonym Table

**File:** `intent_parser.py` (`PART_VOCAB`) and `open_vocab_prompt_map.py` (`REGION_ALIASES`)  
**Status: Partially supports requirements.**

- `PART_VOCAB` covers ~13 parts with Chinese synonyms.
- `REGION_ALIASES` in `open_vocab_prompt_map.py` is a separate code path (used by `is_open_vocab_region()` / `get_prompts_for_region()`), NOT by `parse_intent()`. A term in one system that is absent from the other is silently invisible to some callers.
- Missing long-tail terms: 荷叶边, 流苏, 绑带, 抽绳, 肩缝, 袖缝.
- No fallback translation layer.

### A.3 GroundingDINO / DINO Calls

**File:** `src/fashion_vision/localization/grounding_dino_locator.py`  
**Status: Partially supports requirements.**

What works:
- `mask_to_garment()` grays out non-garment pixels to 128 — mechanism exists and tested.
- `detect()` accepts `garment_mask` parameter.
- Multi-prompt with NMS, area filter, trailing period auto-append, sorted by score.

**Critical gap confirmed in router:**  
`region_localization_router.py` line 137 calls `detect_multi_prompt(crop_image, prompts, garment_mask=None, ...)` — the mask is loaded just above (line 115) but discarded. DINO sees background, arms, hair, adjacent garments inside the bbox. This directly violates Layer A requirement.

Note: `bbox_mask_refiner.py::refine()` does intersect the SAM part-mask with the garment mask — but that is post-detection, not at the DINO detection step.

### A.4 Region Localization Router

**File:** `src/fashion_vision/localization/region_localization_router.py`  
**Status: Partially supports requirements.**

What works:
- Fast-path / open-vocab routing, crop-to-instance, per-part config priority, shape priors, debug metadata.
- `"status": "failed"` on no detection.

**Gaps:**
1. `garment_mask=None` passed to DINO (A.3).
2. `filter_instances()` not called inside `locate_region()` — only in the demo script. If callers use `locate_region()` directly, `garment_ref` filtering is silently skipped.
3. `spatial_anchor` is in the result dict but never acts on anything.
4. Status vocabulary is inconsistent: "failed" is used for all non-success cases, losing semantic information.

### A.5 Garment Instance Masks

**Source:** SAM-HQ Stage 2 (`tools/infer/segment_garments_samhq.py`)  
**Output:** PNG mask files at `outputs/<run>/02_samhq/masks/`  
**Status: Masks exist and are loaded. Quality unverified.**

- `load_binary_mask()` and `resolve_instance_mask_path()` work.
- `bbox_mask_refiner.py` intersects SAM part-mask with garment mask for the final output.
- **Gap:** No systematic IoU evaluation of mask quality on held-out images.
- **Gap:** If mask file is missing, the router silently runs without a mask and returns no warning in the result.
- **Gap:** No mask dilation option for small parts near garment boundaries.

### A.6 Part Detection Configuration

**File:** `src/fashion_vision/localization/part_detection_config.py`  
**Status: Substantially implemented but thresholds are designed, not empirically calibrated.**

- Per-part prompts, box/text thresholds, shape config for 22 parts.
- `drawstring` is registered here but NOT in `PART_VOCAB` — unreachable from Chinese queries.
- All thresholds (box_threshold, text_threshold, max_area_ratio) are authored estimates. No validation data exists yet.

### A.7 Shape Prior Filtering

**File:** `src/fashion_vision/localization/part_shape_priors.py`  
**Status: Substantially implemented but with a critical behavior conflict.**

What works:
- Area ratio, aspect ratio (h/w and w/h), center-x proximity, y_band, x_band checks.
- `_shape_prior_status` and `_shape_prior_reasons` written to each detection dict.

**Critical conflict:**  
When all candidates fail, the code returns the highest-scoring rejected candidate with `_shape_prior_status = "fallback_best_candidate_after_all_rejected"`. The router then returns `"status": "success"` with a geometrically implausible box. **Five existing tests assert this wrong behavior and must be updated.**

**Note on "circularity" for buttons** (Correction 3):  
A bbox-level circularity proxy (aspect ratio w/h ≈ 1.0) is achievable now. **True circularity** (the shape of the detected region, not its bounding box) requires either the SAM mask (post-detection) or a crop-level classifier. The Phase 1 plan only implements bbox aspect ratio checks. This is documented as an approximation.

### A.8 Garment Reference Filtering

**File:** `src/fashion_vision/localization/garment_ref_filter.py`  
**Status: Module implemented but NOT wired into the main localization path; inner/outer logic is heuristic only.**

- `filter_instances()` handles outerwear/top/pants/skirt/dress/inner filtering by class name.
- `inner` uses mask area sort (smallest area = inner) — unreliable for open-front garments.
- Not called inside `locate_region()`. Demo-only.

**Gaps for Requirement 3:**
- No mask containment computation.
- No rule ensemble for inner/outer.
- No ambiguity signal.
- No differentiation between inter-garment seams and intra-garment construction seams.

### A.9 Visualization / Debug Output

**Status: Partially supports requirements.**

Existing: accept/reject overlays (watermark), 7-cell HTML strip per query, collar fast-path overlay.

**Gaps:**
- No visualization of: garment mask, mask-gated DINO input crop, per-box shape rejection reasons.
- No confidence score on overlay boxes.
- No "not detected" banner.
- No inner/outer containment visualization.
- No standardized debug directory structure or result.json schema.

### A.10 Tests

**Status: Good unit tests for config and shape priors; missing tests for new requirements.**

Tested: `part_detection_config` helpers, shape prior filter logic (with wrong fallback assertion), intent parsing, NMS/IoU utilities, debug dict structure.

Missing: Chinese synonym matching for long-tail terms; "all rejected → empty list" behavior; mask-gated DINO call verification; Qwen service mocking; mask containment; `filter_instances()` integration; composite anchor two-step routing.

---

## B. Gap Analysis Against the Three Mentor Requirements

### Requirement 1: Chinese-to-English Query Handling

| Item | Status |
|---|---|
| Local synonym table for common fashion terms | Partially (PART_VOCAB); long-tail terms missing |
| 肩缝, 袖缝, 荷叶边, 流苏, 绑带, 抽绳 in vocab | **NOT IN VOCAB** |
| drawstring in PART_DETECTION_CONFIG | Exists there but unreachable from Chinese query |
| Qwen-VL-7B-Chat fallback for unmatched queries | **NOT IMPLEMENTED** |
| Forced structured JSON output from LLM | **NOT IMPLEMENTED** |
| Timeout (3 s) + deterministic fallback | **NOT IMPLEMENTED** |
| Per-session caching of LLM translations | **NOT IMPLEMENTED** |
| Server deployment plan for Qwen-VL-7B-Chat | **NOT IMPLEMENTED** |
| Zero-shot Chinese text passed verbatim to English-only DINO | **BUG** |

**Files to modify:** `intent_parser.py`, `region_localization_router.py`  
**New files:** `translation_service.py`, server-side `serve_qwen_vl.py`

### Requirement 2: Reducing False Positives for Small Components

| Item | Status |
|---|---|
| Garment mask loaded in router | ✅ Loaded |
| Garment mask passed to DINO (mask-gated) | **BUG — garment_mask=None hardcoded at line 137** |
| Image crop and mask crop using same crop box | **NOT ENFORCED — must be fixed explicitly** |
| Mask/image shape mismatch handling | **NOT IMPLEMENTED** |
| Optional mask dilation for small parts near boundary | **NOT IMPLEMENTED** |
| Configurable mask fill mode (grey/black/white/blur) | **NOT IMPLEMENTED — hardcoded grey (128)** |
| Shape priors for button (bbox aspect ratio ≈ 1.0) | Partial — area_ratio and center_x; aspect ratio check absent |
| Button "circularity" — bbox-level proxy only, not true circularity | **CLARIFICATION NEEDED in plan** |
| Per-part max_area_ratio (not a global 5% cap) | Config exists per part; plan must not imply global cap |
| Shape priors for zipper (elongated h/w ≥ 1.8) | ✅ |
| drawstring reachable from Chinese query | **BUG — not in PART_VOCAB** |
| "not_detected" (not "failed") when all shape priors fail | **NOT IMPLEMENTED — fallback returns worst candidate** |
| Visualization of mask-gated crop + rejection reasons | **NOT IMPLEMENTED** |

**Files to modify:** `region_localization_router.py` (line 137), `part_shape_priors.py`, `grounding_dino_locator.py`, `tests/test_phase2_localization.py`

### Requirement 3: Garment Hierarchy and Composite Anchors

| Item | Status |
|---|---|
| Garment instance masks (SAM-HQ) | ✅ Exist; quality unverified |
| Inner/outer via mask containment | **NOT IMPLEMENTED** |
| Inner/outer via class prior | **NOT IMPLEMENTED (only area sort)** |
| Inner/outer via bbox relation | **NOT IMPLEMENTED** |
| Rule ensemble for robust inner/outer | **NOT IMPLEMENTED** |
| Ambiguity handling for open-front garments | **NOT IMPLEMENTED** |
| Inter-garment boundary seams | **NOT IMPLEMENTED** |
| Intra-garment construction seams (肩缝, 袖缝) | **NOT IMPLEMENTED — separate strategy needed** |
| `garment_ref` filter wired into `locate_region()` | **NOT WIRED** |
| Composite anchor parsing (`spatial_anchor`) | Parsed, NOT acted on |
| One-step DINO with prompt ensemble for compound queries | **NOT IMPLEMENTED** |
| Two-step anchor-crop with side constraints | **NOT IMPLEMENTED** |
| Target-center-inside-anchor validation | **NOT IMPLEMENTED** |
| Coordinate remapping from crop to full image | **NOT IMPLEMENTED** |
| Uncertainty signal for ambiguous hierarchy | **NOT IMPLEMENTED** |
| Visualization of containment decision | **NOT IMPLEMENTED** |

---

## C. Step-by-Step Implementation Roadmap

---

### Phase 1: Immediate Deterministic Fixes (No New Dependencies)

**Goal:** Fix confirmed bugs. All changes are additive or behavioral corrections. No new models or servers required.

---

#### 1.1 Mask-Gated Detection — Rigorous Implementation

**File:** `src/fashion_vision/localization/region_localization_router.py`  
and  `src/fashion_vision/localization/grounding_dino_locator.py`

**The problem (confirmed bug):** `detect_multi_prompt()` is called with `garment_mask=None` at line 137 of the router, even though the garment mask is loaded at line 115. DINO runs on a plain bbox crop.

**Rigorous fix requirements** (Correction 2):

**Rule 1 — Same crop box.** The image crop and the mask crop must be derived from the **exact same** `(x1_clamped, y1_clamped, x2_clamped, y2_clamped, pad_px)` parameter set. The computation must not be duplicated; a single helper must return both:

```python
def _crop_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    inst_bbox: List[int],
    pad_px: int = 8,
) -> tuple[np.ndarray, Optional[np.ndarray], tuple[int, int]]:
    """
    Returns (image_crop, mask_crop, (offset_x, offset_y)).
    mask_crop is None if mask is None.
    Both crops use the exact same clamped bounding box.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst_bbox
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w, x2 + pad_px)
    y2 = min(h, y2 + pad_px)
    if x2 <= x1 or y2 <= y1:
        return image, mask, (0, 0)
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2] if mask is not None else None
    return image_crop, mask_crop, (x1, y1)
```

**Rule 2 — Shape mismatch handling.** If the loaded mask has a different spatial resolution from the image (e.g., loaded from a downsampled PNG), resize it to match before cropping. Always use `cv2.INTER_NEAREST` for binary masks to preserve binary values:

```python
if mask is not None and mask.shape[:2] != image.shape[:2]:
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
```

**Rule 3 — Optional mask dilation for small parts near garment boundaries.** Small parts like buttons and zippers often sit at the edge of the garment mask. Hard mask boundaries may clip them during gating. Add a per-part `mask_dilation_px` config field (default 0; button: 3, zipper: 5) applied before gating:

```python
if dilation_px > 0:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    mask_crop = cv2.dilate(mask_crop, kernel)
```

Add `mask_dilation_px` to `PART_DETECTION_CONFIG` for relevant parts.

**Rule 4 — Configurable fill mode.** Currently `mask_to_garment()` hardcodes grey (128). Make fill mode configurable:

```python
MASK_FILL_MODES = {"grey": 128, "black": 0, "white": 255}

def mask_to_garment(
    image: np.ndarray,
    garment_mask: np.ndarray,
    fill_mode: str = "grey",
) -> np.ndarray:
    ...
```

Grey (128) remains the default. Config field `mask_fill_mode` in `PART_DETECTION_CONFIG` or as a global setting. Blur mode (Gaussian blur on background) is noted as a future option; do not implement now.

**Risk:** Low. Additive changes to an existing mechanism. Mask crop must be tested carefully.  
**Tests:** Unit test: given 640×640 image, 640×640 mask with known non-garment region, assert non-garment pixels in the cropped masked image are exactly 128. Test shape mismatch: mask at 320×320, image at 640×640 — verify resize. Test dilation: known single-pixel garment edge — verify dilation extends coverage.

#### 1.2 Shape Priors: "not_detected" Instead of Best-Rejected Fallback

**File:** `src/fashion_vision/localization/part_shape_priors.py`

**Change:** When `kept` is empty, return `[]`. The existing `if not detections:` check in the router will then produce `"status": "not_detected", "reason": "no_detection_passed_shape_priors"`.

Remove the fallback block:
```python
# DELETE this block:
if not kept:
    best = max(detections, key=lambda d: d.get("score", 0.0))
    best["_shape_prior_status"] = "fallback_best_candidate_after_all_rejected"
    ...
    return [best]

# REPLACE with:
if not kept:
    logger.warning(
        "part_shape_priors: all %d candidates rejected for part=%r — "
        "returning empty list (will produce not_detected status)",
        len(rejected), part,
    )
    return []
```

Add an `allow_fallback: bool = False` parameter for callers that explicitly need the old behavior (currently none; included for extensibility).

**Tests to update** (5 tests assert the wrong behavior):
- `test_zipper_rejects_wide_box` → assert `== []`
- `test_belt_rejects_vertical_box` → assert `== []`
- `test_button_rejects_box_exceeding_max_area` → assert `== []`
- `test_button_rejects_off_center_box` → assert `== []`
- `test_fallback_returns_highest_score` → rename to `test_all_rejected_returns_empty_list`, assert `== []`

Add new test: `test_all_rejected_status_in_router_produces_not_detected` (integration-level, mocking DINO output).

**Risk:** Medium. Observable behavior change. The demo and visual test scripts will now correctly show "not_detected" in cases that previously returned a bad box. Callers that assumed a non-empty result must be audited.

#### 1.3 Unify Status Vocabulary — Five Explicit Statuses

**File:** `src/fashion_vision/localization/region_localization_router.py`  
**Correction 8.**

Replace all uses of `"failed"` with the appropriate specific status:

| Old status | New status | When |
|---|---|---|
| `"failed"` (no DINO boxes) | `"not_detected"` | Part searched but not found |
| `"failed"` (all shape priors rejected) | `"not_detected"` | Part searched, all candidates implausible |
| `"failed"` (no garment mask, no locator) | `"error"` | System precondition missing |
| — | `"unsupported_query"` | Part type not in vocab and not translatable |
| — | `"uncertain"` | Detection found but score is in [low_threshold, high_threshold] range, or hierarchy ambiguous |
| `"success"` | `"success"` | Detection found, above threshold, passed shape priors |

Add a `_STATUS_UNSUPPORTED_QUERY` sentinel in the router for queries where the part is unknown AND the translation service is unavailable or returns no grounding_text.

**Reason field contract:** Every non-success status must include a `"reason"` field with a machine-readable string. Define a closed enum of reason strings in the module:

```python
STATUS_REASONS = {
    "no_detection_above_threshold",
    "no_detection_passed_shape_priors",
    "no_detection_passed_spatial_constraint",
    "part_not_in_vocabulary",
    "translation_service_unavailable",
    "anchor_not_found",
    "target_not_found_in_anchor",
    "target_outside_anchor_bounds",
    "garment_mask_unavailable",
    "image_read_error",
    "model_not_loaded",
    "hierarchy_ambiguous",
}
```

#### 1.4 Add Long-Tail Chinese Terms to PART_VOCAB

**File:** `src/fashion_vision/localization/intent_parser.py`

Add to `PART_VOCAB` and `_PART_TO_GROUNDING_TEXT` (Correction 6 — note the seam type distinction):

| Chinese terms | Canonical part | Grounding text | Seam type |
|---|---|---|---|
| 抽绳, 收绳, 绳子 | `drawstring` | (already in PART_DETECTION_CONFIG) | intra-garment |
| 绑带, 系带, 蝴蝶结绑带 | `tie_strap` | `tie strap on clothing` | intra-garment |
| 荷叶边, 波浪边, 荷叶裙边 | `ruffle` | `ruffle trim on clothing` | intra-garment |
| 流苏, 穗子, 穗饰 | `fringe` | `fringe on clothing` | intra-garment |
| 肩缝, 肩线缝合, 肩部缝线 | `shoulder_seam` | `shoulder seam on garment` | intra-garment construction |
| 袖缝, 袖子缝合, 袖线 | `sleeve_seam` | `sleeve seam on clothing` | intra-garment construction |

**Seam type distinction (Correction 6):**  
- **Intra-garment construction seams** (`shoulder_seam`, `sleeve_seam`): seams within a single garment where fabric pieces are sewn together. They appear as visible lines/ridges on the garment surface. Detection strategy: DINO open-vocab with garment-context prompts. Limited by DINO's pretraining exposure to close-up seam detail — accuracy may be poor without fine-tuning. Document as low-confidence part type.  
- **Inter-garment boundary seams**: the visual boundary where two distinct garment instances (e.g., outerwear and inner top) overlap or meet. Computed geometrically from two masks (see Phase 4.4). These are NOT part types in PART_VOCAB; they require a two-instance query.

Add `shoulder_seam` and `sleeve_seam` to `PART_DETECTION_CONFIG` with appropriate shape configs:
```yaml
shoulder_seam:
  prompts: ["shoulder seam on garment", "seam at shoulder on clothing", "shoulder stitching line"]
  box_threshold: 0.28          # lower threshold — seams are subtle
  text_threshold: 0.23
  shape:
    y_band: [0.0, 0.30]        # shoulder seam at top of garment
    max_area_ratio: 0.15
    # NOTE: no aspect ratio check — seam bbox shape varies widely

sleeve_seam:
  prompts: ["sleeve seam on clothing", "arm seam on garment", "sleeve stitching"]
  box_threshold: 0.25
  text_threshold: 0.20
  shape:
    max_area_ratio: 0.25
    # NOTE: low accuracy expected without fine-tuning; mark results as uncertain
```

**Risk:** Low. Additive only.  
**Tests:** Parameterized test for each new Chinese term confirming correct `part` and `grounding_text`. Test that `drawstring` queries now route correctly (was previously unreachable).

#### 1.5 Wire garment_ref mismatch flag into locate_region()

**File:** `src/fashion_vision/localization/region_localization_router.py`

Immediate safe change: if `intent.garment_ref` is set and the instance's coarse class does not match, set `_garment_ref_mismatch=True` on the instance, log a warning, and include `"garment_ref_matched": False` in the result. This does not block the query; it surfaces information to callers.

Full multi-instance routing (where `filter_instances()` narrows the candidate list) belongs in Phase 4, because it requires a caller-level API change.

#### 1.6 Debug Visualization Script

**New file:** `scripts/visualize_localization_debug.py`

Produces a 7-panel per-query canvas:
1. Original image with garment detection bbox
2. Garment mask overlay (semi-transparent green)
3. Raw bbox crop (no mask — what the old code sent to DINO)
4. Mask-gated crop (what the fixed code sends to DINO)
5. All DINO candidates before shape filter (grey boxes + confidence score label)
6. After shape filter: accepted (green) + rejected (red with `_shape_prior_reasons` text)
7. Final result on original image: bbox (blue, thick) OR "NOT DETECTED" red banner

This visualization is the **primary human-verification artifact for Phase 1**. It must be reviewed on 20–30 sample images before concluding Phase 1 is correct.

**Output format:** follows the debug directory structure defined in Section G.2.

---

### Phase 2: Evaluation and Threshold Calibration

**Goal:** Establish a measured numeric baseline before any modeling decisions.

#### 2.1 Create a Validation Set Using Label Studio

**Correction 9:** Use Label Studio from the start — even for the small Phase 2 validation set. Annotations in COCO JSON format from day one can be directly reused for Phase 6 fine-tuning without re-annotation.

**Installation:**
```bash
pip install label-studio
label-studio start   # opens at http://localhost:8080
```

**Project setup:**
1. Create project: `"Garment Parts — Validation & Fine-tuning"`
2. Upload images from `assets/random_train60/` + DeepFashion2 val set (50–100 images total for Phase 2).
3. Label config:
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="pocket"         background="#FFA39E"/>
    <Label value="zipper"         background="#D4380D"/>
    <Label value="button_cluster" background="#FFC069"/>
    <Label value="drawstring"     background="#AD6800"/>
    <Label value="shoulder_seam"  background="#096DD9"/>
    <Label value="sleeve_seam"    background="#91D5FF"/>
    <Label value="ruffle"         background="#B7EB8F"/>
    <Label value="fringe"         background="#FFD591"/>
    <Label value="tie_strap"      background="#D3ADF7"/>
  </RectangleLabels>
</View>
```

**Note on button annotation:** Label `button_cluster` (the entire row of buttons), not individual buttons. Individual buttons are typically too small for a useful bbox at garment-image scale. DINO will also detect button arrays more reliably than single buttons.

**Annotation guidelines:**
- Draw tight boxes including a 5–10px margin around the part.
- For zippers: tall narrow box along the zipper teeth.
- For negative images (part absent): annotate the image with no boxes. Do not skip it — negatives are required to measure false positive rate.
- For pockets: include the pocket opening + visible pocket fabric, not just the opening.

**Target: 50–100 annotated images covering 5 primary parts** (pocket, zipper, button_cluster, drawstring, collar). Secondary parts (shoulder_seam, ruffle, fringe) can be added later.

**Export:** COCO JSON. In Label Studio: Export → COCO JSON. Save to `data/validation/garment_parts_val_v1.json`.

**Split:** For Phase 2 validation, use all 50–100 images as a validation set. When this set grows to 300+ images for Phase 6 fine-tuning, split into 70% train / 15% val / 15% test.

**Human action required:** 2–4 hours annotation time for 50–100 images at 5 parts.

#### 2.2 Threshold Calibration Protocol

**Do NOT guess or lock thresholds without measurement.** The current values in `PART_DETECTION_CONFIG` are authored estimates.

**Calibration script:** `scripts/calibrate_part_thresholds.py`  
Sweeps `box_threshold` from 0.20 to 0.55 in steps of 0.05. For each threshold:
1. Runs the localization pipeline on the validation set.
2. Computes: TP, FP, FN per part type.
3. Outputs precision-recall curve as CSV and a matplotlib PNG.

**Decision process:**
1. Plot the curve per part type.
2. Choose the highest threshold that keeps recall ≥ 0.50 (i.e., do not miss more than half the present parts).
3. At that threshold, check: is precision ≥ 0.65? If yes, accept. If not, investigate whether the FPs are due to threshold or shape priors.
4. Lock thresholds for the next run. Re-calibrate if the model changes.

**Human action required:** Run the sweep, inspect visualizations at each threshold level. Expect 1–2 hours. Final threshold choice requires your judgment.

#### 2.3 Metrics Definitions

For each part type, measure on the validation set:

| Metric | Definition |
|---|---|
| Precision | TP / (TP + FP) — how many of our detections are correct |
| Recall | TP / (TP + FN) — how many true parts we find |
| Localization accuracy | IoU of predicted bbox vs. ground truth (for TPs only) |
| FP rate | FP / total_negative_images — how often we detect a part that is not there |
| Manual acceptance rate | Human inspects results and judges "would I accept this output?" — target ≥ 65% |

**Acceptance targets (conservative; calibrate empirically):**

| Part | Precision target | Recall target | Comment |
|---|---|---|---|
| pocket | ≥ 0.65 | ≥ 0.50 | Medium difficulty |
| zipper | ≥ 0.60 | ≥ 0.50 | Elongated shape helps |
| button_cluster | ≥ 0.55 | ≥ 0.45 | Harder; many false positives from buttons on other items |
| drawstring | ≥ 0.55 | ≥ 0.45 | Low DINO confidence |
| shoulder_seam | ≥ 0.40 | ≥ 0.35 | Expected low accuracy without fine-tuning |

#### 2.4 Intermediate Evaluation: DINO-tiny vs DINO-base (Correction 12)

Before deciding on fine-tuning, compare DINO-tiny (current) against DINO-base on the same validation set:
- DINO-base: `IDEA-Research/grounding-dino-base` (larger model, ~150M params vs ~48M for tiny, requires more VRAM).
- Run the same calibration script with DINO-base, record the precision-recall curve per part.
- Evaluation matrix:

| Scenario | Action |
|---|---|
| DINO-base precision ≥ targets AND gap over DINO-tiny ≥ 10% | Switch to DINO-base; no fine-tuning needed |
| DINO-base slightly better (< 10% gap) but both below targets | Proceed to Phase 6 fine-tuning |
| DINO-tiny already meets targets | No model change; document result; skip Phase 6 |
| DINO-base meets targets but not DINO-tiny | Use DINO-base as production model |

**This step is mandatory before Phase 6.** Fine-tuning a small model when a larger off-the-shelf model suffices is wasteful.

---

### Phase 3: Qwen-VL-7B-Chat Translation Service

**Goal:** For Chinese queries that do not match any local PART_VOCAB entry, call Qwen-VL-7B-Chat to produce a structured English grounding prompt.

#### 3.1 Local Vocabulary First (Most Queries)

The `parse_intent()` call in `region_localization_router.py` handles all known terms with near-zero latency. Phase 1.4 expands this further. Qwen is called only when `intent.is_zero_shot == True`. This covers < 5% of typical fashion queries.

#### 3.2 Translation Service Client

**New file:** `src/fashion_vision/localization/translation_service.py`

**`TranslationResult` dataclass:**
```python
@dataclass
class TranslationResult:
    original_query:   str
    english_phrase:   str   # e.g. "shoulder seam"
    grounding_text:   str   # e.g. "shoulder seam on garment"
    source:           str   # "local_vocab" | "qwen_vl_llm" | "fallback_literal"
    confidence:       float # 1.0 for local_vocab, 0.0 for fallback_literal
    raw_llm_response: Optional[str] = None   # for debug logging only
```

**Prompt to Qwen-VL-7B-Chat** (text-only call; image input is optional and not required for translation):
```
You are a fashion garment analysis expert. A user is querying about a garment part or detail using Chinese.
Translate it into a short English phrase suitable for an object detection model.

Output ONLY valid JSON. No explanation. No preamble.
Format exactly: {"english_phrase": "...", "grounding_text": "... on clothing"}
Rules:
- english_phrase: 1–4 words, the part name itself.
- grounding_text: 3–7 words, add garment context (e.g. "on clothing", "on garment", "on jacket").
- Examples:
  肩缝 → {"english_phrase": "shoulder seam", "grounding_text": "shoulder seam on garment"}
  荷叶边 → {"english_phrase": "ruffle trim", "grounding_text": "ruffle trim on clothing"}
  流苏 → {"english_phrase": "fringe", "grounding_text": "decorative fringe on garment"}
- If uncertain, still output valid JSON with your best guess.

Query: {query}
```

**Fallback chain:**
```
is_zero_shot=True
  → check cache (dict, key=query.strip().lower())
      HIT  → return cached TranslationResult
      MISS → call Qwen-VL-7B-Chat server (POST, timeout=3.0s)
               OK and valid JSON → parse; add to cache; log
               Timeout / HTTP error / invalid JSON / missing keys
                 → TranslationResult(
                       grounding_text=_zero_shot_noun_phrase(query),
                       source="fallback_literal",
                       confidence=0.0,
                   )
```

When `source == "fallback_literal"`, the router adds `"translation_warning": True` to the result dict. The caller or UI layer should treat this as uncertain.

**Cache note:** In-memory dict per session. Persistent option: `~/.cache/fashion_vision_translations.json` (plain JSON, append-only). Implement in-memory first; persistent cache is Phase 3 stretch.

#### 3.3 Server Deployment — Qwen-VL-7B-Chat

> **Correction 1:** The project requirement specifies **Qwen-VL-7B-Chat**, not Qwen-7B-Chat. These are different models. Qwen-VL-7B-Chat includes a visual encoder; Qwen-7B-Chat is text-only. Although we only send text prompts for translation, the requirement mandates Qwen-VL-7B-Chat.

**VRAM requirement:** Qwen-VL-7B-Chat (full precision FP16) requires ~15–16GB VRAM. INT8 quantization reduces this to ~9–10GB. An RTX 3090 (24GB) or A100 (40GB) is recommended.

**Primary serving stack: vLLM**

vLLM supports Qwen-VL models through its multi-modal pipeline (added in vLLM 0.4+). However, support may vary by vLLM version. Verify compatibility before renting a server.

```bash
# On the rented server (Ubuntu 22.04, CUDA 12.1 recommended)
pip install vllm>=0.4.0

# Check Qwen-VL-7B-Chat is supported:
python -c "from vllm import LLM; LLM('Qwen/Qwen-VL-Chat', dtype='float16')"
# If this raises NotImplementedError or similar → use the fallback stack below
```

If vLLM supports the model:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-VL-Chat \
    --dtype float16 \
    --max-model-len 1024 \
    --port 8000 \
    --host 0.0.0.0
```

**Fallback serving stack: HuggingFace transformers + FastAPI** (Correction 1)

If vLLM does not support Qwen-VL-Chat, use this instead:

```bash
pip install transformers accelerate fastapi uvicorn
```

Server file `serve_qwen_vl.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, logging

app = FastAPI()
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen-VL-Chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
model.eval()

class TranslateRequest(BaseModel):
    query: str
    max_new_tokens: int = 80

@app.post("/translate")
def translate(req: TranslateRequest) -> dict:
    prompt = build_translation_prompt(req.query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=req.max_new_tokens, temperature=0.1)
    response_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info("query=%r response=%r", req.query, response_text)
    return {"text": response_text}

# uvicorn serve_qwen_vl:app --host 0.0.0.0 --port 8000
```

**Client call (in `translation_service.py`):**
```python
import requests, json

def _call_server(query: str, server_url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        resp = requests.post(
            f"{server_url}/translate",
            json={"query": query},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["text"]
        return json.loads(text)
    except Exception:
        return None
```

**Server access from local machine:**
```bash
# Option A: SSH tunnel (recommended for development, no public exposure)
ssh -L 8000:localhost:8000 user@<server_ip>
export QWEN_SERVER_URL=http://localhost:8000

# Option B: if the server has a public IP, use API key authentication
# Add: --header "X-API-Key: your_secret_key" to requests
```

**Security:** Do not expose port 8000 directly without authentication. SSH tunnel is sufficient for development. For production, add an API key check in the FastAPI middleware.

**Logging on the server:** Log every request with `timestamp`, `query`, `response_text`, `latency_ms`. Store in a plain text or SQLite log. This lets you audit LLM usage and catch unexpected outputs.

**Cost estimate (AutoDL):** RTX 3090 ≈ ¥3/hour. At 0.3s per call (transformers inference), 1000 calls = 5 minutes of GPU time = ¥0.25. Caching makes repeated queries free.

**What Qwen-VL-7B-Chat does NOT provide here:** We are NOT passing images to it. We use only the text input for translation. The VL model handles text-only prompts correctly. If future work needs the model to reason about image content for translation context, the image can be added as a second input.

---

### Phase 4: Garment Hierarchy and Mask Containment

**Goal:** Replace the mask-area heuristic with a rigorous rule ensemble for inner/outer determination, and implement inter-garment seam detection.

#### 4.1 Prerequisite: Verify SAM-HQ Mask Quality

**Before writing any containment code**, visually inspect SAM-HQ masks on 20 images with 2+ garment instances (e.g., outerwear over top).

Check per image:
- Does each garment have a separate, non-merged mask?
- Are mask boundaries reasonably clean (not cutting through the visible garment)?
- Is the outerwear mask significantly larger than the inner top mask, as expected?

If masks are merged, wrong, or too noisy for containment math: do not proceed to Phase 4.2. Fix SAM-HQ segmentation quality first (possibly by tuning SAM prompts or adding a secondary verification step).

**Human action required:** 1 hour visual inspection.

#### 4.2 Rule Ensemble for Inner/Outer (Correction 5)

**New file:** `src/fashion_vision/localization/mask_containment.py`

The inner/outer relationship between two garment instances is determined by a **rule ensemble**, not containment alone. Containment is the strongest signal but insufficient for all cases.

**Rule sources (ordered by reliability):**

**Rule 1 — Class prior:** Certain fine classes have strong innate priors.

| Class | Prior |
|---|---|
| `short sleeve outwear`, `long sleeve outwear` | Strong outer prior |
| `sling`, `vest` | Strong inner prior (when another garment present) |
| `short sleeve top`, `long sleeve top` | Weak inner prior (when outwear present) |
| `dress`, `shorts`, `trousers`, `skirt` | Typically standalone — inner/outer N/A |

**Rule 2 — Mask containment ratio:**
- `a_in_b_ratio = area(mask_A ∩ mask_B) / area(mask_A)` — what fraction of A is inside B?
- `b_in_a_ratio = area(mask_A ∩ mask_B) / area(mask_B)`

**Rule 3 — Bbox containment heuristic:**
- Is bbox_A mostly inside bbox_B? Compute: `iou_min = inter / min(area_A_bbox, area_B_bbox)`
- This is coarser than mask containment but useful as a secondary vote.

**Rule 4 — Visible area ratio:**
- If mask_A has much smaller visible (non-overlapped) area than mask_B, it is likely the inner layer.

**Ensemble decision logic:**
```
strong_outer_prior(A) AND mask_containment(A ⊂ B) > 0.60 → A is OUTER, B is INNER
strong_outer_prior(A) AND mask_containment(A ⊂ B) < 0.30 → ADJACENT (open-front coat)
weak priors AND mask_containment in [0.40, 0.60] → AMBIGUOUS (return "uncertain" status)
no class prior AND mask_containment > 0.75 → INNER_OUTER_BY_MASK (lower confidence)
```

**Open-front garment limitation (Correction 5):**  
For open-front coats or blazers, the outerwear mask may only partially overlap the inner garment mask even when worn layered. In this case, containment ratios are in the ambiguous range (0.30–0.60), and the class prior is the primary signal. Mark this case explicitly: `"relationship": "open_front_adjacent"` when outwear class prior is strong but containment < 0.40.

This is a known limitation of Phase 4. Improving open-front handling requires better segmentation or a multi-view approach. Document as incomplete.

**`ContainmentResult` dataclass:**
```python
@dataclass
class ContainmentResult:
    a_in_b_ratio:   float
    b_in_a_ratio:   float
    iou:            float
    relationship:   str     # "a_inside_b" | "b_inside_a" | "adjacent" |
                            # "open_front_adjacent" | "disjoint" | "ambiguous"
    confidence:     float   # 0.0–1.0
    evidence:       dict    # {"class_prior": ..., "mask_containment": ..., "bbox_containment": ...}
```

**Thresholds are initial estimates — MUST be calibrated empirically** (see Phase 2 protocol).

#### 4.3 Inter-Garment Seam Detection

For queries about the boundary between two garment instances (e.g., "外套和内搭的接缝"):

```python
def compute_inter_garment_seam_bbox(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dilation_px: int = 8,
) -> Optional[List[int]]:
    """
    Compute the bounding box of the boundary zone between two garment masks.
    
    Strategy: dilate each mask, find intersection of dilated regions.
    The intersection approximates the garment boundary / seam zone.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_px+1, 2*dilation_px+1))
    dilated_a = cv2.dilate(mask_a.astype(np.uint8), kernel)
    dilated_b = cv2.dilate(mask_b.astype(np.uint8), kernel)
    seam = cv2.bitwise_and(dilated_a, dilated_b)
    if seam.sum() == 0:
        return None  # no overlap even after dilation — garments are not adjacent
    coords = cv2.findNonZero(seam)
    x, y, w, h = cv2.boundingRect(coords)
    return [x, y, x + w, y + h]
```

Note: `dilation_px` is a tunable parameter. 5–10 pixels is typical for 640×640 images. Must be calibrated visually.

**Scope note:** This computes an approximate region, not a precise seam line. For a precise seam, mask erosion or edge detection on the boundary would be needed. Phase 4 implements the approximate version.

#### 4.4 Wire containment into garment_ref_filter

For `garment_ref == "inner"`, replace the mask-area sort with:
1. Compute pairwise `ContainmentResult` for all instance pairs.
2. If a clear relationship is found (`confidence > 0.70`), sort accordingly.
3. If `"ambiguous"` or `"open_front_adjacent"`: return all instances with `_hierarchy_status = "ambiguous"`.

If masks are unavailable: fall back to mask-area sort with `_hierarchy_status = "fallback_area_sort"`.

#### 4.5 Visualization for Hierarchy Decisions

Each hierarchy decision must produce a visualization showing:
1. Two garment masks in different colors (red vs. blue, semi-transparent overlay).
2. Text overlay: `a_in_b=0.82, b_in_a=0.23 → A inside B (conf=0.82)` or `AMBIGUOUS (0.44)`.
3. For seams: the seam bbox overlaid on the original image.

This is mandatory before trusting the hierarchy logic in production.

---

#### 4.6 Inner Garment Detection Under Outerwear (IMPLEMENTED — 2026-07-03)

**Status:** Implemented and validated on 150 FashionAI lapel-design images.
Visual inspection confirms decent overall performance. This subsystem
detects the clothing worn *under* an outerwear garment (e.g., a shirt
visible through an open jacket). It does **not** distinguish between
upper-body and lower-body inner garments — any clothing layer inside
the outerwear mask is treated as "inner garment."

> **Primary limitation (confirmed by inspection):** The single biggest
> quality bottleneck is SAM-HQ outerwear mask accuracy. When the SAM-HQ
> mask of the outer garment bleeds into the inner garment (i.e., the
> outerwear mask includes pixels belonging to the inner layer), the
> neckline-complement calculation is corrupted — the detector sees a
> smaller or distorted complement region, which directly degrades the
> inner mask and bounding box.


##### 4.6.1 Architecture

The subsystem consists of four modules plus a visualization script:

| Module | File | Role |
|---|---|---|
| Detector | `src/fashion_vision/localization/inner_garment_detector.py` | Primary entry point; neckline-complement candidate generation + scoring + downward extension |
| Boundary refiner | `src/fashion_vision/localization/inner_boundary_refiner.py` | Edge/colour/texture-based horizontal and vertical boundary adjustment |
| Mask cleaner | `src/fashion_vision/localization/inner_mask_cleaner.py` | Removes rectangular ROI cropping artifacts (upper-corner residues, side strips) |
| Torso prior | `src/fashion_vision/localization/torso_prior.py` | Builds torso mask from proxy bbox (or keypoints when available) to suppress off-torso candidates |
| Visualization | `scripts/visualize_inner_garment_detection.py` | 4×2 panel debug output; random sampling; summary JSON |
| Tests | `tests/test_inner_garment_detector.py` | 50 unit tests covering scoring, extension, refinement, cleanup, torso thresholds, viz robustness |

##### 4.6.2 Detection Pipeline (Two-Stage)

**Stage 1 — Primary: Neckline-complement geometric analysis**

The key insight: an inner garment visible through an open outerwear
front appears in the *complement* of the outerwear SAM mask — i.e., the
region that lies **inside** the outerwear bounding box and **inside**
the neckline/chest ROI, but **outside** the outerwear mask itself.

```
Image
  → YOLO detects outerwear (e.g. "long sleeve outwear")
  → SAM-HQ segments outerwear mask
  → Build complement search mask:
       inside outerwear_bbox ∩ neckline_ROI ∩ ¬outerwear_mask
  → Generate candidates from 3 sources:
       1. Connected components on complement mask
       2. Canny edge contours within neckline ROI
       3. SAM multimask on neckline ROI
  → Score each candidate (15 weighted metrics):
       - outside_outer_ratio (1.5×)    — how much lies outside outerwear mask
       - inside_bbox_ratio  (1.0×)     — containment within outerwear bbox
       - neckline_overlap   (1.5×)     — overlap with neckline ROI
       - opening_core_overlap (2.0×)   — overlap with centre-front opening core
       - center_score       (1.8×)     — proximity to bbox horizontal centre
       - torso_overlap      (2.0×)     — overlap with torso prior mask
       - upper_position     (1.0×)     — vertical position bias toward top
       - solidity           (1.0×)     — shape compactness
       - area_ratio_bbox    (0.5×)     — size relative to outerwear bbox
       - side_edge_penalty  (-2.0×)    — penalty for touching bbox side edges
  → Select highest-scoring candidate that passes all thresholds
  → Build seed mask from candidate
  → Extend downward through opening region (connected-component merge)
  → Result: seed_mask, seed_bbox, extended_mask
```

**Stage 2 — Fallback: SAM multimask on full outerwear bbox**

If the neckline-complement branch finds nothing (no candidates, or none
pass scoring), fall back to running SAM multimask on the entire
outerwear bounding box. Candidates are filtered by containment ratio,
area ratio, solidity, and edge-touch checks. This is a legacy path
kept for robustness; it is rarely triggered when the neckline branch is
working correctly.

##### 4.6.3 Post-Detection Refinement Pipeline

After the initial mask and bbox are obtained, three post-processing
steps are applied in order:

```
extended_mask, extended_bbox
    │
    ▼
[1] Boundary Refinement  (inner_boundary_refiner.py)
    - Horizontal: composite edge/colour/texture profile → outward peak search
    - Vertical: row-wise Lab-colour + Laplacian-texture scan downward
    - Safety gate: area_ratio ∈ [0.45, 2.80], bbox_area_ratio ∈ [0.45, 3.00],
      centre_shift ≤ 0.18·outer_w, torso_overlap not significantly degraded
    - 150-image run: 79 attempted, 1 accepted, 78 rejected (gate is strict)
    │
    ▼
[2] Artifact Cleanup  (inner_mask_cleaner.py)
    - Soft trapezoidal opening corridor replaces rectangular ROI →
      clips upper-corner right-angle residues
    - Upper-corner colour-consistency suppression (Lab ΔE threshold)
    - Side-strip detection & removal (tall narrow edge components)
    - Main-connected-component preservation with intelligent aux retention
    - Light morphology smoothing (3×3 open + close)
    - Safety gate: cleaned_area ≥ 0.45 · original_area
    - 150-image run: 69 accepted (87.3%), 10 rejected (12.7%),
      avg 7,123 pixels removed per accepted cleanup
    │
    ▼
[3] Torso Prior  (torso_prior.py, applied during scoring)
    - Proxy mode (default): narrower trapezoid from outerwear bbox
      → min_torso_overlap = 0.25 (softer, to avoid over-rejection)
    - Keypoint mode: convex hull of shoulder + hip landmarks
      → min_torso_overlap = 0.35 (stricter, points are reliable)
    - Candidates with insufficient torso overlap are rejected
```

##### 4.6.4 Torso Prior Design

When pose keypoints are unavailable (the common case in this pipeline),
a **proxy torso** is constructed from the outerwear bounding box:

```
Proxy ranges (fractions of outerwear bbox):
  x: 18% – 82%  (narrower than outerwear — removes sleeve regions)
  y:  3% – 88%  (top of chest to bottom of torso)
```

The proxy torso is intentionally *generous* — it over-covers rather
than under-covers. This is because false negatives (rejecting a valid
inner garment because it falls slightly outside the proxy torso) are
more harmful than false positives (accepting a candidate slightly
outside the ideal torso region). The softer `min_torso_overlap = 0.25`
threshold for proxy mode reflects this trade-off.

When keypoints are available, the convex hull of shoulder + hip points
is dilated (9×9 kernel, 2 iterations) to form a more accurate torso
polygon, and a stricter `min_torso_overlap = 0.35` threshold is used.

##### 4.6.5 Key Thresholds

| Parameter | Value | Rationale |
|---|---|---|
| Neckline ROI (x) | 18%–82% of outer_w | Focus on centre-front opening |
| Neckline ROI (y) | 3%–58% of outer_h | Upper portion only |
| Opening ROI (y) | 8%–85% of outer_h | Wider vertical range for downward extension |
| Opening core (x) | 25%–70% of outer_w | Tight centre for scoring |
| Minimum score | 3.0 | Weighted sum of 10 metric components |
| Minimum outside_outer_ratio | 0.45 | Candidate must be mostly outside outerwear mask |
| Minimum neckline_overlap | 0.60 | Strong overlap with neckline region required |
| Refine area_ratio bounds | [0.45, 2.80] | Prevent extreme mask changes |
| Refine centre_shift max | 0.18 × outer_w | ~14 px at 640×640 |
| Cleanup area_ratio min | 0.45 | Reject cleanup that removes >55% of mask |
| Cleanup colour ΔE threshold | 38.0 | Lab CIE76 distance for corner pixel removal |
| Proxy torso min_overlap | 0.25 | Softer — proxy is approximate |
| Keypoint torso min_overlap | 0.35 | Stricter — keypoints are reliable |

##### 4.6.6 Validation Results (150 images, lapel_design_labels)

**Date:** 2026-07-03 | **Seed:** 20260701 | **Device:** CUDA

| Metric | Value |
|---|---|
| Images processed | 150 |
| Outerwear found (DF2 YOLO) | 88 (58.7%) |
| Inner garment found | 79 (89.8% of outerwear) |
| Inner extended (downward merge) | 79 (100% of found) |
| Boundary refine attempted | 79 |
| Boundary refine accepted | 1 (1.3%) |
| Boundary refine rejected | 78 (98.7%) |
| Artifact cleanup accepted | 69 (87.3%) |
| Artifact cleanup rejected | 10 (12.7%) |
| Avg cleanup pixels removed | 7,123 |
| No inner found | 9 (10.2%) |

**Visual inspection notes:** Overall detection quality is decent by
visual inspection. Inner garment masks generally follow the visible
garment boundary. Upper-corner right-angle residues are substantially
reduced by the artifact cleanup module. The boundary refinement safety
gate is appropriately strict — most refinement attempts would have
degraded the mask and are correctly rejected.

**Top candidate rejection reasons (across all candidates):**
area_bbox (345), outside_outer (249), off_center (243), bbox_h (232),
bbox_w (204), low_opening_core (195).

##### 4.6.7 Known Limitations

1. **SAM-HQ outerwear mask quality is the dominant bottleneck.**
   When the outerwear SAM-HQ mask includes pixels belonging to the
   inner garment (mask bleeding), the complement region is reduced or
   fragmented. This directly degrades candidate generation — the
   detector may miss the inner garment entirely or produce a truncated
   mask. This is not a bug in the inner garment detector; it is a
   limitation of the upstream SAM-HQ segmentation. Mitigation would
   require either (a) improving SAM-HQ mask quality specifically for
   open-front outerwear, or (b) using a separate inner-garment SAM
   prompt that operates independently of the outerwear mask.

2. **No upper/lower garment distinction.** The detector treats all
   clothing visible inside the outerwear mask as a single "inner
   garment." It does not separate a shirt from trousers, or a tank top
   from a skirt. For queries requiring upper-body vs. lower-body inner
   garment discrimination, a separate vertical-segmentation step would
   be needed (likely keypoint-driven: hip line as the divider).

3. **Open-front coat ambiguity.** For garments like cardigans, unzipped
   jackets, and blazers worn open, the "complement" region is large and
   continuous with the background. The torso prior and neckline ROI
   constrain this sufficiently for detection, but the mask boundary
   between inner garment and background (not inner vs. outer) can be
   imprecise — particularly at the bottom edge where the inner garment
   tucks into pants/skirt.

4. **Single-instance output.** The current detector returns at most one
   inner garment per outerwear instance. If two inner layers are visible
   (e.g., a shirt *and* a tank top under an open jacket), only the
   highest-scoring candidate is returned. Multi-layer inner detection
   would require iterative masking (subtract the first inner garment and
   re-run detection).

5. **No integration with Fashionpedia part detector.** The inner garment
   detector and the 19-class Fashionpedia YOLO part detector
   (`models/detectors/fashionpedia_yolov8s_19cls_balanced_best.pt`)
   currently operate independently. Fashionpedia can detect collars,
   lapels, and necklines — these could provide stronger localization
   priors for the neckline-complement stage. Integration is deferred.

##### 4.6.8 Usage

```python
from fashion_vision.localization.inner_garment_detector import (
    detect_inner_garment_from_sam,
)

# outer_instance must have bbox_xyxy and a mask (array or path)
inner_result = detect_inner_garment_from_sam(
    image=bgr_image,           # np.ndarray H×W×3
    outer_instance=instance,   # dict with bbox_xyxy, _mask/mask_path
    sam_wrapper=sam,           # SamHqWrapper for SAM-based candidates
)
# Returns dict with bbox_xyxy, mask, score, debug or None
```

The detector is called on-demand from `region_localization_router.py`
when the user intent targets an inner layer and the selected instance
is classified as outerwear. It is **not** run globally on every image.

##### 4.6.9 Debug Visualization

```bash
python scripts/visualize_inner_garment_detection.py \
    --image-dir <path/to/images> \
    --num-images 150 \
    --output-dir outputs/visual_tests/inner_garment_v6 \
    --device cuda \
    --seed 20260701
```

Each output panel (`{stem}_inner_detection.png`) is a 4×2 grid showing:
1. Original image + YOLO outerwear bbox (yellow)
2. Outerwear SAM mask (orange overlay)
3. Neckline ROI (cyan) + opening ROI (purple)
4. Complement search mask (green overlay)
5–7. Top 3 candidates with PASS/REJECT status, scores, and all rejection reasons
8. Selected inner garment with torso ROI (magenta), opening ROI (purple),
   before-refine bbox (orange dashed), before-cleanup bbox (green dashed),
   inner mask (teal overlay), final bbox (blue solid), and subtitle
   showing: method, score, extension status, torso overlap, refine status,
   cleanup status with removed pixels and reason.

A `summary.json` is written to the output directory with counts and
top rejection reasons.

---

### Phase 5: Composite Anchor Handling

**Goal:** Handle compound queries like "外套左边口袋上的扣子" (button on the left pocket of the outerwear).

#### 5.1 Parsing Improvement

`intent_parser.py` already extracts `spatial_anchor`. For Phase 5, add a `target_part` field to `QueryIntent` when `spatial_anchor` is set: the `part` field becomes the target to find within the anchor.

Example: "口袋上的扣子" → `part=button`, `spatial_anchor=pocket`.

#### 5.2 One-Step Strategy with Prompt Ensemble (Correction 7)

For compound queries, generate a prompt ensemble that encodes the anchor-target relationship:

```python
def build_composite_prompts(target_part: str, anchor_part: str) -> list[str]:
    target_text = _PART_TO_GROUNDING_TEXT.get(target_part, target_part)
    anchor_text = _PART_TO_GROUNDING_TEXT.get(anchor_part, anchor_part)
    return [
        f"{target_text} on {anchor_text}",
        f"{target_text} near {anchor_text}",
        f"{target_text} attached to {anchor_text} on clothing",
    ]
```

Apply garment mask gating. Apply shape priors for the target part (not the anchor). If DINO returns a high-confidence detection → done.

**When to use one-step:** `target_part` is a small component (button, zipper, drawstring); `anchor_part` is a medium component (pocket, placket). One-step works when DINO has seen this visual concept in pretraining.

#### 5.3 Two-Step Strategy — With Side Constraints and Validation (Correction 7)

If one-step fails (no detection above threshold):

**Step A — Detect the anchor region with side constraint:**
1. Resolve prompts for `spatial_anchor` from `PART_DETECTION_CONFIG`.
2. If `intent.side` is set ("left" / "right"), run DINO for anchor, then apply `select_side_detection()` to the anchor candidates.
3. Apply shape priors for the anchor part.
4. If multiple anchor candidates survive: pick the highest-confidence one (or the side-selected one).
5. If anchor not found → `"status": "not_detected", "reason": "anchor_not_found"`.

**Step B — Detect target within the anchor crop:**
1. Crop to anchor bbox with 20% padding.
2. Resize the crop to 640×640 (record scale factors `scale_x`, `scale_y`).
3. Run DINO with prompts for the target part.
4. Apply shape priors using `garment_bbox = anchor_bbox` (relative to crop scale).
5. If target not found → `"status": "not_detected", "reason": "target_not_found_in_anchor"`.

**Coordinate remapping:**
```python
scale_x = anchor_crop_w / 640   # 640 = DINO input size
scale_y = anchor_crop_h / 640
full_x1 = anchor_x1 + det["bbox_xyxy"][0] * scale_x
full_y1 = anchor_y1 + det["bbox_xyxy"][1] * scale_y
full_x2 = anchor_x1 + det["bbox_xyxy"][2] * scale_x
full_y2 = anchor_y1 + det["bbox_xyxy"][3] * scale_y
```

**Target-center-inside-anchor validation (Correction 7):**  
After remapping, verify that the center of the target bbox falls within the anchor bbox (with 20% margin tolerance):
```python
target_cx = (full_x1 + full_x2) / 2
target_cy = (full_y1 + full_y2) / 2
margin_x = 0.20 * (anchor_x2 - anchor_x1)
margin_y = 0.20 * (anchor_y2 - anchor_y1)
if not (anchor_x1 - margin_x <= target_cx <= anchor_x2 + margin_x and
        anchor_y1 - margin_y <= target_cy <= anchor_y2 + margin_y):
    return "status": "not_detected", "reason": "target_outside_anchor_bounds"
```

If this check fails, the coordinate mapping is likely wrong or DINO found a different instance of the target. Do not return the result.

#### 5.4 Routing Decision

```
spatial_anchor is not None?
  YES:
    → try one-step (prompt ensemble, 3 prompts)
    → if status == "success": return
    → else: try two-step
      → if anchor found AND target found AND center-in-anchor check passes: return success
      → else: return not_detected with specific reason
  NO:
    → existing single-part detection path (Phase 1/2 fixes)
```

---

### Phase 6: Fine-Tuning Plan (DEFERRED)

**Status: NOT READY FOR IMPLEMENTATION.**

**Correction 13:** Phase 6 fine-tuning requires a **separate, dedicated fine-tuning design document** before any implementation begins. That document must specify:
- Precise data collection pipeline (images, labels, formats)
- Annotation protocol and quality control process
- Training configuration (learning rate, epochs, backbone freeze policy, augmentation)
- Evaluation protocol (metrics, checkpointing, early stopping)
- Checkpoint management and versioning
- A plan for integrating the fine-tuned model back into the localization router

**Trigger condition:** Phase 6 is justified ONLY IF:  
(a) Phase 2 calibration shows precision < 0.55 for any priority part, AND  
(b) Phase 2.4 shows that DINO-base does not close the gap.

If either condition is not met, Phase 6 is cancelled for this part.

**Preliminary scope (for planning only — details in the separate design doc):**

| Part | Min positive samples | Why this many |
|---|---|---|
| pocket | 150 positive + 100 negative | Common part; DINO moderately good; fewer needed |
| zipper | 120 positive + 80 negative | Elongated shape is distinctive; fewer needed |
| button_cluster | 200 positive + 150 negative | High FP risk from badges, decorations |
| drawstring | 100 positive + 80 negative | Rare; expect data scarcity |
| shoulder_seam | 150 positive + 100 negative | Low DINO pretraining coverage |

Model to fine-tune: **DINO-base** (`IDEA-Research/grounding-dino-base`), not DINO-tiny.  
Reason: fine-tuning a larger model yields better generalization on small datasets.

**Do not implement Phase 6 without the separate design document and mentor approval.**

---

## D. Human-in-the-Loop Checklist

Everything in this list requires your eyes and judgment. None of it can be automated.

### Before Phase 1 (Establish Baseline)
- [ ] **D1.** Run `scripts/run_open_vocab_yolo_crop_test.py` on 10–15 images with known garment parts. Inspect the HTML strip. Record which queries succeed and fail. This is your pre-fix baseline.

### After Phase 1 (Verify Fixes)
- [ ] **D2.** Run the new `scripts/visualize_localization_debug.py` on 20–30 images. For each: confirm the mask-gated crop (Panel 4) does not contain background, hair, or adjacent garments.
- [ ] **D3.** Inspect 10 "not_detected" cases. Decide: is the part truly absent, or is this a false negative? Document your finding.
- [ ] **D4.** For each new Chinese term added (Phase 1.4), run a test query and verify the correct part is found and the result makes visual sense.
- [ ] **D5.** Inspect at least 5 cases where `garment_ref_matched=False` appears. Confirm this flag is being set when expected.

### Phase 2 (Calibration)
- [ ] **D6.** Annotate 50–100 images in Label Studio for 5 part types. Aim for ~15 positive + 5 negative per part type. (2–4 hours.)
- [ ] **D7.** Run the threshold sweep. Inspect the precision-recall curves. Choose thresholds that give precision ≥ 0.65 where achievable.
- [ ] **D8.** Inspect 20 visualization outputs at the chosen threshold. Count how many you would accept as "correct" in a real product. Record this number as your manual acceptance rate.
- [ ] **D9.** Decide the acceptable precision-recall tradeoff with your mentor before locking thresholds.
- [ ] **D10.** Run DINO-base on the same validation set. Inspect results visually. Decide: does DINO-base justify the additional VRAM cost?

### Phase 3 (Qwen-VL Server)
- [ ] **D11.** After deploying the server, test 10 long-tail Chinese queries manually. Inspect the JSON responses. Verify they are reasonable English grounding phrases for garment context.
- [ ] **D12.** Test the timeout path: stop the server, submit a query, confirm `source="fallback_literal"` appears in the result and the system does not crash.
- [ ] **D13.** Test the cache: submit the same query twice. Verify the second call does not appear in server logs.

### Phase 4 (Hierarchy)
- [ ] **D14.** Inspect SAM-HQ masks on 20 multi-garment images BEFORE writing containment code. Decide: are the masks clean enough to support containment math?
- [ ] **D15.** After implementing containment, inspect 15–20 multi-garment examples. Does the system correctly identify inner vs. outer? Note failures and adjust thresholds.
- [ ] **D16.** Inspect all "ambiguous" and "open_front_adjacent" cases. Decide: is the uncertainty signal useful, or does it block too many legitimate queries?

### Phase 5 (Composite Anchors)
- [ ] **D17.** Test "口袋上的扣子" (button on pocket) one-step on 10 images. Inspect overlays. Do the results make sense?
- [ ] **D18.** For two-step: inspect the anchor crop panel and the final bbox on the original image. Do they visually align? Is the target-center-inside-anchor check working?
- [ ] **D19.** Collect 5–10 compound query examples from realistic user scenarios. Test each one. Record pass/fail.

### Phase 6 (Fine-Tuning, If Triggered)
- [ ] **D20.** Read and approve the separate fine-tuning design document before any annotation begins.
- [ ] **D21.** Annotate data in Label Studio per the design document's protocol.
- [ ] **D22.** After training, inspect 20 predictions on the held-out test set. Compare against Phase 2 baseline visually.

---

## E. Code Modification Plan

| File | Change | Risk | Test | Phase |
|---|---|---|---|---|
| `region_localization_router.py` | Unify `_crop_to_instance` and mask crop into `_crop_image_and_mask()`; pass `crop_mask` to DINO | Low | Unit test: mask pixels outside bbox become 128 | 1 |
| `region_localization_router.py` | Replace `"failed"` with 5-status vocabulary; add `"reason"` enum | Low | Update existing tests; new "not_detected" tests | 1 |
| `region_localization_router.py` | Add `garment_ref_matched=False` flag when class mismatches | Low | Unit test with mismatching instance | 1 |
| `grounding_dino_locator.py` | Add `fill_mode` parameter to `mask_to_garment()`; add `dilation_px` support | Low | Unit test fill mode options | 1 |
| `part_shape_priors.py` | Remove fallback-best-candidate; return `[]`; add `allow_fallback=False` param | Medium | Update 5 tests; add "all rejected → empty" test | 1 |
| `part_detection_config.py` | Add `mask_dilation_px` field; add `shoulder_seam`, `sleeve_seam`, `ruffle`, `fringe`, `tie_strap` | Low | get_part_shape_config for new parts | 1 |
| `intent_parser.py` | Add 6 new parts to `PART_VOCAB` and `_PART_TO_GROUNDING_TEXT` | Low | Parameterized test per new Chinese term | 1 |
| `tests/test_phase2_localization.py` | Update 5 wrong-fallback tests; add new tests | Low | Run pytest | 1 |
| `scripts/visualize_localization_debug.py` | New 7-panel debug visualization | Low | Manual run on 1 image | 1 |
| `translation_service.py` | New: TranslationService with Qwen-VL client, in-memory cache, timeout, fallback | Medium | Unit tests with mocked HTTP server | 3 |
| `mask_containment.py` | New: ContainmentResult, compute_containment(), compute_inter_garment_seam_bbox() | Low | Synthetic mask unit tests | 4 |
| `garment_ref_filter.py` | Replace area-sort for "inner" with containment + rule ensemble | Medium | Synthetic mask tests; inspect real images | 4 |
| `region_localization_router.py` | Add two-step composite anchor routing; coordinate remapping; center-in-anchor validation | High | Unit test remapping math; integration test | 5 |

---

## F. Test Plan

### F.1 Unit Tests (No GPU Required)

| Test file | Test | Assertion |
|---|---|---|
| `test_intent_parser_extended.py` | New Chinese terms parse correctly | `parse_intent("荷叶边").part == "ruffle"` |
| `test_intent_parser_extended.py` | drawstring reachable | `parse_intent("抽绳").part == "drawstring"` |
| `test_intent_parser_extended.py` | Seam terms route to correct canonical part | `parse_intent("肩缝").part == "shoulder_seam"` |
| `test_intent_parser_extended.py` | True zero-shot (unknown term) | `parse_intent("神秘装饰").is_zero_shot == True` |
| `test_phase2_localization.py` | All shape priors rejected → empty list | `filter_by_shape_priors([wide], "zipper", G) == []` |
| `test_phase2_localization.py` | All rejected → status code is "not_detected" | router-level integration test with mocked DINO |
| `test_router_helpers.py` | Same crop box for image and mask | crop coordinates identical |
| `test_router_helpers.py` | Mask/image shape mismatch → INTER_NEAREST resize | pixel value assertion after resize |
| `test_router_helpers.py` | Mask dilation widens garment region | dilated mask has more non-zero pixels |
| `test_router_helpers.py` | fill_mode="black" sets bg to 0 | pixel value check |
| `test_translation_service.py` | Cached query does not call HTTP | mock HTTP called 0 times on second call |
| `test_translation_service.py` | Timeout returns fallback_literal | `result.source == "fallback_literal"` |
| `test_translation_service.py` | Invalid JSON returns fallback_literal | same |
| `test_translation_service.py` | Valid JSON parsed into TranslationResult | `result.english_phrase == "shoulder seam"` |
| `test_translation_service.py` | `translation_warning` set in router result | `result["translation_warning"] == True` |
| `test_mask_containment.py` | A entirely inside B | `result.relationship == "a_inside_b"` |
| `test_mask_containment.py` | Disjoint masks | `result.relationship == "disjoint"` |
| `test_mask_containment.py` | 50% overlap → ambiguous | `result.relationship == "ambiguous"` |
| `test_mask_containment.py` | Seam bbox computed for adjacent masks | seam bbox is non-None and within image bounds |
| `test_composite_anchor.py` | Coordinate remapping: known bbox in known crop | pixel-exact assertion |
| `test_composite_anchor.py` | Center-inside-anchor check rejects out-of-bounds target | `reason == "target_outside_anchor_bounds"` |
| `test_composite_anchor.py` | Center-inside-anchor with 20% tolerance | margin boundary cases |

### F.2 Tests That Must Be Updated (Phase 1)

| Test | Old assertion | New assertion |
|---|---|---|
| `test_zipper_rejects_wide_box` | `"fallback_best_candidate_after_all_rejected"` | `result == []` |
| `test_belt_rejects_vertical_box` | `"fallback_best_candidate_after_all_rejected"` | `result == []` |
| `test_button_rejects_box_exceeding_max_area` | `"fallback_best_candidate_after_all_rejected"` | `result == []` |
| `test_button_rejects_off_center_box` | `"fallback_best_candidate_after_all_rejected"` | `result == []` |
| `test_fallback_returns_highest_score` | 1 result with score=0.8 | `result == []` (rename test) |

### F.3 Integration Tests (GPU Not Required — Mock DINO)

| Test | Description |
|---|---|
| Router: garment mask passed to DINO | Mock DINO, inspect call args: `garment_mask is not None` |
| Router: `not_detected` when all shape priors fail | Inject pre-rejected detections; assert status |
| Router: `garment_ref_matched=False` when class mismatches | Instance class != intent.garment_ref |
| Router: 5 status vocabulary — no `"failed"` anywhere | Grep the result dicts returned by mock tests |
| Composite anchor: center-in-anchor blocks out-of-anchor detection | Inject remapped bbox outside anchor |

---

## G. Failure Modes and Fallback Behavior

### G.1 Five-Status API Contract (Correction 8)

All callers of `locate_region()` must handle these five statuses:

| Status | When returned | `reason` field present? |
|---|---|---|
| `success` | Part found, score above threshold, passed shape priors | No (implied) |
| `not_detected` | Part was searched but not found (no boxes, all below threshold, all rejected by shape priors or spatial constraints, target not found in anchor) | **YES — always** |
| `unsupported_query` | Part not in vocab, translation service unavailable or returned empty, garment category does not support this part | **YES** |
| `uncertain` | Score in borderline range [low_thresh, high_thresh]; hierarchy relationship ambiguous; translation source is fallback_literal with no confident result | **YES** |
| `error` | System precondition failure: model not loaded, image unreadable, server unreachable, image/mask shape error | **YES** |

**Contract:** The `"status"` field is always present. `"reason"` is present for all non-success statuses. `"bbox"` is null for non-success. `"score"` is present for `"uncertain"` (the borderline score) and null otherwise for non-success.

### G.2 Per-Failure-Mode Behavior Table

| Failure | Status | Reason | Log Level | Notes |
|---|---|---|---|---|
| No garment mask available | `success` or `not_detected` | N/A | WARNING | Run without mask; add `"mask_gated": false` to result |
| Garment mask file corrupt / unreadable | Same as above | N/A | WARNING | Log exception |
| DINO returns no boxes (above threshold) | `not_detected` | `no_detection_above_threshold` | INFO | Normal for absent parts |
| DINO boxes all fail shape priors | `not_detected` | `no_detection_passed_shape_priors` | INFO | Shape rejection reasons in debug |
| All boxes fail spatial constraint (side/direction) | `not_detected` | `no_detection_passed_spatial_constraint` | WARNING | Note: fallback to unfiltered is REMOVED — return not_detected |
| Score in [low_threshold, high_threshold] | `uncertain` | `borderline_confidence` | INFO | Include score in result |
| Qwen-VL timeout | Continue with fallback | N/A | WARNING | `translation_warning: true` in result |
| Qwen-VL invalid JSON | Continue with fallback | N/A | WARNING | Same |
| Qwen-VL server unreachable | Continue with fallback | N/A | ERROR | Log server URL and error |
| Part not in vocab AND translation unavailable | `unsupported_query` | `part_not_translatable` | INFO | |
| Composite anchor not found | `not_detected` | `anchor_not_found` | INFO | |
| Target not in anchor crop | `not_detected` | `target_not_found_in_anchor` | INFO | |
| Target center outside anchor bounds | `not_detected` | `target_outside_anchor_bounds` | INFO | Coordinate issue |
| Hierarchy ambiguous | `uncertain` | `hierarchy_ambiguous` | INFO | Include containment ratios in result |
| SAM-HQ box prompt fails / empty mask | Bbox-fill fallback | N/A | WARNING | `mask_source: "bbox_fill"` in result |
| Image read fails | `error` | `image_read_error` | ERROR | Raise to caller |
| Model not loaded | `error` | `model_not_loaded` | ERROR | Do not silently return empty |

### G.3 Debug Directory Structure and Metadata Schema (Correction 10)

**Directory layout:**
```
outputs/debug_runs/
  {run_id}/
    run_metadata.json
    {image_stem}/
      {query_id}/
        01_garment_mask.png          # garment mask overlaid on image
        02_bbox_crop_raw.png         # plain bbox crop (what old code sent to DINO)
        03_masked_crop_dino.png      # mask-gated crop (what fixed code sends to DINO)
        04_dino_raw_candidates.png   # all DINO boxes before NMS, with scores
        05_after_nms.png             # after NMS, before shape filter
        06_shape_filtered.png        # green=passed, red=rejected+reason text
        07_final_result.png          # final bbox (blue) on original image, or NOT DETECTED banner
        result.json                  # full result + latency metadata
```

**`run_metadata.json` schema:**
```json
{
  "run_id": "20260623_143022_abc1",
  "timestamp": "2026-06-23T14:30:22",
  "model_version": "IDEA-Research/grounding-dino-tiny",
  "config_version": "v1.2",
  "pipeline_version": "3.1.2-phase2"
}
```

**`result.json` schema (per query):**
```json
{
  "run_id": "20260623_143022_abc1",
  "image_id": "000004",
  "query_id": "q001",
  "query": "外套的口袋",
  "model_version": "IDEA-Research/grounding-dino-tiny",
  "config_version": "v1.2",
  "thresholds": {"box_threshold": 0.32, "text_threshold": 0.28},
  "prompts_used": ["clothing pocket", "pocket on garment", "chest pocket"],
  "status": "not_detected",
  "reason": "no_detection_passed_shape_priors",
  "bbox": null,
  "score": null,
  "mask_source": null,
  "mask_gated": true,
  "mask_dilation_px": 0,
  "translation_source": "local_vocab",
  "translation_warning": false,
  "garment_ref_matched": true,
  "intent": {
    "part": "pocket",
    "side": null,
    "garment_ref": "outerwear",
    "direction": null,
    "spatial_anchor": null,
    "is_fast_path": false,
    "is_zero_shot": false
  },
  "debug": {
    "n_candidates_before_nms": 3,
    "n_candidates_after_nms": 2,
    "n_candidates_after_shape": 0,
    "shape_rejection_reasons": [
      "area_ratio 0.38 > max 0.25",
      "area_ratio 0.29 > max 0.25"
    ]
  },
  "latency": {
    "intent_parse_ms": 0.3,
    "translation_ms": 0.0,
    "mask_load_ms": 4.2,
    "mask_crop_apply_ms": 1.1,
    "dino_ms": 187.5,
    "nms_ms": 0.4,
    "shape_filter_ms": 0.9,
    "sam_ms": 0.0,
    "total_ms": 194.4
  }
}
```

### G.4 Latency Budgets (Correction 11)

Target per-component latency on GPU (RTX 3090):

| Component | Budget | Current (estimated) | Notes |
|---|---|---|---|
| Intent parsing | < 1 ms | ~0.3 ms | Already fast |
| Local vocab lookup | < 1 ms | ~0.1 ms | Dict lookup |
| Qwen-VL translation (if called) | < 3000 ms | 500–1500 ms | Cached after first call |
| Mask load from PNG | < 10 ms | ~4–8 ms | Disk I/O dependent |
| Mask crop + apply | < 5 ms | ~1–2 ms | Simple array ops |
| DINO inference (GPU) | < 200 ms | ~150–200 ms | Per-prompt; ~50ms/prompt |
| NMS | < 2 ms | ~0.5 ms | CPU, fast |
| Shape priors | < 5 ms | ~1 ms | CPU, fast |
| SAM refinement (GPU, if used) | < 500 ms | ~300–400 ms | Optional |
| **Total — no SAM, no Qwen** | **< 250 ms** | — | Common fast path |
| **Total — with SAM** | **< 750 ms** | — | Full quality path |
| **Total — with Qwen + SAM** | **< 1500 ms** | — | First-call only; cached |

**If these budgets are violated:** Measure each component with `time.perf_counter_ns()` using the `latency` dict in `result.json`. Do not guess. Fix the slowest component first.

---

## Risks and Assumptions

### Risks

| Risk | Severity | Mitigation |
|---|---|---|
| vLLM does not support Qwen-VL-7B-Chat at the time of deployment | High | Use transformers + FastAPI fallback (Phase 3.3); test vLLM first |
| SAM-HQ mask quality too poor for containment math (occluded garments) | High | Phase 4.1 human inspection; do not write containment code until masks are verified |
| PART_DETECTION_CONFIG thresholds are wrong (designed, not measured) | High | Phase 2 empirical calibration; do not skip |
| Mask gating clips valid part pixels near garment boundary | Medium | Add `mask_dilation_px` per part; tune empirically |
| Button "circularity" check at bbox level has many false positives (square badges, buckles) | Medium | Phase 2 evaluation will surface this; Phase 6 fine-tuning if needed |
| Composite anchor two-step accumulates coordinate error on resize | Medium | Explicit center-in-anchor validation; unit test the remapping |
| Left/right convention (image-left vs. person-anatomical-left) unverified | Medium | Empirical test on known images; from current_project_status.md open gap |
| Qwen-VL-7B-Chat translations imprecise for long-tail fashion terms | Low | Manual inspection of 10 samples (D11); add to local vocab if pattern recurs |
| Open-front garments produce ambiguous hierarchy (always) | Low | Document as known limitation; return "ambiguous" status |
| Shape prior thresholds too strict → high false negative rate after removing fallback | Medium | Phase 2 FN measurement; relax per-part if FN rate unacceptable |

### Assumptions

1. SAM-HQ masks are the authoritative source for garment masks. If they are wrong, both mask gating and hierarchy are compromised.
2. DINO-tiny (zero-shot, no fine-tuning) has enough capacity for common small parts after mask gating and shape priors. If not, Phase 2.4 tests DINO-base before Phase 6 fine-tuning.
3. The Qwen-VL-7B-Chat server is accessible from the local project machine via SSH tunnel or environment variable. Cold-start latency (model loading) is not counted in the per-call budget.
4. "Left" in `select_side_detection()` means image-left (viewer perspective). This is NOT verified empirically and may be person-anatomical-left in the landmark pipeline. Test before trusting side-query results.
5. Per-part `max_area_ratio` in `PART_DETECTION_CONFIG` governs area constraints. There is no global 5% cap. The 5% figure from the mentor's notes is a rough default for very tiny decorative elements only.

---

## What Can Be Implemented Immediately vs. Later

### Immediate (no new dependencies, no server, no new models):
- Phase 1.1–1.6: All five Phase 1 items + debug visualization

### Requires your annotation time but no new tools beyond Label Studio:
- Phase 2.1: Validation set (Label Studio, 2–4 hours annotation)
- Phase 2.2–2.3: Threshold calibration (once validation set exists)
- Phase 2.4: DINO-base comparison (download DINO-base, run calibration script)

### Requires renting a GPU server:
- Phase 3: Qwen-VL-7B-Chat translation service

### Requires pure numpy, no new models:
- Phase 4: mask_containment.py (after Phase 4.1 human mask inspection)
- **Phase 4.6 (Inner garment detection): IMPLEMENTED** — detector, refiner, cleaner, torso prior all built and validated on 150 images. See §4.6.

### Requires careful implementation and testing:
- Phase 5: Composite anchor routing (coordinate remapping is error-prone)

### Blocked by Phase 2 results AND requires separate design document:
- Phase 6: Fine-tuning (do not begin without evidence from Phase 2 and a written design doc)

---

*End of plan v1.2. Phase 1 complete, Phase 4.6 (inner garment) implemented. Remaining phases pending approval and resources.*
