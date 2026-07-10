"""
Unified region localization router (3.1.2).

Single entry point for all region queries regardless of whether they route to
the fast path (landmark + geometry) or the open-vocabulary path (Grounding DINO
+ SAM box-prompt mask).

Routing decision:
    Fast path  : intent.is_fast_path == True (6 structural landmark parts)
    Open-vocab : intent.is_fast_path == False AND intent.part is known vocab
    Zero-shot  : intent.is_zero_shot == True (part not in any vocabulary)

All paths return a unified result dict (see _build_result_base).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2
import numpy as np

from fashion_vision.localization.intent_parser import parse_intent, _zero_shot_noun_phrase
from fashion_vision.localization.anatomical_zoom import (
    apply_anatomical_zoom,
    map_box_from_zoom_to_original,
)
from fashion_vision.localization.region_locator import (
    load_binary_mask,
    locate_region_from_instance,
    resolve_instance_mask_path,
)
from fashion_vision.localization.spatial_constraint import (
    select_side_detection,
    select_direction_detection,
)

if TYPE_CHECKING:
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    from fashion_vision.localization.fashionpedia_part_detector import FashionpediaPartDetector
    from fashion_vision.localization.bbox_mask_refiner import BboxMaskRefiner
    from fashion_vision.models.sam_hq_wrapper import SamHqWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def locate_region(
    query: str,
    instance: Dict[str, Any],
    image: np.ndarray,
    image_width: int,
    image_height: int,
    locator: Optional["GroundingDINOLocator"] = None,
    dino_threshold: float = 0.3,
    prefer_pred_mask: bool = True,
    sam_wrapper: Optional["SamHqWrapper"] = None,
    fashionpedia_detector: Optional["FashionpediaPartDetector"] = None,
) -> Dict[str, Any]:
    """
    Route a natural-language region query to the appropriate backend.

    Args:
        query: Natural language query (Chinese or English).
        instance: Standardised instance record (from instance_schema).
        image: Full image in BGR uint8 H×W×3.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        locator: GroundingDINOLocator instance.  Required for open-vocab and
            zero-shot queries (unless *fashionpedia_detector* covers the part).
            Caller owns the model lifecycle.
        dino_threshold: Minimum score for Grounding DINO detections.
        prefer_pred_mask: Passed to fast-path locate_region_from_instance.
        sam_wrapper: SamHqWrapper for bbox→mask refinement.  If None, the
            open-vocab path returns a bbox-fill pseudo-mask.
        fashionpedia_detector: FashionpediaPartDetector instance.  When provided
            and the queried part is in Fashionpedia core coverage, YOLO runs
            first; DINO is only invoked as fallback when YOLO returns nothing.

    Returns:
        Unified result dict — see _build_result_base() for key set.

    Raises:
        ValueError: If query routes to open-vocab/zero-shot but neither
            *locator* nor *fashionpedia_detector* for the part is provided.
    """
    intent = parse_intent(query)

    # Phase 1.5: surface garment_ref mismatch before routing so both fast-path
    # and open-vocab results carry an accurate garment_ref_matched flag.
    if intent.garment_ref is not None:
        _flag_garment_ref_mismatch(intent.garment_ref, instance)

    # ── Inner-garment detection (SAM-based) ─────────────────────────────
    # When the user targets the inner layer ("内搭") and the selected instance
    # is an outerwear, YOLO often misses the inner top due to occlusion.
    # Use SAM multimask on the full outerwear bbox to find the inner garment's
    # visible contour — which may appear anywhere inside the outerwear
    # (collar, chest opening, abdomen, near the hem).
    if (intent.garment_ref == "inner"
        and instance.get("coarse_class_name") == "outerwear"
        and sam_wrapper is not None):

        from fashion_vision.localization.inner_garment_detector import (
            detect_inner_garment_from_sam,
        )
        logger.info(
            "locate_region: attempting inner-garment detection — "
            "garment_ref=%r, instance coarse_class=%r",
            intent.garment_ref, instance.get("coarse_class_name"),
        )
        inner_inst = detect_inner_garment_from_sam(image, instance, sam_wrapper)
        if inner_inst is not None:
            # Replace garment_mask with inner mask so DINO localizes parts
            # relative to the inner garment, not the outerwear.
            inner_mask = inner_inst.get("mask")
            if inner_mask is not None:
                garment_mask = inner_mask
            # Use inner bbox for cropping (but keep outer bbox for context)
            inner_bbox = inner_inst.get("bbox_xyxy")
            if inner_bbox is not None:
                inst_bbox = inner_bbox
            instance["_garment_ref_mismatch"] = False
            instance["_inner_garment_detected"] = True
            instance["_inner_garment_debug"] = inner_inst.get("debug", {})
            logger.info(
                "locate_region: inner garment detected via SAM — bbox=%s",
                inner_bbox,
            )
        else:
            # Graceful degradation: continue with outerwear instance.
            # The result will carry garment_ref_matched=False so downstream
            # knows the query didn't match the garment type.
            logger.info(
                "locate_region: inner-garment detection failed — "
                "falling back to outerwear-only path",
            )

    # ------------------------------------------------------------------
    # Fast path — landmark + geometry (hem, waist, shoulder, leg_opening)
    # ------------------------------------------------------------------
    if intent.is_fast_path:
        return _build_fast_path_result(
            query, intent, instance, image_width, image_height, prefer_pred_mask,
        )

    # ------------------------------------------------------------------
    # Open-vocab / zero-shot path — Fashionpedia YOLO → Grounding DINO fallback
    # ------------------------------------------------------------------
    _fp_available = (
        fashionpedia_detector is not None
        and _is_fashionpedia_part(intent.part)
    )
    # Neckline / cuff can fall back to fast-path even when no detectors
    # are available — don't raise, let the FP-miss fallback handle it.
    _has_fast_fallback = intent.part in ("neckline", "cuff")
    if locator is None and not _fp_available and not _has_fast_fallback:
        raise ValueError(
            f"Query {query!r} maps to open-vocab/zero-shot part {intent.part!r} "
            "but no GroundingDINOLocator or FashionpediaPartDetector was provided "
            "to locate_region()."
        )

    # ── Neckline / cuff: when no FP detector available, go straight to fast-path ──
    if _has_fast_fallback and not _fp_available:
        return _build_fast_path_result(
            query, intent, instance, image_width, image_height, prefer_pred_mask,
        )

    garment_mask = _load_garment_mask(instance, prefer_pred_mask)

    # Resolve per-part box threshold (shared by FP and DINO).
    box_threshold = _resolve_box_threshold(intent.part, fallback=dino_threshold)

    # Resolve instance bbox for garment-crop and direction filtering.
    inst_bbox = _instance_bbox(instance)

    # Crop image AND mask.
    # Fashionpedia YOLO was trained on full images (no anatomical cropping) —
    # running it on zoomed sub-regions is a distribution shift.  Route FP YOLO
    # through the garment-crop-only path; DINO continues to use anatomical zoom
    # which helps open-vocab detection by giving small parts more pixel budget.
    if inst_bbox is not None and intent.part:
        if _fp_available:
            crop_image, crop_mask, zoom_transform = _crop_image_and_mask_with_transform(
                image, garment_mask, inst_bbox,
            )
        else:
            crop_image, crop_mask, zoom_transform = apply_anatomical_zoom(
                image, garment_mask, inst_bbox, intent.part,
            )
    elif inst_bbox is not None:
        crop_image, crop_mask, zoom_transform = _crop_image_and_mask_with_transform(
            image, garment_mask, inst_bbox,
        )
    else:
        crop_image = image
        crop_mask = garment_mask
        zoom_transform = {
            "offset_x": 0, "offset_y": 0,
            "scale_x": 1.0, "scale_y": 1.0,
            "crop_box": [0, 0, image.shape[1], image.shape[0]],
            "garment_bbox": [0, 0, image.shape[1], image.shape[0]],
            "zoom_applied": False, "zoom_factor": 1.0, "part": "__full_image__",
        }

    # ── Fashionpedia YOLO priority path ──────────────────────────────────────
    # EARLY RETURN: when FP hits, DINO is never called.  Prompts, text_threshold,
    # and mask dilation are resolved lazily inside the DINO fallback block.
    fp_detections: List[Dict[str, Any]] = []
    if _fp_available:
        fp_detections = fashionpedia_detector.detect(  # type: ignore[union-attr]
            crop_image, intent.part, garment_mask=crop_mask, conf=box_threshold,
        )
        # Remap bboxes from zoom-crop → full-image coords (shared with DINO path).
        for d in fp_detections:
            d["bbox_xyxy"] = map_box_from_zoom_to_original(d["bbox_xyxy"], zoom_transform)

    if fp_detections:
        # ── Fashionpedia hit — early return path (DINO never reached) ─────
        logger.info(
            "locate_region: Fashionpedia YOLO hit — part=%r n=%d top_score=%.3f "
            "(DINO skipped)",
            intent.part, len(fp_detections), fp_detections[0]["score"],
        )
        detections = fp_detections
        backend_label = "fashionpedia_yolo"
        n_before_nms = len(detections)
        n_after_nms = len(detections)  # YOLO does its own NMS internally
        # FP path doesn't use DINO prompts or dilation; fill placeholders for debug.
        prompts = [f"<fashionpedia_yolo:{intent.part}>"]
        text_threshold = box_threshold
        _dilation_px = 0
    else:
        # ── Fashionpedia miss ────────────────────────────────────────────
        # Neckline / cuff: fast-path (landmark) fallback.
        if intent.part in ("neckline", "cuff"):
            return _build_fast_path_result(
                query, intent, instance, image_width, image_height, prefer_pred_mask,
            )

        # Other Fashionpedia-core parts: no DINO fallback.
        # If FP YOLO didn't find the part, the garment genuinely lacks it
        # (e.g. no pocket, no zipper).  DINO would only hallucinate.
        if _fp_available:
            logger.info(
                "locate_region: Fashionpedia miss — part=%r (DINO skipped, returning not_detected)",
                intent.part,
            )
            prompts = [f"<fashionpedia_yolo:{intent.part}>"]
            text_threshold = box_threshold
            _dilation_px = 0
            _debug = _build_debug(
                prompts, box_threshold, text_threshold,
                0, 0, pre_shape=[], rejected=[],
                mask_gated=crop_mask is not None,
                dilation_px=0,
                zoom_applied=zoom_transform.get("zoom_applied", False),
                zoom_factor=zoom_transform.get("zoom_factor", 1.0),
                backend="fashionpedia_yolo",
            )
            base = _build_result_base(intent, instance, "fashionpedia_yolo", prompts)
            return {
                **base,
                "status": "not_detected",
                "reason": "fashionpedia_no_detection",
                "bbox": None,
                "score": None,
                "debug": _debug,
            }

        # ── Grounding DINO fallback (only for parts NOT in Fashionpedia) ─
        if locator is None:
            raise ValueError(
                f"Query {query!r} maps to part {intent.part!r} which is not in "
                "Fashionpedia core coverage and no GroundingDINOLocator was provided."
            )
        # Resolve DINO prompts + thresholds (deferred — only when needed).
        prompts, _, _, text_threshold = _resolve_prompts(
            intent, query, fallback_threshold=dino_threshold,
        )
        # Per-part mask dilation (deferred — only for DINO path).
        from fashion_vision.localization.part_detection_config import get_part_shape_config
        _dilation_px: int = get_part_shape_config(intent.part or "").get("mask_dilation_px", 0)

        # return_raw_count=True gives pre-NMS count for debug without extra model calls.
        detections, n_before_nms = locator.detect_multi_prompt(
            crop_image,
            prompts,
            garment_mask=crop_mask,
            threshold=box_threshold,
            return_raw_count=True,
            dilation_px=_dilation_px,
            nms_mode="soft",
            part=intent.part,
        )
        n_after_nms = len(detections)
        backend_label = "open_vocab_grounding_dino"

        # Remap DINO boxes from zoomed/cropped space back to full-image coords.
        for d in detections:
            d["bbox_xyxy"] = map_box_from_zoom_to_original(d["bbox_xyxy"], zoom_transform)

    # Apply spatial constraints: side first, then direction.
    if intent.side is not None:
        detections = select_side_detection(detections, intent.side)

    if intent.direction is not None and inst_bbox is not None:
        detections = select_direction_detection(detections, intent.direction, inst_bbox)

    # Early-exit when no detection found above the score threshold.
    if not detections:
        _debug = _build_debug(
            prompts, box_threshold, text_threshold,
            n_before_nms, n_after_nms,
            pre_shape=[], rejected=[],
            mask_gated=crop_mask is not None,
            dilation_px=_dilation_px,
            zoom_applied=zoom_transform.get("zoom_applied", False),
            zoom_factor=zoom_transform.get("zoom_factor", 1.0),
            backend=backend_label,
        )
        base = _build_result_base(intent, instance, backend_label, prompts)
        return {
            **base,
            "status": "not_detected",
            "reason": "no_detection_above_threshold",
            "bbox": None,
            "score": None,
            "debug": _debug,
        }

    # Shape priors filter — save references BEFORE filtering so we can collect
    # rejected candidates (dicts are mutated in-place by filter_by_shape_priors).
    pre_shape_detections: List[Dict[str, Any]] = list(detections)
    n_before_shape = len(pre_shape_detections)

    if intent.part and inst_bbox is not None:
        from fashion_vision.localization.part_shape_priors import filter_by_shape_priors
        detections = filter_by_shape_priors(detections, intent.part, inst_bbox)
    n_after_shape = len(detections)

    # Collect rejected candidates for debug / visualization.
    rejected_candidates = [
        {
            "bbox_xyxy": list(d["bbox_xyxy"]),
            "score": round(float(d["score"]), 4),
            "reasons": list(d.get("_shape_prior_reasons", [])),
        }
        for d in pre_shape_detections
        if d.get("_shape_prior_status") == "rejected"
    ]

    _debug = _build_debug(
        prompts, box_threshold, text_threshold,
        n_before_nms, n_after_nms,
        pre_shape=[
            {"bbox_xyxy": list(d["bbox_xyxy"]), "score": round(float(d["score"]), 4)}
            for d in pre_shape_detections
        ],
        rejected=rejected_candidates,
        mask_gated=crop_mask is not None,
        dilation_px=_dilation_px,
        zoom_applied=zoom_transform.get("zoom_applied", False),
        zoom_factor=zoom_transform.get("zoom_factor", 1.0),
        backend=backend_label,
    )

    base = _build_result_base(intent, instance, backend_label, prompts)

    if not detections:
        return {
            **base,
            "status": "not_detected",
            "reason": "no_detection_passed_shape_priors",
            "bbox": None,
            "score": None,
            "debug": _debug,
        }

    top = detections[0]
    top_bbox = top["bbox_xyxy"]
    _debug["shape_prior_status"] = top.get("_shape_prior_status")

    # Generate mask via SAM box prompt.
    mask_result = _refine_mask(image, top_bbox, garment_mask, sam_wrapper)

    return {
        **base,
        "status": "success",
        "bbox": top_bbox,                                      # backward-compat: top-1
        "all_bboxes": [d["bbox_xyxy"] for d in detections],   # all kept after shape filter
        "all_scores": [d["score"] for d in detections],
        "bbox_format": "xyxy",
        "score": top["score"],
        "label": top.get("label", ""),
        "mask": mask_result["mask"],
        "mask_source": mask_result["mask_source"],
        "mask_score": mask_result["score"],
        "debug": _debug,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_fast_path_result(
    query: str,
    intent: Any,
    instance: Dict[str, Any],
    image_width: int,
    image_height: int,
    prefer_pred_mask: bool = True,
) -> Dict[str, Any]:
    """Call locate_region_from_instance and enrich with unified schema fields.

    Used by both the pure fast-path branch and the neckline/cuff
    Fashionpedia-miss fallback.
    """
    result = locate_region_from_instance(
        instance=instance,
        query=query,
        image_width=image_width,
        image_height=image_height,
        prefer_pred_mask=prefer_pred_mask,
    )
    result["method"] = "fast_path"
    result.setdefault("part", intent.part)
    result.setdefault("garment_ref", intent.garment_ref)
    result.setdefault("direction", intent.direction)
    result.setdefault("is_zero_shot", False)
    result.setdefault("backend", "fast_path")
    result.setdefault("score", result.get("score", None))
    result.setdefault("prompts_used", None)
    result.setdefault(
        "garment_ref_matched",
        not instance.get("_garment_ref_mismatch", False),
    )
    result.setdefault("mask_source", result.get("mask_source", "landmark_crop"))
    # Ensure debug dict carries backend label when present.
    if "debug" in result and isinstance(result["debug"], dict):
        result["debug"].setdefault("backend", "fast_path")
    return result


def _resolve_box_threshold(part: Optional[str], fallback: float = 0.3) -> float:
    """Return the per-part box_threshold, or *fallback* if unregistered."""
    if not part:
        return fallback
    from fashion_vision.localization.part_detection_config import get_part_thresholds
    box_t, _ = get_part_thresholds(part)
    return box_t


def _is_fashionpedia_part(part: Optional[str]) -> bool:
    """Return True if *part* is in Fashionpedia core coverage."""
    if part is None:
        return False
    from fashion_vision.localization.fashionpedia_part_detector import PART_TO_FP_IDS
    return part in PART_TO_FP_IDS


def _load_garment_mask(
    instance: Dict[str, Any],
    prefer_pred_mask: bool = True,
) -> Optional[np.ndarray]:
    mask_path = resolve_instance_mask_path(instance, prefer_pred_mask=prefer_pred_mask)
    if mask_path is None:
        return None
    try:
        return load_binary_mask(mask_path)
    except FileNotFoundError:
        return None


def _instance_bbox(instance: Dict[str, Any]) -> Optional[List[int]]:
    """Extract the garment instance bounding box from the instance record."""
    for key in ("bbox", "det_bbox", "bbox_xyxy", "detection_bbox"):
        val = instance.get(key)
        if val is not None:
            try:
                return [int(round(float(v))) for v in val]
            except (TypeError, ValueError):
                pass
    return None


def _crop_image_and_mask_with_transform(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    inst_bbox: List[int],
    pad_px: int = 8,
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Same as _crop_image_and_mask but returns a transform dict for uniform API."""
    image_crop, mask_crop, (ox, oy) = _crop_image_and_mask(
        image, mask, inst_bbox, pad_px=pad_px,
    )
    crop_h, crop_w = image_crop.shape[:2]
    h_img, w_img = image.shape[:2]
    transform = {
        "offset_x": ox,
        "offset_y": oy,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "crop_box": [0, 0, crop_w, crop_h],
        "garment_bbox": inst_bbox,
        "zoom_applied": False,
        "zoom_factor": 1.0,
        "part": "__garment_crop__",
    }
    return image_crop, mask_crop, transform


def _crop_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    inst_bbox: List[int],
    pad_px: int = 8,
) -> tuple[np.ndarray, Optional[np.ndarray], tuple[int, int]]:
    """
    Crop image AND mask to the instance bbox using the exact same clamped box.

    Both arrays are sliced with identical ``[y1:y2, x1:x2]`` bounds so the
    mask stays pixel-aligned with the image crop.  If the mask spatial
    dimensions differ from the image (e.g. generated at a different scale),
    it is resized with ``INTER_NEAREST`` before cropping to preserve binary values.

    Args:
        image: Full BGR H×W×3 image.
        mask: Binary H×W garment mask, or None.
        inst_bbox: ``[x1, y1, x2, y2]`` detection bbox in full-image pixel coords.
        pad_px: Padding around the bbox (pixels).

    Returns:
        ``(image_crop, mask_crop, (offset_x, offset_y))``
        ``mask_crop`` is None when ``mask`` is None.
        ``offset_*`` must be added to crop-space DINO bbox coordinates.
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

    if mask is None:
        return image_crop, None, (x1, y1)

    # Resize mask to match image dimensions when they differ (e.g. SAM output
    # at a different scale than the detector output).
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_crop = mask[y1:y2, x1:x2]
    return image_crop, mask_crop, (x1, y1)


def _build_debug(
    prompts: List[str],
    box_threshold: float,
    text_threshold: float,
    n_before_nms: int,
    n_after_nms: int,
    pre_shape: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    mask_gated: bool,
    dilation_px: int,
    zoom_applied: bool = False,
    zoom_factor: float = 1.0,
    backend: str = "open_vocab_grounding_dino",
) -> Dict[str, Any]:
    """Assemble the debug metadata dict for an open-vocab locate_region() result."""
    return {
        "backend": backend,
        "prompts_used": prompts,
        "thresholds_used": {
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            # HuggingFace GDINO post_process uses a single threshold; text_threshold
            # is stored in config for future backends but not forwarded currently.
            "text_threshold_backend_note": "unused_hf_gdino_single_threshold_api",
        },
        "candidate_count_before_nms": n_before_nms,
        "candidate_count_after_nms": n_after_nms,
        "candidate_count_before_shape_filter": len(pre_shape),
        "candidate_count_after_shape_filter": len(pre_shape) - len(rejected),
        "mask_gated": mask_gated,
        "mask_dilation_px": dilation_px,
        "zoom_applied": zoom_applied,
        "zoom_factor": zoom_factor,
        # Saved for visualize_localization_debug.py
        "all_candidates_pre_shape": pre_shape,
        "rejected_candidates": rejected,
    }


def _resolve_prompts(
    intent: Any,
    query: str,
    fallback_threshold: float = 0.3,
) -> tuple[list[str], str, float, float]:
    """
    Choose DINO prompts, backend label, and per-part thresholds.

    Priority:
        1. part_detection_config (Phase 2 — per-part optimised prompts + thresholds).
        2. Registered multi-prompt from open_vocab_prompt_map (legacy fallback).
        3. Single canonical English phrase from intent.grounding_text.
        4. Zero-shot noun extraction from the raw query.

    Returns:
        (prompts, backend_label, box_threshold, text_threshold)
    """
    from fashion_vision.localization.part_detection_config import (
        DEFAULT_TEXT_THRESHOLD,
        PART_DETECTION_CONFIG,
        get_part_prompts,
        get_part_thresholds,
    )

    part = intent.part or ""

    if part:
        # Phase 2: per-part config takes priority over legacy prompt map.
        if part in PART_DETECTION_CONFIG:
            box_t, text_t = get_part_thresholds(part)
            return get_part_prompts(part), "open_vocab_grounding_dino", box_t, text_t

        from fashion_vision.localization.open_vocab_prompt_map import (
            get_prompts_for_region,
            is_open_vocab_region,
        )
        if is_open_vocab_region(part):
            return (
                get_prompts_for_region(part),
                "open_vocab_grounding_dino",
                fallback_threshold,
                DEFAULT_TEXT_THRESHOLD,
            )

        if intent.grounding_text:
            return (
                [intent.grounding_text],
                "open_vocab_grounding_dino",
                fallback_threshold,
                DEFAULT_TEXT_THRESHOLD,
            )

    # Zero-shot: extract noun phrase from raw query.
    noun = _zero_shot_noun_phrase(query)
    logger.info(
        "locate_region: zero-shot fallback for query=%r → noun_phrase=%r",
        query,
        noun,
    )
    return [noun], "zero_shot_grounding_dino", fallback_threshold, DEFAULT_TEXT_THRESHOLD


def _refine_mask(
    image: np.ndarray,
    bbox_xyxy: List[int],
    garment_mask: Optional[np.ndarray],
    sam_wrapper: Optional["SamHqWrapper"],
) -> Dict[str, Any]:
    """Run BboxMaskRefiner if sam_wrapper available, else bbox-fill fallback."""
    if sam_wrapper is None:
        from fashion_vision.localization.bbox_mask_refiner import BboxMaskRefiner
        return BboxMaskRefiner._bbox_fill(image.shape, bbox_xyxy)

    from fashion_vision.localization.bbox_mask_refiner import BboxMaskRefiner
    refiner = BboxMaskRefiner(sam_wrapper)
    return refiner.refine(image, bbox_xyxy, garment_mask)


def _flag_garment_ref_mismatch(garment_ref: str, instance: Dict[str, Any]) -> None:
    """
    Set ``instance["_garment_ref_mismatch"] = True`` when the instance's fine
    class is not one of the classes implied by ``garment_ref``.

    Modifies the instance dict in-place so that both fast-path and open-vocab
    result builders (setdefault and _build_result_base) see the flag.

    Args:
        garment_ref: Parsed garment reference (e.g. "outerwear", "pants").
        instance: Garment instance dict.  Modified in-place when a mismatch is
            detected.  Left unmodified when garment_ref is unrecognised or when
            the instance has no class information.
    """
    from fashion_vision.localization.garment_ref_filter import (
        GARMENT_REF_TO_FINE_CLASSES,
        _class_name,
    )
    # "inner" has no class-based signal — skip.
    target_classes = GARMENT_REF_TO_FINE_CLASSES.get(garment_ref)
    if target_classes is None:
        return
    cls = _class_name(instance)
    if not cls:
        return  # cannot determine class — skip rather than flag as mismatch
    if cls not in frozenset(c.lower() for c in target_classes):
        instance["_garment_ref_mismatch"] = True
        logger.warning(
            "locate_region: garment_ref=%r but instance class=%r — "
            "garment_ref_matched=False in result",
            garment_ref,
            cls,
        )


def _build_result_base(
    intent: Any,
    instance: Dict[str, Any],
    backend_label: str,
    prompts: list[str],
) -> Dict[str, Any]:
    """Build the common fields shared by success and failure results."""
    return {
        "query": intent.raw_query,
        "part": intent.part,
        "garment_ref": intent.garment_ref,
        "direction": intent.direction,
        "spatial_anchor": intent.spatial_anchor,
        "is_zero_shot": intent.is_zero_shot,
        "backend": backend_label,
        "instance_id": instance.get("instance_id"),
        "fine_class": (
            instance.get("fine_class_name")
            or instance.get("class_name")
            or instance.get("category")
        ),
        "garment_ref_matched": not instance.get("_garment_ref_mismatch", False),
        "prompts_used": prompts,
        "bbox_format": "xyxy",
    }
