"""
P1.4b — Region Localization Backend Adapters.

Serving-safe wrappers around real 3.1.2 region localization models.
Each backend returns normalized dicts compatible with ``LocalizedRegion``.
No raw masks, crops, temp paths, or image bytes leak through this layer.

Backends:
    - ``DisabledRegionLocalizationBackend`` — always returns empty (default).
    - ``FashionpediaRegionBackend`` — wraps Fashionpedia YOLOv8s part detector.
    - ``FullRegionLocalizationBackend`` — wraps ``locate_region()`` (skeleton,
      requires garment instances + SAM + DINO; not wired by default).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_sys_path() -> None:
    """Ensure project root and src/ are on sys.path for fashion_vision imports."""
    import sys
    for p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

# ── Part label normalisation ──────────────────────────────────────────────────

# Map raw backend labels → standardised part_type.
# Fashionpedia YOLO class names → canonical part_type.
_FP_LABEL_TO_PART_TYPE: Dict[str, str] = {
    "hood": "hood",
    "collar": "collar",
    "lapel": "lapel",
    "epaulette": "epaulette",
    "sleeve": "sleeve",
    "pocket": "pocket",
    "neckline": "neckline",
    "buckle": "buckle",
    "zipper": "zipper",
    "bow": "bow",
    "fringe": "fringe",
    "ruffle": "ruffle",
    "sequin": "sequin",
    "applique": "applique",
    "bead": "bead",
    "flower": "flower",
    "ribbon": "ribbon",
    "rivet": "rivet",
    "tassel": "tassel",
    # Fast-path labels (from locate_region / region_locator).
    "hem": "hem",
    "waist": "waist",
    "shoulder": "shoulder",
    "leg_opening": "hem",
    # Cuff → sleeve area alias.
    "cuff": "cuff",
    # Generic / unknown.
    "decoration": "decoration",
    "pattern": "pattern",
}

# Map part_type → part_group.
_PART_TYPE_TO_GROUP: Dict[str, str] = {
    "neckline": "collar_area", "collar": "collar_area", "lapel": "collar_area",
    "hood": "collar_area",
    "sleeve": "sleeve_area", "cuff": "sleeve_area",
    "pocket": "pocket_area",
    "zipper": "closure", "button": "closure", "buckle": "closure",
    "hem": "hem_area", "leg_opening": "hem_area",
    "shoulder": "shoulder_area", "epaulette": "shoulder_area", "strap": "shoulder_area",
    "waist": "waist_area",
    "sequin": "decoration", "bead": "decoration", "rivet": "decoration",
    "bow": "decoration", "ribbon": "decoration", "ruffle": "decoration",
    "fringe": "decoration", "tassel": "decoration", "applique": "decoration",
    "flower": "decoration", "decoration": "decoration",
    "pattern": "pattern_area",
    "shoes": "accessory", "bag": "accessory",
}


def _normalize_part_type(raw_label: str) -> str:
    """Map a raw backend label to a canonical ``part_type``."""
    if not raw_label:
        return "unknown"
    label = raw_label.strip().lower()
    return _FP_LABEL_TO_PART_TYPE.get(label, "unknown")


def _part_group_for(part_type: str) -> str:
    """Return the semantic group for a canonical ``part_type``."""
    return _PART_TYPE_TO_GROUP.get(part_type, "unknown")


def _valid_bbox(bbox: Any) -> bool:
    """Return True if *bbox* is [x1,y1,x2,y2] with x2>x1 and y2>y1 and all finite."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    import math
    return (
        math.isfinite(x1) and math.isfinite(y1)
        and math.isfinite(x2) and math.isfinite(y2)
        and x2 > x1 and y2 > y1
    )


# ── Normalisation ────────────────────────────────────────────────────────────


def normalize_region_predictions(
    raw_predictions: List[Dict[str, Any]],
    *,
    source: str = "fashion_vision_3_1_2",
    backend: str = "unknown",
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    """Convert raw 3.1.2 predictions into ``LocalizedRegion``-compatible dicts.

    Rules:
        - Invalid bboxes are silently dropped.
        - Unknown labels → ``"unknown"`` part_type.
        - Confidence preserved if present, else ``None``.
        - ``mask_present`` always ``False`` (no mask through API).
        - ``mask_ref`` always ``None``.
        - No raw image, crop, tensor, path, or bitmap in output.

    Args:
        raw_predictions: List of raw detection dicts.
            Each dict should have at least ``bbox_xyxy`` or ``bbox``,
            and optionally ``label``, ``score``, ``class_id``.
        source: Value for the ``source`` field.
        backend: Value for the ``backend`` field.
        start_index: First ``region_id`` index (for merging multiple backend calls).

    Returns:
        List of dicts compatible with ``LocalizedRegion`` model.
    """
    normalized: List[Dict[str, Any]] = []
    idx = start_index

    for raw in raw_predictions:
        if not isinstance(raw, dict):
            continue

        bbox = raw.get("bbox_xyxy") or raw.get("bbox")
        if not _valid_bbox(bbox):
            continue

        raw_label = raw.get("label", raw.get("part_type", ""))
        part_type = _normalize_part_type(str(raw_label))
        part_group = _part_group_for(part_type)

        confidence = raw.get("score", raw.get("confidence"))
        if not isinstance(confidence, (int, float)):
            confidence = None
        elif isinstance(confidence, (int, float)):
            confidence = float(confidence)

        inst_id = raw.get("_instance_id")  # P1.4e: from bridge
        normalized.append({
            "region_id": f"region_{idx}",
            "part_type": part_type,
            "part_group": part_group,
            "bbox": [float(v) for v in bbox],
            "confidence": confidence,
            "source": source,
            "backend": backend,
            "mask_present": False,
            "mask_ref": None,
            "instance_id": inst_id,
        })
        idx += 1

    return normalized


# ── Backend interface ────────────────────────────────────────────────────────


class RegionLocalizationBackend(ABC):
    """Interface for 3.1.2 region localization backends.

    Implementations accept a decoded image (numpy BGR array) and optional
    query/part parameters, and return a list of ``LocalizedRegion``-compatible
    dicts.
    """

    @abstractmethod
    def locate_regions(
        self,
        image: Any,
        query: Optional[str] = None,
        requested_part: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return normalized localized region dicts for *image*.

        Args:
            image: numpy BGR uint8 H×W×3 array.
            query: Original user query (for intent/disambiguation).
            requested_part: Canonical part_type to localize, or None for all.

        Returns:
            List of dicts compatible with ``LocalizedRegion``.
            Never raises — errors are logged and an empty list is returned.
        """
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier for meta/sources."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """True when this backend is ready to serve requests."""
        ...


# ── Disabled backend ─────────────────────────────────────────────────────────


class DisabledRegionLocalizationBackend(RegionLocalizationBackend):
    """Always returns empty — safe default when 3.1.2 is not enabled."""

    @property
    def backend_name(self) -> str:
        return "disabled"

    @property
    def enabled(self) -> bool:
        return False

    def locate_regions(
        self,
        image: Any = None,
        query: Optional[str] = None,
        requested_part: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return []


# ── Fashionpedia YOLO backend ────────────────────────────────────────────────


class FashionpediaRegionBackend(RegionLocalizationBackend):
    """Wraps ``FashionpediaPartDetector`` for serving-safe inference.

    Runs the 19-class Fashionpedia YOLOv8s model on the full image for each
    requested part.  Does NOT require garment instances, SAM, or DINO.

    Model is loaded lazily on first ``locate_regions()`` call — constructing
    the backend never imports ultralytics or touches CUDA.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> None:
        self._model_path = model_path or str(
            _PROJECT_ROOT / "models" / "detectors"
            / "fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
        )
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._detector: Any = None
        self._load_error: Optional[str] = None

    @property
    def backend_name(self) -> str:
        return "fashionpedia_yolo"

    @property
    def enabled(self) -> bool:
        # ponytail: check model file existence eagerly (lazy load only confirms).
        if self._load_error is not None:
            return False
        if not Path(self._model_path).exists():
            return False
        return True

    # -- lazy load ------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load the YOLO model.  Returns True on success."""
        if self._detector is not None:
            return True
        if self._load_error is not None:
            return False

        # Check model file exists.
        model_path = Path(self._model_path)
        if not model_path.exists():
            self._load_error = f"Model not found: {self._model_path}"
            logger.warning("FashionpediaRegionBackend: %s", self._load_error)
            return False

        # Check optional deps.
        try:
            from importlib import util as importlib_util
            for mod in ("torch", "cv2", "ultralytics"):
                if importlib_util.find_spec(mod) is None:
                    self._load_error = f"Missing dependency: {mod}"
                    logger.warning("FashionpediaRegionBackend: %s", self._load_error)
                    return False
        except Exception:
            pass

        try:
            _ensure_sys_path()
            from fashion_vision.localization.fashionpedia_part_detector import (
                FashionpediaPartDetector,
            )
            self._detector = FashionpediaPartDetector(
                str(model_path), device=self._device,
            )
            logger.info(
                "FashionpediaRegionBackend loaded: model=%s device=%s",
                self._model_path, self._device,
            )
            return True
        except Exception as exc:
            self._load_error = f"Failed to load model: {exc}"
            logger.exception("FashionpediaRegionBackend: load failed")
            return False

    # -- detect single part ---------------------------------------------------

    def _detect_part(
        self, image: Any, target_part: str
    ) -> List[Dict[str, Any]]:
        """Run YOLO for *target_part* and return raw detection dicts."""
        if not self._ensure_loaded():
            return []
        try:
            return self._detector.detect(
                image, target_part, garment_mask=None, conf=self._confidence_threshold,
            )
        except Exception:
            logger.exception(
                "FashionpediaRegionBackend: detect failed for part=%r", target_part,
            )
            return []

    # -- public API -----------------------------------------------------------

    def locate_regions(
        self,
        image: Any,
        query: Optional[str] = None,
        requested_part: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run Fashionpedia YOLO and return normalized regions.

        If *requested_part* is given, only that part is queried.
        Currently queries only the Fashionpedia-supported parts (not landmark-only
        parts like hem/waist/shoulder — those require the full ``locate_region()``
        pipeline).
        """
        if image is None:
            return []

        # ponytail: map requested_part through Fashionpedia coverage.
        # Parts NOT in Fashionpedia (hem, waist, shoulder, leg_opening)
        # require the full locate_region() pipeline (landmark + geometry).
        _ensure_sys_path()
        from fashion_vision.localization.fashionpedia_part_detector import (
            PART_TO_FP_IDS,
        )
        fp_parts = list(PART_TO_FP_IDS.keys())

        if requested_part:
            if requested_part in fp_parts:
                parts_to_query = [requested_part]
            else:
                # Not in Fashionpedia coverage — return empty.
                logger.debug(
                    "FashionpediaRegionBackend: part=%r not in FP coverage, "
                    "returning empty", requested_part,
                )
                return []
        else:
            # No specific part — query all Fashionpedia parts.
            # ponytail: this is expensive (N model calls); detail query uses
            # only a representative subset for now.
            parts_to_query = [
                "neckline", "collar", "lapel", "pocket", "zipper",
                "button", "buckle", "bow", "sequin", "sleeve",
            ]

        all_raw: List[Dict[str, Any]] = []
        for part in parts_to_query:
            detections = self._detect_part(image, part)
            all_raw.extend(detections)

        return normalize_region_predictions(
            all_raw,
            source="fashion_vision_3_1_2",
            backend=self.backend_name,
        )


# ── Full 3.1.2 pipeline backend ─────────────────────────────────────────────


class Full312RegionBackend(RegionLocalizationBackend):
    """Wraps the full ``locate_region()`` pipeline (Fashionpedia YOLO + DINO).

    Uses synthetic full-image instances so 3.1.1 is NOT required.
    Calls the existing ``locate_region()`` function per part — no algorithm
    duplication.  Fashionpedia routing, DINO fallback, shape priors, and
    per-part thresholds are all handled by the existing pipeline.

    Fast-path parts (hem, waist, shoulder, leg_opening) require real garment
    instances with landmark data — they return empty from this backend.

    Models loaded lazily on first ``locate_regions()`` call.
    """

    # Parts that need real garment instances (skipped when using synthetic fallback).
    _FAST_PATH_PARTS = frozenset({"hem", "waist", "shoulder", "leg_opening"})

    # Parts NOT in PART_VOCAB but in Fashionpedia — use grounding text as query.
    _FP_ONLY_PARTS: Dict[str, str] = {
        "hood": "hood on garment",
        "sequin": "sequin decoration on clothing",
        "applique": "applique decoration on clothing",
        "bead": "bead decoration on clothing",
        "flower": "flower decoration on clothing",
        "ribbon": "ribbon on clothing",
        "rivet": "rivet on clothing",
        "tassel": "tassel on clothing",
    }

    def __init__(
        self,
        fp_model_path: Optional[str] = None,
        dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cpu",
        box_threshold: float = 0.3,
        min_confidence: float = 0.5,
    ) -> None:
        self._fp_model_path = fp_model_path or str(
            _PROJECT_ROOT / "models" / "detectors"
            / "fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
        )
        self._dino_model_id = dino_model_id
        self._device = device
        self._box_threshold = box_threshold
        self._min_confidence = min_confidence
        self._fp_detector: Any = None
        self._dino: Any = None
        self._load_error: Optional[str] = None

    @property
    def backend_name(self) -> str:
        return "full312"

    @property
    def enabled(self) -> bool:
        if self._load_error is not None:
            return False
        if not Path(self._fp_model_path).exists():
            return False
        return True

    # -- lazy load -----------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load Fashionpedia YOLO + Grounding DINO. Returns True on success."""
        if self._fp_detector is not None and self._dino is not None:
            return True
        if self._load_error is not None:
            return False

        _ensure_sys_path()

        # Load Fashionpedia.
        fp_path = Path(self._fp_model_path)
        if not fp_path.exists():
            self._load_error = f"Fashionpedia model not found: {self._fp_model_path}"
            logger.warning("Full312RegionBackend: %s", self._load_error)
            return False

        # Check heavy deps.
        try:
            from importlib import util as importlib_util
            for mod in ("torch", "cv2", "ultralytics", "transformers"):
                if importlib_util.find_spec(mod) is None:
                    self._load_error = f"Missing dependency: {mod}"
                    logger.warning("Full312RegionBackend: %s", self._load_error)
                    return False
        except Exception:
            pass

        try:
            from fashion_vision.localization.fashionpedia_part_detector import (
                FashionpediaPartDetector,
            )
            from fashion_vision.localization.grounding_dino_locator import (
                GroundingDINOLocator,
            )
            self._fp_detector = FashionpediaPartDetector(
                str(fp_path), device=self._device,
            )
            # Fix SSL_CERT_FILE if it points to a non-existent path (common on
            # Windows conda envs).  The model is usually cached locally.
            import os as _os
            _cert = _os.environ.get("SSL_CERT_FILE", "")
            if _cert and not _os.path.exists(_cert):
                _os.environ.pop("SSL_CERT_FILE", None)
            try:
                self._dino = GroundingDINOLocator(
                    model_id=self._dino_model_id, device=self._device,
                )
            except Exception:
                # Retry with local_files_only if download fails.
                _os.environ.pop("SSL_CERT_FILE", None)
                self._dino = GroundingDINOLocator(
                    model_id=self._dino_model_id, device=self._device,
                )
            logger.info(
                "Full312RegionBackend loaded: fp=%s dino=%s device=%s",
                self._fp_model_path, self._dino_model_id, self._device,
            )
            return True
        except Exception as exc:
            self._load_error = f"Failed to load models: {exc}"
            logger.exception("Full312RegionBackend: load failed")
            return False

    # -- query builder -------------------------------------------------------

    @staticmethod
    def _query_for_part(part_type: str) -> str:
        """Build a query string that ``parse_intent()`` can resolve to *part_type*.

        For parts in PART_VOCAB, the canonical name works directly.
        For FP-only parts (sequin, applique, etc.), use a custom grounding text.
        For fast-path parts, return the part name (will fail gracefully).
        For unknown parts, return the part name as-is (DINO zero-shot).
        """
        # FP-only parts need custom prompts.
        fp_prompt = Full312RegionBackend._FP_ONLY_PARTS.get(part_type)
        if fp_prompt:
            return fp_prompt
        # Everything else: part name works for PART_VOCAB, zero-shot for others.
        return part_type

    # -- public API ----------------------------------------------------------

    def locate_regions(
        self,
        image: Any,
        query: Optional[str] = None,
        requested_part: Optional[str] = None,
        query_all_parts: bool = False,
        garment_instances: Optional[List[Dict[str, Any]]] = None,
        target_instance_id: Optional[str] = None,
        allow_full_image_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run full 3.1.2 pipeline (Fashionpedia + DINO) and return normalized regions.

        P1.4e: when *garment_instances* from 3.1.1 are available, they are used
        as the ``instance`` for ``locate_region()``.  Synthetic full-image instance
        is only used as fallback.

        Args:
            image: BGR numpy array.
            query: Original user query (unused — routing is by *requested_part*).
            requested_part: Canonical part_type to localize, or None for all.
            query_all_parts: When True, queries representative parts across
                categories (ignoring *requested_part*).
            garment_instances: 3.1.1 garment instance dicts (from VisionContext).
            target_instance_id: If set, only use this instance.
            allow_full_image_fallback: Use synthetic instance when no real
                instances exist (default: True).

        Returns:
            List of ``LocalizedRegion``-compatible dicts with ``instance_id``.
        """
        if image is None or not self._ensure_loaded():
            return []

        h, w = image.shape[:2]

        # Resolve instances.
        instances: List[Dict[str, Any]] = []
        used_synthetic = False

        if garment_instances:
            for gi in garment_instances:
                if not isinstance(gi, dict):
                    continue
                iid = gi.get("instance_id", "")
                if target_instance_id and iid != target_instance_id:
                    continue
                bridge_inst = build_locate_region_instance(gi, w, h)
                instances.append(bridge_inst)
            if not instances and allow_full_image_fallback:
                instances.append(make_synthetic_instance(w, h))
                used_synthetic = True
        elif allow_full_image_fallback:
            instances.append(make_synthetic_instance(w, h))
            used_synthetic = True

        # Resolve parts to query.
        if query_all_parts:
            parts_to_query = [
                "neckline", "collar", "lapel", "pocket", "zipper",
                "sleeve", "cuff", "button", "buckle", "bow",
                "sequin", "hood", "epaulette", "fringe", "ruffle",
            ]
        elif requested_part:
            parts_to_query = [requested_part]
        else:
            parts_to_query = ["neckline"]

        all_raw: List[Dict[str, Any]] = []
        dino_used = False
        fp_used = False

        # Lazy import once.
        from fashion_vision.localization.region_localization_router import (
            locate_region as _locate_region,
        )

        for inst in instances:
            inst_id = inst.get("instance_id", "unknown")
            inst_is_synthetic = (inst_id == "synthetic_full_image")

            for part in parts_to_query:
                # P1.4e: only skip fast-path parts when using synthetic instance.
                if part in self._FAST_PATH_PARTS and inst_is_synthetic:
                    continue

                query_text = self._query_for_part(part)
                try:
                    raw = _locate_region(
                        query=query_text,
                        instance=inst,
                        image=image,
                        image_width=w, image_height=h,
                        locator=self._dino,
                        dino_threshold=self._box_threshold,
                        prefer_pred_mask=False,
                        sam_wrapper=None,
                        fashionpedia_detector=self._fp_detector,
                    )
                except Exception:
                    logger.exception(
                        "Full312RegionBackend: locate_region failed "
                        "part=%r inst=%r", part, inst_id,
                    )
                    continue

                backend_label = raw.get("backend", "unknown")
                if "dino" in backend_label:
                    dino_used = True
                if "fashionpedia" in backend_label or backend_label == "fashionpedia_yolo":
                    fp_used = True

                if raw.get("status") != "success":
                    continue

                bbox = raw.get("bbox")
                score = raw.get("score")
                if bbox is None or score is None:
                    continue
                if isinstance(score, (int, float)) and score < self._min_confidence:
                    continue

                raw_entry = {
                    "bbox_xyxy": bbox,
                    "score": score,
                    "label": part,
                    "backend": backend_label,
                }
                # P1.4e: carry instance_id through to normalized output.
                if not inst_is_synthetic and inst_id != "unknown":
                    raw_entry["_instance_id"] = inst_id
                all_raw.append(raw_entry)

        # Resolve overall backend label.
        if fp_used and dino_used:
            overall_backend = "full312"
        elif fp_used:
            overall_backend = "fashionpedia_yolo"
        elif dino_used:
            overall_backend = "dino"
        else:
            overall_backend = "full312"

        return normalize_region_predictions(
            all_raw,
            source="fashion_vision_3_1_2",
            backend=overall_backend,
        )


# locate_region is imported lazily inside Full312RegionBackend._ensure_loaded()
# to avoid importing torch/transformers/ultralytics at module level.


# ── P1.4e: Instance bridge ──────────────────────────────────────────────────


def build_locate_region_instance(
    garment_instance: Dict[str, Any],
    image_width: int,
    image_height: int,
    mask_resolver: Any = None,
) -> Dict[str, Any]:
    """Convert a 3.1.1 garment_instance into a ``locate_region()`` instance dict.

    Preserves ``instance_id``, category, fine_class_name, and bbox.
    Does NOT resolve mask paths (``mask_ref`` is a placeholder).
    Bboxes are validated and clipped to image bounds.

    Args:
        garment_instance: Dict from ``VisionContext.garment_instances``.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        mask_resolver: Reserved for future mask path resolution.

    Returns:
        Dict compatible with ``locate_region()`` instance parameter.
    """
    inst: Dict[str, Any] = {
        "instance_id": garment_instance.get("instance_id", "unknown"),
    }

    # Category / class — prefer fine-grained names over coarse.
    # Order: fine_class_name first, then class_name, then coarse/category.
    for key in ("fine_class_name", "class_name", "coarse_class_name", "category"):
        val = garment_instance.get(key)
        if val:
            inst.setdefault("fine_class_name", val)
    for key in ("category", "coarse_class_name"):
        val = garment_instance.get(key)
        if val:
            inst.setdefault("category", val)
            inst.setdefault("coarse_class_name", val)

    # Bbox — validate and clip.
    bbox = garment_instance.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        # Clip to image bounds.
        x1 = max(0.0, min(x1, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        x2 = max(0.0, min(x2, float(image_width)))
        y2 = max(0.0, min(y2, float(image_height)))
        if x2 > x1 and y2 > y1:
            inst["bbox"] = [x1, y1, x2, y2]
            inst["bbox_format"] = garment_instance.get("bbox_format", "xyxy_abs_pixels")

    # Mask paths — mask_ref is a placeholder, not a real path.
    # If mask_resolver is provided in future, it would map mask_ref → real path here.
    mask_ref = garment_instance.get("mask_ref")
    mask_present = garment_instance.get("mask_present", False)
    if mask_present and mask_resolver is not None:
        resolved = mask_resolver(mask_ref)
        if resolved:
            inst["pred_mask_path"] = resolved
    # Without resolver: no mask path → locate_region runs without mask gating.

    return inst


def make_synthetic_instance(
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """Create a synthetic full-image instance (fallback when no 3.1.1 results)."""
    return {
        "instance_id": "synthetic_full_image",
        "bbox": [0, 0, image_width, image_height],
        "category": "unknown",
        "fine_class_name": "unknown",
        "coarse_class_name": "unknown",
    }


# ── Factory ──────────────────────────────────────────────────────────────────


def build_region_backend(
    name: str = "disabled",
    *,
    model_path: Optional[str] = None,
    device: str = "cpu",
    confidence_threshold: float = 0.5,
) -> RegionLocalizationBackend:
    """Construct a region localization backend by name.

    Args:
        name: ``"disabled"`` | ``"fashionpedia"`` | ``"full"``.
        model_path: Path to Fashionpedia .pt file (fashionpedia backend only).
        device: ``"cpu"`` | ``"cuda"``.
        confidence_threshold: Minimum score for detections.

    Returns:
        A ``RegionLocalizationBackend`` instance.
    """
    name = name.strip().lower()
    if name == "fashionpedia":
        return FashionpediaRegionBackend(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )
    if name in ("full", "full312"):
        return Full312RegionBackend(
            fp_model_path=model_path,
            device=device,
            box_threshold=0.3,
            min_confidence=confidence_threshold,
        )
    return DisabledRegionLocalizationBackend()


# ── Image decoding (shared utility) ──────────────────────────────────────────


def decode_image_bytes(image_bytes: Any) -> Any:
    """Decode *image_bytes* (bytes or base64 str) to a BGR numpy array.

    Returns None on any failure (never raises).
    """
    import numpy as np

    raw: bytes
    if isinstance(image_bytes, (bytes, bytearray)):
        raw = bytes(image_bytes)
    elif isinstance(image_bytes, str):
        import base64
        try:
            raw = base64.b64decode(image_bytes, validate=True)
        except Exception:
            logger.warning("region_backend: image_bytes is not valid base64")
            return None
    else:
        return None

    try:
        import cv2
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except Exception:
        logger.exception("region_backend: image decode failed")
        return None


# ── Singleton ────────────────────────────────────────────────────────────────

_region_backend: Optional[RegionLocalizationBackend] = None


def _resolve_region_backend_settings() -> Dict[str, Any]:
    """Resolve region backend settings from serving_config.yaml + env overrides.

    Env vars: VISION_REGION_BACKEND, VISION_REGION_ENABLE_REAL,
    VISION_REGION_DEVICE, VISION_REGION_CONFIDENCE_THRESHOLD.
    """
    import os

    defaults: Dict[str, Any] = {
        "backend": "disabled",
        "enable_real": False,
        "model_path": "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
        "device": "cpu",
        "confidence_threshold": 0.5,
        "timeout_ms": 5000,
    }
    try:
        from inference.serving.deps import get_config
        cfg = (get_config().get("region_backend") or {})
        settings = dict(defaults)
        settings["backend"] = str(cfg.get("backend", "disabled"))
        settings["enable_real"] = bool(cfg.get("enable_real", False))
        settings["model_path"] = str(cfg.get("model_path", defaults["model_path"]))
        settings["device"] = str(cfg.get("device", "cpu"))
        settings["confidence_threshold"] = float(cfg.get("confidence_threshold", 0.5))
        settings["timeout_ms"] = int(cfg.get("timeout_ms", 5000))

        env_backend = os.getenv("VISION_REGION_BACKEND")
        if env_backend:
            settings["backend"] = env_backend.strip().lower()
        env_real = os.getenv("VISION_REGION_ENABLE_REAL")
        if env_real is not None:
            settings["enable_real"] = env_real.strip().lower() in ("1", "true", "yes")
        env_device = os.getenv("VISION_REGION_DEVICE")
        if env_device:
            settings["device"] = env_device.strip().lower()
        env_conf = os.getenv("VISION_REGION_CONFIDENCE_THRESHOLD")
        if env_conf:
            settings["confidence_threshold"] = float(env_conf)
        return settings
    except Exception:
        logger.warning("Region backend settings resolution failed — using disabled.")
        return defaults


def get_region_backend() -> RegionLocalizationBackend:
    """Return the process-wide region localization backend singleton.

    Default is ``DisabledRegionLocalizationBackend``.  Real backends are
    selected only when config/env explicitly enables them.
    """
    global _region_backend
    if _region_backend is None:
        settings = _resolve_region_backend_settings()
        if not settings["enable_real"]:
            _region_backend = DisabledRegionLocalizationBackend()
            logger.info("Region backend: disabled (enable_real=false)")
        else:
            _region_backend = build_region_backend(
                name=settings["backend"],
                model_path=settings["model_path"],
                device=settings["device"],
                confidence_threshold=settings["confidence_threshold"],
            )
            logger.info(
                "Region backend: %s (enabled=%s device=%s)",
                _region_backend.backend_name,
                _region_backend.enabled,
                settings["device"],
            )
    return _region_backend


def reset_region_backend() -> None:
    """Reset the singleton (test seam)."""
    global _region_backend
    _region_backend = None
