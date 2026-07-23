"""
P1.1 — RealVisionAttributeProvider: experimental adapter shell for the PRD 3.1
vision pipeline.

STATUS: experimental adapter SHELL.  The serving-side contract (input
validation, timeout, error mapping, output normalization, attribute mapping,
no-leak guarantees) is fully implemented and tested against injectable fake
backends.  The actual 3.1 pipeline invocation is NOT wired in P1.1:
``FashionVision31Backend`` performs dependency/checkpoint probing only and
raises :class:`VisionProviderUnavailable` with a structured reason.  See
``docs/P1_real_vision_provider_adapter.md``.

Hard rules (P1.1):
    - NEVER downloads ``image_url`` (no network access of any kind).
    - NEVER loads model weights or heavy libraries at import time.
    - NEVER returns or logs raw ``image_bytes``.
    - NEVER fabricates attributes / bboxes / masks.
    - NEVER raises uncaught exceptions from ``extract()`` — every backend
      failure maps to a structured warning + empty attributes.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inference.serving.schemas import WarningItem, WarningSeverity
from inference.serving.vision_provider import (
    VisionAttributeProvider,
    VisionAttributeResult,
)

logger = logging.getLogger(__name__)

REAL_PROVIDER_VERSION = "0.1.0"
DEFAULT_TIMEOUT_MS = 1500
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ATTRIBUTE_INFERENCE_YAML = _PROJECT_ROOT / "configs" / "attribute_inference.yaml"

# 3.1 stage-1/2/3 weights (existence checked by probing — never loaded here).
_STAGE_CHECKPOINTS = [
    "models/detectors/yolov8n_deepfashion2_13cls_best.pt",
    "checkpoints/sam_hq/sam_hq_vit_b.pth",
    "outputs/landmark_predictor_resnet18/best.pt",
]

# Heavy modules the 3.1 pipeline needs.  Probed via find_spec — NOT imported.
_REQUIRED_MODULES = ("torch", "torchvision", "cv2", "ultralytics")

# ── Attribute mapping: 3.1 task/field names → serving attribute keys ──────────

ATTRIBUTE_KEY_MAP: Dict[str, str] = {
    # serving-native keys pass through
    "color": "color",
    "fabric": "fabric",
    "style": "style",
    "fit_or_silhouette": "fit_or_silhouette",
    "length": "length",
    "pattern": "pattern",
    "garment_category": "garment_category",
    "sleeve_length": "sleeve_length",
    "collar_design": "collar_design",
    "coat_length": "coat_length",
    "dress_length": "dress_length",
    "skirt_length": "skirt_length",
    "pant_length": "pant_length",
    # 3.1 task names
    "neckline_design": "neckline",
    "neck_design": "neckline",
    "lapel_design": "collar_design",
    "class_name": "garment_category",
    "coarse_class_name": "garment_category",
}

VISUAL_ATTRIBUTE_SOURCE = "vision_provider_real"


# ── Exceptions ─────────────────────────────────────────────────────────────────


class VisionProviderUnavailable(RuntimeError):
    """The real vision backend cannot serve requests (deps/checkpoints/wiring).

    Carries a JSON-safe ``details`` dict for structured surfacing in meta.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}


# ── Warning helper ─────────────────────────────────────────────────────────────


def _warning(code: str, message: str,
             severity: WarningSeverity = WarningSeverity.warn) -> WarningItem:
    return WarningItem(code=code, scope="vision", message=message, severity=severity)


# ── Backends ───────────────────────────────────────────────────────────────────


class UnavailableVisionBackend:
    """Placeholder backend — always raises ``VisionProviderUnavailable``."""

    name = "unavailable"

    def predict(self, **kwargs: Any) -> Dict[str, Any]:
        raise VisionProviderUnavailable(
            "No real vision backend is configured.",
            details={"reason": "backend_unavailable"},
        )


class FashionVision31Backend:
    """Probing shell for the PRD 3.1 pipeline (``tools/infer/garment_pipeline.py``).

    P1.1 scope: dependency + checkpoint probing ONLY.  ``predict()`` never
    loads a model — it raises :class:`VisionProviderUnavailable` with a
    structured report so callers surface an honest ``vision_provider_unavailable``
    warning instead of fabricated attributes.

    Real invocation (temp-file ``GarmentPipeline.run_image`` wrapping) is a
    P1.2 work item — blocked on the 8 missing ``outputs/p2_*/best.pt``
    attribute checkpoints and an approved latency budget.
    """

    name = "fashion_vision_3_1"

    def __init__(self, checkpoint_root: Optional[str] = None,
                 project_root: Optional[Path] = None) -> None:
        self._project_root = Path(project_root) if project_root else _PROJECT_ROOT
        self._checkpoint_root = checkpoint_root or "outputs/"

    # -- probing (filename existence + find_spec only; nothing is loaded) ------

    def _attribute_checkpoints(self) -> List[str]:
        """Read stage-6 checkpoint paths from configs/attribute_inference.yaml."""
        try:
            import yaml
            with open(_ATTRIBUTE_INFERENCE_YAML, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:  # noqa: BLE001 — probing must never crash.
            return []
        paths: List[str] = []
        for task_cfg in (data.get("tasks") or {}).values():
            if isinstance(task_cfg, dict) and task_cfg.get("checkpoint"):
                paths.append(str(task_cfg["checkpoint"]))
        return paths

    def probe(self) -> Dict[str, Any]:
        """Return a JSON-safe availability report without importing/loading anything heavy."""
        missing_modules = [m for m in _REQUIRED_MODULES
                           if importlib_util.find_spec(m) is None]
        checkpoints = list(_STAGE_CHECKPOINTS) + self._attribute_checkpoints()
        missing_checkpoints = [p for p in checkpoints
                               if not (self._project_root / p).exists()]
        return {
            "backend": self.name,
            "available": not missing_modules and not missing_checkpoints,
            "missing_modules": missing_modules,
            "missing_checkpoints": missing_checkpoints,
            "wired": False,  # P1.1: invocation intentionally not wired.
        }

    def predict(self, **kwargs: Any) -> Dict[str, Any]:
        report = self.probe()
        if not report["available"]:
            raise VisionProviderUnavailable(
                "fashion_vision_3_1 backend dependencies/checkpoints missing.",
                details=report,
            )
        raise VisionProviderUnavailable(
            "fashion_vision_3_1 invocation is not wired in P1.1 (experimental "
            "adapter shell). See docs/P1_real_vision_provider_adapter.md.",
            details={**report, "reason": "not_wired_p1_1"},
        )


# ── P1.2 — Real 3.1.1 segmentation-only backend ────────────────────────────────

# Checkpoints required for segmentation-only mode (stages 1+2 ONLY —
# landmark and attribute checkpoints are intentionally NOT required).
_SEGMENTATION_CHECKPOINTS = [
    "models/detectors/yolov8n_deepfashion2_13cls_best.pt",
    "checkpoints/sam_hq/sam_hq_vit_b.pth",
]

_VALID_DEVICES = ("cpu", "cuda", "auto")


class FashionVision31SegmentationBackend:
    """P1.2 — REAL 3.1.1 detection + segmentation backend.

    Runs ``tools/infer/garment_pipeline.py::GarmentPipeline.run_image`` with
    ``run_landmark_and_crops=False`` and ``run_attribute_inference=False`` so
    ONLY stage 1 (YOLO detection) and stage 2 (SAM-HQ segmentation) execute.
    3.1.3 attribute classifiers are never touched (their checkpoints are
    missing locally and are not required for this mode).

    Hard rules:
        - image_bytes only; image_url is NEVER downloaded.
        - Everything happens inside a request-scoped temp dir (fixed internal
          file names, no user input in paths), removed in ``finally``.
        - Output never contains mask bitmaps, absolute paths, or image bytes —
          only bbox / confidence / class labels / ``mask_present`` /
          ``mask_ref`` placeholders.
        - Models load lazily inside ``predict()``; importing this module (or
          constructing the backend) never imports torch/ultralytics or checks
          CUDA.  Device strings resolve to a concrete device only in
          ``predict()`` (``auto`` → cuda if available else cpu).

    ``pipeline_runner`` is the test seam: a callable
    ``(image_path: str, output_dir: str) -> dict`` replacing the real
    ``GarmentPipeline.run_image`` (used by default-safe tests; the real path
    is exercised only by RUN_REAL_VISION_TESTS=1 optional tests).
    """

    name = "fashion_vision_3_1_segmentation"
    stage = "segmentation"

    def __init__(self, *, yolo_device: str = "cpu", sam_device: str = "cpu",
                 cleanup_temp_files: bool = True,
                 project_root: Optional[Path] = None,
                 pipeline_runner: Any = None) -> None:
        self._yolo_device = yolo_device if yolo_device in _VALID_DEVICES else "cpu"
        self._sam_device = sam_device if sam_device in _VALID_DEVICES else "cpu"
        self._cleanup_temp_files = bool(cleanup_temp_files)
        self._project_root = Path(project_root) if project_root else _PROJECT_ROOT
        self._pipeline_runner = pipeline_runner

    # -- probing ----------------------------------------------------------------

    def probe(self) -> Dict[str, Any]:
        """JSON-safe availability report — filename/find_spec checks only."""
        missing_modules = [m for m in _REQUIRED_MODULES
                           if importlib_util.find_spec(m) is None]
        missing_checkpoints = [p for p in _SEGMENTATION_CHECKPOINTS
                               if not (self._project_root / p).exists()]
        return {
            "backend": self.name,
            "mode": "segmentation_only",
            "available": not missing_modules and not missing_checkpoints,
            "missing_modules": missing_modules,
            "missing_checkpoints": missing_checkpoints,
            "wired": True,  # P1.2: real invocation IS wired for this mode.
        }

    # -- helpers ----------------------------------------------------------------

    def _resolve_device(self, device: str) -> str:
        """Resolve ``auto`` lazily — torch is only imported here, inside predict()."""
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    @staticmethod
    def _decode_image_bytes(image_bytes: Any) -> bytes:
        """Return raw bytes from a bytes or base64-encoded str payload."""
        if isinstance(image_bytes, (bytes, bytearray)):
            return bytes(image_bytes)
        if isinstance(image_bytes, str):
            import base64
            try:
                return base64.b64decode(image_bytes, validate=True)
            except Exception as exc:
                raise VisionProviderUnavailable(
                    "image_bytes string is not valid base64.",
                    details={"reason": "image_bytes_decode_failed"},
                ) from exc
        raise VisionProviderUnavailable(
            "image_bytes has unsupported type.",
            details={"reason": "image_bytes_unsupported_type",
                     "type": type(image_bytes).__name__},
        )

    def _run_pipeline(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Invoke the real 3.1.1 pipeline (or the injected test runner)."""
        if self._pipeline_runner is not None:
            return self._pipeline_runner(image_path, output_dir)

        # Real path — lazy import; project root must be on sys.path for
        # `tools.*` namespace imports.
        import sys
        root = str(self._project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from tools.infer.garment_pipeline import (  # noqa: PLC0415 — lazy by design
            GarmentPipeline,
            GarmentPipelineConfig,
        )
        config = GarmentPipelineConfig(
            run_landmark_and_crops=False,     # skip stages 3/4/5
            run_attribute_inference=False,    # never run 3.1.3 classifiers
            yolo_device=self._resolve_device(self._yolo_device),
            sam_device=self._resolve_device(self._sam_device),
        )
        return GarmentPipeline(config).run_image(image_path, output_dir)

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        return (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in bbox))

    def _parse_outputs(self, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse stage-1/2 JSONs into leak-free garment_instances + parse meta."""
        import json as _json

        det_path = output_dir / "01_yolo" / "detections.json"
        seg_path = output_dir / "02_samhq" / "segmentation_results.json"

        detections: List[Dict[str, Any]] = []
        if det_path.exists():
            with open(det_path, "r", encoding="utf-8") as fh:
                det_data = _json.load(fh)
            for image_record in det_data.get("images", []):
                detections.extend(image_record.get("detections", []))

        segmented_det_ids: set = set()
        seg_scores: Dict[int, Any] = {}
        if seg_path.exists():
            with open(seg_path, "r", encoding="utf-8") as fh:
                seg_data = _json.load(fh)
            for image_record in seg_data.get("images", []):
                for seg in image_record.get("segments", []):
                    det_id = seg.get("det_id")
                    segmented_det_ids.add(det_id)
                    seg_scores[det_id] = seg.get("sam_score")

        instances: List[Dict[str, Any]] = []
        invalid_bbox = 0
        for i, det in enumerate(detections):
            bbox = det.get("bbox_xyxy")
            if not self._valid_bbox(bbox):
                invalid_bbox += 1
                continue
            det_id = det.get("det_id")
            has_mask = det_id in segmented_det_ids
            inst: Dict[str, Any] = {
                "instance_id": f"inst_{i}",
                # PRD-facing coarse category + internal fine label (dual-label rule).
                "category": det.get("coarse_class_name", det.get("class_name")),
                "fine_class_name": det.get("fine_class_name", det.get("class_name")),
                "bbox": [float(v) for v in bbox],
                "bbox_format": det.get("bbox_format", "xyxy_abs_pixels"),
                "confidence": det.get("confidence"),
                "mask_present": has_mask,
                "source": "fashion_vision_3_1",
                "stage": "segmentation",
            }
            if has_mask:
                # Non-sensitive placeholder — NEVER a filesystem path.
                inst["mask_ref"] = f"mask_inst_{i}"
                if seg_scores.get(det_id) is not None:
                    inst["sam_score"] = seg_scores[det_id]
            instances.append(inst)

        parse_meta = {
            "num_detections": len(detections),
            "num_segments": len(segmented_det_ids),
            "invalid_bbox_count": invalid_bbox,
        }
        return instances, parse_meta

    # -- public API ---------------------------------------------------------------

    def predict(self, *, image_bytes: Any = None, image_url: Optional[str] = None,
                regions: Optional[List[str]] = None,
                garment_category: Optional[str] = None) -> Dict[str, Any]:
        """Run real 3.1.1 detection+segmentation on *image_bytes*.

        Never downloads *image_url* (the provider already blocks url-only
        input; this guard is defense in depth).
        """
        report = self.probe()
        if not report["available"]:
            raise VisionProviderUnavailable(
                "Segmentation backend dependencies/checkpoints missing.",
                details=report,
            )
        if image_bytes is None:
            raise VisionProviderUnavailable(
                "Segmentation backend requires image_bytes (image_url is never downloaded).",
                details={"reason": "image_bytes_required"},
            )

        raw = self._decode_image_bytes(image_bytes)

        import shutil
        import tempfile
        tmp_root = tempfile.mkdtemp(prefix="p12_seg_")
        try:
            tmp = Path(tmp_root)
            image_path = tmp / "input_image.jpg"   # fixed name — never user input
            output_dir = tmp / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(raw)

            self._run_pipeline(str(image_path), str(output_dir))
            instances, parse_meta = self._parse_outputs(output_dir)

            return {
                "attributes": {},               # 3.1.3 intentionally NOT run
                "garment_instances": instances,
                "regions": [],
                "sources": [{
                    "source_type": "vision_model",
                    "provider": "fashion_vision_3_1",
                    "stage": "segmentation",
                    "source_ref": "local_runtime",
                }],
                "meta": {
                    "vision_backend_mode": "segmentation_only",
                    "num_garment_instances": len(instances),
                    "mask_bitmap_returned": False,
                    "yolo_device": self._yolo_device,
                    "sam_device": self._sam_device,
                    **parse_meta,
                },
            }
        finally:
            if self._cleanup_temp_files:
                shutil.rmtree(tmp_root, ignore_errors=True)


def build_backend(name: str, *, mode: Optional[str] = None,
                  checkpoint_root: Optional[str] = None,
                  yolo_device: str = "cpu", sam_device: str = "cpu",
                  cleanup_temp_files: bool = True) -> Any:
    """Construct a backend by name.  Raises ``VisionProviderUnavailable`` on unknown names."""
    if name == "fashion_vision_3_1_segmentation" or (
            name == "fashion_vision_3_1" and mode == "segmentation_only"):
        return FashionVision31SegmentationBackend(
            yolo_device=yolo_device, sam_device=sam_device,
            cleanup_temp_files=cleanup_temp_files,
        )
    if name == "fashion_vision_3_1":
        return FashionVision31Backend(checkpoint_root=checkpoint_root)
    if name == "unavailable":
        return UnavailableVisionBackend()
    raise VisionProviderUnavailable(
        f"Unknown vision backend: {name!r}", details={"backend": name},
    )


# ── Output normalization ───────────────────────────────────────────────────────


def _json_safe(value: Any, *, max_list: int = 64) -> Any:
    """Best-effort JSON-safe conversion; truncates long lists, drops binary blobs."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<binary:{len(value)}bytes>"
    if isinstance(value, (list, tuple)):
        if len(value) > max_list:
            return [_json_safe(v) for v in list(value)[:max_list]] + [f"<truncated:{len(value) - max_list}>"]
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _normalize_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe instance/region record: bbox kept, mask replaced by a flag."""
    out: Dict[str, Any] = {}
    for key, val in inst.items():
        if key in ("mask", "masks", "segmentation", "mask_bitmap"):
            out["mask_present"] = val is not None
            continue
        out[key] = _json_safe(val)
    return out


def normalize_vision_backend_output(
    raw: Any,
    *,
    backend_name: str = "unknown",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]],
           List[WarningItem], Dict[str, Any]]:
    """Normalize a raw backend prediction into VisionAttributeResult parts.

    Returns ``(attributes, garment_instances, regions, warnings, meta_extra)``.

    Rules:
        - Known keys (via ``ATTRIBUTE_KEY_MAP``) become structured attributes
          ``{"value", "attribute_confidence", "source", "provider"}``.
        - Missing confidence stays ``None`` — NEVER fabricated to 1.0.
        - Unknown attribute keys are preserved in ``meta_extra["unmapped_attribute_keys"]``.
        - bboxes pass through JSON-safe; mask bitmaps are replaced by ``mask_present``.
    """
    warnings: List[WarningItem] = []
    meta_extra: Dict[str, Any] = {}

    if not isinstance(raw, dict):
        warnings.append(_warning(
            "vision_output_schema_mismatch",
            f"Backend output is {type(raw).__name__}, expected dict.",
        ))
        return {}, [], [], warnings, meta_extra

    raw_attrs = raw.get("attributes")
    confidences = raw.get("attribute_confidences") or {}
    attributes: Dict[str, Any] = {}
    unmapped: List[str] = []

    def _add(key: str, value: Any, conf: Any = None) -> None:
        mapped = ATTRIBUTE_KEY_MAP.get(key)
        if mapped is None:
            unmapped.append(key)
            return
        if isinstance(value, dict):
            conf = value.get("attribute_confidence", value.get("confidence", conf))
            value = value.get("value", value.get("label"))
        attributes[mapped] = {
            "value": _json_safe(value),
            "attribute_confidence": conf if isinstance(conf, (int, float)) else None,
            "source": VISUAL_ATTRIBUTE_SOURCE,
            "provider": backend_name,
        }

    if isinstance(raw_attrs, dict):
        for key, value in raw_attrs.items():
            _add(key, value, confidences.get(key))
    elif raw_attrs is not None:
        warnings.append(_warning(
            "vision_output_schema_mismatch",
            f"Backend 'attributes' is {type(raw_attrs).__name__}, expected dict.",
        ))

    # Top-level garment category fields.
    for key in ("garment_category", "class_name", "coarse_class_name"):
        if raw.get(key) and "garment_category" not in attributes:
            _add(key, raw[key], confidences.get(key))

    instances = [
        _normalize_instance(i) for i in (
            raw.get("detections") or raw.get("instances")
            or raw.get("garment_instances") or []
        )
        if isinstance(i, dict)
    ]
    regions = [
        _normalize_instance(r) for r in (raw.get("regions") or [])
        if isinstance(r, dict)
    ]

    if unmapped:
        meta_extra["unmapped_attribute_keys"] = sorted(set(unmapped))
    if not attributes and not instances and not regions:
        warnings.append(_warning(
            "vision_output_empty",
            "Backend returned no recognizable attributes.",
            severity=WarningSeverity.info,
        ))
    return attributes, instances, regions, warnings, meta_extra


# ── Timeout wrapper ────────────────────────────────────────────────────────────


def _call_with_timeout(fn: Any, timeout_ms: int, /, **kwargs: Any) -> Any:
    """Run *fn(**kwargs)* with a wall-clock timeout.

    Uses a single-worker thread (signal-based timeouts are not portable to
    Windows).  On timeout the worker thread cannot be force-killed — it is
    abandoned via ``shutdown(wait=False)``; its eventual result/exception is
    discarded.  Acceptable for P1.1 (backend is a probing shell).
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn, **kwargs)
        return future.result(timeout=max(timeout_ms, 1) / 1000.0)
    finally:
        executor.shutdown(wait=False)


# ── Provider ───────────────────────────────────────────────────────────────────


class RealVisionAttributeProvider(VisionAttributeProvider):
    """Experimental adapter mapping a real vision backend to the serving contract.

    The provider owns the serving contract (validation, timeout, error
    mapping, normalization, no-leak); the injected ``backend_client`` owns the
    actual vision pipeline call.  Default backend is the P1.1 probing shell.
    """

    provider_name = "real"

    def __init__(
        self,
        *,
        backend: str = "fashion_vision_3_1",
        mode: Optional[str] = None,
        checkpoint_root: Optional[str] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        allow_image_url_download: bool = False,
        fail_open_to_mock: bool = True,
        yolo_device: str = "cpu",
        sam_device: str = "cpu",
        cleanup_temp_files: bool = True,
        backend_client: Any = None,
    ) -> None:
        self._timeout_ms = int(timeout_ms)
        self._max_image_bytes = int(max_image_bytes)
        self._allow_image_url_download = bool(allow_image_url_download)
        self._fail_open_to_mock = bool(fail_open_to_mock)
        # backend_client injection is the test seam; nothing heavy is loaded here.
        self._backend = backend_client if backend_client is not None else build_backend(
            backend, mode=mode, checkpoint_root=checkpoint_root,
            yolo_device=yolo_device, sam_device=sam_device,
            cleanup_temp_files=cleanup_temp_files,
        )
        self._backend_name = getattr(self._backend, "name", backend)

    # -- helpers ---------------------------------------------------------------

    def _base_meta(self, *, image: Any, image_url: Optional[str],
                   image_bytes: Any, regions: Optional[List[str]],
                   garment_category: Optional[str],
                   provided_attributes: Optional[Dict[str, Any]],
                   request_id: Optional[str]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "provider": self.provider_name,
            "provider_version": REAL_PROVIDER_VERSION,
            "real_pipeline_enabled": True,
            "vision_backend": self._backend_name,
            "vision_provider_real_enabled": True,
            "has_image": image is not None,
            "has_image_url": bool(image_url),
            "has_image_bytes": image_bytes is not None,
            "requested_regions": list(regions) if regions else [],
            "garment_category": garment_category or "",
            "provided_attributes_present": bool(provided_attributes),
        }
        if request_id:
            meta["request_id"] = request_id
        return meta

    def _result(self, meta: Dict[str, Any], *,
                attributes: Optional[Dict[str, Any]] = None,
                garment_instances: Optional[List[Dict[str, Any]]] = None,
                regions: Optional[List[Dict[str, Any]]] = None,
                sources: Optional[List[Dict[str, Any]]] = None,
                warnings: Optional[List[WarningItem]] = None,
                used_tools: Optional[List[str]] = None) -> VisionAttributeResult:
        return VisionAttributeResult(
            attributes=attributes or {},
            garment_instances=garment_instances or [],
            regions=regions or [],
            sources=sources or [],
            warnings=warnings or [],
            used_tools=used_tools if used_tools is not None else ["real_vision_provider"],
            meta=meta,
        )

    # -- public API (serving contract: same extract() as the mock) --------------

    def extract(
        self,
        *,
        image: Any = None,
        image_url: Optional[str] = None,
        image_bytes: Any = None,
        query: Optional[str] = None,
        garment_category: Optional[str] = None,
        regions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        provided_attributes: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> VisionAttributeResult:
        meta = self._base_meta(
            image=image, image_url=image_url, image_bytes=image_bytes,
            regions=regions, garment_category=garment_category,
            provided_attributes=provided_attributes, request_id=request_id,
        )

        # Rule 1: request-provided attributes are authoritative — never override.
        if provided_attributes:
            return self._result(meta, used_tools=[])

        # Rule 2: no image source at all.
        if not (image_url or image_bytes or image is not None):
            return self._result(meta, warnings=[_warning(
                "vision_input_missing",
                "未提供图片或视觉属性，无法执行视觉属性提取。",
                severity=WarningSeverity.info,
            )])

        # Rule 3: image_url-only input — P1.1 NEVER downloads.
        if image_url and image_bytes is None and image is None:
            meta["error_code"] = "vision_image_url_download_disabled"
            return self._result(meta, warnings=[_warning(
                "vision_image_url_download_disabled",
                "P1.1 real vision provider does not download image_url; "
                "provide image_bytes instead.",
                severity=WarningSeverity.info,
            )])

        # Rule 4: size guard on raw payload (str is base64-ish text per schema).
        if image_bytes is not None and len(image_bytes) > self._max_image_bytes:
            meta["error_code"] = "vision_input_too_large"
            meta["image_bytes_len"] = len(image_bytes)
            return self._result(meta, warnings=[_warning(
                "vision_input_too_large",
                f"image_bytes payload exceeds max {self._max_image_bytes} bytes.",
            )])

        # Rule 5: backend call with timeout + full error mapping.
        t0 = time.perf_counter()
        try:
            raw = _call_with_timeout(
                self._backend.predict, self._timeout_ms,
                image_bytes=image_bytes, image_url=None,  # url never forwarded
                regions=list(regions) if regions else None,
                garment_category=garment_category,
            )
        except FutureTimeoutError:
            meta["error_code"] = "vision_timeout"
            meta["vision_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return self._result(meta, warnings=[_warning(
                "vision_timeout",
                f"Vision backend exceeded {self._timeout_ms} ms timeout.",
            )])
        except VisionProviderUnavailable as exc:
            meta["error_code"] = "vision_provider_unavailable"
            meta["error_details"] = _json_safe(exc.details)
            meta["vision_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return self._result(meta, warnings=[_warning(
                "vision_provider_unavailable",
                "Real vision backend unavailable; no visual attributes produced.",
            )])
        except Exception as exc:  # noqa: BLE001 — endpoint must never crash.
            # NOTE: never include input payloads in the message (no-leak rule).
            meta["error_code"] = "vision_provider_error"
            meta["error_type"] = type(exc).__name__
            meta["vision_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            logger.warning("Real vision backend error: %s", type(exc).__name__)
            return self._result(meta, warnings=[_warning(
                "vision_provider_error",
                "Vision backend call failed; no visual attributes produced.",
            )])

        meta["vision_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        attributes, instances, out_regions, warnings, meta_extra = (
            normalize_vision_backend_output(raw, backend_name=self._backend_name)
        )
        meta.update(meta_extra)

        # Merge JSON-safe backend-level meta (e.g. vision_backend_mode,
        # num_garment_instances) — never overwrites provider core keys.
        backend_meta = raw.get("meta") if isinstance(raw, dict) else None
        if isinstance(backend_meta, dict):
            for key, value in _json_safe(backend_meta).items():
                meta.setdefault(key, value)

        sources = [{
            "type": "visual_attribute",
            "field": key,
            "value": val.get("value") if isinstance(val, dict) else val,
            "attribute_confidence": val.get("attribute_confidence") if isinstance(val, dict) else None,
            "source": VISUAL_ATTRIBUTE_SOURCE,
            "provider": self._backend_name,
        } for key, val in attributes.items()]

        # Backend-declared provenance entries (e.g. segmentation stage) pass
        # through JSON-safe.
        raw_sources = raw.get("sources") if isinstance(raw, dict) else None
        if isinstance(raw_sources, list):
            sources.extend(_json_safe(s) for s in raw_sources if isinstance(s, dict))

        return self._result(
            meta,
            attributes=attributes,
            garment_instances=instances,
            regions=out_regions,
            sources=sources,
            warnings=warnings,
        )
