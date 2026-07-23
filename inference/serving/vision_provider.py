"""
P0a.6 — VisionAttributeProvider Adapter Skeleton.

Defines the ``VisionAttributeProvider`` interface and a
``MockVisionAttributeProvider`` that returns empty results with appropriate
warnings.  Does NOT call any real vision model, image processing library,
or the 3.1 pipeline.

Design principles:
    - Does NOT fabricate attributes, bboxes, or masks.
    - Explicit failure: ``vision_provider_mock`` when image is provided but
      real pipeline is not connected.
    - ``vision_input_missing`` when no image source is provided.
    - Request-provided ``attributes`` take priority and are NEVER overridden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference.serving.schemas import WarningItem, WarningSeverity

logger = logging.getLogger(__name__)
MOCK_PROVIDER_VERSION = "0.1.0"

# ── Warning helpers (prevent magic strings scattered through the module) ──────

_MOCK_WARNING = WarningItem(
    code="vision_provider_mock", scope="vision",
    message="当前视觉属性提供器为 mock，尚未接入真实视觉模型。",
    severity=WarningSeverity.info,
)
_INPUT_MISSING_WARNING = WarningItem(
    code="vision_input_missing", scope="vision",
    message="未提供图片或视觉属性，无法执行视觉属性提取。",
    severity=WarningSeverity.info,
)


# ── Result model ───────────────────────────────────────────────────────────────


@dataclass
class VisionAttributeResult:
    """Structured output from ``VisionAttributeProvider.extract()``."""

    attributes: Dict[str, Any] = field(default_factory=dict)
    garment_instances: List[Dict[str, Any]] = field(default_factory=list)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    used_tools: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": dict(self.attributes),
            "garment_instances": list(self.garment_instances),
            "regions": list(self.regions),
            "sources": list(self.sources),
            "warnings": [w.model_dump() for w in self.warnings],
            "used_tools": list(self.used_tools),
            "meta": dict(self.meta),
        }


# ── Abstract interface ─────────────────────────────────────────────────────────


class VisionAttributeProvider:
    """Interface for extracting visual attributes from garment images.

    Implementations accept ``image`` / ``image_url`` / ``image_bytes`` and
    return a ``VisionAttributeResult``.  The default singleton is
    ``MockVisionAttributeProvider``.
    """

    def extract(
        self,
        *,
        image: Any = None,
        image_url: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        query: Optional[str] = None,
        garment_category: Optional[str] = None,
        regions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        provided_attributes: Optional[Dict[str, Any]] = None,
    ) -> VisionAttributeResult:
        raise NotImplementedError


# ── Mock implementation ────────────────────────────────────────────────────────


class MockVisionAttributeProvider(VisionAttributeProvider):
    """Mock vision provider — NEVER calls a real model.

    Rules:
    1. If ``provided_attributes`` is non-empty, return immediately with empty
       attributes — do NOT override request-provided attributes.
    2. If no image / image_url / image_bytes → ``vision_input_missing``.
    3. If image is present → ``vision_provider_mock``.
    4. Does NOT fabricate any attributes, bboxes, or masks.
    5. P1.4a: when ``mock_regions`` is provided, it's returned as
       ``localized_regions`` when an image source is present.
    """

    def __init__(self, mock_regions: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__()
        self._mock_regions: List[Dict[str, Any]] = list(mock_regions) if mock_regions else []

    def extract(
        self,
        *,
        image: Any = None,
        image_url: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        query: Optional[str] = None,
        garment_category: Optional[str] = None,
        regions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        provided_attributes: Optional[Dict[str, Any]] = None,
    ) -> VisionAttributeResult:
        has_image = bool(image_url or image_bytes or image)

        meta: Dict[str, Any] = {
            "provider": "mock",
            "provider_version": MOCK_PROVIDER_VERSION,
            "real_pipeline_enabled": False,
            "has_image": bool(image is not None),
            "has_image_url": bool(image_url),
            "has_image_bytes": bool(image_bytes is not None),
            "requested_regions": list(regions) if regions else [],
            "garment_category": garment_category or "",
            "provided_attributes_present": bool(provided_attributes),
        }

        # Case A: provided attributes exist — do NOT override.
        if provided_attributes:
            return VisionAttributeResult(
                attributes={},
                used_tools=[],
                meta=meta,
            )

        # Case B: no image source.
        if not has_image:
            meta["has_image"] = False
            return VisionAttributeResult(
                attributes={},
                used_tools=["mock_vision_provider"],
                warnings=[_INPUT_MISSING_WARNING],
                meta=meta,
            )

        # Case C/D: image present but real pipeline not connected.
        return VisionAttributeResult(
            attributes={},
            garment_instances=[],
            regions=list(self._mock_regions),     # P1.4a: mock localized_regions
            used_tools=["mock_vision_provider"],
            warnings=[_MOCK_WARNING],
            meta=meta,
        )


# ── Lazy singleton with config/env-based selection (P1.1) ──────────────────────

_provider: Optional[VisionAttributeProvider] = None


def _resolve_vision_settings() -> Dict[str, Any]:
    """Resolve vision provider settings from serving_config.yaml + env overrides.

    Env vars win over YAML: VISION_PROVIDER, VISION_REAL_ENABLED,
    VISION_TIMEOUT_MS, VISION_CHECKPOINT_ROOT.  Any parse failure falls back
    to safe mock defaults (never raises).
    """
    import os

    defaults: Dict[str, Any] = {
        "provider": "mock",
        "real_enabled": False,
        "backend": "fashion_vision_3_1",
        "mode": None,
        "checkpoint_root": "outputs/",
        "timeout_ms": 1500,
        "max_image_bytes": 5 * 1024 * 1024,
        "allow_image_url_download": False,
        "fail_open_to_mock": True,
        "yolo_device": "cpu",
        "sam_device": "cpu",
        "cleanup_temp_files": True,
    }
    try:
        from inference.serving.deps import get_config
        vision_cfg = (get_config().get("vision") or {})
        real_cfg = (vision_cfg.get("real_provider") or {})
        settings = dict(defaults)
        settings["provider"] = str(vision_cfg.get("provider", "mock"))
        settings["real_enabled"] = bool(vision_cfg.get("real_enabled", False))
        settings["backend"] = str(real_cfg.get("backend", defaults["backend"]))
        settings["mode"] = real_cfg.get("mode") or None
        settings["checkpoint_root"] = str(real_cfg.get("checkpoint_root", defaults["checkpoint_root"]))
        settings["timeout_ms"] = int(real_cfg.get("timeout_ms", defaults["timeout_ms"]))
        settings["max_image_bytes"] = int(real_cfg.get("max_image_bytes", defaults["max_image_bytes"]))
        settings["allow_image_url_download"] = bool(real_cfg.get("allow_image_url_download", False))
        settings["fail_open_to_mock"] = bool(real_cfg.get("fail_open_to_mock", True))
        settings["yolo_device"] = str(real_cfg.get("yolo_device", "cpu"))
        settings["sam_device"] = str(real_cfg.get("sam_device", "cpu"))
        settings["cleanup_temp_files"] = bool(real_cfg.get("cleanup_temp_files", True))

        env_provider = os.getenv("VISION_PROVIDER")
        if env_provider:
            settings["provider"] = env_provider.strip().lower()
        env_real = os.getenv("VISION_REAL_ENABLED")
        if env_real is not None:
            settings["real_enabled"] = env_real.strip().lower() in ("1", "true", "yes")
        env_timeout = os.getenv("VISION_TIMEOUT_MS")
        if env_timeout:
            settings["timeout_ms"] = int(env_timeout)
        env_root = os.getenv("VISION_CHECKPOINT_ROOT")
        if env_root:
            settings["checkpoint_root"] = env_root
        env_mode = os.getenv("VISION_BACKEND_MODE")
        if env_mode:
            settings["mode"] = env_mode.strip().lower()
        env_device = os.getenv("VISION_DEVICE")
        if env_device:
            # One env var sets both stage devices (cpu | cuda | auto).
            device = env_device.strip().lower()
            settings["yolo_device"] = device
            settings["sam_device"] = device
        return settings
    except Exception:  # noqa: BLE001 — config problems must not break serving.
        logger.warning("Vision settings resolution failed — falling back to mock defaults.")
        return defaults


def _select_vision_provider() -> VisionAttributeProvider:
    """Instantiate the provider selected by config/env.  Default: mock.

    Real provider requires BOTH provider=real AND real_enabled=true.
    Setup failures fall back to mock when fail_open_to_mock is set
    (vision_provider_real_disabled semantics are logged, not raised).
    """
    settings = _resolve_vision_settings()
    if settings["provider"] != "real":
        return MockVisionAttributeProvider()
    if not settings["real_enabled"]:
        logger.warning(
            "vision_provider_real_disabled: VISION_PROVIDER=real requested but "
            "real_enabled is false — using MockVisionAttributeProvider.")
        return MockVisionAttributeProvider()
    try:
        # Lazy import: the real adapter module is never imported unless enabled.
        from inference.serving.real_vision_provider import RealVisionAttributeProvider
        provider = RealVisionAttributeProvider(
            backend=settings["backend"],
            mode=settings.get("mode"),
            checkpoint_root=settings["checkpoint_root"],
            timeout_ms=settings["timeout_ms"],
            max_image_bytes=settings["max_image_bytes"],
            allow_image_url_download=settings["allow_image_url_download"],
            fail_open_to_mock=settings["fail_open_to_mock"],
            yolo_device=settings.get("yolo_device", "cpu"),
            sam_device=settings.get("sam_device", "cpu"),
            cleanup_temp_files=settings.get("cleanup_temp_files", True),
        )
        logger.info("RealVisionAttributeProvider selected (backend=%s, mode=%s)",
                    settings["backend"], settings.get("mode"))
        return provider
    except Exception:  # noqa: BLE001 — setup failure must not break serving.
        logger.exception("Real vision provider setup failed")
        if settings.get("fail_open_to_mock", True):
            logger.warning("fail_open_to_mock: falling back to MockVisionAttributeProvider.")
            return MockVisionAttributeProvider()
        raise


def get_vision_provider() -> VisionAttributeProvider:
    """Return the process-wide vision provider singleton.

    Default is ``MockVisionAttributeProvider``.  ``RealVisionAttributeProvider``
    is selected only when config/env explicitly enables it (see
    ``_resolve_vision_settings``).  Does NOT load any model, download any
    resource, or import heavy libraries.
    """
    global _provider
    if _provider is None:
        _provider = _select_vision_provider()
    return _provider
