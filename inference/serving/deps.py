"""
P0a.1 minimal dependency injection / service state.

Loads configuration from ``configs/serving_config.yaml`` so version strings,
module lists and mock placeholders are not hard-coded in application code.

No models are loaded at this stage.  ``ServiceState`` provides the health-check
payload that advertises which modules are implemented vs pending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Resolve project root relative to this file (inference/serving/deps.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "serving_config.yaml"


def _load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the serving YAML config, returning an empty dict on failure so the
    service can still start with built-in defaults."""
    config_path = path or _DEFAULT_CONFIG_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError) as exc:
        # Let callers decide how to handle — return empty so defaults kick in.
        print(f"[deps] WARNING: could not load {config_path}: {exc}")
        return {}


# Module-level config loaded once at import time.
_config = _load_config()
_service_cfg = _config.get("service", {})
_schema_cfg = _config.get("schema", {})
_health_cfg = _config.get("health", {})
_mock_cfg = _config.get("mock", {})

# Public constants (config-driven, with sensible fallbacks).
SERVICE_NAME: str = _service_cfg.get("name", "fashion-vision-serving")
SERVICE_VERSION: str = _service_cfg.get("version", "p0a-api-schema")
SCHEMA_VERSION: str = _schema_cfg.get("version", "1.0.0")
IMPLEMENTED_MODULES: List[str] = _health_cfg.get("implemented_modules", ["schemas", "app", "deps", "intent_classifier"])
PENDING_MODULES: List[str] = _health_cfg.get("pending_modules", [])
VISUAL_KEYWORDS: List[str] = _mock_cfg.get("visual_keywords", ["图片", "看图", "图里", "这张图", "图中"])
MOCK_QA_ANSWER: str = _mock_cfg.get(
    "qa_answer",
    "P0a API schema is ready. Intent routing and services will be implemented in later steps.",
)
MOCK_NEED_IMAGE_CLARIFICATION: str = _mock_cfg.get(
    "need_image_clarification",
    "该问题需要商品图片来进行视觉分析，请上传图片后重试。",
)


@dataclass
class ServiceState:
    """Application-wide service state exposed via ``/v1/health``.

    Attributes:
        ready: Whether the service is accepting traffic.
        version: Build / deployment version.
        implemented_modules: Modules that are currently live.
        pending_modules: Modules planned but not yet implemented.
    """

    ready: bool = True
    version: str = SERVICE_VERSION
    implemented_modules: List[str] = field(default_factory=lambda: list(IMPLEMENTED_MODULES))
    pending_modules: List[str] = field(default_factory=lambda: list(PENDING_MODULES))


# Singleton — mutated only via tests that need to simulate loading / errors.
_service_state = ServiceState()


def get_service_state() -> ServiceState:
    """Return the process-wide ``ServiceState`` singleton."""
    return _service_state


def get_config() -> Dict[str, Any]:
    """Return the raw configuration dict (useful for debugging)."""
    return dict(_config)
