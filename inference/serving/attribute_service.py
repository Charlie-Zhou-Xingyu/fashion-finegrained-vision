"""
P0a.3 — Attribute Service (L1 fast-path).

Given a request's ``attributes`` dict, AttributeService normalises the input,
looks up the requested attribute by name (with alias resolution), applies
source-based template rendering, confidence policies, and fabric-text
sanitisation, and returns a structured ``AttributeAnswer`` ready for use by
the ``QaOrchestrator`` (P0a.5).

Does NOT depend on Redis, RAG, FAISS, BGE, LLM, or the 3.1 vision pipeline.

answer_confidence policy
-------------------------
``answer_confidence`` is derived directly from ``attribute_confidence``:
- If ``attribute_confidence`` is ``None`` → ``answer_confidence = None``
- Otherwise ``answer_confidence = attribute_confidence``
- It is **never** auto-set to 1.0 based on source type (e.g. ``manual_verified``).

Usage::

    from inference.serving.attribute_service import AttributeService

    svc = AttributeService()
    result = svc.answer_attribute(
        "fabric",
        {"fabric": {"value": "纯棉", "attribute_confidence": 0.92, "source": "merchant_input"}},
    )
    # -> AttributeAnswer(answer="这件商品的面料信息标注为纯棉。...", ...)
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from inference.serving.schemas import (
    AttributeValue,
    SourceItem,
    WarningItem,
    WarningSeverity,
)

logger = logging.getLogger(__name__)

# Resolve project root relative to this file.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TEMPLATE_PATH = _PROJECT_ROOT / "configs" / "attribute_templates.yaml"

# ── Allowed fields when constructing AttributeValue from a raw dict ────────────
_ATTR_VALUE_WHITELIST = frozenset({
    "value", "attribute_confidence", "source",
    "composition_verified", "display_value", "unit",
})


# ── Percentage stripping patterns ──────────────────────────────────────────────

# Leading: "100%"/"100％"/"100 %"/"百分百"/"百分之百"
_PCT_LEADING = re.compile(
    r"^(100\s*[%％]|百分百|百分之百)\s*"
)
# Trailing: "100%"/"100％"/"100 %"
_PCT_TRAILING = re.compile(
    r"\s*(100\s*[%％])$"
)


def _is_numeric_string(s: str) -> Optional[float]:
    """Try to parse *s* as a float.  Returns float or None."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ── AttributeAnswer ────────────────────────────────────────────────────────────


class AttributeAnswer:
    """Structured result from ``AttributeService.answer_attribute()``.

    Designed to be trivially converted into ``MultimodalQAData`` by the
    ``QaOrchestrator`` in P0a.5.
    """

    __slots__ = ("answer", "answer_type", "answer_confidence",
                 "sources", "warnings", "used_tools", "meta")

    def __init__(
        self,
        answer: str,
        *,
        answer_type: str = "attribute_query",
        answer_confidence: Optional[float] = None,
        sources: Optional[List[SourceItem]] = None,
        warnings: Optional[List[WarningItem]] = None,
        used_tools: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.answer = answer
        self.answer_type = answer_type
        self.answer_confidence = answer_confidence
        self.sources = sources or []
        self.warnings = warnings or []
        self.used_tools = used_tools or ["attribute_service"]
        self.meta = meta or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dict compatible with ``MultimodalQAData``.

        Sources are expanded to dicts; warnings use ``WarningItem.model_dump()``.
        """
        return {
            "answer": self.answer,
            "answer_type": self.answer_type,
            "answer_confidence": self.answer_confidence,
            "sources": [s.model_dump() for s in self.sources],
            "is_cached": False,
            "need_image": False,
            "clarification": None,
        }


# ── Template loader & validation ───────────────────────────────────────────────


_MANDATORY_TEMPLATE_KEYS = frozenset({"display_name", "unavailable", "default"})
_FABRIC_EXTRA_KEYS = frozenset({"merchant_input", "model_prediction", "manual_verified", "request_raw"})


def _validate_templates(data: Dict[str, Any]) -> List[str]:
    """Validate the loaded template config.  Returns a list of issues (empty = OK).

    Fatal issues raise ``RuntimeError``; warnings are logged.
    """
    issues: List[str] = []

    if "version" not in data:
        issues.append("missing 'version' field")
    thresholds = data.get("thresholds", {})
    if not isinstance(thresholds, dict) or "low_confidence" not in thresholds:
        issues.append("thresholds.low_confidence is required")
    attrs = data.get("attributes")
    if not isinstance(attrs, dict):
        issues.append("attributes must be a dict")
        return issues  # can't validate further

    aliases = data.get("aliases", {})
    if not isinstance(aliases, dict):
        issues.append("aliases must be a dict if present")

    for name, cfg in attrs.items():
        if not isinstance(cfg, dict):
            issues.append(f"attributes.{name} must be a dict")
            continue
        missing = _MANDATORY_TEMPLATE_KEYS - set(cfg.keys())
        if missing:
            issues.append(f"attributes.{name} missing keys: {sorted(missing)}")
        if name == "fabric":
            fabric_missing = _FABRIC_EXTRA_KEYS - set(cfg.keys())
            if fabric_missing:
                issues.append(f"attributes.fabric missing source templates: {sorted(fabric_missing)}")

    return issues


def _load_templates(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate the attribute templates YAML.

    Returns a dict with ``version``, ``thresholds``, ``aliases`` and
    ``attributes``.  Raises ``RuntimeError`` on missing file, YAML error,
    or missing mandatory template keys — the service cannot operate without
    well-formed templates, so fail-fast is correct.
    """
    config_path = path or _DEFAULT_TEMPLATE_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        logger.error("Attribute templates not found: %s", config_path)
        raise RuntimeError(f"Attribute templates not found: {config_path}")
    except yaml.YAMLError as exc:
        logger.error("Attribute templates YAML parse error in %s: %s", config_path, exc)
        raise RuntimeError(f"Invalid YAML in attribute templates: {config_path}") from exc
    except OSError as exc:
        logger.error("Cannot read attribute templates %s: %s", config_path, exc)
        raise RuntimeError(f"Cannot read attribute templates: {config_path}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Attribute templates root must be a dict: {config_path}")

    issues = _validate_templates(data)
    fatal = [i for i in issues if not i.startswith("aliases")]
    if fatal:
        for msg in fatal:
            logger.error("Attribute template validation FAILED: %s", msg)
        raise RuntimeError(
            f"Attribute templates validation failed at {config_path}: "
            f"{'; '.join(fatal)}"
        )
    for msg in issues:
        if msg not in fatal:
            logger.warning("Attribute template validation: %s", msg)

    return data


# ── Normalisation helpers ──────────────────────────────────────────────────────


def _safe_float_confidence(raw: Any) -> Optional[float]:
    """Convert *raw* to a float in [0, 1].  Returns None if invalid or out of range.

    Logs a warning on out-of-range or non-numeric values so operators can
    detect upstream issues without the service crashing.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if 0.0 <= raw <= 1.0:
            return float(raw)
        logger.warning("attribute_confidence out of range: %s", raw)
        return None
    if isinstance(raw, str):
        parsed = _is_numeric_string(raw)
        if parsed is not None and 0.0 <= parsed <= 1.0:
            return parsed
        logger.warning("attribute_confidence non-numeric or out of range: %r", raw)
        return None
    logger.warning("attribute_confidence unsupported type %s: %r", type(raw).__name__, raw)
    return None


def normalize_attributes(
    raw_attrs: Mapping[str, Any],
) -> Dict[str, AttributeValue]:
    """Convert a mixed dict of ``AttributeValue`` / dict / primitive values into
    a uniform ``{name: AttributeValue}`` mapping.

    Rules (revised P0a.3 hardened):
        - ``AttributeValue`` → deep-copied; if ``source`` is None/empty
          it is left as-is (the caller's responsibility).
        - ``dict`` → only whitelisted fields extracted; missing ``source``
          defaults to ``"request_raw"``.
        - ``str`` / ``int`` / ``float`` / ``bool`` →
          ``AttributeValue(value=..., attribute_confidence=None, source="request_raw")``.
        - Unknown types are skipped with a log warning.
        - ``attribute_confidence`` outside [0,1] or non-numeric → clamped to None
          with a warning.

    Returns an empty dict if *raw_attrs* is empty or None.
    """
    if not raw_attrs:
        return {}

    result: Dict[str, AttributeValue] = {}
    for key, val in raw_attrs.items():
        try:
            if isinstance(val, AttributeValue):
                result[key] = deepcopy(val)
            elif isinstance(val, dict):
                # Whitelist extraction.
                kwargs: Dict[str, Any] = {}
                extra: Dict[str, Any] = {}
                for k, v in val.items():
                    if k in _ATTR_VALUE_WHITELIST:
                        kwargs[k] = v
                    else:
                        extra[k] = v
                if extra:
                    logger.debug("attribute_service: ignored unknown keys for %s: %s", key, sorted(extra))
                # Default source.
                if "source" not in kwargs or kwargs["source"] is None:
                    kwargs.setdefault("source", "request_raw")
                # Safe confidence conversion.
                if "attribute_confidence" in kwargs:
                    kwargs["attribute_confidence"] = _safe_float_confidence(
                        kwargs["attribute_confidence"]
                    )
                result[key] = AttributeValue(**kwargs)
            elif isinstance(val, (str, int, float, bool)):
                result[key] = AttributeValue(
                    value=val,
                    attribute_confidence=None,
                    source="request_raw",
                )
            else:
                logger.warning(
                    "attribute_service: skipping unsupported type %s for key=%s",
                    type(val).__name__, key,
                )
        except Exception:
            logger.exception("attribute_service: failed to normalise attribute key=%s", key)
    return result


def sanitize_fabric_value(value: Any, composition_verified: Optional[bool]) -> str:
    """Return a display-safe fabric string by stripping percentage claims when
    *composition_verified* is not ``True``.

    Stripped variants (leading + trailing):
        - ``100%纯棉``
        - ``100％纯棉``
        - ``100 % 纯棉``
        - ``百分百纯棉``
        - ``百分之百纯棉``
        - ``纯棉100%``
        - ``纯棉 100％``

    If *composition_verified* is ``True``, the raw value is returned intact.
    """
    text = str(value).strip() if value is not None else ""
    if composition_verified is True:
        return text
    text = _PCT_LEADING.sub("", text).strip()
    text = _PCT_TRAILING.sub("", text).strip()
    return text


# ── Confidence helpers ─────────────────────────────────────────────────────────


def _confidence_warning(
    attr: AttributeValue,
    thresholds: Dict[str, float],
) -> Optional[WarningItem]:
    """Return a ``WarningItem`` if *attr* confidence is below threshold.

    ``attribute_confidence`` must be a float in [0,1] at this point
    (validation happened during normalisation).
    """
    conf = attr.attribute_confidence
    if conf is None:
        return None
    low = thresholds.get("low_confidence", 0.6)
    if conf < low:
        return WarningItem(
            code="low_attribute_confidence",
            scope=str(attr.value) if attr.value is not None else "",
            message="该属性识别置信度较低，仅供参考。",
            severity=WarningSeverity.warn,
        )
    return None


def _compute_answer_confidence(attr: Optional[AttributeValue]) -> Optional[float]:
    """Derive ``answer_confidence`` from ``AttributeValue.attribute_confidence``.

    - ``attr`` is None (missing) → ``None``
    - ``attr.attribute_confidence`` is None → ``None``
    - otherwise → ``attr.attribute_confidence``

    ``answer_confidence`` is NEVER auto-set to 1.0.
    """
    if attr is None:
        return None
    return attr.attribute_confidence


# ── AttributeService ───────────────────────────────────────────────────────────


class AttributeService:
    """L1 fast-path service for attribute-driven question answering.

    Loads templates from a YAML config, normalises incoming attributes,
    resolves aliases, selects the correct template variant based on source,
    applies confidence policies and fabric sanitisation, and returns a
    structured ``AttributeAnswer``.

    Does NOT call any external service, embedding model, or LLM.
    """

    def __init__(self, template_path: Optional[Path] = None) -> None:
        data = _load_templates(template_path)
        self._version: str = data.get("version", "1.0.0")
        self._thresholds: Dict[str, float] = data.get("thresholds", {})
        self._aliases: Dict[str, str] = data.get("aliases", {})
        self._attr_cfgs: Dict[str, Dict[str, str]] = data.get("attributes", {})
        self._supported_attrs: frozenset = frozenset(self._attr_cfgs.keys())
        logger.info("AttributeService loaded: %d attribute templates, version=%s",
                     len(self._attr_cfgs), self._version)

    # ── Public API ─────────────────────────────────────────────────────────

    def answer_attribute(
        self,
        attribute_name: str,
        attributes: Mapping[str, Any],
        *,
        product_id: Optional[str] = None,
        garment_category: Optional[str] = None,
        locale: str = "zh-CN",
    ) -> AttributeAnswer:
        """Answer a single-attribute query.

        Args:
            attribute_name: The user-facing attribute key (e.g. ``"fabric"``,
                ``"collar"``).  Aliases are resolved internally via
                ``configs/attribute_templates.yaml``.
            attributes: Raw attributes dict — may contain ``AttributeValue``
                objects, plain dicts, or primitive values.
            product_id: Optional; logged but not used for lookup.
            garment_category: e.g. ``"top"``, ``"pants"``.  Used in template
                rendering as ``{garment_label}``.  Falls back to ``"商品"``.
            locale: Locale code (reserved for future multi-locale support).

        Returns:
            ``AttributeAnswer`` — ``meta`` includes ``requested_attribute`` and
            ``resolved_attribute`` for observability.
        """
        norm = normalize_attributes(attributes)
        garment_label = self._resolve_garment_label(garment_category, norm)

        # Resolve alias.
        resolved_name = self._aliases.get(attribute_name, attribute_name)
        attr = norm.get(resolved_name)
        meta: Dict[str, Any] = {
            "requested_attribute": attribute_name,
            "resolved_attribute": resolved_name,
            "resolved_by": "direct",
        }

        warnings: List[WarningItem] = []
        sources: List[SourceItem] = []

        # Length fallback.
        if attr is None and resolved_name == "length":
            attr, fallback_name = self._resolve_length_fallback(norm)
            if attr is not None:
                resolved_name = fallback_name
                meta["resolved_attribute"] = fallback_name
                meta["resolved_by"] = "length_fallback"

        is_unknown = attribute_name not in self._supported_attrs and attribute_name not in self._aliases

        if attr is None:
            template = self._get_template(resolved_name, "unavailable")
            answer_text = self._safe_format(template, garment_label=garment_label, value="")
            warnings.append(WarningItem(
                code="unknown_attribute_type" if is_unknown else "attribute_unavailable",
                scope=attribute_name,
                message=(
                    f"请求的属性 '{attribute_name}' 不在支持的属性列表中"
                    if is_unknown else
                    f"请求的属性 {attribute_name} 不存在或不可用。"
                ),
                severity=WarningSeverity.info,
            ))
        else:
            if is_unknown:
                # Attribute value present for an unsupported type → general template + warning.
                warnings.append(WarningItem(
                    code="unknown_attribute_type",
                    scope=attribute_name,
                    message=f"请求的属性 '{attribute_name}' 不在支持的属性列表中，"
                            f"已尝试从原始输入中提取值。",
                    severity=WarningSeverity.info,
                ))
            # Attribute present.
            display_value = self._render_value(attr, resolved_name)
            template_key = self._template_key_for_source(attr.source)
            answer_text = self._safe_format(
                self._get_template(resolved_name, template_key),
                value=display_value, garment_label=garment_label,
            )

            cw = _confidence_warning(attr, self._thresholds)
            if cw is not None:
                cw.scope = resolved_name
                warnings.append(cw)

            sources.append(SourceItem(
                type="product_attribute",
                field=resolved_name,
                value=attr.value,
                attribute_confidence=attr.attribute_confidence,
                source=attr.source,
            ))

        return AttributeAnswer(
            answer=answer_text,
            answer_type="attribute_query",
            answer_confidence=_compute_answer_confidence(attr),
            sources=sources,
            warnings=warnings,
            meta=meta,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_garment_label(
        garment_category: Optional[str],
        norm: Dict[str, AttributeValue],
    ) -> str:
        if garment_category:
            return garment_category
        gc = norm.get("garment_category")
        if gc and gc.value:
            return str(gc.value)
        return "商品"

    def _resolve_length_fallback(
        self,
        norm: Dict[str, AttributeValue],
    ) -> Tuple[Optional[AttributeValue], str]:
        """Length fallback order: coat → dress → skirt → pant."""
        for candidate in ("coat_length", "dress_length", "skirt_length", "pant_length"):
            a = norm.get(candidate)
            if a is not None and a.value is not None:
                return a, candidate
        return None, "length"

    def _render_value(self, attr: AttributeValue, attr_name: str) -> str:
        raw = str(attr.value) if attr.value is not None else ""
        if attr_name == "fabric":
            return sanitize_fabric_value(raw, attr.composition_verified)
        return raw

    @staticmethod
    def _template_key_for_source(source: Optional[str]) -> str:
        if source in (None, "", "request_raw"):
            return "request_raw"
        if source not in ("merchant_input", "model_prediction", "manual_verified"):
            return "default"
        return source

    def _get_template(self, attr_name: str, key: str) -> str:
        """Resolve the template string for *(attr_name, key)*.

        Falls back: *key* → ``"default"`` → ``"unavailable"`` → bare string.
        """
        cfg = self._attr_cfgs.get(attr_name, {})
        template = cfg.get(key) or cfg.get("default") or cfg.get("unavailable")
        if template is not None:
            return template
        logger.warning("No template for attr=%s key=%s", attr_name, key)
        return "暂未获取到该属性信息，建议查看商品详情页。"

    @staticmethod
    def _safe_format(template: str, **kwargs: str) -> str:
        """Format *template* ignoring unused kwargs so that templates without
        ``{value}`` don't raise ``KeyError``."""
        import string
        formatter = string.Formatter()
        needed = {fname for _, fname, _, _ in formatter.parse(template) if fname is not None}
        subset = {k: v for k, v in kwargs.items() if k in needed}
        return template.format(**subset)


# ── Module-level singleton ─────────────────────────────────────────────────────

_service: Optional[AttributeService] = None


def get_attribute_service() -> AttributeService:
    global _service
    if _service is None:
        _service = AttributeService()
    return _service
