"""
P0b.1 — ContentGenerationService deterministic skeleton.

Generates title / selling_points / short_description / detail_bullets
from structured attributes using safe templates.  Does NOT call any LLM
or external API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference.serving.content_policy import (
    check_input_attributes,
    check_output_text,
    safety_suffix,
)
from inference.serving.errors import ContentInputMissingError, ContentPolicyBlockedError
from inference.serving.schemas import WarningItem, WarningSeverity

logger = logging.getLogger(__name__)


# ── Result model ───────────────────────────────────────────────────────────────


@dataclass
class ContentGenerationResult:
    content_type: str
    generated_content: Any  # str or list[str]
    content_blocks: List[Dict[str, Any]] = field(default_factory=list)
    used_attributes: Dict[str, Any] = field(default_factory=dict)
    blocked_claims: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type,
            "generated_content": self.generated_content,
            "content_blocks": self.content_blocks,
            "used_attributes": self.used_attributes,
            "blocked_claims": self.blocked_claims,
            "sources": self.sources,
            "meta": self.meta,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_attr(attributes: Dict[str, Any], key: str) -> Optional[Any]:
    """Extract a value from primitive or dict-style attributes."""
    val = attributes.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("value")
    return val


def _build_sources(attributes: Dict[str, Any], safe: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for key in safe:
        val = _extract_attr(attributes, key)
        if val is not None:
            src = {"type": "product_attribute", "field": key, "value": val, "source": "request"}
            if isinstance(attributes.get(key), dict):
                ad = attributes[key]
                src["attribute_confidence"] = ad.get("attribute_confidence")
                src["source"] = ad.get("source", "request")
            sources.append(src)
    return sources


def _garment_label(garment_category: Optional[str], attributes: Dict[str, Any]) -> str:
    gc = garment_category or ""
    if not gc:
        raw = attributes.get("garment_category", {})
        gc = raw.get("value", raw) if isinstance(raw, dict) else str(raw or "")
    return gc or "商品"


# ── Service ────────────────────────────────────────────────────────────────────


class ContentGenerationService:
    """Deterministic content generation from structured attributes.

    Supports: title, selling_points, short_description, detail_bullets.
    Does NOT call any LLM, external API, or visual pipeline.
    """

    def generate(
        self,
        *,
        content_type: str = "selling_points",
        attributes: Optional[Dict[str, Any]] = None,
        garment_category: Optional[str] = None,
        target_channel: Optional[str] = None,
        tone: Optional[str] = None,
        language: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        product_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ContentGenerationResult:
        meta: Dict[str, Any] = {
            "generator": "deterministic_template",
            "llm_used": False,
            "content_type": content_type,
            "target_channel": target_channel,
            "tone": tone,
            "language": language,
            "max_length_applied": False,
        }

        warnings: List[WarningItem] = []
        attrs = dict(attributes) if attributes else {}

        # Check input.
        policy = check_input_attributes(attrs)
        safe_attrs = policy["safe_attributes"]
        blocked_claims = policy["blocked"]
        if blocked_claims:
            warnings.append(WarningItem(
                code="content_policy_blocked", scope="content_generation",
                message="部分输入包含高风险或未验证宣称，已从生成内容中移除。",
                severity=WarningSeverity.warn,
            ))
        meta["blocked_claim_count"] = len(blocked_claims)

        garment = _garment_label(garment_category, attrs)
        sources = _build_sources(attrs, safe_attrs)
        meta["attribute_count"] = len(attrs) if attrs else 0
        meta["used_attribute_count"] = len(safe_attrs)

        if not safe_attrs:
            warnings.append(WarningItem(
                code="content_input_missing", scope="content_generation",
                message="缺少生成内容所需的输入信息。",
                severity=WarningSeverity.info,
            ))

        # Dispatch by content_type.
        content_blocks: list = []
        if content_type == "title":
            result = self._generate_title(safe_attrs, garment, max_length, warnings, meta)
            if result:
                content_blocks.append({"type": "title", "text": result,
                                       "source_fields": [k for k in ("color", "fabric", "style") if k in safe_attrs]})
        elif content_type == "selling_points":
            result = self._generate_selling_points(safe_attrs, garment, max_length, warnings, content_blocks)
        elif content_type == "short_description":
            result = self._generate_short_description(safe_attrs, garment, max_length, warnings, content_blocks)
        elif content_type == "detail_bullets":
            result = self._generate_detail_bullets(safe_attrs, garment, max_length, warnings, content_blocks)
        else:
            result = []
            warnings.append(WarningItem(
                code="content_unsupported_type", scope="content_generation",
                message=f"不支持的内容类型: {content_type}",
                severity=WarningSeverity.info,
            ))

        if max_length and isinstance(result, str) and len(result) > max_length:
            result = result[:max_length]
            meta["max_length_applied"] = True

        return ContentGenerationResult(
            content_type=content_type,
            generated_content=result,
            content_blocks=content_blocks,
            used_attributes=dict(safe_attrs),
            blocked_claims=blocked_claims,
            sources=sources,
            warnings=warnings,
            meta=meta,
        )

    # ── Generators ─────────────────────────────────────────────────────────

    def _generate_title(self, attrs: Dict[str, Any], garment: str,
                        max_len: Optional[int], warnings: list,
                        meta: Optional[dict] = None) -> str:
        parts = []
        for key in ("color", "fabric", "style"):
            v = _extract_attr(attrs, key)
            if v:
                parts.append(str(v))
        if not parts:
            return f"{garment}"
        title = "".join(parts) + garment
        if max_len and len(title) > max_len:
            title = title[:max_len]
            if meta is not None:
                meta["max_length_applied"] = True
        return title

    def _generate_selling_points(self, attrs: Dict[str, Any], garment: str,
                                  max_len: Optional[int], warnings: list,
                                  blocks: Optional[list] = None) -> list:
        points: list = []
        for key in ("fabric", "color", "style"):
            v = _extract_attr(attrs, key)
            if v:
                suffix = safety_suffix(key)
                pt = f"{key}信息：{v}，{suffix}"
                if max_len and len(pt) > max_len:
                    pt = pt[:max_len]
                points.append(pt)
                if blocks is not None:
                    blocks.append({"type": "selling_point", "text": pt, "source_fields": [key]})
        return points

    def _generate_short_description(self, attrs: Dict[str, Any], garment: str,
                                     max_len: Optional[int], warnings: list,
                                     blocks: Optional[list] = None) -> str:
        known = []
        src_fields = []
        for key in ("color", "fabric", "style"):
            v = _extract_attr(attrs, key)
            if v:
                known.append(str(v))
                src_fields.append(key)
        base = f"这款{garment}"
        if known:
            base += f"结合{','.join(known)}等已知信息，可用于日常穿搭场景。"
        base += "具体面料成分和尺码信息建议以商品详情页为准。"
        if max_len and len(base) > max_len:
            base = base[:max_len]
        if blocks is not None:
            blocks.append({"type": "short_description", "text": base, "source_fields": src_fields})
        return base

    def _generate_detail_bullets(self, attrs: Dict[str, Any], garment: str,
                                  max_len: Optional[int], warnings: list,
                                  blocks: Optional[list] = None) -> list:
        mapping = {"fabric": "面料", "color": "颜色", "style": "风格"}
        bullets: list = []
        for key, label in mapping.items():
            v = _extract_attr(attrs, key)
            if v:
                text = f"已知{label}信息为{v}，{safety_suffix(key)}"
                if max_len and len(text) > max_len:
                    text = text[:max_len]
                bullet = {"title": label, "text": text}
                bullets.append(bullet)
                if blocks is not None:
                    blocks.append({"type": "detail_bullet", "title": label, "text": text, "source_fields": [key]})
        return bullets


# ── Singleton ──────────────────────────────────────────────────────────────────

_service: Optional[ContentGenerationService] = None


def get_content_generation_service() -> ContentGenerationService:
    global _service
    if _service is None:
        _service = ContentGenerationService()
    return _service
