"""
P0b.1 — Content safety policy for deterministic template generation.

Checks input attributes and generated output against a deterministic
banned-token list.  This is NOT comprehensive content safety — it is a
P0 baseline that blocks the most obvious high-risk marketing claims.
Does NOT call any LLM or external service.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Words that MUST NOT appear in generated content.
# Single-char tokens like "最" are excluded to avoid false-positives
# in normal Chinese text. Use specific multi-char phrases instead.
_BLOCKED_TOKENS: List[str] = [
    "绝对", "100%", "永久",
    "最强", "最佳", "最顶级", "全网最低",
    "顶级",
    "医用", "治疗", "瘦身", "燃脂", "抗菌", "防螨",
    "防晒", "认证", "正品保证", "官方认证为最佳",
    "保证", "一定", "完全",
]

# Words to flag in input attributes but not necessarily in output.
_HIGH_RISK_ATTRIBUTE_KEYS: List[str] = [
    "function", "claim", "功效", "certification", "认证",
]

# Selling-point safety suffix.
_SAFETY_SUFFIX_DEFAULTS = {
    "fabric": "具体成分建议以商品详情页或水洗标为准。",
    "color": "实际观感可能受拍摄光线影响。",
    "style": "可作为日常穿搭参考。",
    "default": "详情以商品实际属性和官方说明为准。",
}


def check_input_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """Scan *attributes* for high-risk keys/values.

    Returns:
        ``{"blocked": [...], "safe_attributes": {...}}``
        Each blocked entry is ``{"field": ..., "reason": "high_risk_key"|"blocked_token"}``.
        Does NOT echo raw risky values in output.
    """
    blocked: List[Dict[str, Any]] = []
    safe: Dict[str, Any] = {}

    for key, val in (attributes or {}).items():
        key_lower = key.lower()
        if any(rk in key_lower for rk in _HIGH_RISK_ATTRIBUTE_KEYS):
            blocked.append({"field": key, "reason": "high_risk_attribute_key"})
            continue
        # Check if value contains blocked token.
        val_str = str(val.get("value", val)) if isinstance(val, dict) else str(val)
        token_hit = None
        for token in _BLOCKED_TOKENS:
            if token in val_str:
                token_hit = token
                break
        if token_hit:
            blocked.append({"field": key, "reason": "blocked_token"})
        else:
            safe[key] = val

    return {
        "blocked": blocked,
        "safe_attributes": safe,
    }


def check_output_text(text: str) -> List[str]:
    """Return list of blocked tokens found in *text*."""
    if not text:
        return []
    found: List[str] = []
    for token in _BLOCKED_TOKENS:
        if token in text:
            found.append(token)
    return found


def safety_suffix(attribute_name: str) -> str:
    """Return a safety suffix for *attribute_name*."""
    if attribute_name in _SAFETY_SUFFIX_DEFAULTS:
        return _SAFETY_SUFFIX_DEFAULTS[attribute_name]
    return _SAFETY_SUFFIX_DEFAULTS["default"]
