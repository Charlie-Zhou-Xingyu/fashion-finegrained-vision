"""
P0a.9.1 — Structured logging helpers with automatic sensitive-field redaction.

Uses standard-library ``logging`` only.  No external dependencies.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

_SENSITIVE_KEYS = frozenset({
    "image_bytes", "password", "token", "secret", "authorization",
    "api_key", "credit_card",
})
_MAX_STRING_LEN = 500


def _redact_value(val: Any) -> Any:
    """Recursively redact sensitive values."""
    if isinstance(val, str):
        if len(val) > _MAX_STRING_LEN:
            return f"<redacted:{len(val)}chars>"
        return val
    if isinstance(val, (int, float, bool, type(None))):
        return val
    if isinstance(val, dict):
        return {k: "<REDACTED>" if k in _SENSITIVE_KEYS else _redact_value(v)
                for k, v in val.items()}
    if isinstance(val, list):
        return [_redact_value(v) for v in val]
    return str(val)[:200]


def redact_payload(obj: Any) -> Any:
    """Return a deep copy of *obj* with sensitive keys redacted.

    Sensitive keys: image_bytes, password, token, secret, authorization,
    api_key, credit_card.  Long strings (>500 chars) are replaced with a
    length summary.
    """
    return _redact_value(deepcopy(obj))


def log_event(event: str, **fields: Any) -> None:
    """Emit a structured JSON log event.

    All field values are automatically redacted.  Expected fields:

        event, request_id, method, path, status_code, process_time_ms,
        warning_count, error_code, route, module
    """
    record: Dict[str, Any] = {"event": event}
    record.update(fields)
    safe = _redact_value(record)
    logging.info(json.dumps(safe, ensure_ascii=False, default=str))
