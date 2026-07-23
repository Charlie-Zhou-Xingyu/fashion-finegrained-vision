"""
P0a.9 — Serving error taxonomy and custom exceptions.

Provides ``ServingErrorCode`` constants, ``ServingError`` and subclasses
that map cleanly to ``UnifiedResponse`` error envelopes.  All exceptions
are JSON-safe and never expose ``image_bytes`` or stack traces to clients.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ServingErrorCode:
    VALIDATION_ERROR = "validation_error"
    INTERNAL_ERROR = "internal_error"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    TIMEOUT = "timeout"
    CONTENT_POLICY_BLOCKED = "content_policy_blocked"
    CONTENT_INPUT_MISSING = "content_input_missing"
    CONTENT_UNSUPPORTED_TYPE = "content_unsupported_type"


class ServingError(Exception):
    """Base exception for predictable serving-layer errors.

    Maps to ``UnifiedResponse`` with ``status="error"``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 500,
        severity: str = "error",
        scope: str = "serving",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.severity = severity
        self.scope = scope
        self.details = details or {}

    def to_warning_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "scope": self.scope,
        }

    def to_response_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.code,
            "error_message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


class UnsupportedOperationError(ServingError):
    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            code=ServingErrorCode.UNSUPPORTED_OPERATION,
            message=message, status_code=501, **kwargs,
        )


class ContentPolicyBlockedError(ServingError):
    def __init__(self, message: str, blocked: Optional[list] = None, **kwargs: Any):
        super().__init__(
            code=ServingErrorCode.CONTENT_POLICY_BLOCKED,
            message=message, status_code=200,
            severity="warn", scope="content_generation",
            details={"blocked_claims": blocked or []},
        )


class ContentInputMissingError(ServingError):
    def __init__(self, message: str = "缺少生成内容所需的输入信息。", **kwargs: Any):
        super().__init__(
            code=ServingErrorCode.CONTENT_INPUT_MISSING,
            message=message, status_code=200,
            severity="info", scope="content_generation",
        )


class DependencyUnavailableError(ServingError):
    def __init__(self, message: str, **kwargs: Any):
        super().__init__(
            code=ServingErrorCode.DEPENDENCY_UNAVAILABLE,
            message=message, status_code=503,
        )
