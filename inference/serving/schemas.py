"""
P0a.1 API schemas — Pydantic v2 models for the fashion-vision serving layer.

All endpoints return a ``UnifiedResponse`` wrapper with disambiguated confidence
fields and structured ``WarningItem`` entries at the top level only.

Status: P0a API schema — intent / attribute / RAG / content services are NOT yet
implemented.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────────


class ResponseStatus(str, Enum):
    success = "success"
    error = "error"
    not_implemented = "not_implemented"


class WarningSeverity(str, Enum):
    info = "info"
    warn = "warn"
    error = "error"


# ── Warning ──────────────────────────────────────────────────────────────────────


class WarningItem(BaseModel):
    """A single structured warning — only appears at the top-level ``warnings`` list.

    Never nest warnings inside ``data`` — keep them at the ``UnifiedResponse`` level.
    """

    code: str
    scope: str
    message: str
    severity: WarningSeverity
    term: Optional[str] = None
    action: Optional[str] = None
    reason: Optional[str] = None


# ── Meta ─────────────────────────────────────────────────────────────────────────


class ResponseMeta(BaseModel):
    """Per-request metadata including version watermarks for cache invalidation."""

    path: Optional[str] = None
    schema_version: str = "1.0.0"
    attr_version: Optional[str] = None
    kb_version: Optional[str] = None
    template_version: Optional[str] = None
    retriever_version: Optional[str] = None
    cache_hit: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)


# ── Unified response wrapper ─────────────────────────────────────────────────────


class UnifiedResponse(BaseModel):
    """Every endpoint returns this envelope.

    ``data`` carries endpoint-specific payload.
    ``warnings`` is the **only** place warnings may appear — never inside ``data``.
    """

    request_id: str
    status: ResponseStatus
    data: Any
    elapsed_ms: float
    used_tools: List[str] = Field(default_factory=list)
    warnings: List[WarningItem] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


# ── Helpers ──────────────────────────────────────────────────────────────────────


def make_request_id() -> str:
    """Generate a short unique request id with ``req_`` prefix."""
    return f"req_{uuid.uuid4().hex[:12]}"


def build_response(
    data: Any,
    *,
    request_id: Optional[str] = None,
    status: ResponseStatus = ResponseStatus.success,
    elapsed_ms: float = 0.0,
    used_tools: Optional[List[str]] = None,
    warnings: Optional[List[WarningItem]] = None,
    meta: Optional[ResponseMeta] = None,
) -> UnifiedResponse:
    """Convenience constructor for ``UnifiedResponse``."""
    return UnifiedResponse(
        request_id=request_id or make_request_id(),
        status=status,
        data=data,
        elapsed_ms=elapsed_ms,
        used_tools=used_tools or [],
        warnings=warnings or [],
        meta=meta or ResponseMeta(),
    )


# ── Attribute value ──────────────────────────────────────────────────────────────


class AttributeValue(BaseModel):
    """A single attribute prediction.

    Uses ``attribute_confidence`` (not a bare ``confidence`` field) to avoid
    ambiguity with intent / RAG / answer confidence.
    """

    value: Optional[Any] = None
    attribute_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: Optional[str] = None
    warning: Optional[str] = None
    composition_verified: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Dialogue ─────────────────────────────────────────────────────────────────────


class DialogueTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# ── Source item (shared by QA and content) ───────────────────────────────────────


class SourceItem(BaseModel):
    """A single evidence source backing an answer.

    Uses ``attribute_confidence`` for attribute sources and ``rag_score`` for
    knowledge-base sources — they are never collapsed into one ``confidence`` field.
    """

    type: str
    id: Optional[str] = None
    field: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    value: Optional[Any] = None
    attribute_confidence: Optional[float] = None
    rag_score: Optional[float] = None
    source: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── /v1/mm/qa ────────────────────────────────────────────────────────────────────


class MultimodalQARequest(BaseModel):
    """Multi-modal QA request.

    Only ``query`` is required.  Pure knowledge questions may omit
    ``product_id`` / ``image_url`` / ``image_bytes`` / ``attributes``.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    product_id: Optional[str] = None
    image_url: Optional[str] = None
    # TODO: add max_length before accepting real base64 uploads (P0a mock).
    image_bytes: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    garment_category: Optional[str] = None
    regions: Optional[List[str]] = None
    session_id: Optional[str] = None
    dialogue_context: List[DialogueTurn] = Field(default_factory=list)
    stream: bool = False
    max_answer_length: int = Field(default=200, ge=1, le=2000)


# ── P1.4a Localized Region ────────────────────────────────────────────────────


class LocalizedRegion(BaseModel):
    """A single localized region from 3.1.2 (bbox-only, no mask/crop/path).

    Forbidden in this model: mask bitmap, crop image, image bytes, temp path,
    local file path, raw tensor, checkpoint path.
    """

    region_id: str
    part_type: str
    part_group: str = "unknown"
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: str = "fashion_vision_3_1_2"
    backend: str = "mock"
    mask_present: bool = False
    mask_ref: Optional[str] = None
    instance_id: Optional[str] = None  # P1.4e: which 3.1.1 garment instance


class LocalizedRegionSummary(BaseModel):
    """Safe subset of LocalizedRegion for API response meta (no mask/path)."""

    region_id: str
    part_type: str
    part_group: str = "unknown"
    bbox: List[float]
    confidence: Optional[float] = None
    source: str = "fashion_vision_3_1_2"
    backend: str = "mock"
    instance_id: Optional[str] = None  # P1.4e


class MultimodalQAData(BaseModel):
    """Payload inside ``UnifiedResponse.data`` for ``/v1/mm/qa``."""

    answer: Optional[str] = None
    answer_type: str = "mock"
    answer_confidence: Optional[float] = None
    intent_confidence: Optional[float] = None
    sources: List[SourceItem] = Field(default_factory=list)
    is_cached: bool = False
    need_image: bool = False
    clarification: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# ── /v1/intent/classify ──────────────────────────────────────────────────────────


class IntentClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    dialogue_context: List[DialogueTurn] = Field(default_factory=list)


class IntentClassifyData(BaseModel):
    primary_intent: str = "fallback_unknown"
    sub_intent: Optional[str] = None
    intent_confidence: float = 0.0
    classifier_level: str = "mock"
    entities: Dict[str, Any] = Field(default_factory=dict)
    alternative_intents: List[Dict[str, Any]] = Field(default_factory=list)


# ── /v1/rag/retrieve ─────────────────────────────────────────────────────────────


class RAGRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    categories: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    include_scores: bool = True
    use_reranker: bool = False
    primary_intent: Optional[str] = None
    sub_intent: Optional[str] = None
    attribute_context: Optional[Dict[str, Any]] = None


class RAGResultItem(BaseModel):
    id: str
    category: str
    term: str
    zh_term: Optional[str] = None
    title: str
    content_snippet: str
    score: Optional[float] = None
    match_type: str = "bm25"
    source: str
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    source_ref: Optional[Dict[str, Any]] = None
    allowed_usage: List[str] = Field(default_factory=list)
    risk_level: Optional[str] = None
    risk_note: Optional[str] = None
    review_status: Optional[str] = None
    reviewed_by: Optional[str] = None
    last_reviewed_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGRetrieveData(BaseModel):
    query: str = ""
    normalized_query: str = ""
    hits: List[RAGResultItem] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


# ── /v1/merchant/content/generate ────────────────────────────────────────────────


class ContentGenerateOptions(BaseModel):
    attribute_tags: bool = True
    selling_points: bool = True
    product_copy: bool = True


class MerchantContentRequest(BaseModel):
    product_id: Optional[str] = None
    image_url: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    style: str = "professional"
    max_copy_length: int = Field(default=150, ge=1, le=2000)
    generate_options: ContentGenerateOptions = Field(default_factory=ContentGenerateOptions)
    batch: bool = False
    content_type: str = "selling_points"
    garment_category: Optional[str] = None


class AttributeTag(BaseModel):
    task: str
    value: Optional[Any] = None
    attribute_confidence: Optional[float] = None
    source: Optional[str] = None
    composition_verified: Optional[bool] = None


class SellingPoint(BaseModel):
    point: str
    source_facts: List[str] = Field(default_factory=list)
    risk_score: Optional[float] = None


class MerchantContentData(BaseModel):
    content_type: str = "selling_points"
    generated_content: Any = None
    content_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    used_attributes: Dict[str, Any] = Field(default_factory=dict)
    blocked_claims: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
