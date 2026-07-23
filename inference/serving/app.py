"""
P0a.1 FastAPI application — schema and endpoint skeletons only.

No intent classifier, attribute service, RAG, cache, content generation, or
LLM models are loaded.  Every business-logic endpoint returns a mock/placeholder
response so the API contract can be validated end-to-end.

Start with::

    uvicorn inference.serving.app:app --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from inference.serving.deps import (
    MOCK_NEED_IMAGE_CLARIFICATION,
    MOCK_QA_ANSWER,
    SCHEMA_VERSION,
    SERVICE_NAME,
    SERVICE_VERSION,
    VISUAL_KEYWORDS,
    get_service_state,
)
from inference.serving.intent_classifier import get_classifier
from inference.serving.qa_orchestrator import get_qa_orchestrator
from inference.serving.vision_provider import get_vision_provider
from inference.serving.content_generation_service import get_content_generation_service
from inference.serving.errors import ServingError
from inference.serving.rag_service import get_rag_service
from inference.serving.schemas import (
    ResponseMeta,
    ResponseStatus,
    UnifiedResponse,
    WarningItem,
    WarningSeverity,
    build_response,
    make_request_id,
    IntentClassifyData,
    IntentClassifyRequest,
    MerchantContentData,
    MerchantContentRequest,
    MultimodalQAData,
    MultimodalQARequest,
    RAGRetrieveData,
    RAGRetrieveRequest,
)


# ── Lifespan ─────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook.  No models to load at P0a."""
    get_service_state().ready = True
    yield


# ── App ──────────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Fashion Vision Serving",
    description="P0a/P0b serving skeleton — deterministic QA, RAG, content generation.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# ── Request-tracing counters (in-process, not thread-safe) ────────────────────
_request_counts: dict = {"total": 0, "errors": 0}
_endpoint_counts: dict = {}
_endpoint_error_counts: dict = {}


# ── ASGI middleware: request_id + process_time header ──────────────────────────


@app.middleware("http")
async def _request_tracing_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or make_request_id()
    request.state.request_id = rid
    t0 = time.perf_counter()
    path = request.url.path

    _request_counts["total"] += 1
    _endpoint_counts[path] = _endpoint_counts.get(path, 0) + 1

    try:
        response = await call_next(request)
    except Exception:
        _request_counts["errors"] += 1
        _endpoint_error_counts[path] = _endpoint_error_counts.get(path, 0) + 1
        raise

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-MS"] = str(elapsed)
    return response


# ── Global error handler ────────────────────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Unify validation errors into the UnifiedResponse error envelope."""
    from fastapi.exceptions import RequestValidationError as RVE
    rid = getattr(request.state, "request_id", make_request_id())
    return JSONResponse(
        status_code=422,
        content={
            "request_id": rid,
            "status": "error",
            "data": None,
            "elapsed_ms": 0.0,
            "used_tools": [],
            "warnings": [{
                "code": "validation_error",
                "message": "请求参数校验失败。",
                "severity": "error",
                "scope": "serving",
            }],
            "meta": {
                "error_code": "validation_error",
                "path": request.url.path,
                "method": request.method,
                "validation_error_count": len(exc.errors()) if hasattr(exc, "errors") else 1,
            },
        },
    )


@app.exception_handler(ServingError)
async def _serving_error_handler(request: Request, exc: ServingError) -> JSONResponse:
    rid = getattr(request.state, "request_id", make_request_id())
    logging.error("ServingError: %s %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": rid,
            "status": "error",
            "data": None,
            "elapsed_ms": 0.0,
            "used_tools": [],
            "warnings": [exc.to_warning_dict()],
            "meta": {"error_code": exc.code, "path": "error_handler"},
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so the API never leaks a bare traceback to callers.

    Detailed error information is logged server-side via ``logging.exception``.
    The client only receives a generic message.
    """
    logging.exception("Unhandled exception in %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "request_id": make_request_id(),
            "status": "error",
            "data": None,
            "elapsed_ms": 0.0,
            "used_tools": [],
            "warnings": [
                {
                    "code": "internal_error",
                    "scope": "server",
                    "message": "An unexpected error occurred. Check server logs for details.",
                    "severity": "error",
                }
            ],
            "meta": {
                "path": "error_handler",
                "schema_version": SCHEMA_VERSION,
                "cache_hit": False,
            },
        },
    )


# ── Shared helpers ───────────────────────────────────────────────────────────────


def _extract_request_id(request: Request) -> str:
    """Return the request id from middleware state, header, or generate new."""
    from_state = getattr(request.state, "request_id", None)
    if from_state:
        return from_state
    header = request.headers.get("X-Request-ID")
    return header if header else make_request_id()


def _needs_image_heuristic(query: str) -> bool:
    """Return True when *query* reads like a visual-reference question.

    This is a P0a placeholder.  The real IntentRouter will replace this logic
    in P0a.2.
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in VISUAL_KEYWORDS)


def _finish(
    request: Request,
    t0: float,
    *,
    data: Any,
    status: ResponseStatus = ResponseStatus.success,
    tools: List[str],
    path: str,
    warnings: List[WarningItem] | None = None,
) -> UnifiedResponse:
    """Shared tail-call: compute elapsed, build ``UnifiedResponse``.

    Extracts the repeated ``t0 / build_response`` boilerplate into one helper
    so every endpoint handler stays concise.
    """
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    raw = data.model_dump() if hasattr(data, "model_dump") else data
    return build_response(
        raw,
        request_id=_extract_request_id(request),
        status=status,
        elapsed_ms=elapsed,
        used_tools=tools,
        warnings=warnings or [],
        meta=ResponseMeta(path=path),
    )


# ── /v1/health ───────────────────────────────────────────────────────────────────


@app.get("/v1/health")
async def health(request: Request) -> UnifiedResponse:
    """Return service readiness, version, and module inventory.

    P0a semantics: ``ready=True`` means the API schema layer is healthy.
    It does NOT imply that models, knowledge bases, or RAG indexes are loaded.
    TODO(P0a.4): after knowledge-base loading is implemented, check KB status
    and set ``ready=False`` + appropriate warnings if the KB failed to load.
    """
    t0 = time.perf_counter()
    state = get_service_state()
    return _finish(
        request, t0,
        data={
            "service": SERVICE_NAME,
            "ready": state.ready,
            "version": state.version,
            "implemented_modules": state.implemented_modules,
            "pending_modules": state.pending_modules,
        },
        tools=["health_check"],
        path="health",
    )


# ── /v1/metrics ──────────────────────────────────────────────────────────────────


@app.get("/v1/metrics")
async def metrics(request: Request) -> UnifiedResponse:
    """Expose in-process request counts (lightweight, no Prometheus dependency)."""
    state = get_service_state()
    data = {
        "service": "fashion-mm-serving",
        "requests_total": _request_counts["total"],
        "errors_total": _request_counts["errors"],
        "warnings_total": 0,  # accumulated elsewhere
        "endpoint_counts": dict(_endpoint_counts),
        "endpoint_error_counts": dict(_endpoint_error_counts),
        "implemented_modules": state.implemented_modules,
        "pending_modules": state.pending_modules,
    }
    return build_response(
        data, request_id=_extract_request_id(request), elapsed_ms=0.0,
        used_tools=["metrics"], meta=ResponseMeta(path="metrics"),
    )


# ── /v1/mm/qa ────────────────────────────────────────────────────────────────────


@app.post("/v1/mm/qa")
async def multimodal_qa(body: MultimodalQARequest, request: Request) -> UnifiedResponse:
    """Real multi-modal QA dispatch via QaOrchestrator.
    P0a.6: vision_provider (mock) is used when image source is present
    and request attributes are absent."""
    t0 = time.perf_counter()
    orchestrator = get_qa_orchestrator()
    result = orchestrator.answer(
        query=body.query,
        image=None,
        image_url=body.image_url,
        image_bytes=body.image_bytes,
        attributes=body.attributes if body.attributes else None,
        garment_category=body.garment_category,
        regions=body.regions,
        context=None,
        request_id=_extract_request_id(request),
    )
    data = MultimodalQAData(
        answer=result.answer,
        answer_type=result.answer_type,
        answer_confidence=result.answer_confidence,
        intent_confidence=result.intent.get("confidence"),
        sources=result.sources,
        is_cached=False,
        need_image=False,
        clarification=None,
        meta=result.meta,
    )
    return _finish(request, t0, data=data.model_dump(), tools=result.used_tools,
                   path=result.meta.get("route", "qa"),
                   warnings=list(result.warnings))


# ── /v1/intent/classify ──────────────────────────────────────────────────────────


@app.post("/v1/intent/classify")
async def intent_classify(body: IntentClassifyRequest, request: Request) -> UnifiedResponse:
    """Classify a user query into a primary / sub intent using keyword and regex
    rules.  Returns ``fallback_unknown`` when no rule matches (this is normal —
    NOT an error)."""
    t0 = time.perf_counter()
    classifier = get_classifier()
    data = classifier.classify(body.query)
    return _finish(request, t0, data=data, tools=["intent_classifier_rule"], path="rule")


# ── /v1/rag/retrieve ─────────────────────────────────────────────────────────────
#
# NOTE: P0a mock endpoints return HTTP 200 with body.status="not_implemented".
# This is intentional for schema-validation / contract-testing.  Once real
# services are wired in, unimplemented paths SHOULD return HTTP 501 or a
# graceful fallback depending on the caller contract.


@app.post("/v1/rag/retrieve")
async def rag_retrieve(body: RAGRetrieveRequest, request: Request) -> UnifiedResponse:
    """Real RAG retrieval — uses RagService for exact/alias/BM25 lookup."""
    t0 = time.perf_counter()
    svc = get_rag_service()
    result = svc.retrieve(
        query=body.query,
        categories=body.categories if body.categories else None,
        primary_intent=body.primary_intent,
        sub_intent=body.sub_intent,
        top_k=body.top_k,
        attribute_context=body.attribute_context,
    )
    hits_data: list[dict] = []
    for h in result.hits:
        hits_data.append({
            "id": h.id, "category": h.category,
            "term": h.term, "zh_term": h.zh_term,
            "title": h.title, "content_snippet": h.content,
            "score": h.score, "match_type": h.match_type,
            "source": h.source, "source_type": h.source_type,
            "source_url": h.source_url,
            "source_ref": h.source_ref,
            "allowed_usage": h.allowed_usage,
            "risk_level": h.risk_level, "risk_note": h.risk_note,
            "review_status": h.review_status,
            "reviewed_by": h.reviewed_by,
            "last_reviewed_at": h.last_reviewed_at,
            "metadata": h.metadata,
        })
    data = RAGRetrieveData(
        query=result.query,
        normalized_query=result.normalized_query,
        hits=hits_data,
        meta=result.meta,
    )
    status = ResponseStatus.success
    # Merge service-level warnings into response-level warnings.
    svc_warnings = list(result.warnings)
    return _finish(
        request, t0,
        data=data.model_dump(),
        status=status,
        tools=["rag_service"],
        path="rag",
        warnings=list(svc_warnings),
    )


# ── /v1/merchant/content/generate ────────────────────────────────────────────────


@app.post("/v1/merchant/content/generate")
async def merchant_content(body: MerchantContentRequest, request: Request) -> UnifiedResponse:
    """Real deterministic content generation (P0b.1). Uses structured attributes
    to produce title / selling_points / short_description / detail_bullets."""
    t0 = time.perf_counter()
    svc = get_content_generation_service()
    result = svc.generate(
        content_type=body.content_type if hasattr(body, 'content_type') else "selling_points",
        attributes=body.attributes if body.attributes else None,
        garment_category=body.garment_category if hasattr(body, 'garment_category') else None,
        target_channel=None,
        tone=body.style if hasattr(body, 'style') else None,
        language=None,
        constraints=None,
        max_length=body.max_copy_length if hasattr(body, 'max_copy_length') else None,
        product_id=body.product_id,
    )
    data = dict(result.to_dict())
    return _finish(
        request, t0, data=data,
        status=ResponseStatus.success,
        tools=["content_generation_service", "deterministic_template"],
        path="content_generation",
        warnings=list(result.warnings),
    )
