"""
P0a.5 — QaOrchestrator: deterministic multi-modal QA dispatch skeleton.

Routes user queries through IntentClassifier → AttributeService / RagService
and returns a unified ``QAOrchestratorResult``.  No LLM, MLLM, visual pipeline,
Redis, FAISS, or BGE dependencies.

Used by ``/v1/mm/qa`` (replaces the P0a mock).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from inference.serving.schemas import WarningItem, WarningSeverity
from inference.serving.vision_context import build_vision_context

logger = logging.getLogger(__name__)

ORCHESTRATOR_VERSION = "0.1.0"

# ── P1.4a region confidence thresholds ────────────────────────────────────────

REGION_CONFIDENCE_THRESHOLD = 0.5       # >= this → reliable evidence
REGION_LOW_CONFIDENCE_THRESHOLD = 0.3   # >= this but < 0.5 → low-confidence, no definitive claim
# < REGION_LOW_CONFIDENCE_THRESHOLD → ignored as unreliable

# P1.4a: safe keys for localized_region summaries (no mask/path/crop/image_bytes).
_LOCALIZED_REGION_SAFE_KEYS = frozenset({
    "region_id", "part_type", "part_group", "bbox",
    "confidence", "source", "backend", "instance_id",  # P1.4e: +instance_id
})


def _build_result_meta(route: str, vc, **extra) -> Dict[str, Any]:
    """Build QAOrchestratorResult.meta from VisionContext + route."""
    if vc is None:
        return {"route": route, "orchestrator_version": ORCHESTRATOR_VERSION, **extra}
    # P1.1: whitelist passthrough of provider-level meta keys (backward
    # compatible — keys only appear when the provider sets them).
    vc_meta = getattr(vc, "meta", {}) or {}
    provider_meta = {k: vc_meta[k] for k in (
        "vision_backend", "vision_latency_ms", "unmapped_attribute_keys",
        "vision_provider_real_enabled", "error_code",
        "vision_backend_mode", "num_garment_instances", "mask_bitmap_returned",
    ) if k in vc_meta}
    return {
        "route": route,
        "attribute_source": getattr(vc, "attribute_source", "none"),
        "provided_attributes_used": getattr(vc, "provided_attributes_used", False),
        "visual_attributes_used": getattr(vc, "visual_attributes_used", False),
        "visual_attributes_present": getattr(vc, "visual_attributes_present", False),
        "vision_provider_used": getattr(vc, "vision_provider_used", False),
        "vision_provider_name": getattr(vc, "vision_provider_name", None),
        "vision_warning_count": len(getattr(vc, "warnings", [])),
        "vision_sources_count": len(getattr(vc, "sources", [])),
        "requested_regions": list(getattr(vc, "requested_regions", [])),
        "garment_instance_count": len(getattr(vc, "garment_instances", [])),
        "visual_region_count": len(getattr(vc, "regions", [])),
        "orchestrator_version": ORCHESTRATOR_VERSION,
        **provider_meta,
        **extra,
    }


# ── P1.3 garment instance helpers ──────────────────────────────────────────────

_GARMENT_INSTANCE_SAFE_KEYS = frozenset({
    "instance_id", "category", "fine_class_name", "bbox",
    "confidence", "mask_present",
})

# ponytail: Chinese display names for PRD 5 coarse categories.
_CATEGORY_CN: Dict[str, str] = {
    "top": "上衣", "pants": "裤子", "skirt": "裙子",
    "outerwear": "外套", "dress": "连衣裙",
}


def _summarize_garment_instances(
    instances: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return JSON-safe summaries of *instances* (safe keys only — no mask/map/path)."""
    return [
        {k: v for k, v in inst.items() if k in _GARMENT_INSTANCE_SAFE_KEYS}
        for inst in instances
    ]


def _build_garment_instance_sources(
    instances: List[Dict[str, Any]],
    provider_name: str = "",
) -> List[Dict[str, Any]]:
    """Convert garment instances to SourceItem-compatible dicts (no mask/path)."""
    sources: List[Dict[str, Any]] = []
    for inst in instances:
        entry: Dict[str, Any] = {
            "type": "garment_instance",
            "id": inst.get("instance_id"),
            "value": inst.get("category"),
            "attribute_confidence": inst.get("confidence"),
            "source": provider_name or "vision_provider",
            "metadata": {},
        }
        for key in ("fine_class_name", "bbox"):
            if key in inst:
                entry["metadata"][key] = inst[key]
        if inst.get("mask_present"):
            entry["metadata"]["mask_present"] = True
        sources.append(entry)
    return sources


# ── P1.4a region helpers ─────────────────────────────────────────────────────


def _summarize_localized_regions(
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return JSON-safe summaries of *regions* (safe keys only — no mask/path/crop)."""
    return [
        {k: v for k, v in r.items() if k in _LOCALIZED_REGION_SAFE_KEYS}
        for r in regions
    ]


def _build_localized_region_sources(
    regions: List[Dict[str, Any]],
    provider_name: str = "",
) -> List[Dict[str, Any]]:
    """Convert localized regions to SourceItem-compatible dicts (no mask/path)."""
    sources: List[Dict[str, Any]] = []
    for r in regions:
        sources.append({
            "type": "localized_region",
            "id": r.get("region_id"),
            "label": r.get("part_type"),
            "confidence": r.get("confidence"),
            "source": provider_name or "vision_provider",
            "metadata": {
                "part_group": r.get("part_group"),
                "bbox": r.get("bbox"),
                "backend": r.get("backend"),
            },
        })
    return sources


def _find_regions_by_part(
    regions: List[Dict[str, Any]],
    part_type: str,
) -> List[Dict[str, Any]]:
    """Return *regions* whose ``part_type`` matches (case-insensitive)."""
    if not part_type:
        return []
    target = part_type.lower().strip()
    return [r for r in regions if (r.get("part_type") or "").lower().strip() == target]


def _filter_reliable_regions(
    regions: List[Dict[str, Any]],
    threshold: float = REGION_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Return regions with confidence >= *threshold* (or confidence is None → excluded)."""
    return [
        r for r in regions
        if isinstance(r.get("confidence"), (int, float)) and r["confidence"] >= threshold
    ]


def _filter_low_confidence_regions(
    regions: List[Dict[str, Any]],
    low: float = REGION_LOW_CONFIDENCE_THRESHOLD,
    high: float = REGION_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Return regions with low <= confidence < high."""
    return [
        r for r in regions
        if isinstance(r.get("confidence"), (int, float))
        and low <= r["confidence"] < high
    ]


def _regions_with_missing_confidence(
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return regions whose confidence is None (missing)."""
    return [r for r in regions if r.get("confidence") is None]


# ── Result model ───────────────────────────────────────────────────────────────


@dataclass
class QAOrchestratorResult:
    query: str
    answer: str
    answer_type: str
    intent: Dict[str, Any] = field(default_factory=dict)
    answer_confidence: Optional[float] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    used_tools: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "intent": self.intent,
            "answer_confidence": self.answer_confidence,
            "sources": self.sources,
            "used_tools": self.used_tools,
            "meta": self.meta,
        }

    def to_qa_data(self) -> Dict[str, Any]:
        """Return a dict compatible with ``MultimodalQAData.model_dump()``."""
        return {
            "answer": self.answer,
            "answer_type": self.answer_type,
            "answer_confidence": self.answer_confidence,
            "intent_confidence": self.intent.get("confidence"),
            "sources": self.sources,
            "is_cached": False,
            "need_image": False,
            "clarification": None,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _attr_name_from_intent(intent: Dict[str, Any], query: str) -> Optional[str]:
    """Extract a plausible attribute name from intent data."""
    sub = intent.get("sub_intent")
    if sub:
        return sub
    entities = intent.get("entities", {})
    attr = entities.get("attribute_name")
    if attr:
        return attr
    # Fallback: try to infer from query keywords.
    for kw, attr_name in [("面料", "fabric"), ("材质", "fabric"), ("颜色", "color"),
                           ("领", "collar"), ("袖", "sleeve"), ("长度", "length"),
                           ("衣长", "length"), ("裤长", "length"), ("裙长", "length")]:
        if kw in query:
            return attr_name
    return "general"


def _knowledge_answer(hits: List[Any], warnings: List[WarningItem]) -> Dict[str, Any]:
    """Build a deterministic knowledge answer from RAG hits."""
    if not hits:
        warnings.append(WarningItem(
            code="no_hits", scope="qa",
            message="未检索到相关知识条目。",
            severity=WarningSeverity.info,
        ))
        return {
            "answer": "暂未检索到相关知识，建议补充更多信息或联系客服获取帮助。",
            "answer_type": "knowledge_answer",
            "answer_confidence": None,
        }

    top = hits[0]
    # RetrievalHit is a dataclass — use attribute access.
    content = getattr(top, "content", "") or ""
    title = getattr(top, "title", "") or ""
    review = getattr(top, "review_status", "") or ""
    answer = f"根据当前知识库，{title}：{content}"
    if review != "reviewed":
        answer += "（提示：该知识条目仍需人工审核，建议结合商品详情页或官方说明确认。）"

    sources = [{
        "type": "knowledge_base",
        "id": getattr(h, "id", ""),
        "title": getattr(h, "title", ""),
        "category": getattr(h, "category", ""),
        "rag_score": getattr(h, "score", None),
        "source": getattr(h, "source", ""),
        "source_type": getattr(h, "source_type", ""),
        "source_url": getattr(h, "source_url", None),
        "source_ref": getattr(h, "source_ref", {}),
        "review_status": getattr(h, "review_status", ""),
        "version": (getattr(h, "metadata", {}) or {}).get("version", ""),
    } for h in hits[:3]]

    return {
        "answer": answer,
        "answer_type": "knowledge_answer",
        "answer_confidence": None,
        "sources": sources,
    }


# ── QaOrchestrator ─────────────────────────────────────────────────────────────


class QaOrchestrator:
    """Deterministic QA dispatch skeleton.

    Routes: attribute_query → AttributeService
             knowledge_qa    → RagService
             craft_explanation / design_explanation → RagService
             styling_advice  → RagService (limited)
             content_generation → unsupported
             fallback_unknown / chat → unsupported

    Does NOT call any LLM, MLLM, visual pipeline, or external service.
    """

    def __init__(
        self,
        intent_classifier: Any,
        attribute_service: Any,
        rag_service: Any,
        vision_provider: Any = None,
    ) -> None:
        self._intent = intent_classifier
        self._attr = attribute_service
        self._rag = rag_service
        self._vision = vision_provider
        logger.info("QaOrchestrator v%s loaded (vision=%s)",
                     ORCHESTRATOR_VERSION,
                     "mock" if vision_provider is None else type(vision_provider).__name__)

    # ── Public API ─────────────────────────────────────────────────────────

    def answer(
        self,
        *,
        query: Optional[str],
        image: Any = None,
        image_url: Optional[str] = None,
        image_bytes: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        garment_category: Optional[str] = None,
        regions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> QAOrchestratorResult:
        warnings: List[WarningItem] = []
        used_tools: List[str] = []
        attrs = dict(attributes) if attributes else {}
        has_attrs = bool(attrs)

        # Vision context: merge provided + visual attributes (P0a.7 single-call).
        vc = build_vision_context(
            vision_provider=self._vision,
            query=query,
            image=image, image_url=image_url, image_bytes=image_bytes,
            provided_attributes=attrs,
            garment_category=garment_category,
            requested_regions=regions,
            context=context,
        )
        effective_attrs = vc.effective_attributes
        vision_warnings = list(vc.warnings)
        if vc.vision_provider_used:
            used_tools.extend(["vision_provider_mock" if vc.vision_provider_name == "MockVisionAttributeProvider" else "vision_provider"])

        # C1: Empty query.
        q = (query or "").strip()
        if not q:
            return QAOrchestratorResult(
                query=query or "", answer="请先输入想了解的问题。",
                answer_type="empty_query",
                warnings=[WarningItem(
                    code="empty_query", scope="qa",
                    message="Query is empty.", severity=WarningSeverity.info,
                )],
                meta=_build_result_meta("empty_query", vc),
            )

        # Classify intent.
        try:
            intent_data = self._intent.classify(q)
        except Exception:
            logger.exception("Intent classification failed")
            intent_data = None

        if intent_data is None:
            return QAOrchestratorResult(
                query=q, answer="当前问题类型暂不支持。",
                answer_type="unsupported",
                warnings=[WarningItem(
                    code="orchestrator_fallback", scope="qa",
                    message="Intent classifier returned null.", severity=WarningSeverity.error,
                )],
                meta={"route": "error", "orchestrator_version": ORCHESTRATOR_VERSION},
            )

        primary = getattr(intent_data, "primary_intent", "fallback_unknown")
        sub = getattr(intent_data, "sub_intent", None)
        conf = getattr(intent_data, "intent_confidence", 0.0)
        entities = getattr(intent_data, "entities", {}) or {}
        intent_dict = {
            "primary_intent": primary, "sub_intent": sub,
            "confidence": conf, "classifier_level": getattr(intent_data, "classifier_level", "rule"),
        }
        used_tools.append("intent_classifier")

        # Low confidence fallback.
        if primary != "fallback_unknown" and conf < 0.3:
            warnings.append(WarningItem(
                code="low_intent_confidence", scope="qa",
                message=f"Intent confidence low ({conf:.2f}).",
                severity=WarningSeverity.warn,
            ))

        warnings.extend(vision_warnings)

        # ── Route ──────────────────────────────────────────────────────────
        if primary == "attribute_query":
            return self._route_attribute(q, effective_attrs, garment_category, intent_dict, warnings, used_tools, vc)
        elif primary in ("knowledge_qa", "design_explanation", "craft_explanation"):
            return self._route_knowledge(q, effective_attrs, primary, sub, warnings, used_tools,
                                         garment_category=garment_category, meta=vc)
        elif primary == "styling_advice":
            return self._route_styling(q, effective_attrs, primary, sub, warnings, used_tools, vc)
        elif primary == "visual_instance_query":
            return self._route_visual_instance(q, primary, sub, entities, warnings, used_tools, vc)
        elif primary.startswith("region_"):
            return self._route_region_query(q, primary, sub, entities, warnings, used_tools, vc,
                                           image_bytes=image_bytes)
        elif primary in ("content_generation",):
            return self._route_unsupported(q, warnings, used_tools,
                                           "当前版本暂未开放商品文案生成。")
        elif primary == "chat":
            return QAOrchestratorResult(
                query=q, answer="你好！我是服饰知识助手，可以帮你查询面料、工艺、搭配等信息。请直接告诉我你想了解的内容。",
                answer_type="unsupported", intent=intent_dict,
                meta={"route": "chat", "orchestrator_version": ORCHESTRATOR_VERSION},
            )
        else:
            return self._route_unsupported(q, warnings, used_tools,
                                           "当前问题类型暂不支持。")

    # ── Route handlers ─────────────────────────────────────────────────────

    def _route_attribute(
        self, q: str, attrs: Dict[str, Any], garment_cat: Optional[str],
        intent: Dict, warnings: List[WarningItem], tools: List[str],
        vc=None,
    ) -> QAOrchestratorResult:
        attr_name = _attr_name_from_intent(intent, q)
        tools.append("attribute_service")
        result = self._attr.answer_attribute(
            attr_name, attrs, garment_category=garment_cat,
        )
        for w in result.warnings:
            warnings.append(w)
        all_sources = [s.model_dump() for s in result.sources]
        if vc is not None and vc.sources:
            all_sources.extend(vc.sources)
        return QAOrchestratorResult(
            query=q, answer=result.answer, answer_type="attribute_answer",
            intent=intent, answer_confidence=result.answer_confidence,
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("attribute_query", vc, attr_name=attr_name,
                                     garment_category=garment_cat or ""),
        )

    def _route_knowledge(
        self, q: str, attrs: Dict[str, Any], primary: str, sub: Optional[str],
        warnings: List[WarningItem], tools: List[str],
        garment_category: Optional[str] = None,
        meta: Any = None,
    ) -> QAOrchestratorResult:
        tools.append("rag_service")
        ctx = dict(attrs) if attrs else {}
        if garment_category:
            ctx.setdefault("garment_category", garment_category)
        r = self._rag.retrieve(
            query=q, primary_intent=primary, sub_intent=sub,
            top_k=3, attribute_context=ctx if ctx else None,
        )
        for w in r.warnings:
            if w.code != "no_hits":
                warnings.append(w)
        ka = _knowledge_answer(r.hits, warnings)
        tools.append("template_answer")
        all_sources = list(ka.get("sources", []))
        if meta is not None and hasattr(meta, "sources") and meta.sources:
            all_sources.extend(meta.sources)
        return QAOrchestratorResult(
            query=q, answer=ka["answer"], answer_type=ka["answer_type"],
            intent={"primary_intent": primary, "sub_intent": sub, "confidence": 0.0},
            answer_confidence=ka["answer_confidence"],
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("knowledge_qa", meta, rag_hit_count=len(r.hits)),
        )

    # ── P1.3 Visual instance query ───────────────────────────────────────────

    def _route_visual_instance(
        self, q: str, primary: str, sub: Optional[str],
        entities: Dict[str, Any],
        warnings: List[WarningItem], tools: List[str],
        vc: Optional[Any] = None,
    ) -> QAOrchestratorResult:
        """Answer garment-instance questions from 3.1.1 segmentation output."""
        tools.append("visual_instance_answer")
        instances = list(getattr(vc, "garment_instances", []) or [])
        intent_dict = {"primary_intent": primary, "sub_intent": sub, "confidence": 1.0}

        if not instances:
            warnings.append(WarningItem(
                code="vision_instances_unavailable", scope="qa",
                message="当前没有可用的服饰实例检测结果。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q, answer="当前没有可用的服饰实例检测结果。",
                answer_type="visual_instance_answer", intent=intent_dict,
                warnings=warnings, used_tools=tools,
                meta=_build_result_meta("visual_instance_query", vc,
                                        primary_intent=primary, sub_intent=sub,
                                        garment_instances_summary=[]),
            )

        summary = _summarize_garment_instances(instances)
        inst_sources = _build_garment_instance_sources(
            instances, provider_name=getattr(vc, "vision_provider_name", "") or "",
        )
        all_sources = list(inst_sources)
        if vc is not None and getattr(vc, "sources", None):
            all_sources.extend(vc.sources)

        garment_ref = entities.get("garment_ref") or ""
        answer: str
        answer_confidence: Optional[float] = None

        if sub == "count":
            cat_counts: Dict[str, int] = {}
            for inst in instances:
                cat = inst.get("category", "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            parts = [f"{_CATEGORY_CN.get(c, c)}{n}件" for c, n in cat_counts.items()]
            answer = "检测到{}件服饰：{}。".format(len(instances), "、".join(parts))
            answer_confidence = None

        elif sub == "detection":
            lines = []
            for inst in instances:
                cn = _CATEGORY_CN.get(inst.get("category", ""), inst.get("category", ""))
                fine = inst.get("fine_class_name", "")
                conf = inst.get("confidence")
                if isinstance(conf, (int, float)):
                    lines.append(f"{cn}（{fine}，置信度{conf:.2f}）")
                else:
                    lines.append(f"{cn}（{fine}）")
            answer = "检测到{}件服饰：{}。".format(len(instances), "；".join(lines))
            answer_confidence = None

        elif sub == "existence":
            # ponytail: entity-extracted refs already map to PDF 5 classes.
            matches = [
                inst for inst in instances
                if inst.get("category") == garment_ref
            ]
            cn = _CATEGORY_CN.get(garment_ref, garment_ref)
            if matches:
                answer = "检测到{}（检测到{}件）。".format(cn, len(matches))
            else:
                answer = "未检测到{}，当前检测结果为：{}。".format(
                    cn if garment_ref else "匹配的服饰",
                    "、".join(
                        _CATEGORY_CN.get(inst.get("category", ""), inst.get("category", ""))
                        for inst in instances
                    ),
                )
            answer_confidence = None

        elif sub == "location":
            target = instances[0]
            if garment_ref:
                matches = [inst for inst in instances if inst.get("category") == garment_ref]
                if matches:
                    target = matches[0]
            bbox = target.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                answer = "检测框：[x1={:.0f}, y1={:.0f}, x2={:.0f}, y2={:.0f}]".format(*bbox)
            else:
                answer = "未找到{}的检测框。".format(
                    _CATEGORY_CN.get(garment_ref, garment_ref or "该服饰"),
                )
            answer_confidence = target.get("confidence")

        elif sub == "segmentation":
            masked = [inst for inst in instances if inst.get("mask_present")]
            n = len(masked)
            if n > 0:
                answer = "检测到{}件服饰有分割掩码结果（共{}件实例）。分割掩码以mask_ref形式存在，不会返回掩码位图。".format(n, len(instances))
            else:
                answer = "当前{}件服饰实例中没有分割掩码结果。".format(len(instances))
            answer_confidence = None

        else:
            # Unknown sub_intent → detection summary.
            lines = []
            for inst in instances:
                cn = _CATEGORY_CN.get(inst.get("category", ""), inst.get("category", ""))
                fine = inst.get("fine_class_name", "")
                lines.append(f"{cn}（{fine}）")
            answer = "检测到{}件服饰：{}。".format(len(instances), "；".join(lines))
            answer_confidence = None

        return QAOrchestratorResult(
            query=q, answer=answer, answer_type="visual_instance_answer",
            intent=intent_dict, answer_confidence=answer_confidence,
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("visual_instance_query", vc,
                                    primary_intent=primary, sub_intent=sub,
                                    garment_instances_summary=summary),
        )

    # ── P1.4b Region backend helper ─────────────────────────────────────────

    @staticmethod
    def _try_region_backend(
        image_bytes: Any,
        query: str,
        warnings: List[WarningItem],
        tools: List[str],
        *,
        query_all_parts: bool = False,
        garment_instances: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Attempt to call the real 3.1.2 region backend.

        Returns a list of ``LocalizedRegion``-compatible dicts on success,
        or an empty list on failure / disabled / unavailable.
        Warnings are appended for error conditions.

        When *query_all_parts* is True (detail queries), all Fashionpedia parts
        are queried instead of extracting a single part from the query.

        P1.4e: *garment_instances* from 3.1.1 are passed through to the backend
        for instance-aware region localization.
        """
        from inference.serving.region_backend import (
            get_region_backend,
            decode_image_bytes,
        )

        backend = get_region_backend()
        if not backend.enabled:
            warnings.append(WarningItem(
                code="region_backend_disabled", scope="vision",
                message="3.1.2 局部区域后端未启用。",
                severity=WarningSeverity.info,
            ))
            return []

        tools.append("region_backend")

        img = decode_image_bytes(image_bytes)
        if img is None:
            warnings.append(WarningItem(
                code="region_backend_error", scope="vision",
                message="3.1.2 图片解码失败，无法执行局部区域检测。",
                severity=WarningSeverity.warn,
            ))
            return []

        # Extract requested part from query for targeted detection.
        requested_part: Optional[str] = None
        if not query_all_parts:
            from inference.serving.region_query_mapper import extract_requested_region_part
            requested_part = extract_requested_region_part(query)

        try:
            regions = backend.locate_regions(
                image=img, query=query, requested_part=requested_part,
                garment_instances=garment_instances,
            )
        except Exception:
            logger.exception("Region backend call failed")
            warnings.append(WarningItem(
                code="region_backend_error", scope="vision",
                message="3.1.2 局部区域检测执行失败。",
                severity=WarningSeverity.warn,
            ))
            return []

        if not regions:
            warnings.append(WarningItem(
                code="region_backend_empty", scope="vision",
                message="3.1.2 后端未检测到任何局部区域。",
                severity=WarningSeverity.info,
            ))

        return regions

    # ── P1.4a Region query routes ────────────────────────────────────────────

    def _route_region_query(
        self, q: str, primary: str, sub: Optional[str],
        entities: Dict[str, Any],
        warnings: List[WarningItem], tools: List[str],
        vc: Optional[Any] = None,
        image_bytes: Any = None,
    ) -> QAOrchestratorResult:
        """Answer local-region questions from 3.1.2 localized_regions.

        P1.4b: when *vc.localized_regions* is empty and a real region backend
        is available (enabled), the backend is called with the decoded image.
        """
        tools.append("region_query_answer")
        regions = list(getattr(vc, "localized_regions", []) or [])
        intent_dict = {"primary_intent": primary, "sub_intent": sub, "confidence": 1.0}

        # P1.4b: try real region backend when no pre-computed regions exist.
        backend_was_called = False
        if not regions and image_bytes is not None:
            regions = self._try_region_backend(
                image_bytes, q, warnings, tools,
                query_all_parts=(primary == "region_detail_query"),
                garment_instances=getattr(vc, "garment_instances", None),
            )
            backend_was_called = "region_backend" in tools

        # No localized regions at all AND backend was NOT called →
        # no evidence source is available (not even mock).
        # When backend WAS called but returned empty, let intent-specific
        # routing handle it (e.g. "未检测到pocket区域").
        if not regions and not backend_was_called:
            warnings.append(WarningItem(
                code="localized_regions_unavailable", scope="qa",
                message="当前没有可用的局部区域检测结果。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q, answer="当前没有可用的局部区域检测结果（3.1.2 未接入或未返回结果）。",
                answer_type="region_query_answer", intent=intent_dict,
                warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent=primary,
                                        localized_regions_summary=[],
                                        regions_used=[]),
            )

        # Extract requested part from query.
        from inference.serving.region_query_mapper import extract_requested_region_part
        requested_part = extract_requested_region_part(q)

        # Common region data for meta.
        summary = _summarize_localized_regions(regions)

        # ── Route by primary intent ──────────────────────────────────────

        if primary == "region_detail_query":
            return self._answer_region_detail(q, regions, warnings, tools, vc,
                                              intent_dict, summary)

        if primary == "region_count_query":
            return self._answer_region_count(
                q, regions, requested_part, warnings, tools, vc, intent_dict, summary)

        if primary == "region_existence_query":
            return self._answer_region_existence(
                q, regions, requested_part, warnings, tools, vc, intent_dict, summary)

        if primary == "region_location_query":
            return self._answer_region_location(
                q, regions, requested_part, warnings, tools, vc, intent_dict, summary)

        if primary == "region_attribute_query":
            return self._answer_region_attribute(
                q, regions, requested_part, warnings, tools, vc, intent_dict, summary,
                image_bytes=image_bytes)

        # Unknown region sub-type — fallback to summary.
        return QAOrchestratorResult(
            query=q, answer="当前暂不支持此类型的区域查询。",
            answer_type="region_query_answer", intent=intent_dict,
            warnings=warnings, used_tools=tools,
            meta=_build_result_meta("region_query", vc,
                                    primary_intent=primary,
                                    localized_regions_summary=summary,
                                    regions_used=[]),
        )

    # -- region_detail_query ---------------------------------------------------

    def _answer_region_detail(
        self, q: str, regions: List[Dict], warnings: List[WarningItem],
        tools: List[str], vc, intent_dict: Dict, summary: List[Dict],
    ) -> QAOrchestratorResult:
        reliable = _filter_reliable_regions(regions)
        low_conf = _filter_low_confidence_regions(regions)
        missing_conf = _regions_with_missing_confidence(regions)

        region_sources = _build_localized_region_sources(
            reliable, provider_name=getattr(vc, "vision_provider_name", "") or "",
        )
        all_sources = list(region_sources)
        if vc is not None and getattr(vc, "sources", None):
            all_sources.extend(vc.sources)

        if low_conf:
            warnings.append(WarningItem(
                code="region_low_confidence", scope="qa",
                message=f"有{len(low_conf)}个局部区域置信度较低，未作为可靠结果返回。",
                severity=WarningSeverity.info,
            ))
        if missing_conf:
            warnings.append(WarningItem(
                code="region_confidence_missing", scope="qa",
                message=f"有{len(missing_conf)}个局部区域缺少置信度信息。",
                severity=WarningSeverity.info,
            ))

        if not reliable:
            warnings.append(WarningItem(
                code="region_not_found", scope="qa",
                message="未检测到可靠的局部细节。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q, answer="当前没有可靠检测到局部细节（如领口、口袋、拉链等）。",
                answer_type="region_query_answer", intent=intent_dict,
                warnings=warnings, used_tools=tools,
                sources=all_sources,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_detail_query",
                                        localized_regions_summary=summary,
                                        regions_used=[]),
            )

        cn_parts = [r.get("part_type", "unknown") for r in reliable]
        # ponytail: simple CN display mapping for common parts.
        _CN_DISPLAY = {
            "neckline": "领口", "collar": "领子", "lapel": "翻领",
            "sleeve": "袖子", "cuff": "袖口", "hem": "下摆",
            "pocket": "口袋", "shoulder": "肩部", "waist": "腰部",
            "zipper": "拉链", "hood": "帽子", "button": "扣子",
            "buckle": "扣环", "bow": "蝴蝶结", "ribbon": "丝带",
            "ruffle": "褶边", "tassel": "流苏", "sequin": "亮片",
            "bead": "珠子", "applique": "贴花", "flower": "花朵装饰",
            "rivet": "铆钉", "pattern": "图案", "decoration": "装饰",
            "strap": "肩带", "epaulette": "肩章",
        }
        parts_cn = [_CN_DISPLAY.get(p, p) for p in cn_parts]
        regions_used = [r.get("region_id") for r in reliable]

        return QAOrchestratorResult(
            query=q,
            answer=f"当前检测到的局部细节包括：{'、'.join(parts_cn)}。",
            answer_type="region_query_answer", intent=intent_dict,
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("region_query", vc,
                                    primary_intent="region_detail_query",
                                    localized_regions_summary=summary,
                                    regions_used=regions_used),
        )

    # -- region_count_query ----------------------------------------------------

    def _answer_region_count(
        self, q: str, regions: List[Dict], requested_part: Optional[str],
        warnings: List[WarningItem], tools: List[str], vc,
        intent_dict: Dict, summary: List[Dict],
    ) -> QAOrchestratorResult:
        if not requested_part:
            warnings.append(WarningItem(
                code="region_not_found", scope="qa",
                message="未能从查询中识别出要计数的局部区域类型。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q, answer="未能从查询中识别出要计数的局部区域类型。",
                answer_type="region_query_answer", intent=intent_dict,
                warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_count_query",
                                        localized_regions_summary=summary,
                                        regions_used=[]),
            )

        matching = _find_regions_by_part(regions, requested_part)
        reliable = _filter_reliable_regions(matching)
        low_conf = _filter_low_confidence_regions(matching)

        if low_conf:
            warnings.append(WarningItem(
                code="region_low_confidence", scope="qa",
                message=f"部分{requested_part}区域置信度较低。",
                severity=WarningSeverity.info,
            ))

        region_sources = _build_localized_region_sources(
            reliable, provider_name=getattr(vc, "vision_provider_name", "") or "",
        )
        all_sources = list(region_sources)
        if vc is not None and getattr(vc, "sources", None):
            all_sources.extend(vc.sources)

        count = len(reliable)
        regions_used = [r.get("region_id") for r in reliable]

        if count > 0:
            return QAOrchestratorResult(
                query=q,
                answer=f"当前可靠检测到 {count} 个{requested_part}区域。",
                answer_type="region_query_answer", intent=intent_dict,
                sources=all_sources, warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_count_query",
                                        localized_regions_summary=summary,
                                        regions_used=regions_used),
            )

        return QAOrchestratorResult(
            query=q,
            answer=f"当前没有可靠检测到{requested_part}区域，不能确认具体数量。",
            answer_type="region_query_answer", intent=intent_dict,
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("region_query", vc,
                                    primary_intent="region_count_query",
                                    localized_regions_summary=summary,
                                    regions_used=[]),
        )

    # -- region_existence_query ------------------------------------------------

    def _answer_region_existence(
        self, q: str, regions: List[Dict], requested_part: Optional[str],
        warnings: List[WarningItem], tools: List[str], vc,
        intent_dict: Dict, summary: List[Dict],
    ) -> QAOrchestratorResult:
        if not requested_part:
            warnings.append(WarningItem(
                code="region_not_found", scope="qa",
                message="未能从查询中识别出要查询的局部区域类型。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q, answer="未能从查询中识别出要查询的局部区域类型。",
                answer_type="region_query_answer", intent=intent_dict,
                warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_existence_query",
                                        localized_regions_summary=summary,
                                        regions_used=[]),
            )

        matching = _find_regions_by_part(regions, requested_part)
        reliable = _filter_reliable_regions(matching)
        low_conf = _filter_low_confidence_regions(matching)

        region_sources = _build_localized_region_sources(
            reliable, provider_name=getattr(vc, "vision_provider_name", "") or "",
        )
        all_sources = list(region_sources)
        if vc is not None and getattr(vc, "sources", None):
            all_sources.extend(vc.sources)

        if reliable:
            conf_strs = [
                f"置信度 {r['confidence']:.2f}" for r in reliable
                if isinstance(r.get("confidence"), (int, float))
            ]
            conf_note = f"，{'、'.join(conf_strs)}" if conf_strs else ""
            return QAOrchestratorResult(
                query=q,
                answer=f"当前检测到 {len(reliable)} 个{requested_part}区域{conf_note}。",
                answer_type="region_query_answer", intent=intent_dict,
                sources=all_sources, warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_existence_query",
                                        localized_regions_summary=summary,
                                        regions_used=[r.get("region_id") for r in reliable]),
            )

        if low_conf:
            warnings.append(WarningItem(
                code="region_low_confidence", scope="qa",
                message=f"检测到可能的{requested_part}区域但置信度较低，不能作为确定性结论。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q,
                answer=f"检测到可能的{requested_part}区域，但置信度较低，不能确认是否存在。",
                answer_type="region_query_answer", intent=intent_dict,
                sources=all_sources, warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_existence_query",
                                        localized_regions_summary=summary,
                                        regions_used=[]),
            )

        return QAOrchestratorResult(
            query=q,
            answer=f"当前没有可靠检测到{requested_part}区域，不能确认是否存在。",
            answer_type="region_query_answer", intent=intent_dict,
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("region_query", vc,
                                    primary_intent="region_existence_query",
                                    localized_regions_summary=summary,
                                    regions_used=[]),
        )

    # -- region_location_query -------------------------------------------------

    def _answer_region_location(
        self, q: str, regions: List[Dict], requested_part: Optional[str],
        warnings: List[WarningItem], tools: List[str], vc,
        intent_dict: Dict, summary: List[Dict],
    ) -> QAOrchestratorResult:
        if not requested_part:
            warnings.append(WarningItem(
                code="region_not_found", scope="qa",
                message="未能从查询中识别出要定位的局部区域类型。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q, answer="未能从查询中识别出要定位的局部区域类型。",
                answer_type="region_query_answer", intent=intent_dict,
                warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_location_query",
                                        localized_regions_summary=summary,
                                        regions_used=[]),
            )

        matching = _find_regions_by_part(regions, requested_part)
        reliable = _filter_reliable_regions(matching)
        low_conf = _filter_low_confidence_regions(matching)

        region_sources = _build_localized_region_sources(
            reliable, provider_name=getattr(vc, "vision_provider_name", "") or "",
        )
        all_sources = list(region_sources)
        if vc is not None and getattr(vc, "sources", None):
            all_sources.extend(vc.sources)

        if reliable:
            r = reliable[0]
            bbox = r.get("bbox")
            conf = r.get("confidence")
            bbox_str = f"bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]" if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else "位置未知"
            conf_str = f"，置信度 {conf:.2f}" if isinstance(conf, (int, float)) else ""
            return QAOrchestratorResult(
                query=q,
                answer=f"检测到{requested_part}区域，位置约为 {bbox_str}{conf_str}。",
                answer_type="region_query_answer", intent=intent_dict,
                sources=all_sources, warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_location_query",
                                        localized_regions_summary=summary,
                                        regions_used=[r.get("region_id")]),
            )

        if low_conf:
            warnings.append(WarningItem(
                code="region_low_confidence", scope="qa",
                message=f"检测到可能的{requested_part}区域但置信度较低。",
                severity=WarningSeverity.info,
            ))
            return QAOrchestratorResult(
                query=q,
                answer=f"检测到可能的{requested_part}区域，但置信度较低，无法给出确定位置。",
                answer_type="region_query_answer", intent=intent_dict,
                sources=all_sources, warnings=warnings, used_tools=tools,
                meta=_build_result_meta("region_query", vc,
                                        primary_intent="region_location_query",
                                        localized_regions_summary=summary,
                                        regions_used=[]),
            )

        return QAOrchestratorResult(
            query=q,
            answer=f"当前没有可靠定位到该局部区域（{requested_part}）。",
            answer_type="region_query_answer", intent=intent_dict,
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("region_query", vc,
                                    primary_intent="region_location_query",
                                    localized_regions_summary=summary,
                                    regions_used=[]),
        )

    # -- region_attribute_query ------------------------------------------------

    def _answer_region_attribute(
        self, q: str, regions: List[Dict], requested_part: Optional[str],
        warnings: List[WarningItem], tools: List[str], vc,
        intent_dict: Dict, summary: List[Dict],
        image_bytes: Any = None,
    ) -> QAOrchestratorResult:
        region_sources: List[Dict] = []
        regions_used: List[str] = []
        found_part = None
        found_bbox: Optional[List[float]] = None
        found_region_id: Optional[str] = None

        if requested_part:
            matching = _find_regions_by_part(regions, requested_part)
            reliable = _filter_reliable_regions(matching)
            if reliable:
                found_part = requested_part
                found_bbox = reliable[0].get("bbox")
                found_region_id = reliable[0].get("region_id")
                regions_used = [r.get("region_id") for r in reliable]
                region_sources = _build_localized_region_sources(
                    reliable, provider_name=getattr(vc, "vision_provider_name", "") or "",
                )

        # P1.4f: try 3.1.3 attribute backend.
        attr_results: List[Dict[str, Any]] = []
        if found_part and found_bbox is not None and image_bytes is not None:
            attr_results = self._try_attribute_backend(
                image_bytes, found_part, found_bbox,
                region_id=found_region_id,
                instance_id=(regions[0].get("instance_id") if regions else None),
                warnings=warnings, tools=tools,
            )

        all_sources = list(region_sources)
        if vc is not None and getattr(vc, "sources", None):
            all_sources.extend(vc.sources)
        # Add attribute results as sources.
        for ar in attr_results:
            all_sources.append({
                "type": "visual_attribute",
                "id": ar.get("region_id", ""),
                "label": ar.get("value"),
                "confidence": ar.get("attribute_confidence"),
                "source": "fashion_vision_3_1_3",
                "metadata": {"task": ar.get("task"), "topk": ar.get("topk", [])},
            })

        # Build answer.
        if found_part and attr_results:
            parts: List[str] = []
            for ar in attr_results:
                val = ar.get("value", "?")
                conf = ar.get("attribute_confidence")
                task = ar.get("task", "")
                if conf is not None:
                    parts.append(f"{task}={val}({conf:.2f})")
                else:
                    parts.append(f"{task}={val}")
            answer = f"检测到{found_part}区域，属性识别结果：{'；'.join(parts)}。"
            meta_extra = {"attribute_results": attr_results}
        elif found_part:
            answer = f"检测到{found_part}区域，但细粒度属性识别（3.1.3）暂未启用或模型不可用。"
            meta_extra = {}
        else:
            answer = f"未检测到对应局部区域，且细粒度属性识别（3.1.3）尚未接入，无法回答此问题。"
            meta_extra = {}

        return QAOrchestratorResult(
            query=q, answer=answer, answer_type="region_query_answer",
            intent=intent_dict,
            sources=all_sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("region_query", vc,
                                    primary_intent="region_attribute_query",
                                    localized_regions_summary=summary,
                                    regions_used=regions_used,
                                    **meta_extra),
        )

    # -- P1.4f: attribute backend helper ----------------------------------------

    @staticmethod
    def _try_attribute_backend(
        image_bytes: Any,
        part_type: str,
        bbox: List[float],
        region_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        warnings: Optional[List[WarningItem]] = None,
        tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Call 3.1.3 attribute classifiers for *part_type* at *bbox*."""
        from inference.serving.attribute_backend import get_attribute_backend
        from inference.serving.region_backend import decode_image_bytes

        backend = get_attribute_backend()
        if not backend.enabled:
            if warnings is not None:
                warnings.append(WarningItem(
                    code="attribute_backend_disabled", scope="vision",
                    message="3.1.3 细粒度属性识别未启用。",
                    severity=WarningSeverity.info,
                ))
            return []

        if tools is not None:
            tools.append("attribute_backend")

        img = decode_image_bytes(image_bytes)
        if img is None:
            return []

        # Map part_type to relevant task names.
        from inference.serving.attribute_backend import _PART_TO_ATTR_TASKS
        task_names = _PART_TO_ATTR_TASKS.get(part_type)

        try:
            return backend.extract_attributes(
                image=img, region_bbox=bbox, task_names=task_names,
                region_id=region_id, instance_id=instance_id,
            )
        except Exception:
            logger.exception("Attribute backend call failed")
            if warnings is not None:
                warnings.append(WarningItem(
                    code="attribute_backend_error", scope="vision",
                    message="3.1.3 属性识别执行失败。",
                    severity=WarningSeverity.warn,
                ))
            return []

    def _route_styling(
        self, q: str, attrs: Dict[str, Any], primary: str, sub: Optional[str],
        warnings: List[WarningItem], tools: List[str],
        vc=None,
    ) -> QAOrchestratorResult:
        tools.append("rag_service")
        r = self._rag.retrieve(
            query=q, primary_intent="styling_advice", sub_intent=sub,
            top_k=3, attribute_context=attrs if attrs else None,
        )
        for w in r.warnings:
            if w.code != "no_hits":
                warnings.append(w)
        if r.hits:
            answer = ("当前版本只能提供基础风格知识检索，暂不生成完整穿搭方案。"
                      "以下为相关参考信息：\n" +
                      getattr(r.hits[0], "content", ""))
            sources = [{
                "type": "knowledge_base",
                "id": getattr(h, "id", ""),
                "title": getattr(h, "title", ""),
                "rag_score": getattr(h, "score", None),
            } for h in r.hits[:2]]
        else:
            answer = "当前版本只能提供基础风格知识检索，暂不生成完整穿搭方案。"
            sources = []
            warnings.append(WarningItem(
                code="no_hits", scope="qa",
                message="未检索到相关风格知识。",
                severity=WarningSeverity.info,
            ))
        tools.append("template_answer")
        return QAOrchestratorResult(
            query=q, answer=answer, answer_type="hybrid_answer",
            intent={"primary_intent": primary, "sub_intent": sub, "confidence": 0.0},
            answer_confidence=None, sources=sources, warnings=warnings, used_tools=tools,
            meta=_build_result_meta("styling_advice", vc, rag_hit_count=len(r.hits)),
        )

    def _route_unsupported(
        self, q: str, warnings: List[WarningItem], tools: List[str], msg: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> QAOrchestratorResult:
        warnings.append(WarningItem(
            code="unsupported_intent", scope="qa",
            message=msg, severity=WarningSeverity.info,
        ))
        return QAOrchestratorResult(
            query=q, answer=msg, answer_type="unsupported",
            warnings=warnings, used_tools=tools,
            meta={**(extra_meta or {}), "route": "unsupported"},
        )


# ── Singleton ──────────────────────────────────────────────────────────────────

_orchestrator: Optional[QaOrchestrator] = None


def get_qa_orchestrator() -> QaOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        from inference.serving.intent_classifier import get_classifier
        from inference.serving.attribute_service import get_attribute_service
        from inference.serving.rag_service import get_rag_service
        from inference.serving.vision_provider import get_vision_provider
        _orchestrator = QaOrchestrator(
            intent_classifier=get_classifier(),
            attribute_service=get_attribute_service(),
            rag_service=get_rag_service(),
            vision_provider=get_vision_provider(),
        )
    return _orchestrator
