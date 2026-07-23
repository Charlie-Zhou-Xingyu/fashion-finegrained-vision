"""
P0a.7 — VisionContext: structured merge of request-provided and vision-derived
attributes for the QA pipeline.

Provides ``VisionContext`` (dataclass) and ``build_vision_context`` (pure
function) so the visual-attribute merging logic is testable in isolation
and does NOT bloat ``QaOrchestrator``.

Does NOT call any real vision model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference.serving.schemas import WarningItem, WarningSeverity


# ── Source builder ─────────────────────────────────────────────────────────────


def build_visual_attribute_sources(
    attributes: Optional[Dict[str, Any]],
    provider_name: str = "vision_provider",
) -> List[Dict[str, Any]]:
    """Convert a dictionary of visual attributes into a list of JSON-safe
    ``SourceItem``-compatible dicts.

    Supports two input formats:

    Primitive:
        ``{"fabric": "棉"}`` → ``{type, field, value, source, provider, attribute_confidence: null}``

    Dict:
        ``{"fabric": {"value": "棉", "attribute_confidence": 0.82, "region_ref": "r1"}}``
        → full source with confidence / region_ref / instance_ref.

    The output NEVER contains image_bytes or bbox/mask data.
    """
    if not attributes:
        return []
    sources: List[Dict[str, Any]] = []
    for key, val in attributes.items():
        if isinstance(val, dict):
            sources.append({
                "type": "product_attribute",
                "field": key,
                "value": val.get("value"),
                "attribute_confidence": val.get("attribute_confidence"),
                "source": val.get("source", provider_name),
                "provider": provider_name,
                "region_ref": val.get("region_ref"),
                "instance_ref": val.get("instance_ref"),
            })
        else:
            sources.append({
                "type": "product_attribute",
                "field": key,
                "value": val,
                "attribute_confidence": None,
                "source": provider_name,
                "provider": provider_name,
                "region_ref": None,
                "instance_ref": None,
            })
    return sources


# ── VisionContext ──────────────────────────────────────────────────────────────


@dataclass
class VisionContext:
    """Structured merge result from ``build_vision_context``."""

    effective_attributes: Dict[str, Any] = field(default_factory=dict)
    attribute_source: str = "none"          # "request" | "vision" | "none"
    provided_attributes_used: bool = False
    visual_attributes_used: bool = False
    visual_attributes_present: bool = False
    vision_provider_used: bool = False
    vision_provider_name: Optional[str] = None
    garment_instances: List[Dict[str, Any]] = field(default_factory=list)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    localized_regions: List[Dict[str, Any]] = field(default_factory=list)  # P1.4a: 3.1.2 local regions
    requested_regions: List[str] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_attributes": dict(self.effective_attributes),
            "attribute_source": self.attribute_source,
            "provided_attributes_used": self.provided_attributes_used,
            "visual_attributes_used": self.visual_attributes_used,
            "visual_attributes_present": self.visual_attributes_present,
            "vision_provider_used": self.vision_provider_used,
            "vision_provider_name": self.vision_provider_name,
            "garment_instances": list(self.garment_instances),
            "regions": list(self.regions),
            "localized_regions": list(self.localized_regions),
            "requested_regions": list(self.requested_regions),
            "warnings": [w.model_dump() for w in self.warnings],
            "sources": list(self.sources),
            "meta": dict(self.meta),
        }


# ── Core logic ─────────────────────────────────────────────────────────────────


def build_vision_context(
    *,
    vision_provider: Any = None,
    query: Optional[str] = None,
    image: Any = None,
    image_url: Optional[str] = None,
    image_bytes: Any = None,
    provided_attributes: Optional[Dict[str, Any]] = None,
    garment_category: Optional[str] = None,
    requested_regions: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> VisionContext:
    """Construct a ``VisionContext`` by merging provided and visual attributes
    according to the fixed priority:

        request attributes > visual attributes > unavailable

    This is the single source of truth for attribute-merging logic in the QA
    pipeline.  QaOrchestrator calls this once and uses the result directly.

    Does NOT mutate any input dicts.
    """
    # ── Step 1: standardise inputs ─────────────────────────────────────────
    req_attrs = deepcopy(provided_attributes) if isinstance(provided_attributes, dict) else {}
    has_req = bool(req_attrs)
    req_regions = list(requested_regions) if requested_regions else []
    has_img = bool(image or image_url or image_bytes)

    vp_name = None
    if vision_provider is not None:
        vp_name = getattr(vision_provider, "__class__", type(vision_provider)).__name__

    # ── Step 2: provided attributes exist → skip vision ────────────────────
    if has_req:
        return VisionContext(
            effective_attributes=req_attrs,
            attribute_source="request",
            provided_attributes_used=True,
            visual_attributes_used=False,
            visual_attributes_present=False,
            vision_provider_used=False,
            vision_provider_name=vp_name,
            requested_regions=req_regions,
            meta={"_step": "provided_attrs"},
        )

    # ── Step 4: no image source → no vision call ───────────────────────────
    if not has_img or vision_provider is None:
        return VisionContext(
            effective_attributes={},
            attribute_source="none",
            provided_attributes_used=False,
            visual_attributes_used=False,
            visual_attributes_present=False,
            vision_provider_used=False,
            vision_provider_name=vp_name,
            requested_regions=req_regions,
        )

    # ── Step 3: call vision provider ───────────────────────────────────────
    try:
        vr = vision_provider.extract(
            image=image, image_url=image_url,
            image_bytes=image_bytes if isinstance(image_bytes, (str, bytes, type(None))) else None,
            query=query, garment_category=garment_category,
            regions=req_regions, context=context,
            provided_attributes=None,
        )
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Vision provider call failed in build_vision_context")
        return VisionContext(
            effective_attributes={},
            attribute_source="none",
            vision_provider_used=True,
            vision_provider_name=vp_name,
            requested_regions=req_regions,
            warnings=[WarningItem(
                code="vision_provider_error", scope="vision",
                message="Vision provider call failed.", severity=WarningSeverity.warn,
            )],
        )

    visual_attrs = dict(vr.attributes) if vr.attributes else {}
    has_visual = bool(visual_attrs)
    sources = build_visual_attribute_sources(visual_attrs, provider_name=vp_name or "vision_provider")
    meta = dict(getattr(vr, "meta", {}) or {})
    meta["_step"] = "vision_provider_called"

    return VisionContext(
        effective_attributes=visual_attrs,
        attribute_source="vision" if has_visual else "none",
        provided_attributes_used=False,
        visual_attributes_used=has_visual,
        visual_attributes_present=has_visual,
        vision_provider_used=True,
        vision_provider_name=vp_name,
        garment_instances=list(getattr(vr, "garment_instances", []) or []),
        regions=list(getattr(vr, "regions", []) or []),
        localized_regions=list(getattr(vr, "regions", []) or []),  # P1.4a: 3.1.2 regions
        requested_regions=req_regions,
        warnings=list(getattr(vr, "warnings", []) or []),
        sources=sources,
        meta=meta,
    )
