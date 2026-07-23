"""P0a.7 — VisionContext unit tests."""

from __future__ import annotations

import json

import pytest

from inference.serving.vision_context import (
    VisionContext,
    build_vision_context,
    build_visual_attribute_sources,
)
from inference.serving.vision_provider import MockVisionAttributeProvider


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_provider():
    return MockVisionAttributeProvider()


@pytest.fixture
def fake_provider():
    """Returns fake visual attributes for testing future real-provider behavior."""
    from inference.serving.vision_provider import VisionAttributeResult
    class FakeProvider:
        def extract(self, **kwargs):
            return VisionAttributeResult(
                attributes={
                    "fabric": {"value": "棉", "attribute_confidence": 0.82,
                               "source": "vision_provider", "region_ref": "region_1"},
                    "color": "蓝色",
                },
                garment_instances=[],
                regions=[],
                warnings=[],
                sources=[],
                used_tools=["fake_vision_provider"],
                meta={"provider": "fake", "real_pipeline_enabled": False},
            )
    return FakeProvider()


# ── Unit tests ─────────────────────────────────────────────────────────────────


def test_provided_attrs_not_call_provider(mock_provider):
    vc = build_vision_context(
        vision_provider=mock_provider,
        provided_attributes={"fabric": "棉"},
        image_url="http://x.com/img.jpg",
    )
    assert vc.attribute_source == "request"
    assert vc.effective_attributes == {"fabric": "棉"}
    assert vc.provided_attributes_used is True
    assert vc.vision_provider_used is False
    assert vc.visual_attributes_used is False


def test_no_attrs_no_image(mock_provider):
    vc = build_vision_context(vision_provider=mock_provider)
    assert vc.attribute_source == "none"
    assert vc.effective_attributes == {}
    assert vc.vision_provider_used is False


def test_image_url_mock_provider(mock_provider):
    vc = build_vision_context(
        vision_provider=mock_provider,
        image_url="http://x.com/img.jpg",
    )
    assert vc.vision_provider_used is True
    assert vc.effective_attributes == {}
    assert len(vc.warnings) == 1
    assert vc.warnings[0].code == "vision_provider_mock"


def test_image_url_fake_provider(fake_provider):
    """Fake provider returns attributes — effective_attrs come from vision."""
    vc = build_vision_context(vision_provider=fake_provider, image_url="http://x.com/a.jpg")
    assert vc.attribute_source == "vision"
    assert vc.effective_attributes["fabric"]["value"] == "棉"
    assert vc.effective_attributes["color"] == "蓝色"
    assert vc.visual_attributes_used is True
    assert vc.visual_attributes_present is True


def test_no_mutation(mock_provider):
    attrs = {"fabric": "棉"}
    build_vision_context(vision_provider=mock_provider, provided_attributes=attrs)
    assert attrs == {"fabric": "棉"}  # not mutated


def test_primitive_source():
    sources = build_visual_attribute_sources({"fabric": "棉"})
    assert len(sources) == 1
    assert sources[0]["field"] == "fabric"
    assert sources[0]["value"] == "棉"
    assert sources[0]["source"] == "vision_provider"


def test_dict_source():
    sources = build_visual_attribute_sources({
        "fabric": {"value": "棉", "attribute_confidence": 0.82, "region_ref": "r1"}
    })
    assert len(sources) == 1
    s = sources[0]
    assert s["attribute_confidence"] == 0.82
    assert s["region_ref"] == "r1"


def test_empty_source():
    assert build_visual_attribute_sources({}) == []
    assert build_visual_attribute_sources(None) == []  # type: ignore


def test_image_bytes_not_leaked():
    secret = "SECRET_123"
    vc = build_vision_context(
        vision_provider=MockVisionAttributeProvider(),
        image_bytes=secret,
    )
    j = json.dumps(vc.to_dict(), ensure_ascii=False)
    assert secret not in j


def test_requested_regions():
    vc = build_vision_context(
        vision_provider=MockVisionAttributeProvider(),
        image_url="http://x.com/a.jpg",
        requested_regions=["collar", "sleeve"],
    )
    assert vc.requested_regions == ["collar", "sleeve"]


def test_meta_counts():
    vc = build_vision_context(
        vision_provider=MockVisionAttributeProvider(),
        image_url="http://x.com/a.jpg",
    )
    assert vc.garment_instances == []
    assert vc.regions == []


def test_to_dict_json():
    vc = build_vision_context(
        vision_provider=MockVisionAttributeProvider(),
        image_url="http://x.com/a.jpg",
    )
    d = vc.to_dict()
    json.dumps(d, ensure_ascii=False)
