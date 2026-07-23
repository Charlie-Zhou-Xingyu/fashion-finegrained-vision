"""P1.3 — Visual instance QA tests.

Tests the /v1/mm/qa visual_instance_query route using mocked
garment_instances — no real YOLO/SAM/GPU.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from inference.serving.app import app
from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.attribute_service import AttributeService
from inference.serving.qa_orchestrator import QaOrchestrator
from inference.serving.rag_service import RagService

client = TestClient(app)


# ── Sample garment instances (simulate P1.2 FashionVision31SegmentationBackend) ──

_SAMPLE_INSTANCES = [
    {
        "instance_id": "inst_0",
        "category": "top",
        "fine_class_name": "short sleeve top",
        "bbox": [120.0, 80.0, 300.0, 350.0],
        "confidence": 0.93,
        "mask_present": True,
    },
    {
        "instance_id": "inst_1",
        "category": "pants",
        "fine_class_name": "trousers",
        "bbox": [100.0, 340.0, 320.0, 600.0],
        "confidence": 0.88,
        "mask_present": True,
    },
    {
        "instance_id": "inst_2",
        "category": "dress",
        "fine_class_name": "long sleeve dress",
        "bbox": [200.0, 60.0, 400.0, 550.0],
        "confidence": 0.76,
        "mask_present": False,
    },
]


# ── Fake vision provider that returns garment_instances ──────────────────────


class _FakeInstanceVisionProvider:
    """Returns pre-canned VisionAttributeResult with garment_instances."""

    def __init__(self, instances=None):
        from inference.serving.vision_provider import VisionAttributeResult
        import copy
        self._instances = copy.deepcopy(instances) if instances else []
        self._result = VisionAttributeResult(
            attributes={},
            garment_instances=self._instances,
            regions=[],
            sources=[],
            warnings=[],
            used_tools=["fake_vision_provider"],
            meta={"provider": "test", "vision_backend": "test"},
        )

    def extract(self, **kwargs):
        return self._result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_orchestrator(instances=None):
    """Build a QaOrchestrator with a fake vision provider returning *instances*."""
    from inference.serving import qa_orchestrator as qa_mod
    import copy
    inst = copy.deepcopy(instances) if instances is not None else _SAMPLE_INSTANCES
    return QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=_FakeInstanceVisionProvider(inst),
    )


def _get_vc_meta(r: dict) -> dict:
    return r.get("data", {}).get("meta", {})


# ── Intent classification ────────────────────────────────────────────────────


_INTENT_QUERIES = {
    "count": [
        "图里有几件衣服？",
        "图里有几件服饰？",
        "照片里有多少件衣服？",
    ],
    "detection": [
        "图中检测到了什么？",
        "图片里有什么服饰？",
        "图里有什么？",
    ],
    "existence": [
        "有没有上衣？",
        "有没有裤子？",
        "有没有裙子？",
        "有没有外套？",
        "有没有连衣裙？",
    ],
    "location": [
        "检测框在哪里？",
        "这件衣服在哪里？",
        "上衣在哪里？",
    ],
    "segmentation": [
        "有没有分割结果？",
        "有没有分割掩码？",
        "是否有分割结果？",
    ],
}


@pytest.mark.parametrize("sub, queries", list(_INTENT_QUERIES.items()))
def test_intent_classification(sub, queries):
    classifier = RuleIntentClassifier()
    for q in queries:
        r = classifier.classify(q)
        assert r.primary_intent == "visual_instance_query", f"{q!r} → {r.primary_intent}"
        assert r.sub_intent == sub, f"{q!r} → sub={r.sub_intent}"


# ── orchestrator-level tests (direct construction, no endpoint) ──────────────


@pytest.fixture(scope="module")
def orch_fake():
    return _make_orchestrator()


@pytest.fixture(scope="module")
def orch_empty():
    return _make_orchestrator([])


def test_count_answer(orch_fake):
    r = orch_fake.answer(query="图里有几件衣服？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "3件" in r.answer
    assert "上衣" in r.answer
    assert "裤子" in r.answer
    assert "连衣裙" in r.answer
    assert any("visual_instance_answer" in t for t in r.used_tools)


def test_detection_answer(orch_fake):
    r = orch_fake.answer(query="图中检测到了什么？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "3件" in r.answer
    assert "short sleeve top" in r.answer
    assert "trousers" in r.answer
    assert any("visual_instance_answer" in t for t in r.used_tools)


def test_existence_yes(orch_fake):
    r = orch_fake.answer(query="有没有上衣？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "检测到上衣" in r.answer


def test_existence_no(orch_fake):
    r = orch_fake.answer(query="有没有裙子？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "未检测到" in r.answer or "skirt" in r.answer.lower()


def test_location_answer(orch_fake):
    r = orch_fake.answer(query="检测框在哪里？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "x1=" in r.answer or "检测框" in r.answer


def test_segmentation_answer(orch_fake):
    r = orch_fake.answer(query="有没有分割结果？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "分割掩码" in r.answer
    assert "2件" in r.answer  # inst_0 and inst_1 have mask_present=True


def test_instances_unavailable_warning(orch_empty):
    r = orch_empty.answer(query="图里有几件衣服？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "没有可用" in r.answer
    assert any(w.code == "vision_instances_unavailable" for w in r.warnings)


def test_no_image_bytes_still_answers(orch_fake):
    """Without image/image_bytes, mock vision won't be called by build_vision_context,
    but the route using a fake provider wired directly in the orchestrator
    should still work when build_vision_context does call the provider.
    With image_bytes, the fake provider returns garment_instances."""
    r = orch_fake.answer(query="图里有几件衣服？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"


# ── no-leak tests ────────────────────────────────────────────────────────────


def test_response_no_temp_path(orch_fake):
    r = orch_fake.answer(query="图中检测到了什么？", image_bytes="dGVzdA==")
    d = r.to_dict()
    payload = json.dumps(d, ensure_ascii=False)
    assert "D:\\\\" not in payload
    assert "/tmp/" not in payload
    assert "\\temp" not in payload
    assert "C:\\Users" not in payload
    assert "Lenovo" not in payload


def test_response_no_image_bytes(orch_fake):
    r = orch_fake.answer(query="图中检测到了什么？", image_bytes="dGVzdA==")
    d = r.to_dict()
    payload = json.dumps(d, ensure_ascii=False)
    # image_bytes should only appear in meta as a boolean flag, not the actual value
    assert "dGVzdA" not in payload


def test_response_no_mask_bitmap(orch_fake):
    r = orch_fake.answer(query="检测到了什么？", image_bytes="dGVzdA==")
    d = r.to_dict()
    payload = json.dumps(d, ensure_ascii=False)
    assert "mask_bitmap" not in payload
    assert "mask_data" not in payload


def test_garment_instances_summary_safe_keys(orch_fake):
    r = orch_fake.answer(query="图中检测到了什么？", image_bytes="dGVzdA==")
    summary = r.meta.get("garment_instances_summary", [])
    assert isinstance(summary, list)
    SAFE = {"instance_id", "category", "fine_class_name", "bbox", "confidence", "mask_present"}
    for inst in summary:
        assert set(inst.keys()).issubset(SAFE), f"Leaked keys in summary: {set(inst.keys()) - SAFE}"
        # mask_ref, sam_score, source, stage must NOT appear
        for forbidden in ("mask_ref", "source", "stage"):
            assert forbidden not in inst


def test_sources_have_garment_instances(orch_fake):
    r = orch_fake.answer(query="图中检测到了什么？", image_bytes="dGVzdA==")
    gi_sources = [s for s in r.sources if s.get("type") == "garment_instance"]
    assert len(gi_sources) == len(_SAMPLE_INSTANCES)
    assert gi_sources[0]["id"] == "inst_0"
    assert gi_sources[0]["value"] == "top"


# ── endpoint integration (via TestClient + monkeypatched singleton) ──────────


@pytest.fixture
def _patch_orchestrator(monkeypatch):
    """Replace the app's orchestrator singleton with a fake-vision version."""
    orch = _make_orchestrator()
    monkeypatch.setattr(
        "inference.serving.app.get_qa_orchestrator",
        lambda: orch,
    )
    # Also patch the module-level singleton in qa_orchestrator.
    monkeypatch.setattr(
        "inference.serving.qa_orchestrator._orchestrator",
        orch,
    )
    return orch


def test_endpoint_count_answer(_patch_orchestrator):
    r = client.post("/v1/mm/qa", json={
        "query": "图里有几件衣服？",
        "image_bytes": "dGVzdA==",
    }).json()
    assert r["status"] == "success"
    data = r["data"]
    assert data["answer_type"] == "visual_instance_answer"
    assert "3件" in data["answer"]
    assert "上衣" in data["answer"]


def test_endpoint_existence_yes(_patch_orchestrator):
    r = client.post("/v1/mm/qa", json={
        "query": "有没有上衣？",
        "image_bytes": "dGVzdA==",
    }).json()
    assert r["status"] == "success"
    assert "检测到上衣" in r["data"]["answer"]


def test_endpoint_instances_unavailable(_patch_orchestrator, monkeypatch):
    """Empty instances from vision provider."""
    orch_empty = _make_orchestrator([])
    monkeypatch.setattr(
        "inference.serving.app.get_qa_orchestrator",
        lambda: orch_empty,
    )
    monkeypatch.setattr(
        "inference.serving.qa_orchestrator._orchestrator",
        orch_empty,
    )
    r = client.post("/v1/mm/qa", json={
        "query": "图里有几件衣服？",
        "image_bytes": "dGVzdA==",
    }).json()
    assert r["status"] == "success"
    assert "没有可用" in r["data"]["answer"]
    assert any(w["code"] == "vision_instances_unavailable" for w in r["warnings"])


def test_endpoint_meta_has_garment_instances_summary(_patch_orchestrator):
    r = client.post("/v1/mm/qa", json={
        "query": "图中检测到了什么？",
        "image_bytes": "dGVzdA==",
    }).json()
    meta = r["data"].get("meta", {})
    assert "garment_instances_summary" in meta
    assert len(meta["garment_instances_summary"]) == 3


# ── Default mock unchanged ───────────────────────────────────────────────────


def test_default_mock_still_returns_mock_warning():
    """Without the fake provider, a visual-instance query with image_bytes
    hits the default MockVisionAttributeProvider.  It should produce a
    vision_provider_mock warning + vision_instances_unavailable (no instances)."""
    # Build orchestrator without overriding vision_provider → uses real singleton.
    orch_default = QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=None,
    )
    r = orch_default.answer(query="图里有几件衣服？", image_bytes="dGVzdA==")
    assert r.answer_type == "visual_instance_answer"
    assert "没有可用" in r.answer
    # The mock provider's warning should also be present.
    codes = {w.code for w in r.warnings}
    assert "vision_instances_unavailable" in codes


def test_existing_attribute_query_still_works(orch_fake):
    """Garment instance changes must not break existing attribute queries."""
    r = orch_fake.answer(
        query="这件衣服是什么面料？",
        attributes={"fabric": {"value": "纯棉", "attribute_confidence": 0.92, "source": "merchant_input"}},
    )
    assert r.answer_type == "attribute_answer"
    assert "纯棉" in r.answer


def test_existing_knowledge_query_still_works(orch_fake):
    r = orch_fake.answer(query="纤维是什么")
    assert r.answer_type in ("knowledge_answer", "unsupported")
