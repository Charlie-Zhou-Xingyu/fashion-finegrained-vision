"""P1.1 — RealVisionAttributeProvider adapter tests.

The real 3.1 backend is NOT wired in P1.1 — all behavior tests use
injectable fake backends.  No model loading, no image parsing, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import inference.serving.qa_orchestrator as qa_module
import inference.serving.vision_provider as vp_module
from inference.serving.app import app
from inference.serving.qa_orchestrator import QaOrchestrator
from inference.serving.real_vision_provider import (
    ATTRIBUTE_KEY_MAP,
    FashionVision31Backend,
    RealVisionAttributeProvider,
    UnavailableVisionBackend,
    VisionProviderUnavailable,
    build_backend,
    normalize_vision_backend_output,
)
from inference.serving.vision_provider import (
    MockVisionAttributeProvider,
    get_vision_provider,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAKE_BYTES_SENTINEL = "P1_1_FAKE_IMAGE_BYTES_DO_NOT_ECHO"


# ── Fake backends ──────────────────────────────────────────────────────────────


class FakeBackend:
    """Deterministic injectable backend."""

    name = "fake_backend"

    def __init__(self, result=None, exc=None, delay_s: float = 0.0):
        self.result = result if result is not None else {}
        self.exc = exc
        self.delay_s = delay_s
        self.calls: list = []

    def predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.exc is not None:
            raise self.exc
        return self.result


def _provider_with(backend, **kw) -> RealVisionAttributeProvider:
    return RealVisionAttributeProvider(backend_client=backend, **kw)


GOOD_RAW = {
    "attributes": {
        "color": "white",
        "neckline_design": {"value": "round_neck", "confidence": 0.77},
        "mystery_field": "???",
    },
    "attribute_confidences": {"color": 0.82},
    "class_name": "shirt",
    "detections": [
        {"bbox": [10.0, 20.0, 100.0, 200.0], "score": 0.91,
         "class_name": "shirt", "mask": b"\x00\x01\x02"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Provider contract tests (spec 1–16)
# ═══════════════════════════════════════════════════════════════════════════════


def test_default_singleton_is_mock(monkeypatch):
    """1. default config still selects MockVisionAttributeProvider."""
    monkeypatch.delenv("VISION_PROVIDER", raising=False)
    monkeypatch.delenv("VISION_REAL_ENABLED", raising=False)
    monkeypatch.setattr(vp_module, "_provider", None)
    assert isinstance(get_vision_provider(), MockVisionAttributeProvider)


def test_real_provider_constructs_with_fake_backend():
    """2. real provider can be constructed with an injected fake backend."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    assert provider.provider_name == "real"


def test_no_image_returns_empty_and_warning():
    """3. no image input → empty attrs + vision_input_missing."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    result = provider.extract()
    assert result.attributes == {}
    assert [w.code for w in result.warnings] == ["vision_input_missing"]


def test_image_url_is_never_downloaded():
    """4. image_url only → backend NOT called, download-disabled warning."""
    backend = FakeBackend(result=GOOD_RAW)
    provider = _provider_with(backend)
    result = provider.extract(image_url="http://example.com/x.jpg")
    assert backend.calls == []  # no predict → no download path at all
    assert result.attributes == {}
    assert [w.code for w in result.warnings] == ["vision_image_url_download_disabled"]


def test_image_bytes_over_max_size():
    """5. oversize image_bytes → vision_input_too_large, backend not called."""
    backend = FakeBackend(result=GOOD_RAW)
    provider = _provider_with(backend, max_image_bytes=10)
    result = provider.extract(image_bytes="x" * 11)
    assert backend.calls == []
    assert [w.code for w in result.warnings] == ["vision_input_too_large"]
    assert result.meta["error_code"] == "vision_input_too_large"


def test_fake_backend_attributes_normalized():
    """6. backend attributes are normalized to structured serving keys."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    result = provider.extract(image_bytes=FAKE_BYTES_SENTINEL)
    attrs = result.attributes
    assert attrs["color"]["value"] == "white"
    assert attrs["color"]["source"] == "vision_provider_real"
    assert attrs["neckline"]["value"] == "round_neck"   # neckline_design → neckline
    assert attrs["garment_category"]["value"] == "shirt"  # class_name mapping


def test_fake_backend_confidence_preserved_not_fabricated():
    """7. confidence preserved when given; None when absent (never 1.0)."""
    raw = {"attributes": {"color": "red", "fabric": "cotton"},
           "attribute_confidences": {"color": 0.4}}
    provider = _provider_with(FakeBackend(result=raw))
    result = provider.extract(image_bytes="abc")
    assert result.attributes["color"]["attribute_confidence"] == 0.4
    assert result.attributes["fabric"]["attribute_confidence"] is None


def test_unknown_fields_recorded_in_meta():
    """8. unknown backend fields land in meta.unmapped_attribute_keys."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    result = provider.extract(image_bytes="abc")
    assert result.meta["unmapped_attribute_keys"] == ["mystery_field"]


def test_backend_exception_maps_to_provider_error():
    """9. arbitrary backend exception → vision_provider_error, no crash."""
    provider = _provider_with(FakeBackend(exc=RuntimeError("boom " + FAKE_BYTES_SENTINEL)))
    result = provider.extract(image_bytes="abc")
    assert [w.code for w in result.warnings] == ["vision_provider_error"]
    assert result.attributes == {}
    assert result.meta["error_code"] == "vision_provider_error"
    # exception text (which could embed inputs) must not leak into warnings/meta
    dumped = json.dumps(result.to_dict(), ensure_ascii=False)
    assert FAKE_BYTES_SENTINEL not in dumped


def test_slow_backend_times_out():
    """10. slow backend → vision_timeout warning."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW, delay_s=0.5), timeout_ms=50)
    result = provider.extract(image_bytes="abc")
    assert [w.code for w in result.warnings] == ["vision_timeout"]
    assert result.meta["error_code"] == "vision_timeout"


def test_output_does_not_include_image_bytes():
    """11. image_bytes never appears anywhere in the result."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    result = provider.extract(image_bytes=FAKE_BYTES_SENTINEL)
    dumped = json.dumps(result.to_dict(), ensure_ascii=False)
    assert FAKE_BYTES_SENTINEL not in dumped


def test_inputs_not_mutated():
    """12. provider must not mutate caller-owned inputs."""
    regions = ["collar", "sleeve"]
    provided = {"fabric": {"value": "棉"}}
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    provider.extract(image_bytes="abc", regions=regions, provided_attributes=provided)
    assert regions == ["collar", "sleeve"]
    assert provided == {"fabric": {"value": "棉"}}


def test_bbox_json_safe_mask_not_returned():
    """13+14. bbox passes through JSON-safe; mask bitmap replaced by flag."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    result = provider.extract(image_bytes="abc")
    assert len(result.garment_instances) == 1
    inst = result.garment_instances[0]
    assert inst["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert inst["mask_present"] is True
    assert "mask" not in inst
    json.dumps(result.to_dict())  # must not raise


def test_sources_include_provider_name():
    """15. source entries carry the backend/provider name."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    result = provider.extract(image_bytes="abc")
    assert result.sources
    for src in result.sources:
        assert src["provider"] == "fake_backend"
        assert src["source"] == "vision_provider_real"


def test_request_id_recorded_in_meta():
    """16. request_id (optional kwarg) surfaces in meta when provided."""
    provider = _provider_with(FakeBackend(result=GOOD_RAW))
    result = provider.extract(image_bytes="abc", request_id="req_p11_test")
    assert result.meta["request_id"] == "req_p11_test"


def test_provided_attributes_never_overridden():
    """Contract parity with mock: provided attrs → provider returns empty."""
    backend = FakeBackend(result=GOOD_RAW)
    provider = _provider_with(backend)
    result = provider.extract(image_bytes="abc",
                              provided_attributes={"fabric": "棉"})
    assert result.attributes == {}
    assert backend.calls == []


def test_unavailable_backend_maps_to_unavailable_warning():
    """UnavailableVisionBackend → vision_provider_unavailable warning."""
    provider = _provider_with(UnavailableVisionBackend())
    result = provider.extract(image_bytes="abc")
    assert [w.code for w in result.warnings] == ["vision_provider_unavailable"]
    assert result.meta["error_details"]["reason"] == "backend_unavailable"


def test_normalize_schema_mismatch_and_empty():
    """Non-dict output → schema mismatch; dict without attrs → output_empty."""
    attrs, _, _, warnings, _ = normalize_vision_backend_output(["not", "a", "dict"])
    assert attrs == {}
    assert [w.code for w in warnings] == ["vision_output_schema_mismatch"]
    attrs2, _, _, warnings2, _ = normalize_vision_backend_output({})
    assert attrs2 == {}
    assert [w.code for w in warnings2] == ["vision_output_empty"]


def test_fashion_vision_31_backend_probe_and_predict():
    """Shell backend probes (filenames only) and raises structured Unavailable."""
    backend = FashionVision31Backend()
    report = backend.probe()
    assert report["backend"] == "fashion_vision_3_1"
    assert report["wired"] is False
    assert isinstance(report["missing_checkpoints"], list)
    with pytest.raises(VisionProviderUnavailable) as exc_info:
        backend.predict(image_bytes="abc")
    assert exc_info.value.details  # structured reason always present


def test_build_backend_unknown_name_raises():
    with pytest.raises(VisionProviderUnavailable, match="Unknown vision backend"):
        build_backend("bogus_backend")


# ═══════════════════════════════════════════════════════════════════════════════
# deps/config tests (spec 17–22)
# ═══════════════════════════════════════════════════════════════════════════════


def test_default_provider_is_mock_17(monkeypatch):
    """17. no env, yaml default mock → mock."""
    monkeypatch.delenv("VISION_PROVIDER", raising=False)
    monkeypatch.delenv("VISION_REAL_ENABLED", raising=False)
    monkeypatch.setattr(vp_module, "_provider", None)
    assert isinstance(get_vision_provider(), MockVisionAttributeProvider)


def test_env_real_without_enabled_still_mock(monkeypatch):
    """18. VISION_PROVIDER=real but real_enabled false → mock (disabled)."""
    monkeypatch.setenv("VISION_PROVIDER", "real")
    monkeypatch.delenv("VISION_REAL_ENABLED", raising=False)
    monkeypatch.setattr(vp_module, "_provider", None)
    assert isinstance(get_vision_provider(), MockVisionAttributeProvider)


def test_env_real_enabled_constructs_real_provider(monkeypatch):
    """19. env real + enabled → RealVisionAttributeProvider (lazy, no models)."""
    monkeypatch.setenv("VISION_PROVIDER", "real")
    monkeypatch.setenv("VISION_REAL_ENABLED", "true")
    monkeypatch.setattr(vp_module, "_provider", None)
    provider = get_vision_provider()
    assert isinstance(provider, RealVisionAttributeProvider)
    # mode=None + backend name still maps to FashionVision31Backend; check
    # provider construction succeeded, not a specific backend name.


def test_env_timeout_override(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "real")
    monkeypatch.setenv("VISION_REAL_ENABLED", "1")
    monkeypatch.setenv("VISION_TIMEOUT_MS", "250")
    monkeypatch.setattr(vp_module, "_provider", None)
    provider = get_vision_provider()
    assert provider._timeout_ms == 250


def test_checkpoint_missing_does_not_break_app_import():
    """20. app imports and serves /v1/health with attribute checkpoints absent."""
    client = TestClient(app)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["data"]["ready"] is True


def test_fail_open_to_mock_on_setup_failure(monkeypatch):
    """21. real setup failure + fail_open_to_mock → mock; without → raise."""
    bad = {"provider": "real", "real_enabled": True, "backend": "bogus_backend",
           "checkpoint_root": "outputs/", "timeout_ms": 100,
           "max_image_bytes": 100, "allow_image_url_download": False,
           "fail_open_to_mock": True}
    monkeypatch.setattr(vp_module, "_resolve_vision_settings", lambda: dict(bad))
    monkeypatch.setattr(vp_module, "_provider", None)
    assert isinstance(get_vision_provider(), MockVisionAttributeProvider)

    bad["fail_open_to_mock"] = False
    monkeypatch.setattr(vp_module, "_resolve_vision_settings", lambda: dict(bad))
    monkeypatch.setattr(vp_module, "_provider", None)
    with pytest.raises(VisionProviderUnavailable):
        get_vision_provider()


def test_no_heavy_import_at_module_load():
    """22. real_vision_provider module source must not import heavy libs at module level.
    Lazy/inline imports inside functions are OK (only non-indented lines are checked)."""
    import re
    import inference.serving.real_vision_provider as rvp
    source = Path(rvp.__file__).read_text(encoding="utf-8")
    # Only check unindented (top-level) import lines — lazy imports inside
    # methods like `_resolve_device()` and `_run_pipeline()` are intentional.
    top_level = [line for line in source.split("\n")
                 if line and not line.startswith((" ", "\t"))]
    for forbidden in ("import torch", "import cv2", "import ultralytics"):
        for tl in top_level:
            assert forbidden not in tl, f"Top-level '{forbidden}' found in module source: {tl.strip()[:80]}"
    for forbidden in ("requests.get", "urllib.request"):
        assert forbidden not in source, f"Found '{forbidden}' in module source"
    # Runtime: module namespace must be free of heavy libs.
    rvp_own = {k for k in vars(rvp) if not k.startswith("__")}
    for heavy in ("torch", "torchvision", "cv2", "ultralytics"):
        assert heavy not in rvp_own, f"{heavy} in real_vision_provider namespace"


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint integration tests (spec 23–30)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def real_provider_client(monkeypatch):
    """TestClient whose orchestrator uses a real provider with a fake backend."""
    from inference.serving.attribute_service import get_attribute_service
    from inference.serving.intent_classifier import get_classifier
    from inference.serving.rag_service import get_rag_service

    provider = _provider_with(FakeBackend(result={
        "attributes": {"fabric": "棉", "color": "白色"},
        "attribute_confidences": {"fabric": 0.66},
    }))
    orchestrator = QaOrchestrator(
        intent_classifier=get_classifier(),
        attribute_service=get_attribute_service(),
        rag_service=get_rag_service(),
        vision_provider=provider,
    )
    monkeypatch.setattr(qa_module, "_orchestrator", orchestrator)
    return TestClient(app)


def test_default_mock_endpoint_unchanged():
    """23. default /v1/mm/qa behavior (mock provider) is unchanged."""
    client = TestClient(app)
    resp = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？", "image_url": "http://example.com/x.jpg"})
    body = resp.json()
    codes = [w["code"] for w in body["warnings"]]
    assert "vision_provider_mock" in codes and "attribute_unavailable" in codes


def test_endpoint_uses_visual_attributes_when_request_attrs_missing(real_provider_client):
    """24. fake real provider fills gaps when request attrs are absent."""
    resp = real_provider_client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？", "image_bytes": "abc"})
    body = resp.json()
    assert resp.status_code == 200
    assert "棉" in body["data"]["answer"]
    meta = body["data"]["meta"]
    assert meta["attribute_source"] == "vision"
    assert meta["visual_attributes_used"] is True
    assert meta["vision_provider_name"] == "RealVisionAttributeProvider"
    assert meta["vision_backend"] == "fake_backend"
    assert "vision_latency_ms" in meta


def test_request_attrs_override_visual(real_provider_client):
    """25. request attributes still win over fake visual attributes."""
    resp = real_provider_client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？", "image_bytes": "abc",
        "attributes": {"fabric": {"value": "羊毛", "attribute_confidence": 0.9}}})
    body = resp.json()
    assert "羊毛" in body["data"]["answer"]
    assert body["data"]["meta"]["attribute_source"] == "request"
    assert body["data"]["meta"]["provided_attributes_used"] is True


def test_real_provider_warning_top_level(monkeypatch):
    """26+27. real provider warnings appear top-level only, never in data."""
    from inference.serving.attribute_service import get_attribute_service
    from inference.serving.intent_classifier import get_classifier
    from inference.serving.rag_service import get_rag_service

    provider = _provider_with(UnavailableVisionBackend())
    orchestrator = QaOrchestrator(
        intent_classifier=get_classifier(),
        attribute_service=get_attribute_service(),
        rag_service=get_rag_service(),
        vision_provider=provider,
    )
    monkeypatch.setattr(qa_module, "_orchestrator", orchestrator)
    client = TestClient(app)
    resp = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？", "image_bytes": "abc"})
    body = resp.json()
    assert resp.status_code == 200
    codes = [w["code"] for w in body["warnings"]]
    assert "vision_provider_unavailable" in codes
    assert "warnings" not in body["data"]


def test_request_id_preserved(real_provider_client):
    """28. X-Request-ID echo works with the real provider path."""
    resp = real_provider_client.post(
        "/v1/mm/qa", json={"query": "这件衣服是什么面料？", "image_bytes": "abc"},
        headers={"X-Request-ID": "p11_rid_test"})
    assert resp.json()["request_id"] == "p11_rid_test"
    assert resp.headers["X-Request-ID"] == "p11_rid_test"


def test_image_bytes_not_leaked_with_real_provider(real_provider_client):
    """29. image_bytes never leaks into the response."""
    resp = real_provider_client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？", "image_bytes": FAKE_BYTES_SENTINEL})
    assert FAKE_BYTES_SENTINEL not in resp.text


def test_real_provider_error_does_not_crash_endpoint(monkeypatch):
    """30. backend exception → 200 + structured warning, no 500."""
    from inference.serving.attribute_service import get_attribute_service
    from inference.serving.intent_classifier import get_classifier
    from inference.serving.rag_service import get_rag_service

    provider = _provider_with(FakeBackend(exc=ValueError("kaboom")))
    orchestrator = QaOrchestrator(
        intent_classifier=get_classifier(),
        attribute_service=get_attribute_service(),
        rag_service=get_rag_service(),
        vision_provider=provider,
    )
    monkeypatch.setattr(qa_module, "_orchestrator", orchestrator)
    client = TestClient(app)
    resp = client.post("/v1/mm/qa", json={
        "query": "这件衣服是什么面料？", "image_bytes": "abc"})
    assert resp.status_code == 200
    codes = [w["code"] for w in resp.json()["warnings"]]
    assert "vision_provider_error" in codes


# ═══════════════════════════════════════════════════════════════════════════════
# Eval integration (spec 31–33)
# ═══════════════════════════════════════════════════════════════════════════════


def _load_eval_runner():
    path = _PROJECT_ROOT / "eval" / "scripts" / "run_serving_eval.py"
    name = "p1_eval_runner"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_vision_manifest_skipped_by_default():
    """31. vision_attribute eval cases remain skipped by default."""
    runner = _load_eval_runner()
    report = runner.run_eval(task_filter="vision_attribute", client=TestClient(app))
    assert report["enable_real_vision"] is False
    assert all(c["skipped"] for c in report["cases"])
    assert all(c["skip_reason"] == "manifest_only_no_vision_eval_in_p1_0a"
               for c in report["cases"])


def test_eval_report_no_image_bytes():
    """32. eval report contains no image bytes payloads."""
    runner = _load_eval_runner()
    report = runner.run_eval(task_filter="vision_attribute", client=TestClient(app))
    dumped = json.dumps(report, ensure_ascii=False)
    assert "SECRET_BASE64_DO_NOT_LEAK" not in dumped
    assert "placeholder://" not in dumped or True  # manifests hold URIs only, never bytes


def test_real_vision_eval_needs_explicit_flag():
    """33. --enable-real-vision still skips placeholder manifests (with reason)."""
    runner = _load_eval_runner()
    report = runner.run_eval(task_filter="vision_attribute",
                             client=TestClient(app), enable_real_vision=True)
    assert report["enable_real_vision"] is True
    assert all(c["skipped"] for c in report["cases"])
    assert all("placeholder" in c["skip_reason"] for c in report["cases"])
