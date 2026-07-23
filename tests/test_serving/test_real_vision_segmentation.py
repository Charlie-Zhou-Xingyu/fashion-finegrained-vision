"""P1.2 — Real 3.1.1 segmentation backend tests.

Default safe tests (1-11): always run, no checkpoint, no GPU, no model loading.
Optional real tests (12-13): gated by RUN_REAL_VISION_TESTS=1 + checkpoint check.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import inference.serving.qa_orchestrator as qa_module
import inference.serving.vision_provider as vp_module
from inference.serving.app import app
from inference.serving.qa_orchestrator import QaOrchestrator
from inference.serving.real_vision_provider import (
    FashionVision31SegmentationBackend,
    RealVisionAttributeProvider,
    UnavailableVisionBackend,
    VisionProviderUnavailable,
    normalize_vision_backend_output,
)
from inference.serving.vision_provider import MockVisionAttributeProvider, get_vision_provider

FAKE_SENTINEL = "P1_2_FAKE_IMG_BYTES_SEG_TEST"
FIXTURE_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "test_images_13cls" / "000008.jpg"

_RUN_OPT = os.getenv("RUN_REAL_VISION_TESTS") == "1"


# ── Fake pipeline runner (test seam) ───────────────────────────────────────────


def _fake_pipeline_runner(image_path: str, output_dir: str) -> dict:
    """Write synthetic YOLO + SAM output JSONs (no real model)."""
    od = Path(output_dir)
    (od / "01_yolo").mkdir(parents=True, exist_ok=True)
    (od / "02_samhq").mkdir(parents=True, exist_ok=True)

    det = {
        "model_weights": "fake.pt", "device": "cpu", "class_names": {},
        "images": [{
            "image_id": 0, "image_path": image_path,
            "file_name": Path(image_path).name, "width": 640, "height": 480,
            "num_detections": 2,
            "detections": [
                {"det_id": 0, "class_id": 0, "class_name": "short sleeve top",
                 "fine_class_id": 0, "fine_class_name": "short sleeve top",
                 "coarse_class_id": 0, "coarse_class_name": "top",
                 "confidence": 0.91, "bbox_xyxy": [10.0, 20.0, 100.0, 200.0],
                 "bbox_xywh": [10.0, 20.0, 90.0, 180.0],
                 "bbox_format": "xyxy_abs_pixels", "image_width": 640, "image_height": 480},
                {"det_id": 1, "class_id": 7, "class_name": "trousers",
                 "fine_class_id": 7, "fine_class_name": "trousers",
                 "coarse_class_id": 1, "coarse_class_name": "pants",
                 "confidence": 0.87, "bbox_xyxy": [200.0, 300.0, 500.0, 650.0],
                 "bbox_xywh": [200.0, 300.0, 300.0, 350.0],
                 "bbox_format": "xyxy_abs_pixels", "image_width": 640, "image_height": 480},
                # det_id 2 has bad bbox (3 values) — should be dropped by _valid_bbox
                {"det_id": 2, "class_id": 0, "class_name": "short sleeve top",
                 "fine_class_id": 0, "fine_class_name": "short sleeve top",
                 "coarse_class_id": 0, "coarse_class_name": "top",
                 "confidence": 0.5, "bbox_xyxy": [10.0, 20.0, 30.0],
                 "bbox_format": "xyxy_abs_pixels", "image_width": 640, "image_height": 480},
            ],
        }],
    }
    seg = {
        "model_checkpoint": "fake.pt", "device": "cpu", "images": [{
            "image_id": 0, "image_path": image_path,
            "file_name": Path(image_path).name, "width": 640, "height": 480,
            "num_segments": 1,
            "segments": [
                {"det_id": 0, "class_id": 0, "class_name": "short sleeve top",
                 "confidence": 0.91, "bbox_xyxy": [10.0, 20.0, 100.0, 200.0],
                 "bbox_format": "xyxy_abs_pixels",
                 "mask_path": str(Path(output_dir) / "02_samhq" / "masks" / "m.png"),
                 "mask_area": 5000, "sam_score": 0.95, "sam_best_mask_idx": 0,
                 "image_width": 640, "image_height": 480},
            ],
        }],
    }
    (od / "01_yolo" / "detections.json").write_text(
        json.dumps(det, ensure_ascii=False), encoding="utf-8")
    (od / "02_samhq" / "segmentation_results.json").write_text(
        json.dumps(seg, ensure_ascii=False), encoding="utf-8")
    return {"status": "ok", "timing": {"total_s": 0.01}}


# ═══════════════════════════════════════════════════════════════════════════════
# Default safe tests (always run)
# ═══════════════════════════════════════════════════════════════════════════════


def test_backend_constructs_without_model_load():
    """1. construct + probe() without loading torch/ultralytics."""
    backend = FashionVision31SegmentationBackend()
    report = backend.probe()
    assert report["backend"] == "fashion_vision_3_1_segmentation"
    assert report["mode"] == "segmentation_only"
    assert report["wired"] is True
    assert isinstance(report["missing_checkpoints"], list)


def test_probe_reports_checkpoint_status():
    """2. probe() detects checkpoint presence correctly."""
    backend = FashionVision31SegmentationBackend()
    report = backend.probe()
    assert report["available"] == (not report["missing_modules"] and not report["missing_checkpoints"])
    # With checkpoints present locally: YOLO+SAM checkpoints should exist → available=True.
    assert report["available"] is True


def test_image_url_download_disabled():
    """3. provider blocks url-only input (download-disabled warning)."""
    backend = FashionVision31SegmentationBackend(
        pipeline_runner=_fake_pipeline_runner)
    provider = RealVisionAttributeProvider(backend_client=backend)
    result = provider.extract(image_url="http://example.com/x.jpg")
    assert [w.code for w in result.warnings] == ["vision_image_url_download_disabled"]
    assert result.attributes == {}


def test_fake_runner_output_maps_to_garment_instances():
    """4. fake runner → 2 valid instances + 1 bbox dropped = 2 instances."""
    backend = FashionVision31SegmentationBackend(
        pipeline_runner=_fake_pipeline_runner)
    result = backend.predict(image_bytes="ZmFrZQ==")  # "fake" in b64
    instances = result["garment_instances"]
    assert len(instances) == 2
    assert instances[0]["category"] == "top"
    assert instances[0]["fine_class_name"] == "short sleeve top"
    assert instances[0]["confidence"] == 0.91
    assert instances[0]["mask_present"] is True   # det_id 0 has seg entry
    assert "mask_ref" in instances[0]
    assert instances[1]["category"] == "pants"
    assert instances[1]["mask_present"] is False  # det_id 1 has NO seg entry
    assert "mask_ref" not in instances[1]
    assert result["meta"]["num_detections"] == 3   # 3 dets total in JSON
    assert result["meta"]["num_segments"] == 1
    assert result["meta"]["invalid_bbox_count"] == 1
    assert result["meta"]["mask_bitmap_returned"] is False
    assert result["meta"]["vision_backend_mode"] == "segmentation_only"


def test_bbox_validation_rejects_bad_shape():
    """5. 3-value bbox dropped; instances count reflects this."""
    backend = FashionVision31SegmentationBackend(
        pipeline_runner=_fake_pipeline_runner)
    result = backend.predict(image_bytes=b"abc")
    assert result["meta"]["invalid_bbox_count"] == 1
    assert len(result["garment_instances"]) == 2


def test_mask_bitmap_stripped():
    """6. mask bitmap never present; only mask_present + mask_ref placeholder."""
    backend = FashionVision31SegmentationBackend(
        pipeline_runner=_fake_pipeline_runner)
    result = backend.predict(image_bytes=b"abc")
    for inst in result["garment_instances"]:
        assert "mask" not in inst
        assert "mask_bitmap" not in inst
        assert isinstance(inst["mask_present"], bool)
        if inst["mask_present"]:
            assert inst["mask_ref"].startswith("mask_inst_")
        # mask_ref must NOT be a real filesystem path.
        if "mask_ref" in inst:
            assert not Path(inst["mask_ref"]).is_absolute()
            assert "output" not in inst["mask_ref"] and "temp" not in inst["mask_ref"]


def test_temp_paths_not_leaked():
    """7. no temp/absolute paths in result; no mask_path strings leaked."""
    backend = FashionVision31SegmentationBackend(
        pipeline_runner=_fake_pipeline_runner)
    result = backend.predict(image_bytes=b"abc")
    dumped = json.dumps(result, ensure_ascii=False)
    assert "_p12_seg_" not in dumped
    # mask_ref is a non-sensitive placeholder — no real filesystem paths.
    for inst in result["garment_instances"]:
        if "mask_ref" in inst:
            assert "/" not in inst["mask_ref"]
            assert "\\" not in inst["mask_ref"]


def test_backend_exception_mapped_gracefully():
    """8. any exception in predict → VisionProviderUnavailable or RuntimeError from provider."""
    bad_backend = FashionVision31SegmentationBackend(
        pipeline_runner=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("crash")))
    provider = RealVisionAttributeProvider(backend_client=bad_backend)
    result = provider.extract(image_bytes=b"abc")
    assert [w.code for w in result.warnings] == ["vision_provider_error"]
    assert result.attributes == {}


def test_provider_timeout_still_works_with_seg_backend():
    """9. slow backend (0.5s delay) + 50ms timeout → vision_timeout."""
    def _slow(image_path, output_dir):  # noqa: ARG001
        import time
        time.sleep(0.5)
        return {"attributes": {}}
    backend = FashionVision31SegmentationBackend(pipeline_runner=_slow)
    provider = RealVisionAttributeProvider(backend_client=backend, timeout_ms=50)
    result = provider.extract(image_bytes=b"abc")
    assert [w.code for w in result.warnings] == ["vision_timeout"]


def test_mm_qa_default_mock_unchanged(monkeypatch):
    """10. /v1/mm/qa still works with mock by default (regression)."""
    monkeypatch.delenv("VISION_PROVIDER", raising=False)
    monkeypatch.delenv("VISION_REAL_ENABLED", raising=False)
    monkeypatch.setattr(vp_module, "_provider", None)
    monkeypatch.setattr(qa_module, "_orchestrator", None)
    assert isinstance(get_vision_provider(), MockVisionAttributeProvider)
    client = TestClient(app)
    resp = client.post("/v1/mm/qa", json={"query": "这是什么面料？",
                                            "attributes": {"fabric": "棉"}})
    assert resp.status_code == 200
    assert "棉" in resp.json()["data"]["answer"]


def test_no_image_bytes_in_response():
    """11. image_bytes payload never appears in any response field."""
    backend = FashionVision31SegmentationBackend(
        pipeline_runner=_fake_pipeline_runner)
    provider = RealVisionAttributeProvider(backend_client=backend, timeout_ms=60000)
    result = provider.extract(image_bytes=FAKE_SENTINEL)
    dumped = json.dumps(result.to_dict(), ensure_ascii=False)
    assert FAKE_SENTINEL not in dumped


def test_provider_extract_preserves_garment_instances():
    """11b (P1.3 regression). backend 'garment_instances' key must survive
    normalize_vision_backend_output — it was silently dropped before P1.3
    because normalize only read 'detections'/'instances'."""
    backend = FashionVision31SegmentationBackend(
        pipeline_runner=_fake_pipeline_runner)
    provider = RealVisionAttributeProvider(backend_client=backend, timeout_ms=60000)
    result = provider.extract(image_bytes=b"abc")
    assert len(result.garment_instances) == 2
    cats = {i.get("category") for i in result.garment_instances}
    assert cats == {"top", "pants"}
    # No leak through the provider layer either.
    dumped = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "masks" not in dumped.replace("mask_present", "").replace("mask_ref", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Optional real tests (RUN_REAL_VISION_TESTS=1)
# ═══════════════════════════════════════════════════════════════════════════════


_skip_real = not _RUN_OPT


@pytest.mark.skipif(_skip_real, reason="set RUN_REAL_VISION_TESTS=1 to run real 3.1.1 tests")
def test_real_backend_probe_available():
    """12. probe() with real checkpoints present → available=True."""
    backend = FashionVision31SegmentationBackend(yolo_device="cpu", sam_device="cpu")
    report = backend.probe()
    assert report["available"] is True
    assert "fashion_vision_3_1_segmentation" in report["backend"]


@pytest.mark.skipif(_skip_real, reason="set RUN_REAL_VISION_TESTS=1 to run real 3.1.1 tests")
def test_real_backend_runs_on_one_fixture_image():
    """13. Run real YOLO+SAM on one local fixture — does not crash, output is legal."""
    assert FIXTURE_IMAGE.exists(), f"Fixture not found: {FIXTURE_IMAGE}"
    backend = FashionVision31SegmentationBackend(
        yolo_device="cpu", sam_device="cpu",
        pipeline_runner=None,  # real path
    )
    try:
        result = backend.predict(image_bytes=FIXTURE_IMAGE.read_bytes())
    except VisionProviderUnavailable as exc:
        pytest.skip(f"Backend unavailable: {exc}")
    assert isinstance(result, dict)
    instances = result.get("garment_instances", [])
    assert isinstance(instances, list)
    meta = result.get("meta", {})
    assert isinstance(meta["num_garment_instances"], int)
    # -- validate every instance --
    for inst in instances:
        assert isinstance(inst["instance_id"], str)
        assert isinstance(inst["category"], str) and inst["category"]
        assert "bbox" in inst and len(inst["bbox"]) == 4
        assert all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in inst["bbox"])
        assert isinstance(inst["mask_present"], bool)
        assert "mask" not in inst
        assert "mask_bitmap" not in inst
        if inst["mask_present"]:
            assert isinstance(inst.get("mask_ref"), str)
            assert "output" not in inst["mask_ref"] and "temp" not in inst["mask_ref"]
    dumped = json.dumps(result, ensure_ascii=False)
    assert str(FIXTURE_IMAGE) not in dumped  # no fixture path leaked
