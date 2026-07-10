"""
Unit tests for FashionpediaPartDetector — mapping, filtering, edge cases.

All tests mock YOLO internals; no GPU or model weights required.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mapping consistency
# ---------------------------------------------------------------------------


def test_core_part_map_has_13_parts() -> None:
    from fashion_vision.localization.fashionpedia_part_detector import FP_CORE_PART_MAP
    assert len(FP_CORE_PART_MAP) == 13
    # Verify a few key entries
    assert FP_CORE_PART_MAP[8] == "zipper"
    assert FP_CORE_PART_MAP[5] == "pocket"
    assert FP_CORE_PART_MAP[1] == "collar"
    assert FP_CORE_PART_MAP[6] == "neckline"


def test_part_to_fp_ids_reverse_consistent() -> None:
    from fashion_vision.localization.fashionpedia_part_detector import (
        FP_CORE_PART_MAP,
        PART_TO_FP_IDS,
    )
    # Allowed aliases: part → fp_id where fp_id maps to a different part.
    _ALIASES = frozenset({"cuff"})
    all_parts = set(FP_CORE_PART_MAP.values())

    for part, fp_ids in PART_TO_FP_IDS.items():
        if part in _ALIASES:
            continue  # skip alias consistency check
        for fp_id in fp_ids:
            assert fp_id in FP_CORE_PART_MAP, (
                f"fp_id {fp_id} for part {part!r} not in FP_CORE_PART_MAP"
            )
            assert FP_CORE_PART_MAP[fp_id] == part, (
                f"fp_id {fp_id} maps to {FP_CORE_PART_MAP[fp_id]!r}, "
                f"not {part!r}"
            )


def test_new_parts_in_part_to_fp_ids() -> None:
    """lapel, epaulette, buckle are in FP core coverage."""
    from fashion_vision.localization.fashionpedia_part_detector import PART_TO_FP_IDS
    assert "lapel" in PART_TO_FP_IDS
    assert "epaulette" in PART_TO_FP_IDS
    assert "buckle" in PART_TO_FP_IDS


# ---------------------------------------------------------------------------
# detect() — class filtering, score sorting, edge cases
# ---------------------------------------------------------------------------


def _make_mock_results(cls_ids, scores, boxes=None):
    """Build a minimal mock ultralytics Results object."""
    if boxes is None:
        boxes = [[10, 10, 50, 50]] * len(cls_ids)
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(boxes, dtype=np.float32)
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array(scores, dtype=np.float32)
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array(cls_ids, dtype=np.float32)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    return [mock_result]


def _make_detector():
    """Create a FashionpediaPartDetector with a mocked YOLO model."""
    from fashion_vision.localization.fashionpedia_part_detector import FashionpediaPartDetector
    det = FashionpediaPartDetector.__new__(FashionpediaPartDetector)
    det._model_path = "mock.pt"
    det._device = "cpu"
    det.model = MagicMock()
    return det


def test_detect_filters_by_target_part() -> None:
    """Only detections matching target_part class IDs are returned."""
    det = _make_detector()
    # Mock returns 4 detections across different classes: zipper(8), pocket(5), collar(1), hood(0)
    det.model.return_value = _make_mock_results(
        cls_ids=[8, 5, 1, 0],
        scores=[0.9, 0.8, 0.7, 0.6],
    )
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Query zipper → should only get cls_id=8
    result = det.detect(img, "zipper", conf=0.25)
    assert len(result) == 1
    assert result[0]["class_id"] == 8
    assert result[0]["label"] == "zipper"
    assert result[0]["backend"] == "fashionpedia_yolo"


def test_detect_sorts_by_score_desc() -> None:
    """Detections are sorted by score descending."""
    det = _make_detector()
    # pocket: cls_id=5, multiple detections
    det.model.return_value = _make_mock_results(
        cls_ids=[5, 5, 5],
        scores=[0.6, 0.9, 0.3],
        boxes=[[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]],
    )
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = det.detect(img, "pocket", conf=0.25)
    assert len(result) == 3
    assert result[0]["score"] == pytest.approx(0.9)
    assert result[1]["score"] == pytest.approx(0.6)
    assert result[2]["score"] == pytest.approx(0.3)


def test_detect_empty_when_no_target_class() -> None:
    """Returns [] when YOLO detects things but none match target_part."""
    det = _make_detector()
    # Only hood (0) and collar (1) — no zipper
    det.model.return_value = _make_mock_results(
        cls_ids=[0, 1],
        scores=[0.9, 0.8],
    )
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = det.detect(img, "zipper", conf=0.25)
    assert result == []


def test_detect_respects_conf_via_yolo() -> None:
    """conf is forwarded to YOLO (which handles internal filtering).

    Our mock bypasses YOLO's internal NMS/conf filter, so a low-score
    detection still appears.  This test verifies the conf kwarg is
    forwarded correctly and the detection dict format is valid even at
    low scores — the real YOLO handles the actual thresholding.
    """
    det = _make_detector()
    det.model.return_value = _make_mock_results(
        cls_ids=[8], scores=[0.2],
    )
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = det.detect(img, "zipper", conf=0.99)
    # conf is forwarded; mock YOLO ignores it → detection survives.
    # In production, YOLO would filter internally.
    assert len(result) == 1
    # Verify conf was forwarded to the model call.
    assert det.model.call_args[1].get("conf") == 0.99


def test_detect_empty_when_yolo_returns_none() -> None:
    """Returns [] when YOLO has no detections at all (boxes is None)."""
    det = _make_detector()
    mock_result = MagicMock()
    mock_result.boxes = None
    det.model.return_value = [mock_result]
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = det.detect(img, "zipper")
    assert result == []


def test_detect_unknown_part_returns_empty() -> None:
    """Returns [] immediately for parts not in PART_TO_FP_IDS."""
    det = _make_detector()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # button is NOT in FP core parts
    result = det.detect(img, "button")
    assert result == []
    # model should not have been called
    det.model.assert_not_called()


def test_detect_none_part_returns_empty() -> None:
    """Returns [] for None part."""
    det = _make_detector()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = det.detect(img, None)
    assert result == []


# ---------------------------------------------------------------------------
# Mask gating
# ---------------------------------------------------------------------------


def test_mask_gating_applied_when_mask_provided() -> None:
    """When garment_mask is provided, non-garment pixels are filled grey."""
    det = _make_detector()
    det.model.return_value = _make_mock_results(
        cls_ids=[8], scores=[0.9],
    )
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1  # center window is garment

    result = det.detect(img, "zipper", garment_mask=mask, conf=0.25)

    # Verify the model received a mask-gated image (and original is unchanged).
    assert result  # detection returned
    assert len(result) == 1
    # Original image should be unchanged (mask_gate copies).
    assert np.all(img == 255)

    # The image passed to model should have grey (128) outside the mask region.
    called_img = det.model.call_args[0][0]
    assert called_img.shape == img.shape
    # Corners should be grey (mask=0 there)
    assert called_img[5, 5, 0] == 128
    assert called_img[5, 5, 1] == 128
    assert called_img[5, 5, 2] == 128
    # Center should be white (mask=1 there)
    assert called_img[50, 50, 0] == 255


def test_mask_gating_mismatched_shape_resized() -> None:
    """Mask is resized to match image when shapes differ."""
    cv2 = pytest.importorskip("cv2", reason="cv2 not available in test env")
    det = _make_detector()
    det.model.return_value = _make_mock_results(
        cls_ids=[8], scores=[0.9],
    )
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # Mask at different resolution
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 1

    result = det.detect(img, "zipper", garment_mask=mask, conf=0.25)
    assert len(result) == 1
    called_img = det.model.call_args[0][0]
    assert called_img.shape == img.shape


# ---------------------------------------------------------------------------
# Backend label in detection dicts
# ---------------------------------------------------------------------------


def test_detection_dict_has_backend_key() -> None:
    """Every returned detection dict includes backend='fashionpedia_yolo'."""
    det = _make_detector()
    det.model.return_value = _make_mock_results(
        cls_ids=[8, 5], scores=[0.9, 0.8],
    )
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = det.detect(img, "zipper", conf=0.25)
    for d in result:
        assert d["backend"] == "fashionpedia_yolo"
        assert "bbox_xyxy" in d
        assert "score" in d
        assert "label" in d
        assert "class_id" in d


# ---------------------------------------------------------------------------
# Mask shape normalization — _mask_gate accepts H×W, H×W×1, H×W×3
# ---------------------------------------------------------------------------


def test_mask_gate_accepts_hw_mask() -> None:
    """_mask_gate works with 2-D H×W mask."""
    from fashion_vision.localization.fashionpedia_part_detector import FashionpediaPartDetector
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    gated = FashionpediaPartDetector._mask_gate(img, mask)
    assert gated.shape == img.shape
    assert gated[5, 5, 0] == 128  # outside mask → grey


def test_mask_gate_accepts_hw1_mask() -> None:
    """_mask_gate squeezes H×W×1 to H×W before indexing."""
    from fashion_vision.localization.fashionpedia_part_detector import FashionpediaPartDetector
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 100, 1), dtype=np.uint8)
    mask[20:80, 20:80, 0] = 1
    gated = FashionpediaPartDetector._mask_gate(img, mask)
    assert gated.shape == img.shape
    assert gated[5, 5, 0] == 128
    assert gated[50, 50, 0] == 255
