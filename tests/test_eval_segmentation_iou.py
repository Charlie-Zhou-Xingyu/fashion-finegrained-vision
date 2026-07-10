"""Unit tests for tools/eval/eval_segmentation_iou.py.

Tests cover the pure helper functions only — no real mask files, GT annotations,
model weights, or YOLO/SAM-HQ inference are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.eval.eval_segmentation_iou import (
    bbox_iou,
    compute_aggregate_stats,
    format_iou_report_md,
    match_predictions_to_gt,
)


# ---------------------------------------------------------------------------
# bbox_iou
# ---------------------------------------------------------------------------


class TestBboxIou:
    def test_identical_boxes_returns_one(self) -> None:
        assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_non_overlapping_returns_zero(self) -> None:
        assert bbox_iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_half_overlap(self) -> None:
        # A=[0,0,10,10] area=100, B=[5,0,15,10] area=100, inter=50, union=150
        assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)

    def test_contained_box_less_than_one(self) -> None:
        # A contains B entirely
        iou = bbox_iou([0, 0, 10, 10], [2, 2, 8, 8])
        # inter = 36, union = 100 + 36 - 36 = 100
        assert iou == pytest.approx(36 / 100)

    def test_zero_area_box_returns_zero(self) -> None:
        assert bbox_iou([5, 5, 5, 5], [0, 0, 10, 10]) == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        a, b = [0, 0, 8, 8], [4, 4, 12, 12]
        assert bbox_iou(a, b) == pytest.approx(bbox_iou(b, a))


# ---------------------------------------------------------------------------
# match_predictions_to_gt
# ---------------------------------------------------------------------------


def _make_pred(cat: str, bbox: list[float]) -> dict:
    return {"target_category": cat, "bbox_xyxy": bbox, "det_id": 0,
            "mask_path": "", "class_name": cat}


def _make_gt(cat: str, bbox: list[float]) -> dict:
    return {"target_category": cat, "bbox": bbox,
            "gt_category_id": 1, "gt_mask": np.zeros((10, 10), dtype=np.uint8)}


class TestMatchPredictionsToGt:
    def test_exact_match(self) -> None:
        preds = [_make_pred("top", [0, 0, 10, 10])]
        gts = [_make_gt("top", [0, 0, 10, 10])]
        matches = match_predictions_to_gt(preds, gts, bbox_iou_threshold=0.5)
        assert len(matches) == 1

    def test_no_match_wrong_category(self) -> None:
        preds = [_make_pred("top", [0, 0, 10, 10])]
        gts = [_make_gt("pants", [0, 0, 10, 10])]
        matches = match_predictions_to_gt(preds, gts, bbox_iou_threshold=0.5)
        assert len(matches) == 0

    def test_no_match_bbox_iou_below_threshold(self) -> None:
        # Barely overlapping — IoU < 0.5
        preds = [_make_pred("top", [0, 0, 10, 10])]
        gts = [_make_gt("top", [8, 8, 20, 20])]  # small overlap
        matches = match_predictions_to_gt(preds, gts, bbox_iou_threshold=0.5)
        assert len(matches) == 0

    def test_greedy_highest_iou_wins(self) -> None:
        # Two GT boxes, pred matches the second better.
        preds = [_make_pred("top", [5, 0, 15, 10])]
        gt_far = _make_gt("top", [0, 0, 6, 10])   # small overlap with pred
        gt_near = _make_gt("top", [5, 0, 15, 10])  # perfect match
        matches = match_predictions_to_gt(preds, [gt_far, gt_near], bbox_iou_threshold=0.5)
        assert len(matches) == 1
        # Should be matched to gt_near (higher IoU)
        _, matched_gt, _ = matches[0]
        assert matched_gt is gt_near

    def test_each_gt_matched_at_most_once(self) -> None:
        # Two preds, one GT — only one match should be produced.
        preds = [
            _make_pred("top", [0, 0, 10, 10]),
            _make_pred("top", [1, 1, 9, 9]),
        ]
        gts = [_make_gt("top", [0, 0, 10, 10])]
        matches = match_predictions_to_gt(preds, gts, bbox_iou_threshold=0.5)
        assert len(matches) == 1

    def test_empty_preds_returns_empty(self) -> None:
        gts = [_make_gt("top", [0, 0, 10, 10])]
        assert match_predictions_to_gt([], gts) == []

    def test_empty_gts_returns_empty(self) -> None:
        preds = [_make_pred("top", [0, 0, 10, 10])]
        assert match_predictions_to_gt(preds, []) == []


# ---------------------------------------------------------------------------
# compute_aggregate_stats
# ---------------------------------------------------------------------------


class TestComputeAggregateStats:
    def test_empty_returns_none_values(self) -> None:
        stats = compute_aggregate_stats([])
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["pct_above_prd_target"] is None

    def test_single_value(self) -> None:
        stats = compute_aggregate_stats([0.9])
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(0.9)
        assert stats["pct_above_prd_target"] == pytest.approx(100.0)

    def test_all_below_target(self) -> None:
        stats = compute_aggregate_stats([0.5, 0.6, 0.7])
        assert stats["pct_above_prd_target"] == pytest.approx(0.0)

    def test_half_above_target(self) -> None:
        stats = compute_aggregate_stats([0.80, 0.80, 0.90, 0.90])
        # target = 0.85; 2 of 4 are ≥ 0.85
        assert stats["pct_above_prd_target"] == pytest.approx(50.0)

    def test_prd_target_boundary(self) -> None:
        stats = compute_aggregate_stats([0.85])
        assert stats["pct_above_prd_target"] == pytest.approx(100.0)

    def test_mean_and_median(self) -> None:
        stats = compute_aggregate_stats([0.6, 0.8, 1.0])
        assert stats["mean"] == pytest.approx(0.8)
        assert stats["median"] == pytest.approx(0.8)
        assert stats["min"] == pytest.approx(0.6)
        assert stats["max"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_iou_report_md
# ---------------------------------------------------------------------------


class TestFormatIouReportMd:
    def _minimal_report(self) -> dict:
        return {
            "meta": {
                "timestamp": "2026-06-17T12:00:00",
                "segmentation_json": "outputs/test/seg.json",
                "ann_dir": "D:/data/annos",
                "bbox_iou_threshold": 0.5,
                "images_evaluated": 2,
                "gt_instances_total": 3,
                "pred_segments_total": 3,
                "matched_pairs": 2,
                "unmatched_gt": 1,
                "unmatched_preds": 1,
            },
            "overall": {
                "count": 2,
                "mean": 0.82,
                "median": 0.82,
                "min": 0.75,
                "max": 0.89,
                "pct_above_prd_target": 50.0,
                "prd_iou_target": 0.85,
            },
            "per_class": {
                "top": {
                    "count": 2, "mean": 0.82, "median": 0.82,
                    "min": 0.75, "max": 0.89,
                    "pct_above_prd_target": 50.0, "prd_iou_target": 0.85,
                }
            },
            "instances": [],
        }

    def test_output_is_nonempty_string(self) -> None:
        md = format_iou_report_md(self._minimal_report())
        assert isinstance(md, str) and len(md) > 0

    def test_contains_prd_pass_fail(self) -> None:
        md = format_iou_report_md(self._minimal_report())
        # mean=0.82 < 0.85 → FAIL
        assert "FAIL" in md

    def test_passing_report_shows_pass(self) -> None:
        r = self._minimal_report()
        r["overall"]["mean"] = 0.90
        md = format_iou_report_md(r)
        assert "PASS" in md

    def test_empty_matches_no_crash(self) -> None:
        r = self._minimal_report()
        r["overall"] = {
            "count": 0, "mean": None, "median": None,
            "min": None, "max": None,
            "pct_above_prd_target": None, "prd_iou_target": 0.85,
        }
        md = format_iou_report_md(r)
        assert "no matched pairs" in md.lower() or "UNKNOWN" in md

    def test_per_class_section_present(self) -> None:
        md = format_iou_report_md(self._minimal_report())
        assert "Per-class" in md
        assert "top" in md

    def test_metadata_in_output(self) -> None:
        md = format_iou_report_md(self._minimal_report())
        assert "2026-06-17" in md
        assert "2" in md  # images_evaluated
