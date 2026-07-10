"""Unit tests for tools/eval/benchmark_all_attribute_tasks.py.

Tests cover pure formatting and aggregation helpers only — no checkpoints
or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.eval.benchmark_all_attribute_tasks import (
    _COARSE_CLASS_TO_TASKS,
    _PRD_LATENCY_TARGET_MS,
    compute_sequential_totals,
    format_latency_report_md,
)


# ---------------------------------------------------------------------------
# compute_sequential_totals
# ---------------------------------------------------------------------------


def _fake_per_task(tasks: list[str], mean_ms: float) -> dict:
    return {
        t: {
            "checkpoint_exists": True,
            "cpu": {"mean": mean_ms, "p50": mean_ms, "p95": mean_ms + 1.0,
                    "min": mean_ms - 0.5, "max": mean_ms + 2.0,
                    "meets_prd_target": mean_ms <= _PRD_LATENCY_TARGET_MS},
        }
        for t in tasks
    }


class TestComputeSequentialTotals:
    def test_top_four_tasks_summed(self) -> None:
        # top: 4 tasks, each 3ms → total 12ms
        per_task = _fake_per_task(
            ["neckline_design", "collar_design", "neck_design", "sleeve_length"],
            mean_ms=3.0,
        )
        totals = compute_sequential_totals(per_task, _COARSE_CLASS_TO_TASKS, "cpu")
        assert totals["top"]["total_mean_ms"] == pytest.approx(12.0)
        assert totals["top"]["tasks_included"] == 4

    def test_pants_single_task(self) -> None:
        per_task = _fake_per_task(["pant_length"], mean_ms=4.0)
        totals = compute_sequential_totals(per_task, _COARSE_CLASS_TO_TASKS, "cpu")
        assert totals["pants"]["total_mean_ms"] == pytest.approx(4.0)
        assert totals["pants"]["tasks_included"] == 1

    def test_missing_task_excluded_from_sum(self) -> None:
        # Only provide sleeve_length for top (3 of 4 expected)
        per_task = _fake_per_task(["sleeve_length"], mean_ms=5.0)
        totals = compute_sequential_totals(per_task, _COARSE_CLASS_TO_TASKS, "cpu")
        assert totals["top"]["tasks_included"] == 1
        assert totals["top"]["tasks_expected"] == 4
        assert totals["top"]["total_mean_ms"] == pytest.approx(5.0)

    def test_meets_prd_when_total_under_20ms(self) -> None:
        per_task = _fake_per_task(["pant_length"], mean_ms=4.0)
        totals = compute_sequential_totals(per_task, _COARSE_CLASS_TO_TASKS, "cpu")
        assert totals["pants"]["meets_prd_target"] is True

    def test_fails_prd_when_total_over_20ms(self) -> None:
        # top has 4 tasks; each 6ms → total 24ms > 20ms
        per_task = _fake_per_task(
            ["neckline_design", "collar_design", "neck_design", "sleeve_length"],
            mean_ms=6.0,
        )
        totals = compute_sequential_totals(per_task, _COARSE_CLASS_TO_TASKS, "cpu")
        assert totals["top"]["meets_prd_target"] is False

    def test_empty_per_task_gives_zero_totals(self) -> None:
        totals = compute_sequential_totals({}, _COARSE_CLASS_TO_TASKS, "cpu")
        for cls_data in totals.values():
            assert cls_data["total_mean_ms"] == 0.0
            assert cls_data["tasks_included"] == 0


# ---------------------------------------------------------------------------
# format_latency_report_md
# ---------------------------------------------------------------------------


def _minimal_report() -> dict:
    return {
        "meta": {
            "timestamp": "2026-06-17T12:00:00",
            "inference_config": "configs/attribute_inference.yaml",
            "warmup": 20,
            "runs": 200,
            "devices_tested": ["cpu"],
            "prd_latency_target_ms": 20.0,
        },
        "per_task": {
            "collar_design": {
                "checkpoint": "outputs/p2_collar_design_resnet18_seed2/best.pt",
                "checkpoint_exists": True,
                "arch": "resnet18",
                "img_size": 224,
                "cpu": {
                    "mean": 5.1, "median": 5.0, "p50": 5.0, "p95": 6.0,
                    "p99": 7.0, "min": 4.5, "max": 9.0,
                    "meets_prd_target": True,
                },
            },
            "neckline_design": {
                "checkpoint": "outputs/p2_neckline_design_resnet18_seed2/best.pt",
                "checkpoint_exists": False,
                "arch": "resnet18",
                "img_size": 224,
            },
        },
        "sequential_totals": {
            "cpu": {
                "top": {
                    "tasks_included": 1,
                    "tasks_expected": 4,
                    "total_mean_ms": 5.1,
                    "total_p95_ms": 6.0,
                    "meets_prd_target": True,
                }
            }
        },
    }


class TestFormatLatencyReportMd:
    def test_output_is_string(self) -> None:
        md = format_latency_report_md(_minimal_report())
        assert isinstance(md, str) and len(md) > 0

    def test_prd_pass_row_present(self) -> None:
        md = format_latency_report_md(_minimal_report())
        assert "PASS" in md

    def test_missing_checkpoint_marked(self) -> None:
        md = format_latency_report_md(_minimal_report())
        assert "missing" in md.lower()

    def test_sequential_totals_section_present(self) -> None:
        md = format_latency_report_md(_minimal_report())
        assert "Sequential Total" in md or "sequential" in md.lower()

    def test_task_names_appear(self) -> None:
        md = format_latency_report_md(_minimal_report())
        assert "collar_design" in md
        assert "neckline_design" in md

    def test_all_fail_shows_fail(self) -> None:
        r = _minimal_report()
        # Set mean above PRD target
        r["per_task"]["collar_design"]["cpu"]["mean"] = 25.0
        r["per_task"]["collar_design"]["cpu"]["meets_prd_target"] = False
        md = format_latency_report_md(r)
        assert "FAIL" in md

    def test_timestamp_in_output(self) -> None:
        md = format_latency_report_md(_minimal_report())
        assert "2026-06-17" in md

    def test_prd_constant_is_20ms(self) -> None:
        assert _PRD_LATENCY_TARGET_MS == pytest.approx(20.0)
