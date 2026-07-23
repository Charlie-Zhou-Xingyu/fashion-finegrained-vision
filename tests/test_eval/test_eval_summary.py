"""P1.0a — Summary script tests (spec section 9.3).

Covers: reading a report, failed case ids in output, clear errors on
missing/invalid files, and no secret leakage in the summary output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import load_summarizer  # noqa: E402

summarizer = load_summarizer()


def _fake_report() -> dict:
    return {
        "schema_version": "1.0.0",
        "generated_at": "2026-07-15T00:00:00Z",
        "mode": "report_only",
        "task_filter": "all",
        "summary": {"total": 3, "passed": 1, "failed": 1, "skipped": 1, "pass_rate": 0.5},
        "task_summaries": {
            "intent": {"total": 2, "passed": 1, "failed": 1, "skipped": 0, "pass_rate": 0.5},
            "vision_attribute": {"total": 1, "passed": 0, "failed": 0, "skipped": 1,
                                 "pass_rate": None},
        },
        "thresholds": {"intent": 0.8},
        "threshold_passed": False,
        "threshold_detail": {"intent": {"threshold": 0.8, "pass_rate": 0.5, "passed": False}},
        "cases": [
            {"id": "intent_ok", "task_type": "intent", "passed": True, "skipped": False,
             "status_code": 200, "checks": [], "warnings": [], "error": None},
            {"id": "intent_bad", "task_type": "intent", "passed": False, "skipped": False,
             "status_code": 200,
             "checks": [{"name": "primary_intent", "passed": False,
                         "expected": "a", "actual": "b", "message": ""}],
             "warnings": [], "error": None},
            {"id": "vision_attr_001", "task_type": "vision_attribute", "passed": False,
             "skipped": True, "status_code": None, "checks": [], "warnings": [],
             "error": None},
        ],
    }


# ── 31. Summary reads report ───────────────────────────────────────────────────


def test_summary_reads_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_fake_report(), ensure_ascii=False), encoding="utf-8")
    report = summarizer.load_report(path)
    lines = summarizer.summarize(report)
    text = "\n".join(lines)
    assert "schema_version : 1.0.0" in text
    assert "mode           : report_only" in text
    assert "pass_rate=0.5" in text


# ── 32. Failed case ids present ────────────────────────────────────────────────


def test_summary_contains_failed_case_ids(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_fake_report(), ensure_ascii=False), encoding="utf-8")
    text = "\n".join(summarizer.summarize(summarizer.load_report(path)))
    assert "intent_bad" in text
    assert "primary_intent" in text  # top failed checks section
    assert "vision_attr_001" in text  # skipped section
    assert "threshold_passed : False" in text


# ── 33. Missing file → clear error ─────────────────────────────────────────────


def test_summary_missing_file_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        summarizer.load_report(tmp_path / "nope.json")


# ── 34. Invalid JSON → clear error ─────────────────────────────────────────────


def test_summary_invalid_json_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        summarizer.load_report(path)


# ── 35. No secrets in summary output ───────────────────────────────────────────


def test_summary_output_contains_no_secret(tmp_path):
    report = _fake_report()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    text = "\n".join(summarizer.summarize(summarizer.load_report(path)))
    assert "SECRET_BASE64_DO_NOT_LEAK" not in text
    assert "image_bytes" not in text
