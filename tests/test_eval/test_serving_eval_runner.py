"""P1.0a — Eval runner tests (spec section 9.2).

Covers: dataset loading, per-task runs, vision skip, report structure,
checker behaviors (request_id echo, no data.warnings, forbidden substrings,
required hit ids, llm_used=false, content_blocks, no secret leak), and
report-only vs fail-on-threshold semantics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import load_runner  # noqa: E402

from inference.serving.app import app  # noqa: E402

runner = load_runner()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ── 15. All datasets load ──────────────────────────────────────────────────────


def test_runner_loads_all_datasets():
    by_task = runner.load_all_datasets(list(runner.TASK_FILES.keys()))
    assert set(by_task.keys()) == set(runner.TASK_FILES.keys())
    assert all(len(cases) > 0 for cases in by_task.values())


# ── 16–18. Per-task runs ───────────────────────────────────────────────────────


def test_runner_task_intent(client):
    report = runner.run_eval(task_filter="intent", client=client)
    assert report["task_filter"] == "intent"
    assert report["summary"]["total"] >= 30
    assert report["summary"]["skipped"] == 0


def test_runner_task_content_generation(client):
    report = runner.run_eval(task_filter="content_generation", client=client)
    assert report["summary"]["total"] >= 20
    ts = report["task_summaries"]["content_generation"]
    assert ts["pass_rate"] is not None


def test_runner_task_vision_attribute_all_skipped(client):
    report = runner.run_eval(task_filter="vision_attribute", client=client)
    s = report["summary"]
    assert s["total"] >= 10
    assert s["skipped"] == s["total"]
    assert s["pass_rate"] is None
    assert all(c["skipped"] for c in report["cases"])
    assert all(c["status_code"] is None for c in report["cases"])


# ── 19–21. Report structure ────────────────────────────────────────────────────


def test_report_is_json_safe(client):
    report = runner.run_eval(task_filter="intent", client=client)
    dumped = json.dumps(report, ensure_ascii=False)
    assert isinstance(json.loads(dumped), dict)


def test_report_top_level_keys(client):
    report = runner.run_eval(task_filter="intent", client=client)
    for key in ("schema_version", "generated_at", "mode", "task_filter",
                "summary", "task_summaries", "thresholds", "threshold_passed",
                "cases"):
        assert key in report
    assert report["mode"] == "report_only"


def test_case_results_contain_checks(client):
    report = runner.run_eval(task_filter="intent", client=client)
    non_skipped = [c for c in report["cases"] if not c["skipped"]]
    assert non_skipped
    for c in non_skipped:
        assert isinstance(c["checks"], list) and c["checks"]
        for ch in c["checks"]:
            assert {"name", "passed", "expected", "actual", "message"} <= set(ch.keys())


# ── 22. request_id echo checker ────────────────────────────────────────────────


def test_request_id_echo_checker():
    ok = runner.check_request_id_echo("case_x", {"request_id": "eval_case_x"})
    assert ok["passed"]
    bad = runner.check_request_id_echo("case_x", {"request_id": "req_other"})
    assert not bad["passed"]


def test_request_id_echo_end_to_end(client):
    report = runner.run_eval(task_filter="rag_retrieval", client=client)
    for c in report["cases"]:
        echo = [ch for ch in c["checks"] if ch["name"] == "request_id_echo"]
        assert echo and echo[0]["passed"], c["id"]


# ── 23. no data.warnings checker ───────────────────────────────────────────────


def test_no_data_warnings_checker():
    ok = runner.check_no_data_warnings({"data": {"answer": "x"}})
    assert ok["passed"]
    bad = runner.check_no_data_warnings({"data": {"warnings": []}})
    assert not bad["passed"]


# ── 24. forbidden_substrings checker ───────────────────────────────────────────


def test_forbidden_substrings_checker():
    body = {"data": {"answer": "包含抗菌宣称"}, "warnings": []}
    result = runner.check_forbidden_substrings({"forbidden_substrings": ["抗菌"]}, body)
    assert not result["passed"]
    clean = runner.check_forbidden_substrings({"forbidden_substrings": ["抗菌"]},
                                              {"data": {"answer": "普通棉"}})
    assert clean["passed"]


def test_forbidden_substrings_ignores_blocked_claims_field():
    body = {"data": {"answer": "ok", "blocked_claims": [{"field": "抗菌", "reason": "x"}]}}
    result = runner.check_forbidden_substrings({"forbidden_substrings": ["抗菌"]}, body)
    assert result["passed"], "redacted blocked_claims must not cause false positives"


# ── 25. rag required_hit_ids checker ───────────────────────────────────────────


def test_required_hit_ids_checker():
    data = {"hits": [{"id": "fiber_term_001"}, {"id": "cotton_001"}]}
    ok = runner.check_required_hit_ids({"required_hit_ids": ["fiber_term_001"]}, data)
    assert ok["passed"]
    bad = runner.check_required_hit_ids({"required_hit_ids": ["missing_id"]}, data)
    assert not bad["passed"]


def test_required_hit_ids_end_to_end(client):
    resp = client.post("/v1/rag/retrieve", json={"query": "纤维"},
                       headers={"X-Request-ID": "eval_unit_rag"})
    assert resp.status_code == 200
    hits = resp.json()["data"]["hits"]
    assert any(h["id"] == "fiber_term_001" for h in hits)


# ── 26. llm_used=false checker ─────────────────────────────────────────────────


def test_llm_used_false_checker():
    ok = runner.check_llm_used_false({"llm_used": False}, {"meta": {"llm_used": False}})
    assert ok["passed"]
    bad = runner.check_llm_used_false({"llm_used": False}, {"meta": {"llm_used": True}})
    assert not bad["passed"]


# ── 27. content_blocks_present checker ─────────────────────────────────────────


def test_content_blocks_present_checker():
    ok = runner.check_content_blocks_present(
        {"content_blocks_present": True}, {"content_blocks": [{"type": "title"}]})
    assert ok["passed"]
    bad = runner.check_content_blocks_present(
        {"content_blocks_present": True}, {"content_blocks": []})
    assert not bad["passed"]
    ok_empty = runner.check_content_blocks_present(
        {"content_blocks_present": False}, {"content_blocks": []})
    assert ok_empty["passed"]


# ── 28. no_secret_leak checker ─────────────────────────────────────────────────


def test_no_secret_leak_checker():
    case_input = {"image_bytes": "FAKE_BYTES_XYZ"}
    leaked = runner.check_no_secret_leak(case_input, {"data": {"answer": "FAKE_BYTES_XYZ"}})
    assert not leaked["passed"]
    clean = runner.check_no_secret_leak(case_input, {"data": {"answer": "ok"}})
    assert clean["passed"]


def test_report_redacts_image_bytes(client):
    """The image_bytes payload must never appear in the report JSON."""
    report = runner.run_eval(task_filter="mm_qa", client=client)
    dumped = json.dumps(report, ensure_ascii=False)
    assert "EVAL_FAKE_IMAGE_BYTES_DO_NOT_ECHO_p1a" not in dumped


# ── 29–30. Thresholds ──────────────────────────────────────────────────────────


def test_thresholds_default_report_only(client):
    """run_eval never raises on low pass rates — mode stays report_only."""
    report = runner.run_eval(task_filter="intent", client=client)
    assert report["mode"] == "report_only"
    assert isinstance(report["threshold_passed"], bool)


def test_evaluate_thresholds_helper_failure():
    summaries = {"intent": {"total": 10, "passed": 1, "failed": 9,
                            "skipped": 0, "pass_rate": 0.1}}
    ok, detail = runner.evaluate_thresholds(summaries)
    assert not ok
    assert detail["intent"]["passed"] is False


def test_evaluate_thresholds_helper_excludes_skipped():
    summaries = {"vision_attribute": {"total": 10, "passed": 0, "failed": 0,
                                      "skipped": 10, "pass_rate": None}}
    ok, detail = runner.evaluate_thresholds(summaries)
    assert ok
    assert detail["vision_attribute"]["passed"] is None


def test_cli_fail_on_threshold_exit_code(tmp_path, monkeypatch):
    """--fail-on-threshold returns non-zero when a task is under threshold."""
    fake_report = {
        "schema_version": "1.0.0", "generated_at": "x", "mode": "report_only",
        "task_filter": "intent",
        "summary": {"total": 10, "passed": 1, "failed": 9, "skipped": 0, "pass_rate": 0.1},
        "task_summaries": {"intent": {"total": 10, "passed": 1, "failed": 9,
                                      "skipped": 0, "pass_rate": 0.1}},
        "thresholds": dict(runner.DEFAULT_THRESHOLDS),
        "threshold_passed": False,
        "threshold_detail": {"intent": {"threshold": 0.8, "pass_rate": 0.1, "passed": False}},
        "cases": [],
    }
    monkeypatch.setattr(runner, "run_eval", lambda task_filter="all": fake_report)
    out = tmp_path / "r.json"
    rc = runner.main(["--task", "intent", "--output", str(out), "--fail-on-threshold"])
    assert rc == 1
    rc_report_only = runner.main(["--task", "intent", "--output", str(out)])
    assert rc_report_only == 0
