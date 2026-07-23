"""P1.0a — Eval dataset & case schema tests (spec section 9.1).

Covers: file existence, per-line JSON parsing, required fields, enum
validity, type constraints, global id uniqueness, minimum case counts,
placeholder-only vision URIs, and no-secret rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import DATASETS_DIR, SCHEMA_PATH, load_runner  # noqa: E402

runner = load_runner()

MIN_CASE_COUNTS = {
    "intent": 30,
    "attribute_qa": 25,
    "rag_retrieval": 25,
    "mm_qa": 20,
    "content_generation": 20,
    "vision_attribute": 10,
}


def _all_lines():
    """Yield (task, filename, lineno, raw_line) for every dataset line."""
    for task, filename in runner.TASK_FILES.items():
        path = DATASETS_DIR / filename
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if line:
                    yield task, filename, lineno, line


def _all_cases():
    by_task = runner.load_all_datasets(list(runner.TASK_FILES.keys()))
    for task, cases in by_task.items():
        for case in cases:
            yield task, case


# ── 1. Files exist ─────────────────────────────────────────────────────────────


def test_schema_file_exists():
    assert SCHEMA_PATH.exists()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(schema["required"]) == set(runner.REQUIRED_FIELDS)


@pytest.mark.parametrize("task", list(runner.TASK_FILES.keys()))
def test_dataset_file_exists(task):
    assert (DATASETS_DIR / runner.TASK_FILES[task]).exists()


# ── 2. Every line parses as JSON ───────────────────────────────────────────────


def test_every_line_is_valid_json():
    for task, filename, lineno, line in _all_lines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{filename}:{lineno} invalid JSON: {exc}")
        assert isinstance(obj, dict), f"{filename}:{lineno} not an object"


# ── 3–9. Field-level schema constraints ────────────────────────────────────────


def test_required_fields_present():
    for task, case in _all_cases():
        for field in runner.REQUIRED_FIELDS:
            assert field in case, f"{case.get('id')} missing {field}"


def test_task_type_valid_and_matches_file():
    for task, case in _all_cases():
        assert case["task_type"] in runner.TASK_TYPES
        assert case["task_type"] == task


def test_difficulty_valid():
    for _, case in _all_cases():
        assert case["difficulty"] in runner.VALID_DIFFICULTY, case["id"]


def test_review_status_valid():
    for _, case in _all_cases():
        assert case["review_status"] in runner.VALID_REVIEW_STATUS, case["id"]


def test_object_fields_are_objects():
    for _, case in _all_cases():
        for field in ("input", "expected", "source_ref"):
            assert isinstance(case[field], dict), f"{case['id']}.{field}"


def test_tags_is_list():
    for _, case in _all_cases():
        assert isinstance(case["tags"], list), case["id"]


def test_notes_is_string():
    for _, case in _all_cases():
        assert isinstance(case["notes"], str), case["id"]


# ── 10. Global id uniqueness ───────────────────────────────────────────────────


def test_case_ids_globally_unique():
    ids = [case["id"] for _, case in _all_cases()]
    assert len(ids) == len(set(ids))


def test_duplicate_id_detected_by_loader(tmp_path):
    """The loader must reject duplicate ids across datasets."""
    # Build two tiny datasets that share an id.
    base = {
        "task_type": "intent", "input": {"query": "x"}, "expected": {},
        "tags": [], "difficulty": "easy", "review_status": "seed",
        "source_ref": {}, "notes": "",
    }
    a = dict(base, id="dup_001")
    b = dict(base, id="dup_001", task_type="mm_qa")
    (tmp_path / runner.TASK_FILES["intent"]).write_text(
        json.dumps(a, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / runner.TASK_FILES["mm_qa"]).write_text(
        json.dumps(b, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(runner.EvalCaseValidationError, match="duplicate"):
        runner.load_all_datasets(["intent", "mm_qa"], tmp_path)


# ── 11. Minimum case counts ────────────────────────────────────────────────────


@pytest.mark.parametrize("task,minimum", sorted(MIN_CASE_COUNTS.items()))
def test_min_case_counts(task, minimum):
    cases = runner.load_dataset(task)
    assert len(cases) >= minimum, f"{task}: {len(cases)} < {minimum}"


def test_total_case_count_at_least_130():
    total = sum(len(runner.load_dataset(t)) for t in runner.TASK_FILES)
    assert total >= 130


# ── 12. Vision manifest placeholder-only URIs ──────────────────────────────────


def test_vision_image_uris_are_placeholder_only():
    for case in runner.load_dataset("vision_attribute"):
        uri = case["input"]["image_uri"]
        assert uri.startswith("placeholder://"), case["id"]


def test_vision_manifest_rejects_non_placeholder(tmp_path):
    bad = {
        "id": "vis_bad", "task_type": "vision_attribute",
        "input": {"image_uri": "http://example.com/x.jpg"},
        "expected": {}, "tags": [], "difficulty": "easy",
        "review_status": "seed", "source_ref": {}, "notes": "",
    }
    (tmp_path / runner.TASK_FILES["vision_attribute"]).write_text(
        json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(runner.EvalCaseValidationError, match="placeholder"):
        runner.load_dataset("vision_attribute", tmp_path)


# ── 13–14. No secrets in datasets ──────────────────────────────────────────────


def test_datasets_do_not_contain_secret_marker():
    for _, filename, lineno, line in _all_lines():
        assert "SECRET_BASE64_DO_NOT_LEAK" not in line, f"{filename}:{lineno}"


def test_datasets_do_not_contain_obvious_token_fields():
    suspicious = ("api_key", "apikey", "access_token", "authorization", "password")
    for _, case in _all_cases():
        raw = json.dumps(case, ensure_ascii=False).lower()
        for token in suspicious:
            assert f'"{token}"' not in raw, f"{case['id']} contains field {token}"


def test_validator_rejects_missing_field():
    bad = {"id": "x", "task_type": "intent"}
    issues = runner.validate_case(bad, expected_task_type="intent")
    assert issues and any("missing required field" in i for i in issues)


def test_validator_rejects_bad_enums():
    base = {
        "id": "x", "task_type": "intent", "input": {}, "expected": {},
        "tags": [], "difficulty": "impossible", "review_status": "nope",
        "source_ref": {}, "notes": "",
    }
    issues = runner.validate_case(base, expected_task_type="intent")
    assert any("difficulty" in i for i in issues)
    assert any("review_status" in i for i in issues)
