"""P0a.4.0 — Knowledge base schema validation tests.

Validates:
    - YAML parseability
    - Top-level structure (version, locale, source_policy, documents)
    - Every document has all required fields
    - All documents have review_status == "manual_review_required"
    - All documents have source_ref with document_title and page_start
    - Content does not contain banned absolute-claim phrases
    - IDs are globally unique
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_PATH = _PROJECT_ROOT / "configs" / "knowledge_base.yaml"

REQUIRED_DOC_FIELDS = frozenset({
    "id", "category", "term", "zh_term", "aliases", "title", "content",
    "allowed_usage", "risk_level", "source", "source_type", "source_url",
    "source_ref", "review_status", "reviewed_by", "last_reviewed_at",
    "version", "tags",
})

BANNED_CONTENT_WORDS = [
    "保证", "绝对", "一定", "100%", "完全",
    "最环保", "最可持续", "官方认证为最佳",
]

ALLOWED_CATEGORIES = frozenset({
    "fabric", "material", "fiber", "sustainability", "supply_chain", "term",
})

ALLOWED_RISK_LEVELS = frozenset({"low", "medium", "high"})

ALLOWED_REVIEW_STATUSES = frozenset({
    "seed_unreviewed", "manual_review_required", "reviewed", "deprecated",
})


@pytest.fixture(scope="module")
def kb_data():
    with open(_KB_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Top-level structure ────────────────────────────────────────────────────────


def test_kb_parses(kb_data):
    assert isinstance(kb_data, dict)


def test_kb_has_version(kb_data):
    assert "version" in kb_data
    assert kb_data["version"] == "1.0.0"


def test_kb_has_locale(kb_data):
    assert "locale" in kb_data


def test_kb_has_source_policy(kb_data):
    assert "source_policy" in kb_data
    assert "default_review_status" in kb_data["source_policy"]


def test_kb_has_documents(kb_data):
    assert "documents" in kb_data
    assert isinstance(kb_data["documents"], list)


def test_kb_documents_non_empty(kb_data):
    docs = kb_data["documents"]
    assert len(docs) > 0, "Seed KB must contain at least one document"


# ── Per-document required fields ────────────────────────────────────────────────


def test_all_documents_have_required_fields(kb_data):
    for i, doc in enumerate(kb_data["documents"]):
        missing = REQUIRED_DOC_FIELDS - set(doc.keys())
        assert not missing, (
            f"Document {i} (id={doc.get('id', '??')}) missing fields: {sorted(missing)}"
        )


def test_all_documents_have_manual_review_required(kb_data):
    for i, doc in enumerate(kb_data["documents"]):
        assert doc["review_status"] == "manual_review_required", (
            f"Document {i} (id={doc.get('id')}) has review_status={doc['review_status']!r}; "
            f"all seed entries must be 'manual_review_required'"
        )
        assert doc["reviewed_by"] is None
        assert doc["last_reviewed_at"] is None


# ── source_ref ──────────────────────────────────────────────────────────────────


def test_all_documents_have_source_ref_document_title(kb_data):
    for i, doc in enumerate(kb_data["documents"]):
        sr = doc.get("source_ref", {})
        assert sr.get("document_title"), (
            f"Document {i} (id={doc.get('id')}) missing source_ref.document_title"
        )


def test_all_documents_have_source_ref_page_start(kb_data):
    for i, doc in enumerate(kb_data["documents"]):
        sr = doc.get("source_ref", {})
        assert sr.get("page_start") is not None, (
            f"Document {i} (id={doc.get('id')}) missing source_ref.page_start"
        )


# ── Content constraints ─────────────────────────────────────────────────────────


def test_content_no_banned_words(kb_data):
    for doc in kb_data["documents"]:
        content = doc.get("content", "")
        for word in BANNED_CONTENT_WORDS:
            assert word not in content, (
                f"Document {doc['id']}: content contains banned word {word!r}"
            )


# ── ID uniqueness ───────────────────────────────────────────────────────────────


def test_ids_are_unique(kb_data):
    ids = [doc["id"] for doc in kb_data["documents"]]
    assert len(ids) == len(set(ids)), (
        f"Duplicate IDs found: {[i for i in ids if ids.count(i) > 1]}"
    )


def test_ids_are_snake_case(kb_data):
    for doc in kb_data["documents"]:
        doc_id = doc["id"]
        assert " " not in doc_id, f"ID {doc_id!r} contains spaces"
        assert doc_id == doc_id.lower(), f"ID {doc_id!r} is not lowercase"


# ── Field value constraints ─────────────────────────────────────────────────────


def test_categories_are_allowed(kb_data):
    for doc in kb_data["documents"]:
        assert doc["category"] in ALLOWED_CATEGORIES, (
            f"Document {doc['id']}: category={doc['category']!r} not in {sorted(ALLOWED_CATEGORIES)}"
        )


def test_risk_levels_are_allowed(kb_data):
    for doc in kb_data["documents"]:
        assert doc["risk_level"] in ALLOWED_RISK_LEVELS, (
            f"Document {doc['id']}: risk_level={doc['risk_level']!r}"
        )


def test_review_statuses_are_allowed(kb_data):
    for doc in kb_data["documents"]:
        assert doc["review_status"] in ALLOWED_REVIEW_STATUSES, (
            f"Document {doc['id']}: review_status={doc['review_status']!r}"
        )


def test_aliases_are_non_empty(kb_data):
    for doc in kb_data["documents"]:
        assert isinstance(doc.get("aliases"), list), (
            f"Document {doc['id']}: aliases must be a list"
        )
        assert len(doc["aliases"]) > 0, (
            f"Document {doc['id']}: aliases must not be empty"
        )


def test_allowed_usage_are_valid(kb_data):
    valid = {"knowledge_qa", "explanation", "internal_test"}
    for doc in kb_data["documents"]:
        for usage in doc["allowed_usage"]:
            assert usage in valid, (
                f"Document {doc['id']}: allowed_usage={usage!r} not in {sorted(valid)}"
            )


def test_source_is_consistent(kb_data):
    for doc in kb_data["documents"]:
        assert doc["source"] == "materials_terminology_guide_2020", (
            f"Document {doc['id']}: source={doc['source']!r} — "
            f"all seed entries should come from the same source"
        )


# ── Source registry (A5) ──────────────────────────────────────────────────────


def test_sources_top_level_exists(kb_data):
    assert "sources" in kb_data, "Top-level 'sources' key is required"
    assert isinstance(kb_data["sources"], dict)


def test_sources_have_copyright_and_license(kb_data):
    for key, src in kb_data["sources"].items():
        assert "copyright" in src, f"Source {key} missing 'copyright'"
        assert "license" in src, f"Source {key} missing 'license'"
        assert "usage_note" in src, f"Source {key} missing 'usage_note'"


def test_document_source_in_registry(kb_data):
    registry = set(kb_data.get("sources", {}).keys())
    for doc in kb_data["documents"]:
        assert doc["source"] in registry, (
            f"Document {doc['id']}: source={doc['source']!r} not found in sources registry. "
            f"Registered sources: {sorted(registry)}"
        )


# ── risk_note (A3) ────────────────────────────────────────────────────────────


def test_documents_have_risk_note(kb_data):
    for doc in kb_data["documents"]:
        assert "risk_note" in doc, (
            f"Document {doc['id']}: missing 'risk_note' field"
        )
        # Can be empty string, but field must exist.
