"""P2 — grounded MLLM generation on the knowledge route."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from inference.llm.mllm_client import MockMLLMClient
from inference.rag.generation import REFUSAL_PHRASE, build_grounded_prompt, check_grounding
from inference.rag.settings import GenerationSettings
from inference.serving.attribute_service import AttributeService
from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.qa_orchestrator import QaOrchestrator, _decode_image_bytes
from inference.serving.rag_service import RagService
from inference.serving.schemas import WarningItem, WarningSeverity

KNOWLEDGE_QUERY = "什么是纤维"


@dataclass
class FakeResult:
    text: Optional[str]
    used_tools: List[str] = field(default_factory=lambda: ["fake_mllm"])
    warnings: List[WarningItem] = field(default_factory=list)
    latency_ms: float = 12.5
    meta: Dict[str, Any] = field(default_factory=lambda: {"provider": "fake"})


class FakeMLLM:
    def __init__(self, text: Optional[str], available: bool = True, raise_exc: bool = False) -> None:
        self.text, self.available, self.raise_exc = text, available, raise_exc
        self.calls: List[dict] = []

    def is_available(self) -> bool:
        return self.available

    def chat(self, **kw) -> FakeResult:
        self.calls.append(kw)
        if self.raise_exc:
            raise RuntimeError("boom")
        return FakeResult(self.text)


class EmptyRag:
    def retrieve(self, **kw):
        from inference.serving.rag_service import RetrievalResult
        return RetrievalResult(query=kw.get("query", ""), normalized_query="", hits=[], used_tools=["rag_service"])


def _orch(mllm: Any, enabled: bool = True, rag: Any = None, **gen_kw) -> QaOrchestrator:
    return QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=rag or RagService(),
        mllm_client=mllm,
        generation_settings=GenerationSettings(enabled=enabled, **gen_kw),
    )


# ── Pure helpers ───────────────────────────────────────────────────────────────


def test_prompt_numbers_hits_and_returns_ids():
    hits = RagService().retrieve(KNOWLEDGE_QUERY, top_k=3).hits
    system, user, ids = build_grounded_prompt("问题？", hits, max_context_hits=2)
    assert ids == [h.id for h in hits[:2]]
    assert "[1] " in user and "[2] " in user and "[3] " not in user
    assert hits[0].title in user and "用户问题：问题？" in user
    assert REFUSAL_PHRASE in system


@pytest.mark.parametrize("text,n,expected", [
    ("羊毛是动物纤维[1]。", 3, (True, [1])),
    ("见[2]和[3]，也见[1]。", 3, (True, [1, 2, 3])),
    ("见[7]。", 3, (False, [])),                 # out-of-range citation is not grounding
    (REFUSAL_PHRASE + "。", 3, (True, [])),       # explicit refusal is acceptable
    ("羊毛很保暖。", 3, (False, [])),
    ("", 3, (False, [])),
])
def test_check_grounding(text, n, expected):
    assert check_grounding(text, n) == expected


def test_decode_image_bytes():
    assert _decode_image_bytes(None) is None
    assert _decode_image_bytes(b"\x89PNG") == b"\x89PNG"
    assert _decode_image_bytes("aGVsbG8=") == b"hello"
    assert _decode_image_bytes("data:image/png;base64,aGVsbG8=") == b"hello"
    assert _decode_image_bytes(12) is None


# ── Orchestrator behaviour ─────────────────────────────────────────────────────


def test_generation_disabled_uses_template_and_never_calls_mllm():
    mllm = FakeMLLM("羊毛[1]")
    r = _orch(mllm, enabled=False).answer(query=KNOWLEDGE_QUERY)
    assert r.answer_type == "knowledge_answer"
    assert r.meta["generation"] == "template"
    assert "template_answer" in r.used_tools and "fake_mllm" not in r.used_tools
    assert mllm.calls == []


def test_mock_client_is_skipped_with_info_warning():
    r = _orch(MockMLLMClient()).answer(query=KNOWLEDGE_QUERY)
    assert r.meta["generation"] == "template"
    assert any(w.code == "mllm_generation_skipped" and w.severity == WarningSeverity.info for w in r.warnings)
    assert "template_answer" in r.used_tools


def test_grounded_answer_replaces_template():
    mllm = FakeMLLM("纤维是纺织材料的基本单元[1]。")
    r = _orch(mllm).answer(query=KNOWLEDGE_QUERY)
    assert r.answer == "纤维是纺织材料的基本单元[1]。"
    assert r.answer_type == "knowledge_answer"
    assert r.meta["generation"] == "mllm"
    assert r.meta["cited_context_indices"] == [1]
    assert r.meta["cited_knowledge_ids"] == [r.sources[0]["id"]]
    assert r.meta["mllm_provider"] == "fake" and r.meta["mllm_latency_ms"] == 12.5
    assert "fake_mllm" in r.used_tools and "template_answer" not in r.used_tools
    assert r.sources and r.sources[0]["type"] == "knowledge_base"    # sources unchanged
    call = mllm.calls[0]
    assert call["temperature"] == 0.0 and call["max_new_tokens"] == 256
    assert "[1] " in call["query"] and call["system_prompt"]
    assert call["context"]["knowledge_ids"] == [s["id"] for s in r.sources[:3]]


def test_ungrounded_answer_falls_back_to_template():
    r = _orch(FakeMLLM("羊毛很保暖，随便说的。")).answer(query=KNOWLEDGE_QUERY)
    assert r.meta["generation"] == "template"
    assert "根据当前知识库" in r.answer
    assert any(w.code == "generation_ungrounded" and w.severity == WarningSeverity.warn for w in r.warnings)
    assert "template_answer" in r.used_tools


def test_ungrounded_answer_accepted_when_citation_not_required():
    r = _orch(FakeMLLM("羊毛很保暖。"), require_citation=False).answer(query=KNOWLEDGE_QUERY)
    assert r.answer == "羊毛很保暖。" and r.meta["generation"] == "mllm"
    assert r.meta["cited_knowledge_ids"] == []


def test_refusal_is_accepted_as_grounded():
    r = _orch(FakeMLLM(REFUSAL_PHRASE)).answer(query=KNOWLEDGE_QUERY)
    assert r.answer == REFUSAL_PHRASE and r.meta["generation"] == "mllm"


def test_empty_and_failing_mllm_fall_back():
    r = _orch(FakeMLLM("   ")).answer(query=KNOWLEDGE_QUERY)
    assert r.meta["generation"] == "template"
    assert any(w.code == "mllm_empty_answer" for w in r.warnings)
    r = _orch(FakeMLLM(None, raise_exc=True)).answer(query=KNOWLEDGE_QUERY)
    assert r.meta["generation"] == "template"
    assert any(w.code == "mllm_generation_failed" for w in r.warnings)


def test_no_hits_means_no_generation_call():
    mllm = FakeMLLM("凭空捏造[1]")
    r = _orch(mllm, rag=EmptyRag()).answer(query=KNOWLEDGE_QUERY)
    assert mllm.calls == []
    assert r.meta["generation"] == "template" and r.meta["rag_hit_count"] == 0


def test_image_bytes_are_decoded_and_forwarded():
    mllm = FakeMLLM("见[1]。")
    _orch(mllm).answer(query=KNOWLEDGE_QUERY, image_bytes="data:image/png;base64,aGVsbG8=")
    assert mllm.calls[0]["image_bytes"] == b"hello"


def test_max_context_hits_limits_prompt():
    mllm = FakeMLLM("见[1]。")
    _orch(mllm, max_context_hits=1).answer(query=KNOWLEDGE_QUERY)
    assert "[2] " not in mllm.calls[0]["query"]
