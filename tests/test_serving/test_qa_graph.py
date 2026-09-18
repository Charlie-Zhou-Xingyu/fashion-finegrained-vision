"""P2 — LangGraph knowledge-QA graph tests (fakes only; no model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("langgraph")

from inference.agent.qa_graph import (
    KNOWLEDGE_INTENTS, STRICT_RETRY_SUFFIX, UNSUPPORTED_ANSWER,
    QAGraphServices, build_qa_graph, run_qa_graph,
)
from inference.rag.generation import REFUSAL_PHRASE
from inference.rag.settings import GenerationSettings
from inference.serving.intent_classifier import RuleIntentClassifier
from inference.serving.rag_service import RagService, RetrievalResult

KNOWLEDGE_QUERY = "什么是纤维"


@dataclass
class FakeResult:
    text: Optional[str]
    used_tools: List[str] = field(default_factory=lambda: ["fake_mllm"])
    warnings: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class ScriptedMLLM:
    """Returns the scripted replies in order; raises when a reply is an Exception."""

    def __init__(self, replies: List[Any], available: bool = True) -> None:
        self.replies, self.available, self.calls = list(replies), available, []

    def is_available(self) -> bool:
        return self.available

    def chat(self, **kw) -> FakeResult:
        self.calls.append(kw)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResult(reply)


class EmptyRag:
    def retrieve(self, **kw):
        return RetrievalResult(query=kw.get("query", ""), normalized_query="", hits=[])


@pytest.fixture(scope="module")
def rag() -> RagService:
    return RagService()


def _services(mllm: Any, rag_service: Any, **gen_kw) -> QAGraphServices:
    return QAGraphServices(
        intent_classifier=RuleIntentClassifier(), rag_service=rag_service, mllm_client=mllm,
        generation=GenerationSettings(**{"enabled": True, **gen_kw}),
    )


def test_grounded_first_try(rag):
    mllm = ScriptedMLLM(["纤维是纺织的基本单元[1]。"])
    s = run_qa_graph(_services(mllm, rag), KNOWLEDGE_QUERY)
    assert s["trace"] == ["classify", "retrieve", "generate", "verify"]
    assert s["generation"] == "mllm" and s["grounded"] is True
    assert s["answer"] == "纤维是纺织的基本单元[1]。"
    assert s["cited_ids"] == [s["hits"][0].id]
    assert s["attempts"] == 1 and s["graph_version"]
    assert s["intent"]["primary_intent"] in KNOWLEDGE_INTENTS


def test_retry_once_with_strict_prompt_then_succeed(rag):
    mllm = ScriptedMLLM(["随口一说。", "纤维见条目[1]。"])
    s = run_qa_graph(_services(mllm, rag), KNOWLEDGE_QUERY, max_retries=1)
    assert s["trace"] == ["classify", "retrieve", "generate", "verify", "generate", "verify"]
    assert s["generation"] == "mllm" and s["attempts"] == 2
    assert STRICT_RETRY_SUFFIX not in mllm.calls[0]["query"]
    assert mllm.calls[1]["query"].endswith(STRICT_RETRY_SUFFIX)
    assert mllm.calls[1]["context"]["attempt"] == 2
    assert [w["code"] for w in s["warnings"]].count("generation_ungrounded") == 1


def test_gives_up_after_max_retries_and_uses_template(rag):
    mllm = ScriptedMLLM(["胡说一", "胡说二"])
    s = run_qa_graph(_services(mllm, rag), KNOWLEDGE_QUERY, max_retries=1)
    assert s["trace"][-1] == "template" and s["trace"].count("generate") == 2
    assert s["generation"] == "template" and "根据当前知识库" in s["answer"]
    assert s["cited_ids"] == []
    codes = [w["code"] for w in s["warnings"]]
    assert codes.count("generation_ungrounded") == 2 and "generation_fallback_template" in codes


def test_zero_retries(rag):
    mllm = ScriptedMLLM(["胡说"])
    s = run_qa_graph(_services(mllm, rag), KNOWLEDGE_QUERY, max_retries=0)
    assert s["trace"] == ["classify", "retrieve", "generate", "verify", "template"]


def test_refusal_counts_as_grounded(rag):
    s = run_qa_graph(_services(ScriptedMLLM([REFUSAL_PHRASE]), rag), KNOWLEDGE_QUERY)
    assert s["generation"] == "mllm" and s["cited_ids"] == []


def test_unavailable_mllm_goes_straight_to_template(rag):
    mllm = ScriptedMLLM(["never"], available=False)
    s = run_qa_graph(_services(mllm, rag), KNOWLEDGE_QUERY)
    assert s["trace"] == ["classify", "retrieve", "template"]
    assert mllm.calls == [] and s["generation"] == "template"


def test_generation_disabled_goes_to_template(rag):
    mllm = ScriptedMLLM(["never"])
    s = run_qa_graph(_services(mllm, rag, enabled=False), KNOWLEDGE_QUERY)
    assert s["trace"] == ["classify", "retrieve", "template"] and mllm.calls == []


def test_no_evidence_means_no_model_call():
    mllm = ScriptedMLLM(["凭空[1]"])
    s = run_qa_graph(_services(mllm, EmptyRag()), KNOWLEDGE_QUERY)
    assert s["trace"] == ["classify", "retrieve", "template"] and mllm.calls == []
    assert "暂未检索到相关知识" in s["answer"]


def test_transport_failure_is_not_retried(rag):
    mllm = ScriptedMLLM([RuntimeError("down"), "会被忽略[1]"])
    s = run_qa_graph(_services(mllm, rag), KNOWLEDGE_QUERY, max_retries=1)
    assert s["trace"] == ["classify", "retrieve", "generate", "verify", "template"]
    assert len(mllm.calls) == 1
    assert "mllm_generation_failed" in [w["code"] for w in s["warnings"]]


def test_non_knowledge_intent_is_unsupported(rag):
    mllm = ScriptedMLLM(["never"])
    s = run_qa_graph(_services(mllm, rag), "这件衣服是什么面料？")     # attribute_query
    assert s["trace"] == ["classify", "unsupported"]
    assert s["answer"] == UNSUPPORTED_ANSWER and mllm.calls == []


def test_compiled_graph_is_reusable(rag):
    services = _services(ScriptedMLLM(["a[1]", "b[1]"]), rag)
    graph = build_qa_graph(services)
    s1 = run_qa_graph(services, KNOWLEDGE_QUERY, graph=graph)
    s2 = run_qa_graph(services, KNOWLEDGE_QUERY, graph=graph)
    assert s1["answer"] == "a[1]" and s2["answer"] == "b[1]"
    assert s2["trace"] == ["classify", "retrieve", "generate", "verify"]   # state does not leak
