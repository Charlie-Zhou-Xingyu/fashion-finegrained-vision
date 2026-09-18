"""Knowledge-QA agent graph (LangGraph).

Why a graph when ``QaOrchestrator`` already routes?  The orchestrator is a
single-pass dispatcher; the two things it cannot express cleanly are (1) a
*bounded* retry when the model's answer fails the grounding check and (2) an
inspectable trace of which node ran with what state.  Both are the normal
job of an agent runtime, so they live here, and every node delegates to the
existing services rather than re-implementing them:

    START → classify ─┬─ knowledge intent ─→ retrieve ─┬─ evidence + MLLM ─→ generate → verify ─┬─ grounded ─→ END
                      │                                │                                        ├─ retry (≤ max_retries) → generate
                      └─ other ─→ unsupported → END     └─ otherwise ─→ template → END           └─ give up ─→ template → END

Grounding policy is the same as the orchestrator's (``inference.rag.generation``):
a reply must cite a valid ``[n]`` or refuse explicitly; the retry re-prompts
with a stricter instruction once, then falls back to the deterministic
template.  The graph never answers from the model without evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict

from inference.rag.generation import REFUSAL_PHRASE, build_grounded_prompt, check_grounding
from inference.rag.settings import GenerationSettings

logger = logging.getLogger(__name__)

GRAPH_VERSION = "0.1.0"
KNOWLEDGE_INTENTS = frozenset({"knowledge_qa", "design_explanation", "craft_explanation"})
UNSUPPORTED_ANSWER = "当前问题类型暂不支持。"
STRICT_RETRY_SUFFIX = (
    "\n\n注意：上一次回答没有标注引用编号，未被采纳。请在每个结论后标注对应条目编号（如[1]）；"
    f"若条目无法支持回答，请只回复“{REFUSAL_PHRASE}”。"
)


class QAGraphState(TypedDict, total=False):
    query: str
    image_bytes: Optional[bytes]
    intent: Dict[str, Any]
    hits: List[Any]
    context_ids: List[str]
    answer: str
    generation: str            # "mllm" | "template" | "unsupported"
    grounded: bool
    cited_ids: List[str]
    attempts: int
    max_retries: int
    warnings: List[Dict[str, str]]
    trace: List[str]


@dataclass
class QAGraphServices:
    intent_classifier: Any
    rag_service: Any
    mllm_client: Any = None
    generation: GenerationSettings = GenerationSettings(enabled=True)


def _warn(code: str, message: str, severity: str = "warn", scope: str = "agent") -> Dict[str, str]:
    return {"code": code, "scope": scope, "message": message, "severity": severity}


def _warnings_to_dicts(items: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for w in items or []:
        if hasattr(w, "model_dump"):
            d = w.model_dump()
            out.append({k: str(d.get(k, "")) for k in ("code", "scope", "message", "severity")})
        elif isinstance(w, dict):
            out.append({k: str(w.get(k, "")) for k in ("code", "scope", "message", "severity")})
    return out


def _template_answer(hits: List[Any]) -> str:
    from inference.serving.qa_orchestrator import _knowledge_answer
    return _knowledge_answer(list(hits), [])["answer"]


# ── Node factories ─────────────────────────────────────────────────────────────


def _make_nodes(svc: QAGraphServices) -> Dict[str, Callable[[QAGraphState], QAGraphState]]:
    gen = svc.generation

    def classify(state: QAGraphState) -> QAGraphState:
        data = svc.intent_classifier.classify(state["query"])
        intent = {
            "primary_intent": getattr(data, "primary_intent", "fallback_unknown"),
            "sub_intent": getattr(data, "sub_intent", None),
            "confidence": getattr(data, "intent_confidence", 0.0),
        }
        return {"intent": intent, "trace": state.get("trace", []) + ["classify"]}

    def retrieve(state: QAGraphState) -> QAGraphState:
        intent = state["intent"]
        r = svc.rag_service.retrieve(
            query=state["query"], primary_intent=intent["primary_intent"],
            sub_intent=intent.get("sub_intent"), top_k=gen.max_context_hits,
        )
        warnings = state.get("warnings", []) + [
            w for w in _warnings_to_dicts(r.warnings) if w["code"] != "no_hits"
        ]
        return {"hits": list(r.hits), "warnings": warnings, "trace": state.get("trace", []) + ["retrieve"]}

    def generate(state: QAGraphState) -> QAGraphState:
        attempts = state.get("attempts", 0)
        system_prompt, user_prompt, ctx_ids = build_grounded_prompt(state["query"], state["hits"], gen.max_context_hits)
        if attempts > 0:
            user_prompt += STRICT_RETRY_SUFFIX
        warnings = list(state.get("warnings", []))
        try:
            res = svc.mllm_client.chat(
                query=user_prompt, system_prompt=system_prompt,
                image_bytes=state.get("image_bytes"),
                context={"knowledge_ids": ctx_ids, "attempt": attempts + 1},
                max_new_tokens=gen.max_new_tokens, temperature=0.0,
            )
            answer = (getattr(res, "text", None) or "").strip()
            warnings += _warnings_to_dicts(getattr(res, "warnings", []))
            failed = False
        except Exception as exc:
            logger.exception("qa_graph: MLLM generation failed")
            answer = ""
            warnings.append(_warn("mllm_generation_failed", f"大模型生成失败：{type(exc).__name__}"))
            failed = True
        return {
            "answer": answer, "context_ids": ctx_ids,
            # a transport failure is not worth retrying with a stricter prompt
            "attempts": (state.get("max_retries", 0) + 1) if failed else attempts + 1,
            "warnings": warnings, "trace": state.get("trace", []) + ["generate"],
        }

    def verify(state: QAGraphState) -> QAGraphState:
        ok, cites = check_grounding(state.get("answer", ""), len(state.get("context_ids", [])))
        ok = ok or not gen.require_citation and bool(state.get("answer"))
        warnings = list(state.get("warnings", []))
        if not ok:
            warnings.append(_warn("generation_ungrounded",
                                  f"第{state.get('attempts', 0)}次回答未引用知识条目。", severity="info"))
        ctx = state.get("context_ids", [])
        return {
            "grounded": ok, "generation": "mllm" if ok else state.get("generation", ""),
            "cited_ids": [ctx[i - 1] for i in cites if 0 < i <= len(ctx)],
            "warnings": warnings, "trace": state.get("trace", []) + ["verify"],
        }

    def template(state: QAGraphState) -> QAGraphState:
        warnings = list(state.get("warnings", []))
        if state.get("attempts", 0) > 0 and not state.get("grounded", False):
            warnings.append(_warn("generation_fallback_template", "模型回答多次未引用知识条目，已回退为模板回答。"))
        return {
            "answer": _template_answer(state.get("hits", [])), "generation": "template",
            "cited_ids": [], "warnings": warnings, "trace": state.get("trace", []) + ["template"],
        }

    def unsupported(state: QAGraphState) -> QAGraphState:
        return {"answer": UNSUPPORTED_ANSWER, "generation": "unsupported", "cited_ids": [],
                "trace": state.get("trace", []) + ["unsupported"]}

    return {"classify": classify, "retrieve": retrieve, "generate": generate,
            "verify": verify, "template": template, "unsupported": unsupported}


# ── Routing predicates ─────────────────────────────────────────────────────────


def _route_after_classify(state: QAGraphState) -> str:
    return "retrieve" if state["intent"]["primary_intent"] in KNOWLEDGE_INTENTS else "unsupported"


def _make_route_after_retrieve(svc: QAGraphServices) -> Callable[[QAGraphState], str]:
    def route(state: QAGraphState) -> str:
        if not state.get("hits"):
            return "template"
        if not svc.generation.enabled or svc.mllm_client is None:
            return "template"
        try:
            if not svc.mllm_client.is_available():
                return "template"
        except Exception:
            return "template"
        return "generate"
    return route


def _route_after_verify(state: QAGraphState) -> str:
    if state.get("grounded"):
        return "end"
    if state.get("attempts", 0) <= state.get("max_retries", 0):
        return "generate"
    return "template"


# ── Public API ─────────────────────────────────────────────────────────────────


def build_qa_graph(services: QAGraphServices):
    """Compile the graph. Requires ``langgraph`` (imported here, not at module load)."""
    from langgraph.graph import END, START, StateGraph

    nodes = _make_nodes(services)
    g = StateGraph(QAGraphState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route_after_classify, {"retrieve": "retrieve", "unsupported": "unsupported"})
    g.add_conditional_edges("retrieve", _make_route_after_retrieve(services), {"generate": "generate", "template": "template"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", _route_after_verify, {"end": END, "generate": "generate", "template": "template"})
    g.add_edge("template", END)
    g.add_edge("unsupported", END)
    return g.compile()


def run_qa_graph(
    services: QAGraphServices, query: str, *, image_bytes: Optional[bytes] = None,
    max_retries: int = 1, graph: Any = None,
) -> Dict[str, Any]:
    """Run the graph once and return the final state as a plain dict."""
    graph = graph or build_qa_graph(services)
    initial: QAGraphState = {
        "query": (query or "").strip(), "image_bytes": image_bytes,
        "attempts": 0, "max_retries": max(0, int(max_retries)),
        "warnings": [], "trace": [], "cited_ids": [], "generation": "",
    }
    final = graph.invoke(initial)
    final.pop("image_bytes", None)
    final["graph_version"] = GRAPH_VERSION
    return dict(final)
