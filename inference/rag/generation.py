"""Grounded answer generation helpers (prompt construction + citation check).

Pure functions, no model access.  The orchestrator decides *whether* to call
the MLLM; this module decides *what* to send and whether the reply can be
trusted to be grounded in the retrieved knowledge.

Grounding policy: the model is told to cite context entries as ``[n]`` and
to decline explicitly when the entries do not cover the question.  A reply is
accepted only if it cites at least one valid entry OR contains the explicit
refusal phrase; anything else falls back to the deterministic template answer
(never an uncited free-form claim).
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple

REFUSAL_PHRASE = "当前知识库暂未覆盖该问题"

SYSTEM_PROMPT = (
    "你是电商服饰知识助手。你只能依据下面给出的、带编号的知识条目回答用户问题；"
    "回答时用[编号]标注引用的条目，例如[1]。"
    f"如果条目不足以回答，请直接回复“{REFUSAL_PHRASE}”，不要编造，不要引入条目之外的事实。"
    "如果条目标注为未审核，请在回答末尾提醒用户以商品详情页或官方说明为准。"
    "回答使用简体中文，简洁、准确。"
)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def _hit_field(hit: Any, name: str, default: str = "") -> str:
    val = getattr(hit, name, None)
    if val is None and isinstance(hit, dict):
        val = hit.get(name)
    return str(val) if val else default


def build_grounded_prompt(
    query: str, hits: Sequence[Any], max_context_hits: int = 3,
) -> Tuple[str, str, List[str]]:
    """Return ``(system_prompt, user_prompt, context_doc_ids)``.

    ``context_doc_ids[i]`` is the KB id cited as ``[i+1]`` in the prompt.
    """
    chosen = list(hits)[: max(1, max_context_hits)]
    lines: List[str] = []
    ids: List[str] = []
    for i, h in enumerate(chosen, start=1):
        title = _hit_field(h, "title")
        content = _hit_field(h, "content")
        source = _hit_field(h, "source", "unknown")
        review = _hit_field(h, "review_status", "unknown")
        review_cn = "已审核" if review == "reviewed" else "未审核"
        lines.append(f"[{i}] {title}：{content}（来源：{source}；审核状态：{review_cn}）")
        ids.append(_hit_field(h, "id"))
    user_prompt = (
        "知识条目：\n" + "\n".join(lines) +
        f"\n\n用户问题：{query.strip()}\n请仅依据上述条目回答，并标注引用编号。"
    )
    return SYSTEM_PROMPT, user_prompt, ids


def check_grounding(text: str, n_context: int) -> Tuple[bool, List[int]]:
    """Return ``(is_grounded, cited_indices)``.

    Grounded ⇔ cites ≥1 index within ``1..n_context`` or contains the refusal
    phrase.  Citations outside the range are ignored (a model hallucinating
    ``[7]`` with three entries does not count as grounded).
    """
    if not text:
        return False, []
    cited = sorted({int(m) for m in _CITATION_RE.findall(text)})
    valid = [c for c in cited if 1 <= c <= n_context]
    if valid:
        return True, valid
    if REFUSAL_PHRASE in text:
        return True, []
    return False, []
