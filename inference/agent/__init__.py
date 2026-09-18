"""P2 — LangGraph orchestration shell for knowledge QA (PRD §3.4).

``qa_graph`` expresses the knowledge route as an explicit state machine
(classify → retrieve → generate → verify, with a bounded grounding-retry
loop) over the *same* services the deterministic ``QaOrchestrator`` uses.
It adds a traceable state and a retry policy; it does not add new tools.
LangGraph is imported lazily so the serving package works without it.
"""
