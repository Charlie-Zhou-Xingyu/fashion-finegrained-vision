"""
P0a.4.2-hardened — RagService: Exact / Alias / BM25-like retrieval.

Loads the seed knowledge base and retrieval config.  Supports exact-match,
alias-match, title-match, and simplified BM25-like token-overlap scoring with
category filtering, stable ranking, and comprehensive metadata passthrough.

Does NOT depend on FAISS, BGE, Redis, LLM, or the 3.1 vision pipeline.
Does NOT generate answers — only returns ranked hits with full metadata.
"""

from __future__ import annotations

import json
import logging
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from inference.serving.schemas import WarningItem, WarningSeverity

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_KB_PATH = _PROJECT_ROOT / "configs" / "knowledge_base.yaml"
_DEFAULT_RETRIEVAL_CONFIG_PATH = _PROJECT_ROOT / "configs" / "retrieval_config.yaml"

# ── Data models ────────────────────────────────────────────────────────────────


@dataclass
class RetrievalHit:
    id: str
    category: str
    term: str
    zh_term: Optional[str]
    title: str
    content: str
    score: float
    match_type: str
    source: str
    source_type: str
    source_url: Optional[str]
    source_ref: Dict[str, Any]
    allowed_usage: List[str]
    risk_level: str
    risk_note: Optional[str]
    review_status: str
    reviewed_by: Optional[str]
    last_reviewed_at: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "category": self.category,
            "term": self.term, "zh_term": self.zh_term,
            "title": self.title, "content": self.content,
            "score": self.score, "match_type": self.match_type,
            "source": self.source, "source_type": self.source_type,
            "source_url": self.source_url, "source_ref": dict(self.source_ref),
            "allowed_usage": list(self.allowed_usage),
            "risk_level": self.risk_level, "risk_note": self.risk_note,
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "last_reviewed_at": self.last_reviewed_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class RetrievalResult:
    query: str
    normalized_query: str
    hits: List[RetrievalHit]
    used_tools: List[str] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "hits": [h.to_dict() for h in self.hits],
            "used_tools": self.used_tools,
            "warnings": [w.model_dump() for w in self.warnings],
            "meta": self.meta,
        }


_MATCH_PRIORITY = {"exact": 0, "alias": 1, "title": 2, "bm25": 3}

# ── KB validation ──────────────────────────────────────────────────────────────

_REQUIRED_DOC_KEYS = frozenset({
    "id", "category", "term", "zh_term", "aliases", "title", "content",
    "allowed_usage", "risk_level", "source", "source_type", "source_url",
    "source_ref", "review_status", "reviewed_by", "last_reviewed_at",
    "version", "tags",
})
_REQUIRED_SOURCE_KEYS = frozenset({
    "title", "publisher", "year", "version", "copyright",
    "source_type", "source_url", "license", "usage_note",
})
_VALID_REVIEW_STATUSES = frozenset({"seed_unreviewed", "manual_review_required", "reviewed", "deprecated"})
_VALID_RISK_LEVELS = frozenset({"low", "medium", "high"})


def _validate_kb(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in ("version", "locale", "source_policy", "sources", "documents"):
        if key not in data:
            errors.append(f"KB missing top-level key: {key}")

    sources = data.get("sources", {})
    if not isinstance(sources, dict):
        errors.append("KB sources must be a dict")
    else:
        for sk, sv in sources.items():
            if not isinstance(sv, dict):
                errors.append(f"sources.{sk} must be a dict")
                continue
            missing_src = _REQUIRED_SOURCE_KEYS - set(sv.keys())
            if missing_src:
                errors.append(f"sources.{sk} missing: {sorted(missing_src)}")

    docs = data.get("documents")
    if not isinstance(docs, list):
        errors.append("KB documents must be a list")
        return errors

    ids_seen: set = set()
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            errors.append(f"KB documents[{i}] is not a dict")
            continue
        missing = _REQUIRED_DOC_KEYS - set(doc.keys())
        if missing:
            errors.append(f"KB documents[{i}] (id={doc.get('id','?')}) missing: {sorted(missing)}")
        doc_id = doc.get("id")
        if doc_id in ids_seen:
            errors.append(f"KB documents[{i}] duplicate id: {doc_id!r}")
        ids_seen.add(doc_id or "")
        if not isinstance(doc.get("aliases"), list):
            errors.append(f"KB documents[{i}] aliases must be a list")
        if not isinstance(doc.get("tags"), list):
            errors.append(f"KB documents[{i}] tags must be a list")
        if not isinstance(doc.get("allowed_usage"), list):
            errors.append(f"KB documents[{i}] allowed_usage must be a list")
        sr = doc.get("source_ref")
        if not isinstance(sr, dict) or "document_title" not in sr or "page_start" not in sr:
            errors.append(f"KB documents[{i}] source_ref missing document_title / page_start")
        if doc.get("source") not in sources:
            errors.append(f"KB documents[{i}] source={doc.get('source','?')!r} not in sources registry")
        if doc.get("review_status") not in _VALID_REVIEW_STATUSES:
            errors.append(f"KB documents[{i}] invalid review_status: {doc.get('review_status')!r}")
        if doc.get("risk_level") not in _VALID_RISK_LEVELS:
            errors.append(f"KB documents[{i}] invalid risk_level: {doc.get('risk_level')!r}")
    return errors


# ── Query normalization ────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation + "，。！？、；：""''【】《》（）…—") + r"\s]+")
_SPACES_RE = re.compile(r"\s+")


def normalize_query(query: Optional[str]) -> str:
    if not query:
        return ""
    q = query.strip()
    q = q.lower()
    q = _PUNCT_RE.sub(" ", q)
    q = _SPACES_RE.sub(" ", q)
    return q.strip()


# ── Tokenization ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    parts = re.findall(r"[a-zA-Z]+|.", text)
    for part in parts:
        if re.match(r"[a-zA-Z]+", part):
            if len(part) >= 2:
                tokens.append(part.lower())
            for i in range(len(part) - 1):
                tokens.append(part[i:i+2].lower())
        else:
            for ch in part:
                if ch.strip():
                    tokens.append(ch)
            for i in range(len(part) - 1):
                bigram = part[i:i+2]
                if bigram.strip():
                    tokens.append(bigram)
    return [t for t in tokens if len(t.strip()) >= 1]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_int(val: Any, default: int) -> int:
    """Convert *val* to int if possible; return *default* otherwise."""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val.strip())
        except (ValueError, TypeError):
            pass
    return default


def _safe_strings(val: Any) -> List[str]:
    """Flatten *val* to a list of string tokens for attribute_context expansion."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, (int, float, bool)):
        return [str(val)]
    if isinstance(val, list):
        result: List[str] = []
        for item in val:
            result.extend(_safe_strings(item))
        return result
    # dict / nested — skip.
    return []


def _expand_query(query: str, attribute_context: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Return (expanded_query, context_keys) by appending safe string tokens from
    *attribute_context* to the original query for BM25 enrichment.

    Exact/alias matching is NOT influenced by attribute_context.
    """
    if not attribute_context:
        return query, []
    parts = [query]
    keys_used: List[str] = []
    for key, val in attribute_context.items():
        tokens = _safe_strings(val)
        if tokens:
            parts.extend(tokens)
            keys_used.append(key)
    expanded = " ".join(parts) if len(parts) > 1 else query
    return expanded, keys_used


# ── RagService ─────────────────────────────────────────────────────────────────


class RagService:
    """Local knowledge-base retrieval service.

    Supports exact-match, alias-match, title-match, and simplified BM25-like
    token-overlap scoring.  Does NOT use FAISS, BGE, or any embedding model.
    """

    def __init__(
        self,
        kb_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        cfg_path = config_path or _DEFAULT_RETRIEVAL_CONFIG_PATH
        with open(cfg_path, "r", encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh) or {}

        kb_p = kb_path or _DEFAULT_KB_PATH
        with open(kb_p, "r", encoding="utf-8") as fh:
            kb_data = yaml.safe_load(fh) or {}
        errors = _validate_kb(kb_data)
        if errors:
            for e in errors:
                logger.error("KB validation FAILED: %s", e)
            raise RuntimeError(f"KB validation failed ({len(errors)} errors)")
        self._docs: List[Dict[str, Any]] = kb_data.get("documents", [])
        self._sources: Dict[str, Any] = kb_data.get("sources", {})
        self._kb_version: str = kb_data.get("version", "unknown")

        self._doc_tokens: List[List[str]] = []
        for doc in self._docs:
            fields = [doc.get(k, "") or "" for k in ("term", "zh_term", "title", "content")]
            fields += [a for a in doc.get("aliases", []) if a]
            fields += [t for t in doc.get("tags", []) if t]
            self._doc_tokens.append(_tokenize(" ".join(fields)))

        logger.info("RagService loaded: %d documents, %d sources, kb_version=%s",
                     len(self._docs), len(self._sources), self._kb_version)

    # ── Public API ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: Optional[str],
        *,
        categories: Optional[List[str]] = None,
        primary_intent: Optional[str] = None,
        sub_intent: Optional[str] = None,
        top_k: Any = None,
        attribute_context: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        warnings: List[WarningItem] = []
        ret_cfg = self._cfg.get("retrieval", {})
        default_top = ret_cfg.get("default_top_k", 3)
        max_top = ret_cfg.get("max_top_k", 10)

        # Top-k handling (A5).
        requested_top_k: Any = top_k
        parsed_top_k = _safe_int(top_k, default_top)
        final_top_k = parsed_top_k
        warn_clamped = False
        if parsed_top_k <= 0:
            final_top_k = default_top
            warn_clamped = True
        elif parsed_top_k > max_top:
            final_top_k = max_top
            warn_clamped = True
        # Detect non-numeric string that parsed to default but wasn't explicitly 0 or default.
        if isinstance(top_k, str):
            try:
                int(top_k.strip())
            except (ValueError, TypeError):
                final_top_k = default_top
                warn_clamped = True
        if warn_clamped:
            warnings.append(WarningItem(
                code="top_k_clamped", scope="rag",
                message=f"top_k clamped from {top_k!r} to {final_top_k}",
                severity=WarningSeverity.info,
            ))

        # Normalize query (A4).
        nq = normalize_query(query)
        if not nq:
            warnings.append(WarningItem(
                code="empty_query", scope="rag",
                message="Query is empty after normalization.",
                severity=WarningSeverity.info,
            ))
            return RetrievalResult(
                query=query or "", normalized_query=nq,
                hits=[], used_tools=["rag_service"],
                warnings=warnings,
                meta={"effective_categories": categories or [], "top_k": final_top_k,
                     "requested_top_k": top_k, "kb_version": self._kb_version,
                     "attribute_context_keys": [], "expanded_query": nq},
            )

        # Empty KB.
        if not self._docs:
            warnings.append(WarningItem(
                code="kb_empty", scope="rag",
                message="Knowledge base is empty.",
                severity=WarningSeverity.warn,
            ))
            return RetrievalResult(
                query=query or "", normalized_query=nq,
                hits=[], used_tools=["rag_service"],
                warnings=warnings,
                meta={"effective_categories": categories or [], "top_k": final_top_k,
                     "requested_top_k": top_k, "kb_version": self._kb_version,
                     "attribute_context_keys": [], "expanded_query": nq},
            )

        # Resolve categories (A2, A3).
        effective_cats, cat_warnings = self._resolve_categories(
            categories, primary_intent, sub_intent,
        )
        warnings.extend(cat_warnings)

        # Attribute context expansion (A8).
        expanded_nq, ctx_keys = _expand_query(nq, attribute_context)

        # Search (A2: filtered BEFORE scoring).
        hits = self._search(nq, expanded_nq, effective_cats)
        hits = self._deduplicate_and_rank(hits)

        # Clamp scores to [0, 1] (A6).
        for h in hits:
            h.score = round(max(0.0, min(1.0, h.score)), 4)

        hits = hits[:final_top_k]

        if not hits and not any(w.code == "unknown_category" for w in warnings):
            warnings.append(WarningItem(
                code="no_hits", scope="rag",
                message="未检索到相关知识条目。",
                severity=WarningSeverity.info,
            ))

        return RetrievalResult(
            query=query or "", normalized_query=nq,
            hits=hits, used_tools=["rag_service"],
            warnings=warnings,
            meta={
                "effective_categories": effective_cats,
                "top_k": final_top_k,
                "requested_top_k": requested_top_k,
                "kb_version": self._kb_version,
                "attribute_context_keys": ctx_keys,
                "expanded_query": expanded_nq,
            },
        )

    # ── Internals ───────────────────────────────────────────────────────────

    def _resolve_categories(
        self,
        categories: Optional[List[str]],
        primary_intent: Optional[str],
        sub_intent: Optional[str],
    ) -> Tuple[List[str], List[WarningItem]]:
        warnings: List[WarningItem] = []
        if categories:
            known = set(self._cfg.get("category_boost", {}).keys())
            unknown = [c for c in categories if c not in known]
            if unknown:
                warnings.append(WarningItem(
                    code="unknown_category", scope="rag",
                    message=f"Unknown categories: {unknown}",
                    severity=WarningSeverity.info,
                ))
            return list(categories), warnings
        if primary_intent:
            intent_key = f"{primary_intent}/{sub_intent}" if sub_intent else primary_intent
            intent_map = self._cfg.get("intent_category_map", {})
            mapped = intent_map.get(intent_key, intent_map.get(primary_intent, []))
            return list(mapped), warnings
        return [], warnings

    def _search(self, nq: str, expanded_nq: str, categories: List[str]) -> List[RetrievalHit]:
        score_cfg = self._cfg.get("scores", {})
        nq_tokens = _tokenize(expanded_nq)
        exact_score = score_cfg.get("exact_match", 1.0)
        contains_score = score_cfg.get("contains_term_match", 0.96)
        alias_score = score_cfg.get("alias_match", 0.92)
        title_score_val = score_cfg.get("title_match", 0.85)
        bm25_min = score_cfg.get("bm25_min_score", 0.25)
        bm25_max = score_cfg.get("bm25_max_score", 0.75)
        cat_boost = self._cfg.get("category_boost", {})

        raw_hits: List[Tuple[int, float, str]] = []

        for idx, doc in enumerate(self._docs):
            if categories and doc.get("category") not in categories:
                continue

            term_lower = (doc.get("term") or "").lower()
            zh = (doc.get("zh_term") or "")
            ttl = (doc.get("title") or "").lower()
            aliases = [a.lower() for a in doc.get("aliases", [])]

            if nq == term_lower or nq == zh.lower():
                raw_hits.append((idx, exact_score, "exact"))
                continue
            if term_lower and term_lower in nq:
                raw_hits.append((idx, contains_score, "exact"))
                continue
            if zh and zh in nq:
                raw_hits.append((idx, contains_score, "exact"))
                continue
            if ttl and ttl in nq:
                raw_hits.append((idx, contains_score, "exact"))
                continue
            alias_hit = False
            for a in aliases:
                if a and (a in nq or nq == a or nq in a):
                    raw_hits.append((idx, alias_score, "alias"))
                    alias_hit = True
                    break
            if alias_hit:
                continue
            if ttl and any(tok in nq_tokens for tok in _tokenize(ttl)):
                raw_hits.append((idx, title_score_val, "title"))
                continue
            bm25 = self._bm25_score(nq_tokens, idx)
            if bm25 >= bm25_min:
                clamped = min(bm25, bm25_max)
                raw_hits.append((idx, clamped, "bm25"))

        final_hits: List[RetrievalHit] = []
        for idx, score, mtype in raw_hits:
            cat = self._docs[idx].get("category", "")
            boost = cat_boost.get(cat, 1.0)
            final_hits.append(self._make_hit(idx, score * boost, mtype))
        return final_hits

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        if not query_tokens:
            return 0.0
        doc_toks = self._doc_tokens[doc_idx]
        doc_set = set(doc_toks)
        overlap = sum(1 for t in query_tokens if t in doc_set)
        if overlap == 0:
            return 0.0
        total_docs = max(len(self._doc_tokens), 1)
        idf_sum = 0.0
        for t in query_tokens:
            if t in doc_set:
                df = sum(1 for dt in self._doc_tokens if t in set(dt))
                idf_sum += 1.0 / (1.0 + df / total_docs)
        raw = (overlap / max(len(query_tokens), 1)) * (idf_sum / max(len(query_tokens), 1))
        return min(raw, 1.0)

    def _make_hit(self, idx: int, score: float, match_type: str) -> RetrievalHit:
        doc = self._docs[idx]
        return RetrievalHit(
            id=doc["id"], category=doc.get("category", ""),
            term=doc.get("term", ""), zh_term=doc.get("zh_term"),
            title=doc.get("title", ""), content=doc.get("content", ""),
            score=round(score, 4), match_type=match_type,
            source=doc.get("source", ""), source_type=doc.get("source_type", ""),
            source_url=doc.get("source_url"), source_ref=dict(doc.get("source_ref", {})),
            allowed_usage=list(doc.get("allowed_usage", [])),
            risk_level=doc.get("risk_level", ""), risk_note=doc.get("risk_note"),
            review_status=doc.get("review_status", ""),
            reviewed_by=doc.get("reviewed_by"),
            last_reviewed_at=doc.get("last_reviewed_at"),
            metadata={
                "tags": list(doc.get("tags", [])),
                "version": doc.get("version", ""),
            },
        )

    @staticmethod
    def _deduplicate_and_rank(hits: List[RetrievalHit]) -> List[RetrievalHit]:
        best: Dict[str, RetrievalHit] = {}
        for h in hits:
            if h.id not in best:
                best[h.id] = h
            else:
                existing = best[h.id]
                if h.score > existing.score:
                    best[h.id] = h
                elif h.score == existing.score:
                    if _MATCH_PRIORITY.get(h.match_type, 99) < _MATCH_PRIORITY.get(existing.match_type, 99):
                        best[h.id] = h
        ranked = sorted(best.values(), key=lambda h: (-h.score, _MATCH_PRIORITY.get(h.match_type, 99), h.id))
        return ranked


# ── Singleton (lazy init — KB is NOT loaded at module import time) ─────────────

_service: Optional[RagService] = None


def get_rag_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service
