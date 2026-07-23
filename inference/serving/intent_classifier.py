"""
P0a.2 — Rule-based intent classifier.

Loads intent rules from ``configs/intent_taxonomy.yaml`` and classifies a
natural-language query into a primary / sub intent via keyword and regex
matching.  No embedding models, ONNX, or LLM dependencies.

Typical latency: < 200 µs per call (pure string operations).

The taxonomy splits ``primary_intent`` (top-level: ``attribute_query``,
``knowledge_qa``, ``styling_advice``, …) from ``sub_intent`` (detail:
``fabric``, ``collar``, ``match``, …).  Downstream routing (P0a.5
QaOrchestrator) will dispatch on the pair ``(primary_intent, sub_intent)``.

Usage::

    from inference.serving.intent_classifier import RuleIntentClassifier

    classifier = RuleIntentClassifier()
    result = classifier.classify("这件衣服是什么面料？")
    # -> IntentClassifyData(
    #        primary_intent="attribute_query", sub_intent="fabric",
    #        intent_confidence=0.95, classifier_level="rule",
    #    )
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from inference.serving.schemas import IntentClassifyData

logger = logging.getLogger(__name__)

# Resolve project root relative to this file.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TAXONOMY_PATH = _PROJECT_ROOT / "configs" / "intent_taxonomy.yaml"

_REQUIRED_ENTRY_KEYS = {"primary_intent"}


def _validate_taxonomy(data: Dict[str, Any]) -> List[str]:
    """Perform basic structural validation on the loaded taxonomy.

    Returns a list of warning / error messages.  An empty list means OK.
    """
    issues: List[str] = []
    intents = data.get("intents")
    if not isinstance(intents, list):
        issues.append("taxonomy.intents must be a list")
        return issues

    for i, entry in enumerate(intents):
        if not isinstance(entry, dict):
            issues.append(f"intents[{i}] is not a dict: {entry}")
            continue

        # Required keys.
        for key in _REQUIRED_ENTRY_KEYS:
            if key not in entry or not entry[key]:
                issues.append(f"intents[{i}] missing required key '{key}'")

        # keywords must be list[str].
        keywords = entry.get("keywords", [])
        if not isinstance(keywords, list):
            issues.append(f"intents[{i}].keywords must be a list, got {type(keywords).__name__}")

        # patterns must be list[str].
        patterns = entry.get("patterns", [])
        if not isinstance(patterns, list):
            issues.append(f"intents[{i}].patterns must be a list, got {type(patterns).__name__}")

    return issues


def _load_taxonomy(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate the intent taxonomy YAML.

    Returns a dict with ``intents`` and ``default_intent`` keys.
    On failure, logs an error and returns a minimal fallback that only
    produces ``fallback_unknown``.  The service will **start** but every
    query will land in the fallback.
    """
    config_path = path or _DEFAULT_TAXONOMY_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        logger.error("Intent taxonomy not found: %s — all queries will fallback", config_path)
        return {"intents": [], "default_intent": "fallback_unknown"}
    except yaml.YAMLError as exc:
        logger.error("Intent taxonomy YAML parse error in %s: %s", config_path, exc)
        return {"intents": [], "default_intent": "fallback_unknown"}
    except OSError as exc:
        logger.error("Cannot read intent taxonomy %s: %s", config_path, exc)
        return {"intents": [], "default_intent": "fallback_unknown"}

    if not isinstance(data, dict) or "intents" not in data:
        logger.error("Intent taxonomy at %s is missing 'intents' key", config_path)
        return {"intents": [], "default_intent": "fallback_unknown"}

    # Structural validation.
    issues = _validate_taxonomy(data)
    if issues:
        for msg in issues:
            logger.warning("Intent taxonomy validation: %s", msg)
        # Continue with whatever is loadable — a broken rule is better than
        # no rules, but we want operators to see the warnings.

    return data


class RuleIntentClassifier:
    """Classify a Chinese fashion query via keyword and regex rules.

    The classifier evaluates rules top-to-bottom from the taxonomy YAML.
    The first rule whose keyword or pattern matches the query wins.

    No embedding models, ONNX runtime, or LLM calls are involved —
    classification completes in < 200 µs on commodity hardware.

    Taxonomy loading errors are logged and the classifier silently degrades
    to returning ``fallback_unknown`` for every query.
    """

    KEYWORD_CONFIDENCE = 0.95
    PATTERN_CONFIDENCE = 0.80
    FALLBACK_CONFIDENCE = 0.0
    CLASSIFIER_LEVEL = "rule"

    def __init__(self, taxonomy_path: Optional[Path] = None) -> None:
        taxonomy = _load_taxonomy(taxonomy_path)
        self._intents: List[Dict[str, Any]] = taxonomy.get("intents", [])
        self._default_intent: str = taxonomy.get("default_intent", "fallback_unknown")

        # Pre-compile regex patterns for speed (one-time cost at import).
        self._compiled: List[Tuple[Dict[str, Any], List[re.Pattern]]] = []
        for entry in self._intents:
            patterns = [re.compile(p) for p in entry.get("patterns", [])]
            self._compiled.append((entry, patterns))

        logger.info(
            "RuleIntentClassifier loaded: %d intents, %d pre-compiled patterns",
            len(self._intents),
            sum(len(ps) for _, ps in self._compiled),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, query: str) -> IntentClassifyData:
        """Classify *query* and return an ``IntentClassifyData``.

        Returns ``fallback_unknown`` (primary_intent only) with confidence 0.0
        when no rule matches.
        """
        query_stripped = query.strip()
        if not query_stripped:
            return self._fallback()

        for entry, patterns in self._compiled:
            # 1. Keyword match (highest confidence).
            for kw in entry.get("keywords", []):
                if kw in query_stripped:
                    return IntentClassifyData(
                        primary_intent=entry["primary_intent"],
                        sub_intent=entry.get("sub_intent"),
                        intent_confidence=self.KEYWORD_CONFIDENCE,
                        classifier_level=self.CLASSIFIER_LEVEL,
                        entities=self._extract_entities(entry, query_stripped),
                    )

            # 2. Regex pattern match.
            for pat in patterns:
                if pat.search(query_stripped):
                    return IntentClassifyData(
                        primary_intent=entry["primary_intent"],
                        sub_intent=entry.get("sub_intent"),
                        intent_confidence=self.PATTERN_CONFIDENCE,
                        classifier_level=self.CLASSIFIER_LEVEL,
                        entities=self._extract_entities(entry, query_stripped),
                    )

        # 3. Nothing matched.
        return self._fallback()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _fallback(self) -> IntentClassifyData:
        """Return the default fallback_unknown result."""
        return IntentClassifyData(
            primary_intent=self._default_intent,
            sub_intent=None,
            intent_confidence=self.FALLBACK_CONFIDENCE,
            classifier_level=self.CLASSIFIER_LEVEL,
        )

    @staticmethod
    def _extract_entities(entry: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Extract simple named entities from the query.

        This is a lightweight heuristic — not a full NER model.  P0a.5
        QaOrchestrator will use primary_intent / sub_intent for routing;
        entities are supplementary hints.

        Currently supported entity types:
            color        — e.g. "红色", "蓝色", "黑色", "白色"
            garment_ref  — e.g. "外套"→outerwear, "裤子"→pants
            fabric       — e.g. "棉", "麻", "丝", "羊毛"
            attribute_name — inferred from sub_intent (e.g. "fabric", "collar")

        Returns a dict; may be empty.
        """
        entities: Dict[str, Any] = {}

        # Color extraction
        colors = ["红色", "蓝色", "黑色", "白色", "绿色", "黄色", "紫色",
                   "粉色", "灰色", "棕色", "橙色", "米色", "卡其色"]
        for c in colors:
            if c in query:
                entities["color"] = c
                break

        # Garment-reference extraction
        garment_hints = {
            "外套": "outerwear", "上衣": "top", "裤子": "pants",
            "裙子": "skirt", "连衣裙": "dress", "衬衫": "shirt",
            "T恤": "tshirt", "卫衣": "hoodie", "夹克": "jacket",
        }
        for zh, en in garment_hints.items():
            if zh in query:
                entities["garment_ref"] = en
                break

        # Fabric extraction (partial list)
        fabric_hints = ["棉", "麻", "丝", "羊毛", "羊绒", "涤纶", "尼龙", "牛仔"]
        for f in fabric_hints:
            if f in query:
                entities["fabric"] = f
                break

        # Attribute name hint from the matched sub_intent.
        sub = entry.get("sub_intent")
        if sub:
            entities["attribute_name"] = sub

        # Pass through the primary_intent for downstream routing.
        entities["primary_intent"] = entry.get("primary_intent", "")
        return entities

    @property
    def intent_count(self) -> int:
        """Number of registered intent rules."""
        return len(self._intents)

    @property
    def default_intent(self) -> str:
        """The fallback intent used when no rule matches."""
        return self._default_intent


# ── Module-level convenience ────────────────────────────────────────────────────

# Singleton for use by the FastAPI app.  Loaded once at import time.
_classifier: Optional[RuleIntentClassifier] = None


def get_classifier() -> RuleIntentClassifier:
    """Return the process-wide ``RuleIntentClassifier`` singleton.

    Loads taxonomy on first call; subsequent calls return the cached instance.
    """
    global _classifier
    if _classifier is None:
        _classifier = RuleIntentClassifier()
    return _classifier
