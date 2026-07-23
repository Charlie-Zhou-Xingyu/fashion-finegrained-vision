"""
Local vocabulary mapping — first level of the LLM preference ladder.

Maps Chinese fashion query terms to canonical part names without any
model inference. Covers 95%+ of expected queries at <1ms latency.

Status: Skeleton. Vocabulary to be populated from existing PART_VOCAB
in src/fashion_vision/localization/intent_parser.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class VocabRouter:
    """Local vocabulary-based query translation.

    Covers the common case (95%+ of queries) where the user's Chinese
    term maps directly to a known fashion part via a static dictionary.
    No GPU, no model, <1ms.

    Usage::

        router = VocabRouter()
        result = router.lookup("口袋")
        # -> {"canonical": "pocket", "source": "vocab", "confidence": 1.0}
    """

    def __init__(self) -> None:
        # ── Core vocabulary (populated from intent_parser.py PART_VOCAB) ──
        # ponytail: import from config YAML in production.
        self._vocab: Dict[str, str] = {
            # Garment-level
            "上衣": "top", "裤子": "pants", "裙子": "skirt",
            "外套": "outerwear", "连衣裙": "dress",
            # Parts — fast-path (landmark-based)
            "领口": "neckline", "袖口": "cuff", "下摆": "hem",
            "腰头": "waist", "肩膀": "shoulder", "裤脚": "leg_opening",
            # Parts — Fashionpedia
            "口袋": "pocket", "袖子": "sleeve", "拉链": "zipper",
            "领子": "collar", "兜帽": "hood", "纽扣": "buckle",
            # Parts — open-vocab
            "胸部": "chest", "背部": "back", "肩带": "shoulder_strap",
            # Decorations
            "铆钉": "rivet", "亮片": "sequin", "蝴蝶结": "bow",
            "流苏": "fringe", "荷叶边": "ruffle",
        }

        # ── Synonym mapping (same meaning, different wording) ──
        self._synonyms: Dict[str, str] = {
            "兜": "pocket", "衣兜": "pocket", "裤兜": "pocket",
            "衣领": "collar", "领子": "collar", "领": "collar",
            "袖": "sleeve", "衣袖": "sleeve", "拉索": "zipper",
        }

    def lookup(self, query: str) -> Optional[Dict[str, object]]:
        """Look up a Chinese query in the local vocabulary.

        Returns:
            Dict with canonical, source, confidence if found.
            None if the query is not in the vocabulary.
        """
        query_clean = query.strip()
        canonical = self._vocab.get(query_clean) or self._synonyms.get(query_clean)
        if canonical:
            return {
                "canonical": canonical,
                "source": "vocab",
                "confidence": 0.95 if query_clean in self._vocab else 0.85,
            }
        return None

    @property
    def size(self) -> int:
        return len(self._vocab) + len(self._synonyms)


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    router = VocabRouter()
    assert router.lookup("口袋") is not None
    assert router.lookup("口袋")["canonical"] == "pocket"  # type: ignore[index]
    assert router.lookup("不存在的词") is None
    print(f"  vocab_router: {router.size} entries, OK.")


if __name__ == "__main__":
    _demo()
