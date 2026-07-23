"""
Synonym dictionary — second level of the LLM preference ladder.

Handles fuzzy matches and common misspellings for Chinese fashion terms.
In-process, <1ms, covers ~3% of queries that miss the exact vocab match.

Status: Skeleton. Synonym entries to be expanded from real query logs.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class SynonymDict:
    """Fuzzy synonym matching for fashion query terms."""

    def __init__(self) -> None:
        # ── Character-level variant mapping ──
        self._char_variants: Dict[str, str] = {
            # Traditional → Simplified
            "領": "领", "褲": "裤", "裙": "裙", "襯": "衬",
            # Common input method artifacts
            "口袋儿": "口袋", "领口儿": "领口",
        }

        # ── Phrase-level reordering ──
        self._phrase_map: Dict[str, str] = {
            "长袖上衣": "long_sleeve_top",
            "短袖上衣": "short_sleeve_top",
            "无袖": "sleeveless",
            "V领": "v_neck",
            "圆领": "round_neck",
        }

    def normalize(self, query: str) -> str:
        """Normalize a query by applying character and phrase variants."""
        q = query.strip()
        for old, new in self._char_variants.items():
            q = q.replace(old, new)
        return q

    def lookup(self, query: str) -> Optional[str]:
        """Look up a normalized query in the phrase map."""
        normalized = self.normalize(query)
        return self._phrase_map.get(normalized)

    @property
    def size(self) -> int:
        return len(self._char_variants) + len(self._phrase_map)


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    sd = SynonymDict()
    assert sd.normalize("褲子") == "裤子"
    print(f"  synonym_dict: {sd.size} entries, OK.")


if __name__ == "__main__":
    _demo()
