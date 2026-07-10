"""
Unified query intent parser for fashion region localization.

Supersedes the two divergent parsers that existed before Phase 0:
  - src/fashion_vision/localization/query_parser.py (parse_region_type)
  - tools/demo/query_region_online_demo.py::infer_region_from_query

Design:
  ``part`` uses the canonical internal names that match region_locator.py and
  the landmark pipeline (neckline, cuff, hem, waist, shoulder, leg_opening).

  ``crop_region`` is a derived property that translates to the "region" field
  values stored in crop records by crop_garment_regions_from_landmarks.py
  (collar, sleeve, hem, waist, pant_leg, shoulder).

  ``component`` gives the legacy left_sleeve/right_sleeve string expected by
  the existing crop-selection logic when a side query targets the cuff.

  ``grounding_text`` gives the English noun phrase for Grounding DINO (Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Parts routed to the existing landmark + geometry pipeline (no new model needed).
# neckline and cuff were migrated to Fashionpedia-first routing (Phase 3);
# they fall back to fast-path only when Fashionpedia YOLO returns empty.
FAST_PATH_PARTS: frozenset[str] = frozenset({
    "hem", "waist", "leg_opening", "shoulder",
})

# Translate internal part name → crop record "region" field value.
# crop_garment_regions_from_landmarks.py writes collar/sleeve/pant_leg naming.
_PART_TO_CROP_REGION: dict[str, str] = {
    "neckline": "collar",
    "cuff": "sleeve",
    "leg_opening": "pant_leg",
}

# English noun phrase for Grounding DINO text input (Phase 2).
# Garment-context prefixes ("clothing zipper" vs bare "zipper") reduce confusion
# with visually similar non-garment objects (necklaces, bags, bag straps).
_PART_TO_GROUNDING_TEXT: dict[str, str] = {
    "neckline": "neckline",
    "cuff": "sleeve cuff",
    "hem": "hem",
    "waist": "waist",
    "shoulder": "shoulder",
    "leg_opening": "leg opening",
    "zipper": "clothing zipper",        # bare "zipper" confuses with necklaces
    "button": "clothing button",        # bare "button" confuses with decorative badges
    "pocket": "clothing pocket",        # bare "pocket" confuses with bags
    "placket": "front placket",         # bare "placket" has low token salience
    "pattern": "fabric pattern",
    "belt": "clothing belt",            # bare "belt" confuses with bag straps
    "collar_stand": "collar stand",
    # Fashionpedia core parts (Phase 3).
    "lapel": "lapel on jacket",
    "epaulette": "shoulder epaulette on garment",
    "buckle": "belt buckle on clothing",
    # Long-tail construction / decorative parts (Phase 1)
    "drawstring": "drawstring on hood",
    "tie_strap": "tie strap on clothing",
    "ruffle": "ruffle trim on clothing",
    "fringe": "fringe on clothing",
    "strap": "clothing strap on garment",
    "bag": "bag handbag backpack on person",
    "shoes": "shoes footwear sneakers",
    # ponytail: seam parts — low expected accuracy without fine-tuning; DINO
    # has limited exposure to close-up intra-garment seam details.
    "shoulder_seam": "shoulder seam on garment",
    "sleeve_seam": "sleeve seam on clothing",
}

# Side words: longer strings first to prevent "左" matching inside "左边".
_SIDE_WORDS: list[tuple[str, str]] = [
    ("左手边", "left"), ("左侧", "left"), ("左边", "left"), ("左", "left"),
    ("右手边", "right"), ("右侧", "right"), ("右边", "right"), ("右", "right"),
    ("left", "left"), ("right", "right"),
]

# Special compound parts checked before generic parts.
# Each entry: (canonical_part, keywords, implied_garment_ref).
# "连衣裙下摆" must come before "裙摆" so the longer token wins.
_SPECIAL_PARTS: list[tuple[str, list[str], Optional[str]]] = [
    ("hem", ["连衣裙下摆", "连衣裙裙摆", "裙装下摆"], "dress"),
    ("hem", ["裙摆", "裙子下摆", "半裙下摆", "裙底", "裙边"], "skirt"),
]

# Regular part vocabulary.  Within each list, longer keywords appear first so
# substring matching is unambiguous (e.g. "sleeve cuff" before "sleeve").
PART_VOCAB: dict[str, list[str]] = {
    "neckline": [
        "衣领", "领口", "领子", "领部", "脖颈", "脖子",
        "neckline", "collar", "neck",
    ],
    "cuff": [
        "袖口", "衣袖", "袖子", "袖部", "袖",
        "sleeve cuff", "sleeve end", "cuff", "sleeve",
    ],
    "hem": [
        "下摆", "衣摆", "底边", "下边",
        "hem", "bottom",
    ],
    "waist": [
        "腰围", "腰线", "腰部", "裤腰", "裙腰", "收腰", "腰",
        "waistline", "waist",
    ],
    "shoulder": [
        "肩部", "肩膀", "肩线",
        "shoulder",
    ],
    "leg_opening": [
        "裤管", "腿部", "裤脚", "裤口", "裤腿",
        "leg opening", "pant leg", "trouser leg", "pant hem", "trouser hem", "pant_leg",
    ],
    # Open-vocabulary parts for the Grounding DINO backend (Phase 2).
    "zipper": ["拉链", "拉锁", "zipper", "zip"],
    "button": ["纽扣", "扣子", "button", "buttons"],
    "pocket": ["口袋", "衣兜", "兜", "pocket"],
    "placket": ["门襟", "placket"],
    "pattern": ["碎花", "花纹", "图案", "印花", "pattern", "print"],
    "belt": ["腰带", "皮带", "belt"],
    "collar_stand": ["领座", "领底", "collar stand"],
    # Fashionpedia core parts (Phase 3 — Fashionpedia-first routing).
    "lapel":       ["翻领", "驳领", "西装领", "lapel"],
    "epaulette":   ["肩章", "肩袢", "epaulette"],
    "buckle":      ["扣环", "皮带扣", "buckle"],
    "strap":       ["吊带", "肩带", "细吊带", "strap"],
    "bag":         ["包", "包包", "手提包", "背包", "挎包", "bag"],
    "shoes":       ["鞋子", "鞋", "鞋履", "shoes", "shoe"],
    # Long-tail construction / decorative parts (Phase 1 addition).
    # "drawstring" was already in PART_DETECTION_CONFIG but unreachable from
    # Chinese queries; adding it here closes that gap.
    "drawstring":    ["抽绳", "收绳", "绳子", "drawstring"],
    "tie_strap":     ["绑带", "系带", "tie strap", "tie"],
    "ruffle":        ["荷叶边", "波浪边", "荷叶裙边", "ruffle", "ruffle trim"],
    "fringe":        ["流苏", "穗子", "穗饰", "fringe"],
    # Intra-garment construction seams — expected low DINO accuracy; results
    # should be treated as uncertain until fine-tuned or verified.
    "shoulder_seam": ["肩缝", "肩线缝合", "肩部缝线", "shoulder seam", "shoulder stitching"],
    "sleeve_seam":   ["袖缝", "袖子缝合", "袖线", "sleeve seam", "sleeve stitching"],
}

# Garment reference words: longer strings first ("连衣裙" before "裙子").
_GARMENT_REF_WORDS: list[tuple[str, str]] = [
    ("上衣外套", "outerwear"), ("连衣裙", "dress"),
    ("外套", "outerwear"), ("外衣", "outerwear"),
    ("半裙", "skirt"), ("裙子", "skirt"),
    ("裤子", "pants"),
    ("内搭", "inner"), ("里面", "inner"),
]

# Direction words: viewer-perspective spatial qualifiers beyond left/right.
# Longer strings first to avoid partial matches.
_DIRECTION_WORDS: list[tuple[str, str]] = [
    ("前胸", "front_upper"), ("胸前", "front_upper"),
    ("上方", "upper"), ("上边", "upper"), ("上部", "upper"),
    ("下方", "lower"), ("下边", "lower"), ("下部", "lower"),
    ("背后", "back"), ("后背", "back"),
]


@dataclass
class QueryIntent:
    """Structured result of parsing a natural language fashion region query."""

    raw_query: str
    part: Optional[str] = None          # canonical internal part name
    side: Optional[str] = None          # "left" / "right" / None
    garment_ref: Optional[str] = None   # "outerwear" / "skirt" / "dress" / "pants" / None
    direction: Optional[str] = None     # "upper" / "lower" / "front_upper" / "back" / None
    spatial_anchor: Optional[str] = None  # part name extracted from "X附近" / "X上的" patterns
    is_fast_path: bool = False          # True → existing landmark pipeline
    is_zero_shot: bool = False          # True → part unknown, zero-shot DINO fallback
    _matched_keyword: Optional[str] = field(default=None, repr=False)

    @property
    def crop_region(self) -> Optional[str]:
        """
        Region name as stored in pipeline crop records.

        crop_garment_regions_from_landmarks.py writes "collar"/"sleeve"/"pant_leg"
        instead of the internal "neckline"/"cuff"/"leg_opening".  Use this when
        searching existing crop JSON records.
        """
        if self.part is None:
            return None
        return _PART_TO_CROP_REGION.get(self.part, self.part)

    @property
    def component(self) -> Optional[str]:
        """
        Legacy component filter for landmark crop-selection.

        Returns "left_sleeve" or "right_sleeve" when the query targets a cuff
        with a side constraint; None otherwise.
        """
        if self.part == "cuff":
            if self.side == "left":
                return "left_sleeve"
            if self.side == "right":
                return "right_sleeve"
        return None

    @property
    def grounding_text(self) -> Optional[str]:
        """English noun phrase suitable as Grounding DINO text input (Phase 2)."""
        if self.part is None:
            return None
        return _PART_TO_GROUNDING_TEXT.get(self.part, self.part)


def _zero_shot_noun_phrase(query: str) -> str:
    """
    Extract a best-effort English-like noun phrase from a Chinese/English query
    for zero-shot Grounding DINO prompting when no vocabulary keyword matched.

    Strips common structural prefixes (garment refs, side words, possessives)
    and returns the remainder.  The result may still be Chinese — callers should
    treat it as best-effort and display a low-confidence warning.

    LLM-based translation can be injected by replacing this function in Phase 3
    without touching the routing logic.

    Args:
        query: Raw user query string.

    Returns:
        Cleaned remainder string, falling back to the full query if stripping
        yields an empty result.
    """
    _STRIP_PREFIXES = [
        "这件衣服的", "衣服的", "这件的",
        "外套的", "内搭的", "裙子的", "连衣裙的", "裤子的", "半裙的",
        "左边的", "右边的", "左侧的", "右侧的", "左手边的", "右手边的",
        "上面的", "下面的", "上方的", "下方的", "胸前的", "前胸的",
        "这件", "那件",
    ]
    q = query.strip()
    for prefix in _STRIP_PREFIXES:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
    # Remove trailing 的
    if q.endswith("的"):
        q = q[:-1].strip()
    return q if q else query.strip()


def parse_intent(query: str) -> QueryIntent:
    """
    Parse a natural language region query into a structured QueryIntent.

    Supports Chinese and English, including side words (左边/right), garment
    references (外套/连衣裙/裤子), direction words (胸前/上方), compound region
    terms (裙摆), spatial anchors (X附近/X上的), and open-vocabulary part names
    (拉链, 口袋).  Unrecognized queries set is_zero_shot=True instead of failing.

    Args:
        query: Natural language region query string.

    Returns:
        QueryIntent with extracted part, side, garment_ref, direction,
        spatial_anchor, is_fast_path, and is_zero_shot flags.
    """
    import re

    q = query.strip()
    ql = q.lower()

    # 1. Side words — longest first to avoid "左" matching inside "左边".
    side: Optional[str] = None
    for kw, direction_val in _SIDE_WORDS:
        if kw.lower() in ql:
            side = direction_val
            break

    # 2. Direction words (up/down/front/back) — longest first.
    direction: Optional[str] = None
    for kw, dir_val in _DIRECTION_WORDS:
        if kw.lower() in ql:
            direction = dir_val
            break

    # 3. Garment reference — longest first to avoid "裙" before "连衣裙".
    garment_ref: Optional[str] = None
    for kw, ref in _GARMENT_REF_WORDS:
        if kw.lower() in ql:
            garment_ref = ref
            break

    # 4. Spatial anchor: match "X附近" or "X上的" patterns, extract anchor part.
    spatial_anchor: Optional[str] = None
    for pattern in (r"(.+?)附近", r"(.+?)上的"):
        m = re.search(pattern, ql)
        if m:
            candidate = m.group(1).strip()
            # Try to match candidate against PART_VOCAB keywords.
            for part_name, keywords in PART_VOCAB.items():
                if any(kw.lower() in candidate for kw in keywords):
                    spatial_anchor = part_name
                    break
            if spatial_anchor:
                break

    # 5. Special compound parts — must precede generic hem to catch 裙摆/连衣裙下摆.
    part: Optional[str] = None
    matched_kw: Optional[str] = None
    for canonical, keywords, implied_ref in _SPECIAL_PARTS:
        for kw in keywords:
            if kw.lower() in ql:
                part = canonical
                matched_kw = kw
                if garment_ref is None and implied_ref is not None:
                    garment_ref = implied_ref
                break
        if part is not None:
            break

    # 6. Regular part vocab — longest-match across ALL parts so that a specific
    #    keyword like "腰带" (belt, 2 chars) beats the shorter "腰" (waist, 1 char)
    #    even though waist appears earlier in PART_VOCAB.
    if part is None:
        best_len = 0
        for part_name, keywords in PART_VOCAB.items():
            for kw in keywords:
                if kw.lower() in ql and len(kw) > best_len:
                    part = part_name
                    matched_kw = kw
                    best_len = len(kw)

    # 7. Routing: fast path for all 6 standard regions, including sided queries.
    # Side selection for fast-path parts (e.g. left cuff) is handled downstream
    # by intent.component → "left_sleeve" / "right_sleeve" in crop selection.
    # Open-vocab parts (zipper, pocket, …) always go to Grounding DINO regardless.
    # Unknown parts (part is None) go to zero-shot DINO fallback.
    is_fast = part in FAST_PATH_PARTS
    is_zero_shot = part is None

    return QueryIntent(
        raw_query=query,
        part=part,
        side=side,
        garment_ref=garment_ref,
        direction=direction,
        spatial_anchor=spatial_anchor,
        is_fast_path=is_fast,
        is_zero_shot=is_zero_shot,
        _matched_keyword=matched_kw,
    )


if __name__ == "__main__":
    _CASES = [
        "领口",
        "左边袖口",              # sided cuff → fast_path=True, component=left_sleeve
        "右侧的袖口",
        "裙摆",
        "连衣裙下摆",
        "外套的拉链",
        "口袋",
        "腰部",
        "裤腿",
        "裤子的裤腿",            # garment_ref=pants → NEW
        "碎花图案",
        "胸前的口袋",            # direction=front_upper → NEW
        "上方的装饰",            # direction=upper → NEW
        "袖口附近的装饰",        # spatial_anchor=cuff → NEW
        "帽兜上的拉链",          # spatial_anchor=? part=zipper → NEW
        "这件衣服的扣子",        # zero_shot=False (button in vocab)
        "肩缝处",               # zero_shot=True → NEW
        "this garment neckline",
        "left sleeve cuff",
    ]
    for _q in _CASES:
        _i = parse_intent(_q)
        print(
            f"{_q!r:30s} → part={_i.part!r:14s} side={_i.side!r:8s}"
            f" ref={_i.garment_ref!r:10s} dir={_i.direction!r:12s}"
            f" anchor={_i.spatial_anchor!r:10s} fast={_i.is_fast_path}"
            f" zero={_i.is_zero_shot}"
        )
