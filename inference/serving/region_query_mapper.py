"""
P1.4a — Chinese query to part-type mapping.

Deterministic mapping from Chinese region keywords to standardised part_type
values used in ``localized_regions``.  No NLP, no embeddings, no external deps.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# ── Primary mapping: Chinese keyword → part_type, part_group ──────────────────
#
# ponytail: simple ordered dict; first match wins for overlapping keywords.
# The list is ordered from most specific to most general.

_CHINESE_TO_PART: Dict[str, Tuple[str, str]] = {
    # collar area
    "领口": ("neckline", "collar_area"),       # 领口
    "领子": ("collar", "collar_area"),           # 领子
    "翻领": ("lapel", "collar_area"),            # 翻领
    "立领": ("collar", "collar_area"),           # 立领
    "圆领": ("collar", "collar_area"),           # 圆领
    "V领": ("collar", "collar_area"),                # V领
    # sleeve area
    "袖口": ("cuff", "sleeve_area"),             # 袖口
    "袖子": ("sleeve", "sleeve_area"),           # 袖子
    "袖缘": ("cuff", "sleeve_area"),             # 袖缘
    # hem
    "下摆": ("hem", "hem_area"),                 # 下摆
    "衣摆": ("hem", "hem_area"),                 # 衣摆
    "裙摆": ("hem", "hem_area"),                 # 裙摆
    # pocket
    "口袋": ("pocket", "pocket_area"),           # 口袋
    "兆袋": ("pocket", "pocket_area"),           # 兜袋
    # shoulder
    "肩章": ("epaulette", "shoulder_area"),      # 肩章
    "肩部": ("shoulder", "shoulder_area"),       # 肩部
    "肩带": ("strap", "shoulder_area"),          # 肩带
    # waist
    "腰部": ("waist", "waist_area"),             # 腰部
    "腰带扣": ("buckle", "waist_area"),      # 腰带扣
    "腰带": ("waist", "waist_area"),             # 腰带
    # closure
    "拉链": ("zipper", "closure"),               # 拉链
    "扣子": ("button", "closure"),               # 扣子
    "纽扣": ("button", "closure"),               # 纽扣
    "按扣": ("button", "closure"),               # 按扣
    "扣环": ("buckle", "closure"),               # 扣环
    # hood
    "帽兜": ("hood", "collar_area"),             # 帽兜
    "帽子": ("hood", "collar_area"),             # 帽子
    "连帽": ("hood", "collar_area"),             # 连帽
    # decorations
    "蝴蝶结": ("bow", "decoration"),         # 蝴蝶结
    "丝带": ("ribbon", "decoration"),            # 丝带
    "褶边": ("ruffle", "decoration"),            # 褶边
    "流苏": ("tassel", "decoration"),            # 流苏
    "亮片": ("sequin", "decoration"),            # 亮片
    "珠子": ("bead", "decoration"),              # 珠子
    "铆钉": ("rivet", "decoration"),             # 铆钉
    "贴花": ("applique", "decoration"),          # 贴花
    "花朵装饰": ("flower", "decoration"),  # 花朵装饰
    "花朵": ("flower", "decoration"),            # 花朵
    # pattern
    "图案": ("pattern", "pattern_area"),         # 图案
    "花纹": ("pattern", "pattern_area"),         # 花纹
    # generic decoration
    "装饰": ("decoration", "decoration"),        # 装饰
    "细节": ("decoration", "decoration"),        # 细节
    "局部设计": ("decoration", "decoration"),  # 局部设计
    # accessories
    "鞋子": ("shoes", "accessory"),              # 鞋子
    "鞋": ("shoes", "accessory"),                    # 鞋
    "包": ("bag", "accessory"),                      # 包
}


def extract_requested_region_part(query: str) -> Optional[str]:
    """Return the standardised ``part_type`` for a Chinese query, or None.

    Scans *query* for known Chinese region keywords.  First match wins
    (longer/more-specific keywords appear earlier in the mapping).

    Returns the ``part_type`` string (e.g. ``"neckline"``), not the Chinese
    keyword or the (part_type, part_group) tuple.

    Examples::

        >>> extract_requested_region_part("领口在哪里？")
        'neckline'
        >>> extract_requested_region_part("有没有口袋？")
        'pocket'
        >>> extract_requested_region_part("这是什么颜色？")
        None
    """
    if not query:
        return None
    for keyword, (part_type, _part_group) in _CHINESE_TO_PART.items():
        if keyword in query:
            return part_type
    return None


def extract_requested_region_part_with_group(query: str) -> Optional[Tuple[str, str]]:
    """Return ``(part_type, part_group)`` for a Chinese query, or None.

    Same matching logic as ``extract_requested_region_part`` but returns both
    the part type and its semantic group.
    """
    if not query:
        return None
    for keyword, (part_type, part_group) in _CHINESE_TO_PART.items():
        if keyword in query:
            return (part_type, part_group)
    return None


# ── Part group lookup ────────────────────────────────────────────────────────

_PART_TYPE_TO_GROUP: Dict[str, str] = {
    part_type: part_group
    for _kw, (part_type, part_group) in _CHINESE_TO_PART.items()
}


def part_group_for(part_type: str) -> str:
    """Return the semantic group for a known *part_type*, or ``"unknown"``."""
    return _PART_TYPE_TO_GROUP.get(part_type, "unknown")
