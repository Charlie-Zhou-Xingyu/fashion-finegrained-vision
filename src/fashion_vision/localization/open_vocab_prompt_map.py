"""
Open-vocabulary prompt map for fashion local region localization.

This module maps normalized region types to English text prompts used by
open-vocabulary detectors such as GroundingDINO, Florence, OWL-ViT, or CLIPSeg.

The prompts are intentionally English because most open-vocabulary detection
models are trained with English text.
"""

from __future__ import annotations

from typing import Dict, List


OPEN_VOCAB_REGION_PROMPTS: Dict[str, List[str]] = {
    # ------------------------------------------------------------------
    # Common local parts
    # ------------------------------------------------------------------
    "pocket": [
        "pocket",
        "clothing pocket",
        "garment pocket",
    ],
    "button": [
        "button",
        "clothing button",
        "garment button",
    ],
    "zipper": [
        "zipper",
        "clothing zipper",
        "zip",
    ],
    "logo": [
        "logo",
        "brand logo",
        "printed logo",
    ],
    "pattern": [
        "pattern",
        "clothing pattern",
        "printed pattern",
        "fabric pattern",
    ],
    "decoration": [
        "decoration",
        "decorative detail",
        "clothing decoration",
        "ornament",
    ],

    # ------------------------------------------------------------------
    # Craft / visual details
    # ------------------------------------------------------------------
    "embroidery": [
        "embroidery",
        "embroidered pattern",
        "embroidered decoration",
    ],
    "print": [
        "print",
        "printed graphic",
        "printed pattern",
    ],
    "lace": [
        "lace",
        "lace trim",
        "lace decoration",
    ],
    "beading": [
        "beading",
        "beaded decoration",
        "beads",
    ],
    "sequin": [
        "sequin",
        "sequins",
        "sequin decoration",
    ],
    "bow": [
        "bow",
        "ribbon bow",
        "decorative bow",
    ],

    # ------------------------------------------------------------------
    # Garment construction parts
    # ------------------------------------------------------------------
    "collar": [
        "collar",
        "shirt collar",
        "clothing collar",
    ],
    "placket": [
        "placket",
        "shirt placket",
        "front placket",
    ],
    "strap": [
        "strap",
        "shoulder strap",
        "clothing strap",
    ],
    "belt": [
        "belt",
        "waist belt",
        "clothing belt",
    ],
    "hood": [
        "hood",
        "clothing hood",
        "jacket hood",
    ],

    # ------------------------------------------------------------------
    # Bag / shoe parts
    # ------------------------------------------------------------------
    "bag_handle": [
        "bag handle",
        "handbag handle",
    ],
    "bag_strap": [
        "bag strap",
        "shoulder bag strap",
    ],
    "shoe_upper": [
        "shoe upper",
        "upper part of shoe",
    ],
    "shoe_sole": [
        "shoe sole",
        "sole of shoe",
    ],
}


REGION_ALIASES: Dict[str, str] = {
    # Chinese aliases
    "口袋": "pocket",
    "袋子": "pocket",
    "纽扣": "button",
    "扣子": "button",
    "拉链": "zipper",
    "logo": "logo",
    "标志": "logo",
    "图案": "pattern",
    "花纹": "pattern",
    "印花": "print",
    "装饰": "decoration",
    "刺绣": "embroidery",
    "蕾丝": "lace",
    "钉珠": "beading",
    "珠饰": "beading",
    "亮片": "sequin",
    "蝴蝶结": "bow",
    "领子": "collar",
    "衣领": "collar",
    "门襟": "placket",
    "肩带": "strap",
    "吊带": "strap",
    "腰带": "belt",
    "帽子": "hood",
    "帽兜": "hood",

    # English aliases
    "pockets": "pocket",
    "buttons": "button",
    "zippers": "zipper",
    "logos": "logo",
    "patterns": "pattern",
    "decorations": "decoration",
    "embroidered": "embroidery",
    "printed": "print",
    "lace trim": "lace",
    "sequins": "sequin",
    "ribbon": "bow",
    "ribbon bow": "bow",
    "collars": "collar",
    "straps": "strap",
    "belts": "belt",
}


def normalize_open_vocab_region(region_type: str) -> str:
    """
    Normalize open-vocabulary region type.

    Args:
        region_type: Raw region type or alias.

    Returns:
        Normalized region type.
    """
    region = region_type.lower().strip().replace(" ", "_")

    if region in OPEN_VOCAB_REGION_PROMPTS:
        return region

    # Try original string alias.
    raw = region_type.strip()
    if raw in REGION_ALIASES:
        return REGION_ALIASES[raw]

    # Try lowercase alias.
    lower = region_type.lower().strip()
    if lower in REGION_ALIASES:
        return REGION_ALIASES[lower]

    # Try underscore to space alias.
    space_form = region.replace("_", " ")
    if space_form in REGION_ALIASES:
        return REGION_ALIASES[space_form]

    return region


def is_open_vocab_region(region_type: str) -> bool:
    """
    Check whether region is supported by open-vocabulary prompt map.

    Args:
        region_type: Region type.

    Returns:
        Whether supported.
    """
    normalized = normalize_open_vocab_region(region_type)
    return normalized in OPEN_VOCAB_REGION_PROMPTS


def get_prompts_for_region(region_type: str) -> List[str]:
    """
    Get text prompts for a region.

    Args:
        region_type: Region type.

    Returns:
        Prompt list.
    """
    normalized = normalize_open_vocab_region(region_type)
    return OPEN_VOCAB_REGION_PROMPTS.get(normalized, [normalized.replace("_", " ")])
