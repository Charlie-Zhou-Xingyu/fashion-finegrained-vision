"""
Category mapping utilities for fashion fine-grained vision.

This module defines the unified 8-class target taxonomy required by the
project PRD and provides mapping rules from DeepFashion2 official category
IDs to the unified target taxonomy.

The system-level taxonomy contains 8 categories:
    - top
    - pants
    - skirt
    - outwear
    - dress
    - shoes
    - bag
    - accessory

DeepFashion2 official clothing categories can be mapped to part of this
taxonomy. Categories such as shoes, bag, and accessory require additional
datasets, custom annotations, or open-vocabulary detection modules.

Business note:
    The DeepFashion2 category ``vest`` is mapped to ``outwear`` in this
    project. In fashion e-commerce scenarios, vest may represent an outer
    garment such as a zippered vest or sleeveless jacket. If more fine-grained
    attributes become available later, vest can be further separated into
    inner-wear vest and outer-wear vest.

Scope and ID convention:
    This module is scoped to the **DeepFashion2 ground-truth annotation
    parsing path**.  Category IDs here are **1-based** (1–13), matching the
    raw DeepFashion2 JSON annotation format where ``item_id`` 1 = short
    sleeve top, 2 = long sleeve top, etc.

    This module is used by ``deepfashion2_parser.py`` for GT annotation
    parsing (e.g. SAM-HQ evaluation with GT boxes).  It is **not** used
    for YOLO model inference output.

    For YOLO model output category mapping (0-based IDs, 5-class PRD
    taxonomy), use ``configs/category_mapping.yaml`` loaded via
    ``tools.eval.category_mapping.load_category_mapping()``.

    The two mapping systems differ intentionally:

    * This module: vest (id=5, 1-based) → ``outwear`` (8-class annotation
      taxonomy).
    * ``category_mapping.yaml``: vest (id=4, 0-based) → ``top`` (5-class
      PRD taxonomy per CLAUDE.md policy).

    Both are correct for their respective data paths.  Do not consolidate
    them without a full audit of all callers.
"""

from __future__ import annotations

from typing import Dict


TARGET_CATEGORIES: Dict[str, Dict[str, str]] = {
    "top": {
        "name_en": "top",
        "name_zh": "上衣",
    },
    "pants": {
        "name_en": "pants",
        "name_zh": "裤子",
    },
    "skirt": {
        "name_en": "skirt",
        "name_zh": "裙子",
    },
    "outwear": {
        "name_en": "outwear",
        "name_zh": "外套",
    },
    "dress": {
        "name_en": "dress",
        "name_zh": "连衣裙",
    },
    "shoes": {
        "name_en": "shoes",
        "name_zh": "鞋子",
    },
    "bag": {
        "name_en": "bag",
        "name_zh": "包包",
    },
    "accessory": {
        "name_en": "accessory",
        "name_zh": "配饰",
    },
}


DEEPFASHION2_CATEGORY_ID_TO_NAME: Dict[int, str] = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}


DEEPFASHION2_TO_TARGET_CATEGORY: Dict[int, str] = {
    1: "top",
    2: "top",
    3: "outwear",
    4: "outwear",
    5: "outwear",
    6: "top",
    7: "pants",
    8: "pants",
    9: "skirt",
    10: "dress",
    11: "dress",
    12: "dress",
    13: "dress",
}


OPEN_VOCAB_CATEGORY_PROMPTS: Dict[str, str] = {
    "top": "top, shirt, blouse, t-shirt, upper garment",
    "pants": "pants, trousers, shorts, jeans",
    "skirt": "skirt",
    "outwear": "coat, jacket, outwear, vest, sleeveless jacket",
    "dress": "dress",
    "shoes": "shoes, footwear, sneakers, boots, sandals, heels",
    "bag": "bag, handbag, backpack, purse, shoulder bag, tote bag",
    "accessory": (
        "accessory, hat, belt, scarf, necklace, glasses, watch, "
        "earrings, bracelet"
    ),
}


SUPPLEMENTARY_DATASET_PLAN: Dict[str, list[str]] = {
    "shoes": [
        "UT Zappos",
        "Fashionpedia",
        "OpenImages",
        "LVIS",
        "Custom Label Studio annotations",
    ],
    "bag": [
        "Fashionpedia",
        "ModaNet",
        "OpenImages",
        "LVIS",
        "COCO",
        "Custom Label Studio annotations",
    ],
    "accessory": [
        "Fashionpedia",
        "ModaNet",
        "OpenImages",
        "LVIS",
        "Custom Label Studio annotations",
    ],
}


def get_target_category(category_key: str) -> Dict[str, str]:
    """
    Get target category metadata by category key.

    Args:
        category_key: Unified target category key, such as ``top`` or
            ``dress``.

    Returns:
        A dictionary containing English and Chinese category names.

    Raises:
        KeyError: If the category key is not defined in target taxonomy.
    """
    if category_key not in TARGET_CATEGORIES:
        raise KeyError(f"Unknown target category key: {category_key}")

    return TARGET_CATEGORIES[category_key]


def map_deepfashion2_category(category_id: int) -> Dict[str, object]:
    """
    Map a DeepFashion2 category ID to the unified target taxonomy.

    Args:
        category_id: DeepFashion2 official category ID.

    Returns:
        A dictionary containing raw category information and mapped target
        category information.

    Raises:
        KeyError: If the input category ID is not a valid DeepFashion2
            category ID.
    """
    if category_id not in DEEPFASHION2_CATEGORY_ID_TO_NAME:
        raise KeyError(f"Unknown DeepFashion2 category ID: {category_id}")

    target_key = DEEPFASHION2_TO_TARGET_CATEGORY[category_id]
    target_info = get_target_category(target_key)

    return {
        "raw_category_id": category_id,
        "raw_category_name": DEEPFASHION2_CATEGORY_ID_TO_NAME[category_id],
        "target_category": target_key,
        "target_category_zh": target_info["name_zh"],
    }


def is_supported_target_category(category_key: str) -> bool:
    """
    Check whether a category key belongs to the unified 8-class taxonomy.

    Args:
        category_key: Category key to check.

    Returns:
        True if the category key exists in target taxonomy, otherwise False.
    """
    return category_key in TARGET_CATEGORIES


def get_all_target_categories() -> Dict[str, Dict[str, str]]:
    """
    Get all unified target categories.

    Returns:
        A dictionary of all target categories.
    """
    return TARGET_CATEGORIES.copy()


def get_open_vocab_prompt(category_key: str) -> str:
    """
    Get open-vocabulary text prompt for a target category.

    This function is prepared for later GroundingDINO or open-vocabulary
    detection integration.

    Args:
        category_key: Unified target category key.

    Returns:
        Text prompt for open-vocabulary detection.

    Raises:
        KeyError: If the category key is not supported.
    """
    if category_key not in OPEN_VOCAB_CATEGORY_PROMPTS:
        raise KeyError(f"No open-vocabulary prompt for: {category_key}")

    return OPEN_VOCAB_CATEGORY_PROMPTS[category_key]


def get_supplementary_dataset_plan(category_key: str) -> list[str]:
    """
    Get supplementary dataset plan for categories that are not sufficiently
    covered by DeepFashion2.

    Args:
        category_key: Unified target category key.

    Returns:
        A list of recommended supplementary data sources.

    Raises:
        KeyError: If the category has no supplementary dataset plan.
    """
    if category_key not in SUPPLEMENTARY_DATASET_PLAN:
        raise KeyError(
            f"No supplementary dataset plan defined for: {category_key}"
        )

    return SUPPLEMENTARY_DATASET_PLAN[category_key]
