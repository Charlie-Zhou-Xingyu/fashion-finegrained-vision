"""
Per-part Grounding DINO prompt configs and detection thresholds (Phase 2, 3.1.2).

Priority in region_localization_router: this config takes precedence over
open_vocab_prompt_map for any registered part.

Shape config keys (consumed by part_shape_priors.filter_by_shape_priors):
    min/max_area_ratio          - fraction of garment bbox area
    min/max_aspect_ratio_h_over_w
    min/max_aspect_ratio_w_over_h
    prefer_center_x             - reject if center_x drifts > center_x_tolerance
    center_x_tolerance          - fraction of garment width (default 0.30)
    y_band                      - [lo, hi] normalized garment-relative y
    x_band                      - [lo, hi] normalized garment-relative x
"""
from __future__ import annotations

DEFAULT_BOX_THRESHOLD: float = 0.30
DEFAULT_TEXT_THRESHOLD: float = 0.25

# Garment-context prefixes ("clothing zipper" vs bare "zipper") reduce DINO
# confusion with visually similar non-garment objects (bag zippers, etc.).
PART_DETECTION_CONFIG: dict[str, dict] = {
    # ── Small fastener parts (high-precision thresholds) ─────────────────────
    "button": {
        "prompts": [
            "a round plastic button sewn along the front placket of a shirt",
            "a small circular button on the center front opening of a jacket",
            "a button fastener on a shirt cuff at the sleeve end",
            "a row of buttons running vertically down a cardigan front",
            "a wooden button on a coat front closure",
        ],
        "box_threshold": 0.35,
        "text_threshold": 0.30,
        "shape": {
            "max_area_ratio": 0.10,  # relaxed from 0.06; shape_priors was killing all button dets
            # ponytail: prefer_center_x removed — buttons on cuffs and side closures
            # were being falsely rejected. DINO handles button localization fine raw.
            "mask_dilation_px": 3,
        },
    },
    "zipper": {
        "prompts": [
            "a vertical metal zipper line on the front of a jacket",
            "a central zipper closure running down a coat",
            "a long thin zipper on clothing",
            "metal zipper teeth on a garment front",
        ],
        "box_threshold": 0.40,
        "text_threshold": 0.35,
        "shape": {
            "min_aspect_ratio_h_over_w": 1.2,  # relaxed from 1.8; re-calibrate after v3
            "max_area_ratio": 0.30,
            "prefer_center_x": True,
            "mask_dilation_px": 5,   # zipper teeth may straddle the garment edge
        },
    },
    # ── Mid-size structural parts ─────────────────────────────────────────────
    "pocket": {
        "prompts": [
            "a sewn fabric patch pocket on a jacket",
            "a square pocket on the chest of a shirt",
            "a side pocket opening on a pair of pants",
            "a flap pocket stitched onto a garment",
        ],
        "box_threshold": 0.32,
        "text_threshold": 0.28,
        "shape": {
            "min_area_ratio": 0.01,
            "max_area_ratio": 0.25,
        },
    },
    "belt": {
        "prompts": [
            "a waist belt tied around a coat",
            "a fabric belt strap around the waist of a dress",
            "a buckled belt on a garment",
        ],
        "box_threshold": 0.38,
        "text_threshold": 0.32,
        "shape": {
            "min_aspect_ratio_w_over_h": 2.0,
            "y_band": [0.35, 0.75],
            "max_area_ratio": 0.35,
        },
    },
    "placket": {
        "prompts": [
            "the front button placket of a dress shirt",
            "the central front opening on a jacket",
            "a visible placket running down the center of a coat",
        ],
        "box_threshold": 0.35,
        "text_threshold": 0.30,
        "shape": {
            "min_aspect_ratio_h_over_w": 1.5,
            "prefer_center_x": True,
            "max_area_ratio": 0.25,
        },
    },
    "strap": {
        "prompts": [
            "a thin shoulder strap on a dress",
            "a fabric strap on a garment",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "max_area_ratio": 0.35,
        },
    },
    # ── Upper garment parts ───────────────────────────────────────────────────
    "hood": {
        "prompts": [
            "a hood attached to a jacket",
            "a fabric hood on a hoodie sweatshirt",
            "the hood of a coat",
        ],
        "box_threshold": 0.33,
        "text_threshold": 0.28,
        "shape": {
            "y_band": [0.0, 0.45],
            "max_area_ratio": 0.40,
        },
    },
    "collar": {
        "prompts": [
            "a folded fabric collar around the neck of a shirt",
            "a pointed shirt collar with stiff fabric structure",
            "a jacket lapel collar around the neckline area",
            "a structured collar with visible fabric fold at neck",
            "the fabric collar of a coat",
        ],
        "box_threshold": 0.33,
        "text_threshold": 0.28,
        "shape": {
            "y_band": [0.0, 0.30],
            "max_area_ratio": 0.25,
        },
    },
    "neckline": {
        "prompts": [
            "the neckline opening showing skin at the top of a dress",
            "a curved cutout neck opening exposing skin",
            "the edge of the neck opening on a top",
            "skin-visible neckline border of a garment",
            "the neck opening where the garment meets bare skin",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "y_band": [0.0, 0.30],
            "max_area_ratio": 0.20,
        },
    },
    "collar_stand": {
        "prompts": ["collar stand on shirt", "shirt collar stand"],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "y_band": [0.0, 0.30],
            "max_area_ratio": 0.10,
        },
    },
    "epaulette": {
        "prompts": [
            "a structured shoulder epaulette on a military-style shirt",
            "a fabric shoulder tab with a button on a jacket",
            "a decorative epaulette on the shoulder of a coat",
            "a small rectangular fabric flap on the shoulder of a uniform top",
            "a buttoned epaulette strap on the shoulder of a garment",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "y_band": [0.0, 0.25],
            "max_area_ratio": 0.08,
        },
    },
    "cuff": {
        "prompts": [
            "a sleeve cuff at the end of a shirt sleeve",
            "a buttoned cuff on a dress shirt",
            "the ribbed cuff of a jacket sleeve",
        ],
        "box_threshold": 0.33,
        "text_threshold": 0.28,
        "shape": {
            "max_area_ratio": 0.15,
        },
    },
    "sleeve": {
        "prompts": [
            "a long sleeve on a shirt",
            "a jacket sleeve",
            "the sleeve of a garment",
        ],
        "box_threshold": 0.32,
        "text_threshold": 0.27,
        "shape": {
            "max_area_ratio": 0.50,
        },
    },
    # ── Lower garment parts ───────────────────────────────────────────────────
    "hem": {
        "prompts": [
            "the bottom hem of a dress",
            "a stitched hem at the lower edge of a skirt",
            "the finished hemline of a garment",
        ],
        "box_threshold": 0.32,
        "text_threshold": 0.27,
        "shape": {
            "y_band": [0.55, 1.0],
            "max_area_ratio": 0.45,
        },
    },
    "pant_leg": {
        "prompts": [
            "a trouser leg on a pair of pants",
            "the leg opening of a pair of trousers",
            "a pant leg on clothing",
        ],
        "box_threshold": 0.32,
        "text_threshold": 0.27,
        "shape": {
            "y_band": [0.4, 1.0],
            "max_area_ratio": 0.55,
        },
    },
    "buckle": {
        "prompts": [
            "a metal belt buckle at the waist of trousers",
            "a rectangular metal buckle on the front of pants",
            "an adjustable metal strap buckle on a belt",
            "a shiny metal buckle fastening on clothing",
            "a belt buckle with metal frame at waist level",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "y_band": [0.30, 0.65],
            "max_area_ratio": 0.06,
        },
    },
    # ── Surface decorations (lower threshold — patterns are subtle) ───────────
    "logo": {
        "prompts": [
            "a logo printed on a shirt",
            "a brand logo on clothing",
            "a printed logo on a t-shirt",
        ],
        "box_threshold": 0.32,
        "text_threshold": 0.27,
        "shape": {
            "max_area_ratio": 0.12,
        },
    },
    "pattern": {
        "prompts": [
            "a fabric pattern on clothing",
            "a printed pattern on a garment",
            "a textile pattern on a shirt",
        ],
        "box_threshold": 0.28,
        "text_threshold": 0.23,
        "shape": {
            "min_area_ratio": 0.02,
            "max_area_ratio": 0.90,
        },
    },
    "print": {
        "prompts": [
            "a printed graphic on a t-shirt",
            "a screen printed design on clothing",
        ],
        "box_threshold": 0.28,
        "text_threshold": 0.23,
        "shape": {
            "min_area_ratio": 0.01,
            "max_area_ratio": 0.70,
        },
    },
    "embroidery": {
        "prompts": [
            "embroidered decoration on a garment",
            "an embroidered pattern stitched on clothing",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "max_area_ratio": 0.30,
        },
    },
    "decoration": {
        "prompts": [
            "a decorative detail on a garment",
            "an ornamental decoration on clothing",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "max_area_ratio": 0.40,
        },
    },
    "lace": {
        "prompts": [
            "lace trim on a dress",
            "delicate lace decoration on clothing",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "max_area_ratio": 0.35,
        },
    },
    "rivet": {
        "prompts": [
            "metal stud on fabric",
            "small metal dot on clothing",
            "rivet on jeans",
            "metal fastener on denim",
            "round metal rivet on garment",
        ],
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "shape": {
            "max_area_ratio": 0.04,  # relaxed from 0.02; rivets vary in crop size
        },
    },
    "beading": {
        "prompts": [
            "beaded decoration on a garment",
            "a row of beads sewn onto clothing",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "max_area_ratio": 0.25,
        },
    },
    "sequin": {
        "prompts": [
            "shiny iridescent sequin discs sewn onto an evening dress",
            "small glittering sequin embellishments reflecting light on fabric",
            "a cluster of metallic sequin decorations on a garment",
            "sequin patches shimmering on clothing surface",
            "reflective sequin discs attached to dress fabric",
        ],
        "box_threshold": 0.28,
        "text_threshold": 0.23,
        "shape": {
            "max_area_ratio": 0.60,
        },
    },
    "bow": {
        "prompts": [
            "a tied fabric bow at the waistline of a dress",
            "a decorative satin ribbon bow on a garment neckline",
            "a small fabric bow on the shoulder of a top",
            "a knotted decorative bow made of cloth on clothing",
            "a ribbon bow tied on a blouse",
        ],
        "box_threshold": 0.32,
        "text_threshold": 0.27,
        "shape": {
            "max_area_ratio": 0.18,
        },
    },
    # ── Long-tail parts added Phase 1 ─────────────────────────────────────────
    "drawstring": {
        # Was registered here before but unreachable from Chinese queries; now
        # also in PART_VOCAB so "抽绳" routes here correctly.
        "prompts": ["drawstring on hood", "clothing drawstring", "hoodie drawstring"],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "max_area_ratio": 0.15,
        },
    },
    "tie_strap": {
        "prompts": ["tie strap on clothing", "bow tie strap on garment", "fabric tie on clothing"],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "shape": {
            "max_area_ratio": 0.20,
        },
    },
    "ruffle": {
        "prompts": ["ruffle trim on clothing", "ruffle edge on garment", "frilly ruffle on dress"],
        "box_threshold": 0.28,
        "text_threshold": 0.23,
        "shape": {
            "max_area_ratio": 0.45,
        },
    },
    "fringe": {
        "prompts": [
            "tasseled fringe trim dangling along the hemline of a skirt",
            "decorative thread fringe hanging from the edge of clothing",
            "a row of dangling fabric fringe trim on a garment",
            "fringe border decoration swaying at garment bottom edge",
            "thin thread fringe attached to clothing hem",
        ],
        "box_threshold": 0.28,
        "text_threshold": 0.23,
        "shape": {
            "max_area_ratio": 0.30,
        },
    },
    # ponytail: seam parts have low expected accuracy without fine-tuning.
    # DINO has limited pretraining exposure to close-up intra-garment seam detail.
    "shoulder_seam": {
        "prompts": [
            "shoulder seam on garment",
            "seam at shoulder on clothing",
            "shoulder stitching line",
        ],
        "box_threshold": 0.28,
        "text_threshold": 0.23,
        "shape": {
            "y_band": [0.0, 0.30],
            "max_area_ratio": 0.15,
        },
    },
    "sleeve_seam": {
        "prompts": ["sleeve seam on clothing", "arm seam on garment", "sleeve stitching"],
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "shape": {
            "max_area_ratio": 0.25,
        },
    },
    # ── Non-garment accessories (DINO open-vocab path) ────────────────────
    "bag": {
        "prompts": [
            "a handbag held by a person",
            "a backpack worn on the back",
            "a shoulder bag carried by someone",
            "a bag on a fashion model",
        ],
        "box_threshold": 0.25,
        "text_threshold": 0.22,
        "shape": {
            "max_area_ratio": 0.40,
        },
    },
    "shoes": {
        "prompts": [
            "a pair of shoes worn on feet at the bottom of the image",
            "sneakers on a person's feet at the lower frame",
            "leather shoes at the very bottom edge of a fashion photo",
            "footwear visible at the lower portion of the picture",
            "shoes on the ground beneath a standing person",
        ],
        "box_threshold": 0.25,
        "text_threshold": 0.22,
        "shape": {
            "y_band": [0.75, 1.0],  # mentor: shoes only in lower 25%
            "max_area_ratio": 0.30,
        },
    },
}


def get_part_prompts(part: str, fallback_prompt: str | None = None) -> list[str]:
    """Return DINO text prompts for ``part``.

    Args:
        part: Canonical internal part name (e.g. "zipper").
        fallback_prompt: Used when ``part`` is not in PART_DETECTION_CONFIG.
            If also None, the bare part name (underscores → spaces) is used.
    """
    cfg = PART_DETECTION_CONFIG.get(part)
    if cfg:
        return list(cfg["prompts"])
    if fallback_prompt is not None:
        return [fallback_prompt]
    return [part.replace("_", " ")]


def get_part_thresholds(part: str) -> tuple[float, float]:
    """Return ``(box_threshold, text_threshold)`` for ``part``.

    Defaults to ``(DEFAULT_BOX_THRESHOLD, DEFAULT_TEXT_THRESHOLD)`` for
    unknown parts.
    """
    cfg = PART_DETECTION_CONFIG.get(part)
    if cfg:
        return (cfg["box_threshold"], cfg["text_threshold"])
    return (DEFAULT_BOX_THRESHOLD, DEFAULT_TEXT_THRESHOLD)


def get_part_shape_config(part: str) -> dict:
    """Return the shape-prior config dict for ``part`` (may be empty dict)."""
    cfg = PART_DETECTION_CONFIG.get(part)
    if cfg:
        return dict(cfg.get("shape", {}))
    return {}
