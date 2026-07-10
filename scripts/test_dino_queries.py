#!/usr/bin/env python3
"""
Quick DINO query A/B test for shoes, button, rivet, sequin.
Runs different text prompts on same GT images, compares hit rate.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict, List

import cv2, torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PER_RESULT = PROJECT_ROOT / "data/validation/eval_v2/per_result.jsonl"

# ── Prompt variants to test ────────────────────────────────────────────────────

QUERY_VARIANTS: Dict[str, List[Dict]] = {
    "shoes": [
        {
            "name": "v0_current",
            "prompts": [
                "a pair of shoes worn on feet at the bottom of the image",
                "sneakers on a person's feet at the lower frame",
                "leather shoes at the very bottom edge of a fashion photo",
                "footwear visible at the lower portion of the picture",
                "shoes on the ground beneath a standing person",
            ],
        },
        {
            "name": "v1_no_position",
            "prompts": [
                "a pair of shoes on feet",
                "sneakers worn by a person",
                "high heels on a fashion model",
                "footwear on the ground",
            ],
        },
        {
            "name": "v2_brand_style",
            "prompts": [
                "shoes",
                "sneakers",
                "footwear",
                "high heels",
                "boots on feet",
            ],
        },
        {
            "name": "v3_toe_heel_partial",
            "prompts": [
                "the toe of a shoe",
                "the heel of a shoe",
                "a shoe sole",
                "footwear at the edge of the image",
                "the tip of a shoe peeking into frame",
            ],
        },
        {
            "name": "v4_bare_minimal",
            "prompts": [
                "shoe",
                "foot",
                "sneaker",
                "heel",
                "toe of footwear",
            ],
        },
    ],
    "button": [
        {
            "name": "v0_current",
            "prompts": [
                "a round plastic button sewn along the front placket of a shirt",
                "a small circular button on the center front opening of a jacket",
                "a button fastener on a shirt cuff at the sleeve end",
                "a row of buttons running vertically down a cardigan front",
                "a wooden button on a coat front closure",
            ],
        },
        {
            "name": "v1_simple",
            "prompts": [
                "a button on clothing",
                "a round button on a shirt",
                "small buttons on a jacket",
                "button fastener",
            ],
        },
        {
            "name": "v2_closeup",
            "prompts": [
                "button",
                "small round button",
                "circular button",
                "clothing button",
                "sewn button on garment",
            ],
        },
        {
            "name": "v3_contrast",
            "prompts": [
                "a dark button on a light shirt",
                "a white button on dark fabric",
                "a contrasting button on clothing",
                "a visible button on a garment front",
                "a distinct round button on fabric",
            ],
        },
        {
            "name": "v4_single_word",
            "prompts": [
                "button",
                "button.",
                "a button.",
            ],
        },
    ],
    "rivet": [
        {
            "name": "v0_current",
            "prompts": [
                "a small round metal rivet on the corner of a jeans pocket",
                "a copper rivet reinforcing a denim seam on clothing",
                "a flat metal stud rivet attached to garment fabric",
                "a tiny metallic rivet fastener on clothing",
                "a decorative metal rivet embedded in denim",
            ],
        },
        {
            "name": "v1_simple",
            "prompts": [
                "rivet on clothing",
                "metal rivet on jeans",
                "small metal stud on garment",
                "decorative rivet",
            ],
        },
        {
            "name": "v2_metal",
            "prompts": [
                "metal stud",
                "small metal dot on fabric",
                "rivet",
                "metal fastener on denim",
                "round metal rivet",
            ],
        },
        {
            "name": "v3_tiny_circle",
            "prompts": [
                "small metal circle on fabric",
                "tiny metallic dot on clothing",
                "a metal spot on jeans",
                "circular metal stud on denim",
                "small shiny dot on garment",
            ],
        },
        {
            "name": "v4_bare_metal",
            "prompts": [
                "metal dot",
                "silver dot",
                "brass stud",
                "copper dot on fabric",
                "tiny metal circle",
            ],
        },
    ],
    "sequin": [
        {
            "name": "v0_current",
            "prompts": [
                "shiny iridescent sequin discs sewn onto an evening dress",
                "small glittering sequin embellishments reflecting light on fabric",
                "a cluster of metallic sequin decorations on a garment",
                "sequin patches shimmering on clothing surface",
                "reflective sequin discs attached to dress fabric",
            ],
        },
        {
            "name": "v1_simple",
            "prompts": [
                "sequins on a dress",
                "sequin decorations on clothing",
                "shiny sequins on fabric",
            ],
        },
        {
            "name": "v2_glitter",
            "prompts": [
                "sequin",
                "glittering sequin",
                "shiny small disc on dress",
                "sparkly sequin embellishment",
                "metallic sequin on garment",
            ],
        },
        {
            "name": "v3_sparkle_reflection",
            "prompts": [
                "sparkling decoration on fabric",
                "glittery surface on clothing",
                "light reflection on shiny fabric dots",
                "shimmering embellishment on dress",
                "glittering texture on garment",
            ],
        },
    ],
}


def _box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def load_dino():
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    model_path = str(PROJECT_ROOT / "models/grounding_dino_tiny")
    locator = GroundingDINOLocator(model_id=model_path, device="cuda")
    return locator


def main():
    # Load samples from per_result
    results = [json.loads(l) for l in open(PER_RESULT, encoding='utf-8')]

    # Group by part
    samples: Dict[str, List] = {}
    for r in results:
        part = r["part"]
        if part not in QUERY_VARIANTS:
            continue
        if part not in samples:
            samples[part] = []
        samples[part].append(r)

    # Kick sequin to DINO path (was FP_CORE in eval)
    # We test it anyway

    print("Loading DINO...")
    dino = load_dino()
    print("DINO loaded.\n")

    for part, variants in QUERY_VARIANTS.items():
        part_samples = samples[part][:20]  # max 20 per part
        print(f"{'='*60}")
        print(f"PART: {part} ({len(part_samples)} samples)")
        print(f"{'='*60}")

        for variant in variants:
            name = variant["name"]
            prompts = variant["prompts"]
            hits = 0
            total = 0

            for s in part_samples:
                img = cv2.imread(s["image_path"])
                if img is None:
                    continue
                H, W = img.shape[:2]

                # Crop to garment if available
                if s.get("garment_bbox"):
                    gx1, gy1, gx2, gy2 = [int(v) for v in s["garment_bbox"]]
                    gx1, gy1 = max(0, gx1), max(0, gy1)
                    gx2, gy2 = min(W, gx2), min(H, gy2)
                    crop = img[gy1:gy2, gx1:gx2]
                    ox, oy = gx1, gy1
                else:
                    crop = img
                    ox, oy = 0, 0

                gt_bbox = s["gt_bbox"]
                # Shift GT to crop coords
                gt_crop = [gt_bbox[0] - ox, gt_bbox[1] - oy,
                           gt_bbox[2] - ox, gt_bbox[3] - oy]

                # Run DINO
                try:
                    dets = dino.detect_multi_prompt(crop, prompts, threshold=0.25)
                except Exception:
                    dets = []

                # Check best IoU
                best_iou = 0.0
                for d in dets:
                    b = d["bbox_xyxy"]
                    iou = _box_iou(b, gt_crop)
                    if iou > best_iou:
                        best_iou = iou

                total += 1
                if best_iou > 0.3:
                    hits += 1

            acc = hits / total * 100 if total > 0 else 0
            print(f"  {name:<20s}: {hits:>2d}/{total} hits @IoU>0.3 = {acc:.1f}%  prompts={prompts[:2]}...")

        print()


if __name__ == "__main__":
    main()
