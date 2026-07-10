#!/usr/bin/env python3
"""
Grounding DINO Tiny vs Base A/B comparison.

Tests both models on the eval_v2 validation set and reports:
  1. Accuracy: IoU>0.3 hit rate per part
  2. Latency: single detect() and multi-prompt detect_multi_prompt()
  3. VRAM: GPU memory usage after model load

Usage:
    # Both models local
    python scripts/compare_dino_tiny_vs_base.py \
        --tiny models/grounding_dino_tiny \
        --base models/grounding_dino_base

    # Use HF hub directly (needs internet)
    python scripts/compare_dino_tiny_vs_base.py \
        --tiny IDEA-Research/grounding-dino-tiny \
        --base IDEA-Research/grounding-dino-base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
from fashion_vision.localization.part_detection_config import (
    PART_DETECTION_CONFIG,
    get_part_prompts,
    get_part_thresholds,
    DEFAULT_BOX_THRESHOLD,
)

# ── Parts to include in the comparison ─────────────────────────────────────
# Focus on parts where tiny has known issues + structural parts for regression check
FOCUS_PARTS: List[str] = [
    # Structural (high-value, must not regress)
    "collar", "neckline", "lapel",
    # Small fasteners (tiny known ceiling)
    "button", "rivet",
    # Decorations (prompt-sensitive)
    "sequin", "fringe", "ruffle", "bow",
    # Accessories (tiny known ceiling)
    "shoes", "bag",
    # Others for coverage
    "zipper", "pocket", "buckle", "epaulette", "hood",
]

PER_RESULT = PROJECT_ROOT / "data" / "validation" / "eval_v2" / "per_result.jsonl"
MAX_SAMPLES_PER_PART = 15  # cap per part (8GB laptop GPU)
LATENCY_WARMUP = 3
LATENCY_REPEATS = 10


def _box_iou(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def load_samples(parts: List[str]) -> Dict[str, List[Dict]]:
    """Load eval samples grouped by part, capped at MAX_SAMPLES_PER_PART."""
    if not PER_RESULT.exists():
        print(f"ERROR: eval data not found at {PER_RESULT}")
        print("Run eval_validation_v2.py first to generate per_result.jsonl")
        sys.exit(1)

    samples: Dict[str, List[Dict]] = defaultdict(list)
    with open(PER_RESULT, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            part = r["part"]
            if part in parts and len(samples[part]) < MAX_SAMPLES_PER_PART:
                samples[part].append(r)
    return dict(samples)


def get_gpu_vram_mb() -> float:
    """Return currently allocated GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def crop_garment(img: np.ndarray, sample: Dict) -> Tuple[np.ndarray, int, int]:
    """Crop image to garment bbox. Returns (crop, offset_x, offset_y)."""
    H, W = img.shape[:2]
    if sample.get("garment_bbox"):
        gx1, gy1, gx2, gy2 = [int(v) for v in sample["garment_bbox"]]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(W, gx2), min(H, gy2)
        return img[gy1:gy2, gx1:gx2], gx1, gy1
    return img, 0, 0


def bench_accuracy(
    locator: GroundingDINOLocator,
    samples: Dict[str, List[Dict]],
    label: str,
) -> Dict[str, Dict[str, Any]]:
    """Run accuracy benchmark. Returns per-part hit counts and rates."""
    results: Dict[str, Dict] = {}
    print(f"\n{'='*60}")
    print(f"ACCURACY: {label}")
    print(f"{'='*60}")

    for part in FOCUS_PARTS:
        part_samples = samples.get(part, [])
        if not part_samples:
            print(f"  {part:<20s}: no samples, skipping")
            continue

        # Get prompts and threshold from part_detection_config
        prompts = get_part_prompts(part)
        box_threshold, _ = get_part_thresholds(part)

        hits = 0
        total = 0
        best_ious: List[float] = []

        for s in part_samples:
            img = cv2.imread(s["image_path"])
            if img is None:
                continue
            crop, ox, oy = crop_garment(img, s)

            gt_bbox = s["gt_bbox"]
            gt_crop = [
                gt_bbox[0] - ox, gt_bbox[1] - oy,
                gt_bbox[2] - ox, gt_bbox[3] - oy,
            ]

            try:
                dets = locator.detect_multi_prompt(
                    crop, prompts, threshold=box_threshold,
                )
            except Exception as e:
                print(f"  WARN: {part} detect failed: {e}")
                dets = []

            best_iou = 0.0
            for d in dets:
                iou = _box_iou(d["bbox_xyxy"], gt_crop)
                if iou > best_iou:
                    best_iou = iou

            best_ious.append(best_iou)
            total += 1
            if best_iou > 0.3:
                hits += 1

        rate = hits / total * 100 if total > 0 else 0
        mean_iou = np.mean(best_ious) if best_ious else 0
        results[part] = {
            "hits": hits, "total": total, "rate": rate,
            "mean_iou": mean_iou, "best_ious": best_ious,
        }
        print(f"  {part:<20s}: {hits:>2d}/{total} @IoU>0.3 = {rate:5.1f}%  mean_IoU={mean_iou:.3f}")

    return results


def bench_latency(
    locator: GroundingDINOLocator,
    label: str,
) -> Dict[str, float]:
    """Benchmark single-detect and multi-prompt latency on a sample image."""
    print(f"\n{'='*60}")
    print(f"LATENCY: {label}")
    print(f"{'='*60}")

    # Use a real garment crop from eval data if available
    test_img = np.zeros((480, 640, 3), dtype=np.uint8) + 128  # grey fallback
    if PER_RESULT.exists():
        with open(PER_RESULT, encoding="utf-8") as f:
            first = json.loads(f.readline())
            img = cv2.imread(first["image_path"])
            if img is not None:
                crop, _, _ = crop_garment(img, first)
                test_img = crop

    # Pick a representative part with multiple prompts
    prompts = get_part_prompts("zipper")  # 4 prompts
    single_prompt = prompts[:1]
    multi_prompts = prompts

    results: Dict[str, float] = {}

    # Warmup
    for _ in range(LATENCY_WARMUP):
        locator.detect(test_img, single_prompt[0], threshold=0.3)
        locator.detect_multi_prompt(test_img, multi_prompts, threshold=0.3)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Single detect
    times = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        locator.detect(test_img, single_prompt[0], threshold=0.3)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    results["single_detect_ms"] = np.mean(times)
    results["single_detect_std_ms"] = np.std(times)
    print(f"  single detect : {results['single_detect_ms']:.1f} ± {results['single_detect_std_ms']:.1f} ms")

    # Multi-prompt detect
    times = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        locator.detect_multi_prompt(test_img, multi_prompts, threshold=0.3)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    results["multi_prompt_ms"] = np.mean(times)
    results["multi_prompt_std_ms"] = np.std(times)
    results["n_prompts"] = len(multi_prompts)
    print(f"  multi-prompt ({len(multi_prompts)} prompts): {results['multi_prompt_ms']:.1f} ± {results['multi_prompt_std_ms']:.1f} ms")

    return results


def print_summary(
    acc_tiny: Dict, acc_base: Dict,
    lat_tiny: Dict, lat_base: Dict,
    vram_tiny: float, vram_base: float,
) -> None:
    """Print final comparison table."""
    print(f"\n\n{'='*80}")
    print("FINAL COMPARISON: Grounding DINO Tiny vs Base")
    print(f"{'='*80}")

    # ── Accuracy ──
    print(f"\n{'Part':<16s} {'Tiny hits':>10s} {'Tiny %':>8s} {'Base hits':>10s} {'Base %':>8s} {'Delta':>8s}")
    print("-" * 62)
    tiny_wins = 0
    base_wins = 0
    for part in FOCUS_PARTS:
        t = acc_tiny.get(part, {})
        b = acc_base.get(part, {})
        if not t or not b:
            continue
        t_rate = t["rate"]
        b_rate = b["rate"]
        delta = b_rate - t_rate
        marker = " <<<" if delta > 5 else (" >>>" if delta < -5 else "")
        print(f"  {part:<14s} {t['hits']:>3d}/{t['total']:<3d} {t_rate:>6.1f}% {b['hits']:>3d}/{b['total']:<3d} {b_rate:>6.1f}% {delta:>+7.1f}%{marker}")
        if delta > 1:
            base_wins += 1
        elif delta < -1:
            tiny_wins += 1
    print(f"\n  Base better on {base_wins} parts, Tiny better on {tiny_wins} parts")

    # ── Collar+neckline+lapel merged ──
    for label, merge_parts in [("collar+neckline+lapel", ["collar", "neckline", "lapel"])]:
        t_hits = sum(acc_tiny.get(p, {}).get("hits", 0) for p in merge_parts)
        t_total = sum(acc_tiny.get(p, {}).get("total", 0) for p in merge_parts)
        b_hits = sum(acc_base.get(p, {}).get("hits", 0) for p in merge_parts)
        b_total = sum(acc_base.get(p, {}).get("total", 0) for p in merge_parts)
        t_rate = t_hits / t_total * 100 if t_total else 0
        b_rate = b_hits / b_total * 100 if b_total else 0
        print(f"  {label:<14s} {t_hits:>3d}/{t_total:<3d} {t_rate:>6.1f}% {b_hits:>3d}/{b_total:<3d} {b_rate:>6.1f}% {b_rate - t_rate:>+7.1f}%")

    # ── Latency ──
    print(f"\n{'Metric':<30s} {'Tiny':>12s} {'Base':>12s} {'Ratio':>8s}")
    print("-" * 62)
    for key, label in [
        ("single_detect_ms", "Single detect"),
        ("multi_prompt_ms", f"Multi-prompt ({lat_tiny.get('n_prompts', '?')} prompts)"),
    ]:
        tv = lat_tiny.get(key, 0)
        bv = lat_base.get(key, 0)
        ratio = bv / tv if tv > 0 else float("inf")
        print(f"  {label:<30s} {tv:>8.1f} ms {bv:>8.1f} ms {ratio:>7.2f}x")

    # ── VRAM ──
    print(f"\n{'Metric':<30s} {'Tiny':>12s} {'Base':>12s} {'Delta':>10s}")
    print("-" * 62)
    print(f"  {'GPU VRAM':<30s} {vram_tiny:>8.0f} MB {vram_base:>8.0f} MB {vram_base - vram_tiny:>+9.0f} MB")

    # ── Verdict ──
    print(f"\n{'─'*60}")
    avg_delta = np.mean([
        acc_base.get(p, {}).get("rate", 0) - acc_tiny.get(p, {}).get("rate", 0)
        for p in FOCUS_PARTS if acc_tiny.get(p) and acc_base.get(p)
    ])
    lat_ratio = lat_base.get("multi_prompt_ms", 0) / max(lat_tiny.get("multi_prompt_ms", 1), 1)
    print(f"Avg accuracy delta: {avg_delta:+.1f}%")
    print(f"Latency ratio (base/tiny): {lat_ratio:.2f}x")
    if avg_delta > 5 and lat_ratio < 2.5:
        print("VERDICT: SWITCH to DINO-base — accuracy gain justifies latency cost")
    elif avg_delta > 2 and lat_ratio < 2.0:
        print("VERDICT: CONSIDER switching — moderate gain at acceptable cost")
    elif avg_delta < 0:
        print("VERDICT: STAY with DINO-tiny — base is worse on accuracy")
    else:
        print("VERDICT: STAY with DINO-tiny — gain too small for latency cost")
        print("  Consider DINO-base only for specific hard parts (shoes, rivet, sequin)")


def _load_and_bench(
    model_id: str, device: str, label: str,
    samples: Optional[Dict[str, List[Dict]]] = None,
) -> Dict[str, Any]:
    """Load one model, run accuracy + latency, unload, return results."""
    print(f"\n{'='*60}")
    print(f"Loading {label}: {model_id}")
    print(f"{'='*60}")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    vram_before = get_gpu_vram_mb()
    loc = GroundingDINOLocator(model_id=model_id, device=device)
    vram = get_gpu_vram_mb() - vram_before
    n_params = sum(p.numel() for p in loc._model.parameters())
    print(f"  {n_params:,} params, VRAM +{vram:.0f} MB")

    result: Dict[str, Any] = {"vram_mb": vram, "n_params": n_params}

    # Accuracy
    if samples is not None:
        result["accuracy"] = bench_accuracy(loc, samples, label)

    # Latency
    result["latency"] = bench_latency(loc, label)

    # Unload
    print(f"  Unloading {label}...")
    del loc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result


def main():
    parser = argparse.ArgumentParser(description="DINO Tiny vs Base A/B comparison")
    parser.add_argument("--tiny", default="models/grounding_dino_tiny",
                        help="Path or HF ID for DINO-tiny")
    parser.add_argument("--base", default="models/grounding_dino_base",
                        help="Path or HF ID for DINO-base")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip accuracy benchmark (latency + VRAM only)")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("WARNING: CUDA not available, falling back to CPU (latency not meaningful)")

    # Load eval samples once
    samples = None
    if not args.skip_accuracy:
        samples = load_samples(FOCUS_PARTS)
        print(f"Loaded {sum(len(v) for v in samples.values())} samples across {len(samples)} parts")

    # ── Run Tiny (load → bench → unload) ─────────────────────────────────
    res_tiny = _load_and_bench(args.tiny, args.device, "DINO-tiny", samples)

    # ── Run Base (load → bench → unload) ─────────────────────────────────
    res_base = _load_and_bench(args.base, args.device, "DINO-base", samples)

    # ── Summary ───────────────────────────────────────────────────────────
    if not args.skip_accuracy:
        acc_tiny = res_tiny.get("accuracy", {})
        acc_base = res_base.get("accuracy", {})
        print_summary(acc_tiny, acc_base,
                      res_tiny["latency"], res_base["latency"],
                      res_tiny["vram_mb"], res_base["vram_mb"])

        # Save CSV
        csv_path = PROJECT_ROOT / "outputs" / "dino_tiny_vs_base.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("part,tiny_hits,tiny_total,tiny_rate,base_hits,base_total,base_rate,delta\n")
            for part in FOCUS_PARTS:
                t = acc_tiny.get(part, {})
                b = acc_base.get(part, {})
                if not t or not b:
                    continue
                f.write(f"{part},{t['hits']},{t['total']},{t['rate']:.1f},{b['hits']},{b['total']},{b['rate']:.1f},{b['rate'] - t['rate']:.1f}\n")
        print(f"\nResults saved to {csv_path}")
    else:
        print(f"\nLatency: tiny={res_tiny['latency']['multi_prompt_ms']:.1f}ms, base={res_base['latency']['multi_prompt_ms']:.1f}ms")
        print(f"VRAM:    tiny={res_tiny['vram_mb']:.0f}MB, base={res_base['vram_mb']:.0f}MB")


if __name__ == "__main__":
    main()
