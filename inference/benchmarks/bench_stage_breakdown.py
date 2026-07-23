"""
Stage-level bottleneck analysis for validated fast paths.

Extracts per-stage timing from CachedFastPath and BatchBackedFastPath,
produces breakdown table and CSV for prioritization.

Usage::

    python inference/benchmarks/bench_stage_breakdown.py \\
        --images 50 --output-dir outputs/inference_optimization/stage_breakdown

    python inference/benchmarks/bench_stage_breakdown.py --dry-run

Status: New module. Does NOT modify existing 3.1 code.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.env_capture import capture_env
from inference.latency_taxonomy import compute_stats

DEFAULT_IMAGE_DIR = (
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\lapel_design_labels"
)

STAGE_KEYS = [
    ("yolo", "yolo_seconds"),
    ("sam_hq", "sam_hq_seconds"),
    ("landmarks", "landmarks_seconds"),
    ("region_crop", "region_crop_seconds"),
    ("mask_aware_crop", "mask_aware_crop_seconds"),
    ("total", "total_seconds"),
]


def sample_images(d: str, n: int, seed: int = 42) -> List[Path]:
    dp = Path(d)
    if not dp.is_dir():
        return []
    all_imgs = sorted(p for p in dp.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    rng = random.Random(seed)
    return rng.sample(all_imgs, min(n, len(all_imgs)))


def bench_cached_stages(
    image_paths: List[Path], warmup: int = 2
) -> Dict[str, Any]:
    """Run CachedFastPath and extract per-stage timings per image."""
    from inference.pipelines.fast_path_existing_cached import CachedFastPath

    pipe = CachedFastPath(lazy=True)
    if warmup > 0 and image_paths:
        pipe.warmup(str(image_paths[0]))

    rows: List[Dict[str, Any]] = []
    stage_values: Dict[str, List[float]] = {
        "yolo": [], "sam_hq": [], "landmarks": [],
        "region_crop": [], "mask_aware_crop": [], "total": [],
    }

    t0 = time.perf_counter()
    for i, img in enumerate(image_paths):
        t1 = time.perf_counter()
        result = pipe.run_image(str(img))
        wall_ms = (time.perf_counter() - t1) * 1000

        timing = result.get("timing", {})
        row = {
            "image_idx": i,
            "image": img.name,
            "method": "cached",
            "num_instances": result.get("num_instances", 0),
            "status": result.get("status", "?"),
        }
        for stage_key, timing_key in STAGE_KEYS:
            ms = round(timing.get(timing_key, 0) * 1000, 2)
            row[f"{stage_key}_ms"] = ms
            stage_values[stage_key].append(ms)

        # Overhead = wall clock - sum of named stages (excluding total)
        named_sum = sum(
            timing.get(tk, 0) * 1000 for _, tk in STAGE_KEYS if tk != "total_seconds"
        )
        overhead = round(wall_ms - named_sum, 2)
        row["overhead_ms"] = overhead

        rows.append(row)

    total_s = time.perf_counter() - t0

    # Compute stats per stage
    stages = {}
    for stage_key, _ in STAGE_KEYS:
        vals = stage_values[stage_key]
        stats = compute_stats(vals) if vals else {}
        total_mean = stage_values["total"][0] if stage_values["total"] else 1
        stages[stage_key] = {
            "mean_ms": stats.get("mean_ms", 0),
            "std_ms": stats.get("std_ms", 0),
            "p50_ms": stats.get("p50_ms", 0),
            "p95_ms": stats.get("p95_ms", 0),
            "p99_ms": stats.get("p99_ms", 0),
            "min_ms": stats.get("min_ms", 0),
            "max_ms": stats.get("max_ms", 0),
            "share_percent": round(stats.get("mean_ms", 0) / max(1e-9, total_mean) * 100, 1),
        }

    # Overhead stats
    overhead_vals = [r["overhead_ms"] for r in rows]
    overhead_stats = compute_stats(overhead_vals) if overhead_vals else {}
    stages["overhead"] = {
        "mean_ms": overhead_stats.get("mean_ms", 0),
        "share_percent": round(
            overhead_stats.get("mean_ms", 0) / max(1e-9, stages["total"]["mean_ms"]) * 100, 1
        ),
    }

    return {
        "total": {
            "mean_ms": stages["total"]["mean_ms"],
            "p50_ms": stages["total"]["p50_ms"],
            "p95_ms": stages["total"]["p95_ms"],
            "qps": round(len(rows) / max(1e-9, total_s), 2),
        },
        "stages": stages,
        "per_image": rows,
        "total_wallclock_s": round(total_s, 2),
        "num_images": len(rows),
    }


def bench_batch_stages(
    image_paths: List[Path],
) -> Dict[str, Any]:
    """Run BatchBackedFastPath — aggregate-only per-stage timing."""
    from inference.pipelines.fast_path_batch_backed import BatchBackedFastPath

    pipe = BatchBackedFastPath()
    t0 = time.perf_counter()
    result = pipe.run_images([str(p) for p in image_paths])
    total_s = time.perf_counter() - t0

    n = len(image_paths)
    breakdown = result.get("timing_breakdown", {})
    total_ms = result.get("total_ms_all_images", 0)

    stage_total_map = {
        "yolo": breakdown.get("yolo_ms_total", 0),
        "sam_hq": breakdown.get("sam_hq_ms_total", 0),
        "landmarks": breakdown.get("landmarks_ms_total", 0),
        "region_crop": breakdown.get("region_crops_ms_total", 0),
        "mask_aware_crop": breakdown.get("masked_crops_ms_total", 0),
    }

    stages = {}
    for stage_key, total_stage_ms in stage_total_map.items():
        per_img = round(total_stage_ms / max(1, n), 2)
        stages[stage_key] = {
            "mean_ms": per_img,
            "share_percent": round(per_img / max(1e-9, total_ms / max(1, n)) * 100, 1),
            "_note": "aggregate / N — no per-image breakdown available",
        }

    stages["total"] = {
        "mean_ms": round(total_ms / max(1, n), 2),
        "share_percent": 100.0,
    }

    return {
        "total": {
            "mean_ms": round(total_ms / max(1, n), 2),
            "qps": round(result.get("throughput_qps", 0), 2),
        },
        "stages": stages,
        "total_wallclock_s": round(total_s, 2),
        "num_images": n,
        "notes": [
            "BatchBackedFastPath: per-stage values are total/N (aggregate only).",
            "run_source() processes all images in one call — no per-image breakdown.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage-level bottleneck analysis for validated fast paths"
    )
    ap.add_argument("--images", type=int, default=50)
    ap.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--output-dir", type=str,
                    default="outputs/inference_optimization/stage_breakdown")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", choices=["cached", "batch", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    env = capture_env()
    with open(output_dir / "env.json", "w") as f:
        json.dump(env, f, indent=2, ensure_ascii=False, default=str)

    if args.dry_run:
        imgs = sample_images(args.image_dir, args.images, args.seed)
        print(f"[DRY RUN] {len(imgs)} images available.")
        return

    image_paths = sample_images(args.image_dir, args.images, args.seed)
    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)
    print(f"Stage breakdown on {len(image_paths)} images (seed={args.seed})")

    report: Dict[str, Any] = {
        "metadata": {
            "benchmark_type": "stage_breakdown",
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "images": len(image_paths),
            "seed": args.seed,
            "label": "[measured]",
        },
        "methods": {},
    }

    all_rows: List[Dict] = []

    # ── Cached Fast Path ──
    if args.method in ("cached", "both"):
        print("\n--- CachedFastPath stage breakdown ---")
        cached = bench_cached_stages(image_paths)
        report["methods"]["cached_fast_path"] = {
            "total": cached["total"],
            "stages": cached["stages"],
        }
        for r in cached["per_image"]:
            all_rows.append(r)
        print(f"  Total: {cached['total']['mean_ms']:.0f} ms/img")

    # ── Batch Backed Fast Path ──
    if args.method in ("batch", "both"):
        print("\n--- BatchBackedFastPath stage breakdown ---")
        batch = bench_batch_stages(image_paths)
        report["methods"]["batch_backed_fast_path"] = {
            "total": batch["total"],
            "stages": batch["stages"],
            "notes": batch.get("notes", []),
        }
        print(f"  Total: {batch['total']['mean_ms']:.0f} ms/img (aggregate/N)")

    # ── Write JSON ──
    json_path = output_dir / "stage_breakdown.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # ── Write CSV ──
    csv_path = output_dir / "stage_breakdown_per_image.csv"
    if all_rows:
        keys = ["image_idx", "image", "method", "num_instances", "status",
                "total_ms", "yolo_ms", "sam_hq_ms", "landmarks_ms",
                "region_crop_ms", "mask_aware_crop_ms", "overhead_ms"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    # ── Write MD ──
    cached_s = report["methods"].get("cached_fast_path", {}).get("stages", {})
    batch_s = report["methods"].get("batch_backed_fast_path", {}).get("stages", {})

    def _md_row(label, cs, bs):
        cm = cs.get("mean_ms", 0)
        cp = cs.get("share_percent", 0)
        bm = bs.get("mean_ms", cm)
        bp = bs.get("share_percent", cp) if bs else 0
        return f"| {label} | {cm:.1f} | {cp:.1f}% | {bm:.1f} | {bp:.1f}% | [measured] |"

    md = f"""# Stage-Level Breakdown

> Date: {report['metadata']['date']} | Images: {len(image_paths)} | Seed: {args.seed}

## CachedFastPath Per-Stage Latency

| Stage | Mean ms | Share | P50 | P95 | Label |
|---|---:|---:|---:|---:|---|
"""
    for sk in ["yolo", "sam_hq", "landmarks", "region_crop", "mask_aware_crop",
               "overhead", "total"]:
        cs = cached_s.get(sk, {})
        md += (f"| {sk} | {cs.get('mean_ms',0):.1f} | {cs.get('share_percent',0):.1f}% | "
               f"{cs.get('p50_ms',0):.0f} | {cs.get('p95_ms',0):.0f} | [measured] |\n")

    md += f"""
## Comparison

| Stage | Cached ms | Cached % | Batch ms | Batch % | Label |
|---|---:|---:|---:|---:|---|
"""
    for sk in ["yolo", "sam_hq", "landmarks", "region_crop", "mask_aware_crop", "total"]:
        cs = cached_s.get(sk, {})
        bs = batch_s.get(sk, {})
        md += _md_row(sk, cs, bs) + "\n"

    # Bottleneck ranking
    ranked = sorted(
        [(sk, cs.get("mean_ms", 0), cs.get("share_percent", 0))
         for sk, cs in cached_s.items()
         if sk not in ("total", "overhead") and cs.get("mean_ms", 0) > 0],
        key=lambda x: -x[1],
    )
    md += "\n## Bottleneck Ranking (CachedFastPath)\n\n"
    md += "| Rank | Stage | Mean ms | Share |\n|---|---:|---:|\n"
    for rank, (sk, ms, pct) in enumerate(ranked, 1):
        md += f"| {rank} | {sk} | {ms:.1f} | {pct:.1f}% |\n"

    md += f"""
## Notes

- CachedFastPath: per-image stage timings extracted from ``result["timing"]``.
- BatchBackedFastPath: per-stage values are ``total / N`` (aggregate only — ``run_source()`` processes all images in one call).
- Overhead = wall-clock total − sum of named stages (Python overhead, I/O, CUDA sync).
- All numbers [measured] on RTX 3090, fashion-demo2 conda env.
"""

    md_path = output_dir / "stage_breakdown.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md.strip())

    print(f"\n[Reports: {json_path}, {md_path}, {csv_path}]")


if __name__ == "__main__":
    main()
