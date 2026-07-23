"""
Benchmark: Original run_image vs CachedFastPath vs BatchBackedFastPath.

Compares three approaches without modifying any existing 3.1 code.

Usage::

    python inference/benchmarks/bench_cached_fast_path.py \\
        --images 10 --output-dir outputs/inference_optimization/cached_fast_path

    python inference/benchmarks/bench_cached_fast_path.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.env_capture import capture_env

DEFAULT_IMAGE_DIR = (
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\lapel_design_labels"
)


def sample_images(d: str, n: int, seed: int = 42) -> List[Path]:
    d = Path(d)
    if not d.is_dir():
        return []
    all_imgs = sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    rng = random.Random(seed)
    return rng.sample(all_imgs, min(n, len(all_imgs)))


def bench_original_run_image(
    image_paths: List[Path], warmup: int = 1
) -> Dict[str, Any]:
    """Benchmark original GarmentPipeline.run_image() — repeated calls."""
    from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig
    import tempfile

    config = GarmentPipelineConfig(
        run_landmark_and_crops=True,
        run_attribute_inference=False,
        save_yolo_vis=False,
        save_yolo_crops=False,
    )

    with tempfile.TemporaryDirectory(prefix="orig_") as td:
        tmp = Path(td)
        pipeline = GarmentPipeline(config)

        # Warmup
        if warmup > 0:
            for _ in range(warmup):
                pipeline.run_image(str(image_paths[0]), str(tmp / "warmup"))

        latencies: List[float] = []
        t0 = time.perf_counter()
        for i, img in enumerate(image_paths):
            out = tmp / f"img_{i:04d}"
            out.mkdir(parents=True, exist_ok=True)
            t1 = time.perf_counter()
            pipeline.run_image(str(img), str(out))
            latencies.append((time.perf_counter() - t1) * 1000)
        total_s = time.perf_counter() - t0

    return {
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "total_wallclock_s": round(total_s, 2),
        "throughput_qps": round(len(latencies) / max(1e-9, total_s), 2),
        "num_images": len(latencies),
    }


def bench_cached_fast_path(
    image_paths: List[Path], warmup: int = 1
) -> Dict[str, Any]:
    """Benchmark CachedFastPath.run_image() — true cached models."""
    from inference.pipelines.fast_path_existing_cached import CachedFastPath

    pipe = CachedFastPath(lazy=True)
    if warmup > 0 and image_paths:
        pipe.warmup(str(image_paths[0]))

    latencies: List[float] = []
    t0 = time.perf_counter()
    for img in image_paths:
        t1 = time.perf_counter()
        pipe.run_image(str(img))
        latencies.append((time.perf_counter() - t1) * 1000)
    total_s = time.perf_counter() - t0

    return {
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "total_wallclock_s": round(total_s, 2),
        "throughput_qps": round(len(latencies) / max(1e-9, total_s), 2),
        "num_images": len(latencies),
    }


def bench_batch_backed(
    image_paths: List[Path],
) -> Dict[str, Any]:
    """Benchmark BatchBackedFastPath.run_images() — single run_source() call."""
    from inference.pipelines.fast_path_batch_backed import BatchBackedFastPath

    pipe = BatchBackedFastPath()

    t0 = time.perf_counter()
    result = pipe.run_images([str(p) for p in image_paths])
    total_s = time.perf_counter() - t0

    return {
        "mean_ms": result["per_image_ms_avg"],
        "total_wallclock_s": round(total_s, 2),
        "throughput_qps": result["throughput_qps"],
        "num_images": result["num_images"],
        "pipeline_total_ms": result["total_ms_all_images"],
        "timing_breakdown": result.get("timing_breakdown", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark cached vs uncached Fast Path")
    ap.add_argument("--images", type=int, default=10)
    ap.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--output-dir", type=str,
                    default="outputs/inference_optimization/cached_fast_path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    env = capture_env()
    with open(output_dir / "env.json", "w") as f:
        json.dump(env, f, indent=2, ensure_ascii=False, default=str)

    if args.dry_run:
        paths_ok = {
            "image_dir": Path(args.image_dir).is_dir(),
            "images_available": len(sample_images(args.image_dir, args.images, args.seed)),
        }
        print("[DRY RUN]")
        for k, v in paths_ok.items():
            print(f"  [{'OK' if v else 'MISSING'}] {k}")
        if all(paths_ok.values()):
            print("  All paths OK. Ready to run real benchmark.")
        return

    # Sample images
    image_paths = sample_images(args.image_dir, args.images, args.seed)
    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)
    print(f"Sampled {len(image_paths)} images.")

    # ── Method 1: Original run_image ──
    print("\n--- Method 1: Original GarmentPipeline.run_image() ---")
    m1 = bench_original_run_image(image_paths, warmup=args.warmup)
    print(f"  Mean: {m1['mean_ms']:.0f} ms/img, QPS: {m1['throughput_qps']:.1f}")

    # ── Method 2: Cached Fast Path ──
    print("\n--- Method 2: CachedFastPath.run_image() ---")
    m2 = bench_cached_fast_path(image_paths, warmup=args.warmup)
    print(f"  Mean: {m2['mean_ms']:.0f} ms/img, QPS: {m2['throughput_qps']:.1f}")

    # ── Method 3: Batch-Backed Fast Path ──
    print("\n--- Method 3: BatchBackedFastPath.run_images() ---")
    m3 = bench_batch_backed(image_paths)
    print(f"  Per-image avg: {m3['mean_ms']:.0f} ms, QPS: {m3['throughput_qps']:.1f}")

    # ── Compute speedups ──
    speedup_cached = m1["mean_ms"] / max(1e-9, m2["mean_ms"])
    speedup_batch = m1["mean_ms"] / max(1e-9, m3["mean_ms"])

    print(f"\n=== SPEEDUP ===")
    print(f"  CachedFastPath vs original:    {speedup_cached:.1f}×")
    print(f"  BatchBackedFastPath vs original: {speedup_batch:.1f}×")

    # ── Build report ──
    report = {
        "metadata": {
            "benchmark_type": "cached_fast_path_comparison",
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "images": args.images,
            "seed": args.seed,
        },
        "results": {
            "original_run_image": m1,
            "cached_fast_path": m2,
            "batch_backed_fast_path": m3,
            "speedup_cached_vs_original": round(speedup_cached, 2),
            "speedup_batch_vs_original": round(speedup_batch, 2),
        },
    }

    json_path = output_dir / "cached_fast_path_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # ── Write markdown ──
    md = f"""# Cached Fast Path Benchmark

> Date: {report['metadata']['date']} | Images: {args.images} | Seed: {args.seed}

## Results

| Method | Mean ms/img | P50 | P95 | QPS | Label |
|---|---:|---:|---:|---:|---|
| Original repeated run_image() | {m1['mean_ms']:.0f} | {m1['p50_ms']:.0f} | {m1['p95_ms']:.0f} | {m1['throughput_qps']:.1f} | [measured] |
| CachedFastPath.run_image() | {m2['mean_ms']:.0f} | {m2['p50_ms']:.0f} | {m2['p95_ms']:.0f} | {m2['throughput_qps']:.1f} | [measured] |
| BatchBackedFastPath.run_images() | {m3['mean_ms']:.0f} | — | — | {m3['throughput_qps']:.1f} | [measured] |

## Speedup

| Comparison | Ratio |
|---|---|
| CachedFastPath vs original run_image | **{speedup_cached:.1f}×** |
| BatchBackedFastPath vs original run_image | **{speedup_batch:.1f}×** |

## Notes

- CachedFastPath: YOLO + SAM-HQ + Landmark loaded once, reused across calls. ~135 lines adapted from stages 1-2 inference loops. Stages 3-5 imported unchanged.
- BatchBackedFastPath: ZERO replicated inference logic. Thin wrapper around GarmentPipeline.run_source(). Best for batch/offline processing. Does NOT help single-image interactive latency.
"""
    md_path = output_dir / "cached_fast_path_benchmark.md"
    with open(md_path, "w") as f:
        f.write(md.strip())

    # ── CSV ──
    csv_path = output_dir / "cached_fast_path_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("method,mean_ms,p50_ms,p95_ms,qps\n")
        f.write(f"original,{m1['mean_ms']},{m1['p50_ms']},{m1['p95_ms']},{m1['throughput_qps']}\n")
        f.write(f"cached,{m2['mean_ms']},{m2['p50_ms']},{m2['p95_ms']},{m2['throughput_qps']}\n")
        f.write(f"batch,{m3['mean_ms']},-,_,{m3['throughput_qps']}\n")

    print(f"\n[Reports: {json_path}, {md_path}, {csv_path}]")


if __name__ == "__main__":
    main()
