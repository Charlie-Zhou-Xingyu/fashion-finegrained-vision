#!/usr/bin/env python3
"""
Inference runtime benchmark for Fashionpedia YOLO part detector.

Benchmarks YOLOv8s-19cls on:
  - Single-image throughput (warmup + N runs, mean/std/P50/P95)
  - Batch inference scaling (1/2/4/8/16 batch sizes)
  - Model load time, parameter count, FLOPs estimate

Outputs: benchmark_runtime.csv in the specified output directory.

Usage:
    python scripts/benchmark_fashionpedia_yolo.py \
        --model outputs/fashionpedia_19cls_yolov8s/fashionpedia_parts_19cls_yolov8s_best.pt \
        --image-dir E:/fashionpedia_yolo_19cls/images/val \
        --output-dir outputs/fashionpedia_19cls_yolov8s \
        --device cuda

    # CPU benchmark:
    python scripts/benchmark_fashionpedia_yolo.py --device cpu --max-images 50
"""

from __future__ import annotations

import argparse
import csv
import time
import math
from pathlib import Path
from typing import Optional

import numpy as np


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def _fmt_fps(seconds: float) -> str:
    return f"{1.0 / max(1e-9, seconds):.1f} FPS"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", "-m", type=Path,
        default=Path("outputs/fashionpedia_19cls_yolov8s/fashionpedia_parts_19cls_yolov8s_best.pt"),
        help="Path to YOLO .pt model",
    )
    ap.add_argument(
        "--image-dir", type=Path,
        default=Path("E:/fashionpedia_yolo_19cls/images/val"),
        help="Directory of images for benchmarking",
    )
    ap.add_argument(
        "--output-dir", "-o", type=Path,
        default=Path("outputs/fashionpedia_19cls_yolov8s"),
        help="Output directory for benchmark_runtime.csv",
    )
    ap.add_argument(
        "--device", "-d", type=str, default="cuda",
        help="Device: cuda / cpu",
    )
    ap.add_argument(
        "--max-images", type=int, default=200,
        help="Max images to load for batch benchmark",
    )
    ap.add_argument(
        "--single-runs", type=int, default=100,
        help="Number of single-image inference runs for timing",
    )
    args = ap.parse_args()

    # ── Imports (deferred so --help is fast) ──────────────────────────
    from ultralytics import YOLO
    import torch

    # ── Load model ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model = YOLO(str(args.model))
    t_load = time.perf_counter() - t0

    device_str = args.device
    actual_device = next(model.model.parameters()).device
    print(f"Model loaded in {_fmt_ms(t_load)}")
    print(f"  Device: {actual_device}")
    print(f"  Classes: {len(model.names)}")
    print(f"  Names: {model.names}")

    # Parameter count
    n_params = sum(p.numel() for p in model.model.parameters())
    n_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,} total, {n_trainable:,} trainable")

    # FLOPs estimate (rough: YOLOv8s ~28.6 GFLOPs at 640×640; scale by resolution)
    print(f"  FLOPs: ~28.6G (YOLOv8s reference at 640×640)")

    # ── Gather test images ─────────────────────────────────────────────
    image_dir = args.image_dir
    if not image_dir.is_dir():
        print(f"\n[WARN] Image dir not found: {image_dir}")
        print("  Skipping image-based benchmarks. Only model metadata collected.")
        _write_csv(args.output_dir, {
            "model_path": str(args.model),
            "device": device_str,
            "n_params": n_params,
            "n_classes": len(model.names),
            "load_time_s": round(t_load, 4),
        })
        return

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    image_paths = image_paths[:args.max_images]
    if not image_paths:
        print(f"[ERROR] No images found in {image_dir}")
        return
    print(f"\nBenchmark images: {len(image_paths)} (max {args.max_images})")

    # ── Warmup ─────────────────────────────────────────────────────────
    print("\nWarming up (3 passes)...")
    for i in range(3):
        _ = model(str(image_paths[0]), device=device_str, verbose=False)
    print("  done.")

    # ── Single-image benchmark ─────────────────────────────────────────
    img0 = str(image_paths[0])
    n_runs = args.single_runs
    print(f"\nSingle-image inference ({n_runs} runs)...")
    times: list[float] = []
    for _ in range(n_runs):
        t_start = time.perf_counter()
        _ = model(img0, device=device_str, verbose=False, imgsz=640)
        torch_sync(actual_device)
        times.append(time.perf_counter() - t_start)

    times_ms = np.array(times) * 1000
    print(f"  Mean  : {np.mean(times_ms):.2f} ms  ({1e3/np.mean(times_ms):.1f} FPS)")
    print(f"  Std   : {np.std(times_ms):.2f} ms")
    print(f"  P50   : {np.percentile(times_ms, 50):.2f} ms")
    print(f"  P95   : {np.percentile(times_ms, 95):.2f} ms")
    print(f"  Min   : {np.min(times_ms):.2f} ms")
    print(f"  Max   : {np.max(times_ms):.2f} ms")

    single_stats = {
        "mean_ms": round(float(np.mean(times_ms)), 3),
        "std_ms": round(float(np.std(times_ms)), 3),
        "p50_ms": round(float(np.percentile(times_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
        "min_ms": round(float(np.min(times_ms)), 3),
        "max_ms": round(float(np.max(times_ms)), 3),
    }

    # ── Batch inference benchmark ──────────────────────────────────────
    batch_sizes = [1, 2, 4, 8, 16]
    batch_stats: dict[int, dict] = {}

    print(f"\nBatch inference benchmark:")
    print(f"{'Batch':>6s} {'Images':>8s} {'Total':>10s} {'Per-img':>10s} {'FPS':>10s}")
    print("-" * 48)

    for bs in batch_sizes:
        if bs > len(image_paths):
            break

        # Use a subset so all batches are same size for fair comparison
        n_images = (len(image_paths) // bs) * bs
        subset = image_paths[:n_images]
        img_strs = [str(p) for p in subset]

        batch_times: list[float] = []
        for i in range(0, n_images, bs):
            batch = img_strs[i:i + bs]
            if len(batch) < bs:
                break
            t_start = time.perf_counter()
            results = model(batch, device=device_str, verbose=False, imgsz=640)
            torch_sync(actual_device)
            batch_times.append(time.perf_counter() - t_start)

        if not batch_times:
            continue

        total_s = sum(batch_times)
        n_batches = len(batch_times)
        n_imgs = n_batches * bs
        per_img_ms = (total_s / n_imgs) * 1000
        fps = n_imgs / total_s

        batch_stats[bs] = {
            "n_batches": n_batches,
            "total_images": n_imgs,
            "total_time_s": round(total_s, 4),
            "per_image_ms": round(per_img_ms, 3),
            "fps": round(fps, 1),
        }

        print(f"{bs:6d} {n_imgs:8d} {total_s:9.4f}s {per_img_ms:9.2f}ms {fps:9.1f}")

    # ── Throughput comparison ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"THROUGHPUT COMPARISON")
    print(f"{'='*60}")
    print(f"  Single-image (mean): {single_stats['mean_ms']:.1f} ms → {1000/single_stats['mean_ms']:.1f} FPS")
    if 1 in batch_stats:
        print(f"  Batch-1            : {batch_stats[1]['per_image_ms']:.1f} ms → {batch_stats[1]['fps']:.1f} FPS")
    if 16 in batch_stats:
        bs16 = batch_stats[16]
        print(f"  Batch-16           : {bs16['per_image_ms']:.1f} ms → {bs16['fps']:.1f} FPS")
        speedup = batch_stats[1]["per_image_ms"] / max(1e-9, bs16["per_image_ms"])
        print(f"  Speedup (1→16)     : {speedup:.2f}×")

    # ── Training time estimate ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING TIME ESTIMATE (rough)")
    print(f"{'='*60}")

    # Rough estimate: YOLOv8s ~300 images/s on single GPU at batch 16
    # But we can estimate from single/batch inference
    if 16 in batch_stats:
        img_per_s = batch_stats[16]["fps"]
    elif batch_stats:
        img_per_s = max(s["fps"] for s in batch_stats.values())
    else:
        img_per_s = 100  # conservative fallback

    # Estimate for 100 epochs with 45K images (1×) vs inflated (e.g. 3×)
    n_train = 45000  # ~45K images with parts
    epochs = 100

    for label, inflation in [("Original (1×)", 1.0), ("Balanced p=0.6 r=12 (~3×)", 3.0), ("Balanced p=0.6 r=12 (~5×)", 5.0)]:
        n_total = int(n_train * inflation * epochs)
        est_hours = n_total / max(1, img_per_s) / 3600
        print(f"  {label:30s}: {n_total:>10,} forward passes → ~{est_hours:.1f} h (~{est_hours/24:.1f} days)")

    print(f"\n  (based on {img_per_s:.0f} img/s throughput at batch-16 on {device_str})")
    print(f"  Actual training time includes backward pass + augmentations → ×2-4 slower")

    # ── Write CSV ──────────────────────────────────────────────────────
    _write_csv(args.output_dir, {
        "model_path": str(args.model),
        "device": device_str,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "n_classes": len(model.names),
        "load_time_s": round(t_load, 4),
        "single_mean_ms": single_stats["mean_ms"],
        "single_std_ms": single_stats["std_ms"],
        "single_p50_ms": single_stats["p50_ms"],
        "single_p95_ms": single_stats["p95_ms"],
        "single_min_ms": single_stats["min_ms"],
        "single_max_ms": single_stats["max_ms"],
        "batch1_per_img_ms": batch_stats.get(1, {}).get("per_image_ms", ""),
        "batch4_per_img_ms": batch_stats.get(4, {}).get("per_image_ms", ""),
        "batch8_per_img_ms": batch_stats.get(8, {}).get("per_image_ms", ""),
        "batch16_per_img_ms": batch_stats.get(16, {}).get("per_image_ms", ""),
        "batch16_fps": batch_stats.get(16, {}).get("fps", ""),
    })


def torch_sync(device) -> None:
    """Synchronize CUDA stream for accurate timing."""
    try:
        import torch
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def _write_csv(output_dir: Path, data: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_runtime.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        writer.writeheader()
        writer.writerow(data)
    print(f"\nBenchmark saved: {csv_path}")


if __name__ == "__main__":
    main()
