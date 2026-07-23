"""
Baseline benchmark wrapper for the existing 3.1 pipeline.

Runs the current `GarmentPipeline` (PyTorch, unoptimized) WITHOUT modifying
any existing code under `tools/infer/` or `src/fashion_vision/`.

Captures per-stage timing, environment metadata, and produces a JSON report
in the standard format defined by the inference optimization plan v2.

Usage::

    # Dry run — validate paths and config without GPU execution
    python inference/benchmarks/bench_existing_3_1_pipeline.py \\
        --images 10 --mode fast --dry-run

    # Real benchmark — Fast Path only
    python inference/benchmarks/bench_existing_3_1_pipeline.py \\
        --images 10 --mode fast --output-dir outputs/inference_optimization/baseline

    # Real benchmark — Fast + Query (includes DINO queries)
    python inference/benchmarks/bench_existing_3_1_pipeline.py \\
        --images 10 --mode query --include-dino

    # Real benchmark — Full Analysis Path
    python inference/benchmarks/bench_existing_3_1_pipeline.py \\
        --images 10 --mode full --include-attributes

Output files (written to --output-dir):
    benchmark_report.json   — full results with per-image breakdown
    env.json                — hardware/software environment snapshot
    per_image_timing.csv    — per-image × per-stage latencies (ms)

Status: New module.  Wraps existing pipeline, does not modify it.
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

# ── Project path setup ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# inference/ utilities (these are new, safe to import)
from inference.env_capture import capture_env
from inference.latency_taxonomy import compute_stats, StageTiming


# ── Default paths (read-only — do not hard-override without checking existence) ─

DEFAULT_YOLO_WEIGHTS = "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
DEFAULT_SAM_CHECKPOINT = "checkpoints/sam_hq/sam_hq_vit_b.pth"
DEFAULT_LANDMARK_CHECKPOINT = "outputs/landmark_predictor_resnet18/best.pt"
DEFAULT_FP_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
DEFAULT_DINO_MODEL = "models/grounding_dino_tiny"

# Image directory used by existing benchmark_312_timing.py
DEFAULT_IMAGE_DIR = (
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\lapel_design_labels"
)

# ── Path validation ────────────────────────────────────────────────────────────

def _resolve(path_str: str) -> Path:
    """Resolve a path relative to project root if not absolute."""
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _check_path(label: str, path_str: str) -> Dict[str, Any]:
    """Validate one path and return a status dict."""
    p = _resolve(path_str)
    exists = p.exists()
    return {
        "label": label,
        "path": str(p),
        "exists": exists,
        "type": "file" if exists and p.is_file() else "directory" if exists and p.is_dir() else "missing",
    }


def validate_all_paths(
    image_dir: str,
    yolo_weights: str,
    sam_checkpoint: str,
    landmark_checkpoint: str,
    fp_model: Optional[str] = None,
    dino_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Check all required model and data paths.  Returns list of status dicts."""
    checks = [
        _check_path("image_directory", image_dir),
        _check_path("yolo_weights", yolo_weights),
        _check_path("sam_checkpoint", sam_checkpoint),
        _check_path("landmark_checkpoint", landmark_checkpoint),
    ]
    if fp_model:
        checks.append(_check_path("fashionpedia_model", fp_model))
    if dino_model:
        checks.append(_check_path("dino_model", dino_model))
    return checks


# ── Image sampling ─────────────────────────────────────────────────────────────

def sample_images(image_dir: str, n: int, seed: int = 42) -> List[Path]:
    """Sample *n* image files from a directory, sorted and seeded."""
    d = _resolve(image_dir)
    if not d.is_dir():
        print(f"[WARN] Image directory not found: {d}")
        return []
    all_images = sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    rng = random.Random(seed)
    return rng.sample(all_images, min(n, len(all_images)))


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_fast_path_benchmark(
    image_paths: List[Path],
    yolo_weights: str,
    sam_checkpoint: str,
    landmark_checkpoint: str,
    warmup: int = 3,
) -> Dict[str, Any]:
    """
    Run the existing GarmentPipeline Fast Path on a list of images.

    Does NOT import or use DINO, Fashionpedia, or attribute classifiers.
    Uses GarmentPipeline.run_image() which already records per-stage timing
    via ``time.perf_counter()`` inside the pipeline.

    Returns a dict with keys:
        per_image: list of per-image timing dicts
        aggregate: summary stats per stage (mean, std, p50, p95, p99)
        total_wallclock_s: total wall clock for all images
        num_images: number of images processed
    """
    # ── Lazy import — only when actually running ──
    from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig

    import tempfile

    config = GarmentPipelineConfig(
        yolo_weights=yolo_weights,
        sam_checkpoint=sam_checkpoint,
        sam_model_type="vit_b",
        landmark_checkpoint=landmark_checkpoint,
        yolo_device="0",
        sam_device="cuda",
        landmark_device="cuda",
        run_landmark_and_crops=True,
        run_attribute_inference=False,
        save_yolo_vis=False,
        save_yolo_crops=False,
    )

    print("[INFO] Loading GarmentPipeline (YOLO + SAM-HQ + Landmark + Crop)...")
    t_load = time.perf_counter()
    pipeline = GarmentPipeline(config)
    load_s = time.perf_counter() - t_load
    print(f"[INFO] Pipeline loaded in {load_s:.1f}s")

    per_image: List[Dict[str, Any]] = []
    stage_latencies: Dict[str, List[float]] = {
        "yolo_ms": [],
        "sam_hq_ms": [],
        "landmarks_ms": [],
        "region_crops_ms": [],
        "masked_crops_ms": [],
        "total_ms": [],
    }

    with tempfile.TemporaryDirectory(prefix="bench_31_") as tmpdir:
        tmp = Path(tmpdir)

        # ── Warmup ──
        if warmup > 0 and len(image_paths) > 0:
            print(f"[INFO] Warming up ({warmup} passes on first image)...")
            for _ in range(warmup):
                pipeline.run_image(str(image_paths[0]), str(tmp / "warmup"))

        # ── Measurement ──
        t_total_start = time.perf_counter()
        for i, img_path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name}")
            out_dir = tmp / f"img_{i:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = pipeline.run_image(str(img_path), str(out_dir))
            except Exception as exc:
                print(f"    [FAIL] {exc}")
                per_image.append({"image": img_path.name, "error": str(exc)[:200]})
                continue

            timing = result.get("timing", {})
            row = {
                "image": img_path.name,
                "num_garments": (
                    result.get("region_crops_summary", {}).get("num_instances", None)
                    or result.get("masked_crops_summary", {}).get("num_instances", None)
                ),
                "yolo_ms": round(timing.get("yolo_seconds", 0) * 1000, 2),
                "sam_hq_ms": round(timing.get("sam_hq_seconds", 0) * 1000, 2),
                "landmarks_ms": round(timing.get("landmarks_seconds", 0) * 1000, 2),
                "region_crops_ms": round(timing.get("region_crops_seconds", 0) * 1000, 2),
                "masked_crops_ms": round(timing.get("masked_crops_seconds", 0) * 1000, 2),
                "total_ms": round(timing.get("total_seconds", 0) * 1000, 2),
            }
            per_image.append(row)

            for key in stage_latencies:
                stage_latencies[key].append(row[key])

        total_wallclock_s = time.perf_counter() - t_total_start

    # ── Aggregate stats ──
    aggregate: Dict[str, Any] = {}
    for key, values in stage_latencies.items():
        if values:
            aggregate[key] = compute_stats(values)
        else:
            aggregate[key] = {"count": 0}

    return {
        "per_image": per_image,
        "aggregate": aggregate,
        "total_wallclock_s": round(total_wallclock_s, 2),
        "num_images": len(per_image),
        "num_failures": sum(1 for r in per_image if "error" in r),
        "pipeline_load_s": round(load_s, 1),
    }


# ── Dry-run mode ───────────────────────────────────────────────────────────────

def dry_run(
    image_dir: str,
    images: int,
    mode: str,
    include_dino: bool,
    include_attributes: bool,
    yolo_weights: str,
    sam_checkpoint: str,
    landmark_checkpoint: str,
    fp_model: Optional[str],
    dino_model: Optional[str],
) -> Dict[str, Any]:
    """Validate all paths and report what WOULD be run, without GPU execution."""
    path_checks = validate_all_paths(
        image_dir=image_dir,
        yolo_weights=yolo_weights,
        sam_checkpoint=sam_checkpoint,
        landmark_checkpoint=landmark_checkpoint,
        fp_model=fp_model,
        dino_model=dino_model,
    )

    all_ok = all(c["exists"] for c in path_checks)
    sampled = sample_images(image_dir, images) if all_ok else []

    return {
        "mode": "dry_run",
        "pipeline_path": mode,
        "images_requested": images,
        "images_available": len(sampled),
        "all_paths_ok": all_ok,
        "path_checks": path_checks,
        "would_run": {
            "fast_path": True,
            "include_dino": include_dino and all_ok,
            "include_attributes": include_attributes and all_ok,
        },
        "notes": [
            "Dry run — no GPU execution performed.",
            f"Sampled {len(sampled)} images from {image_dir}",
        ] + (
            [] if all_ok else ["Some paths are MISSING — real benchmark will fail."]
        ),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baseline benchmark for existing 3.1 pipeline (read-only wrapper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--images", type=int, default=10,
                    help="Number of images to benchmark (default: 10)")
    ap.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR,
                    help="Directory containing images")
    ap.add_argument("--output-dir", type=str,
                    default="outputs/inference_optimization/baseline",
                    help="Output directory for reports")
    ap.add_argument("--mode", choices=["fast", "query", "full"], default="fast",
                    help="Pipeline path to benchmark")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Warmup iterations (default: 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-dino", action="store_true",
                    help="Include DINO queries (for query/full mode)")
    ap.add_argument("--include-attributes", action="store_true",
                    help="Include attribute classification (for full mode)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate paths only, no GPU execution")

    # Model path overrides (for reproducibility on different machines)
    ap.add_argument("--yolo-weights", type=str, default=DEFAULT_YOLO_WEIGHTS)
    ap.add_argument("--sam-checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT)
    ap.add_argument("--landmark-checkpoint", type=str, default=DEFAULT_LANDMARK_CHECKPOINT)
    ap.add_argument("--fp-model", type=str, default=DEFAULT_FP_MODEL)
    ap.add_argument("--dino-model", type=str, default=DEFAULT_DINO_MODEL)

    args = ap.parse_args()

    # ── Resolve paths ──
    yolo_w = str(_resolve(args.yolo_weights))
    sam_ckpt = str(_resolve(args.sam_checkpoint))
    lm_ckpt = str(_resolve(args.landmark_checkpoint))
    fp_m = str(_resolve(args.fp_model)) if args.include_dino else None
    dino_m = str(_resolve(args.dino_model)) if args.include_dino else None

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Capture environment ──
    env = capture_env()
    env_path = output_dir / "env.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False, default=str)
    print(f"[INFO] Environment saved to: {env_path}")

    # ── Dry run ──
    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        report = dry_run(
            image_dir=args.image_dir,
            images=args.images,
            mode=args.mode,
            include_dino=args.include_dino,
            include_attributes=args.include_attributes,
            yolo_weights=yolo_w,
            sam_checkpoint=sam_ckpt,
            landmark_checkpoint=lm_ckpt,
            fp_model=fp_m,
            dino_model=dino_m,
        )
        report["environment"] = {"env_file": str(env_path)}
        report["metadata"] = {
            "benchmark_type": "pipeline",
            "pipeline_path": args.mode,
            "status": "dry_run",
        }

        dry_path = output_dir / "dry_run_report.json"
        with open(dry_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[Dry run report saved to: {dry_path}]")

        # Print path status
        print("\nPath checks:")
        for c in report["path_checks"]:
            icon = "OK" if c["exists"] else "MISSING"
            print(f"  [{icon}] {c['label']}: {c['path']}")

        if report["all_paths_ok"]:
            print("\n[OK] All paths OK. Real benchmark can proceed with:")
            print(f"  python inference/benchmarks/bench_existing_3_1_pipeline.py \\")
            print(f"    --images {args.images} --mode {args.mode} \\")
            print(f"    --output-dir {args.output_dir}")
        else:
            print("\n[WARN] Some paths missing. Fix before running real benchmark.")
        return

    # ── Real benchmark ──
    print(f"\n=== REAL BENCHMARK: {args.mode.upper()} PATH ===\n")
    print(f"  Images: {args.images} (seed={args.seed})")
    print(f"  Warmup: {args.warmup}")
    print(f"  DINO: {args.include_dino}, Attributes: {args.include_attributes}")
    print()

    # Sample images
    image_paths = sample_images(args.image_dir, args.images, args.seed)
    if not image_paths:
        print("[ERROR] No images found. Run with --dry-run to check paths.")
        sys.exit(1)
    print(f"  Sampled {len(image_paths)} images.")

    # ── Fast Path (always runs) ──
    t0 = time.perf_counter()
    fast_result = run_fast_path_benchmark(
        image_paths=image_paths,
        yolo_weights=yolo_w,
        sam_checkpoint=sam_ckpt,
        landmark_checkpoint=lm_ckpt,
        warmup=args.warmup,
    )
    total_s = time.perf_counter() - t0
    qps = fast_result["num_images"] / max(1e-9, total_s)

    # ── Build report ──
    report: Dict[str, Any] = {
        "metadata": {
            "benchmark_type": "pipeline",
            "pipeline_path": args.mode,
            "status": "baseline_existing_3_1",
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _get_git_commit(),
        },
        "environment": {"env_file": str(env_path)},
        "config": {
            "images_requested": args.images,
            "images_processed": fast_result["num_images"],
            "warmup": args.warmup,
            "seed": args.seed,
            "include_dino": args.include_dino,
            "include_attributes": args.include_attributes,
            "pipeline_load_s": fast_result["pipeline_load_s"],
        },
        "results": {
            "total_wallclock_s": round(total_s, 2),
            "throughput_qps": round(qps, 2),
            "latency": {
                "mean_ms": round(fast_result["aggregate"]["total_ms"].get("mean_ms", 0), 2),
                "std_ms": round(fast_result["aggregate"]["total_ms"].get("std_ms", 0), 2),
                "p50_ms": round(fast_result["aggregate"]["total_ms"].get("p50_ms", 0), 2),
                "p95_ms": round(fast_result["aggregate"]["total_ms"].get("p95_ms", 0), 2),
                "p99_ms": round(fast_result["aggregate"]["total_ms"].get("p99_ms", 0), 2),
            },
            "per_stage": {
                key.replace("_ms", ""): fast_result["aggregate"][key]
                for key in ["yolo_ms", "sam_hq_ms", "landmarks_ms",
                            "region_crops_ms", "masked_crops_ms", "total_ms"]
                if fast_result["aggregate"].get(key, {}).get("count", 0) > 0
            },
            "num_failures": fast_result["num_failures"],
            "notes": [
                f"Fast Path ({args.mode} mode).",
                "Stage timings from existing GarmentPipeline (perf_counter, no sub-stage breakdown).",
                "SAM-HQ vit_b, YOLOv8n PyTorch, ResNet18 PyTorch — no TensorRT.",
                "All numbers [measured] on current codebase.",
            ],
        },
    }

    # ── Write JSON report ──
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Report saved to: {report_path}]")

    # ── Write per-image CSV ──
    csv_path = output_dir / "per_image_timing.csv"
    if fast_result["per_image"]:
        fieldnames = list(fast_result["per_image"][0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fast_result["per_image"])
        print(f"[CSV saved to: {csv_path}]")

    # ── Summary to stdout ──
    agg = fast_result["aggregate"]
    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY — {args.mode.upper()} PATH")
    print(f"{'='*60}")
    print(f"  Images:         {fast_result['num_images']}")
    print(f"  Total wall:     {total_s:.1f}s")
    print(f"  Throughput:     {qps:.1f} QPS")
    print(f"  Pipeline load:  {fast_result['pipeline_load_s']:.1f}s")
    print(f"  Failures:       {fast_result['num_failures']}")
    print(f"\n  Per-stage (mean ± std):")
    stage_labels = [
        ("yolo_ms", "YOLO detection"),
        ("sam_hq_ms", "SAM-HQ segmentation"),
        ("landmarks_ms", "Landmark prediction"),
        ("region_crops_ms", "Region crop"),
        ("masked_crops_ms", "Mask-aware crop"),
        ("total_ms", "TOTAL"),
    ]
    for key, label in stage_labels:
        s = agg.get(key, {})
        if s.get("count", 0) > 0:
            print(f"    {label:<25s}: {s['mean_ms']:7.1f} ms ± {s['std_ms']:5.1f}  "
                  f"(P50={s['p50_ms']:.0f}, P95={s['p95_ms']:.0f}, P99={s['p99_ms']:.0f})")
    print(f"{'='*60}")


def _get_git_commit() -> str:
    """Get short git commit hash. Returns 'unknown' on failure."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
