"""
Unified CLI for running inference benchmarks.

Supports three benchmark types:
    microbench  — Model-only, synthetic tensor, no I/O
    stage       — Single stage with real input from disk
    pipeline    — End-to-end pipeline on real images

Usage::

    python inference/benchmarks/benchmark_runner.py microbench \
        --model-type yolo --engine engines/yolov8n_fp16.engine --runs 1000

    python inference/benchmarks/benchmark_runner.py stage \
        --stage yolo --images 100 --config configs/fast_path_bench.yaml

    python inference/benchmarks/benchmark_runner.py pipeline \
        --path fast --batch-size 8 --images 500

Status: New module. Skeleton only. Full implementation proceeds Week 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Sub-command handlers (skeletons) ───────────────────────────────────────────

def run_microbench(
    model_type: str,
    engine_path: Optional[Path] = None,
    pt_path: Optional[Path] = None,
    batch_size: int = 1,
    runs: int = 1000,
    warmup: int = 100,
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a model-only microbenchmark.

    Measures pure engine/pytorch forward pass with synthetic input on GPU.
    No H2D, no I/O, no postprocessing included.
    """
    print(f"[microbench] model_type={model_type}, batch={batch_size}, runs={runs}")
    print(f"  engine: {engine_path}, pytorch: {pt_path}")
    # ── TODO Week 1: implement per-model-type microbench ──
    # 1. Create synthetic input tensor on GPU
    # 2. Load engine or PyTorch model
    # 3. Warmup (discard)
    # 4. Measure N runs with torch.cuda.synchronize() each
    # 5. Compute stats via latency_taxonomy.compute_stats()
    print("  [SKELETON] Not yet implemented — Week 1 task.")
    return {"status": "not_implemented", "model_type": model_type}


def run_stage_bench(
    stage: str,
    images: int = 100,
    config: Optional[Path] = None,
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single-stage benchmark with real input from disk."""
    print(f"[stage-bench] stage={stage}, images={images}")
    # ── TODO Week 1: implement per-stage benchmarks ──
    # 1. Load model (TRT engine or PyTorch)
    # 2. Load images from config or CLI args
    # 3. For each image: run preprocess → model → postprocess
    # 4. Record per-sub-stage timings via StageTimer
    # 5. Aggregate and report
    print("  [SKELETON] Not yet implemented — Week 1 task.")
    return {"status": "not_implemented", "stage": stage}


def run_pipeline_bench(
    path: str,
    batch_size: int = 1,
    images: int = 500,
    config: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run end-to-end pipeline benchmark."""
    print(f"[pipeline-bench] path={path}, batch={batch_size}, images={images}")
    # ── TODO Week 1: implement per-path pipeline benchmarks ──
    # 1. Load all models for the chosen path
    # 2. Build pipeline (fast/query/full)
    # 3. Run on N images, record total wall-clock + per-stage breakdown
    # 4. Report QPS, P50/P95/P99, per-stage timing
    print("  [SKELETON] Not yet implemented — Week 1 task.")
    return {"status": "not_implemented", "path": path}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inference benchmark runner — microbench, stage, or pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # microbench
    p_micro = sub.add_parser("microbench", help="Model-only microbenchmark")
    p_micro.add_argument("--model-type", required=True,
                         choices=["yolo", "sam_encoder", "sam_decoder", "landmark",
                                   "fashionpedia", "dino", "attribute"])
    p_micro.add_argument("--engine", type=Path, help="TensorRT engine path")
    p_micro.add_argument("--pt-path", type=Path, help="PyTorch model path (fallback)")
    p_micro.add_argument("--batch-size", type=int, default=1)
    p_micro.add_argument("--runs", type=int, default=1000)
    p_micro.add_argument("--warmup", type=int, default=100)
    p_micro.add_argument("--output", type=Path)

    # stage
    p_stage = sub.add_parser("stage", help="Single-stage benchmark")
    p_stage.add_argument("--stage", required=True,
                         choices=["yolo", "sam", "landmark", "crop", "mask_crop",
                                   "dino", "fashionpedia", "attributes"])
    p_stage.add_argument("--images", type=int, default=100)
    p_stage.add_argument("--config", type=Path)
    p_stage.add_argument("--output", type=Path)

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="End-to-end pipeline benchmark")
    p_pipe.add_argument("--path", required=True,
                        choices=["fast", "query", "full"])
    p_pipe.add_argument("--batch-size", type=int, default=1)
    p_pipe.add_argument("--images", type=int, default=500)
    p_pipe.add_argument("--config", type=Path)
    p_pipe.add_argument("--output-dir", type=Path)

    args = ap.parse_args()

    t0 = time.perf_counter()

    if args.mode == "microbench":
        result = run_microbench(
            model_type=args.model_type,
            engine_path=args.engine,
            pt_path=args.pt_path,
            batch_size=args.batch_size,
            runs=args.runs,
            warmup=args.warmup,
            output=args.output,
        )
    elif args.mode == "stage":
        result = run_stage_bench(
            stage=args.stage,
            images=args.images,
            config=args.config,
            output=args.output,
        )
    elif args.mode == "pipeline":
        result = run_pipeline_bench(
            path=args.path,
            batch_size=args.batch_size,
            images=args.images,
            config=args.config,
            output_dir=args.output_dir,
        )
    else:
        ap.print_help()
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    result["cli_elapsed_s"] = round(elapsed, 2)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
