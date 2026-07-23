"""
Model-only microbenchmarks.

Measures pure engine/PyTorch forward pass with synthetic input tensors
already resident on GPU. No H2D transfer, no file I/O, no postprocessing.

Usage::

    python inference/benchmarks/microbench_model.py \
        --model-type yolo --engine engines/yolov8n_fp16.engine \
        --batch-size 1 --runs 1000 --warmup 100

Status: Skeleton. Full implementation proceeds Week 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _create_synthetic_input(
    model_type: str, batch_size: int, device: str = "cuda"
) -> Any:
    """Create a synthetic input tensor for the given model type.

    NOT YET IMPLEMENTED — returns None.
    TODO Week 1: allocate tensors on GPU matching each model's input shape.
    """
    # Shapes per model type:
    #   yolo:         (B, 3, 640, 640)
    #   sam_encoder:  (B, 3, 1024, 1024)
    #   sam_decoder:  (B, 256, 64, 64) + box prompts
    #   landmark:     (B, 3, 256, 256)
    #   fashionpedia: (B, 3, 640, 640)
    #   dino:         (B, 3, 800, 800) + text embeddings
    #   attribute:    (B, 3, 224, 224)
    print(f"  [SKELETON] Synthetic input for {model_type}, batch={batch_size}")
    return None


def run_microbench(
    model_type: str,
    engine_path: Optional[Path] = None,
    pt_path: Optional[Path] = None,
    batch_size: int = 1,
    runs: int = 1000,
    warmup: int = 100,
) -> Dict[str, Any]:
    """Run a model-only microbenchmark.

    TODO Week 1:
    1. Load engine (TensorRT) or PyTorch model
    2. Create synthetic input tensor on GPU
    3. Warmup (discard results)
    4. Measure N runs with torch.cuda.synchronize() each
    5. Compute stats via latency_taxonomy.compute_stats()
    """
    print(f"[microbench] {model_type}, batch={batch_size}, runs={runs}")
    return {
        "status": "not_implemented",
        "model_type": model_type,
        "batch_size": batch_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-only microbenchmark")
    ap.add_argument("--model-type", required=True,
                    choices=["yolo", "sam_encoder", "sam_decoder", "landmark",
                              "fashionpedia", "dino", "attribute"])
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--pt-path", type=Path)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_microbench(
        model_type=args.model_type,
        engine_path=args.engine,
        pt_path=args.pt_path,
        batch_size=args.batch_size,
        runs=args.runs,
        warmup=args.warmup,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
