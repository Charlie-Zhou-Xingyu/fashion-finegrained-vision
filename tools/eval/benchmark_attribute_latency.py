"""Benchmark single-instance inference latency for an attribute classifier.

Measures the forward-pass wall time of a trained attribute classification model
using synthetic input tensors only.  No dataset or image files are read.

CUDA synchronisation is applied before and after each timed pass when running
on GPU to ensure accurate wall-clock measurement.

Usage
-----
::

    python tools/eval/benchmark_attribute_latency.py \\
        --checkpoint outputs/p2_collar_design_multiview_v2_pipeline_resnet18_seed2/best.pt \\
        --arch resnet18 \\
        --img-size 224 \\
        --batch-size 1 \\
        --warmup 20 \\
        --runs 200 \\
        --device auto \\
        --out outputs/eval_latency/collar_design.json

If ``--num-classes`` is omitted, the number of output classes is inferred from
the ``fc.weight`` shape in the checkpoint state dict.  Pass ``--num-classes``
explicitly if inference fails.

Output JSON
-----------
::

    {
      "checkpoint": "...",
      "arch": "resnet18",
      "num_classes": 5,
      "img_size": 224,
      "batch_size": 1,
      "device": "cpu",
      "warmup_runs": 20,
      "timed_runs": 200,
      "latency_ms_per_batch": {"mean": 3.2, "median": 3.1, "p50": 3.1,
                                "p95": 4.1, "p99": 5.2, "min": 2.9, "max": 8.3},
      "latency_ms_per_image": {"mean": 3.2, ...},
      "throughput_images_per_sec": 312.5,
      "prd_latency_target_ms": 20.0,
      "meets_prd_target": true
    }

``meets_prd_target`` evaluates ``latency_ms_per_image["mean"] <= 20.0``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# sys.path: project root (for models/) and src/ (for fashion_vision/)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.attribute_classifier import build_attribute_classifier  # noqa: E402
from fashion_vision.attributes.task_registry import (  # noqa: E402
    infer_num_classes_from_state,
    load_checkpoint_state,
)

# PRD 3.1.3 latency requirement.
_PRD_LATENCY_TARGET_MS: float = 20.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _compute_stats(times_ms: list[float]) -> dict[str, float]:
    """Compute descriptive latency statistics over a list of timings.

    Args:
        times_ms: Per-run wall-time measurements in milliseconds.

    Returns:
        Dict with keys: ``mean``, ``median``, ``p50``, ``p95``, ``p99``,
        ``min``, ``max`` (all in milliseconds).

    Raises:
        ValueError: If *times_ms* is empty.
    """
    if not times_ms:
        raise ValueError("times_ms must not be empty.")
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p50":    float(np.percentile(arr, 50)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# num_classes resolution
# ---------------------------------------------------------------------------


def _resolve_num_classes(
    num_classes_arg: int | None,
    state: dict[str, torch.Tensor],
    checkpoint_path: Path,
) -> int:
    """Resolve ``num_classes`` from the CLI argument or the checkpoint state dict.

    Args:
        num_classes_arg: Value passed via ``--num-classes``, or ``None``.
        state: Cleaned model state dict from :func:`load_checkpoint_state`.
        checkpoint_path: Path to the checkpoint (for error messages only).

    Returns:
        Resolved ``num_classes`` as a positive integer.

    Raises:
        ValueError: If *num_classes_arg* is ``None`` and inference from the
            state dict fails (no recognised final-layer key found).
    """
    if num_classes_arg is not None:
        return num_classes_arg

    nc = infer_num_classes_from_state(state)
    if nc is None:
        raise ValueError(
            f"Cannot infer --num-classes from checkpoint: {checkpoint_path}\n"
            "None of the recognised final-layer keys (fc.weight, "
            "classifier.weight, head.weight) were found in the state dict.\n"
            "Pass --num-classes explicitly, e.g. --num-classes 8"
        )
    return nc


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_benchmark(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> list[float]:
    """Run warmup passes then timed passes; return per-run wall times in ms.

    CUDA synchronisation is applied before and after each timed call when
    *device* is a CUDA device, ensuring accurate wall-clock measurement.

    Args:
        model: Model in eval mode, already on *device*.
        x: Synthetic input tensor already on *device*.
        device: The device the model and tensor reside on.
        warmup: Number of un-timed warm-up forward passes.
        runs: Number of timed forward passes.

    Returns:
        List of per-run elapsed times in milliseconds (length == *runs*).
    """
    is_cuda = device.type == "cuda"
    times_ms: list[float] = []

    with torch.no_grad():
        # Warm-up — ensures CUDA kernels are compiled and caches are warm.
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()

        # Timed passes.
        for _ in range(runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1_000.0)

    return times_ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark single-instance inference latency for an attribute classifier. "
            "Uses synthetic tensors only — no dataset or image files are read."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Model architecture name passed to build_attribute_classifier().",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        dest="num_classes",
        help=(
            "Number of output classes.  If omitted, inferred from the "
            "fc.weight shape in the checkpoint state dict."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        dest="img_size",
        help="Square input image size in pixels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        dest="batch_size",
        help="Batch size for the synthetic input tensor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string: 'auto' (CUDA if available), 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of un-timed warm-up forward passes.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of timed forward passes.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the results JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic input.",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    """Resolve the device string to a :class:`torch.device`.

    Args:
        device_str: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable benchmark summary to stdout.

    Args:
        result: The full result dict that will also be saved as JSON.
    """
    sep = "-" * 60
    print(sep)
    print(f"  Checkpoint : {result['checkpoint']}")
    print(f"  Arch       : {result['arch']}  |  num_classes: {result['num_classes']}")
    print(f"  Device     : {result['device']}  |  img_size: {result['img_size']}  |  batch: {result['batch_size']}")
    print(f"  Runs       : warmup={result['warmup_runs']}  timed={result['timed_runs']}")
    print(sep)

    stats_b = result["latency_ms_per_batch"]
    stats_i = result["latency_ms_per_image"]
    print("  Latency (per batch):")
    print(f"    mean={stats_b['mean']:.3f} ms  median={stats_b['median']:.3f} ms  "
          f"p95={stats_b['p95']:.3f} ms  p99={stats_b['p99']:.3f} ms  "
          f"min={stats_b['min']:.3f} ms  max={stats_b['max']:.3f} ms")
    if result["batch_size"] > 1:
        print("  Latency (per image):")
        print(f"    mean={stats_i['mean']:.3f} ms  median={stats_i['median']:.3f} ms  "
              f"p95={stats_i['p95']:.3f} ms  p99={stats_i['p99']:.3f} ms")
    print(f"  Throughput : {result['throughput_images_per_sec']:.1f} images/sec")
    print(sep)

    target = result["prd_latency_target_ms"]
    mean_img = stats_i["mean"]
    status = "PASS" if result["meets_prd_target"] else "FAIL"
    print(f"  PRD target : mean per-image latency <= {target} ms")
    print(f"  Result     : {mean_img:.3f} ms  →  [{status}]")
    print(sep)


def main() -> None:
    """Entry point: parse args, load model, run benchmark, report results."""
    args = parse_args()

    # Validate inputs before doing any heavy work.
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")
    if args.runs < 1:
        raise ValueError(f"--runs must be >= 1, got {args.runs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.img_size <= 0:
        raise ValueError(f"--img-size must be positive, got {args.img_size}")

    device = _resolve_device(args.device)

    # Load checkpoint state dict.
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    state = load_checkpoint_state(args.checkpoint, device)

    # Resolve num_classes.
    num_classes = _resolve_num_classes(args.num_classes, state, args.checkpoint)
    print(f"[INFO] arch={args.arch!r}  num_classes={num_classes}  device={device}")

    # Build and load model.
    model = build_attribute_classifier(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Build synthetic input tensor.
    torch.manual_seed(args.seed)
    x = torch.zeros(args.batch_size, 3, args.img_size, args.img_size, device=device)

    print(
        f"[INFO] Benchmarking: batch_size={args.batch_size}  "
        f"warmup={args.warmup}  runs={args.runs}"
    )

    # Run benchmark.
    times_ms_per_batch = _run_benchmark(
        model=model,
        x=x,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
    )

    # Compute per-batch and per-image stats.
    times_ms_per_image = [t / args.batch_size for t in times_ms_per_batch]
    total_images = args.batch_size * args.runs
    total_time_s = sum(times_ms_per_batch) / 1_000.0
    throughput = total_images / total_time_s if total_time_s > 0 else float("inf")

    stats_batch = _compute_stats(times_ms_per_batch)
    stats_image = _compute_stats(times_ms_per_image)
    meets_prd = stats_image["mean"] <= _PRD_LATENCY_TARGET_MS

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "arch": args.arch,
        "num_classes": num_classes,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "seed": args.seed,
        "latency_ms_per_batch": stats_batch,
        "latency_ms_per_image": stats_image,
        "throughput_images_per_sec": round(throughput, 2),
        "prd_latency_target_ms": _PRD_LATENCY_TARGET_MS,
        "meets_prd_target": meets_prd,
    }

    _print_summary(result)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[INFO] Results saved to: {args.out}")


if __name__ == "__main__":
    main()
