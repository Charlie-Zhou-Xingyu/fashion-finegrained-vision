"""
Latency taxonomy utilities for per-sub-stage timing.

Defines a standard breakdown for every pipeline stage:
    preprocess -> H2D transfer -> model forward -> postprocess -> D2H transfer

Provides context managers and decorators that instrument code with
nanosecond-resolution wall-clock timers and CUDA synchronization.

Usage::

    from inference.latency_taxonomy import StageTimer, record_stage

    timer = StageTimer()
    with timer.substage("preprocess"):
        tensor = preprocess(image)
    with timer.substage("model_forward"):
        output = model(tensor)
    timer.report()  # -> dict with per-sub-stage latencies

Status: New module. Not yet integrated into existing pipeline.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SubstageTiming:
    """Timing for a single sub-stage."""
    name: str
    elapsed_ms: float = 0.0
    count: int = 0

    @property
    def avg_ms(self) -> float:
        return self.elapsed_ms / max(1, self.count)


@dataclass
class StageTiming:
    """Timing for one pipeline stage, decomposed into sub-stages.

    Canonical sub-stages (not every stage has all):
        preprocess   - CPU/GPU preprocessing (resize, normalize, letterbox)
        h2d          - Host-to-device data transfer
        model_forward - Pure model/engine forward pass
        postprocess  - GPU/CPU postprocessing (NMS, decode, threshold)
        d2h          - Device-to-host data transfer
        overhead     - Python overhead, dict construction, etc.

    Each sub-stage's elapsed_ms includes CUDA synchronization where applicable.
    """

    stage_name: str
    substages: Dict[str, SubstageTiming] = field(default_factory=dict)
    total_ms: float = 0.0
    batch_size: int = 1

    @property
    def model_only_ms(self) -> float:
        """Return model forward time only (no pre/post)."""
        mf = self.substages.get("model_forward")
        return mf.avg_ms if mf else 0.0

    @property
    def overhead_ms(self) -> float:
        """Total overhead: total - sum of all named substages."""
        named = sum(s.elapsed_ms for s in self.substages.values())
        return self.total_ms - named


# ── Timer ──────────────────────────────────────────────────────────────────────

class StageTimer:
    """Context-manager-based timer for per-sub-stage latency decomposition.

    Automatically synchronizes CUDA before each sub-stage timing boundary
    to avoid measuring asynchronous kernel launch overhead as model latency.

    Usage::

        timer = StageTimer("yolo_detection")
        with timer.substage("preprocess"):
            tensor = letterbox(image)
        with timer.substage("model_forward"):
            with torch.cuda.synchronize():
                output = engine.infer(tensor)
        with timer.substage("postprocess"):
            boxes = nms(output)
        report = timer.finalize()
        # -> StageTiming with per-sub-stage breakdown
    """

    def __init__(self, stage_name: str, sync_cuda: bool = True) -> None:
        self.stage_name = stage_name
        self._sync_cuda = sync_cuda
        self._substages: Dict[str, SubstageTiming] = {}
        self._t_start: float = 0.0

    def _sync(self) -> None:
        if self._sync_cuda:
            import torch  # lazy — only imported when CUDA sync is requested
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    @contextmanager
    def substage(self, name: str) -> Iterator[None]:
        """Time a named sub-stage. Synchronizes CUDA at entry and exit."""
        self._sync()
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._sync()
            elapsed_ns = time.perf_counter_ns() - t0
            elapsed_ms = elapsed_ns / 1_000_000
            if name not in self._substages:
                self._substages[name] = SubstageTiming(name=name)
            self._substages[name].elapsed_ms += elapsed_ms
            self._substages[name].count += 1

    def start(self) -> None:
        """Start the total stage timer."""
        self._sync()
        self._t_start = time.perf_counter_ns()

    def stop(self) -> float:
        """Stop the total stage timer. Returns total elapsed in ms."""
        self._sync()
        elapsed_ms = (time.perf_counter_ns() - self._t_start) / 1_000_000
        return elapsed_ms

    def finalize(self, batch_size: int = 1) -> StageTiming:
        """Build a StageTiming report from accumulated sub-stage measurements."""
        self._sync()
        return StageTiming(
            stage_name=self.stage_name,
            substages=dict(self._substages),
            batch_size=batch_size,
        )


# ── Decorator for function-level timing ────────────────────────────────────────

def record_stage(stage_name: str, sync_cuda: bool = True):
    """Decorator that times a function as a pipeline stage.

    The decorated function receives an extra keyword argument ``timer``
    containing a StageTimer instance.

    Example::

        @record_stage("yolo_detect")
        def detect(image, *, timer=None):
            with timer.substage("preprocess"):
                ...
            with timer.substage("model_forward"):
                ...
            return boxes
    """

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = StageTimer(stage_name, sync_cuda=sync_cuda)
            timer.start()
            kwargs["timer"] = timer
            result = func(*args, **kwargs)
            elapsed = timer.stop()
            timing = timer.finalize()
            timing.total_ms = elapsed
            if hasattr(result, "__dict__") and not isinstance(result, dict):
                result._timing = timing  # type: ignore[attr-defined]
            elif isinstance(result, dict):
                result["_timing"] = timing
            return result

        return wrapper

    return decorator


# ── Report formatting ──────────────────────────────────────────────────────────

def format_stage_table(timing: StageTiming, precision: int = 2) -> str:
    """Format a single StageTiming as an ASCII table string."""
    lines = [f"Stage: {timing.stage_name} (batch={timing.batch_size})"]
    lines.append(f"{'Sub-stage':<20} {'Total(ms)':>10} {'Avg(ms)':>10} {'Count':>7}")
    lines.append("-" * 47)
    for name in ("preprocess", "h2d", "model_forward", "postprocess", "d2h", "overhead"):
        ss = timing.substages.get(name)
        if ss and ss.count > 0:
            lines.append(
                f"{name:<20} {ss.elapsed_ms:>10.{precision}f} "
                f"{ss.avg_ms:>10.{precision}f} {ss.count:>7}"
            )
    lines.append("-" * 47)
    lines.append(f"{'TOTAL':<20} {timing.total_ms:>10.{precision}f}")
    if timing.overhead_ms > 0.01:
        lines.append(f"  overhead (unnamed): {timing.overhead_ms:.{precision}f} ms")
    return "\n".join(lines)


def format_pipeline_report(
    stage_timings: List[StageTiming], total_ms: float, num_images: int
) -> str:
    """Format a full pipeline benchmark report."""
    lines = [
        "=" * 70,
        f"Pipeline Benchmark Report",
        f"Images: {num_images}",
        f"Total wall-clock: {total_ms:.1f} ms ({total_ms/1000:.2f} s)",
        f"When throughput: {num_images / (total_ms/1000):.1f} QPS",
        "=" * 70,
    ]
    for st in stage_timings:
        lines.append(format_stage_table(st))
        lines.append("")
    return "\n".join(lines)


# ── Benchmark statistics ───────────────────────────────────────────────────────

def compute_stats(
    values: List[float],
) -> Dict[str, float]:
    """Compute summary statistics for a list of latency values (in ms).

    Returns dict with: mean, std, min, max, p50, p95, p99, count.
    """
    if not values:
        return {"count": 0}
    import math

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    sorted_vals = sorted(values)

    def percentile(p: float) -> float:
        k = (p / 100) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[f] * (c - k)
        d1 = sorted_vals[c] * (k - f)
        return d0 + d1

    return {
        "count": n,
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "min_ms": round(sorted_vals[0], 3),
        "max_ms": round(sorted_vals[-1], 3),
        "p50_ms": round(percentile(50), 3),
        "p95_ms": round(percentile(95), 3),
        "p99_ms": round(percentile(99), 3),
    }


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Smoke test: ensure timer and stats functions run without error."""
    timer = StageTimer("demo_stage")
    with timer.substage("preprocess"):
        time.sleep(0.001)
    with timer.substage("model_forward"):
        time.sleep(0.005)
    timing = timer.finalize()
    assert timing.substages["preprocess"].count == 1
    assert timing.substages["model_forward"].count == 1
    assert timing.substages["preprocess"].elapsed_ms > 0

    stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["count"] == 5
    assert abs(stats["mean_ms"] - 3.0) < 0.1
    assert stats["p50_ms"] == 3.0

    print(format_stage_table(timing))
    print("  latency_taxonomy: all checks passed.")


if __name__ == "__main__":
    _demo()
