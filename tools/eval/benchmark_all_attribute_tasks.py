"""Benchmark all 8 attribute classifier checkpoints from attribute_inference.yaml.

Iterates over every task defined in ``configs/attribute_inference.yaml``, loads
the real checkpoint, and measures per-image inference latency using synthetic
input tensors.  No dataset or image files are read.

Latency is measured for:
  - CPU (always)
  - CUDA GPU (if torch.cuda.is_available())

Reports per-task mean / p50 / p95 latency, PRD target pass/fail for each task,
and an estimated sequential total across all tasks enabled for a given coarse
garment class.

Usage
-----
::

    python tools/eval/benchmark_all_attribute_tasks.py \\
        --inference-config configs/attribute_inference.yaml \\
        --output-dir outputs/benchmarks/attribute_latency \\
        --warmup 20 \\
        --runs 200

Outputs
-------
``outputs/benchmarks/attribute_latency/latency_report.json``
``outputs/benchmarks/attribute_latency/latency_report.md``
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# sys.path — project root + src/
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
    load_id_to_label,
)

# Reuse pure helpers from the single-task benchmark.
from tools.eval.benchmark_attribute_latency import (  # noqa: E402
    _compute_stats,
    _run_benchmark,
    _resolve_device,
)

logger = logging.getLogger(__name__)

_PRD_LATENCY_TARGET_MS: float = 20.0

# Coarse class → tasks enabled by attribute_group_mapping.yaml (reference copy for
# the sequential-total estimate).  These must stay in sync with the YAML.
_COARSE_CLASS_TO_TASKS: dict[str, list[str]] = {
    "top":       ["neckline_design", "collar_design", "neck_design", "sleeve_length"],
    "outerwear": ["lapel_design", "coat_length", "sleeve_length"],
    "pants":     ["pant_length"],
    "skirt":     ["skirt_length"],
    "dress":     ["neckline_design", "skirt_length", "sleeve_length"],
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_inference_config(path: Path) -> dict[str, dict[str, Any]]:
    """Load attribute_inference.yaml and return the ``tasks`` sub-dict.

    Args:
        path: Path to ``attribute_inference.yaml``.

    Returns:
        Dict mapping task name → task config.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the YAML lacks a ``tasks`` key.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"attribute_inference.yaml not found: {path}\n"
            "Expected: configs/attribute_inference.yaml"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "tasks" not in raw:
        raise KeyError(f"YAML at {path} has no 'tasks' key — check the format.")
    return raw["tasks"]


# ---------------------------------------------------------------------------
# Report formatting helpers (testable without weights)
# ---------------------------------------------------------------------------


def compute_sequential_totals(
    per_task_results: dict[str, dict[str, Any]],
    coarse_class_to_tasks: dict[str, list[str]],
    device_key: str,
) -> dict[str, dict[str, float]]:
    """Estimate sequential total latency for each coarse garment class.

    Sums per-task mean latencies for the tasks enabled per coarse class.
    Tasks with no latency data (checkpoint missing) are skipped.

    Args:
        per_task_results: ``{task: {device_key: {mean, p50, p95, ...}, ...}}``
        coarse_class_to_tasks: Mapping from coarse class name to task list.
        device_key: Device string used as key in per_task_results (e.g. ``"cpu"``).

    Returns:
        ``{coarse_class: {"tasks_included": N, "total_mean_ms": float, ...}}``
    """
    totals: dict[str, dict[str, Any]] = {}
    for coarse_cls, tasks in coarse_class_to_tasks.items():
        total_mean = 0.0
        total_p95 = 0.0
        included = 0
        for task in tasks:
            stats = (per_task_results.get(task) or {}).get(device_key)
            if stats and stats.get("mean") is not None:
                total_mean += stats["mean"]
                total_p95 += stats.get("p95", stats["mean"])
                included += 1
        totals[coarse_cls] = {
            "tasks_included": included,
            "tasks_expected": len(tasks),
            "total_mean_ms": round(total_mean, 3),
            "total_p95_ms": round(total_p95, 3),
            "meets_prd_target": total_mean <= _PRD_LATENCY_TARGET_MS,
        }
    return totals


def format_latency_report_md(report: dict[str, Any]) -> str:
    """Render the full benchmark report as a Markdown string.

    Args:
        report: Dict returned by :func:`run_all_benchmarks`.

    Returns:
        Markdown text (no trailing newline).
    """
    meta = report.get("meta", {})
    per_task = report.get("per_task", {})
    sequential = report.get("sequential_totals", {})

    lines: list[str] = [
        "# Attribute Classifier Latency Benchmark Report",
        "",
        f"> Generated: {meta.get('timestamp', 'unknown')}",
        f"> PRD latency target: ≤ {_PRD_LATENCY_TARGET_MS} ms per image",
        f"> Warmup runs: {meta.get('warmup', 0)} | Timed runs: {meta.get('runs', 0)}",
        f"> Devices tested: {', '.join(meta.get('devices_tested', []))}",
        "",
    ]

    for device in meta.get("devices_tested", []):
        lines += [
            f"## Per-task Latency — {device.upper()}",
            "",
            "| Task | Checkpoint exists | Mean ms | p50 ms | p95 ms | PRD (≤20ms) |",
            "|---|---|---:|---:|---:|---|",
        ]
        for task in sorted(per_task.keys()):
            task_data = per_task[task]
            ckpt_ok = task_data.get("checkpoint_exists", False)
            stats = task_data.get(device)
            if not ckpt_ok:
                lines.append(f"| {task} | ✗ missing | — | — | — | — |")
            elif stats is None:
                lines.append(f"| {task} | ✓ | (error) | — | — | — |")
            else:
                mean = stats.get("mean", 0.0)
                p50 = stats.get("p50", 0.0)
                p95 = stats.get("p95", 0.0)
                ok = "✓ PASS" if mean <= _PRD_LATENCY_TARGET_MS else "✗ FAIL"
                lines.append(
                    f"| {task} | ✓ | {mean:.2f} | {p50:.2f} | {p95:.2f} | {ok} |"
                )
        lines.append("")

        if sequential:
            lines += [
                f"## Sequential Total Latency per Coarse Class — {device.upper()}",
                "",
                "*(Sum of per-task mean latencies for all tasks enabled for each garment class.)*",
                "",
                "| Coarse class | Tasks | Total mean ms | Total p95 ms | ≤ 20ms? |",
                "|---|---:|---:|---:|---|",
            ]
            totals = report.get("sequential_totals", {}).get(device, {})
            for cls_name, cls_data in sorted(totals.items()):
                n_inc = cls_data["tasks_included"]
                n_exp = cls_data["tasks_expected"]
                t_mean = cls_data["total_mean_ms"]
                t_p95 = cls_data["total_p95_ms"]
                ok = "✓" if cls_data["meets_prd_target"] else "✗"
                task_str = f"{n_inc}/{n_exp}"
                lines.append(
                    f"| {cls_name} | {task_str} | {t_mean:.2f} | {t_p95:.2f} | {ok} |"
                )
            lines.append("")

    lines += [
        "## PRD 3.1.3 Summary",
        "",
        "| PRD Requirement | Status |",
        "|---|---|",
    ]
    for device in meta.get("devices_tested", []):
        all_pass = all(
            (per_task[t].get(device) or {}).get("mean", 999) <= _PRD_LATENCY_TARGET_MS
            for t in per_task
            if per_task[t].get("checkpoint_exists")
        )
        status = "✓ ALL PASS" if all_pass else "✗ SOME FAIL"
        lines.append(f"| Per-task ≤ 20ms ({device}) | {status} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-task benchmark
# ---------------------------------------------------------------------------


def _benchmark_one_task(
    task: str,
    task_cfg: dict[str, Any],
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict[str, Any] | None:
    """Load one task's checkpoint and run the latency benchmark.

    Args:
        task: Task name (for logging).
        task_cfg: Task config dict from attribute_inference.yaml.
        device: Torch device.
        warmup: Warmup passes.
        runs: Timed passes.

    Returns:
        Stats dict from :func:`_compute_stats`, or ``None`` on error.
    """
    ckpt_rel = task_cfg.get("checkpoint", "")
    ckpt = _PROJECT_ROOT / ckpt_rel if not Path(ckpt_rel).is_absolute() else Path(ckpt_rel)
    arch = task_cfg.get("arch", "resnet18")
    img_size = int(task_cfg.get("img_size", 224))

    # Load label map to determine num_classes reliably.
    label_map_rel = task_cfg.get("label_map", "")
    label_map_path = (
        _PROJECT_ROOT / label_map_rel
        if not Path(label_map_rel).is_absolute()
        else Path(label_map_rel)
    )
    num_classes: int | None = None
    if label_map_path.exists():
        try:
            id_to_label = load_id_to_label(str(label_map_path))
            num_classes = len(id_to_label)
        except Exception as exc:
            logger.warning("Task %r: could not load label map: %s", task, exc)

    try:
        state = load_checkpoint_state(ckpt, device)
    except Exception as exc:
        logger.error("Task %r: failed to load checkpoint %s: %s", task, ckpt, exc)
        return None

    if num_classes is None:
        num_classes = infer_num_classes_from_state(state)
    if num_classes is None:
        logger.error("Task %r: cannot determine num_classes.", task)
        return None

    try:
        model = build_attribute_classifier(arch=arch, num_classes=num_classes, pretrained=False)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        logger.error("Task %r: failed to build/load model: %s", task, exc)
        return None

    x = torch.zeros(1, 3, img_size, img_size, device=device)

    times_ms = _run_benchmark(model, x, device, warmup=warmup, runs=runs)
    return _compute_stats(times_ms)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_all_benchmarks(
    inference_config: Path,
    warmup: int = 20,
    runs: int = 200,
    force_cpu_only: bool = False,
) -> dict[str, Any]:
    """Benchmark all attribute tasks across all available devices.

    Args:
        inference_config: Path to ``attribute_inference.yaml``.
        warmup: Warmup passes per task.
        runs: Timed passes per task.
        force_cpu_only: If True, skip CUDA even when available.

    Returns:
        Full report dict ready for JSON serialisation.
    """
    task_configs = load_inference_config(inference_config)

    devices_to_test: list[torch.device] = [torch.device("cpu")]
    if not force_cpu_only and torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))

    device_names = [str(d) for d in devices_to_test]
    logger.info("Devices to benchmark: %s", device_names)
    logger.info("Tasks: %s", sorted(task_configs.keys()))

    per_task: dict[str, dict[str, Any]] = {}

    for task, task_cfg in sorted(task_configs.items()):
        ckpt_rel = task_cfg.get("checkpoint", "")
        ckpt = _PROJECT_ROOT / ckpt_rel if not Path(ckpt_rel).is_absolute() else Path(ckpt_rel)
        ckpt_exists = ckpt.exists()

        per_task[task] = {
            "checkpoint": str(ckpt),
            "checkpoint_exists": ckpt_exists,
            "arch": task_cfg.get("arch", "resnet18"),
            "img_size": task_cfg.get("img_size", 224),
        }

        if not ckpt_exists:
            logger.warning("Task %r: checkpoint not found: %s", task, ckpt)
            continue

        for device in devices_to_test:
            dev_key = str(device)
            logger.info("Benchmarking task=%r  device=%s  warmup=%d  runs=%d",
                        task, dev_key, warmup, runs)
            t_start = time.perf_counter()
            stats = _benchmark_one_task(task, task_cfg, device, warmup, runs)
            elapsed = time.perf_counter() - t_start
            logger.info("  → mean %.2f ms  p95 %.2f ms  (wall %.1f s)",
                        (stats or {}).get("mean", 0.0),
                        (stats or {}).get("p95", 0.0),
                        elapsed)

            if stats is not None:
                stats["meets_prd_target"] = stats["mean"] <= _PRD_LATENCY_TARGET_MS
            per_task[task][dev_key] = stats

    # Sequential total estimates.
    seq_totals: dict[str, dict[str, Any]] = {}
    for device in devices_to_test:
        dev_key = str(device)
        seq_totals[dev_key] = compute_sequential_totals(
            per_task, _COARSE_CLASS_TO_TASKS, dev_key
        )

    return {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "inference_config": str(inference_config),
            "warmup": warmup,
            "runs": runs,
            "devices_tested": device_names,
            "prd_latency_target_ms": _PRD_LATENCY_TARGET_MS,
        },
        "per_task": per_task,
        "sequential_totals": seq_totals,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark all attribute classifier checkpoints from attribute_inference.yaml. "
            "Uses synthetic tensors — no dataset or image files are read."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inference-config", type=Path,
        default=Path("configs/attribute_inference.yaml"),
        help="Path to attribute_inference.yaml.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/benchmarks/attribute_latency"),
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Skip CUDA even when available.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    report = run_all_benchmarks(
        inference_config=args.inference_config,
        warmup=args.warmup,
        runs=args.runs,
        force_cpu_only=args.cpu_only,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "latency_report.json"
    md_path = args.output_dir / "latency_report.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(
        format_latency_report_md(report) + "\n", encoding="utf-8"
    )

    logger.info("JSON report : %s", json_path)
    logger.info("MD report   : %s", md_path)

    # Console summary.
    per_task = report["per_task"]
    meta = report["meta"]
    print(f"\n{'='*65}")
    print(f"  Attribute Latency Benchmark — {meta['timestamp']}")
    print(f"  warmup={meta['warmup']}  runs={meta['runs']}")
    print(f"{'='*65}")

    for device in meta["devices_tested"]:
        print(f"\n  Device: {device.upper()}")
        print(f"  {'Task':<22} {'Mean ms':>9} {'p50 ms':>9} {'p95 ms':>9} {'PRD':>8}")
        print(f"  {'-'*60}")
        for task in sorted(per_task.keys()):
            td = per_task[task]
            if not td.get("checkpoint_exists"):
                print(f"  {task:<22} {'MISSING':>9}")
                continue
            stats = td.get(device)
            if stats is None:
                print(f"  {task:<22} {'ERROR':>9}")
                continue
            prd = "PASS" if stats["meets_prd_target"] else "FAIL"
            print(
                f"  {task:<22} {stats['mean']:>8.2f} {stats['p50']:>8.2f}"
                f" {stats['p95']:>8.2f} {prd:>8}"
            )

        seq = report["sequential_totals"].get(device, {})
        if seq:
            print(f"\n  Sequential totals ({device}):")
            for cls_name, cls_d in sorted(seq.items()):
                tasks_str = f"{cls_d['tasks_included']}/{cls_d['tasks_expected']} tasks"
                ok = "PASS" if cls_d["meets_prd_target"] else "FAIL"
                print(
                    f"    {cls_name:<12} {tasks_str:<12}"
                    f" total_mean={cls_d['total_mean_ms']:.2f} ms  [{ok}]"
                )
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
