"""
Capture hardware and software environment for reproducible benchmarks.

Records GPU, driver, CUDA, PyTorch, TensorRT versions and system specs.
Outputs JSON suitable for inclusion in benchmark reports.

Usage::

    python inference/env_capture.py --output benchmark_env.json

    # Or as library:
    from inference.env_capture import capture_env
    env = capture_env()
    print(env["gpu_name"])

Status: New module. Ready for use in Week 0 benchmark hygiene.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a command and return stripped stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, shell=False
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_nvidia_smi() -> Dict[str, Any]:
    """Query nvidia-smi for GPU details."""
    info: Dict[str, Any] = {}
    try:
        raw = _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,"
            "power.limit,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        if not raw:
            return info
        lines = raw.split("\n")
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                gpus.append({
                    "name": parts[0],
                    "driver": parts[1],
                    "vram_total_mb": parts[2],
                    "vram_free_mb": parts[3],
                    "power_limit_w": parts[4],
                    "sm_clock_mhz": parts[5],
                    "mem_clock_mhz": parts[6],
                    "temp_c": parts[7],
                })
        info["gpus"] = gpus
        info["gpu_count"] = len(gpus)
    except Exception:
        pass
    return info


def _get_cuda_version() -> Optional[str]:
    """Best-effort CUDA version detection."""
    # Try nvcc first
    ver = _run(["nvcc", "--version"])
    if ver:
        for line in ver.split("\n"):
            if "release" in line:
                return line.strip().split("release")[-1].strip().split(",")[0]
    # Fallback: check nvidia-smi
    smi = _run(["nvidia-smi"])
    if smi:
        for line in smi.split("\n"):
            if "CUDA Version" in line:
                return line.strip().split(":")[-1].strip()
    return None


def _get_torch_info() -> Dict[str, Any]:
    """Get PyTorch version and CUDA availability."""
    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version_torch"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_name_torch"] = torch.cuda.get_device_name(0)
            info["gpu_count_torch"] = torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = "not installed"
    return info


def _get_trt_info() -> Dict[str, Any]:
    """Get TensorRT version."""
    info: Dict[str, Any] = {}
    try:
        import tensorrt
        info["tensorrt_version"] = tensorrt.__version__
    except ImportError:
        info["tensorrt_version"] = "not installed"
    return info


def _get_system_info() -> Dict[str, Any]:
    """Get CPU, RAM, OS info."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "cpu": platform.processor() or _run(["wmic", "cpu", "get", "name"]).split("\n")[1].strip() if platform.system() == "Windows" else platform.machine(),
        "cpu_cores_physical": os.cpu_count(),
        "python_version": sys.version.split()[0],
    }


def capture_env(
    gpu_clock_locked: Optional[bool] = None,
    cudnn_benchmark: Optional[bool] = None,
) -> Dict[str, Any]:
    """Capture full environment for benchmark reproducibility.

    Args:
        gpu_clock_locked: Whether GPU clocks were locked during benchmark.
        cudnn_benchmark: Whether cudnn.benchmark was enabled during benchmark.

    Returns:
        Dict with keys: system, gpu, torch, tensorrt, benchmark_config.
    """
    env: Dict[str, Any] = {
        "captured_at": _run(["date", "/T"]).strip() if platform.system() == "Windows" else _run(["date"]).strip(),
        "system": _get_system_info(),
        "gpu": _get_nvidia_smi(),
        "torch": _get_torch_info(),
        "tensorrt": _get_trt_info(),
    }

    # Add CUDA version if not already in torch section
    if not env["torch"].get("cuda_version_torch"):
        cuda_ver = _get_cuda_version()
        if cuda_ver:
            env["gpu"]["cuda_version_nvcc"] = cuda_ver

    # Benchmark-specific config
    env["benchmark_config"] = {
        "gpu_clock_locked": gpu_clock_locked,
        "cudnn_benchmark": cudnn_benchmark,
    }

    return env


def capture_env_cli() -> None:
    """CLI entry point: capture env and write JSON."""
    ap = argparse.ArgumentParser(
        description="Capture hardware/software environment for benchmark reproducibility"
    )
    ap.add_argument(
        "--output", "-o", type=Path, default=Path("benchmark_env.json"),
        help="Output JSON file path"
    )
    ap.add_argument(
        "--gpu-clock-locked", type=lambda s: s.lower() == "true", default=None,
        help="Whether GPU clock was locked during benchmark"
    )
    ap.add_argument(
        "--cudnn-benchmark", type=lambda s: s.lower() == "true", default=None,
        help="Whether cudnn.benchmark was enabled"
    )
    args = ap.parse_args()

    env = capture_env(
        gpu_clock_locked=args.gpu_clock_locked,
        cudnn_benchmark=args.cudnn_benchmark,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False, default=str)

    print(f"Environment captured to: {args.output}")
    print(f"  GPU: {env['torch'].get('gpu_name_torch', 'unknown')}")
    print(f"  PyTorch: {env['torch'].get('torch_version', 'unknown')}")
    print(f"  TensorRT: {env['tensorrt'].get('tensorrt_version', 'unknown')}")


if __name__ == "__main__":
    capture_env_cli()
