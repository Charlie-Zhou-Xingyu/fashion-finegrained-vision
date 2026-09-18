"""
GPU-server-only. Builds a TensorRT-LLM engine for a text-only Qwen checkpoint.

PRD 3.2 / 4.3.2: weight-only INT4 (AWQ) quantization, paged KV-cache, INT8
KV-cache quantization, PagedAttention via TensorRT-LLM's gpt_attention
plugin. Targets TensorRT-LLM 0.10.0, which bundles TensorRT 10.0.1 — NOT
the PRD's "TensorRT 8.6.1" (that pin belongs to a much older TensorRT-LLM
line and is incompatible with 0.10; see docs/tensorrt_llm_integration.md).

Scope: this script drives TensorRT-LLM's generic ModelOpt quantize.py path,
which loads the checkpoint through HF AutoModelForCausalLM and assumes a
plain decoder-only module tree. That path CRASHES on Qwen-VL-Chat
('VisionTransformer' object has no attribute 'layers') because the vision
tower is embedded in the same top-level class. Use it for text-only Qwen
checkpoints (Qwen-7B-Chat, Qwen1.5-*). For Qwen-VL-Chat use the two-engine
recipe in docs/tensorrt_llm_integration.md#multimodal (examples/qwen/
convert_checkpoint.py + examples/qwenvl/vit_onnx_trt.py), which was
verified end-to-end.

This script does NOT run on the Mac dev machine — it requires an NVIDIA
GPU, CUDA 12.x, and a working ``tensorrt_llm`` install. It fails fast and
loudly if that environment is missing, rather than pretending to build
anything (no fabricated engine files).

For local dev / code review, use --dry-run: it prints the exact shell
commands without executing them or requiring tensorrt_llm to be
importable. The command-construction functions are pure and unit-tested
under tests/test_engines/.

Pipeline (matches TensorRT-LLM's unified ModelOpt quantization flow,
current as of 0.9-0.11 — see docs/tensorrt_llm_integration.md for the
version-compat caveat):

    1. examples/quantization/quantize.py
           HF checkpoint --qformat int4_awq --kv_cache_dtype int8
           --> TensorRT-LLM checkpoint (quantized + calibrated)
    2. trtllm-build
           TensorRT-LLM checkpoint --paged_kv_cache enable
           --> .engine file
    3. Register engine metadata into inference/engines/registry.json

Usage (on the GPU server)::

    python inference/engines/qwen_vl_trtllm_builder.py \\
        --hf-model-dir /models/Qwen-VL-7B-Chat \\
        --output-root /engines/qwen-vl-7b \\
        --calib-dataset ccdv/cnn_dailymail

Usage (dry-run, on the Mac, to review/test the command construction)::

    python inference/engines/qwen_vl_trtllm_builder.py --dry-run \\
        --hf-model-dir /models/Qwen-VL-7B-Chat --output-root /tmp/out
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_REGISTRY_PATH = _PROJECT_ROOT / "inference" / "engines" / "registry.json"


# ── Config ────────────────────────────────────────────────────────────────


@dataclass
class BuildConfig:
    hf_model_dir: str
    output_root: str
    engine_name: str = ""      # registry key — defaults to basename(hf_model_dir) if empty
    model_label: str = ""      # human-readable model name for the registry — same default
    dtype: str = "float16"
    qformat: str = "int4_awq"          # int4_awq | int8_sq | fp8 | full_prec
    kv_cache_dtype: str = "int8"       # int8 | fp8 | (omit for fp16 kv cache)
    calib_dataset: str = "ccdv/cnn_dailymail"
    calib_size: int = 512
    tp_size: int = 1
    max_batch_size: int = 8
    max_input_len: int = 2048
    max_output_len: int = 512
    paged_kv_cache: bool = True
    remove_input_padding: bool = True
    context_fmha: bool = True
    tensorrt_llm_examples_root: str = "third_party/TensorRT-LLM/examples"

    def __post_init__(self) -> None:
        basename = Path(self.hf_model_dir.rstrip("/")).name
        if not self.engine_name:
            self.engine_name = f"{basename}_trtllm"
        if not self.model_label:
            self.model_label = basename

    @property
    def quant_ckpt_dir(self) -> str:
        return str(Path(self.output_root) / "trt_ckpt")

    @property
    def engine_dir(self) -> str:
        return str(Path(self.output_root) / "trt_engine")


# ── Pure command builders (unit-testable without a GPU) ─────────────────────


def build_quantize_command(cfg: BuildConfig) -> List[str]:
    """examples/quantization/quantize.py invocation.

    ModelOpt-based unified quantization entry point. Produces a TensorRT-LLM
    checkpoint (weights + config.json) with weight-only INT4 AWQ and INT8
    paged KV-cache calibration baked in — this is where the KV-cache
    quantization PRD requirement is actually satisfied.
    """
    script = str(Path(cfg.tensorrt_llm_examples_root) / "quantization" / "quantize.py")
    cmd = [
        sys.executable, script,
        "--model_dir", cfg.hf_model_dir,
        "--dtype", cfg.dtype,
        "--qformat", cfg.qformat,
        "--output_dir", cfg.quant_ckpt_dir,
        "--calib_size", str(cfg.calib_size),
        "--tp_size", str(cfg.tp_size),
    ]
    if cfg.qformat != "full_prec":
        cmd += ["--calib_dataset", cfg.calib_dataset]
    if cfg.kv_cache_dtype:
        cmd += ["--kv_cache_dtype", cfg.kv_cache_dtype]
    return cmd


def build_trtllm_build_command(cfg: BuildConfig) -> List[str]:
    """``trtllm-build`` invocation.

    --paged_kv_cache enable is PagedAttention (vLLM-style block-based KV
    cache) — the PRD 4.3.2 "集成PagedAttention技术提升吞吐量" requirement.
    """
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", cfg.quant_ckpt_dir,
        "--output_dir", cfg.engine_dir,
        "--gemm_plugin", cfg.dtype,
        "--gpt_attention_plugin", cfg.dtype,
        "--max_batch_size", str(cfg.max_batch_size),
        "--max_input_len", str(cfg.max_input_len),
        "--max_output_len", str(cfg.max_output_len),
    ]
    cmd.append("--paged_kv_cache")
    cmd.append("enable" if cfg.paged_kv_cache else "disable")
    cmd.append("--remove_input_padding")
    cmd.append("enable" if cfg.remove_input_padding else "disable")
    cmd.append("--context_fmha")
    cmd.append("enable" if cfg.context_fmha else "disable")
    return cmd


# ── Environment guard ─────────────────────────────────────────────────────


class GpuEnvironmentError(RuntimeError):
    """Raised when this script is run somewhere that cannot build a TRT-LLM
    engine (no CUDA GPU, or tensorrt_llm not installed). Deliberately not
    caught anywhere — a build must fail loudly, not fall back silently."""


def check_gpu_environment() -> Dict[str, Any]:
    """Verify tensorrt_llm + a CUDA GPU are actually present.

    Raises GpuEnvironmentError with an actionable message on any Mac /
    CPU-only / missing-dependency environment. Returns a small info dict
    on success (used in the registry entry's ``build_env``).
    """
    if platform.system() == "Darwin":
        raise GpuEnvironmentError(
            "This script builds a TensorRT-LLM engine, which requires an NVIDIA "
            "CUDA GPU and has no macOS build. You're on macOS (Darwin) — this "
            "must run on the GPU server instead. Use --dry-run here to review "
            "the commands, then run for real on the server."
        )
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError as exc:
        raise GpuEnvironmentError(
            "tensorrt_llm is not installed in this environment. Install it on "
            "the GPU server: pip install -r requirements-gpu-server.txt "
            "(see docs/tensorrt_llm_integration.md for the exact CUDA/driver "
            "version this needs)."
        ) from exc
    try:
        import torch
        if not torch.cuda.is_available():
            raise GpuEnvironmentError(
                "No CUDA device visible to torch.cuda.is_available(). Check "
                "nvidia-smi and the container's --gpus flag."
            )
        gpu_name = torch.cuda.get_device_name(0)
    except GpuEnvironmentError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GpuEnvironmentError(f"Could not query CUDA device: {exc}") from exc

    return {"tensorrt_llm_version": tensorrt_llm.__version__, "gpu_name": gpu_name}


# ── Registry ──────────────────────────────────────────────────────────────


def load_registry(path: Path = _DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def register_engine(
    entry: Dict[str, Any],
    path: Path = _DEFAULT_REGISTRY_PATH,
) -> None:
    """Append (or replace, by ``name``) an engine entry in registry.json."""
    registry = load_registry(path)
    registry["engines"] = [e for e in registry["engines"] if e.get("name") != entry["name"]]
    registry["engines"].append(entry)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def make_registry_entry(cfg: BuildConfig, build_env: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": cfg.engine_name,
        "model": cfg.model_label,
        "framework": "tensorrt_llm",
        "engine_dir": cfg.engine_dir,
        "quant_ckpt_dir": cfg.quant_ckpt_dir,
        "dtype": cfg.dtype,
        "qformat": cfg.qformat,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "paged_kv_cache": cfg.paged_kv_cache,
        "tp_size": cfg.tp_size,
        "max_batch_size": cfg.max_batch_size,
        "max_input_len": cfg.max_input_len,
        "max_output_len": cfg.max_output_len,
        "build_env": build_env,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Orchestration ────────────────────────────────────────────────────────


def run_build(cfg: BuildConfig, dry_run: bool = False,
              registry_path: Path = _DEFAULT_REGISTRY_PATH) -> Optional[Dict[str, Any]]:
    quantize_cmd = build_quantize_command(cfg)
    build_cmd = build_trtllm_build_command(cfg)

    if dry_run:
        print("[DRY RUN] Step 1/3 — quantize (INT4-AWQ weights + INT8 paged KV-cache calibration):")
        print("  " + " ".join(quantize_cmd))
        print("[DRY RUN] Step 2/3 — trtllm-build (PagedAttention engine):")
        print("  " + " ".join(build_cmd))
        print(f"[DRY RUN] Step 3/3 — would register engine '{cfg.engine_name}' "
              f"into {registry_path}")
        return None

    build_env = check_gpu_environment()

    print("[1/3] Quantizing checkpoint (this downloads calibration data and can take a while)...")
    subprocess.run(quantize_cmd, check=True)

    print("[2/3] Building TensorRT-LLM engine...")
    subprocess.run(build_cmd, check=True)

    print("[3/3] Registering engine metadata...")
    entry = make_registry_entry(cfg, build_env)
    register_engine(entry, path=registry_path)
    print(f"Done. Engine at {cfg.engine_dir}")
    return entry


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model-dir", required=True, help="Path to HF Qwen checkpoint (text-only architectures)")
    p.add_argument("--output-root", required=True, help="Output root for quant checkpoint + engine")
    p.add_argument("--engine-name", default="", help="Registry key — defaults to basename(hf-model-dir)")
    p.add_argument("--model-label", default="", help="Human-readable model name for the registry — same default")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--qformat", default="int4_awq", choices=["int4_awq", "int8_sq", "fp8", "full_prec"])
    p.add_argument("--kv-cache-dtype", default="int8", choices=["int8", "fp8", "none"])
    p.add_argument("--calib-dataset", default="ccdv/cnn_dailymail")
    p.add_argument("--calib-size", type=int, default=512)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--max-batch-size", type=int, default=8,
                    help="Hard cap on concurrent sequences per forward pass — this, not the "
                         "quantization, is the first throughput ceiling (8 -> ~11 QPS, 64 -> ~58 QPS "
                         "on an RTX 4090 at 100 output tokens). 64 was the measured sweet spot; past "
                         "it per-batch latency step-jumps. 96 OOMs at build time on 24GB.")
    p.add_argument("--max-input-len", type=int, default=2048)
    p.add_argument("--max-output-len", type=int, default=512)
    p.add_argument("--no-paged-kv-cache", action="store_true")
    p.add_argument("--trtllm-examples-root", default="third_party/TensorRT-LLM/examples")
    p.add_argument("--registry-path", default=str(_DEFAULT_REGISTRY_PATH))
    p.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing (safe on non-GPU machines)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = BuildConfig(
        hf_model_dir=args.hf_model_dir,
        output_root=args.output_root,
        engine_name=args.engine_name,
        model_label=args.model_label,
        dtype=args.dtype,
        qformat=args.qformat,
        kv_cache_dtype="" if args.kv_cache_dtype == "none" else args.kv_cache_dtype,
        calib_dataset=args.calib_dataset,
        calib_size=args.calib_size,
        tp_size=args.tp_size,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        paged_kv_cache=not args.no_paged_kv_cache,
        tensorrt_llm_examples_root=args.trtllm_examples_root,
    )
    run_build(cfg, dry_run=args.dry_run, registry_path=Path(args.registry_path))


if __name__ == "__main__":
    main()
