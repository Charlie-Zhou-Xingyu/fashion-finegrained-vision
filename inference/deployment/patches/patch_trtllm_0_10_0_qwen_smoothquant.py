"""
Fix TensorRT-LLM 0.10.0 SmoothQuant (W8A8) builds for Qwen v1 checkpoints.

Symptom: `trtllm-build` on an `examples/qwen/convert_checkpoint.py --smoothquant`
checkpoint dies with
    RuntimeError: ... 'transformer.layers.0.mlp.fc.weight'
    Updated: (11008, 4096), original: (22016, 4096)

Root cause (in the library, not the converter): `models/qwen/model.py`
deliberately halves `config.intermediate_size` when it builds GatedMLP
("Qwen's real inter_size is one half of what's in the config"), but
`quantization/quantize.py::smooth_quantize_plugin` replaces that layer with
SmoothQuantGatedMLP and re-derives the width from `config.intermediate_size`
instead of reusing `layer.mlp.ffn_hidden_size`. Every other model family has
the two equal, so only Qwen v1 trips it. The checkpoint's 11008 is correct.

Fix: prefer the width the layer was actually constructed with. Non-gated
MLPs (no `ffn_hidden_size` attribute) fall through to the original expression.

Idempotent: asserts exactly one match before writing, prints "already
patched" on a second run. Keeps a `.orig_0.10.0` backup beside the file.
Run inside the environment that has tensorrt_llm installed:

    python inference/deployment/patches/patch_trtllm_0_10_0_qwen_smoothquant.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

OLD = (
    "        mlp_hidden_size = config.hidden_size * 4 "
    "if config.intermediate_size is None else config.intermediate_size\n"
)
NEW = (
    "        # Qwen(v1)'s model class halves config.intermediate_size when it builds GatedMLP;\n"
    "        # re-deriving the width from config here produced a 2x-wide SmoothQuant MLP that\n"
    "        # could not load the (correct) checkpoint. Prefer the size the layer was built with.\n"
    "        mlp_hidden_size = getattr(layer.mlp, 'ffn_hidden_size', None) or (\n"
    "            config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size)\n"
)
MARKER = "getattr(layer.mlp, 'ffn_hidden_size', None)"


def main() -> int:
    import tensorrt_llm  # noqa: F401 — locate the installed package, not a hardcoded path
    from tensorrt_llm.quantization import quantize as q

    target = Path(q.__file__)
    if target.suffix != ".py":
        print(f"refusing to patch non-source file {target}", file=sys.stderr)
        return 2
    src = target.read_text(encoding="utf-8")

    if MARKER in src:
        print(f"already patched: {target}")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"expected exactly one match of the original line in {target}, found {n} — "
              f"this patch targets tensorrt_llm 0.10.0 (installed: {tensorrt_llm.__version__}); "
              "not modifying anything.", file=sys.stderr)
        return 1

    backup = target.with_name(target.name + ".orig_0.10.0")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(src.replace(OLD, NEW), encoding="utf-8")
    compile(target.read_text(encoding="utf-8"), str(target), "exec")  # syntax check
    print(f"patched: {target}\nbackup:  {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
