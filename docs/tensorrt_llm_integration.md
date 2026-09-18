# TensorRT-LLM / Qwen-VL Integration

Covers PRD §3.2 (多模态大模型推理优化模块) and the "推理优化" row of §4.2's
tech-stack table. Scope: build a TensorRT-LLM engine for Qwen-VL-7B-Chat with
weight-only INT4 (AWQ) quantization, INT8 paged KV-cache quantization, and
PagedAttention, then serve it behind an OpenAI-compatible HTTP endpoint that
`inference/llm/mllm_client.py` talks to.

## Where things run

| Piece | Runs on | Why |
|---|---|---|
| `inference/llm/mllm_client.py` | anywhere (Mac dev, CPU serving box) | HTTP client only, no model loaded in-process |
| `inference/serving/app.py` (existing orchestrator) | anywhere | calls `mllm_client` over HTTP when wired in |
| `inference/engines/qwen_vl_trtllm_builder.py` | **GPU server only** | shells out to TensorRT-LLM's checkpoint-convert + `trtllm-build`, both CUDA-only |
| `inference/serving/trtllm_server.py` | **GPU server only** | loads the built engine, serves `/v1/chat/completions` |

Apple Silicon (M-series Mac) cannot run any GPU-only row — TensorRT-LLM,
TensorRT, and `onnxruntime-gpu` have no macOS build, full stop. This isn't a
missing-package problem; there's no wheel to install. `requirements.txt`
stays Mac/CPU-safe; GPU-only deps live in `requirements-gpu-server.txt`.

## Known issue: the PRD's own version pin is inconsistent

§4.2 lists "TensorRT-LLM 0.10、TensorRT 8.6、ONNX Runtime 1.17" with a
version-requirement column of "TensorRT 8.6.1". **TensorRT-LLM 0.10 does not
pair with TensorRT 8.6.1** — that TensorRT version belongs to a much older
TensorRT-LLM release (roughly the 0.5–0.7 line). TensorRT-LLM 0.10 (released
2024-06) bundles **TensorRT 10.0.1** as a transitive dependency. Installing
`tensorrt_llm==0.10.0` will pull TensorRT 10.0.1 regardless of what's
manually pinned — trying to force 8.6.1 alongside it breaks the install.

**What we actually used** (verified working on a rented RunPod RTX 4090,
Ubuntu 24.04, driver 570.195/CUDA 12.8):

| Component | Version | Note |
|---|---|---|
| CUDA (driver) | 12.8 (host) | backward-compatible; wheels below target 12.1 runtime, which is fine |
| Python | 3.10.21 (conda-forge env) | TensorRT-LLM 0.10 only ships cp310/cp311 wheels — **cp312 fails** (see below) |
| PyTorch | 2.3.0+cu121 | `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121` |
| TensorRT-LLM | 0.10.0 | `pip install --extra-index-url https://pypi.nvidia.com tensorrt_llm==0.10.0` |
| TensorRT | 10.0.1 | pulled automatically by tensorrt_llm, do not pin 8.6.1 separately |

Requirements.txt keeps this pin for engine-build reproducibility, but treat
it as a starting point, not gospel — TensorRT-LLM's pinned CUDA/PyTorch
combination has moved with almost every minor release; check
`pip index versions tensorrt_llm` and the release's own requirements before
reusing this on a different base image.

## Environment pitfalls hit during setup (and fixes)

These cost real debugging time on the rented box — recorded so the next
setup skips them:

1. **Base-image Python is 3.12, TensorRT-LLM 0.10 needs 3.10/3.11.** The
   RunPod `pytorch:...-torch280-ubuntu2404` image ships system Python 3.12
   with no conda. `pip install tensorrt_llm==0.10.0` under 3.12 falls back to
   building from an sdist and fails at `metadata-generation-failed` — it
   isn't meant to be source-built. Fix: install Miniconda and create a
   `python=3.10` env instead of using the system Python or a plain venv.

2. **`conda create` fails with a ToS error on fresh Miniconda installs.**
   Recent conda versions gate the `defaults` channel (`repo.anaconda.com/pkgs/main`,
   `pkgs/r`) behind a Terms-of-Service acceptance prompt, which breaks
   non-interactively (`CondaToSNonInteractiveError`). If the env-create step
   fails silently, later commands (`conda activate`, `pip install`) run in
   `base`, not `trtllm`, often against a different Python entirely
   (miniconda's own base env floated to Python 3.14 during this setup) —
   every subsequent install then targets the wrong interpreter. Fix: create
   the env from `conda-forge` instead of accepting the ToS:
   `conda create -n trtllm python=3.10 -y -c conda-forge --override-channels`.
   **Always verify `python3 --version` after `conda activate` before
   installing anything** — this failure mode is silent otherwise.

3. **No sshd running by default in the container.** `service ssh start`
   fails with `sshd: no hostkeys available -- exiting` on a fresh container
   (host keys aren't generated). Fix: `ssh-keygen -A` before `service ssh
   start`.

4. **Pasting large multi-line scripts into RunPod's JupyterLab web terminal
   silently corrupts them.** xterm.js drops/reorders bytes on big pastes
   under latency — this produced a file that was a byte-salad of the
   docstring's start and the argparse block's end, with no error at paste
   time. Symptoms: garbled terminal echo, stray `-bash: 1: command not
   found` lines. Two reliable workarounds: (a) upload the file through
   JupyterLab's own file-browser upload button (real HTTP file transfer, not
   keystroke injection) instead of pasting into the terminal; (b) once
   direct SSH access works, write the file via `ssh host 'cat > path' <<
   'EOF' ... EOF` from a real terminal (not the browser one) — always
   `ast.parse()` (or equivalent) the result to confirm it isn't truncated.

5. **Direct SSH to some hosts times out at the banner-exchange stage**
   (TCP handshake completes, `SSH-2.0-...` banner never arrives) while ICMP
   and the TCP SYN/ACK both succeed. Seen against both a China-hosted GPU
   rental (SeetaCloud) and, intermittently, a US-hosted RunPod IP — client
   network path dependent, not something to debug by retrying against a
   different destination. If this happens, fall back to the platform's
   browser-based terminal (HTTPS, generally unaffected) for setup, and keep
   retrying direct SSH once the network path changes (different network,
   VPN, etc.) — it isn't a server-side problem, so there's nothing to fix on
   the instance itself.

6. **SmoothQuant (W8A8) build fails on Qwen v1 with a shape mismatch — a
   TensorRT-LLM 0.10.0 library bug, located to the line.** `trtllm-build`
   dies loading `transformer.layers.0.mlp.fc.weight`: checkpoint (11008, 4096),
   model expects (22016, 4096). Root cause, confirmed by reading the source
   (not by guessing): `models/qwen/model.py` deliberately halves
   `config.intermediate_size` (22016 → 11008, "Qwen's real inter_size is one
   half of what's in the config") when it builds `GatedMLP`, but
   `quantization/quantize.py::smooth_quantize_plugin` then *replaces* that
   layer with `SmoothQuantGatedMLP` and re-derives the width from
   `config.intermediate_size` instead of reusing `layer.mlp.ffn_hidden_size`.
   Every other family has the two values equal, so only Qwen v1 trips it. Not
   a converter bug (the converter's 11008 is correct; llama's SQ converter
   writes the same unfused layout), not `fuse_gate_mlp` (`--use_fused_mlp`
   defaults off and has no SQ auto-enable), not `hidden_act` (both
   checkpoints say `silu`) — each of those was checked and eliminated before
   touching anything. Fix applied on the GPU box (backup kept at
   `quantize.py.orig_0.10.0`): prefer the size the layer was actually built
   with —
   `mlp_hidden_size = getattr(layer.mlp, 'ffn_hidden_size', None) or (...original expression...)`.
   One line, reversible, no weight surgery; non-gated MLPs fall through to
   the original path. The README's claim that SmoothQuant "supports Qwen
   models" is not true on 0.10.0 without this.

## Multimodal — how real image input actually works (verified end-to-end)

Two TensorRT engines, not one. Confirmed by actually building and running
both on the RunPod RTX 4090:

1. **`qwen_vl_trtllm_builder.py`** builds the **text decoder** engine — this
   is the piece `trtllm-build --paged_kv_cache enable` and the INT4/INT8
   quantization apply to, and the piece that dominates cost/latency for a 7B
   model. It does **not** touch the vision tower.
2. The vision encoder is a **separate, plain TensorRT engine** (not
   TensorRT-LLM), built via TensorRT-LLM's own `examples/qwenvl/` directory
   — this is NOT under `examples/multimodal/` (that directory's
   `build_visual_engine.py` supports llava/vila/cogvlm/fuyu/neva/pix2struct/
   nougat but has no Qwen-VL code path at all; `examples/qwenvl/` is the
   actual Qwen-VL-specific one, easy to miss since it's not cross-referenced
   from `examples/multimodal/`'s README).

### The real pipeline (this is what we ran)

```bash
# 1. Vision encoder: HF checkpoint -> ONNX -> plain TensorRT engine
cd TensorRT-LLM/examples/qwenvl
python3 vit_onnx_trt.py --pretrained_model_path <Qwen-VL-Chat dir>
#   loads the FULL HF model (AutoModelForCausalLM, needs the GPU) just to
#   reach model.transformer.visual, torch.onnx.exports THAT submodule, then
#   builds a TensorRT engine from the ONNX in the SAME process — the full
#   7B model is still resident in GPU memory during the TRT build step,
#   which throws `OutOfMemory` on a 24GB GPU (a real bug/oversight in the
#   reference script, not a config problem). Fix: rerun with --only_trt,
#   which skips reloading the HF model and builds straight from the
#   already-exported ONNX — GPU memory is clean by then. Output:
#   plan/visual_encoder/visual_encoder_fp16.plan (~3.9GB, ~100s build).
#   ViT output shape: [-1, 256, 4096] — 256 vision tokens/image, 4096-dim,
#   matching Qwen-VL's known architecture.

# 2. Text decoder: Qwen-VL-Chat's checkpoint -> Qwen-family converter
#    (NOT quantize.py/ModelOpt — see the crash below) -> trtllm-build
python3 ../qwen/convert_checkpoint.py --model_dir=<Qwen-VL-Chat dir> \
    --output_dir=./tllm_checkpoint_1gpu --dtype float16 \
    --use_weight_only --weight_only_precision int4 --int8_kv_cache
trtllm-build --checkpoint_dir=./tllm_checkpoint_1gpu \
    --gemm_plugin=float16 --gpt_attention_plugin=float16 \
    --lookup_plugin=float16 --max_input_len=2048 --max_output_len=1024 \
    --max_batch_size=8 --max_prompt_embedding_table_size=2048 \
    --remove_input_padding=enable --paged_kv_cache=enable \
    --output_dir=./trt_engines/Qwen-VL-7B-Chat

# 3. Run both together
python3 run.py --tokenizer_dir=<Qwen-VL-Chat dir> \
    --qwen_engine_dir=./trt_engines/Qwen-VL-7B-Chat --vit_engine_dir=./plan \
    --images_path='{"image": "./pics/demo.jpeg"}' --input_dir='{"image": "image.pt"}' \
    --input_text="这张图片里有什么？"
```

**Why `examples/quantization/quantize.py` (the generic ModelOpt path used
for the text-only Qwen-7B-Chat build above) crashes on Qwen-VL-Chat's
checkpoint but `examples/qwen/convert_checkpoint.py` doesn't**: `quantize.py`
loads the checkpoint via `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)`, which instantiates Qwen-VL's *full* model class —
vision tower included — then walks that live module tree assuming a plain
decoder-only structure (`model.layers`), and crashes with `'VisionTransformer'
object has no attribute 'layers'` the moment it hits `model.transformer.visual`.
`convert_checkpoint.py` works at the raw state_dict level instead — it reads
tensor names directly out of the safetensors files and only picks up keys
matching the expected `transformer.h.*` / `transformer.wte` naming pattern.
`visual.*`-prefixed keys just don't match anything it's looking for, so they're
silently skipped. No special-casing for Qwen-VL exists in that script; it works
by construction, not by any VL-specific logic — confirmed by grepping the
script for "visual"/"VisionTransformer" (zero matches) and then watching it
convert cleanly with no errors, right down to normal INT8-KV-cache calibration.

### Embedding injection mechanism

`--lookup_plugin=float16` + `--max_prompt_embedding_table_size=2048` is
TensorRT-LLM's p-tuning / prompt-tuning embedding-table mechanism, repurposed
here for image conditioning rather than tunable soft prompts: the 256
per-image vision embeddings from the ViT engine get placed in a lookup table
that the text engine's forward pass substitutes in at specific input
positions, instead of the token embedding it would normally look up. 2048 /
256 = headroom for 8 images per request at this setting.

### Measured latency (RTX 4090, this exact build — not PRD targets, actual numbers)

- ViT (image → 256×4096 embeddings): **56ms**
- Qwen text decoder (embeddings + text → answer): **1.02s**
- Combined: ~1.08s for one image + short Chinese question → coherent, visually
  accurate answer (verified against the actual demo image content, not just
  "it didn't crash")

This is total latency for a cold `run.py` invocation including engine
deserialization on this call; it is not a warm-server steady-state number and
not the PRD's ≤400ms/≥60QPS target — those need an actual load test (e.g.
against `inference/serving/trtllm_server.py` running as a persistent
process), which is still open work.

### Server integration — done and verified over real HTTP

`inference/serving/trtllm_server.py` now auto-detects mode from env vars at
startup: `MLLM_TEXT_ENGINE_DIR` alone → text-only via `hlapi.LLM`;
`MLLM_TEXT_ENGINE_DIR` + `MLLM_VIT_ENGINE_DIR` + `MLLM_TOKENIZER_DIR` +
`MLLM_QWENVL_EXAMPLES_DIR` → multimodal, importing `QWenInfer` and the ViT
session-running logic straight from the vendored `examples/qwenvl/run.py`
(not reimplemented — see the reasoning above) with one deliberate change:
the ViT `Session` is loaded once at startup and cached, instead of
`vit_process()`'s reference behavior of reloading the engine file from disk
on every call, which is fine for a one-shot CLI script but wasteful for a
long-running server.

Verified with a real HTTP request through `fastapi.testclient.TestClient`
(base64 `data:image/jpeg` URI in the request body, exactly how a real API
client would send one) — `/health` correctly reports `{"mode": "multimodal"}`,
and `POST /v1/chat/completions` with the demo image + "这张图片里有什么？"
returned HTTP 200 with the same correct description as the raw script test,
end-to-end latency 1254ms (base64 decode + temp file + ViT + prompt
injection + generation, all through the actual HTTP layer — not just the
underlying script).

### hlapi gotchas hit while testing generation (0.10.0 specifically)

- `tensorrt_llm.hlapi` in 0.10.0 has no top-level `SamplingParams` — it's
  `SamplingConfig(end_id=None, pad_id=None, beam_width=1, max_new_tokens=None)`,
  no `temperature` field.
- `LLM.__init__` takes a positional `ModelConfig(model_dir=...)`, not a bare
  `model=` kwarg.
- `LLM.generate()` with raw string prompts asserts on a missing tokenizer
  internally even when a tokenizer was passed to `LLM()` — this looks like a
  genuine bug in this point release. Workaround: tokenize yourself and pass
  token ID lists instead of strings.
- Running generation via `python3 -c "..."` hangs indefinitely (0% GPU
  util, CPU time stops climbing) — the hlapi executor spawns a worker
  process that can't bootstrap without a real importable `__main__` module.
  Always run generation through an actual `.py` file with
  `if __name__ == "__main__":`, never `-c`.

## Build steps (on the GPU server)

```bash
# one-time environment setup
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p ./miniconda3 && source ./miniconda3/bin/activate
conda create -n trtllm python=3.10 -y -c conda-forge --override-channels
conda activate trtllm
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install --extra-index-url https://pypi.nvidia.com tensorrt_llm==0.10.0
git clone --depth 1 --branch v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git
pip install -r TensorRT-LLM/examples/quantization/requirements.txt

# qwen_vl_trtllm_builder.py drives the generic ModelOpt quantize.py path, which
# CRASHES on Qwen-VL-Chat's checkpoint (see "Multimodal" below). Use it for
# text-only Qwen checkpoints (e.g. Qwen/Qwen-7B-Chat, ~15GB) only:
python inference/engines/qwen_vl_trtllm_builder.py \
    --hf-model-dir ./models/Qwen-7B-Chat \
    --output-root ./engine_out \
    --trtllm-examples-root ./TensorRT-LLM/examples \
    --registry-path ./registry/registry.json
# For Qwen-VL-Chat itself, follow the two-engine recipe in the Multimodal
# section (examples/qwen/convert_checkpoint.py + examples/qwenvl/vit_onnx_trt.py).

# serve it
uvicorn inference.serving.trtllm_server:app --host 0.0.0.0 --port 8082
```

Always run `--dry-run` first on a new machine/config — it validates the
exact `quantize.py` / `trtllm-build` command lines without spending GPU time
or downloading the model.

## Benchmarks and measured limits (RTX 4090, TensorRT-LLM 0.10.0 Python runtime)

All numbers: Qwen-VL text decoder, INT4 weight-only + INT8 KV cache,
8 short e-commerce Chinese prompts, `max_new_tokens=100`, token-ID inputs,
3 repeated trials per point (all three listed). Own script — TensorRT-LLM's
`benchmarks/` suite only hardcodes falcon/llama-2/gpt-j and needs a C++
binary this pip install doesn't ship.

**Single-stream latency (bs=1):** P50 ≈ 611–613 ms for 100 tokens ≈ 6.1 ms/token.
Decode at bs=1 is memory-bandwidth-bound: engine weights 5.74 GB, of which
`vocab_embedding` (~1.24 GB) is a sparse lookup and `lm_head` (~1.24 GB,
**F16 — it is in the checkpoint's `exclude_modules`, i.e. not quantized**) is
read in full every token → ~4.5 GB/token → ~4.5 ms floor at ~1008 GB/s.
Measured 6.1 ms ≈ 74% of the bandwidth floor. Consequences: further weight
quantization has < ~26% headroom (lm_head INT8 ≈ 10% of it), and the PRD's
≤400 ms target at 100 output tokens is not reachable on this card via
quantization — it is equivalent to "≤ ~65 output tokens", or needs
speculative decoding.

**Throughput vs `--max_batch_size` (QPS = batch / mean wall time):**

| engine max_bs | batch | wall time (s), 3 trials | QPS |
|---|---|---|---|
| 8 | 1 / 2 / 4 / 8 | 0.61 → 0.82 | 1.63 / 3.21 / 6.26 / 9.80 |
| 64 | 1 / 8 / 16 / 32 | 0.614 / 0.717 / 0.748 / 0.809 | 1.63 / 11.15 / 21.40 / 39.53 |
| 64 | 64 | 1.103 1.106 1.105 | **57.94** (peak) |
| 80 | 48 | 1.064 0.978 0.975 | 47.73 |
| 80 | 64 | 1.207 1.103 1.103 | 56.25 |
| 80 | 70 | 1.461 1.485 1.467 | 47.59 |
| 80 | 72 | 1.466 1.464 1.464 | 49.15 |
| 80 | 76 | 1.480 1.477 1.474 | 51.47 |
| 80 | 80 | 1.499 1.500 1.498 | 53.37 |
| 96 | — | build-time OOM (6.4 GB tactic allocation fails) | — |

Readings: (1) `--max_batch_size` is the first throughput ceiling — 8 → 64
took QPS from 9.8 to 57.94, 96.6% of the PRD's 60. (2) Per-batch wall time
is a **step function** at 64: 48→64 costs +13%, then a fixed +0.33 s appears
past 64 and stays flat through 80 (bs=72, a multiple of 8, is no better than
70, so it is a 64 boundary, not 8-alignment). **Hypothesis, not
profiler-confirmed:** the INT4 WeightOnlyQuantMatmul plugin pads to the next
tensor-core tile / switches tactic past 64. Verify with nsys or TRT's layer
profiler. (3) Because 70→80 is flat, QPS climbs linearly in that band and
extrapolates to ~60 at bs≈90 — blocked only by memory; right-sizing
`max_input_len`/`max_output_len`/`tokens_per_block` to the real length
distribution is the lever to unlock it. (4) An engine built for max_bs=80 is
~3% slower at bs=64 than one built for 64 — build for the batch you run.

**W8A8 SmoothQuant vs W4A16 (same bs64 build flags, same prompts, 3 trials
each; SQ engine needs the library patch from pitfall 6):**

| batch | W4A16 mean / QPS | W8A8 mean / QPS | W8A8 slower by |
|---|---|---|---|
| 1 | 0.614 s / 1.63 | 0.996 s / 1.00 | 1.62× |
| 8 | 0.717 s / 11.15 | 1.077 s / 7.43 | 1.50× |
| 32 | 0.809 s / 39.53 | 1.172 s / 27.31 | 1.45× |
| 64 | 1.105 s / 57.94 | 1.293 s / 49.50 | 1.17× |

Negative result for the "INT8 GEMM wins at high batch" hypothesis at every
tested batch. It is bandwidth-consistent (INT8 weights read 2× the bytes of
INT4) and the gap narrows monotonically — W8A8's wall time grows only +30%
from bs=1→64 vs W4A16's +80%, the signature of better compute efficiency
from a worse baseline; a linear extrapolation of the two slopes crosses at
bs≈120 (extrapolation, beyond what a 24 GB card can build). The benchmark
is also structurally unfair to W8A8: ~10-token prompts make prefill
negligible, and prefill is the compute-bound phase where INT8 GEMMs pay
off. Decision at the time: keep W4A16 (bs64) canonical; revisit W8A8 only with
realistic 500–1500-token prompts. **Revisited — see the next two blocks.**

**Realistic-length prompts (system + fabric knowledge + attribute JSON +
question; 500 and 1000 input tokens, 100 output, 2 trials; 256 image tokens
stood in by equal text — same decode/KV cost, prefill differs only by one
embedding lookup; `max_input_len=1024` caps this at 1000):**

| in | bs | W4A16 QPS | W8A8 QPS | prefill tok/s W4 / W8 |
|---|---|---|---|---|
| 500 | 1 | 1.50 | 0.97 | 750 / 486 |
| 500 | 8 | 6.97 | 6.10 | 3484 / 3052 |
| 500 | 32 | 11.21 | **14.80** (+32%) | 5607 / 7402 |
| 500 | 64 | 10.84 | **14.08** (+30%) | 5420 / 7042 |
| 1000 | 1 | 1.38 | 0.95 | 1375 / 946 |
| 1000 | 8 | 4.97 | 5.16 | 4971 / 5156 |
| 1000 | 32 | 5.68 | **7.20** (+27%) | 5683 / 7203 |
| 1000 | 64 | 5.61 | **8.12** (+45%) | 5613 / 8124 |

The "W8A8 wins when prefill dominates" prediction is confirmed and grows
with prompt length; W4A16's prefill saturates at ~5.5k tok/s and stops
scaling past bs=32. More importantly for the PRD: **with realistic inputs
peak QPS is 8–14, not 58** — the 60 QPS target is 4–5× away on this card.
1000 input tokens add only ~110 ms to single-request latency. Serving
choice: route by input length — W4A16 for short QA, W8A8 for RAG-augmented
long requests.

**Accuracy (FP16 reference vs W4A16 vs W8A8):** two independent metrics.
(1) TensorRT-LLM's own `examples/summarize.py`, cnn_dailymail, 20 articles:

| | rouge1 | rouge2 | rougeL |
|---|---|---|---|
| FP16 (HF) | 25.59 | 7.93 | 18.62 |
| W4A16 (RTN INT4) | 24.80 | 7.93 | 17.84 |
| W8A8 (SmoothQuant) | 28.31 | 10.72 | 20.26 |

(2) Greedy-decode agreement on 20 Chinese fashion prompts, FP16 as
reference, top_k=1 everywhere (TRT via `tensorrt_llm.runtime.ModelRunner`;
note Qwen's tokenizer sets `eod_id` not `eos_token_id`, and ModelRunner
asserts on `end_id=None`): W4A16 diverges at the first content token on
every prompt, W8A8 after 8.6 tokens on average. Both metrics agree:
W8A8 ≈ FP16 > W4A16 — RTN INT4 costs ~3–4% relative ROUGE, real but small;
"W8A8 above FP16" is n=20 noise, not a gain. Limits: English news, no chat
template, 20 samples, not a fashion task set (no reliable Fashion-VQA source
was found; nothing was downloaded from a guessed URL). Confound checked:
`generation_config.json` has no repetition_penalty and `do_sample=False`
was passed explicitly, so the HF reference is pure greedy. Root cause of the
W4A16 gap: the Qwen-VL conversion path only offers RTN, not AWQ.

**Evidence for the numbers above** is in `docs/benchmarks/2026-09-18-rtx4090/`
— reconstructed verbatim from the session's tool outputs after the pod was
stopped before its `archive/` could be fetched; its README states exactly
what was recovered (every result line, registry, build stats, versions, the
scripts as run) and what was not (`pip freeze`, the per-prompt eval JSONs,
the SQ engine's build stats). The scripts that produced them are in
`inference/benchmarks/`.

**Next for throughput (D — inflight batching, no Triton needed):** the pip
0.10.0 install exposes the C++ executor — `tensorrt_llm.bindings.executor`
has `Executor`, `ExecutorConfig`, `BatchingType.INFLIGHT`, `SchedulerConfig`,
`KvCacheConfig` — and the bs64 engine already satisfies its prerequisites
(`paged_kv_cache=True`, `remove_input_padding=True`, `max_num_tokens=65536`,
`tokens_per_block=64` default). The static-batch numbers above therefore
have a direct upgrade path: an executor-backed serving loop plus a
multi-client Poisson-arrival load generator, measuring QPS and latency
percentiles under real concurrency. ~30–45 min of GPU; not run yet.

**Caveats that make these numbers optimistic:** prompts are ~10 tokens
(prefill ≈ 0; the real PRD prompt is RAG knowledge + attribute JSON + 256
image tokens ≈ 500–1500 tokens); this is static batching through the
Python runtime, which TensorRT-LLM itself labels experimental — production
serving needs the C++ executor / Triton backend for inflight batching; and
this is a 4090 (Ada, ~660 INT8 TOPS, 1008 GB/s), not the PRD's 3090 (Ampere,
~285 TOPS, 936 GB/s) — on a 3090 expect ~7% worse at bs=1 and roughly half
the high-batch throughput, i.e. 60 QPS on a single 3090 is unlikely with this
architecture.

**INT8 fallback warnings are expected noise, not partial failure.** The
bs=64 build log has 1490 `Missing scale and zero-point ... fall back to
non-int8` lines across all 32 layers. Breakdown: layernorm-decomposition
ops 768, MLP 288, qkv 128, residual adds 130, only 4 lines matching
KV-cache-related names (not one per layer). The engine is W4A16 —
activations are FP16 by design; `--int8_kv_cache` puts only the KV cache in
INT8, and once INT8 is globally allowed TensorRT warns for every
intermediate tensor without a calibration scale. Those tensors were never
INT8 targets.

**Vision path is real, not language prior.** demo.jpeg ships with
TensorRT-LLM (possible training/eval exposure) and "woman + dog on a beach"
is guessable from text alone, so we ran three synthetic solid-color images
(RGB 168/32/120, 24/178/140, 232/150/18) asking for the background color:
answers 洋红色 / 翡翠绿色 / 橙色 — 3/3, two of them more precise than our own
labels. A language-only prior cannot do that on blank images.

**Process hygiene findings:** `tensorrt_llm.hlapi.LLM.__del__` raises on a
missing `mpi_session` attribute and the executor's worker is not reclaimed —
a finished script kept 23 GB of GPU memory allocated until killed. Working
around it with `os._exit(0)` then silently discarded all buffered stdout
(results lost). Use `python -u` + per-result `flush()` + append-to-file, and
treat this hlapi as unsuitable for a long-lived server without a process
watchdog and GPU-memory health check.

## Consuming it from the app

Point the orchestrator side at the running `trtllm_server.py` via
`configs/serving_config.yaml`:

```yaml
mllm:
  provider: real
  real_enabled: true
  endpoint: "http://<gpu-server-ip>:8082"
```

or the equivalent env vars (`MLLM_PROVIDER=real MLLM_REAL_ENABLED=true
MLLM_ENDPOINT=http://<gpu-server-ip>:8082`). Default stays `mock` — nothing
calls a real model unless both are explicitly set, matching the rest of this
codebase's provider-selection convention (see `vision_provider.py`).
