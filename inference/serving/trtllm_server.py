"""
GPU-server-only. OpenAI-compatible ``/v1/chat/completions`` wrapper around
TensorRT-LLM engines built by inference/engines/qwen_vl_trtllm_builder.py
(text-only) or TensorRT-LLM's own examples/qwenvl/ pipeline (multimodal).

This is the process ``inference.llm.mllm_client.HttpMLLMClient`` talks to.
Runs as a separate process from the CPU-side ``inference.serving.app`` —
same split as documented in inference/llm/llm_client.py: heavy model,
separate GPU, HTTP boundary in between.

Backends (``MLLM_BACKEND``):

``executor`` (default) — the C++ ``tensorrt_llm.bindings.executor`` with
inflight batching. Concurrent HTTP requests are enqueued individually and
scheduled per iteration by the runtime; a background thread drains
``await_responses`` and resolves per-request asyncio futures. This is the
path measured in docs/tensorrt_llm_integration.md §D (QPS saturates at
``max_batch_size``; 1.28× over static batching on heterogeneous lengths).
Optional ``MLLM_KV_BLOCK_REUSE=1`` (prefix caching — the PRD prompt shares
system + RAG knowledge across requests) and ``MLLM_CHUNKED_CONTEXT=1``
(needs an engine built with ``--use_paged_context_fmha enable``).

``hlapi`` — ``tensorrt_llm.hlapi.LLM``: one request at a time, kept for
comparison. Known 0.10.0 issue: its worker leaks GPU memory on shutdown.

Multimodal (``MLLM_VIT_ENGINE_DIR`` + ``MLLM_TOKENIZER_DIR`` +
``MLLM_QWENVL_EXAMPLES_DIR`` set): image requests go through the vendored
``examples/qwenvl/run.py`` ``QWenInfer`` (ChatML with image placeholders,
prompt-table embedding injection — verified end-to-end: ViT 56 ms, decoder
1.02 s, correct description). That path is serialized (one image request at
a time); text-only requests still use the executor. Routing image requests
through the executor (``Request.prompt_tuning_config``) is the next step.

Request hardening (all env-tunable, all reported in ``/health``):
  MLLM_MAX_NEW_TOKENS   clamp on max_tokens           (default: engine max_output_len, ≤256)
  MLLM_MAX_INPUT_TOKENS reject 413 above this          (default: engine max_input_len)
  MLLM_MAX_IMAGE_BYTES  reject 413 above this          (default 5 MiB)
  MLLM_MAX_QUEUE        reject 429 when this many requests are already in flight
                        (default 4 × max_batch_size)
  MLLM_REQUEST_TIMEOUT_S 504 + executor.cancel_request (default 30 s)

Run on the GPU server with::

    export MLLM_TEXT_ENGINE_DIR=/workspace/trtllm_work/qwenvl_engine_bs96_pcf
    export MLLM_TOKENIZER_DIR=/workspace/trtllm_work/models/Qwen-VL-7B-Chat
    export MLLM_KV_BLOCK_REUSE=1 MLLM_CHUNKED_CONTEXT=1
    # optional image support:
    export MLLM_VIT_ENGINE_DIR=/workspace/trtllm_work/TensorRT-LLM/examples/qwenvl/plan
    export MLLM_QWENVL_EXAMPLES_DIR=/workspace/trtllm_work/TensorRT-LLM/examples/qwenvl
    uvicorn inference.serving.trtllm_server:app --host 0.0.0.0 --port 8082
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import logging
import os
import statistics
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS_CAP = 256
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_state: Dict[str, Any] = {
    "mode": "not_loaded",       # "text_only" | "multimodal" | "not_loaded"
    "backend": None,            # "executor" | "hlapi"
    "llm": None,                # hlapi backend: tensorrt_llm.hlapi.LLM
    "executor": None,           # executor backend: ExecutorBackend
    "qinfer": None,             # multimodal: QWenInfer instance
    "vit_session": None,
    "vit_stream": None,
    "preprocess": None,
    "text_engine_dir": None,
    "engine_meta": {},
    "limits": {},
    "mm_lock": None,            # asyncio.Lock serialising QWenInfer calls
}


# ── Limits / pure helpers (unit-tested without a GPU) ────────────────────────


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def read_engine_meta(engine_dir: str) -> Dict[str, Any]:
    """max_batch_size / max_input_len / max_output_len / max_num_tokens / quant from the engine's config.json."""
    try:
        with open(os.path.join(engine_dir, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return {}
    bc = cfg.get("build_config", {}) or {}
    pc = (cfg.get("pretrained_config", {}) or {}).get("quantization", {}) or {}
    return {
        "max_batch_size": bc.get("max_batch_size"),
        "max_input_len": bc.get("max_input_len"),
        "max_output_len": bc.get("max_output_len"),
        "max_num_tokens": bc.get("max_num_tokens"),
        "paged_context_fmha": (bc.get("plugin_config", {}) or {}).get("use_paged_context_fmha"),
        "quant_algo": pc.get("quant_algo"),
        "kv_cache_quant_algo": pc.get("kv_cache_quant_algo"),
    }


def resolve_limits(engine_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Env overrides on top of engine-derived defaults. Never returns a limit above what the engine can do."""
    eng_out = int(engine_meta.get("max_output_len") or DEFAULT_MAX_NEW_TOKENS_CAP)
    eng_in = int(engine_meta.get("max_input_len") or 1024)
    eng_bs = int(engine_meta.get("max_batch_size") or 8)
    max_new = min(_env_int("MLLM_MAX_NEW_TOKENS", min(DEFAULT_MAX_NEW_TOKENS_CAP, eng_out)), eng_out)
    max_in = min(_env_int("MLLM_MAX_INPUT_TOKENS", eng_in), eng_in)
    return {
        "max_new_tokens": max(1, max_new),
        "max_input_tokens": max(1, max_in),
        "max_image_bytes": max(1, _env_int("MLLM_MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES)),
        "max_queue": max(1, _env_int("MLLM_MAX_QUEUE", 4 * eng_bs)),
        "request_timeout_s": max(0.1, _env_float("MLLM_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S)),
    }


def clamp_max_tokens(requested: Any, cap: int) -> Tuple[int, Optional[str]]:
    """Clamp the client's max_tokens into [1, cap]; returns (value, warning|None)."""
    try:
        val = int(requested)
    except (TypeError, ValueError):
        return cap, f"max_tokens={requested!r} not an int; using {cap}"
    if val < 1:
        return 1, f"max_tokens={val} < 1; using 1"
    if val > cap:
        return cap, f"max_tokens={val} clamped to server cap {cap}"
    return val, None


def build_chatml(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Qwen(-VL)-Chat's ChatML template for a single-turn text request."""
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")


def _extract_text_and_image(messages: List["ChatMessage"]) -> Tuple[str, Optional[str], Optional[str], Optional[bytes]]:
    """Flatten OpenAI-style content into (system_prompt|None, user_text, image_url, image_bytes).
    Only the last message's image is used — no multi-turn multi-image history yet."""
    parts: List[str] = []
    system_prompt: Optional[str] = None
    image_url: Optional[str] = None
    image_bytes: Optional[bytes] = None
    for msg in messages:
        if isinstance(msg.content, str):
            if msg.role == "system" and system_prompt is None:
                system_prompt = msg.content
            else:
                parts.append(msg.content if msg.role == "user" else f"{msg.role}: {msg.content}")
            continue
        for block in msg.content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    _, _, b64 = url.partition(",")
                    image_bytes = base64.b64decode(b64)
                else:
                    image_url = url
    return system_prompt, "\n".join(parts), image_url, image_bytes


class LatencyWindow:
    """Ring buffer of recent latencies for /health (p50/p95 over the last N)."""

    def __init__(self, n: int = 512) -> None:
        self._d: Deque[float] = deque(maxlen=n)
        self.completed = 0
        self.rejected: Dict[str, int] = {"413": 0, "429": 0, "504": 0, "500": 0}

    def add(self, seconds: float) -> None:
        self._d.append(seconds)
        self.completed += 1

    def summary(self) -> Dict[str, Any]:
        vals = sorted(self._d)
        if not vals:
            return {"completed": self.completed, "rejected": dict(self.rejected), "n": 0}
        k95 = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
        return {"completed": self.completed, "rejected": dict(self.rejected), "n": len(vals),
                "p50_ms": round(statistics.median(vals) * 1000, 1), "p95_ms": round(vals[k95] * 1000, 1)}


_latency = LatencyWindow()


# ── Executor backend ──────────────────────────────────────────────────────────


class ExecutorBackend:
    """Owns one ``tensorrt_llm.bindings.executor.Executor`` and a response-drain thread.

    ``submit()`` is called from the asyncio loop: it enqueues the request and returns
    a future; the drain thread resolves the future via ``loop.call_soon_threadsafe``.
    """

    def __init__(self, engine_dir: str, tokenizer_dir: str, *, block_reuse: bool, chunked: bool) -> None:
        import tensorrt_llm.bindings.executor as ex
        from transformers import AutoTokenizer

        self.ex = ex
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        self.end_id = getattr(self.tok, "im_end_id", None) or getattr(self.tok, "eod_id", None) or self.tok.eos_token_id
        self.pad_id = getattr(self.tok, "eod_id", None) or self.end_id
        kv = ex.KvCacheConfig(enable_block_reuse=block_reuse)
        cfg = ex.ExecutorConfig(max_beam_width=1, kv_cache_config=kv, enable_chunked_context=chunked,
                                batching_type=ex.BatchingType.INFLIGHT)
        self.executor = ex.Executor(engine_dir, ex.ModelType.DECODER_ONLY, cfg)
        self.block_reuse, self.chunked = block_reuse, chunked
        self._pending: Dict[int, Tuple[asyncio.AbstractEventLoop, asyncio.Future, List[int]]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="trtllm-executor-drain", daemon=True)
        self._thread.start()

    # -- tokenisation ---------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        try:
            return list(self.tok.encode(text, allowed_special="all"))
        except TypeError:
            return list(self.tok.encode(text))

    def decode(self, ids: List[int]) -> str:
        cut = ids.index(self.end_id) if self.end_id in ids else len(ids)
        return self.tok.decode(ids[:cut], skip_special_tokens=True)

    # -- request lifecycle ----------------------------------------------------

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._pending)

    def submit(self, loop: asyncio.AbstractEventLoop, input_ids: List[int], max_new_tokens: int,
               temperature: float, min_tokens: Optional[int] = None) -> Tuple[int, asyncio.Future]:
        ex = self.ex
        greedy = temperature is None or temperature <= 0.0
        kw: Dict[str, Any] = {"beam_width": 1}
        if min_tokens:
            kw["min_length"] = int(min(min_tokens, max_new_tokens))
        sc = ex.SamplingConfig(top_k=1, **kw) if greedy else \
            ex.SamplingConfig(temperature=float(temperature), top_p=0.8, **kw)
        req = ex.Request(input_token_ids=input_ids, max_new_tokens=max_new_tokens, streaming=False,
                         sampling_config=sc, output_config=ex.OutputConfig(exclude_input_from_output=True),
                         end_id=self.end_id, pad_id=self.pad_id)
        fut: asyncio.Future = loop.create_future()
        with self._lock:                       # register before enqueue: the drain thread may be faster than us
            rid = self.executor.enqueue_request(req)
            self._pending[rid] = (loop, fut, [])
        return rid, fut

    def cancel(self, rid: int) -> None:
        with self._lock:
            self._pending.pop(rid, None)
        try:
            self.executor.cancel_request(rid)
        except Exception:  # noqa: BLE001 — already finished
            pass

    def _drain(self) -> None:
        wait = datetime.timedelta(milliseconds=200)
        while not self._stop.is_set():
            try:
                responses = self.executor.await_responses(wait)
            except Exception:  # noqa: BLE001
                if self._stop.is_set():
                    return
                logger.exception("executor.await_responses failed")
                time.sleep(0.05)
                continue
            for r in responses:
                with self._lock:
                    entry = self._pending.get(r.request_id)
                if entry is None:
                    continue                    # cancelled / timed out
                loop, fut, buf = entry
                if r.has_error():
                    with self._lock:
                        self._pending.pop(r.request_id, None)
                    msg = getattr(r, "error_msg", None)          # 0.10: attribute, not get_error_msg()
                    msg = msg() if callable(msg) else (msg or "executor error")
                    loop.call_soon_threadsafe(self._resolve, fut, RuntimeError(str(msg)))
                    continue
                buf.extend(r.result.output_token_ids[0])
                if r.result.is_final:
                    with self._lock:
                        self._pending.pop(r.request_id, None)
                    loop.call_soon_threadsafe(self._resolve, fut, list(buf))

    @staticmethod
    def _resolve(fut: asyncio.Future, value: Any) -> None:
        if fut.done():
            return
        if isinstance(value, Exception):
            fut.set_exception(value)
        else:
            fut.set_result(value)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.executor.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ── hlapi backend (comparison only) ───────────────────────────────────────────


def _load_text_only_llm(engine_dir: str, tokenizer_dir: Optional[str]):
    from tensorrt_llm.hlapi import LLM, ModelConfig

    cfg = ModelConfig(model_dir=engine_dir)
    kwargs: Dict[str, Any] = {}
    if tokenizer_dir:
        kwargs["tokenizer"] = tokenizer_dir
    return LLM(cfg, **kwargs)


# ── Multimodal path (reuses examples/qwenvl/run.py, not reimplemented) ──────


def _load_multimodal(vit_engine_dir: str, text_engine_dir: str, tokenizer_dir: str,
                     qwenvl_examples_dir: str) -> None:
    """Load both engines once at startup. Mutates ``_state`` in place.

    Imports QWenInfer and Preprocss straight from the vendored TensorRT-LLM
    checkout rather than vendoring copies into this repo, so a `git pull` /
    version bump of that checkout is the only place those classes need to
    stay in sync.
    """
    import sys

    import torch
    from tensorrt_llm.runtime import Session

    if qwenvl_examples_dir not in sys.path:
        sys.path.insert(0, qwenvl_examples_dir)
    from run import QWenInfer  # type: ignore[import-not-found]  # vendored, not a repo module
    from vit_onnx_trt import Preprocss  # type: ignore[import-not-found]

    vit_engine_path = os.path.join(vit_engine_dir, "visual_encoder", "visual_encoder_fp16.plan")
    with open(vit_engine_path, "rb") as f:
        engine_buffer = f.read()
    _state["vit_session"] = Session.from_serialized_engine(engine_buffer)
    _state["vit_stream"] = torch.cuda.current_stream().cuda_stream
    _state["preprocess"] = Preprocss(448)  # matches examples/qwenvl/vit_onnx_trt.py's fixed size

    qinfer = QWenInfer(tokenizer_dir, text_engine_dir, "info", None, None, num_beams=1)
    qinfer.qwen_model_init()
    _state["qinfer"] = qinfer
    _state["text_engine_dir"] = text_engine_dir


def _run_vit(image_path: str):
    """Mirrors examples/qwenvl/run.py's vit_process(), but against an
    already-loaded Session instead of reloading the engine file per call."""
    import tensorrt as trt
    import torch
    from tensorrt_llm.runtime import TensorInfo

    def _trt_dtype_to_torch(dtype):
        if dtype == trt.float16:
            return torch.float16
        if dtype == trt.float32:
            return torch.float32
        if dtype == trt.int32:
            return torch.int32
        raise TypeError(f"{dtype} is not supported")

    device = torch.device("cuda")
    image = _state["preprocess"].encode([image_path]).to(device)
    images = image.contiguous()
    session = _state["vit_session"]
    visual_inputs = {"input": images.float()}
    visual_output_info = session.infer_shapes([TensorInfo("input", trt.DataType.FLOAT, images.shape)])
    visual_outputs = {
        t.name: torch.empty(tuple(t.shape), dtype=_trt_dtype_to_torch(t.dtype), device="cuda")
        for t in visual_output_info
    }
    ok = session.run(visual_inputs, visual_outputs, _state["vit_stream"])
    torch.cuda.synchronize()
    if not ok:
        raise RuntimeError("ViT TensorRT session run failed")
    return visual_outputs["output"]


# ── Startup ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    vit_engine_dir = os.getenv("MLLM_VIT_ENGINE_DIR")
    text_engine_dir = os.getenv("MLLM_TEXT_ENGINE_DIR")
    tokenizer_dir = os.getenv("MLLM_TOKENIZER_DIR")
    qwenvl_examples_dir = os.getenv("MLLM_QWENVL_EXAMPLES_DIR")
    backend = (os.getenv("MLLM_BACKEND") or "executor").strip().lower()
    _state["mm_lock"] = asyncio.Lock()

    if text_engine_dir:
        _state["engine_meta"] = read_engine_meta(text_engine_dir)
        _state["limits"] = resolve_limits(_state["engine_meta"])
        try:
            if backend == "executor":
                if not tokenizer_dir:
                    raise RuntimeError("MLLM_BACKEND=executor needs MLLM_TOKENIZER_DIR")
                _state["executor"] = ExecutorBackend(
                    text_engine_dir, tokenizer_dir,
                    block_reuse=_env_bool("MLLM_KV_BLOCK_REUSE"), chunked=_env_bool("MLLM_CHUNKED_CONTEXT"))
            else:
                _state["llm"] = _load_text_only_llm(text_engine_dir, tokenizer_dir)
            _state["backend"] = backend
            _state["mode"] = "text_only"
            logger.info("Loaded text engine %s via %s (limits=%s)", text_engine_dir, backend, _state["limits"])
        except Exception:  # noqa: BLE001 — server should still start and report unhealthy
            logger.exception("Failed to load text engine — /health will report not ready.")
            _state["mode"] = "not_loaded"

    if _state["mode"] != "not_loaded" and vit_engine_dir and tokenizer_dir and qwenvl_examples_dir:
        try:
            _load_multimodal(vit_engine_dir, text_engine_dir, tokenizer_dir, qwenvl_examples_dir)
            _state["mode"] = "multimodal"
            logger.info("Loaded multimodal engines: text=%s vit=%s", text_engine_dir, vit_engine_dir)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load ViT / QWenInfer — continuing text-only.")

    if _state["mode"] == "not_loaded":
        logger.warning("No engine configured/loaded. /v1/chat/completions will return 503.")
    yield
    if _state["executor"] is not None:
        _state["executor"].shutdown()
    _state.update({"llm": None, "executor": None, "qinfer": None, "vit_session": None})


app = FastAPI(title="Qwen-VL TensorRT-LLM Server", lifespan=lifespan)


# ── OpenAI-compatible schema (minimal subset actually used by mllm_client) ──


class ChatMessage(BaseModel):
    role: str
    content: Any  # str, or list[{"type": "text"|"image_url", ...}]


class ChatCompletionRequest(BaseModel):
    model: str = "qwen-vl-7b-chat"
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.2
    min_tokens: Optional[int] = None   # non-OpenAI; benchmark aid — forces at least this many new tokens


@app.get("/health")
async def health() -> Dict[str, Any]:
    exe = _state["executor"]
    return {
        "status": "ok" if _state["mode"] != "not_loaded" else "engine_not_loaded",
        "mode": _state["mode"],
        "backend": _state["backend"],
        "engine": _state["engine_meta"],
        "limits": _state["limits"],
        "executor": None if exe is None else {"in_flight": exe.in_flight, "block_reuse": exe.block_reuse,
                                              "chunked_context": exe.chunked},
        "latency": _latency.summary(),
    }


def _reject(code: int, detail: str):
    from fastapi import HTTPException
    _latency.rejected[str(code)] = _latency.rejected.get(str(code), 0) + 1
    raise HTTPException(status_code=code, detail=detail)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    mode = _state["mode"]
    if mode == "not_loaded":
        _reject(503, "No TensorRT-LLM engine loaded. Set MLLM_TEXT_ENGINE_DIR (+MLLM_TOKENIZER_DIR for "
                     "the executor backend; +MLLM_VIT_ENGINE_DIR/MLLM_QWENVL_EXAMPLES_DIR for images).")
    limits = _state["limits"]
    system_prompt, prompt, image_url, image_bytes = _extract_text_and_image(req.messages)
    if image_bytes is not None and len(image_bytes) > limits["max_image_bytes"]:
        _reject(413, f"image is {len(image_bytes)} bytes; limit {limits['max_image_bytes']}")
    max_new, warn = clamp_max_tokens(req.max_tokens, limits["max_new_tokens"])
    warnings: List[str] = [warn] if warn else []
    t0 = time.perf_counter()

    if mode == "multimodal" and (image_url or image_bytes):
        text, mm_warnings = await _generate_multimodal(prompt, image_url, image_bytes, max_new)
        warnings += mm_warnings
        finish = "stop"
    else:
        if image_url or image_bytes:
            warnings.append("image input received but no ViT engine is loaded — answering text-only.")
        if _state["backend"] == "executor":
            text, finish = await _generate_executor(prompt, system_prompt, max_new, req.temperature, limits,
                                                    min_tokens=req.min_tokens)
        else:
            text, finish = _generate_hlapi(prompt, max_new, req.temperature), "stop"

    latency = time.perf_counter() - t0
    _latency.add(latency)
    return {
        "id": "chatcmpl-trtllm",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {},
        "meta": {"backend": _state["backend"], "mode": mode, "latency_ms": latency * 1000,
                 "max_new_tokens": max_new, "warnings": warnings},
    }


async def _generate_executor(prompt: str, system_prompt: Optional[str], max_new: int, temperature: float,
                             limits: Dict[str, Any], min_tokens: Optional[int] = None) -> Tuple[str, str]:
    exe: ExecutorBackend = _state["executor"]
    if exe.in_flight >= limits["max_queue"]:
        _reject(429, f"{exe.in_flight} requests in flight; limit {limits['max_queue']}")
    ids = exe.encode(build_chatml(prompt, system_prompt or DEFAULT_SYSTEM_PROMPT))
    if len(ids) > limits["max_input_tokens"]:
        _reject(413, f"prompt is {len(ids)} tokens; limit {limits['max_input_tokens']}")
    loop = asyncio.get_running_loop()
    rid, fut = exe.submit(loop, ids, max_new, temperature, min_tokens=min_tokens)
    try:
        out_ids = await asyncio.wait_for(fut, timeout=limits["request_timeout_s"])
    except asyncio.TimeoutError:
        exe.cancel(rid)
        _reject(504, f"generation exceeded {limits['request_timeout_s']} s")
    except RuntimeError as exc:
        _reject(500, f"executor error: {exc}")
    finish = "length" if len(out_ids) >= max_new and exe.end_id not in out_ids else "stop"
    return exe.decode(out_ids), finish


def _generate_hlapi(prompt: str, max_new_tokens: int, temperature: float) -> str:
    from tensorrt_llm.hlapi import SamplingConfig

    llm = _state["llm"]
    if llm is None:
        _reject(503, "hlapi engine not loaded.")
    sc = SamplingConfig(max_new_tokens=max_new_tokens)
    out = list(llm.generate([prompt], sc))
    return out[0].text


async def _generate_multimodal(
    prompt: str, image_url: Optional[str], image_bytes: Optional[bytes], max_new_tokens: int
) -> Tuple[str, List[str]]:
    warnings: List[str] = ["multimodal path is serialized (QWenInfer); text-only requests use the executor"]
    tmp_path = None
    try:
        if image_bytes:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(image_bytes)
                tmp_path = f.name
        elif image_url:
            import requests

            resp = requests.get(image_url, timeout=10, stream=True)
            resp.raise_for_status()
            data = resp.raw.read(_state["limits"]["max_image_bytes"] + 1)
            if len(data) > _state["limits"]["max_image_bytes"]:
                _reject(413, "downloaded image exceeds MLLM_MAX_IMAGE_BYTES")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(data)
                tmp_path = f.name

        async with _state["mm_lock"]:
            loop = asyncio.get_running_loop()
            image_embeds = await loop.run_in_executor(None, _run_vit, tmp_path)
            qinfer = _state["qinfer"]
            text = await loop.run_in_executor(
                None, lambda: qinfer.qwen_infer(image_embeds, [{"image": tmp_path}], prompt, max_new_tokens,
                                                num_beams=1, history=[]))
        return text, warnings
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
