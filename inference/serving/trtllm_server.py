"""
GPU-server-only. OpenAI-compatible ``/v1/chat/completions`` wrapper around
TensorRT-LLM engines built by inference/engines/qwen_vl_trtllm_builder.py
(text-only) or TensorRT-LLM's own examples/qwenvl/ pipeline (multimodal).

This is the process ``inference.llm.mllm_client.HttpMLLMClient`` talks to.
Runs as a separate process from the CPU-side ``inference.serving.app`` —
same split as documented in inference/llm/llm_client.py: heavy model,
separate GPU, HTTP boundary in between.

Two modes, auto-detected at startup from env vars:

Text-only (``MLLM_VIT_ENGINE_DIR`` unset): loads a single engine via
``tensorrt_llm.hlapi.LLM`` — solid, stable API surface as of 0.10.0, used
for e.g. the Qwen-7B-Chat engine ``qwen_vl_trtllm_builder.py`` builds.

Multimodal (``MLLM_VIT_ENGINE_DIR`` + ``MLLM_TOKENIZER_DIR`` set): real
image-conditioned generation. This does NOT reimplement TensorRT-LLM's
Qwen-VL runtime (exact ChatML tokenization, prompt-table embedding
injection) — it imports and reuses ``QWenInfer`` / the ViT session-running
logic directly from the vendored TensorRT-LLM checkout's
``examples/qwenvl/run.py``, because that code correctly handles a lot of
Qwen-VL-specific detail (image placeholder token positions, fake-prompt-id
ranges, the exact chat template) that would be easy to get subtly wrong
rewriting from scratch without repeated GPU verification. This exact code
path (vit_process + QWenInfer.qwen_infer) was run end-to-end and verified:
ViT 56ms, text decoder 1.02s, correct image description output. See
docs/tensorrt_llm_integration.md#multimodal for the full trace.

Heavy imports (tensorrt_llm, torch, the vendored qwenvl module) happen
lazily inside functions, not at module level, so this file can still be
imported/linted/tested on a non-GPU machine.

Run on the GPU server with::

    export MLLM_TEXT_ENGINE_DIR=/workspace/trtllm_work/qwenvl_engine_bs64
    export MLLM_TOKENIZER_DIR=/workspace/trtllm_work/models/Qwen-VL-7B-Chat
    export MLLM_VIT_ENGINE_DIR=/workspace/trtllm_work/TensorRT-LLM/examples/qwenvl/plan
    export MLLM_QWENVL_EXAMPLES_DIR=/workspace/trtllm_work/TensorRT-LLM/examples/qwenvl
    uvicorn inference.serving.trtllm_server:app --host 0.0.0.0 --port 8082
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_state: Dict[str, Any] = {
    "mode": "not_loaded",       # "text_only" | "multimodal" | "not_loaded"
    "llm": None,                # text_only mode: tensorrt_llm.hlapi.LLM
    "qinfer": None,             # multimodal mode: QWenInfer instance
    "vit_session": None,        # multimodal mode: cached tensorrt_llm.runtime.Session
    "vit_stream": None,
    "preprocess": None,         # multimodal mode: Preprocss instance (image_size=448)
    "text_engine_dir": None,
}


# ── Text-only path (tensorrt_llm.hlapi) ──────────────────────────────────


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
    already-loaded Session instead of reloading the engine file per call —
    the reference script is a one-shot CLI tool where that doesn't matter;
    a long-running server needs the engine loaded once at startup."""
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

    if vit_engine_dir and text_engine_dir and tokenizer_dir and qwenvl_examples_dir:
        try:
            _load_multimodal(vit_engine_dir, text_engine_dir, tokenizer_dir, qwenvl_examples_dir)
            _state["mode"] = "multimodal"
            logger.info("Loaded multimodal engines: text=%s vit=%s", text_engine_dir, vit_engine_dir)
        except Exception:  # noqa: BLE001 — server should still start and report unhealthy
            logger.exception("Failed to load multimodal engines — falling back to not_loaded.")
            _state["mode"] = "not_loaded"
    elif text_engine_dir:
        try:
            _state["llm"] = _load_text_only_llm(text_engine_dir, tokenizer_dir)
            _state["mode"] = "text_only"
            logger.info("Loaded text-only engine from %s", text_engine_dir)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load text-only engine — /health will report not ready.")
            _state["mode"] = "not_loaded"
    else:
        logger.warning(
            "No engine configured. Set MLLM_TEXT_ENGINE_DIR (text-only) or "
            "MLLM_TEXT_ENGINE_DIR+MLLM_VIT_ENGINE_DIR+MLLM_TOKENIZER_DIR+"
            "MLLM_QWENVL_EXAMPLES_DIR (multimodal). Server starts but "
            "/v1/chat/completions will return 503."
        )
    yield
    _state["llm"] = None
    _state["qinfer"] = None
    _state["vit_session"] = None


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


def _extract_text_and_image(messages: List[ChatMessage]) -> tuple[str, Optional[str], Optional[bytes]]:
    """Flatten OpenAI-style multimodal message content into
    (prompt_text, image_url, image_bytes). Only the last message's image is used —
    this server does not support multi-turn multi-image history yet."""
    parts: List[str] = []
    image_url: Optional[str] = None
    image_bytes: Optional[bytes] = None
    for msg in messages:
        if isinstance(msg.content, str):
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
    return "\n".join(parts), image_url, image_bytes


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _state["mode"] != "not_loaded" else "engine_not_loaded",
        "mode": _state["mode"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    from fastapi import HTTPException

    mode = _state["mode"]
    if mode == "not_loaded":
        raise HTTPException(
            status_code=503,
            detail="No TensorRT-LLM engine loaded. Set MLLM_TEXT_ENGINE_DIR "
                   "(and MLLM_VIT_ENGINE_DIR/MLLM_TOKENIZER_DIR/MLLM_QWENVL_EXAMPLES_DIR "
                   "for image support) and restart.",
        )

    prompt, image_url, image_bytes = _extract_text_and_image(req.messages)
    t0 = time.perf_counter()

    if mode == "multimodal" and (image_url or image_bytes):
        text, warnings = await _generate_multimodal(prompt, image_url, image_bytes, req.max_tokens)
    else:
        warnings = []
        if image_url or image_bytes:
            warnings.append(
                "image input received but this server is running in text_only mode "
                "(no MLLM_VIT_ENGINE_DIR configured) — answering text-only."
            )
        text = _generate_text_only(prompt, req.max_tokens, req.temperature)

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "id": "chatcmpl-trtllm",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {},
        "meta": {"backend": "trtllm", "mode": mode, "latency_ms": latency_ms, "warnings": warnings},
    }


def _generate_text_only(prompt: str, max_new_tokens: int, temperature: float) -> str:
    from tensorrt_llm.hlapi import SamplingConfig

    llm = _state["llm"]
    if llm is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Text-only engine not loaded.")
    sc = SamplingConfig(max_new_tokens=max_new_tokens)
    out = list(llm.generate([prompt], sc))
    return out[0].text


async def _generate_multimodal(
    prompt: str, image_url: Optional[str], image_bytes: Optional[bytes], max_new_tokens: int
) -> tuple[str, List[str]]:
    warnings: List[str] = []
    tmp_path = None
    try:
        if image_bytes:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(image_bytes)
                tmp_path = f.name
        elif image_url:
            import requests

            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name

        image_embeds = _run_vit(tmp_path)
        qinfer = _state["qinfer"]
        text = qinfer.qwen_infer(
            image_embeds,
            [{"image": tmp_path}],
            prompt,
            max_new_tokens,
            num_beams=1,
            history=[],
        )
        return text, warnings
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
