"""
Optimized SAM-HQ segmentation path.

Drop-in replacement for ``segment_garments_samhq.py`` core loop:
    - One ``set_image()`` per image (same as original).
    - Optional batched ``predict_boxes()`` (original does one-by-one).
    - Optional FP16 autocast (original is FP32 only).

Does NOT modify tools/infer/, src/fashion_vision/, or configs/.

Usage::

    from inference.optimized.segment_garments_sam_optimized import (
        segment_garments_sam_optimized,
    )
    result = segment_garments_sam_optimized(
        image_rgb, boxes_xyxy, "checkpoints/sam_hq/sam_hq_vit_b.pth",
    )
    # result["masks"]      — np.ndarray (N, H, W)
    # result["timing"]     — dict with set_image_ms, predict_ms, total_ms
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inference.wrappers.sam_wrapper import SamHqWrapper

logger = logging.getLogger(__name__)


def segment_garments_sam_optimized(
    image_rgb: np.ndarray,
    boxes_xyxy: Union[np.ndarray, list, torch.Tensor],
    checkpoint: str,
    model_type: str = "vit_b",
    use_fp16: bool = True,
    use_batched_boxes: bool = True,
    device: str = "cuda",
    return_numpy: bool = True,
) -> Dict[str, Any]:
    """Optimized SAM-HQ segmentation — one set_image, optional batched + FP16.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image, H x W x 3, uint8.
    boxes_xyxy : shape (N, 4) or (4,)
        Box prompts in original image coordinates (xyxy).
    checkpoint : str
        Path to SAM-HQ .pth checkpoint.
    model_type : str
        ``vit_b`` (default), ``vit_l``, ``vit_h``.
    use_fp16 : bool
        Enable ``torch.amp.autocast('cuda')``.  Does NOT call ``model.half()``.
    use_batched_boxes : bool
        If True, single ``predict_boxes()`` call for all boxes.
        If False, ``predict()`` loop (one call per box).
    device : str
        Torch device.  Falls back to CPU if CUDA unavailable.
    return_numpy : bool
        Convert outputs to numpy arrays.

    Returns
    -------
    dict with keys:
        masks : np.ndarray or torch.Tensor, shape (N, H, W)
        scores : np.ndarray or torch.Tensor, shape (N,)
        logits : np.ndarray or torch.Tensor, shape (N, H, W)
        timing : dict
            set_image_ms, predict_ms, total_ms, num_boxes,
            set_image_calls (always 1), predict_calls (1 or N),
            use_fp16, use_batched_boxes
    """
    # ── Input normalisation ────────────────────────────────────────────
    boxes_np: np.ndarray = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(1, 4)
    if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError(
            f"boxes_xyxy must be (N, 4) or (4,), got shape {boxes_np.shape}"
        )
    num_boxes = boxes_np.shape[0]

    # ── Device check ───────────────────────────────────────────────────
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable — falling back to CPU")
        device = "cpu"

    # ── Load wrapper ───────────────────────────────────────────────────
    wrapper = SamHqWrapper(
        checkpoint=checkpoint,
        model_type=model_type,
        device=device,
        use_fp16=use_fp16,
    )

    # ── Empty boxes early-return ───────────────────────────────────────
    if num_boxes == 0:
        h, w = image_rgb.shape[:2]
        return {
            "masks": np.zeros((0, h, w), dtype=bool),
            "scores": np.zeros((0,), dtype=np.float32),
            "logits": np.zeros((0, h, w), dtype=np.float32),
            "timing": {
                "set_image_ms": 0.0,
                "predict_ms": 0.0,
                "total_ms": 0.0,
                "num_boxes": 0,
                "set_image_calls": 0,
                "predict_calls": 0,
                "use_fp16": use_fp16,
                "use_batched_boxes": use_batched_boxes,
            },
        }

    # ── set_image (ONCE) ──────────────────────────────────────────────
    wrapper.set_image(image_rgb)
    set_image_ms = wrapper.last_timing.get("set_image_ms", 0.0)

    # ── predict ────────────────────────────────────────────────────────
    if use_batched_boxes:
        masks, scores, logits = wrapper.predict_boxes(boxes_np)
        predict_calls = 1
    else:
        all_masks: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        for box in boxes_np:
            m, s, l = wrapper.predict(box=box)
            all_masks.append(m[0] if m.ndim == 3 else m)
            all_scores.append(s[0] if s.ndim == 1 else s)
            all_logits.append(l[0] if l.ndim == 3 else l)
        masks = np.stack(all_masks, axis=0)
        scores = np.array(all_scores, dtype=np.float32)
        logits = np.stack(all_logits, axis=0)
        predict_calls = num_boxes

    predict_ms = wrapper.last_timing.get("predict_boxes_ms", 0.0)
    total_ms = set_image_ms + predict_ms

    # ── Return ─────────────────────────────────────────────────────────
    if not return_numpy:
        import torch as _torch
        masks = _torch.as_tensor(masks, device=device)
        scores = _torch.as_tensor(scores, device=device)
        logits = _torch.as_tensor(logits, device=device)

    return {
        "masks": masks,
        "scores": scores,
        "logits": logits,
        "timing": {
            "set_image_ms": set_image_ms,
            "predict_ms": predict_ms,
            "total_ms": total_ms,
            "num_boxes": num_boxes,
            "set_image_calls": 1,
            "predict_calls": predict_calls,
            "use_fp16": use_fp16,
            "use_batched_boxes": use_batched_boxes,
        },
    }
