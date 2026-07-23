"""
SAM-HQ wrapper with FP16 autocast and batched box prediction.

    from inference.wrappers.sam_wrapper import SamHqWrapper

    wrapper = SamHqWrapper("checkpoints/sam_hq/sam_hq_vit_b.pth", use_fp16=True)
    wrapper.set_image(image_rgb)

    # single box
    masks, scores, logits = wrapper.predict_boxes(box_xyxy)

    # batched boxes
    masks, scores, logits = wrapper.predict_boxes(boxes_xyxy)

Does NOT modify tools/infer/, src/fashion_vision/, or configs/.
"""

from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SamHqWrapper:
    """SAM-HQ ViT-B predictor with FP16 autocast + batched box prediction.

    Parameters
    ----------
    checkpoint : str
        Path to .pth checkpoint.
    model_type : str
        SAM model type (``vit_b``, ``vit_l``, ``vit_h``).
    device : str
        Torch device string.
    use_fp16 : bool
        Enable ``torch.amp.autocast('cuda')`` for set_image / predict_boxes.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda",
        use_fp16: bool = True,
    ) -> None:
        self._checkpoint = str(checkpoint)
        self._model_type = model_type
        self._device = device
        self.use_fp16 = use_fp16 and device == "cuda"

        self._sam: Optional[torch.nn.Module] = None
        self._predictor = None
        self._loaded = False
        self._image_is_set = False

        # ── Optional: last-timing record ───────────────────────────────
        self.last_timing: Dict[str, float] = {}

    # ── Lazy load ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        third_party = str(_PROJECT_ROOT / "third_party" / "sam-hq")
        if third_party not in sys.path:
            sys.path.insert(0, third_party)
        from segment_anything import sam_model_registry, SamPredictor

        self._sam = sam_model_registry[self._model_type](
            checkpoint=self._checkpoint,
        )
        self._sam.to(device=self._device)
        self._sam.eval()
        self._predictor = SamPredictor(self._sam)
        self._loaded = True

    # ── Autocast context ───────────────────────────────────────────────

    def _autocast_context(self):
        """Return autocast or nullcontext, compatible with old/new PyTorch."""
        if not self.use_fp16:
            return nullcontext()
        try:
            # New API (PyTorch >= 2.0)
            return torch.amp.autocast("cuda")
        except (TypeError, AttributeError):
            # Old API fallback
            return torch.cuda.amp.autocast()  # type: ignore[attr-defined]

    # ── Public API ─────────────────────────────────────────────────────

    def set_image(self, image: np.ndarray) -> None:
        """Encode image with the SAM image encoder.

        Parameters
        ----------
        image : np.ndarray
            RGB image in H×W×3 uint8 format.
        """
        self._ensure_loaded()
        self._image_is_set = False
        with self._autocast_context(), torch.inference_mode():
            t0 = time.perf_counter()
            self._predictor.set_image(image)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        self._image_is_set = True
        self.last_timing["set_image_ms"] = (t1 - t0) * 1000

    def predict_boxes(
        self,
        boxes: Union[np.ndarray, list, torch.Tensor],
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks from box prompts (supports batched input).

        Parameters
        ----------
        boxes : shape (N, 4) or (4,)
            xyxy box coordinates in original image space.
        multimask_output : bool
            If True, returns 3 masks per box.

        Returns
        -------
        masks : np.ndarray, shape (N, H, W) if multimask_output=False
            Binary masks as numpy bool arrays.
        scores : np.ndarray, shape (N,)
            IoU prediction scores.
        logits : np.ndarray, shape (N, H, W)
            Low-resolution logits (before sigmoid).
        """
        if not self._image_is_set:
            raise RuntimeError(
                "set_image() must be called before predict_boxes()"
            )

        # Normalise input.
        boxes_np = np.asarray(boxes, dtype=np.float32)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np.reshape(1, 4)
        if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
            raise ValueError(
                f"boxes must be (N, 4) or (4,), got {boxes_np.shape}"
            )

        with self._autocast_context(), torch.inference_mode():
            t0 = time.perf_counter()

            # Transform boxes from original-image coords to model coords.
            boxes_torch = torch.as_tensor(
                boxes_np, dtype=torch.float32, device=self._device,
            )
            transformed = self._predictor.transform.apply_boxes_torch(
                boxes_torch, self._predictor.original_size,
            )

            masks_t, scores_t, logits_t = self._predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=multimask_output,
            )

            torch.cuda.synchronize()
            t1 = time.perf_counter()

        self.last_timing["predict_boxes_ms"] = (t1 - t0) * 1000
        # ponytail: also expose as generic predict_ms for compat
        self.last_timing["predict_ms"] = self.last_timing["predict_boxes_ms"]

        # Convert to numpy.
        masks_np = masks_t.cpu().numpy().astype(bool)
        scores_np = scores_t.cpu().numpy().astype(np.float32)
        logits_np = logits_t.cpu().numpy().astype(np.float32)

        return masks_np, scores_np, logits_np

    # ── Backward-compatible alias ──────────────────────────────────────

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> tuple:
        """Legacy single-box predict (compatible with SamPredictor).

        Delegates to ``predict_boxes()`` when only *box* is given.
        Otherwise falls through to the underlying predictor.
        """
        if point_coords is None and point_labels is None and box is not None:
            masks, scores, logits = self.predict_boxes(
                box, multimask_output=multimask_output,
            )
            if return_logits:
                return masks, scores, logits
            return masks, scores, logits

        # Fallback to raw predictor for complex prompt combinations.
        self._ensure_loaded()
        with self._autocast_context(), torch.inference_mode():
            return self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def model(self) -> torch.nn.Module:
        self._ensure_loaded()
        return self._sam

    @property
    def predictor(self):
        self._ensure_loaded()
        return self._predictor

    @property
    def device_str(self) -> str:
        return self._device


# ── Helper ────────────────────────────────────────────────────────────

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute binary mask IoU."""
    a_bool = np.asarray(a).astype(bool)
    b_bool = np.asarray(b).astype(bool)
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(intersection / union) if union > 0 else 1.0


def stats(values: list) -> dict:
    """Return mean / p50 / p95 / min / max for a list of floats."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
