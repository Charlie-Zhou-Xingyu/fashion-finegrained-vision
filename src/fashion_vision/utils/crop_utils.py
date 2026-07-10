"""Canonical region crop utility for PRD 3.1.3 attribute inference.

This module provides a single, deterministic crop function used consistently
across val, test, and live inference.  Training augmentation (RandomResizedCrop)
remains in the dataset transform pipelines and is NOT replicated here.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image


def crop_region_from_image(
    image: np.ndarray,
    bbox_xyxy: Union[list, tuple],
    mask: Union[np.ndarray, None] = None,
    expand_ratio: float = 0.15,
    target_size: int = 224,
    background_fill: str = "keep",
) -> Image.Image:
    """Crop a garment region from an image and resize to a square PIL Image.

    The crop is expanded by ``expand_ratio`` on each side (as a fraction of
    ``max(width, height)`` of the original bbox), then clamped to the image
    boundary.  The result is resized with bilinear interpolation; no letterboxing
    is applied.

    The function is deterministic and does not apply any random augmentation.

    Args:
        image: RGB image as a NumPy array of shape ``(H, W, 3)``, dtype uint8.
        bbox_xyxy: Bounding box ``[x1, y1, x2, y2]`` in absolute pixel coordinates.
        mask: Optional binary mask of shape ``(H, W)`` or ``(H, W, 1)``, same
            spatial size as *image*.  Only used when *background_fill* is
            ``"zero"`` or ``"mean"``.
        expand_ratio: Fraction of ``max(bbox_w, bbox_h)`` added to each side
            before cropping.  Default 0.15 matches the multiview_v2 pipeline.
        target_size: Square output size in pixels.  Default 224.
        background_fill: How to handle pixels outside the mask.  One of:

            * ``"keep"`` — ignore mask, return raw crop (default).
            * ``"zero"`` — set background pixels to 0.
            * ``"mean"`` — replace background pixels with the per-channel mean
              of foreground pixels.

    Returns:
        PIL ``Image`` in RGB mode of size ``(target_size, target_size)``.

    Raises:
        ValueError: If *background_fill* is not one of the three valid values.
        ValueError: If *bbox_xyxy* produces a zero-area region after clamping.
        ValueError: If *mask* spatial dimensions do not match *image*.
        ValueError: If *target_size* is not positive.
        ValueError: If *image* is not a 3-channel array.
    """
    _validate_inputs(image, bbox_xyxy, mask, expand_ratio, target_size, background_fill)

    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)

    # Expand bbox
    bw = x2 - x1
    bh = y2 - y1
    pad = expand_ratio * max(bw, bh)
    x1e = max(0.0, x1 - pad)
    y1e = max(0.0, y1 - pad)
    x2e = min(float(w), x2 + pad)
    y2e = min(float(h), y2 + pad)

    # Integer pixel coordinates
    cx1 = int(round(x1e))
    cy1 = int(round(y1e))
    cx2 = int(round(x2e))
    cy2 = int(round(y2e))

    # Clamp to valid range
    cx1 = max(0, min(cx1, w - 1))
    cy1 = max(0, min(cy1, h - 1))
    cx2 = max(cx1 + 1, min(cx2, w))
    cy2 = max(cy1 + 1, min(cy2, h))

    if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
        raise ValueError(
            f"bbox_xyxy {bbox_xyxy} produced a zero-area crop after clamping "
            f"to image bounds ({w}×{h})."
        )

    crop = image[cy1:cy2, cx1:cx2].copy()

    if background_fill != "keep" and mask is not None:
        mask_2d = mask.squeeze() if mask.ndim == 3 else mask
        mask_crop = mask_2d[cy1:cy2, cx1:cx2].astype(bool)
        crop = _apply_background_fill(crop, mask_crop, background_fill)

    pil_crop = Image.fromarray(crop, mode="RGB")
    return pil_crop.resize((target_size, target_size), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_background_fill(
    crop: np.ndarray,
    mask_crop: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Apply background fill to *crop* where *mask_crop* is False.

    Args:
        crop: uint8 RGB array of shape ``(H, W, 3)``.
        mask_crop: Boolean array of shape ``(H, W)``; True = foreground.
        mode: ``"zero"`` or ``"mean"``.

    Returns:
        Modified crop array (same shape and dtype).
    """
    result = crop.copy()
    bg = ~mask_crop  # shape (H, W)

    if mode == "zero":
        result[bg] = 0
    elif mode == "mean":
        fg_pixels = crop[mask_crop]  # shape (N, 3) or empty
        if fg_pixels.size > 0:
            fill = fg_pixels.mean(axis=0).round().astype(np.uint8)
        else:
            fill = np.zeros(3, dtype=np.uint8)
        result[bg] = fill

    return result


def _validate_inputs(
    image: np.ndarray,
    bbox_xyxy: Union[list, tuple],
    mask: Union[np.ndarray, None],
    expand_ratio: float,
    target_size: int,
    background_fill: str,
) -> None:
    """Raise ValueError for invalid inputs before any processing begins."""
    valid_fills = {"keep", "zero", "mean"}
    if background_fill not in valid_fills:
        raise ValueError(
            f"background_fill must be one of {valid_fills}, got {background_fill!r}."
        )

    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be an (H, W, 3) array, got shape {image.shape}."
        )

    if len(bbox_xyxy) != 4:
        raise ValueError(
            f"bbox_xyxy must have 4 elements [x1, y1, x2, y2], got {len(bbox_xyxy)}."
        )

    x1_v, y1_v, x2_v, y2_v = (float(v) for v in bbox_xyxy)
    if x2_v <= x1_v or y2_v <= y1_v:
        raise ValueError(
            f"bbox_xyxy must have x2 > x1 and y2 > y1 (zero-area input), "
            f"got {list(bbox_xyxy)}."
        )

    if mask is not None:
        h, w = image.shape[:2]
        mask_hw = mask.squeeze() if mask.ndim == 3 else mask
        if mask_hw.shape != (h, w):
            raise ValueError(
                f"mask spatial dims {mask_hw.shape} do not match image ({h}, {w})."
            )
