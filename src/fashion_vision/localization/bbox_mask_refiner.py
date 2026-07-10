"""
Bbox-to-mask refiner for open-vocabulary region localization (3.1.2).

Converts a Grounding DINO bounding box into a fine-grained binary mask by
running SAM-HQ in box-prompt mode.  Falls back to a filled-rectangle
pseudo-mask when SAM fails or returns an empty result.

The caller owns the SamHqWrapper lifecycle; this class holds no model state.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

if TYPE_CHECKING:
    from fashion_vision.models.sam_hq_wrapper import SamHqWrapper

logger = logging.getLogger(__name__)

BboxXYXY = Union[List[int], List[float], Tuple[float, ...], np.ndarray]


class BboxMaskRefiner:
    """
    Refine a detection bounding box into a segmentation mask using SAM-HQ.

    Args:
        sam_wrapper: An already-loaded SamHqWrapper instance.
    """

    def __init__(self, sam_wrapper: "SamHqWrapper") -> None:
        self._sam = sam_wrapper

    def refine(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: BboxXYXY,
        garment_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run SAM box-prompt on ``bbox_xyxy`` and return a binary mask.

        Steps:
        1. Convert image BGR → RGB.
        2. Call ``sam_wrapper.predict_with_box(image_rgb, bbox_xyxy)``.
        3. Intersect the result with ``garment_mask`` if provided, so the
           local-part mask never bleeds outside the garment silhouette.
        4. On any failure, fill ``bbox_xyxy`` as a rectangle pseudo-mask.

        Args:
            image_bgr: Full image in BGR uint8 H×W×3 format.
            bbox_xyxy: Detection bbox [x1, y1, x2, y2] in full-image coords.
            garment_mask: Optional binary H×W mask for the parent garment
                instance.  Non-zero pixels = garment.

        Returns:
            Dict with keys:
                mask (np.ndarray | None): H×W uint8 binary mask (0/255).
                mask_source (str): "sam_box_prompt" | "bbox_fill" | "failed".
                score (float): SAM confidence score, or 0.0 for fallbacks.
        """
        bbox = [int(round(float(v))) for v in bbox_xyxy]
        if len(bbox) != 4:
            logger.error("bbox_mask_refiner: bbox must have 4 values, got %s", bbox)
            return {"mask": None, "mask_source": "failed", "score": 0.0}

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        try:
            result = self._sam.predict_with_box(image_rgb, bbox)
        except Exception as exc:
            logger.warning("bbox_mask_refiner: SAM predict_with_box failed: %s", exc)
            return self._bbox_fill(image_bgr.shape, bbox)

        raw_mask = result.get("mask")
        score = float(result.get("score", 0.0))

        if raw_mask is None or not isinstance(raw_mask, np.ndarray):
            logger.warning("bbox_mask_refiner: SAM returned no mask")
            return self._bbox_fill(image_bgr.shape, bbox)

        # Ensure 2-D uint8 binary mask.
        mask = (raw_mask > 0).astype(np.uint8) * 255
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if mask.sum() == 0:
            logger.warning("bbox_mask_refiner: SAM returned empty mask, using bbox fill")
            return self._bbox_fill(image_bgr.shape, bbox)

        # Intersect with garment silhouette to prevent bleeding.
        if garment_mask is not None:
            garment_bin = (garment_mask > 0).astype(np.uint8)
            # Squeeze H×W×1 → H×W so shapes match candidate mask.
            if garment_bin.ndim == 3 and garment_bin.shape[2] == 1:
                garment_bin = garment_bin[:, :, 0]
            if garment_bin.shape == mask.shape:
                mask = np.where(garment_bin > 0, mask, 0).astype(np.uint8)
            else:
                logger.warning(
                    "bbox_mask_refiner: garment_mask shape %s != mask shape %s, skipping intersection",
                    garment_bin.shape,
                    mask.shape,
                )

        return {"mask": mask, "mask_source": "sam_box_prompt", "score": score}

    @staticmethod
    def _bbox_fill(
        image_shape: Tuple[int, ...],
        bbox: List[int],
    ) -> Dict[str, Any]:
        """Return a filled-rectangle pseudo-mask as a graceful fallback."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        return {"mask": mask, "mask_source": "bbox_fill", "score": 0.0}


if __name__ == "__main__":
    # Smoke test: verify BboxMaskRefiner._bbox_fill without a real SAM model.
    shape = (480, 640, 3)
    bbox = [100, 80, 300, 250]
    result = BboxMaskRefiner._bbox_fill(shape, bbox)
    assert result["mask_source"] == "bbox_fill"
    assert result["mask"][80:250, 100:300].all(), "fill region should be non-zero"
    assert result["mask"][0, 0] == 0, "outside fill region should be zero"
    print("bbox_mask_refiner smoke test passed.")
