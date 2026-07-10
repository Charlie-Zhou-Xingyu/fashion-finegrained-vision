"""
Grounding DINO backend for open-vocabulary region localization (3.1.2 Phase 2).

Uses HuggingFace transformers — no mmcv required.
Model: IDEA-Research/grounding-dino-tiny (default).

Lifecycle: instantiate once and reuse across queries; the caller (router) owns
the model instance.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch


# ── Class-aware IoU thresholds for multi-instance parts ──────────────────────
# Lower thresholds allow adjacent detections (two pockets, button rows) to all
# survive NMS instead of having all-but-one suppressed.
MULTI_INSTANCE_IOU: dict[str, float] = {
    "pocket": 0.30,
    "button": 0.20,
    "rivet": 0.15,
    "bead": 0.15,
    "sequin": 0.10,
    "buckle": 0.35,
    "epaulette": 0.40,   # two epaulettes (left + right shoulder)
    "bow": 0.30,
    "flower": 0.25,
    "tassel": 0.20,
    "applique": 0.30,
    "fringe": 0.35,
    "ribbon": 0.30,
    "stud": 0.15,
}


def _box_iou(a: list, b: list) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class GroundingDINOLocator:
    """Detect garment sub-regions with a text query via Grounding DINO."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
    ) -> None:
        """
        Load processor and model from HuggingFace hub.

        Args:
            model_id: HF model repo ID.
            device: "cuda" or "cpu".
        """
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self._device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = GroundingDinoForObjectDetection.from_pretrained(model_id)
        self._model.to(self._device).eval()

    def detect(
        self,
        image: np.ndarray,
        text_query: str,
        garment_mask: Optional[np.ndarray] = None,
        threshold: float = 0.3,
        min_bbox_area_ratio: float = 0.003,
        fill_mode: str = "grey",
        dilation_px: int = 0,
    ) -> list[dict]:
        """
        Detect sub-regions matching text_query within the image.

        Args:
            image: BGR uint8 H×W×3 numpy array (OpenCV format).
            text_query: English noun phrase, e.g. "clothing zipper", "coat pocket".
            garment_mask: Optional binary H×W mask. Non-garment pixels are filled
                according to ``fill_mode`` so DINO focuses on the garment.
                Must already match ``image`` spatial dimensions; resize before
                calling if needed.
            threshold: Minimum confidence score for returned detections.
            min_bbox_area_ratio: Drop detections whose bbox area is smaller than
                this fraction of the image area. Filters tiny jewellery / noise.
            fill_mode: Background fill mode for non-garment pixels.
                ``"grey"`` (128) / ``"black"`` (0) / ``"white"`` (255).
                Default ``"grey"``.
            dilation_px: Dilate the garment mask by this many pixels before
                applying it. Use for small parts near garment boundaries (e.g.
                buttons near seams). 0 = no dilation.

        Returns:
            List of dicts sorted descending by score::

                [{"bbox_xyxy": [x1, y1, x2, y2], "score": float, "label": str}, ...]

            Empty list when nothing exceeds threshold.
        """
        if garment_mask is not None:
            image = self.mask_to_garment(image, garment_mask, fill_mode=fill_mode, dilation_px=dilation_px)

        # ponytail: GDINO requires trailing period for reliable scoring
        if not text_query.rstrip().endswith("."):
            text_query = text_query.rstrip() + "."

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        min_area = min_bbox_area_ratio * H * W

        inputs = self._processor(
            images=image_rgb,
            text=text_query,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            target_sizes=[(H, W)],
        )[0]

        detections = []
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            detections.append({
                "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)],
                "score": float(score),
                "label": str(label),
            })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    def detect_multi_prompt(
        self,
        image: np.ndarray,
        prompts: list[str],
        garment_mask: Optional[np.ndarray] = None,
        threshold: float = 0.3,
        min_bbox_area_ratio: float = 0.003,
        nms_iou_threshold: float = 0.5,
        return_raw_count: bool = False,
        fill_mode: str = "grey",
        dilation_px: int = 0,
        nms_mode: str = "greedy",
        part: Optional[str] = None,
    ) -> "list[dict] | tuple[list[dict], int]":
        """
        Run detect() for each prompt phrase, merge results, deduplicate by NMS.

        Use this when multiple phrasings describe the same part
        (e.g. ["zipper", "clothing zipper", "zip"]) to improve recall without
        raising the score threshold.

        Args:
            prompts: Ordered list of text prompts to try.
            nms_iou_threshold: IoU above which two boxes are considered duplicates;
                the lower-scoring one is dropped.  Overridden per-part when *part*
                is in ``MULTI_INSTANCE_IOU``.
            return_raw_count: If True, return ``(detections, raw_count)`` where
                ``raw_count`` is the total number of detections across all prompts
                before NMS.  Default False preserves the original return type.
            fill_mode: Background fill mode passed to each ``detect()`` call.
            dilation_px: Mask dilation radius in pixels passed to each ``detect()`` call.
            nms_mode: ``"greedy"`` (default, legacy) or ``"soft"`` (score decay
                instead of removal — preserves adjacent detections).
            part: Canonical part name.  When provided, class-aware IoU thresholds
                from ``MULTI_INSTANCE_IOU`` override *nms_iou_threshold*.

        Returns:
            NMS-filtered detections sorted descending by score.
            Each dict carries an extra "prompt" key recording which phrase matched.
            If ``return_raw_count=True``, returns ``(detections, raw_count)`` instead.
        """
        all_dets: list[dict] = []
        for prompt in prompts:
            for d in self.detect(image, prompt, garment_mask, threshold, min_bbox_area_ratio,
                                 fill_mode=fill_mode, dilation_px=dilation_px):
                all_dets.append({**d, "prompt": prompt})

        # Class-aware IoU threshold: multi-instance parts (pocket, button, …)
        # use a lower IoU so adjacent valid detections both survive.
        effective_iou = nms_iou_threshold
        if part and part in MULTI_INSTANCE_IOU:
            effective_iou = MULTI_INSTANCE_IOU[part]

        if nms_mode == "soft":
            result = self._soft_nms(all_dets, effective_iou)
        else:
            result = self._greedy_nms(all_dets, effective_iou)

        if return_raw_count:
            return result, len(all_dets)
        return result

    @staticmethod
    def _greedy_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """Greedy NMS: keep a box only if its IoU with every already-kept box is below threshold."""
        if not detections:
            return []
        dets = sorted(detections, key=lambda d: d["score"], reverse=True)
        kept: list[dict] = []
        for d in dets:
            b1 = d["bbox_xyxy"]
            if all(_box_iou(b1, k["bbox_xyxy"]) < iou_threshold for k in kept):
                kept.append(d)
        return kept

    # Legacy alias — kept for backward compatibility with external callers.
    _nms = _greedy_nms

    @staticmethod
    def _soft_nms(
        detections: list[dict],
        iou_threshold: float = 0.5,
        sigma: float = 0.5,
        score_threshold: float = 0.001,
    ) -> list[dict]:
        """Soft-NMS: Gaussian penalty decay instead of hard removal.

        Each box's score is decayed by ``exp(-iou² / sigma)`` when it overlaps
        with a higher-scoring box above *iou_threshold*.  This preserves
        adjacent detections (e.g. two pockets, button rows) that greedy NMS
        would suppress.

        Args:
            detections: Unsorted list of detection dicts with ``score`` and
                ``bbox_xyxy`` keys.  Scores are mutated in-place.
            iou_threshold: IoU above which decay is applied.
            sigma: Gaussian decay width (smaller = more aggressive decay).
            score_threshold: Boxes whose score falls below this are removed.

        Returns:
            Detections sorted descending by (decayed) score.
        """
        if not detections:
            return []

        dets = sorted(detections, key=lambda d: d["score"], reverse=True)
        n = len(dets)
        # Work on a copy so callers can inspect original scores.
        scores = [d["score"] for d in dets]

        for i in range(n):
            if scores[i] < score_threshold:
                continue
            b_i = dets[i]["bbox_xyxy"]
            for j in range(i + 1, n):
                if scores[j] < score_threshold:
                    continue
                iou = _box_iou(b_i, dets[j]["bbox_xyxy"])
                if iou > iou_threshold:
                    # Gaussian penalty
                    decay = math.exp(-(iou * iou) / sigma)
                    scores[j] *= decay

        # Update scores in-place and filter
        result: list[dict] = []
        for d, s in zip(dets, scores):
            if s >= score_threshold:
                d["score"] = float(s)
                result.append(d)

        result.sort(key=lambda d: d["score"], reverse=True)
        return result

    @staticmethod
    def mask_to_garment(
        image: np.ndarray,
        garment_mask: np.ndarray,
        fill_mode: str = "grey",
        dilation_px: int = 0,
    ) -> np.ndarray:
        """
        Fill pixels outside the garment mask so DINO attends to the garment only.

        Args:
            image: BGR uint8 H×W×3.
            garment_mask: Binary H×W mask (non-zero = garment). Must match
                ``image`` spatial dimensions; no internal resize is performed.
            fill_mode: How to fill non-garment pixels.
                ``"grey"`` → 128 (default, neutral mid-tone).
                ``"black"`` → 0.
                ``"white"`` → 255.
            dilation_px: Dilate the mask by this many pixels before applying.
                Useful for small parts (buttons, zippers) whose bounding boxes
                sit at the garment edge and might otherwise be clipped by a hard
                mask boundary. 0 = no dilation.

        Returns:
            Copy of ``image`` with non-garment pixels filled.

        Raises:
            ValueError: If ``fill_mode`` is not one of the accepted values.
        """
        _FILL_VALUES: dict[str, int] = {"grey": 128, "black": 0, "white": 255}
        if fill_mode not in _FILL_VALUES:
            raise ValueError(
                f"fill_mode must be one of {list(_FILL_VALUES)}, got {fill_mode!r}"
            )

        mask = garment_mask
        # Squeeze H×W×1 → H×W so boolean indexing works on H×W×3 arrays.
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if dilation_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
            )
            mask = cv2.dilate(mask.astype(np.uint8), kernel)

        out = image.copy()
        out[mask == 0] = _FILL_VALUES[fill_mode]
        return out


if __name__ == "__main__":
    import sys

    img_path = "assets/random_train60/images/000004.jpg"
    mask_path = (
        "outputs/test_pipeline_smoke/02_samhq/masks/"
        "000004_det000_long sleeve top_mask.png"
    )

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: cannot read {img_path}", file=sys.stderr)
        sys.exit(1)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"WARNING: mask not found at {mask_path}, running without mask")

    # ponytail: cpu so smoke test runs without GPU
    loc = GroundingDINOLocator(device="cpu")
    print("collar:", loc.detect(img, "collar", mask))
    print("zipper:", loc.detect(img, "zipper", mask))
    print("Smoke test passed.")
