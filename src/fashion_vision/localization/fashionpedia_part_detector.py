"""
Fashionpedia YOLOv8s 19-class part detector for 3.1.2.

Priority fast detector that runs before the Grounding DINO fallback.
Covers 13 high-confidence parts (including neckline) plus a cuff→sleeve alias.
Six additional decoration-class parts (applique, bead, flower, ribbon, rivet,
tassel) are detected by the model but not exposed as independent PART_VOCAB
entries — they are decoration candidates tracked in debug metadata.

Usage::

    detector = FashionpediaPartDetector("models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt")
    detections = detector.detect(crop_image, "zipper", garment_mask=mask, conf=0.25)
    # → [{"bbox_xyxy": [...], "score": 0.87, "label": "zipper", "class_id": 8, "backend": "fashionpedia_yolo"}, ...]
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Fashionpedia YOLO class ID → internal part name (core 12) ──────────────
FP_CORE_PART_MAP: Dict[int, str] = {
    0: "hood",
    1: "collar",
    2: "lapel",
    3: "epaulette",
    4: "sleeve",
    5: "pocket",
    6: "neckline",
    7: "buckle",
    8: "zipper",
    11: "bow",
    13: "fringe",
    16: "ruffle",
    17: "sequin",
}

# Decoration-class parts — detected by FP model but not in PART_VOCAB.
# Their detections are tracked in debug metadata when the generic "decoration"
# query routes through DINO (or when the model incidentally detects them).
FP_DECORATION_CLASSES: Dict[int, str] = {
    9: "applique",
    10: "bead",
    12: "flower",
    14: "ribbon",
    15: "rivet",
    18: "tassel",
}

# ── Reverse map: internal part → [FP class IDs] ────────────────────────────


def _build_reverse_map(forward: Dict[int, str]) -> Dict[str, List[int]]:
    """Build part→[fp_ids] from fp_id→part, grouping duplicates if any."""
    rev: Dict[str, List[int]] = {}
    for fp_id, part in forward.items():
        rev.setdefault(part, []).append(fp_id)
    return rev


PART_TO_FP_IDS: Dict[str, List[int]] = _build_reverse_map(FP_CORE_PART_MAP)

# Alias: "cuff" queries use Fashionpedia sleeve class (4).
# The YOLO label stays "sleeve", but the result part stays "cuff".
PART_TO_FP_IDS["cuff"] = [4]

# All FP classes that the model can detect (core + decoration).
_ALL_FP_CLASSES: Dict[int, str] = {**FP_CORE_PART_MAP, **FP_DECORATION_CLASSES}


class FashionpediaPartDetector:
    """YOLOv8s 19-class Fashionpedia part detector.

    Caller owns the model lifecycle — instantiate once and reuse across
    ``locate_region()`` calls, same pattern as ``GroundingDINOLocator``.
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """Load the YOLO model.

        Args:
            model_path: Path to a ``.pt`` weights file.
            device: ``"cuda"``, ``"cpu"``, or ``"cuda:0"`` etc.
        """
        from ultralytics import YOLO

        self._model_path = model_path
        self._device = device
        self.model = YOLO(model_path)
        logger.info(
            "FashionpediaPartDetector loaded: model=%s device=%s nc=%d",
            model_path,
            device,
            len(self.model.names),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        target_part: str,
        garment_mask: Optional[np.ndarray] = None,
        conf: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Run YOLO on *image* and return detections matching *target_part*.

        Args:
            image: BGR uint8 crop (already zoomed by the router).
            target_part: Canonical internal part name (e.g. ``"zipper"``).
            garment_mask: Optional binary H×W mask — when provided, non-garment
                pixels are filled with grey (128) before inference so the model
                ignores background / adjacent garments.
            conf: Minimum confidence threshold.

        Returns:
            List of detection dicts sorted by score descending::

                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "score": float,
                    "label": str,
                    "class_id": int,
                    "backend": "fashionpedia_yolo",
                }

            Empty list when no detection for *target_part* exceeds *conf*.
            Bboxes are in the same coordinate space as *image* (crop-local).
        """
        fp_ids = PART_TO_FP_IDS.get(target_part)
        if fp_ids is None:
            return []

        # Mask-gate: fill non-garment pixels so the model ignores background.
        infer_image = image
        if garment_mask is not None:
            infer_image = self._mask_gate(image, garment_mask)

        # YOLO inference expects contiguous BGR array.
        results = self.model(
            infer_image,
            device=self._device,
            conf=conf,
            verbose=False,
        )

        if results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()       # [[x1,y1,x2,y2], ...]
        scores = results[0].boxes.conf.cpu().numpy()       # [score, ...]
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # [cls_id, ...]

        target_set = frozenset(fp_ids)
        detections: List[Dict[str, Any]] = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            if cls_id not in target_set:
                continue
            detections.append({
                "bbox_xyxy": [float(v) for v in box],
                "score": float(score),
                "label": _ALL_FP_CLASSES.get(cls_id, f"cls_{cls_id}"),
                "class_id": int(cls_id),
                "backend": "fashionpedia_yolo",
            })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_gate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill non-garment pixels with grey (128) to suppress background.

        Mutates a copy so the original *image* is unchanged.
        """
        # Squeeze H×W×1 → H×W so boolean indexing works on H×W×3 arrays.
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.shape[:2] != image.shape[:2]:
            import cv2
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        gated = image.copy()
        gated[mask == 0] = 128
        return gated
