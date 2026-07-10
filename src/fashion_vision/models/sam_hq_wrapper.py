"""
SAM-HQ model wrapper.

This module provides a project-level wrapper for SAM-HQ box-prompt based
mask prediction. It hides third-party implementation details and exposes a
stable interface for 3.1.1 fashion instance segmentation.

The wrapper supports:
    - lazy import of SAM-HQ dependencies
    - checkpoint validation
    - image setting
    - box-prompt prediction
    - latency measurement
    - robust error messages

Expected third-party dependency:
    The SAM-HQ repository should be installed or added to PYTHONPATH.

Common import paths may include:
    - segment_anything
    - segment_anything_hq

The exact available import path depends on the SAM-HQ installation method.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


class SamHqImportError(ImportError):
    """
    Raised when SAM-HQ dependencies cannot be imported.
    """


class SamHqWrapper:
    """
    Wrapper for SAM-HQ box-prompt mask prediction.

    Args:
        checkpoint: Path to SAM-HQ checkpoint file.
        model_type: SAM-HQ model type, such as ``vit_b``, ``vit_l`` or
            ``vit_h``.
        device: Runtime device, such as ``cuda`` or ``cpu``.
        multimask_output: Whether to return multiple masks from SAM-HQ.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_type: str = "vit_b",
        device: str = "cuda",
        multimask_output: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.model_type = model_type
        self.device = device
        self.multimask_output = multimask_output

        self._validate_checkpoint()

        self.sam_model: Any | None = None
        self.predictor: Any | None = None

        self._load_model()

    def _validate_checkpoint(self) -> None:
        """
        Validate SAM-HQ checkpoint path.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If checkpoint path is not a file.
        """
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"SAM-HQ checkpoint does not exist: {self.checkpoint}"
            )

        if not self.checkpoint.is_file():
            raise ValueError(
                f"SAM-HQ checkpoint path is not a file: {self.checkpoint}"
            )

    def _load_model(self) -> None:
        """
        Load SAM-HQ model and predictor.

        Raises:
            SamHqImportError: If SAM-HQ Python package cannot be imported.
            KeyError: If model type is not supported by the registry.
        """
        registry, predictor_cls = self._import_sam_hq()

        if self.model_type not in registry:
            available_types = sorted(registry.keys())
            raise KeyError(
                f"Unsupported SAM-HQ model_type={self.model_type}. "
                f"Available types: {available_types}"
            )

        self.sam_model = registry[self.model_type](
            checkpoint=str(self.checkpoint)
        )
        self.sam_model.to(device=self.device)
        self.predictor = predictor_cls(self.sam_model)

    @staticmethod
    def _import_sam_hq() -> tuple[Dict[str, Any], Any]:
        """
        Import SAM-HQ registry and predictor.

        The exact import path may differ depending on how SAM-HQ is installed.
        This function tries common import paths and provides a clear error
        message when all imports fail.

        Returns:
            A tuple of ``model_registry`` and ``predictor_class``.

        Raises:
            SamHqImportError: If no supported import path is available.
        """
        import_errors: list[str] = []

        try:
            from segment_anything import SamPredictor, sam_model_registry

            return sam_model_registry, SamPredictor
        except ImportError as error:
            import_errors.append(f"segment_anything: {error}")

        try:
            from segment_anything_hq import SamPredictor, sam_model_registry

            return sam_model_registry, SamPredictor
        except ImportError as error:
            import_errors.append(f"segment_anything_hq: {error}")

        message = (
            "Failed to import SAM-HQ dependencies. Please install SAM-HQ or "
            "add the SAM-HQ repository to PYTHONPATH. Tried import paths:\n"
            + "\n".join(import_errors)
        )
        raise SamHqImportError(message)

    def predict_with_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray,
    ) -> Dict[str, Any]:
        """
        Predict a binary mask using one box prompt.

        Args:
            image_rgb: RGB image array with shape ``H x W x 3``.
            box_xyxy: Box prompt in ``[x1, y1, x2, y2]`` format.

        Returns:
            Dictionary containing:
                - mask: Binary uint8 mask with shape ``H x W``.
                - score: SAM-HQ confidence score.
                - sam_latency_ms: Prediction latency in milliseconds.

        Raises:
            RuntimeError: If predictor is not initialized.
            ValueError: If image or box input is invalid.
        """
        self._validate_image(image_rgb)
        box = self._validate_box(box_xyxy)

        if self.predictor is None:
            raise RuntimeError("SAM-HQ predictor is not initialized.")

        start_time = time.perf_counter()

        self.predictor.set_image(image_rgb)

        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=self.multimask_output,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        mask, score = self._select_best_mask(masks, scores)

        return {
            "mask": mask.astype(np.uint8),
            "score": float(score),
            "sam_latency_ms": float(latency_ms),
        }

    def predict_all_masks(
        self,
        image_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray,
    ) -> list[dict]:
        """
        Predict ALL candidate masks using one box prompt (multimask mode).

        Unlike :meth:`predict_with_box`, which returns only the highest-scoring
        mask, this method returns all candidates SAM produces when
        ``multimask_output=True``.  This is used by inner-garment detection to
        search for a secondary object (inner collar) within an outerwear crop.

        Args:
            image_rgb: RGB image array with shape ``H x W x 3``.
            box_xyxy: Box prompt in ``[x1, y1, x2, y2]`` format.

        Returns:
            List of dicts sorted descending by score::

                [{"mask": np.ndarray (H×W uint8), "score": float}, ...]

            Typically 3 candidates.  Empty list on failure.

        Raises:
            RuntimeError: If predictor is not initialized.
            ValueError: If image or box input is invalid.
        """
        self._validate_image(image_rgb)
        box = self._validate_box(box_xyxy)

        if self.predictor is None:
            raise RuntimeError("SAM-HQ predictor is not initialized.")

        start_time = time.perf_counter()

        self.predictor.set_image(image_rgb)

        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True,   # force multimask regardless of constructor setting
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        masks_array = np.asarray(masks)
        scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)

        candidates: list[dict] = []
        for i in range(len(scores_array)):
            m = masks_array[i]
            candidates.append({
                "mask": (m > 0).astype(np.uint8) * 255,
                "score": float(scores_array[i]),
                "sam_latency_ms": float(latency_ms),
            })

        candidates.sort(key=lambda d: d["score"], reverse=True)
        return candidates

    @staticmethod
    def _validate_image(image_rgb: np.ndarray) -> None:
        """
        Validate RGB image input.

        Args:
            image_rgb: RGB image array.

        Raises:
            ValueError: If image format is invalid.
        """
        if not isinstance(image_rgb, np.ndarray):
            raise ValueError("image_rgb must be a numpy.ndarray.")

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                "image_rgb must have shape H x W x 3. "
                f"Got shape: {image_rgb.shape}"
            )

        if image_rgb.size == 0:
            raise ValueError("image_rgb must not be empty.")

    @staticmethod
    def _validate_box(
        box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray,
    ) -> np.ndarray:
        """
        Validate and convert box prompt.

        Args:
            box_xyxy: Box in ``[x1, y1, x2, y2]`` format.

        Returns:
            Box as numpy array with shape ``(4,)``.

        Raises:
            ValueError: If box is invalid.
        """
        box = np.asarray(box_xyxy, dtype=np.float32)

        if box.shape != (4,):
            raise ValueError(
                f"box_xyxy must have shape (4,), got shape: {box.shape}"
            )

        x1, y1, x2, y2 = box.tolist()

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box_xyxy: {box.tolist()}")

        return box

    @staticmethod
    def _select_best_mask(
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Select the best mask from SAM-HQ prediction outputs.

        Args:
            masks: Predicted masks with shape ``N x H x W`` or ``H x W``.
            scores: Predicted confidence scores.

        Returns:
            Best mask and corresponding score.

        Raises:
            ValueError: If prediction output is invalid.
        """
        if masks is None or scores is None:
            raise ValueError("SAM-HQ returned empty masks or scores.")

        masks_array = np.asarray(masks)
        scores_array = np.asarray(scores, dtype=np.float32)

        if masks_array.ndim == 2:
            return masks_array.astype(np.uint8), float(scores_array.reshape(-1)[0])

        if masks_array.ndim != 3:
            raise ValueError(
                "Expected masks with shape N x H x W or H x W, "
                f"got shape: {masks_array.shape}"
            )

        if scores_array.size == 0:
            raise ValueError("SAM-HQ returned empty score array.")

        best_index = int(np.argmax(scores_array))
        best_mask = masks_array[best_index]
        best_score = float(scores_array[best_index])

        return best_mask.astype(np.uint8), best_score
