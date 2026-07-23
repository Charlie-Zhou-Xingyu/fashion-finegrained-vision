"""
TensorRT-accelerated YOLO wrapper with automatic PyTorch fallback.

Usage::

    from inference.wrappers.yolo_wrapper import YOLOWrapper

    wrapper = YOLOWrapper(
        engine_path="inference/engines/yolov8n_fp16.engine",
        pt_path="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
    )
    results = wrapper.detect(image_bgr)  # list[dict] with bbox, score, class_id

Status: Skeleton. The TensorRT codepath is NOT yet implemented.
Current behavior: always loads PyTorch model via ultralytics.YOLO.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class YOLOWrapper:
    """YOLOv8 garment detector wrapper.

    Design:
    - If engine_path exists and is a valid .engine file, loads TensorRT engine.
    - Otherwise, falls back to ultralytics YOLO (.pt).
    - Both paths expose the same detect() interface.
    - Accuracy validation: mAP50 on val2000 must be within 0.5% of PyTorch baseline.

    At present, no TensorRT engine exists, so this always falls back to PyTorch.
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        pt_path: Optional[str] = None,
        imgsz: int = 640,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self._imgsz = imgsz
        self._device = device
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._use_trt = False

        # ── Try TensorRT first ──
        if engine_path and Path(engine_path).exists():
            # ponytail: TRT loader not yet implemented.
            # TODO Week 3: implement _load_trt_engine() using tensorrt Python API
            # or pycuda. For now, fall through to PyTorch.
            self._use_trt = False
            self._engine = None
            self._engine_context = None

        # ── Fallback: PyTorch ──
        if not self._use_trt:
            self._load_pytorch(pt_path)

    def _load_pytorch(self, pt_path: Optional[str]) -> None:
        """Load YOLO model from .pt checkpoint."""
        from ultralytics import YOLO

        if pt_path is None:
            raise ValueError("pt_path is required when no TensorRT engine is available")

        full_path = Path(pt_path)
        if not full_path.is_absolute():
            # Resolve relative to project root
            from pathlib import Path as P
            project_root = P(__file__).resolve().parent.parent.parent
            full_path = project_root / pt_path

        if not full_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {full_path}")

        t0 = time.perf_counter()
        self._model = YOLO(str(full_path))
        self._model.to(self._device)
        self._use_trt = False
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[YOLOWrapper] Loaded PyTorch model in {load_ms:.0f} ms: {full_path}")

    # ponytail: _load_trt_engine() stub — implement Week 3 when first engine is built.
    # def _load_trt_engine(self, engine_path: str) -> None: ...

    def detect(
        self,
        image: np.ndarray,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run YOLO detection on a BGR image (numpy array, H×W×3).

        Returns list of dicts with keys:
            bbox_xyxy, confidence, class_id, class_name, fine_class_id,
            coarse_class_id, coarse_class_name
        """
        if self._use_trt:
            return self._detect_trt(image)
        return self._detect_pytorch(image, verbose)

    def _detect_pytorch(
        self, image: np.ndarray, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """Run PyTorch YOLO inference."""
        results = self._model(image, verbose=verbose, imgsz=self._imgsz)
        detections: List[Dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                if conf < self._conf:
                    continue
                xyxy = boxes.xyxy[i].tolist()
                detections.append({
                    "bbox_xyxy": xyxy,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": self._model.names.get(cls_id, str(cls_id)),
                })
        return detections

    def _detect_trt(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run TensorRT engine inference. NOT YET IMPLEMENTED."""
        raise NotImplementedError(
            "TensorRT codepath not implemented. "
            "Build engine first (Week 3), then implement this method."
        )

    # ── Compatibility properties ──────────────────────────────────────────

    @property
    def names(self) -> Dict[int, str]:
        """Class ID -> class name mapping."""
        if self._use_trt:
            # ponytail: hardcoded DeepFashion2 13-class names for TRT path.
            # TODO Week 3: load from engine metadata or config.
            return {
                0: "short sleeve top", 1: "long sleeve top",
                2: "short sleeve outwear", 3: "long sleeve outwear",
                4: "vest", 5: "sling",
                6: "shorts", 7: "trousers", 8: "skirt",
                9: "short sleeve dress", 10: "long sleeve dress",
                11: "vest dress", 12: "sling dress",
            }
        return self._model.names

    @property
    def is_trt(self) -> bool:
        return self._use_trt


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Verify the wrapper loads (PyTorch fallback) without error."""
    import sys
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

    pt = str(project_root / "models" / "detectors" / "yolov8n_deepfashion2_13cls_best.pt")
    if not Path(pt).exists():
        print(f"[SKIP] YOLO checkpoint not found at {pt}")
        return

    wrapper = YOLOWrapper(pt_path=pt)
    assert not wrapper.is_trt, "Expected PyTorch fallback (no TRT engine)"
    print(f"  Classes: {len(wrapper.names)}")
    print("  yolo_wrapper: PyTorch fallback OK.")


if __name__ == "__main__":
    _demo()
