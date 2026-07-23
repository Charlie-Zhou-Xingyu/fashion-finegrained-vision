"""
Cached Fast Path — reuses YOLO + SAM-HQ + Landmark models across images.

Wraps the existing 5-stage pipeline WITHOUT modifying any code under
``tools/infer/``, ``src/fashion_vision/``, or ``configs/``.

Status: **Experimental.**

This cached fast path intentionally replicates a small amount of YOLO/SAM-HQ
stage logic (~135 lines) from ``tools/infer/`` to avoid repeated model loading.
It is intended for inference optimization experiments and interactive/single-image
serving prototypes.

**For stable batch/offline processing, prefer**
:class:`inference.pipelines.fast_path_batch_backed.BatchBackedFastPath`
because it delegates to the existing ``GarmentPipeline.run_source()``
implementation and does **not** duplicate core inference logic.

Key risks of this experimental module:
    - ~135 lines replicated from stages 1-2 must stay in sync with upstream
      changes to ``predict_garments_yolo.py`` and ``segment_garments_samhq.py``.
    - Output format may diverge from the reference pipeline if the upstream
      detection/segmentation dict schema changes.
    - Validated on 10 images only; 50-image validation pending.

Design:
    - YOLO model loaded once, reused via ``model.predict()``.
    - SAM-HQ model loaded once, ``SamPredictor`` reused.
    - Landmark model loaded once, passed to existing ``run_segmentation_json_mode``.
    - Stages 3-5 (region crop, mask-aware crop) import existing functions unchanged.
    - Only stages 1-2 inference loops are replicated (~135 lines total).

Usage::

    from inference.pipelines.fast_path_existing_cached import CachedFastPath

    pipe = CachedFastPath()
    pipe.warmup("test_image.jpg")
    result = pipe.run_image("input.jpg")
    # result["timing"] → per-stage wall-clock timings
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── Project-relative path resolution ───────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


# ── Default model paths (same as GarmentPipeline defaults) ─────────────────────

_DEFAULT_YOLO_WEIGHTS = "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
_DEFAULT_SAM_CHECKPOINT = "checkpoints/sam_hq/sam_hq_vit_b.pth"
_DEFAULT_SAM_MODEL_TYPE = "vit_b"
_DEFAULT_LANDMARK_CHECKPOINT = "outputs/landmark_predictor_resnet18/best.pt"
_DEFAULT_LANDMARK_MODEL = "resnet18"
_DEFAULT_LANDMARK_IMAGE_SIZE = 256
_DEFAULT_LANDMARK_MAX_LANDMARKS = 39
_DEFAULT_LANDMARK_PAD_RATIO = 0.05


# ── Cached Fast Path ───────────────────────────────────────────────────────────

class CachedFastPath:
    """Cached-model fast path: YOLO detection + SAM-HQ segmentation + Landmark + crops.

    .. warning::
       **Experimental.** This class replicates ~135 lines of YOLO/SAM-HQ stage
       logic from ``tools/infer/``. For stable batch processing, prefer
       :class:`BatchBackedFastPath` which has zero replicated logic.

    Models are loaded ONCE (lazy) and reused across all ``run_image()`` calls.
    This eliminates ~640 ms per-image model reloading overhead measured in the
    lifecycle benchmark (2026-07-10).

    The output dict format matches ``GarmentPipeline.run_image()`` where possible
    to simplify benchmarking and future drop-in replacement.
    """

    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = _DEFAULT_SAM_MODEL_TYPE,
        landmark_checkpoint: Optional[str] = None,
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        lazy: bool = True,
    ) -> None:
        """
        Args:
            yolo_weights: Path to YOLOv8n .pt weights.
            sam_checkpoint: Path to SAM-HQ .pth checkpoint.
            sam_model_type: SAM-HQ variant (vit_b, vit_l, vit_h).
            landmark_checkpoint: Path to ResNet18 landmark .pt checkpoint.
            device: Torch device string (e.g., "cuda", "cpu").
            imgsz: YOLO inference image size.
            conf: YOLO confidence threshold.
            iou: YOLO NMS IoU threshold.
            lazy: If True, models are loaded on first use (warmup or run_image).
                  If False, models are loaded immediately in __init__.
        """
        self._yolo_weights = yolo_weights or _DEFAULT_YOLO_WEIGHTS
        self._sam_checkpoint = sam_checkpoint or _DEFAULT_SAM_CHECKPOINT
        self._sam_model_type = sam_model_type
        self._landmark_checkpoint = landmark_checkpoint or _DEFAULT_LANDMARK_CHECKPOINT
        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou

        # Model slots — populated on first use
        self._yolo_model: Any = None
        self._sam_predictor: Any = None
        self._sam_model_registry: Any = None
        self._landmark_model: Any = None
        self._landmark_device_obj: Any = None
        self._category_mapping: Any = None

        if not lazy:
            self._ensure_loaded()

    # ── Public API ─────────────────────────────────────────────────────────

    def warmup(self, image_path: Optional[str] = None) -> None:
        """Load all models and optionally run one dummy inference.

        Without warmup, the first ``run_image()`` call pays model loading + CUDA
        JIT compilation cost (up to 2-3 seconds cold).  With warmup, the first
        ``run_image()`` is fast.
        """
        self._ensure_loaded()
        if image_path is not None:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="cfp_warmup_") as td:
                self.run_image(image_path, output_dir=td)

    def run_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full 5-stage Fast Path on a single image.

        Args:
            image_path: Path to input image (jpg/png).
            output_dir: If provided, intermediate outputs are saved to this
                        directory.  If None, a temporary directory is used.

        Returns:
            Dict with keys: status, timing, paths, num_instances, notes.
        """
        import tempfile
        _temp_dir_context: Any = (
            tempfile.TemporaryDirectory(prefix="cfp_")
            if output_dir is None
            else None
        )
        if output_dir is not None:
            out_root = Path(output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
        else:
            out_root = Path(_temp_dir_context.__enter__())  # type: ignore[union-attr]

        try:
            self._ensure_loaded()
            result = self._run_stages(image_path, out_root)
        finally:
            if _temp_dir_context is not None:
                _temp_dir_context.__exit__(None, None, None)

        return result

    # ── Model loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load all models if not already loaded."""
        if self._yolo_model is None:
            self._load_yolo()
        if self._sam_predictor is None:
            self._load_sam()
        if self._landmark_model is None:
            self._load_landmark()

    def _load_yolo(self) -> None:
        from ultralytics import YOLO
        t0 = time.perf_counter()
        weights = str(_resolve(self._yolo_weights))
        self._yolo_model = YOLO(weights)
        self._yolo_model.to(self._device)
        ms = (time.perf_counter() - t0) * 1000
        print(f"[CachedFastPath] YOLO loaded in {ms:.0f} ms: {weights}")

    def _load_sam(self) -> None:
        import torch
        t0 = time.perf_counter()

        # ponytail: same import path as segment_garments_samhq.py
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            from segment_anything_hq import sam_model_registry, SamPredictor

        checkpoint = str(_resolve(self._sam_checkpoint))
        sam = sam_model_registry[self._sam_model_type](checkpoint=checkpoint)
        sam.to(device=self._device)
        self._sam_predictor = SamPredictor(sam)

        ms = (time.perf_counter() - t0) * 1000
        print(f"[CachedFastPath] SAM-HQ loaded in {ms:.0f} ms: {checkpoint}")

    def _load_landmark(self) -> None:
        import torch
        import sys
        _src = str(_PROJECT_ROOT / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)

        from tools.infer.infer_landmarks_for_predictions import load_landmark_model

        device_obj = torch.device(self._device)
        t0 = time.perf_counter()
        self._landmark_model = load_landmark_model(
            checkpoint_path=str(_resolve(self._landmark_checkpoint)),
            model_name=_DEFAULT_LANDMARK_MODEL,
            max_landmarks=_DEFAULT_LANDMARK_MAX_LANDMARKS,
            device=device_obj,
        )
        self._landmark_device_obj = device_obj
        ms = (time.perf_counter() - t0) * 1000
        print(f"[CachedFastPath] Landmark loaded in {ms:.0f} ms")

    def _get_category_mapping(self) -> Any:
        if self._category_mapping is None:
            import sys
            _src = str(_PROJECT_ROOT / "src")
            if _src not in sys.path:
                sys.path.insert(0, _src)
            from tools.eval.category_mapping import load_category_mapping
            mapping_yaml = _PROJECT_ROOT / "configs" / "category_mapping.yaml"
            self._category_mapping = load_category_mapping(mapping_yaml)
        return self._category_mapping

    # ── Stage runners ──────────────────────────────────────────────────────

    def _run_stages(self, image_path: str, out_root: Path) -> Dict[str, Any]:
        """Run all 5 stages, building intermediate outputs in out_root."""
        timing: Dict[str, float] = {}
        notes: List[str] = []
        t_total = time.perf_counter()

        # ── Stage 1: YOLO detection ────────────────────────────────────
        t0 = time.perf_counter()
        yolo_dir = out_root / "01_yolo"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        detections = self._run_yolo(image_path, yolo_dir)
        timing["yolo_seconds"] = round(time.perf_counter() - t0, 4)

        num_detections = len(detections)
        if num_detections == 0:
            timing["total_seconds"] = round(time.perf_counter() - t_total, 4)
            return {
                "status": "ok",
                "num_instances": 0,
                "timing": timing,
                "notes": ["No garments detected."],
            }

        # ── Stage 2: SAM-HQ segmentation ───────────────────────────────
        t0 = time.perf_counter()
        sam_dir = out_root / "02_samhq"
        sam_dir.mkdir(parents=True, exist_ok=True)
        sam_results = self._run_sam(image_path, detections, sam_dir)
        timing["sam_hq_seconds"] = round(time.perf_counter() - t0, 4)

        # ── Stage 3: Landmark prediction ───────────────────────────────
        t0 = time.perf_counter()
        lm_dir = out_root / "03_landmarks"
        lm_dir.mkdir(parents=True, exist_ok=True)
        landmarks_data = self._run_landmark(sam_results, lm_dir)
        timing["landmarks_seconds"] = round(time.perf_counter() - t0, 4)

        # ── Stage 4: Region crop ───────────────────────────────────────
        t0 = time.perf_counter()
        crop_dir = out_root / "04_region_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        region_data = self._run_region_crop(landmarks_data, crop_dir)
        timing["region_crop_seconds"] = round(time.perf_counter() - t0, 4)

        # ── Stage 5: Mask-aware crop ───────────────────────────────────
        t0 = time.perf_counter()
        mask_dir = out_root / "05_region_masked_crops"
        mask_dir.mkdir(parents=True, exist_ok=True)
        masked_data = self._run_mask_aware_crop(region_data, sam_results, mask_dir)
        timing["mask_aware_crop_seconds"] = round(time.perf_counter() - t0, 4)

        timing["total_seconds"] = round(time.perf_counter() - t_total, 4)

        num_instances = (
            region_data.get("summary", {}).get("num_instances", 0)
            if isinstance(region_data, dict) else 0
        )

        return {
            "status": "ok",
            "num_instances": num_instances,
            "timing": timing,
            "notes": notes,
            "paths": {
                "yolo_dir": str(yolo_dir),
                "sam_dir": str(sam_dir),
                "landmark_dir": str(lm_dir),
                "region_crop_dir": str(crop_dir),
                "masked_crop_dir": str(mask_dir),
            },
        }

    # ── Stage 1 internals ─────────────────────────────────────────────────

    def _run_yolo(self, image_path: str, out_dir: Path) -> List[Dict[str, Any]]:
        """Run cached YOLO model on one image. Returns list of detection dicts."""
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]

        results = self._yolo_model.predict(
            source=image_path,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        if len(results) != 1:
            return []

        result = results[0]
        detections: List[Dict[str, Any]] = []
        mapping = self._get_category_mapping()
        model_names = self._yolo_model.names

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for det_id, (bbox, conf_val, class_id) in enumerate(
                zip(boxes_xyxy, confs, class_ids)
            ):
                bbox_xyxy = [
                    float(max(0, bbox[0])),
                    float(max(0, bbox[1])),
                    float(min(width, bbox[2])),
                    float(min(height, bbox[3])),
                ]
                bbox_w = bbox_xyxy[2] - bbox_xyxy[0]
                bbox_h = bbox_xyxy[3] - bbox_xyxy[1]
                class_id = int(class_id)
                class_name = model_names.get(class_id, str(class_id))
                coarse_id = mapping.map_13_to_5.get(class_id, class_id)

                detections.append({
                    "det_id": det_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "fine_class_id": class_id,
                    "fine_class_name": class_name,
                    "coarse_class_id": coarse_id,
                    "coarse_class_name": mapping.prd_5cls.get(coarse_id, class_name),
                    "confidence": float(conf_val),
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_xywh": [bbox_xyxy[0], bbox_xyxy[1], bbox_w, bbox_h],
                    "bbox_format": "xyxy_abs_pixels",
                    "image_width": width,
                    "image_height": height,
                })

        # Save detections.json for downstream stage compatibility
        all_results = {
            "task": "predict_garments_yolo_cached",
            "weights": str(_resolve(self._yolo_weights)),
            "source": image_path,
            "imgsz": self._imgsz,
            "conf": self._conf,
            "iou": self._iou,
            "device": self._device,
            "class_names": {str(k): v for k, v in model_names.items()},
            "images": [{
                "image_id": 0,
                "image_path": image_path,
                "file_name": Path(image_path).name,
                "width": width,
                "height": height,
                "num_detections": len(detections),
                "detections": detections,
            }],
        }
        json_path = out_dir / "detections.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        return detections

    # ── Stage 2 internals ─────────────────────────────────────────────────

    def _run_sam(
        self, image_path: str, detections: List[Dict], out_dir: Path
    ) -> Dict[str, Any]:
        """Run cached SAM-HQ predictor. Returns segmentation results dict."""
        import cv2

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self._sam_predictor.set_image(image_rgb)

        masks_dir = out_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        image_segments: List[Dict[str, Any]] = []

        for det in detections:
            det_id = int(det["det_id"])
            class_name = str(det["class_name"])
            confidence = float(det["confidence"])
            bbox_xyxy = [float(v) for v in det["bbox_xyxy"]]

            # Skip tiny boxes
            area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
            if area < 10.0:
                continue

            box_np = np.array([
                max(0, bbox_xyxy[0]),
                max(0, bbox_xyxy[1]),
                min(width, bbox_xyxy[2]),
                min(height, bbox_xyxy[3]),
            ])

            try:
                masks, scores, logits = self._sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_np,
                    multimask_output=False,
                )
            except TypeError:
                masks, scores, logits = self._sam_predictor.predict(
                    box=box_np, multimask_output=False,
                )

            best_idx = int(scores.argmax())
            best_mask = masks[best_idx].astype(bool)
            best_score = float(scores[best_idx])
            mask_u8 = (best_mask.astype(np.uint8) * 255)

            mask_name = f"{Path(image_path).stem}_det{det_id:03d}_{class_name}_mask.png"
            mask_path = masks_dir / mask_name
            cv2.imwrite(str(mask_path), mask_u8)

            image_segments.append({
                "det_id": det_id,
                "class_id": int(det["class_id"]),
                "class_name": class_name,
                "confidence": confidence,
                "bbox_xyxy": bbox_xyxy,
                "bbox_format": "xyxy_abs_pixels",
                "mask_path": str(mask_path),
                "mask_area": int(best_mask.sum()),
                "sam_score": best_score,
                "sam_best_mask_idx": best_idx,
                "image_width": width,
                "image_height": height,
            })

        results = {
            "task": "segment_garments_samhq_cached",
            "sam_checkpoint": str(_resolve(self._sam_checkpoint)),
            "model_type": self._sam_model_type,
            "device": self._device,
            "images": [{
                "image_id": 0,
                "image_path": image_path,
                "file_name": Path(image_path).name,
                "width": width,
                "height": height,
                "num_detections": len(detections),
                "num_segments": len(image_segments),
                "detections": detections,
                "segments": image_segments,
            }],
        }

        json_path = out_dir / "segmentation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results

    # ── Stage 3-5 (imported from existing pipeline) ────────────────────────

    def _run_landmark(
        self, sam_results: Dict, out_dir: Path
    ) -> Dict[str, Any]:
        """Run landmark prediction using cached model + existing inference function."""
        import sys
        _src = str(_PROJECT_ROOT / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)

        from tools.infer.infer_landmarks_for_predictions import run_segmentation_json_mode

        # Save sam_results to JSON for file-based pipeline compatibility
        seg_json = out_dir / "_input_segmentation.json"
        with open(seg_json, "w", encoding="utf-8") as f:
            json.dump(sam_results, f, ensure_ascii=False, indent=2)

        args = Namespace(
            segmentation_json=str(seg_json),
            output_dir=str(out_dir),
            checkpoint=str(_resolve(self._landmark_checkpoint)),
            model=_DEFAULT_LANDMARK_MODEL,
            image_size=_DEFAULT_LANDMARK_IMAGE_SIZE,
            max_landmarks=_DEFAULT_LANDMARK_MAX_LANDMARKS,
            pad_ratio=_DEFAULT_LANDMARK_PAD_RATIO,
            device=self._device,
            save_visualizations=False,
            draw_index=False,
            draw_name=False,
            use_mask_quality=True,
            refine_with_mask=False,
            max_mask_refine_distance=20.0,
            add_landmark_schema=True,
            filter_landmarks_by_category=False,
            drop_invalid_landmarks=False,
        )

        run_segmentation_json_mode(
            args=args,
            model=self._landmark_model,
            device=self._landmark_device_obj,
        )

        # Load the output landmarks JSON that run_segmentation_json_mode writes
        lm_json = out_dir / "landmarks_results.json"
        if lm_json.exists():
            with open(lm_json, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _run_region_crop(
        self, landmarks_data: Dict, out_dir: Path
    ) -> Dict[str, Any]:
        """Run region crop generation (stage 4)."""
        import sys
        _src = str(_PROJECT_ROOT / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)

        from tools.crop.crop_garment_regions_from_landmarks import run_region_crop

        # Save landmarks data to JSON for file-based pipeline compatibility
        lm_json = out_dir / "_input_landmarks.json"
        with open(lm_json, "w", encoding="utf-8") as f:
            json.dump(landmarks_data, f, ensure_ascii=False, indent=2)

        args = Namespace(
            landmarks_json=str(lm_json),
            output_dir=str(out_dir),
            regions=["collar", "sleeve", "hem", "waist", "pant_leg"],
            use_category_regions=True,
            max_outside_distance=5.0,
            min_points=2,
            pad_ratio=0.35,
            single_point_box_ratio=0.18,
            fallback=True,
            save_debug=False,
        )
        return run_region_crop(args)

    def _run_mask_aware_crop(
        self, region_data: Dict, sam_results: Dict, out_dir: Path
    ) -> Dict[str, Any]:
        """Run mask-aware crop application (stage 5)."""
        import sys
        _src = str(_PROJECT_ROOT / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)

        from tools.crop.apply_samhq_mask_to_region_crops import run_apply_samhq_mask

        # Find the region_crops.json that run_region_crop wrote
        region_json = out_dir.parent / "04_region_crops" / "region_crops.json"
        if not region_json.exists():
            # ponytail: region_data may already have the path
            region_json = out_dir / "_input_region_crops.json"
            with open(region_json, "w", encoding="utf-8") as f:
                json.dump(region_data, f, ensure_ascii=False, indent=2)

        seg_json = out_dir / "_input_segmentation.json"
        if not seg_json.exists():
            with open(seg_json, "w", encoding="utf-8") as f:
                json.dump(sam_results, f, ensure_ascii=False, indent=2)
        else:
            # Use the one from stage 2
            seg_json = out_dir.parent / "02_samhq" / "segmentation_results.json"

        if not region_json.exists():
            # It was written by stage 4
            region_json = out_dir.parent / "04_region_crops" / "region_crops.json"

        args = Namespace(
            region_crops_json=str(region_json),
            segmentation_json=str(seg_json),
            output_dir=str(out_dir),
            background="white",
            transparent=False,
            min_mask_area_ratio=0.005,
        )
        return run_apply_samhq_mask(args)


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Verify the class instantiates without error (models lazy-loaded by default)."""
    pipe = CachedFastPath(lazy=True)
    assert pipe._yolo_model is None
    assert pipe._sam_predictor is None
    assert pipe._landmark_model is None
    print("[CachedFastPath] Lazy-load constructor OK (no models loaded).")


if __name__ == "__main__":
    _demo()
