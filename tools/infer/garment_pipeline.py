from __future__ import annotations

import json
import time
from argparse import Namespace
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import torch

from tools.infer.predict_garments_yolo import run_inference as run_yolo_inference
from tools.infer.segment_garments_samhq import run as run_samhq_segmentation
from tools.infer.infer_landmarks_for_predictions import (
    load_landmark_model,
    run_segmentation_json_mode,
)
from tools.crop.crop_garment_regions_from_landmarks import run_region_crop
from tools.crop.apply_samhq_mask_to_region_crops import run_apply_samhq_mask


@dataclass
class GarmentPipelineConfig:
    yolo_weights: str = "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
    sam_checkpoint: str = "checkpoints/sam_hq/sam_hq_vit_b.pth"
    sam_model_type: str = "vit_b"

    landmark_checkpoint: str = "outputs/landmark_predictor_resnet18/best.pt"
    landmark_model: str = "resnet18"
    landmark_image_size: int = 256
    landmark_max_landmarks: int = 39
    landmark_pad_ratio: float = 0.05

    yolo_imgsz: int = 640
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    yolo_device: str = "0"
    sam_device: str = "cuda"
    landmark_device: str = "cuda"

    save_yolo_vis: bool = True
    save_yolo_crops: bool = False
    save_landmark_visualizations: bool = False
    draw_landmark_index: bool = False
    draw_landmark_name: bool = False

    region_crop_regions: tuple[str, ...] = (
        "collar",
        "sleeve",
        "hem",
        "waist",
        "pant_leg",
    )
    use_category_regions: bool = True
    region_max_outside_distance: float = 5.0
    region_min_points: int = 2
    region_pad_ratio: float = 0.35
    region_single_point_box_ratio: float = 0.18
    region_fallback: bool = True

    masked_crop_background: str = "white"
    masked_crop_transparent: bool = False
    min_mask_area_ratio: float = 0.005

    # Stages 3-5: landmark prediction + region crops + masked crops.
    # Set False for 3.1.2 open-vocab queries that only need YOLO+SAM.
    run_landmark_and_crops: bool = True

    # Stage 6: fine-grained attribute inference (opt-in)
    run_attribute_inference: bool = False
    attribute_device: str = "auto"
    attribute_topk: int = 3
    attribute_inference_config: str = ""   # empty → use configs/attribute_inference.yaml
    attribute_group_mapping_config: str = ""  # empty → use configs/attribute_group_mapping.yaml


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_round(x: float, ndigits: int = 4) -> float:
    return round(float(x), ndigits)


class GarmentPipeline:
    """
    Function-call based garment pipeline.

    This class does not call subprocess.
    It directly reuses functions from:
      - predict_garments_yolo.py
      - segment_garments_samhq.py
      - infer_landmarks_for_predictions.py
      - crop_garment_regions_from_landmarks.py
      - apply_samhq_mask_to_region_crops.py
    """

    def __init__(self, config: Optional[GarmentPipelineConfig] = None):
        self.config = config or GarmentPipelineConfig()
        self._landmark_model = None
        self._landmark_device = None
        self._attribute_pipeline = None  # lazy-loaded on first Stage 6 call
        self._sam_wrapper = None         # lazy-loaded on first open-vocab bbox→mask call

    def _get_landmark_device(self) -> torch.device:
        device_str = self.config.landmark_device
        if device_str == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested for landmark but not available. Use CPU.")
            device_str = "cpu"
        return torch.device(device_str)

    def _get_attribute_pipeline(self):
        """Lazy-load and cache the Stage 6 GarmentAttributePipeline.

        Deferred import keeps attribute-inference deps out of the module's
        top-level import graph when Stage 6 is disabled.

        Returns:
            A ready-to-use :class:`GarmentAttributePipeline` instance.
        """
        if self._attribute_pipeline is not None:
            return self._attribute_pipeline

        # fashion_vision lives under src/; add it to sys.path if needed.
        import sys as _sys
        _src_dir = str(Path(__file__).resolve().parents[2] / "src")
        if _src_dir not in _sys.path:
            _sys.path.insert(0, _src_dir)

        from fashion_vision.attributes.garment_attribute_pipeline import (  # noqa: PLC0415
            GarmentAttributePipeline,
            AttributePipelineConfig,
        )

        attr_cfg = AttributePipelineConfig(
            device=self.config.attribute_device,
            topk=self.config.attribute_topk,
        )
        if self.config.attribute_inference_config:
            attr_cfg.inference_config_path = _Path(self.config.attribute_inference_config)
        if self.config.attribute_group_mapping_config:
            attr_cfg.group_mapping_path = _Path(self.config.attribute_group_mapping_config)

        self._attribute_pipeline = GarmentAttributePipeline(attr_cfg)
        return self._attribute_pipeline

    def get_sam_wrapper(self):
        """
        Lazy-load and cache a SamHqWrapper for bbox-to-mask refinement.

        This is a second SAM instance used exclusively by the open-vocab
        localization path (DINO bbox → SAM box prompt → local mask).
        It is independent of Stage 2 segmentation, which uses its own loader.

        Returns:
            A ready-to-use SamHqWrapper instance.
        """
        if self._sam_wrapper is not None:
            return self._sam_wrapper

        import sys as _sys
        _src_dir = str(Path(__file__).resolve().parents[2] / "src")
        if _src_dir not in _sys.path:
            _sys.path.insert(0, _src_dir)

        from fashion_vision.models.sam_hq_wrapper import SamHqWrapper  # noqa: PLC0415

        self._sam_wrapper = SamHqWrapper(
            checkpoint=self.config.sam_checkpoint,
            model_type=self.config.sam_model_type,
            device=self.config.sam_device,
        )
        return self._sam_wrapper

    def _get_landmark_model(self):
        """
        Lazy load landmark model.
        YOLO and SAM are still loaded inside their existing stage functions.
        Later we can further optimize them into persistent models too.
        """
        if self._landmark_model is not None:
            return self._landmark_model, self._landmark_device

        device = self._get_landmark_device()
        model = load_landmark_model(
            checkpoint_path=self.config.landmark_checkpoint,
            model_name=self.config.landmark_model,
            max_landmarks=self.config.landmark_max_landmarks,
            device=device,
        )

        self._landmark_model = model
        self._landmark_device = device

        return model, device

    def run_source(
        self,
        source: str,
        output_dir: str,
        max_images: Optional[int] = None,
    ) -> dict[str, Any]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        yolo_output_dir = output_root / "01_yolo"
        sam_output_dir = output_root / "02_samhq"
        landmark_output_dir = output_root / "03_landmarks"
        region_crop_output_dir = output_root / "04_region_crops"
        masked_crop_output_dir = output_root / "05_region_masked_crops"

        timing: dict[str, float] = {}
        t_total = time.perf_counter()

        # -----------------------------
        # 1. YOLO
        # -----------------------------
        t0 = time.perf_counter()
        yolo_args = Namespace(
            weights=self.config.yolo_weights,
            source=source,
            output_dir=str(yolo_output_dir),
            imgsz=self.config.yolo_imgsz,
            conf=self.config.yolo_conf,
            iou=self.config.yolo_iou,
            device=self.config.yolo_device,
            save_vis=self.config.save_yolo_vis,
            save_crops=self.config.save_yolo_crops,
            max_images=max_images,
        )
        run_yolo_inference(yolo_args)
        timing["yolo_seconds"] = safe_round(time.perf_counter() - t0)

        detections_json = yolo_output_dir / "detections.json"

        # -----------------------------
        # 2. SAM-HQ
        # -----------------------------
        t0 = time.perf_counter()
        sam_args = Namespace(
            detections_json=str(detections_json),
            sam_checkpoint=self.config.sam_checkpoint,
            model_type=self.config.sam_model_type,
            output_dir=str(sam_output_dir),
            device=self.config.sam_device,
            multimask_output=False,
            mask_alpha=0.45,
            min_box_area=10.0,
        )
        run_samhq_segmentation(sam_args)
        timing["sam_hq_seconds"] = safe_round(time.perf_counter() - t0)

        segmentation_json = sam_output_dir / "segmentation_results.json"

        # Defaults for stages 3-5 (set when run_landmark_and_crops=False).
        landmarks_json: Optional[Path] = None
        region_crops_json: Optional[Path] = None
        region_masked_crops_json: Optional[Path] = None
        region_data: dict[str, Any] = {}
        masked_data: dict[str, Any] = {}

        if self.config.run_landmark_and_crops:
            # -----------------------------
            # 3. Landmarks
            # -----------------------------
            t0 = time.perf_counter()
            landmark_model, landmark_device = self._get_landmark_model()

            landmark_args = Namespace(
                segmentation_json=str(segmentation_json),
                output_dir=str(landmark_output_dir),
                checkpoint=self.config.landmark_checkpoint,
                model=self.config.landmark_model,
                image_size=self.config.landmark_image_size,
                max_landmarks=self.config.landmark_max_landmarks,
                pad_ratio=self.config.landmark_pad_ratio,
                device=self.config.landmark_device,
                save_visualizations=self.config.save_landmark_visualizations,
                draw_index=self.config.draw_landmark_index,
                draw_name=self.config.draw_landmark_name,
                use_mask_quality=True,
                refine_with_mask=False,
                max_mask_refine_distance=20.0,
                add_landmark_schema=True,
                filter_landmarks_by_category=False,
                drop_invalid_landmarks=False,
            )

            run_segmentation_json_mode(
                args=landmark_args,
                model=landmark_model,
                device=landmark_device,
            )
            timing["landmarks_seconds"] = safe_round(time.perf_counter() - t0)

            landmarks_json = landmark_output_dir / "landmarks_results.json"

            # -----------------------------
            # 4. Region crops
            # -----------------------------
            t0 = time.perf_counter()
            region_args = Namespace(
                landmarks_json=str(landmarks_json),
                output_dir=str(region_crop_output_dir),
                regions=list(self.config.region_crop_regions),
                use_category_regions=self.config.use_category_regions,
                max_outside_distance=self.config.region_max_outside_distance,
                min_points=self.config.region_min_points,
                pad_ratio=self.config.region_pad_ratio,
                single_point_box_ratio=self.config.region_single_point_box_ratio,
                fallback=self.config.region_fallback,
                save_debug=False,
            )
            region_data = run_region_crop(region_args)
            timing["region_crops_seconds"] = safe_round(time.perf_counter() - t0)

            region_crops_json = region_crop_output_dir / "region_crops.json"

            # -----------------------------
            # 5. Mask-aware crops
            # -----------------------------
            t0 = time.perf_counter()
            masked_args = Namespace(
                region_crops_json=str(region_crops_json),
                segmentation_json=str(segmentation_json),
                output_dir=str(masked_crop_output_dir),
                background=self.config.masked_crop_background,
                transparent=self.config.masked_crop_transparent,
                min_mask_area_ratio=self.config.min_mask_area_ratio,
            )
            masked_data = run_apply_samhq_mask(masked_args)
            timing["masked_crops_seconds"] = safe_round(time.perf_counter() - t0)

            region_masked_crops_json = masked_crop_output_dir / "region_masked_crops.json"

        # -----------------------------
        # 6. Attribute inference (opt-in)
        # -----------------------------
        attributes_jsonl: Optional[Path] = None
        attr_results: list[dict[str, Any]] = []

        if self.config.run_attribute_inference:
            t0 = time.perf_counter()
            attr_output_dir = output_root / "06_attributes"
            attr_output_dir.mkdir(parents=True, exist_ok=True)
            attributes_jsonl = attr_output_dir / "predictions.jsonl"

            # Prefer masked crops (stage 5) for cleaner background; fall back to stage 4.
            crops_input = (
                region_masked_crops_json
                if region_masked_crops_json.exists()
                else region_crops_json
            )

            attr_pipeline = self._get_attribute_pipeline()
            attr_results = attr_pipeline.predict_from_json(crops_input)

            with attributes_jsonl.open("w", encoding="utf-8") as fh:
                for rec in attr_results:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            timing["attribute_seconds"] = safe_round(time.perf_counter() - t0)

        timing["total_seconds"] = safe_round(time.perf_counter() - t_total)

        result = {
            "task": "garment_pipeline_function_call",
            "status": "success",
            "source": source,
            "output_dir": str(output_root),
            "config": asdict(self.config),
            "timing": timing,
            "paths": {
                "detections_json": str(detections_json),
                "segmentation_json": str(segmentation_json),
                "landmarks_json": str(landmarks_json) if landmarks_json else None,
                "region_crops_json": str(region_crops_json) if region_crops_json else None,
                "region_masked_crops_json": str(region_masked_crops_json) if region_masked_crops_json else None,
                "attributes_jsonl": str(attributes_jsonl) if attributes_jsonl else None,
                "pipeline_summary_json": str(output_root / "pipeline_summary.json"),
            },
            "region_crops_summary": region_data.get("summary", {}),
            "masked_crops_summary": masked_data.get("summary", {}),
            "attributes_summary": {
                "num_instances": len(attr_results),
                "num_with_attributes": sum(1 for r in attr_results if r["attributes"]),
                "num_errors": sum(1 for r in attr_results if r["error"]),
            } if self.config.run_attribute_inference else None,
        }

        save_json(result, output_root / "pipeline_summary.json")

        return result

    def run_image(
        self,
        image_path: str,
        output_dir: str,
    ) -> dict[str, Any]:
        return self.run_source(
            source=image_path,
            output_dir=output_dir,
            max_images=1,
        )
