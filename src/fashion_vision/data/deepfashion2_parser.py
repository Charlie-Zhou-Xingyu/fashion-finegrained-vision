"""
DeepFashion2 dataset parser.

This module reads DeepFashion2 annotations and converts garment instances
into a unified internal format for SAM-HQ based segmentation evaluation.

The parser extracts:
    - image path
    - annotation path
    - image size
    - garment instance ID
    - bounding box
    - raw DeepFashion2 category
    - mapped 8-class target category
    - ground-truth binary mask
    - DeepFashion2 clothing landmarks

The parser is designed for the first stage of 3.1.1 fashion instance
segmentation, where DeepFashion2 ground-truth bounding boxes are used as
SAM-HQ box prompts.

It is also used as the data foundation for later local-region localization
tasks, where DeepFashion2 landmarks can help locate neckline, cuff, hem,
shoulder, waist, and other garment parts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from fashion_vision.data.class_mapping import map_deepfashion2_category
from fashion_vision.data.landmarks import (
    count_visible_landmarks,
    parse_flat_landmarks,
)


LOGGER = logging.getLogger(__name__)


class DeepFashion2Parser:
    """
    Parser for DeepFashion2 image and annotation files.

    Args:
        root: DeepFashion2 dataset root directory.
        split: Dataset split, such as ``train`` or ``validation``.
        image_dir: Relative image directory from dataset root.
        annotation_dir: Relative annotation directory from dataset root.
        min_bbox_area: Minimum bounding box area for a valid instance.
        skip_empty_mask: Whether to skip instances with empty masks.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "validation",
        image_dir: str | None = None,
        annotation_dir: str | None = None,
        min_bbox_area: int = 1,
        skip_empty_mask: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / (
            image_dir if image_dir is not None else f"{split}/image"
        )
        self.annotation_dir = self.root / (
            annotation_dir if annotation_dir is not None else f"{split}/annos"
        )
        self.min_bbox_area = min_bbox_area
        self.skip_empty_mask = skip_empty_mask

        self._validate_paths()

    def _validate_paths(self) -> None:
        """
        Validate dataset directories.

        Raises:
            FileNotFoundError: If required directories do not exist.
            ValueError: If paths are not directories.
        """
        if not self.root.exists():
            raise FileNotFoundError(
                f"DeepFashion2 root does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise ValueError(f"DeepFashion2 root is not a directory: {self.root}")

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"DeepFashion2 image directory does not exist: "
                f"{self.image_dir}"
            )

        if not self.image_dir.is_dir():
            raise ValueError(
                f"DeepFashion2 image path is not a directory: {self.image_dir}"
            )

        if not self.annotation_dir.exists():
            raise FileNotFoundError(
                f"DeepFashion2 annotation directory does not exist: "
                f"{self.annotation_dir}"
            )

        if not self.annotation_dir.is_dir():
            raise ValueError(
                f"DeepFashion2 annotation path is not a directory: "
                f"{self.annotation_dir}"
            )

    def list_image_ids(self) -> List[str]:
        """
        List image IDs available in the image directory.

        Returns:
            Sorted list of image IDs without file extension.
        """
        image_paths = []
        image_paths.extend(self.image_dir.glob("*.jpg"))
        image_paths.extend(self.image_dir.glob("*.jpeg"))
        image_paths.extend(self.image_dir.glob("*.png"))

        image_ids = sorted({path.stem for path in image_paths})
        return image_ids

    def load_sample(self, image_id: str) -> Dict[str, Any]:
        """
        Load one DeepFashion2 sample by image ID.

        Args:
            image_id: Image ID without file extension, such as ``000001``.

        Returns:
            A dictionary containing image metadata and parsed instances.

        Raises:
            FileNotFoundError: If image or annotation file does not exist.
            ValueError: If image cannot be read.
        """
        image_path = self._find_image_path(image_id)
        annotation_path = self.annotation_dir / f"{image_id}.json"

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation file does not exist: {annotation_path}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        height, width = image.shape[:2]

        annotation = self._load_annotation(annotation_path)
        instances = self.parse_instances(annotation, height, width)

        return {
            "image_id": image_id,
            "image_path": image_path,
            "annotation_path": annotation_path,
            "width": width,
            "height": height,
            "instances": instances,
        }

    def _find_image_path(self, image_id: str) -> Path:
        """
        Find image path by trying common image extensions.

        Args:
            image_id: Image ID without extension.

        Returns:
            Existing image path.

        Raises:
            FileNotFoundError: If no image file is found.
        """
        for extension in [".jpg", ".jpeg", ".png"]:
            image_path = self.image_dir / f"{image_id}{extension}"
            if image_path.exists():
                return image_path

        raise FileNotFoundError(
            f"Image file not found for image_id={image_id} "
            f"in {self.image_dir}"
        )

    @staticmethod
    def _load_annotation(annotation_path: str | Path) -> Dict[str, Any]:
        """
        Load one annotation JSON file.

        Args:
            annotation_path: Path to annotation JSON.

        Returns:
            Parsed annotation dictionary.

        Raises:
            ValueError: If annotation root is not a dictionary.
        """
        path = Path(annotation_path)

        with path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)

        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation root must be a dictionary: {path}")

        return annotation

    def parse_instances(
        self,
        annotation: Dict[str, Any],
        height: int,
        width: int,
    ) -> List[Dict[str, Any]]:
        """
        Parse garment instances from one DeepFashion2 annotation.

        Args:
            annotation: DeepFashion2 annotation dictionary.
            height: Image height.
            width: Image width.

        Returns:
            List of parsed garment instances.
        """
        instances: List[Dict[str, Any]] = []

        for key, value in sorted(annotation.items()):
            if not key.startswith("item"):
                continue

            if not isinstance(value, dict):
                LOGGER.warning("Skip invalid item field: %s", key)
                continue

            try:
                instance = self._parse_single_instance(
                    instance_id=key,
                    item=value,
                    height=height,
                    width=width,
                )
            except (KeyError, ValueError, TypeError) as error:
                LOGGER.warning(
                    "Skip instance %s due to parsing error: %s",
                    key,
                    error,
                )
                continue

            if instance is not None:
                instances.append(instance)

        return instances

    def _parse_single_instance(
        self,
        instance_id: str,
        item: Dict[str, Any],
        height: int,
        width: int,
    ) -> Dict[str, Any] | None:
        """
        Parse one garment instance.

        Args:
            instance_id: Instance key in annotation, such as ``item1``.
            item: DeepFashion2 item annotation.
            height: Image height.
            width: Image width.

        Returns:
            Parsed instance dictionary, or None if skipped.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If bbox or mask is invalid.
        """
        category_id = int(item["category_id"])
        bbox = self._parse_bbox(item["bounding_box"], width, height)
        bbox_area = self._bbox_area(bbox)

        if bbox_area < self.min_bbox_area:
            LOGGER.debug(
                "Skip %s because bbox area %.2f is smaller than %d",
                instance_id,
                bbox_area,
                self.min_bbox_area,
            )
            return None

        segmentation = item.get("segmentation", [])
        gt_mask = self.polygon_to_mask(segmentation, height, width)

        if self.skip_empty_mask and gt_mask.sum() == 0:
            LOGGER.debug("Skip %s because gt mask is empty.", instance_id)
            return None

        category_info = map_deepfashion2_category(category_id)

        landmarks = self._parse_item_landmarks(
            instance_id=instance_id,
            item=item,
        )

        return {
            "instance_id": instance_id,
            "source_item_id": instance_id,

            # Category fields.
            # category_id is kept for downstream standardized schema.
            "category_id": category_id,
            "raw_category_id": category_info["raw_category_id"],
            "raw_category_name": category_info["raw_category_name"],
            "target_category": category_info["target_category"],
            "target_category_zh": category_info["target_category_zh"],

            # Geometry and mask fields.
            "bbox": bbox,
            "bbox_format": "xyxy",
            "segmentation": segmentation,
            "gt_mask": gt_mask,

            # Landmark fields.
            "landmarks": landmarks,
            "num_landmarks": len(landmarks),
            "num_visible_landmarks": count_visible_landmarks(landmarks),
        }

    @staticmethod
    def _parse_item_landmarks(
        instance_id: str,
        item: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Parse landmark field from a DeepFashion2 item.

        Different annotation versions may use different field names. This
        function tries several common names:

            - landmarks
            - landmark
            - landmark_points

        Args:
            instance_id: Instance ID for logging.
            item: Raw DeepFashion2 item dictionary.

        Returns:
            Structured landmark list. Returns an empty list if no valid
            landmark field is found.
        """
        raw_landmarks = (
            item.get("landmarks")
            or item.get("landmark")
            or item.get("landmark_points")
            or []
        )

        if raw_landmarks in (None, [], {}):
            return []

        try:
            return parse_flat_landmarks(raw_landmarks)
        except (ValueError, TypeError) as error:
            LOGGER.warning(
                "Failed to parse landmarks for %s: %s",
                instance_id,
                error,
            )
            return []

    @staticmethod
    def _parse_bbox(
        bbox: List[float] | tuple[float, ...],
        width: int,
        height: int,
    ) -> List[float]:
        """
        Parse and clip a bounding box to image boundaries.

        DeepFashion2 bounding boxes are expected to be in ``xyxy`` format.

        Args:
            bbox: Bounding box in ``[x1, y1, x2, y2]`` format.
            width: Image width.
            height: Image height.

        Returns:
            Clipped bounding box as list of floats.

        Raises:
            ValueError: If bbox is invalid.
        """
        if len(bbox) != 4:
            raise ValueError(f"Expected bbox with 4 values, got: {bbox}")

        x1, y1, x2, y2 = [float(value) for value in bbox]

        x1 = max(0.0, min(x1, float(width - 1)))
        y1 = max(0.0, min(y1, float(height - 1)))
        x2 = max(0.0, min(x2, float(width - 1)))
        y2 = max(0.0, min(y2, float(height - 1)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")

        return [x1, y1, x2, y2]

    @staticmethod
    def _bbox_area(bbox: List[float]) -> float:
        """
        Compute bounding box area.

        Args:
            bbox: Bounding box in ``[x1, y1, x2, y2]`` format.

        Returns:
            Bounding box area.
        """
        x1, y1, x2, y2 = bbox
        return float((x2 - x1) * (y2 - y1))

    @staticmethod
    def polygon_to_mask(
        segmentation: List[Any],
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Convert polygon segmentation to a binary mask.

        Args:
            segmentation: DeepFashion2 polygon segmentation. It is usually a
                list of polygons, where each polygon is a flattened list of
                coordinates ``[x1, y1, x2, y2, ...]``.
            height: Image height.
            width: Image width.

        Returns:
            Binary mask with shape ``height x width`` and dtype uint8.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        if not segmentation:
            return mask

        polygons = DeepFashion2Parser._normalize_segmentation(segmentation)

        for polygon in polygons:
            if len(polygon) < 6:
                continue

            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            points = np.round(points).astype(np.int32)

            if points.shape[0] >= 3:
                cv2.fillPoly(mask, [points], 1)

        return mask

    @staticmethod
    def _normalize_segmentation(segmentation: List[Any]) -> List[List[float]]:
        """
        Normalize DeepFashion2 segmentation field to a list of polygons.

        Args:
            segmentation: Raw segmentation field.

        Returns:
            List of flattened polygon coordinate lists.
        """
        if not segmentation:
            return []

        if all(isinstance(value, (int, float)) for value in segmentation):
            return [list(map(float, segmentation))]

        polygons: List[List[float]] = []

        for polygon in segmentation:
            if not isinstance(polygon, list):
                continue

            if all(isinstance(value, (int, float)) for value in polygon):
                polygons.append(list(map(float, polygon)))

        return polygons
