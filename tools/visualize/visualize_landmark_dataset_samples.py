"""
Visualize DeepFashion2 landmark dataset samples.

This script checks whether:
    1. bbox crop is correct
    2. landmark coordinate normalization is correct
    3. visible / occluded points are displayed properly

Example:
    python tools/visualize_landmark_dataset_samples.py ^
      --jsonl data/processed/deepfashion2_landmarks/validation.jsonl ^
      --output-dir outputs/landmark_dataset_vis ^
      --num-samples 30 ^
      --image-size 256
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

from fashion_vision.landmarks.dataset import DeepFashion2LandmarkDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DeepFashion2 landmark dataset samples."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to train.jsonl or validation.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output visualization directory.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Dataset crop image size.",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.05,
        help="Crop bbox padding ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Optional category filter, e.g. 'short sleeve top'.",
    )
    return parser.parse_args()


def tensor_image_to_uint8_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert image tensor [3,H,W] to uint8 RGB image.

    Args:
        image_tensor: Tensor image.

    Returns:
        RGB uint8 image.
    """
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def draw_landmarks(sample: Dict[str, Any]) -> np.ndarray:
    """
    Draw landmarks on crop image.

    Args:
        sample: Dataset sample.

    Returns:
        Visualization BGR image.
    """
    image_rgb = tensor_image_to_uint8_rgb(sample["image"])
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    h, w = image_bgr.shape[:2]

    landmarks = sample["landmarks"].detach().cpu().numpy()
    visibility = sample["visibility"].detach().cpu().numpy()
    valid = sample["valid"].detach().cpu().numpy()

    for i in range(landmarks.shape[0]):
        v = int(visibility[i])
        is_valid = float(valid[i]) > 0.5

        if v <= 0 or not is_valid:
            continue

        x = int(round(float(landmarks[i, 0]) * w))
        y = int(round(float(landmarks[i, 1]) * h))

        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        # visibility == 2: green, visibility == 1: orange
        if v == 2:
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)

        cv2.circle(image_bgr, (x, y), 3, color, -1)
        cv2.putText(
            image_bgr,
            str(i + 1),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    title = f"{sample['image_id']} {sample['instance_id']} {sample['category_name']}"
    subtitle = (
        f"present={sample['num_present_landmarks']} "
        f"visible={sample['num_visible_landmarks']}"
    )

    cv2.rectangle(image_bgr, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(
        image_bgr,
        title,
        (5, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        subtitle,
        (5, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image_bgr


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = None
    if args.category.strip():
        categories = [args.category.strip()]

    dataset = DeepFashion2LandmarkDataset(
        jsonl_path=args.jsonl,
        image_size=args.image_size,
        pad_ratio=args.pad_ratio,
        categories=categories,
    )

    print(f"[INFO] Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check jsonl path or category filter.")

    random.seed(args.seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: min(args.num_samples, len(indices))]

    for rank, index in enumerate(indices):
        sample = dataset[index]
        vis = draw_landmarks(sample)

        safe_category = sample["category_name"].replace(" ", "_")
        filename = (
            f"{rank:03d}_"
            f"{sample['image_id']}_"
            f"{sample['instance_id']}_"
            f"{safe_category}.jpg"
        )

        output_path = output_dir / filename
        cv2.imwrite(str(output_path), vis)

    print(f"[INFO] Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
