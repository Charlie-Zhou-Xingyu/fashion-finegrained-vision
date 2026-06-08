"""
Visualize garment landmark predictor outputs.

This script loads a trained landmark predictor checkpoint and draws:
    - GT landmarks in green/orange
    - predicted landmarks in red

Example:
    python tools/visualize_landmark_predictions.py ^
      --jsonl data/processed/deepfashion2_landmarks/validation.jsonl ^
      --checkpoint outputs/landmark_predictor_resnet18/best.pt ^
      --output-dir outputs/landmark_prediction_vis ^
      --num-samples 50 ^
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
from fashion_vision.landmarks.model import build_landmark_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize garment landmark predictor outputs."
    )
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-landmarks", type=int, default=39)
    parser.add_argument("--pad-ratio", type=float, default=0.05)

    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--category", type=str, default="")
    parser.add_argument(
        "--draw-invalid-pred",
        action="store_true",
        help="Draw predicted points even when GT landmark is invalid.",
    )

    return parser.parse_args()


def tensor_image_to_uint8_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    max_landmarks: int,
    device: torch.device,
) -> torch.nn.Module:
    model = build_landmark_model(
        model_name=model_name,
        max_landmarks=max_landmarks,
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def draw_gt_and_pred(
    sample: Dict[str, Any],
    pred_landmarks: np.ndarray,
    draw_invalid_pred: bool = False,
) -> np.ndarray:
    """
    Draw GT and predicted landmarks.

    GT:
        visible: green
        occluded: orange
    Pred:
        red circle + red line from GT to pred if GT valid
    """
    image_rgb = tensor_image_to_uint8_rgb(sample["image"])
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    h, w = image_bgr.shape[:2]

    gt_landmarks = sample["landmarks"].detach().cpu().numpy()
    visibility = sample["visibility"].detach().cpu().numpy()
    valid = sample["valid"].detach().cpu().numpy()

    for i in range(gt_landmarks.shape[0]):
        v = int(visibility[i])
        is_valid = float(valid[i]) > 0.5

        if not is_valid and not draw_invalid_pred:
            continue

        pred_x = int(round(float(pred_landmarks[i, 0]) * w))
        pred_y = int(round(float(pred_landmarks[i, 1]) * h))

        pred_x = max(0, min(w - 1, pred_x))
        pred_y = max(0, min(h - 1, pred_y))

        # Predicted point: red
        cv2.circle(image_bgr, (pred_x, pred_y), 3, (0, 0, 255), -1)

        if is_valid and v > 0:
            gt_x = int(round(float(gt_landmarks[i, 0]) * w))
            gt_y = int(round(float(gt_landmarks[i, 1]) * h))

            gt_x = max(0, min(w - 1, gt_x))
            gt_y = max(0, min(h - 1, gt_y))

            if v == 2:
                gt_color = (0, 255, 0)
            else:
                gt_color = (0, 165, 255)

            cv2.circle(image_bgr, (gt_x, gt_y), 4, gt_color, 1)
            cv2.line(image_bgr, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 255), 1)

            cv2.putText(
                image_bgr,
                str(i + 1),
                (gt_x + 4, gt_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                gt_color,
                1,
                cv2.LINE_AA,
            )

    title = f"{sample['image_id']} {sample['instance_id']} {sample['category_name']}"
    legend = "GT visible=green, GT occluded=orange, Pred=red"

    cv2.rectangle(image_bgr, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(
        image_bgr,
        title,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        legend,
        (5, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image_bgr


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    categories = None
    if args.category.strip():
        categories = [args.category.strip()]

    dataset = DeepFashion2LandmarkDataset(
        jsonl_path=args.jsonl,
        image_size=args.image_size,
        max_landmarks=args.max_landmarks,
        pad_ratio=args.pad_ratio,
        categories=categories,
    )

    print(f"[INFO] Dataset size: {len(dataset)}")

    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        max_landmarks=args.max_landmarks,
        device=device,
    )

    random.seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: min(args.num_samples, len(indices))]

    for rank, index in enumerate(indices):
        sample = dataset[index]

        image = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image)[0].detach().cpu().numpy()

        vis = draw_gt_and_pred(
            sample=sample,
            pred_landmarks=pred,
            draw_invalid_pred=bool(args.draw_invalid_pred),
        )

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
