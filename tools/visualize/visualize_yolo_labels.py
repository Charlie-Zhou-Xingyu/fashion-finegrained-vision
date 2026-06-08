from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_CLASS_NAMES = ["top", "pants", "skirt", "outwear", "dress"]

CLASS_COLORS = {
    0: (255, 80, 80),      # top - red
    1: (80, 180, 255),     # pants - blue
    2: (255, 180, 80),     # skirt - orange
    3: (160, 100, 255),    # outwear - purple
    4: (80, 220, 120),     # dress - green
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO detection labels on images."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="YOLO dataset directory, e.g. data/processed/deepfashion2_yolo_debug",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "validation", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualize_yolo_labels",
        help="Output directory for visualization images.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of random images to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include images with empty or missing label files.",
    )
    return parser.parse_args()


def normalize_split(split: str) -> str:
    if split == "validation":
        return "val"
    return split


def parse_class_names(value: str) -> List[str]:
    names = [x.strip() for x in value.split(",") if x.strip()]
    if not names:
        return DEFAULT_CLASS_NAMES
    return names


def find_image_files(image_dir: Path) -> List[Path]:
    image_files: List[Path] = []
    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(image_dir.glob(pattern))
    return sorted(image_files)


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Returns list of:
      (class_id, x_center, y_center, width, height)
    """
    if not label_path.exists():
        return []

    lines = label_path.read_text(encoding="utf-8").splitlines()
    labels: List[Tuple[int, float, float, float, float]] = []

    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            print(f"[WARN] Invalid label line: {label_path}:{line_idx}: {line}")
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            print(f"[WARN] Failed to parse label line: {label_path}:{line_idx}: {line}")
            continue

        labels.append((class_id, x_center, y_center, width, height))

    return labels


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    cx = x_center * image_width
    cy = y_center * image_height
    w = box_width * image_width
    h = box_height * image_height

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return x1, y1, x2, y2


def clip_xyxy(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width - 1))
    y2 = max(0, min(int(round(y2)), image_height - 1))

    return x1, y1, x2, y2


def get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def draw_label_box(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = bbox

    # bbox rectangle
    line_width = 3
    for offset in range(line_width):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color,
        )

    # text background
    try:
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except Exception:
        text_w, text_h = 120, 20

    bg_x1 = x1
    bg_y1 = max(0, y1 - text_h - 6)
    bg_x2 = min(x1 + text_w + 8, x2)
    bg_y2 = y1

    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
    draw.text((bg_x1 + 4, bg_y1 + 2), text, fill=(255, 255, 255), font=font)


def visualize_one_image(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: List[str],
) -> Dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    labels = read_yolo_label(label_path)

    draw = ImageDraw.Draw(image)
    font = get_font()

    num_valid = 0
    num_invalid = 0

    for label in labels:
        class_id, x_center, y_center, box_width, box_height = label

        if not (0 <= class_id < len(class_names)):
            num_invalid += 1
            continue

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 <= box_width <= 1.0
            and 0.0 <= box_height <= 1.0
        ):
            num_invalid += 1
            continue

        bbox_float = yolo_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            box_width=box_width,
            box_height=box_height,
            image_width=image_width,
            image_height=image_height,
        )
        bbox = clip_xyxy(bbox_float, image_width=image_width, image_height=image_height)

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            num_invalid += 1
            continue

        class_name = class_names[class_id]
        color = CLASS_COLORS.get(class_id, (255, 255, 0))
        text = f"{class_id}:{class_name}"

        draw_label_box(draw, bbox=bbox, text=text, color=color, font=font)

        num_valid += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "output_path": str(output_path),
        "num_labels": len(labels),
        "num_valid": num_valid,
        "num_invalid": num_invalid,
        "image_width": image_width,
        "image_height": image_height,
    }


def main() -> None:
    args = parse_args()

    dataset_dir: Path = args.dataset_dir
    split = normalize_split(args.split)
    output_dir: Path = args.output_dir
    class_names = parse_class_names(args.class_names)

    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    image_files = find_image_files(image_dir)

    if not args.include_empty:
        image_files = [
            image_path
            for image_path in image_files
            if (label_dir / f"{image_path.stem}.txt").exists()
            and (label_dir / f"{image_path.stem}.txt").read_text(encoding="utf-8").strip()
        ]

    if not image_files:
        raise RuntimeError(f"No image files found for split={split} in {image_dir}")

    random.seed(args.seed)
    sample_count = min(args.num_samples, len(image_files))
    sampled_images = random.sample(image_files, sample_count)

    output_split_dir = output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset_dir: {dataset_dir}")
    print(f"[INFO] split: {split}")
    print(f"[INFO] images found: {len(image_files)}")
    print(f"[INFO] samples: {sample_count}")
    print(f"[INFO] output_dir: {output_split_dir}")

    summaries: List[Dict[str, object]] = []

    for idx, image_path in enumerate(sampled_images, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        output_path = output_split_dir / f"{image_path.stem}_yolo_vis.jpg"

        summary = visualize_one_image(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            class_names=class_names,
        )
        summaries.append(summary)

        print(
            f"[{idx:03d}/{sample_count:03d}] "
            f"{image_path.name} -> {output_path} "
            f"labels={summary['num_labels']} valid={summary['num_valid']} invalid={summary['num_invalid']}"
        )

    print("[INFO] Visualization finished.")


if __name__ == "__main__":
    main()
