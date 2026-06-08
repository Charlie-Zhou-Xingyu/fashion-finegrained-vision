"""
Build DeepFashion2 landmark predictor dataset index.

This script converts original DeepFashion2 annotation JSON files into a
JSONL dataset for garment landmark prediction.

Input:
    DeepFashion2 original directory:
        deepfashion2/
        ├── train/
        │   ├── image/
        │   └── annos/
        └── validation/
            ├── image/
            └── annos/

Output:
    data/processed/deepfashion2_landmarks/train.jsonl
    data/processed/deepfashion2_landmarks/validation.jsonl
    data/processed/deepfashion2_landmarks/summary.json

Each JSONL line corresponds to one garment instance.

Example:
    python tools/build_deepfashion2_landmark_dataset.py ^
      --deepfashion2-root D:/Aliintern/fashion-ai-data/deepfashion2 ^
      --output-dir data/processed/deepfashion2_landmarks ^
      --splits train,validation
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DeepFashion2 landmark dataset JSONL index."
    )
    parser.add_argument(
        "--deepfashion2-root",
        type=str,
        required=True,
        help="Root directory of DeepFashion2 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for JSONL files.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated splits to process, e.g. train,validation.",
    )
    parser.add_argument(
        "--min-present-landmarks",
        type=int,
        default=2,
        help="Minimum number of visibility>0 landmarks required.",
    )
    parser.add_argument(
        "--allow-missing-image",
        action="store_true",
        help="Keep samples even if image file is missing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_image_path(image_dir: Path, image_id: str) -> Optional[Path]:
    """
    Find image file by image id.

    Args:
        image_dir: Image directory.
        image_id: Image id without extension.

    Returns:
        Image path or None.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate

    return None


def normalize_bbox_xyxy(raw_bbox: Any) -> Optional[List[float]]:
    """
    Normalize DeepFashion2 bounding_box to xyxy.

    DeepFashion2 usually stores bounding_box as:
        [x1, y1, x2, y2]

    Args:
        raw_bbox: Raw bbox.

    Returns:
        bbox xyxy or None.
    """
    if not isinstance(raw_bbox, list):
        return None

    if len(raw_bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    except Exception:
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def normalize_landmarks(raw_landmarks: Any) -> List[Dict[str, Any]]:
    """
    Normalize raw DeepFashion2 landmarks.

    DeepFashion2 landmark format:
        [x1, y1, v1, x2, y2, v2, ...]

    Args:
        raw_landmarks: Raw landmark list.

    Returns:
        Normalized landmark dict list.
    """
    if not isinstance(raw_landmarks, list):
        return []

    if len(raw_landmarks) == 0:
        return []

    if len(raw_landmarks) % 3 != 0:
        return []

    landmarks: List[Dict[str, Any]] = []
    num_points = len(raw_landmarks) // 3

    for i in range(num_points):
        x = raw_landmarks[i * 3]
        y = raw_landmarks[i * 3 + 1]
        visibility = raw_landmarks[i * 3 + 2]

        try:
            landmark = {
                "index": i + 1,
                "x": float(x),
                "y": float(y),
                "visibility": int(visibility),
            }
        except Exception:
            continue

        landmarks.append(landmark)

    return landmarks


def count_landmarks(landmarks: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Count landmarks.

    Args:
        landmarks: Normalized landmarks.

    Returns:
        num_landmarks, num_visible, num_present
    """
    num_landmarks = len(landmarks)
    num_visible = 0
    num_present = 0

    for landmark in landmarks:
        visibility = int(landmark.get("visibility", 0))
        x = float(landmark.get("x", 0.0))
        y = float(landmark.get("y", 0.0))

        if visibility == 2 and x > 0 and y > 0:
            num_visible += 1

        if visibility > 0 and x > 0 and y > 0:
            num_present += 1

    return num_landmarks, num_visible, num_present


def iter_deepfashion2_items(
    split: str,
    deepfashion2_root: Path,
    min_present_landmarks: int,
    allow_missing_image: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Iterate one DeepFashion2 split and build sample records.

    Args:
        split: train or validation.
        deepfashion2_root: Dataset root.
        min_present_landmarks: Minimum present landmarks.
        allow_missing_image: Whether to keep missing image samples.

    Returns:
        samples, stats
    """
    split_dir = deepfashion2_root / split
    anno_dir = split_dir / "annos"
    image_dir = split_dir / "image"

    if not anno_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {anno_dir}")

    if not image_dir.exists() and not allow_missing_image:
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    anno_paths = sorted(anno_dir.glob("*.json"))

    samples: List[Dict[str, Any]] = []

    stats: Dict[str, Any] = {
        "split": split,
        "num_anno_files": len(anno_paths),
        "num_samples": 0,
        "num_skipped_missing_image": 0,
        "num_skipped_invalid_bbox": 0,
        "num_skipped_no_landmarks": 0,
        "num_skipped_few_landmarks": 0,
        "category_counter": Counter(),
        "category_landmark_count": defaultdict(Counter),
    }

    for anno_path in anno_paths:
        image_id = anno_path.stem
        image_path = find_image_path(image_dir=image_dir, image_id=image_id)

        if image_path is None and not allow_missing_image:
            stats["num_skipped_missing_image"] += 1
            continue

        anno = load_json(anno_path)

        for key, value in anno.items():
            if not str(key).startswith("item"):
                continue

            if not isinstance(value, dict):
                continue

            instance_id = str(key)
            category_name = str(value.get("category_name", "")).lower().strip()
            category_id = value.get("category_id", None)

            bbox = normalize_bbox_xyxy(value.get("bounding_box"))
            if bbox is None:
                stats["num_skipped_invalid_bbox"] += 1
                continue

            landmarks = normalize_landmarks(value.get("landmarks"))
            if not landmarks:
                stats["num_skipped_no_landmarks"] += 1
                continue

            num_landmarks, num_visible, num_present = count_landmarks(landmarks)

            if num_present < min_present_landmarks:
                stats["num_skipped_few_landmarks"] += 1
                continue

            sample = {
                "image_id": image_id,
                "image_path": str(image_path) if image_path is not None else None,
                "anno_path": str(anno_path),
                "split": split,
                "instance_id": instance_id,
                "category_name": category_name,
                "category_id": category_id,
                "bbox_xyxy": bbox,
                "landmarks": landmarks,
                "num_landmarks": num_landmarks,
                "num_visible_landmarks": num_visible,
                "num_present_landmarks": num_present,
            }

            samples.append(sample)

            stats["category_counter"][category_name] += 1
            stats["category_landmark_count"][category_name][num_landmarks] += 1

    stats["num_samples"] = len(samples)

    # Convert counters to normal dicts for JSON serialization.
    stats["category_counter"] = dict(stats["category_counter"])
    stats["category_landmark_count"] = {
        category: dict(counter)
        for category, counter in stats["category_landmark_count"].items()
    }

    return samples, stats


def write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
    """
    Write samples to JSONL.

    Args:
        samples: Sample records.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    deepfashion2_root = Path(args.deepfashion2_root)
    output_dir = Path(args.output_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: Dict[str, Any] = {
        "deepfashion2_root": str(deepfashion2_root),
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split in splits:
        print(f"[INFO] Processing split: {split}")

        samples, stats = iter_deepfashion2_items(
            split=split,
            deepfashion2_root=deepfashion2_root,
            min_present_landmarks=int(args.min_present_landmarks),
            allow_missing_image=bool(args.allow_missing_image),
        )

        output_jsonl = output_dir / f"{split}.jsonl"
        write_jsonl(samples=samples, path=output_jsonl)

        all_stats["splits"][split] = stats

        print(f"[INFO] Split: {split}")
        print(f"[INFO] Annotation files: {stats['num_anno_files']}")
        print(f"[INFO] Samples: {stats['num_samples']}")
        print(f"[INFO] Output: {output_jsonl}")

    summary_path = output_dir / "summary.json"
    save_json(all_stats, summary_path)

    print("[INFO] Done.")
    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
