"""
Export DeepFashion2 landmark inspection images by raw category.

This script scans DeepFashion2 annotations, groups garment instances by
raw_category_name, and exports landmark visualization images for manual
inspection.

Purpose:
    Build reliable landmark-region mapping for local region localization,
    e.g. neckline / cuff / hem / waist / leg opening.

Example:
    python tools/export_landmark_inspection_by_category.py ^
        --config configs/inference/sam_box_prompt.yaml ^
        --output-root outputs/landmark_inspection ^
        --max-per-category 100

Optional:
    python tools/export_landmark_inspection_by_category.py ^
        --config configs/inference/sam_box_prompt.yaml ^
        --categories "short sleeve top,long sleeve top,trousers,skirt" ^
        --max-per-category 100 ^
        --random-sample
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fashion_vision.data.deepfashion2_parser import DeepFashion2Parser
from fashion_vision.utils.config import load_yaml_config, require_config_keys
from fashion_vision.utils.image import read_image_rgb
from fashion_vision.utils.json_io import save_json
from fashion_vision.utils.logger import setup_logger


DEFAULT_TARGET_RAW_CATEGORIES = [
    "short sleeve top",
    "long sleeve top",
    "vest",
    "sling",
    "short sleeve outwear",
    "long sleeve outwear",
    "vest dress",
    "sling dress",
    "short sleeve dress",
    "long sleeve dress",
    "trousers",
    "shorts",
    "skirt",
]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Export DeepFashion2 landmark inspection images by category."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/landmark_inspection",
        help="Output root directory.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=100,
        help="Maximum number of instances to export per raw category.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to scan.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help=(
            "Optional comma-separated raw category names. "
            "If omitted, use default DeepFashion2 categories."
        ),
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomize image order before scanning.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--visible-only",
        action="store_true",
        help="Only draw strictly visible landmarks with visibility == 2.",
    )
    parser.add_argument(
        "--draw-occluded",
        action="store_true",
        help="Draw occluded landmarks with visibility == 1 as hollow yellow points.",
    )
    return parser.parse_args()


def normalize_category_name(name: str) -> str:
    """
    Normalize category name for folder/file names.

    Args:
        name: Raw category name.

    Returns:
        Normalized name.
    """
    return (
        name.lower()
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def parse_category_filter(raw: Optional[str]) -> List[str]:
    """
    Parse category filter argument.

    Args:
        raw: Comma-separated raw category names.

    Returns:
        Category list.
    """
    if raw is None:
        return DEFAULT_TARGET_RAW_CATEGORIES

    categories = [item.strip() for item in raw.split(",") if item.strip()]
    if not categories:
        return DEFAULT_TARGET_RAW_CATEGORIES

    return categories


def prepare_output_dirs(output_root: str | Path) -> Dict[str, Path]:
    """
    Prepare output dirs.

    Args:
        output_root: Output root.

    Returns:
        Directory dict.
    """
    output_root = Path(output_root)

    dirs = {
        "root": output_root,
        "metrics": output_root / "metrics",
        "logs": output_root / "logs",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def build_parser(config: Dict[str, Any]) -> DeepFashion2Parser:
    """
    Build DeepFashion2 parser.

    Args:
        config: Loaded config.

    Returns:
        DeepFashion2Parser.
    """
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})

    return DeepFashion2Parser(
        root=dataset_cfg["root"],
        split=dataset_cfg.get("split", "validation"),
        image_dir=dataset_cfg.get("image_dir"),
        annotation_dir=dataset_cfg.get("annotation_dir"),
        min_bbox_area=int(inference_cfg.get("min_bbox_area", 1)),
        skip_empty_mask=bool(inference_cfg.get("skip_empty_mask", True)),
    )


def get_raw_category_name(instance: Dict[str, Any]) -> str:
    """
    Get raw category name from instance.

    Args:
        instance: Instance dict.

    Returns:
        Raw category name.
    """
    return str(
        instance.get(
            "raw_category_name",
            instance.get(
                "category_name",
                instance.get("raw_category", "unknown"),
            ),
        )
    )


def get_target_category(instance: Dict[str, Any]) -> str:
    """
    Get target category.

    Args:
        instance: Instance dict.

    Returns:
        Target category.
    """
    return str(
        instance.get(
            "target_category",
            instance.get("category", "unknown"),
        )
    )


def draw_text_with_background(
    image_bgr: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.55,
    text_color: Tuple[int, int, int] = (0, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Draw text with black background.

    Args:
        image_bgr: BGR image.
        text: Text.
        origin: Bottom-left text origin.
        font_scale: Font scale.
        text_color: Text color.
        bg_color: Background color.
    """
    x, y = origin
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        image_bgr,
        (x, y - text_h - baseline - 4),
        (x + text_w + 6, y + baseline + 4),
        bg_color,
        -1,
    )
    cv2.putText(
        image_bgr,
        text,
        (x + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_landmarks_on_image(
    image_rgb: np.ndarray,
    instance: Dict[str, Any],
    visible_only: bool = False,
    draw_occluded: bool = True,
) -> np.ndarray:
    """
    Draw bbox and landmarks on image.

    Args:
        image_rgb: RGB image.
        instance: Instance dict.
        visible_only: Whether to only draw visibility == 2.
        draw_occluded: Whether to draw visibility == 1.

    Returns:
        Visualization RGB image.
    """
    image_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    bbox = instance.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        label = (
            f"{instance.get('instance_id', 'item')}: "
            f"{get_raw_category_name(instance)} -> {get_target_category(instance)}"
        )
        draw_text_with_background(
            image_bgr=image_bgr,
            text=label,
            origin=(max(0, x1), max(22, y1)),
        )

    landmarks = instance.get("landmarks", [])

    for point in landmarks:
        visibility = int(point.get("visibility", 0))
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        index = int(point.get("index", 0))

        if visibility <= 0 or x <= 0 or y <= 0:
            continue

        if visible_only and visibility != 2:
            continue

        if visibility == 1 and not draw_occluded:
            continue

        px = int(round(x))
        py = int(round(y))

        if visibility == 2:
            point_color = (0, 0, 255)
            cv2.circle(image_bgr, (px, py), 4, point_color, -1)
            cv2.circle(image_bgr, (px, py), 6, (255, 255, 255), 1)
            text_color = (0, 0, 255)
        elif visibility == 1:
            point_color = (0, 255, 255)
            cv2.circle(image_bgr, (px, py), 5, point_color, 2)
            text_color = (0, 255, 255)
        else:
            continue

        cv2.putText(
            image_bgr,
            str(index),
            (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(image_rgb: np.ndarray, output_path: str | Path) -> None:
    """
    Save RGB image.

    Args:
        image_rgb: RGB image.
        output_path: Output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def sanitize_landmarks_for_json(
    landmarks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Make landmarks JSON serializable.

    Args:
        landmarks: Landmark list.

    Returns:
        JSON serializable list.
    """
    result = []

    for point in landmarks:
        result.append(
            {
                "index": int(point.get("index", 0)),
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "visibility": int(point.get("visibility", 0)),
                "present": bool(point.get("present", False)),
                "visible": bool(point.get("visible", False)),
                "occluded": bool(point.get("occluded", False)),
                "absent": bool(point.get("absent", False)),
            }
        )

    return result


def category_quota_reached(
    counts: Dict[str, int],
    category_name: str,
    max_per_category: int,
) -> bool:
    """
    Check if category quota is reached.

    Args:
        counts: Export counts.
        category_name: Raw category name.
        max_per_category: Max per category.

    Returns:
        Whether quota reached.
    """
    return counts.get(category_name, 0) >= max_per_category


def all_quotas_reached(
    counts: Dict[str, int],
    categories: List[str],
    max_per_category: int,
) -> bool:
    """
    Check if all category quotas are reached.

    Args:
        counts: Export counts.
        categories: Target categories.
        max_per_category: Max per category.

    Returns:
        Whether all quotas reached.
    """
    for category in categories:
        if counts.get(category, 0) < max_per_category:
            return False

    return True


def run(args: argparse.Namespace) -> None:
    """
    Run export.

    Args:
        args: Parsed args.
    """
    config = load_yaml_config(args.config)
    require_config_keys(config, ["dataset"])

    output_dirs = prepare_output_dirs(args.output_root)

    logger = setup_logger(
        name="export_landmark_inspection",
        log_file=output_dirs["logs"] / "export_landmarks.log",
        level=config.get("logging", {}).get("level", "INFO"),
    )

    target_categories = parse_category_filter(args.categories)
    target_category_set = set(target_categories)

    logger.info("Target raw categories: %s", target_categories)
    logger.info("Max per category: %d", args.max_per_category)

    parser = build_parser(config)
    image_ids = parser.list_image_ids()

    if args.random_sample:
        rng = random.Random(args.random_seed)
        rng.shuffle(image_ids)

    if args.max_images is not None:
        image_ids = image_ids[: int(args.max_images)]

    counts: Dict[str, int] = defaultdict(int)
    skipped_counts: Dict[str, int] = defaultdict(int)
    exported_records: List[Dict[str, Any]] = []

    for image_id in tqdm(image_ids, desc="Exporting landmark inspection"):
        if all_quotas_reached(
            counts=counts,
            categories=target_categories,
            max_per_category=int(args.max_per_category),
        ):
            logger.info("All category quotas reached. Stop early.")
            break

        try:
            sample = parser.load_sample(image_id)
        except Exception as error:
            logger.warning("Skip image %s due to error: %s", image_id, error)
            continue

        image_path = sample.get("image_path")
        if image_path is None:
            logger.warning("Skip image %s due to missing image_path.", image_id)
            continue

        try:
            image_rgb = read_image_rgb(str(image_path))
        except Exception as error:
            logger.warning("Failed to read image %s: %s", image_path, error)
            continue

        for instance in sample.get("instances", []):
            raw_category_name = get_raw_category_name(instance)

            if raw_category_name not in target_category_set:
                skipped_counts[raw_category_name] += 1
                continue

            if category_quota_reached(
                counts=counts,
                category_name=raw_category_name,
                max_per_category=int(args.max_per_category),
            ):
                continue

            landmarks = instance.get("landmarks", [])

            if not landmarks:
                skipped_counts[f"{raw_category_name}_no_landmarks"] += 1
                continue

            instance_id = str(instance.get("instance_id", "unknown"))
            safe_category = normalize_category_name(raw_category_name)

            category_dir = output_dirs["root"] / safe_category
            category_dir.mkdir(parents=True, exist_ok=True)

            output_name = f"{image_id}_{instance_id}_{safe_category}.jpg"
            output_path = category_dir / output_name

            vis_rgb = draw_landmarks_on_image(
                image_rgb=image_rgb,
                instance=instance,
                visible_only=bool(args.visible_only),
                draw_occluded=bool(args.draw_occluded),
            )

            save_image_rgb(vis_rgb, output_path)

            record = {
                "image_id": str(image_id),
                "image_path": str(image_path),
                "instance_id": instance_id,
                "raw_category_name": raw_category_name,
                "target_category": get_target_category(instance),
                "bbox": [
                    float(value)
                    for value in instance.get("bbox", [])
                ],
                "num_landmarks": int(len(landmarks)),
                "num_present_landmarks": int(
                    sum(
                        1
                        for point in landmarks
                        if bool(point.get("present", False))
                    )
                ),
                "num_visible_landmarks": int(
                    sum(
                        1
                        for point in landmarks
                        if bool(point.get("visible", False))
                    )
                ),
                "visualization": str(output_path),
                "landmarks": sanitize_landmarks_for_json(landmarks),
            }

            exported_records.append(record)
            counts[raw_category_name] += 1

    summary = {
        "task": "export_landmark_inspection_by_category",
        "output_root": str(output_dirs["root"]),
        "max_per_category": int(args.max_per_category),
        "target_categories": target_categories,
        "counts": dict(counts),
        "skipped_counts": dict(skipped_counts),
        "num_exported": int(len(exported_records)),
        "visible_only": bool(args.visible_only),
        "draw_occluded": bool(args.draw_occluded),
        "records": exported_records,
    }

    index_path = output_dirs["root"] / "index.json"
    counts_path = output_dirs["metrics"] / "category_counts.json"

    save_json(summary, index_path)
    save_json(
        {
            "counts": dict(counts),
            "skipped_counts": dict(skipped_counts),
            "num_exported": int(len(exported_records)),
        },
        counts_path,
    )

    logger.info("Finished exporting landmark inspection images.")
    logger.info("Num exported: %d", len(exported_records))
    logger.info("Counts: %s", dict(counts))
    logger.info("Index saved to: %s", index_path)
    logger.info("Counts saved to: %s", counts_path)


def main() -> None:
    """
    Main entry.
    """
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
