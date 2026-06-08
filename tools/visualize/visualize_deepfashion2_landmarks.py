"""
Visualize DeepFashion2 clothing landmarks.

This script reads DeepFashion2 images and annotations, parses clothing
landmarks, and saves landmark visualization images.

Example:
    python tools/visualize_deepfashion2_landmarks.py \
        --config configs/inference/sam_box_prompt.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fashion_vision.data.deepfashion2_parser import DeepFashion2Parser
from fashion_vision.data.landmarks import (
    count_visible_landmarks,
    sanitize_landmarks_for_json,
)
from fashion_vision.utils.config import load_yaml_config, require_config_keys
from fashion_vision.utils.image import read_image_rgb
from fashion_vision.utils.json_io import save_json
from fashion_vision.utils.logger import setup_logger
from fashion_vision.visualization.landmark_visualizer import (
    save_landmark_visualization,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize DeepFashion2 landmarks."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def build_parser(config: Dict[str, Any]) -> DeepFashion2Parser:
    """
    Build DeepFashion2 parser from config.

    Args:
        config: Loaded YAML config.

    Returns:
        DeepFashion2Parser instance.
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


def prepare_landmark_output_dirs(output_root: str | Path) -> Dict[str, Path]:
    """
    Prepare output directories for landmark visualization.

    Args:
        output_root: Output root.

    Returns:
        Output directory dictionary.
    """
    output_root = Path(output_root)

    dirs = {
        "root": output_root,
        "visualizations": output_root / "visualizations",
        "predictions": output_root / "predictions",
        "logs": output_root / "logs",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def sanitize_instance_landmark_record(
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert instance landmark information to JSON record.

    Args:
        instance: Parsed DeepFashion2 instance.

    Returns:
        JSON-serializable instance landmark record.
    """
    landmarks = instance.get("landmarks", [])

    return {
        "instance_id": str(instance.get("instance_id", "unknown")),
        "category_id": int(instance.get("category_id", -1))
        if instance.get("category_id") is not None
        else None,
        "category": str(
            instance.get(
                "target_category",
                instance.get("category", "unknown"),
            )
        ),
        "category_name": str(instance.get("category_name", "")),
        "bbox": [float(value) for value in instance.get("bbox", [])],
        "num_landmarks": int(len(landmarks)),
        "num_visible_landmarks": int(count_visible_landmarks(landmarks)),
        "landmarks": sanitize_landmarks_for_json(landmarks),
    }


def run(config: Dict[str, Any]) -> None:
    """
    Run landmark visualization.

    Args:
        config: Loaded config.
    """
    require_config_keys(config, ["dataset"])

    landmark_cfg = config.get("landmark_visualization", {})
    inference_cfg = config.get("inference", {})

    output_root = landmark_cfg.get(
        "output_root",
        "outputs/deepfashion2_landmarks",
    )
    output_dirs = prepare_landmark_output_dirs(output_root)

    logger = setup_logger(
        name="deepfashion2_landmark_visualization",
        log_file=output_dirs["logs"] / "run.log",
        level=config.get("logging", {}).get("level", "INFO"),
    )

    parser = build_parser(config)
    image_ids = parser.list_image_ids()

    max_images = inference_cfg.get("max_images")
    random_sample = bool(inference_cfg.get("random_sample", False))
    random_seed = int(inference_cfg.get("random_seed", 42))

    if random_sample:
        rng = np.random.default_rng(random_seed)
        image_ids = list(rng.permutation(image_ids))
        logger.info("Random sampling enabled. random_seed=%d", random_seed)

    if max_images is not None:
        image_ids = image_ids[: int(max_images)]

    logger.info("Start DeepFashion2 landmark visualization.")
    logger.info("Number of images to process: %d", len(image_ids))
    logger.info("Output root: %s", output_dirs["root"])

    processed_images = 0
    processed_instances = 0
    total_visible_landmarks = 0
    image_index: List[Dict[str, Any]] = []

    for image_id in tqdm(image_ids, desc="Visualizing landmarks"):
        try:
            sample = parser.load_sample(image_id)
            image_rgb = read_image_rgb(sample["image_path"])
        except (FileNotFoundError, ValueError, KeyError) as error:
            logger.warning(
                "Skip image %s due to loading error: %s",
                image_id,
                error,
            )
            continue

        instances = sample.get("instances", [])

        if not instances:
            logger.info("No instances for image %s.", image_id)
            continue

        instance_records = []
        image_visible_landmarks = 0

        for instance in instances:
            record = sanitize_instance_landmark_record(instance)
            instance_records.append(record)
            processed_instances += 1
            image_visible_landmarks += record["num_visible_landmarks"]

        total_visible_landmarks += image_visible_landmarks

        visualization_path = (
            output_dirs["visualizations"] / f"{image_id}_landmarks.jpg"
        )
        save_landmark_visualization(
            image_rgb=image_rgb,
            instances=instances,
            output_path=visualization_path,
            draw_bbox=bool(landmark_cfg.get("draw_bbox", True)),
            draw_label=bool(landmark_cfg.get("draw_label", True)),
            draw_index=bool(landmark_cfg.get("draw_index", True)),
            visible_only=bool(landmark_cfg.get("visible_only", True)),
        )

        prediction_path = (
            output_dirs["predictions"] / f"{image_id}_landmarks.json"
        )
        prediction_record = {
            "image_id": str(image_id),
            "image_path": str(sample["image_path"]),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "num_instances": int(len(instance_records)),
            "num_visible_landmarks": int(image_visible_landmarks),
            "instances": instance_records,
            "visualization": str(visualization_path),
        }
        save_json(prediction_record, prediction_path)

        image_index.append(
            {
                "image_id": str(image_id),
                "image_path": str(sample["image_path"]),
                "prediction_json": str(prediction_path),
                "visualization": str(visualization_path),
                "num_instances": int(len(instance_records)),
                "num_visible_landmarks": int(image_visible_landmarks),
            }
        )

        processed_images += 1

    index_record = {
        "task": "deepfashion2_landmark_visualization",
        "output_root": str(output_dirs["root"]),
        "num_images": int(processed_images),
        "num_instances": int(processed_instances),
        "num_visible_landmarks": int(total_visible_landmarks),
        "images": image_index,
    }

    index_path = output_dirs["root"] / "index.json"
    save_json(index_record, index_path)

    logger.info("Finished DeepFashion2 landmark visualization.")
    logger.info("Processed images: %d", processed_images)
    logger.info("Processed instances: %d", processed_instances)
    logger.info("Visible landmarks: %d", total_visible_landmarks)
    logger.info("Index JSON saved to: %s", index_path)


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    config = load_yaml_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
