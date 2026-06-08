"""
Summarize DeepFashion2 landmark statistics.

This script scans DeepFashion2 annotations and reports landmark counts by
target category and raw category. It helps prepare local-region localization
rules for neckline, cuff, hem, waist, shoulder, etc.

Example:
    python tools/summarize_deepfashion2_landmarks.py \
        --config configs/inference/sam_box_prompt.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fashion_vision.data.deepfashion2_parser import DeepFashion2Parser
from fashion_vision.utils.config import load_yaml_config, require_config_keys
from fashion_vision.utils.json_io import save_json
from fashion_vision.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Summarize DeepFashion2 landmark statistics."
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


def prepare_output_dirs(output_root: str | Path) -> Dict[str, Path]:
    """
    Prepare output dirs.

    Args:
        output_root: Output root.

    Returns:
        Dict of dirs.
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


def _safe_mean(values: List[float]) -> float:
    """
    Safe mean.

    Args:
        values: Numeric list.

    Returns:
        Mean value or 0.
    """
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_median(values: List[float]) -> float:
    """
    Safe median.

    Args:
        values: Numeric list.

    Returns:
        Median value or 0.
    """
    if not values:
        return 0.0
    return float(np.median(values))


def summarize_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize a list of instance records.

    Args:
        records: Instance-level landmark records.

    Returns:
        Summary dictionary.
    """
    num_landmarks = [record["num_landmarks"] for record in records]
    num_visible = [record["num_visible_landmarks"] for record in records]
    num_present = [record["num_present_landmarks"] for record in records]

    index_counter: Dict[str, int] = defaultdict(int)
    visible_index_counter: Dict[str, int] = defaultdict(int)
    present_index_counter: Dict[str, int] = defaultdict(int)

    for record in records:
        for point in record["landmarks"]:
            index = str(point["index"])
            index_counter[index] += 1

            if point.get("present", False):
                present_index_counter[index] += 1

            if point.get("visible", False):
                visible_index_counter[index] += 1

    return {
        "num_instances": int(len(records)),
        "mean_num_landmarks": _safe_mean(num_landmarks),
        "median_num_landmarks": _safe_median(num_landmarks),
        "mean_num_present_landmarks": _safe_mean(num_present),
        "median_num_present_landmarks": _safe_median(num_present),
        "mean_num_visible_landmarks": _safe_mean(num_visible),
        "median_num_visible_landmarks": _safe_median(num_visible),
        "landmark_index_count": dict(index_counter),
        "present_landmark_index_count": dict(present_index_counter),
        "visible_landmark_index_count": dict(visible_index_counter),
    }


def run(config: Dict[str, Any]) -> None:
    """
    Run summarization.

    Args:
        config: Loaded config.
    """
    require_config_keys(config, ["dataset"])

    output_root = config.get("landmark_summary", {}).get(
        "output_root",
        "outputs/deepfashion2_landmarks",
    )
    output_dirs = prepare_output_dirs(output_root)

    logger = setup_logger(
        name="deepfashion2_landmark_summary",
        log_file=output_dirs["logs"] / "summary.log",
        level=config.get("logging", {}).get("level", "INFO"),
    )

    parser = build_parser(config)
    image_ids = parser.list_image_ids()

    inference_cfg = config.get("inference", {})
    max_images = inference_cfg.get("max_images")
    random_sample = bool(inference_cfg.get("random_sample", False))
    random_seed = int(inference_cfg.get("random_seed", 42))

    if random_sample:
        rng = np.random.default_rng(random_seed)
        image_ids = list(rng.permutation(image_ids))

    if max_images is not None:
        image_ids = image_ids[: int(max_images)]

    all_records: List[Dict[str, Any]] = []
    by_target_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_raw_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    logger.info("Start landmark summary.")
    logger.info("Number of images to scan: %d", len(image_ids))

    processed_images = 0

    for image_id in tqdm(image_ids, desc="Summarizing landmarks"):
        try:
            sample = parser.load_sample(image_id)
        except (FileNotFoundError, ValueError, KeyError) as error:
            logger.warning("Skip image %s due to error: %s", image_id, error)
            continue

        for instance in sample.get("instances", []):
            landmarks = instance.get("landmarks", [])

            num_present = sum(
                1 for point in landmarks if point.get("present", False)
            )
            num_visible = sum(
                1 for point in landmarks if point.get("visible", False)
            )

            record = {
                "image_id": str(image_id),
                "instance_id": str(instance.get("instance_id", "unknown")),
                "category_id": int(instance.get("category_id", -1)),
                "raw_category_name": str(
                    instance.get("raw_category_name", "unknown")
                ),
                "target_category": str(
                    instance.get("target_category", "unknown")
                ),
                "num_landmarks": int(len(landmarks)),
                "num_present_landmarks": int(num_present),
                "num_visible_landmarks": int(num_visible),
                "landmarks": landmarks,
            }

            all_records.append(record)
            by_target_category[record["target_category"]].append(record)
            by_raw_category[record["raw_category_name"]].append(record)

        processed_images += 1

    summary = {
        "task": "deepfashion2_landmark_summary",
        "num_images": int(processed_images),
        "num_instances": int(len(all_records)),
        "overall": summarize_group(all_records),
        "by_target_category": {
            category: summarize_group(records)
            for category, records in by_target_category.items()
        },
        "by_raw_category": {
            category: summarize_group(records)
            for category, records in by_raw_category.items()
        },
    }

    summary_path = output_dirs["metrics"] / "landmark_summary.json"
    save_json(summary, summary_path)

    logger.info("Finished landmark summary.")
    logger.info("Processed images: %d", processed_images)
    logger.info("Processed instances: %d", len(all_records))
    logger.info("Summary saved to: %s", summary_path)


def main() -> None:
    """
    Main entry.
    """
    args = parse_args()
    config = load_yaml_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
