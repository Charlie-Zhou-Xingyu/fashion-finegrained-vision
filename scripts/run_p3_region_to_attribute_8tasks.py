"""Run existing P3 region-to-attribute predictor for 8 FashionAI tasks.

Task configuration is loaded at startup from two YAML files:

  configs/attribute_inference.yaml
      Per-task model settings: checkpoint, label_map, arch, img_size,
      region_filter (the --region CLI arg), and optional class_contains.

  configs/attribute_group_mapping.yaml
      Per-task routing settings read via CategoryGate:
      crop_input_type and component_contains.

The dict produced by build_task_configs() is structurally identical to the
former hardcoded TASK_CONFIGS, so all downstream subprocess behaviour is
unchanged.  Override the YAML paths at runtime with --inference-config and
--group-mapping-config.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Project root and src/ path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fashion_vision.attributes.category_gate import (  # noqa: E402
    load_attribute_group_mapping,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_INFERENCE_CONFIG = ROOT / "configs" / "attribute_inference.yaml"
_DEFAULT_GROUP_MAPPING_CONFIG = ROOT / "configs" / "attribute_group_mapping.yaml"

DEFAULT_REGION_CROPS_JSON = (
    "outputs/p3_region_to_attribute_neckline/expanded_region_crops/region_crops_with_expanded.json"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_task_configs(
    inference_config: Path,
    group_mapping_config: Path,
) -> dict[str, dict[str, Any]]:
    """Build the task configuration dict from YAML files.

    Merges per-task model settings (from ``attribute_inference.yaml``) with
    per-task routing settings (from ``attribute_group_mapping.yaml``) to
    produce a dict equivalent to the former hardcoded ``TASK_CONFIGS``.

    The returned dict maps task name → config dict with keys:

    * ``checkpoint``        — path string, relative to project root
    * ``label_map``         — path string, relative to project root
    * ``arch``              — model architecture name (e.g. ``"resnet18"``)
    * ``img_size``          — input image size (int)
    * ``region``            — ``--region`` arg for predict_region_attribute_batch.py
    * ``crop_input_type``   — ``--crop-input-type`` arg
    * ``component_contains``— ``--component-contains`` arg value, or ``None``
    * ``class_contains``    — ``--class-contains`` arg value, or ``None``

    Args:
        inference_config: Path to ``attribute_inference.yaml``.
        group_mapping_config: Path to ``attribute_group_mapping.yaml``.

    Returns:
        Task configuration dict.

    Raises:
        FileNotFoundError: If either YAML file does not exist.
        KeyError: If the inference YAML lacks a ``tasks`` section.
        ValueError: If a task entry is missing required fields
            (``checkpoint`` or ``label_map``).
    """
    if not inference_config.exists():
        raise FileNotFoundError(
            f"Inference config not found: {inference_config}\n"
            "Create configs/attribute_inference.yaml or pass --inference-config."
        )

    raw: dict[str, Any] = yaml.safe_load(
        inference_config.read_text(encoding="utf-8")
    )
    if "tasks" not in raw:
        raise KeyError(
            f"'tasks' section missing from {inference_config}. "
            "Expected a mapping of task names under the 'tasks' key."
        )

    mapping = load_attribute_group_mapping(group_mapping_config)

    task_configs: dict[str, dict[str, Any]] = {}
    for task, task_data in raw["tasks"].items():
        _validate_task_entry(task, task_data, inference_config)

        crop_type = mapping.task_to_crop_type.get(task)
        if crop_type is None:
            logger.warning(
                "Task %r not found in task_to_crop_type in %s; "
                "defaulting to 'expanded_crop'.",
                task,
                group_mapping_config,
            )
            crop_type = "expanded_crop"

        task_configs[task] = {
            "checkpoint": task_data["checkpoint"],
            "label_map": task_data["label_map"],
            "arch": str(task_data.get("arch", "resnet18")),
            "img_size": int(task_data.get("img_size", 224)),
            "region": str(task_data.get("region_filter", "all")),
            "crop_input_type": crop_type,
            "component_contains": mapping.task_to_component_filter.get(task),
            "class_contains": task_data.get("class_contains") or None,
        }

    return task_configs


def _validate_task_entry(
    task: str,
    task_data: dict[str, Any],
    config_path: Path,
) -> None:
    """Raise ValueError if required per-task fields are absent.

    Args:
        task: Task name string.
        task_data: Raw task dict from the YAML.
        config_path: YAML file path (used in error messages only).

    Raises:
        ValueError: If ``checkpoint`` or ``label_map`` are missing.
    """
    for required_field in ("checkpoint", "label_map"):
        if required_field not in task_data:
            raise ValueError(
                f"Task {task!r} in {config_path} is missing required "
                f"field '{required_field}'."
            )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_cmd(cmd: list[Any], cwd: Path) -> None:
    """Print and execute a subprocess command.

    Args:
        cmd: Command and arguments as a list.
        cwd: Working directory for the subprocess.

    Raises:
        subprocess.CalledProcessError: If the command exits with non-zero status.
    """
    print("\n[RUN]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load task configs from YAML, and run all tasks."""
    parser = argparse.ArgumentParser(
        description="Run existing P3 region-to-attribute predictor for 8 FashionAI tasks."
    )
    parser.add_argument(
        "--region-crops-json",
        default=DEFAULT_REGION_CROPS_JSON,
        help="Existing region crops JSON generated by the P3 pipeline.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/p3_region_to_attribute_8tasks",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Tasks to run (space-separated). "
            "Defaults to all tasks defined in attribute_inference.yaml."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--vis-limit", type=int, default=200)
    parser.add_argument("--vis-cols", type=int, default=5)
    parser.add_argument("--vis-page-size", type=int, default=50)
    parser.add_argument("--vis-thumb-size", type=int, default=160)
    parser.add_argument("--vis-text-height", type=int, default=90)
    parser.add_argument("--vis-font-size", type=int, default=10)
    parser.add_argument("--skip-vis", action="store_true")
    # Config path overrides — optional; defaults point to standard YAML locations.
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to attribute_inference.yaml. "
            f"Default: {_DEFAULT_INFERENCE_CONFIG}"
        ),
    )
    parser.add_argument(
        "--group-mapping-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to attribute_group_mapping.yaml. "
            f"Default: {_DEFAULT_GROUP_MAPPING_CONFIG}"
        ),
    )
    args = parser.parse_args()

    # Resolve YAML paths
    inference_config = (
        Path(args.inference_config) if args.inference_config
        else _DEFAULT_INFERENCE_CONFIG
    )
    group_mapping_config = (
        Path(args.group_mapping_config) if args.group_mapping_config
        else _DEFAULT_GROUP_MAPPING_CONFIG
    )

    # Build task configs from YAML files
    task_configs = build_task_configs(inference_config, group_mapping_config)

    # Resolve tasks: default to all keys defined in the YAML
    tasks: list[str] = args.tasks if args.tasks is not None else list(task_configs.keys())

    # Resolve region crops JSON
    region_crops_json = Path(args.region_crops_json)
    if not region_crops_json.is_absolute():
        region_crops_json = ROOT / region_crops_json
    if not region_crops_json.exists():
        raise FileNotFoundError(region_crops_json)

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        if task not in task_configs:
            raise KeyError(
                f"Unknown task: {task!r}. "
                f"Available tasks: {list(task_configs)}"
            )

        cfg = task_configs[task]
        out_dir = output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_jsonl = out_dir / "predictions.jsonl"
        summary_json = out_dir / "summary.json"
        vis_dir = out_dir / "vis"

        checkpoint = ROOT / cfg["checkpoint"]
        label_map = ROOT / cfg["label_map"]

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if not label_map.exists():
            raise FileNotFoundError(label_map)

        cmd: list[Any] = [
            sys.executable,
            "tools/demo/predict_region_attribute_batch.py",
            "--region-crops-json",
            str(region_crops_json),
            "--checkpoint",
            str(checkpoint),
            "--label-map",
            str(label_map),
            "--task",
            task,
            "--arch",
            cfg["arch"],
            "--img-size",
            str(cfg["img_size"]),
            "--topk",
            str(args.topk),
            "--device",
            args.device,
            "--region",
            cfg["region"],
            "--crop-input-type",
            cfg["crop_input_type"],
            "--output-jsonl",
            str(pred_jsonl),
            "--output-summary",
            str(summary_json),
        ]

        if cfg.get("component_contains"):
            cmd += ["--component-contains", cfg["component_contains"]]
        if cfg.get("class_contains"):
            cmd += ["--class-contains", cfg["class_contains"]]
        if args.max_samples and args.max_samples > 0:
            cmd += ["--max-samples", str(args.max_samples)]

        run_cmd(cmd, ROOT)

        if not args.skip_vis:
            vis_cmd: list[Any] = [
                sys.executable,
                "scripts/visualize_region_attribute_predictions.py",
                "--pred-jsonl",
                str(pred_jsonl),
                "--output-dir",
                str(vis_dir),
                "--image-field",
                "auto",
                "--limit",
                str(args.vis_limit),
                "--cols",
                str(args.vis_cols),
                "--page-size",
                str(args.vis_page_size),
                "--thumb-size",
                str(args.vis_thumb_size),
                "--text-height",
                str(args.vis_text_height),
                "--font-size",
                str(args.vis_font_size),
            ]
            run_cmd(vis_cmd, ROOT)

        print(f"[OK] {task}: {out_dir}", flush=True)

    print(f"\n[DONE] outputs saved to: {output_root}", flush=True)


if __name__ == "__main__":
    main()
