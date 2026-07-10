"""PRD 3.1.3 attribute pipeline: connects 3.1.2 region-crop outputs to attribute classifiers.

This module bridges the garment pipeline (3.1.1 detection + 3.1.2 region localisation)
and the FashionAI attribute classifiers (3.1.3).  It reads the ``region_crops.json``
produced by ``tools/infer/garment_pipeline.py`` stage 4 (or stage 5 masked crops),
groups records by ``det_id``, and runs the appropriate attribute tasks per garment
instance.

Input JSON structure (``region_crops.json``)
--------------------------------------------
::

    {
      "crops": [
        {
          "det_id":            "img001_det0",
          "class_name":        "long sleeve top",
          "region":            "collar",
          "component":         "front_collar",
          "success":           true,
          "expanded_crop_path":"outputs/.../expanded_crop.jpg",
          "upper_crop_path":   "outputs/.../upper_crop.jpg",
          ...
        },
        ...
      ]
    }

Per-instance attribute output
-----------------------------
::

    {
      "det_id":           "img001_det0",
      "fine_class_name":  "long sleeve top",
      "coarse_class_name":"top",
      "num_crops":        3,
      "attributes": {
        "neckline_design": {
          "label": "V-shape",
          "score": 0.87,
          "topk": [{"label": "V-shape", "score": 0.87}, ...]
        },
        "sleeve_length": { ... }
      }
    }

Task routing is config-driven:

* ``configs/attribute_inference.yaml`` — per-task checkpoint, region_filter, class_contains
* ``configs/attribute_group_mapping.yaml`` — per-task crop_type, component_contains, coarse→tasks
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path: add src/ when this module is run directly as a CLI script.
# (When imported as a package, src/ is already on the path.)
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parents[2]  # src/fashion_vision/attributes/ → src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fashion_vision.attributes.category_gate import (  # noqa: E402
    AttributeGroupMapping,
    get_enabled_tasks,
    load_attribute_group_mapping,
)
from fashion_vision.attributes.task_registry import (  # noqa: E402
    LoadedTask,
    load_inference_config,
    load_tasks_for_class,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = _SRC_DIR.parent  # src/ → project root
_DEFAULT_INFERENCE_CONFIG = _PROJECT_ROOT / "configs" / "attribute_inference.yaml"
_DEFAULT_GROUP_MAPPING = _PROJECT_ROOT / "configs" / "attribute_group_mapping.yaml"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AttributePipelineConfig:
    """Configuration for :class:`GarmentAttributePipeline`.

    Attributes:
        inference_config_path: Path to ``attribute_inference.yaml``.
        group_mapping_path: Path to ``attribute_group_mapping.yaml``.
        device: Inference device.  ``"auto"`` selects CUDA if available.
        topk: Number of top-k predictions to return per task.
    """

    inference_config_path: Path = field(
        default_factory=lambda: _DEFAULT_INFERENCE_CONFIG
    )
    group_mapping_path: Path = field(default_factory=lambda: _DEFAULT_GROUP_MAPPING)
    device: str = "auto"
    topk: int = 3


# ---------------------------------------------------------------------------
# Pure module-level helpers (testable without models)
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    """Resolve ``"auto"``, ``"cpu"``, or ``"cuda[:<N>]"`` to a :class:`torch.device`.

    Args:
        device_str: Device string.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _infer_coarse_class(
    fine_class_name: str,
    mapping: AttributeGroupMapping,
) -> str | None:
    """Map a fine class name to a PRD coarse class name using substring matching.

    Iterates ``coarse_class_to_fine_substrings`` from the group mapping config.
    Returns the first matching coarse class, or ``None`` if no match is found.

    Args:
        fine_class_name: DeepFashion2 fine class name, e.g. ``"long sleeve top"``.
        mapping: Loaded :class:`AttributeGroupMapping`.

    Returns:
        Coarse class name (e.g. ``"top"``), or ``None``.
    """
    name_lower = fine_class_name.lower()
    for coarse_class, substrings in mapping.coarse_class_to_fine_substrings.items():
        for sub in substrings:
            if sub.lower() in name_lower:
                return coarse_class
    return None


def _get_crop_path(record: dict[str, Any], crop_type: str) -> str | None:
    """Return the crop file path for *crop_type* from a region-crop record.

    Mirrors the fallback chain in ``tools/demo/predict_region_attribute_batch.py``
    so both scripts behave identically when selecting crop inputs.

    Args:
        record: Single crop record dict from ``region_crops.json``.
        crop_type: One of ``"expanded_crop"``, ``"upper_crop"``,
            ``"masked_crop"``, ``"image_crop"``, ``"raw_region_crop"``.

    Returns:
        Absolute or relative path string, or ``None`` if no matching key exists.
    """
    if crop_type == "expanded_crop":
        return (
            record.get("expanded_crop_path")
            or record.get("image_crop_path")
            or record.get("crop_path")
        )
    if crop_type == "upper_crop":
        return (
            record.get("upper_crop_path")
            or record.get("expanded_crop_path")
            or record.get("image_crop_path")
            or record.get("crop_path")
        )
    if crop_type == "masked_crop":
        return record.get("masked_crop_path")
    if crop_type == "image_crop":
        return record.get("image_crop_path") or record.get("crop_path")
    # raw_region_crop or unknown
    return record.get("crop_path")


def _select_crop_record(
    crops: list[dict[str, Any]],
    region_filter: str,
    class_contains: str | None,
    component_contains: str | None,
) -> dict[str, Any] | None:
    """Find the first crop record from *crops* that passes all three filters.

    Filters applied in order:

    1. ``success == True`` (required).
    2. ``region == region_filter`` unless *region_filter* is ``"all"``.
    3. ``class_contains`` substring present in ``class_name`` (if set).
    4. ``component_contains`` substring present in ``component`` (if set).

    Args:
        crops: List of crop records for one garment instance (same ``det_id``).
        region_filter: Region type to match, or ``"all"`` to skip region filtering.
        class_contains: Required substring in ``class_name``, or ``None``.
        component_contains: Required substring in ``component``, or ``None``.

    Returns:
        First matching record dict, or ``None`` if none match.
    """
    for crop in crops:
        if not crop.get("success", False):
            continue

        region = str(crop.get("region", ""))
        if region_filter != "all" and region != region_filter:
            continue

        if class_contains:
            class_name = str(crop.get("class_name", ""))
            if class_contains not in class_name:
                continue

        if component_contains:
            component = str(crop.get("component", ""))
            if component_contains not in component:
                continue

        return crop

    return None


def _run_inference(
    loaded: LoadedTask,
    crop_path: Path,
    topk: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one attribute classifier forward pass and return top-k predictions.

    Args:
        loaded: Pre-loaded :class:`LoadedTask` (model in eval mode, on *device*).
        crop_path: Path to the crop image file.
        topk: Number of top-k predictions to include in the result.
        device: Device the model is on (used for the input tensor).

    Returns:
        Dict with keys:

        * ``"label"``  — top-1 label string
        * ``"score"``  — top-1 softmax confidence (float)
        * ``"topk"``   — list of ``{"label": str, "score": float}`` dicts
    """
    image = Image.open(crop_path).convert("RGB")
    x = loaded.transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = loaded.model(x)
        probs = torch.softmax(logits, dim=1)[0]

    k = min(topk, probs.numel())
    confs, ids = torch.topk(probs, k=k)

    topk_list = [
        {
            "label": loaded.id_to_label[int(label_id)],
            "score": round(float(conf), 6),
        }
        for label_id, conf in zip(ids.cpu().tolist(), confs.cpu().tolist())
    ]

    return {
        "label": topk_list[0]["label"],
        "score": topk_list[0]["score"],
        "topk": topk_list,
    }


def _load_raw_task_data(yaml_path: Path) -> dict[str, dict[str, Any]]:
    """Load raw per-task fields from ``attribute_inference.yaml``.

    Returns the ``tasks`` mapping verbatim so that ``region_filter`` and
    ``class_contains`` are accessible alongside the parsed
    :class:`~fashion_vision.attributes.task_registry.AttributeTaskConfig`.

    Args:
        yaml_path: Path to ``attribute_inference.yaml``.

    Returns:
        Dict mapping task name → raw YAML task entry.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        KeyError: If the YAML lacks a ``tasks`` section.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"attribute_inference.yaml not found: {yaml_path}")
    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if "tasks" not in raw:
        raise KeyError(f"'tasks' section missing from {yaml_path}")
    return raw["tasks"]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class GarmentAttributePipeline:
    """Connects 3.1.2 region-crop records to 3.1.3 attribute classifiers.

    Models are loaded lazily on the first prediction call for each coarse
    garment class and cached for subsequent calls.  Loading all tasks for all
    classes at ``__init__`` time is avoided because individual deployments
    typically serve only one or two garment classes per request.

    Example::

        config = AttributePipelineConfig(device="auto", topk=3)
        pipeline = GarmentAttributePipeline(config)
        results = pipeline.predict_from_json(
            Path("outputs/pipeline/04_region_crops/region_crops.json")
        )
        for r in results:
            print(r["det_id"], r["attributes"])
    """

    def __init__(self, config: AttributePipelineConfig | None = None) -> None:
        """Initialise the pipeline.  Does NOT load any models yet.

        Args:
            config: Pipeline configuration.  Defaults to
                :class:`AttributePipelineConfig` with standard config paths.
        """
        self._config = config or AttributePipelineConfig()
        self._device = _resolve_device(self._config.device)

        # Load configs eagerly (lightweight YAML only — no model weights).
        self._inference_config = load_inference_config(
            self._config.inference_config_path
        )
        self._mapping = load_attribute_group_mapping(self._config.group_mapping_path)
        self._raw_task_data = _load_raw_task_data(self._config.inference_config_path)

        # Task model cache: coarse_class_name → {task_name: LoadedTask}
        self._task_cache: dict[str, dict[str, LoadedTask]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tasks_for_class(self, coarse_class_name: str) -> dict[str, LoadedTask]:
        """Return (and lazily load) the task models for *coarse_class_name*.

        Args:
            coarse_class_name: PRD coarse class name, e.g. ``"top"``.

        Returns:
            Dict mapping task name → :class:`~fashion_vision.attributes.task_registry.LoadedTask`.
        """
        if coarse_class_name not in self._task_cache:
            logger.info(
                "Loading attribute task models for class %r on %s …",
                coarse_class_name,
                self._device,
            )
            self._task_cache[coarse_class_name] = load_tasks_for_class(
                coarse_class_name,
                self._inference_config,
                self._mapping,
                self._device,
            )
        return self._task_cache[coarse_class_name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_instance(
        self,
        crops: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Predict all applicable attribute tasks for one garment instance.

        *crops* must be all region-crop records that share the same ``det_id``
        (as grouped from ``region_crops.json``).  The method reads ``class_name``
        from the first successful record to determine the garment class, then
        dispatches each enabled attribute task.

        Missing or inaccessible crop files are skipped with a warning; they do
        not raise exceptions.

        Args:
            crops: Non-empty list of crop records for a single garment instance.
                Each record is a dict as defined in ``region_crops.json``
                (keys: ``det_id``, ``class_name``, ``region``, ``component``,
                ``success``, ``expanded_crop_path``, ``upper_crop_path``, …).

        Returns:
            Dict mapping task name → attribute prediction dict (``label``,
            ``score``, ``topk``).  Empty dict if the garment class is unknown
            or no enabled tasks have a usable crop.
        """
        # Derive fine class name from the first successful record.
        fine_class_name: str = ""
        for crop in crops:
            if crop.get("success", False):
                fine_class_name = str(crop.get("class_name", ""))
                break

        if not fine_class_name:
            logger.warning(
                "predict_instance: no successful crop with class_name; "
                "skipping attribute prediction."
            )
            return {}

        coarse_class_name = _infer_coarse_class(fine_class_name, self._mapping)
        if coarse_class_name is None:
            logger.warning(
                "predict_instance: cannot infer coarse class from %r; "
                "skipping attribute prediction.",
                fine_class_name,
            )
            return {}

        enabled_tasks = get_enabled_tasks(coarse_class_name, self._mapping)
        if not enabled_tasks:
            return {}

        task_models = self._get_tasks_for_class(coarse_class_name)
        attributes: dict[str, Any] = {}

        for task in enabled_tasks:
            if task not in task_models:
                # Model not in inference config — already warned by load_tasks_for_class.
                continue

            raw = self._raw_task_data.get(task, {})
            region_filter = str(raw.get("region_filter", "all"))
            class_contains: str | None = raw.get("class_contains") or None
            component_contains: str | None = (
                self._mapping.task_to_component_filter.get(task)
            )
            crop_type: str = self._mapping.task_to_crop_type.get(task, "expanded_crop")

            crop_record = _select_crop_record(
                crops, region_filter, class_contains, component_contains
            )
            if crop_record is None:
                logger.debug(
                    "Task %r: no matching crop record "
                    "(region_filter=%r, class_contains=%r, component_contains=%r).",
                    task, region_filter, class_contains, component_contains,
                )
                continue

            crop_path_str = _get_crop_path(crop_record, crop_type)
            if not crop_path_str:
                logger.debug("Task %r: crop path is empty for crop_type=%r.", task, crop_type)
                continue

            crop_path = Path(crop_path_str)
            if not crop_path.exists():
                logger.warning(
                    "Task %r: crop file not found: %s", task, crop_path
                )
                continue

            try:
                attributes[task] = _run_inference(
                    loaded=task_models[task],
                    crop_path=crop_path,
                    topk=self._config.topk,
                    device=self._device,
                )
            except Exception as exc:
                logger.warning(
                    "Task %r: inference failed for %s: %s",
                    task, crop_path, exc,
                )

        return attributes

    def predict_from_json(
        self,
        region_crops_json: Path,
        max_instances: int = 0,
    ) -> list[dict[str, Any]]:
        """Load a region-crops JSON file and predict attributes for all instances.

        Records in the JSON are grouped by ``det_id``.  For each group,
        :meth:`predict_instance` is called.  Per-instance errors are caught and
        reported in the result dict rather than propagating.

        Args:
            region_crops_json: Path to a ``region_crops.json`` file produced by
                the garment pipeline (stage 4 or 5).  Expected structure:
                ``{"crops": [...]}``.
            max_instances: If positive, stop after this many unique instances.
                ``0`` means no limit.

        Returns:
            List of per-instance dicts, each containing:

            * ``"det_id"`` — instance identifier
            * ``"fine_class_name"`` — DeepFashion2 fine class (from crop records)
            * ``"coarse_class_name"`` — PRD coarse class, or ``None``
            * ``"num_crops"`` — number of crop records for this instance
            * ``"attributes"`` — dict mapping task → prediction, or ``{}``
            * ``"error"`` — exception repr if prediction failed, else ``None``

        Raises:
            FileNotFoundError: If *region_crops_json* does not exist.
            ValueError: If the JSON lacks the ``"crops"`` key.
        """
        if not region_crops_json.exists():
            raise FileNotFoundError(
                f"region_crops_json not found: {region_crops_json}"
            )

        data: dict[str, Any] = json.loads(
            region_crops_json.read_text(encoding="utf-8")
        )
        all_crops: list[dict[str, Any]] = data.get("crops", [])
        if not isinstance(all_crops, list):
            raise ValueError(
                f"Expected 'crops' to be a list in {region_crops_json}, "
                f"got {type(all_crops).__name__}."
            )

        # Group crop records by a globally unique instance key.
        # det_id is only unique within each image; incorporate the image_path
        # stem to prevent cross-image collisions when the JSON covers many images.
        # Falls back to raw det_id when image_path is absent (backward compat).
        # Use explicit None checks so that det_id=0 (integer) is preserved.
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for crop in all_crops:
            image_path = str(crop.get("image_path") or "")
            image_stem = Path(image_path).stem if image_path else ""
            _det = crop.get("det_id")
            if _det is None:
                _det = crop.get("instance_id")
            raw_det_id = "" if _det is None else str(_det)
            instance_key = f"{image_stem}__det{raw_det_id}" if image_stem else raw_det_id
            grouped[instance_key].append(crop)

        results: list[dict[str, Any]] = []

        for det_id, crops in grouped.items():
            if max_instances > 0 and len(results) >= max_instances:
                break

            fine_class_name = ""
            for crop in crops:
                if crop.get("success", False):
                    fine_class_name = str(crop.get("class_name", ""))
                    break

            coarse_class_name = (
                _infer_coarse_class(fine_class_name, self._mapping)
                if fine_class_name
                else None
            )

            error_repr: str | None = None
            attributes: dict[str, Any] = {}
            try:
                attributes = self.predict_instance(crops)
            except Exception as exc:
                error_repr = repr(exc)
                logger.error(
                    "predict_from_json: instance %r failed: %s", det_id, exc
                )

            results.append(
                {
                    "det_id": det_id,
                    "fine_class_name": fine_class_name,
                    "coarse_class_name": coarse_class_name,
                    "num_crops": len(crops),
                    "attributes": attributes,
                    "error": error_repr,
                }
            )

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> "argparse.Namespace":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "PRD 3.1.3 attribute pipeline: predict fine-grained garment attributes "
            "from 3.1.2 region-crop outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--region-crops-json",
        type=Path,
        required=True,
        help="Path to region_crops.json produced by the garment pipeline (stage 4/5).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL file (one per-instance result per line).",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Path to attribute_inference.yaml. Default: {_DEFAULT_INFERENCE_CONFIG}",
    )
    parser.add_argument(
        "--group-mapping-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Path to attribute_group_mapping.yaml. Default: {_DEFAULT_GROUP_MAPPING}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top-k predictions per task.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        dest="max_instances",
        help="Limit instances processed. 0 = unlimited.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    inference_config_path = (
        Path(args.inference_config) if args.inference_config
        else _DEFAULT_INFERENCE_CONFIG
    )
    group_mapping_path = (
        Path(args.group_mapping_config) if args.group_mapping_config
        else _DEFAULT_GROUP_MAPPING
    )

    config = AttributePipelineConfig(
        inference_config_path=inference_config_path,
        group_mapping_path=group_mapping_path,
        device=args.device,
        topk=args.topk,
    )

    pipeline = GarmentAttributePipeline(config)

    logger.info("Running attribute pipeline on: %s", args.region_crops_json)
    results = pipeline.predict_from_json(
        args.region_crops_json,
        max_instances=args.max_instances,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    num_with_attrs = sum(1 for r in results if r["attributes"])
    num_errors = sum(1 for r in results if r["error"])
    logger.info(
        "Done. %d instances processed, %d with attributes, %d errors. "
        "Output: %s",
        len(results),
        num_with_attrs,
        num_errors,
        args.output_jsonl,
    )


if __name__ == "__main__":
    main()
