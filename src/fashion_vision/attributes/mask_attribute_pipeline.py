"""PRD 3.1.3 direct inference: image + target region mask → fine-grained attributes.

Implements the PRD-facing entry point for attribute extraction from a product
image and a binary target region mask.  Reuses all existing infrastructure
without duplicating any routing or inference logic:

* ``crop_utils.crop_region_from_image``  — deterministic mask-aware crop
* ``_infer_coarse_class``                — fine→coarse class mapping
* ``GarmentAttributePipeline``           — task routing and model inference

Flow::

    image (RGB, H×W×3) + binary mask (H×W)
        + garment_category (str) + component_type (str)
            ├─ _mask_bbox_xyxy()                        → tight bbox
            ├─ crop_region_from_image(image, bbox, mask) → masked_crop (PIL)
            ├─ crop_region_from_image(image, bbox, None)  → raw_crop   (PIL)
            ├─ _make_overlay(image, mask)                → overlay     (PIL)
            ├─ save artifacts to output_dir
            ├─ _build_synthetic_record(...)              → crop record dict
            └─ GarmentAttributePipeline.predict_instance([record])
                   → {task: {label, score, topk}}

The synthetic crop record sets every crop-path key to the saved masked crop so
that every attribute task receives it regardless of which ``crop_type`` it
prefers (``upper_crop``, ``expanded_crop``, etc.).

Example::

    from fashion_vision.attributes.mask_attribute_pipeline import (
        predict_attributes_from_mask,
    )
    result = predict_attributes_from_mask(
        image_path="assets/random_train60/images/000004.jpg",
        mask_path="outputs/test_pipeline_smoke/02_samhq/masks/000004_det000_long sleeve top_mask.png",
        garment_category="top",
        component_type="collar",
        output_dir="outputs/smoke_test_attr_from_mask",
        topk=3,
        device="cpu",
    )
    for task, pred in result["attributes"].items():
        print(f"{task}: {pred['label']} ({pred['score']:.3f})")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path: add src/ when run as a CLI script
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parents[2]  # .../src/fashion_vision/attributes/ → src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fashion_vision.attributes.garment_attribute_pipeline import (  # noqa: E402
    AttributePipelineConfig,
    GarmentAttributePipeline,
    _infer_coarse_class,
)
from fashion_vision.utils.crop_utils import crop_region_from_image  # noqa: E402

logger = logging.getLogger(__name__)

_PROJECT_ROOT = _SRC_DIR.parent

# ---------------------------------------------------------------------------
# Garment category normalisation
# ---------------------------------------------------------------------------

# Maps user-supplied strings to the 5 PRD coarse class names.
# Covers direct names, aliases, and common synonyms.
_COARSE_CLASS_ALIASES: dict[str, str] = {
    "top": "top",
    "upper": "top",
    "shirt": "top",
    "blouse": "top",
    "tee": "top",
    "vest": "top",
    "sling": "top",
    "pants": "pants",
    "bottom": "pants",
    "trousers": "pants",
    "shorts": "pants",
    "skirt": "skirt",
    "outerwear": "outerwear",
    "outwear": "outerwear",
    "coat": "outerwear",
    "jacket": "outerwear",
    "dress": "dress",
}

# Representative DeepFashion2 fine class for each PRD coarse class.
# Used to build synthetic crop records so that GarmentAttributePipeline.predict_instance()
# can infer the coarse class via its substring-matching logic.
_COARSE_TO_REPRESENTATIVE_FINE: dict[str, str] = {
    "top":       "long sleeve top",
    "pants":     "trousers",
    "skirt":     "skirt",
    "outerwear": "long sleeve outwear",
    "dress":     "long sleeve dress",
}

# ---------------------------------------------------------------------------
# Component type → crop record field mapping
# ---------------------------------------------------------------------------

# Maps component_type → (region, component) for the synthetic crop record.
#
# The ``region`` field must match the task's ``region_filter`` exactly for
# tasks with non-"all" filters (e.g. collar tasks use region_filter="collar").
# The ``component`` field must satisfy ``component_contains`` substring filters
# (e.g. sleeve_length requires "sleeve" in component).
#
# Tasks with region_filter="all" accept any region; their component filter still
# applies.  Unknown component types fall back to (component_type, component_type).
_COMPONENT_TO_RECORD_FIELDS: dict[str, tuple[str, str]] = {
    "collar":   ("collar",      "collar"),
    "neckline": ("collar",      "neckline"),
    "neck":     ("collar",      "neck"),
    "lapel":    ("collar",      "lapel"),
    "sleeve":   ("sleeve",      "sleeve"),
    "cuff":     ("cuff",        "sleeve"),    # sleeve_length component_contains="sleeve"
    "hem":      ("hem",         "hem"),
    "waist":    ("waist",       "waist"),
    "leg":      ("leg_opening", "pant_leg"),
    "pant_leg": ("leg_opening", "pant_leg"),
    "pant":     ("pant_leg",    "pant"),      # pant_length component_contains="pant"
}


# ---------------------------------------------------------------------------
# Pure helpers (all testable without model weights or real files)
# ---------------------------------------------------------------------------


def _normalize_garment_category(
    garment_category: str,
    mapping: Any,
) -> str:
    """Map a user-supplied garment category to a PRD coarse class name.

    Resolution order:
    1. Direct lookup in :data:`_COARSE_CLASS_ALIASES` (handles common aliases).
    2. :func:`~fashion_vision.attributes.garment_attribute_pipeline._infer_coarse_class`
       substring matching for DeepFashion2 fine class names.

    Args:
        garment_category: User input (e.g. ``"top"``, ``"upper"``,
            ``"long sleeve top"``).
        mapping: Loaded
            :class:`~fashion_vision.attributes.category_gate.AttributeGroupMapping`.

    Returns:
        One of the 5 PRD coarse class names.

    Raises:
        ValueError: If no mapping can be resolved.
    """
    key = garment_category.lower().strip()
    if key in _COARSE_CLASS_ALIASES:
        return _COARSE_CLASS_ALIASES[key]
    coarse = _infer_coarse_class(garment_category, mapping)
    if coarse is not None:
        return coarse
    raise ValueError(
        f"Cannot map garment_category {garment_category!r} to a known coarse class. "
        f"Accepted coarse classes: {sorted(set(_COARSE_CLASS_ALIASES.values()))}. "
        f"Aliases accepted: {sorted(_COARSE_CLASS_ALIASES)}. "
        f"DeepFashion2 fine class names (e.g. 'long sleeve top') are also accepted."
    )


def _get_region_component(component_type: str) -> tuple[str, str]:
    """Map a component type string to ``(region, component)`` crop-record fields.

    Args:
        component_type: User input (e.g. ``"collar"``, ``"neckline"``,
            ``"sleeve"``).

    Returns:
        ``(region, component)`` where *region* satisfies the task's
        ``region_filter`` and *component* satisfies ``component_contains``
        filters.  Unknown types fall back to ``(component_type, component_type)``.
    """
    key = component_type.lower().strip()
    if key in _COMPONENT_TO_RECORD_FIELDS:
        return _COMPONENT_TO_RECORD_FIELDS[key]
    logger.warning(
        "Unknown component_type %r — using as both region and component. "
        "Tasks with strict region_filter values may not match.",
        component_type,
    )
    return key, key


def _load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as a (H, W, 3) uint8 RGB numpy array.

    Args:
        path: Path to an image file (JPEG, PNG, etc.).

    Returns:
        RGB array, dtype uint8.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def _load_binary_mask(path: Path) -> np.ndarray:
    """Load a mask file as a (H, W) boolean numpy array.

    Any pixel with grayscale value > 0 is treated as foreground (``True``).

    Args:
        path: Path to a mask image (PNG, etc.).

    Returns:
        Boolean array of shape (H, W).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    return np.array(Image.open(path).convert("L")) > 0


def _mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight bounding box of the foreground region in *mask*.

    Args:
        mask: Boolean array of shape (H, W).

    Returns:
        ``(x1, y1, x2, y2)`` in absolute pixel coordinates.  ``x2`` and
        ``y2`` are exclusive (one past the last foreground pixel column/row).

    Raises:
        ValueError: If *mask* contains no foreground pixels.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError(
            "Mask is empty — no foreground pixels found. "
            "Provide a mask with at least one non-zero pixel."
        )
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return x1, y1, x2 + 1, y2 + 1


def _make_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 80, 80),
    alpha: float = 0.45,
) -> Image.Image:
    """Render the mask as a semi-transparent coloured overlay on the image.

    Args:
        image_rgb: RGB array (H, W, 3) uint8.
        mask: Boolean array (H, W); True = foreground.
        color: RGB tint colour for foreground pixels. Default red.
        alpha: Overlay opacity (0 = invisible, 1 = solid). Default 0.45.

    Returns:
        PIL RGB Image with the mask region tinted.
    """
    result = image_rgb.astype(np.float32).copy()
    fg = mask.astype(bool)
    result[fg] = (1.0 - alpha) * result[fg] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def _build_synthetic_record(
    masked_crop_path: Path,
    raw_crop_path: Path,
    fine_class_name: str,
    region: str,
    component: str,
) -> dict[str, Any]:
    """Build a crop record dict compatible with ``GarmentAttributePipeline.predict_instance()``.

    All crop-type path keys (``expanded_crop_path``, ``upper_crop_path``,
    ``masked_crop_path``, ``crop_path``) point to *masked_crop_path* so that
    every attribute task gets the mask-aware crop regardless of which
    ``crop_type`` it prefers.  The raw crop is stored in ``image_crop_path``
    for completeness.

    Args:
        masked_crop_path: Path to the saved mask-aware crop.
        raw_crop_path: Path to the saved raw (no fill) crop.
        fine_class_name: DeepFashion2 fine class name (used by
            ``predict_instance`` to resolve the coarse class via substring
            matching).
        region: Region label (must match the task's ``region_filter`` value).
        component: Component label (must satisfy ``component_contains`` filters).

    Returns:
        Synthetic crop record dict ready for ``predict_instance``.
    """
    s = str(masked_crop_path)
    return {
        "det_id": "mask_inference",
        "class_name": fine_class_name,
        "region": region,
        "component": component,
        "success": True,
        "crop_path": s,
        "expanded_crop_path": s,
        "upper_crop_path": s,
        "masked_crop_path": s,
        "image_crop_path": str(raw_crop_path),
    }


# ---------------------------------------------------------------------------
# MaskAttributePipeline
# ---------------------------------------------------------------------------


class MaskAttributePipeline:
    """PRD 3.1.3 direct inference: image + binary mask → attribute labels.

    Wraps :class:`~fashion_vision.attributes.garment_attribute_pipeline.GarmentAttributePipeline`
    to accept raw images and binary region masks rather than a pre-generated
    ``region_crops.json``.

    Models are loaded lazily on the first prediction call for each coarse
    garment class and cached for subsequent calls.

    Example::

        pipeline = MaskAttributePipeline()
        result = pipeline.predict(
            image_path="assets/random_train60/images/000004.jpg",
            mask_path="outputs/test_pipeline_smoke/02_samhq/masks/000004_det000_long sleeve top_mask.png",
            garment_category="top",
            component_type="collar",
            output_dir="outputs/smoke_test_attr_from_mask",
        )
        for task, pred in result["attributes"].items():
            print(task, pred["label"], pred["score"])
    """

    def __init__(self, config: AttributePipelineConfig | None = None) -> None:
        """Initialise. Does NOT load any model weights yet.

        Args:
            config: Pipeline configuration.  Defaults to
                :class:`AttributePipelineConfig` with standard YAML paths.
        """
        self._pipeline = GarmentAttributePipeline(config or AttributePipelineConfig())
        self._mapping = self._pipeline._mapping

    def predict(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        garment_category: str,
        component_type: str,
        output_dir: Union[str, Path],
        topk: int = 3,
        background_fill: str = "mean",
    ) -> dict[str, Any]:
        """Predict fine-grained attributes from a product image and region mask.

        Args:
            image_path: Path to the full product image.
            mask_path: Path to the binary target region mask.  Any non-zero
                pixel is foreground.  Spatial dimensions must match
                *image_path*.
            garment_category: Garment class or alias (e.g. ``"top"``,
                ``"upper"``, ``"long sleeve top"``).
            component_type: Region being analysed (e.g. ``"collar"``,
                ``"neckline"``, ``"sleeve"``).
            output_dir: Directory for saving crop images and overlay.
            topk: Number of top-k predictions per task.
            background_fill: Background fill mode for the masked crop —
                ``"mean"`` (default), ``"zero"``, or ``"keep"``.

        Returns:
            JSON-serialisable dict:

            .. code-block:: json

                {
                  "image_path":       "...",
                  "mask_path":        "...",
                  "garment_category": "...",
                  "coarse_class":     "...",
                  "component_type":   "...",
                  "bbox_xyxy":        [x1, y1, x2, y2],
                  "crop_path":        "... (masked crop)",
                  "raw_crop_path":    "... (no background fill)",
                  "overlay_path":     "... (mask overlay)",
                  "attributes": {
                    "task_name": {"label": "...", "score": 0.0, "topk": [...]}
                  }
                }

        Raises:
            FileNotFoundError: If *image_path* or *mask_path* is missing.
            ValueError: If the mask is empty, spatial dimensions differ, or
                *garment_category* cannot be resolved.
        """
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load image and mask
        image_rgb = _load_image_rgb(image_path)
        mask = _load_binary_mask(mask_path)

        if image_rgb.shape[:2] != mask.shape:
            raise ValueError(
                f"Image spatial dims {image_rgb.shape[:2]} do not match "
                f"mask dims {mask.shape}. Both must be (H, W)."
            )

        # 2. Compute tight bbox from mask foreground pixels
        bbox_xyxy = list(_mask_bbox_xyxy(mask))

        # 3. Generate masked crop, raw crop, and overlay
        masked_crop: Image.Image = crop_region_from_image(
            image_rgb,
            bbox_xyxy,
            mask=mask.astype(np.uint8),
            expand_ratio=0.15,
            target_size=224,
            background_fill=background_fill,
        )
        raw_crop: Image.Image = crop_region_from_image(
            image_rgb,
            bbox_xyxy,
            mask=None,
            expand_ratio=0.15,
            target_size=224,
            background_fill="keep",
        )
        overlay: Image.Image = _make_overlay(image_rgb, mask)

        # 4. Save all artifacts
        stem = f"{image_path.stem}_{component_type}"
        masked_crop_path = output_dir / f"{stem}_masked_crop.jpg"
        raw_crop_path = output_dir / f"{stem}_raw_crop.jpg"
        overlay_path = output_dir / f"{stem}_overlay.jpg"

        masked_crop.save(masked_crop_path, quality=95)
        raw_crop.save(raw_crop_path, quality=95)
        overlay.save(overlay_path, quality=95)

        logger.info("Saved masked crop: %s", masked_crop_path)
        logger.info("Saved raw crop:    %s", raw_crop_path)
        logger.info("Saved overlay:     %s", overlay_path)

        # 5. Resolve garment class and component mapping
        coarse_class = _normalize_garment_category(garment_category, self._mapping)
        fine_class_name = _COARSE_TO_REPRESENTATIVE_FINE[coarse_class]
        region, component = _get_region_component(component_type)

        logger.info(
            "Routing: category=%r → coarse=%r, fine=%r, region=%r, component=%r",
            garment_category, coarse_class, fine_class_name, region, component,
        )

        # 6. Build synthetic crop record and run inference via existing pipeline
        record = _build_synthetic_record(
            masked_crop_path=masked_crop_path,
            raw_crop_path=raw_crop_path,
            fine_class_name=fine_class_name,
            region=region,
            component=component,
        )

        prev_topk = self._pipeline._config.topk
        self._pipeline._config.topk = topk
        try:
            attributes = self._pipeline.predict_instance([record])
        finally:
            self._pipeline._config.topk = prev_topk

        if not attributes:
            logger.warning(
                "No attribute predictions produced for category=%r component=%r. "
                "Check that enabled tasks exist for coarse_class=%r and that "
                "component_type maps to a matching region/component filter.",
                garment_category, component_type, coarse_class,
            )

        return {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "garment_category": garment_category,
            "coarse_class": coarse_class,
            "component_type": component_type,
            "bbox_xyxy": bbox_xyxy,
            "crop_path": str(masked_crop_path),
            "raw_crop_path": str(raw_crop_path),
            "overlay_path": str(overlay_path),
            "attributes": attributes,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def predict_attributes_from_mask(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    garment_category: str,
    component_type: str,
    output_dir: Union[str, Path],
    topk: int = 3,
    device: str = "cpu",
    background_fill: str = "mean",
) -> dict[str, Any]:
    """Convenience wrapper: create a fresh pipeline and predict once.

    For repeated predictions, instantiate :class:`MaskAttributePipeline` directly
    to benefit from model weight caching across calls.

    Args:
        image_path: Path to the full product image.
        mask_path: Path to the binary target region mask.
        garment_category: Garment class or alias.
        component_type: Region being analysed.
        output_dir: Directory for saving artifacts.
        topk: Number of top-k predictions per task.
        device: PyTorch device — ``"cpu"``, ``"cuda"``, or ``"auto"``.
        background_fill: Background fill mode — ``"mean"``, ``"zero"``,
            or ``"keep"``.

    Returns:
        Same dict as :meth:`MaskAttributePipeline.predict`.
    """
    config = AttributePipelineConfig(device=device, topk=topk)
    pipeline = MaskAttributePipeline(config)
    return pipeline.predict(
        image_path=image_path,
        mask_path=mask_path,
        garment_category=garment_category,
        component_type=component_type,
        output_dir=output_dir,
        topk=topk,
        background_fill=background_fill,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "PRD 3.1.3: image + target region mask → fine-grained attribute labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, type=Path,
                        help="Path to the full product image.")
    parser.add_argument("--mask", required=True, type=Path,
                        help="Path to the binary target region mask.")
    parser.add_argument("--garment-category", required=True, type=str,
                        help="Garment class (e.g. 'top', 'upper', 'pants', 'dress').")
    parser.add_argument("--component-type", required=True, type=str,
                        help="Component region (e.g. 'collar', 'neckline', 'sleeve').")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--background-fill", type=str, default="mean",
                        choices=["mean", "zero", "keep"])
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    result = predict_attributes_from_mask(
        image_path=args.image,
        mask_path=args.mask,
        garment_category=args.garment_category,
        component_type=args.component_type,
        output_dir=args.output_dir,
        topk=args.topk,
        device=args.device,
        background_fill=args.background_fill,
    )

    out_dir = Path(args.output_dir)
    json_path = out_dir / "predictions.json"
    jsonl_path = out_dir / "predictions.jsonl"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("predictions.json  → %s", json_path)
    logger.info("predictions.jsonl → %s", jsonl_path)

    attrs = result.get("attributes", {})
    if attrs:
        logger.info("Attribute predictions:")
        for task, pred in attrs.items():
            topk_str = ", ".join(
                f"{t['label']}({t['score']:.2f})" for t in pred.get("topk", [])[1:]
            )
            logger.info("  %-20s %s (%.3f)  alts: %s", task, pred["label"], pred["score"], topk_str)
    else:
        logger.warning(
            "No attribute predictions produced. "
            "Verify --garment-category and --component-type are compatible."
        )


if __name__ == "__main__":
    main()
