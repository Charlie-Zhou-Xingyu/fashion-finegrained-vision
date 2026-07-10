from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fashion_vision.localization.intent_parser import PART_VOCAB, parse_intent
from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig


REGION_DISPLAY_NAME = {
    "collar": "collar / 领口",
    "sleeve": "sleeve / 袖子",
    "hem": "hem / 下摆",
    "waist": "waist / 腰部",
    "pant_leg": "pant_leg / 裤腿",
}


UPPER_BODY_CLASSES = {
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear",
    "long sleeve outwear",
    "vest",
    "sling",
}


DRESS_CLASSES = {
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress",
}


LOWER_BODY_CLASSES = {
    "shorts",
    "trousers",
    "skirt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Online query demo: input one garment image and a natural language "
            "region query, then return local bbox/mask/overlay."
        )
    )

    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument(
        "--query", type=str, default="",
        help="Natural language query (single-query mode).",
    )
    parser.add_argument(
        "--queries", type=str, default="",
        help=(
            "Comma-separated queries for batch mode, e.g. '口袋,拉链,袖子'. "
            "Models load once; one subfolder per query. Mutually exclusive with --query."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/query_region_online_demo",
        help="Output root directory.",
    )
    parser.add_argument(
        "--save-debug-artifacts",
        action="store_true",
        default=False,
        help="Save full pipeline intermediates + debug.json per query. Default: query-only output.",
    )

    parser.add_argument(
        "--target-class",
        type=str,
        default="",
        help=(
            "Optional garment class hint, e.g. trousers, skirt, long sleeve top. "
            "If omitted, the demo may infer class hint from query such as 裙摆."
        ),
    )
    parser.add_argument(
        "--target-det-id",
        type=int,
        default=None,
        help="Optional detection id. If provided, only crops from this det_id are considered.",
    )
    parser.add_argument(
        "--prefer-component",
        type=str,
        default="",
        help="Optional component hint, e.g. left_sleeve, right_sleeve.",
    )
    parser.add_argument(
        "--reuse-pipeline-dir",
        type=str,
        default="",
        help=(
            "Optional existing pipeline output dir containing 05_region_masked_crops. "
            "If set, skip running GarmentPipeline and reuse existing results."
        ),
    )

    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="checkpoints/sam_hq/sam_hq_vit_b.pth",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument(
        "--landmark-checkpoint",
        type=str,
        default="outputs/landmark_predictor_resnet18/best.pt",
    )

    parser.add_argument("--yolo-device", type=str, default="0")
    parser.add_argument("--sam-device", type=str, default="cuda")
    parser.add_argument("--landmark-device", type=str, default="cuda")

    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--yolo-imgsz", type=int, default=640)

    parser.add_argument(
        "--fp-model",
        type=str,
        default="models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt",
        help="Path to Fashionpedia YOLOv8s part detector weights.",
    )
    parser.add_argument(
        "--fp-device", type=str, default="cuda",
        help="Device for Fashionpedia YOLO (cuda / cpu).",
    )

    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for selected region mask.",
    )

    args = parser.parse_args()

    # ── Validate --query / --queries exclusivity ──
    if args.queries and args.query:
        parser.error("--query and --queries are mutually exclusive.")
    if not args.queries and not args.query:
        parser.error("Either --query or --queries is required.")

    return args


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sanitize_filename_part(text: Any) -> str:
    value = str(text).strip()

    if not value:
        value = "empty"

    return (
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def garment_group(class_name: Any) -> str:
    """
    Map DeepFashion2 class name to semantic garment group.
    """
    name = str(class_name).strip().lower()

    if name in UPPER_BODY_CLASSES:
        return "upper"

    if name in DRESS_CLASSES:
        return "dress"

    if name in LOWER_BODY_CLASSES:
        return "lower"

    return "other"


def waist_class_priority(class_name: Any) -> int:
    """
    Priority only for waist query.

    Smaller value means higher priority.

    Rule:
        upper waist > dress waist > lower waist > others
    """
    group = garment_group(class_name)

    if group == "upper":
        return 0
    if group == "dress":
        return 1
    if group == "lower":
        return 2

    return 9


_OUTWEAR_CLASSES = {
    "short sleeve outwear", "long sleeve outwear",
}
_TOP_CLASSES = {
    "short sleeve top", "long sleeve top", "vest", "sling",
}
_PANTS_CLASSES = {
    "shorts", "trousers",
}


def target_class_matches(class_name: Any, target_class: str) -> bool:
    """
    Match user-provided or query-inferred class hint.

    Special sentinel values:
        __dress__   → any dress class
        __outwear__ → any outwear class
        __top__     → any top class
        __pants__   → any pants class
    """
    if not target_class:
        return True

    cls = str(class_name).strip().lower()
    target = str(target_class).strip().lower()

    if target == "__dress__":
        return cls in DRESS_CLASSES
    if target == "__outwear__":
        return cls in _OUTWEAR_CLASSES
    if target == "__top__":
        return cls in _TOP_CLASSES
    if target == "__pants__":
        return cls in _PANTS_CLASSES

    return cls == target or target in cls


def target_component_matches(component: Any, target_component: str | None) -> bool:
    if not target_component:
        return True

    comp = str(component).strip().lower()
    target = str(target_component).strip().lower()

    return comp == target


def target_det_matches(det_id: Any, target_det_id: int | None) -> bool:
    if target_det_id is None:
        return True

    try:
        return int(det_id) == int(target_det_id)
    except Exception:
        return False


def candidate_sort_key(
    record: dict[str, Any],
    target_region: str,
) -> tuple[Any, ...]:
    """
    Deterministic ordering key for valid candidates.

    Important:
        Only waist uses clothing-category priority.
        Other regions do not use upper/dress/lower priority.

    This is not a model score.
    This is just deterministic ordering for ties.
    """
    component = str(record.get("component", "")).strip().lower()

    if target_region == "waist":
        semantic_priority = waist_class_priority(record.get("class_name", ""))
    else:
        semantic_priority = 0

    fallback_priority = 1 if record.get("fallback", False) else 0

    try:
        num_reliable_landmarks = int(record.get("num_reliable_landmarks", 0))
    except Exception:
        num_reliable_landmarks = 0

    try:
        mask_area_ratio = float(record.get("mask_area_ratio", 0.0))
    except Exception:
        mask_area_ratio = 0.0

    try:
        det_id = int(record.get("det_id", 9999))
    except Exception:
        det_id = 9999

    return (
        semantic_priority,
        fallback_priority,
        -num_reliable_landmarks,
        -mask_area_ratio,
        det_id,
        component,
    )


def selection_reason(
    selected: dict[str, Any],
    target_region: str,
    target_component: str | None = None,
    target_class: str = "",
    target_det_id: int | None = None,
) -> str:
    class_name = str(selected.get("class_name", ""))
    component = str(selected.get("component", ""))
    det_id = selected.get("det_id")
    group = garment_group(class_name)

    if target_region == "waist":
        if group == "upper":
            reason = "waist query selected upper-body garment waist by default"
        elif group == "dress":
            reason = "waist query selected dress waist because no upper-body waist was selected"
        elif group == "lower":
            reason = "waist query selected lower-body waist because no upper/dress waist was selected"
        else:
            reason = "waist query selected an available waist candidate"
    elif target_region == "hem" and target_class == "skirt":
        reason = "skirt hem query selected skirt hem"
    elif target_region == "hem" and target_class == "__dress__":
        reason = "dress hem query selected dress hem"
    else:
        reason = f"{target_region} query selected matching candidate without garment-category priority"

    constraints = []

    if target_component:
        constraints.append(f"component={target_component}")

    if target_class:
        if target_class == "__dress__":
            constraints.append("class=dress")
        else:
            constraints.append(f"class={target_class}")

    if target_det_id is not None:
        constraints.append(f"det_id={target_det_id}")

    if constraints:
        reason += "; constraints: " + ", ".join(constraints)

    reason += f"; selected det_id={det_id}, class={class_name}, component={component}"

    return reason


def select_best_record(
    masked_crops_data: dict[str, Any],
    target_region: str,
    target_component: str | None = None,
    target_class: str = "",
    target_det_id: int | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Select best record using deterministic rules.

    No arbitrary score is used.

    Rules:
        - region must match
        - target_det_id, if provided, must match
        - target_class, if provided or inferred from query, must match
        - target_component, if provided, must match
        - only waist uses upper > dress > lower priority
        - other regions have no garment-category priority
    """
    crops = masked_crops_data.get("crops", [])

    if not isinstance(crops, list):
        return None, []

    candidates: list[dict[str, Any]] = []

    for record in crops:
        if not isinstance(record, dict):
            continue

        if not record.get("masked_success", False):
            continue

        region = str(record.get("region", "")).strip().lower()
        if region != target_region:
            continue

        if not target_det_matches(record.get("det_id"), target_det_id):
            continue

        if not target_class_matches(record.get("class_name", ""), target_class):
            continue

        if not target_component_matches(record.get("component", ""), target_component):
            continue

        item = dict(record)
        item["_selection_sort_key"] = candidate_sort_key(
            record=record,
            target_region=target_region,
        )
        candidates.append(item)

    candidates.sort(key=lambda x: x["_selection_sort_key"])

    for index, item in enumerate(candidates, start=1):
        item["_selection_rank"] = index

    if not candidates:
        return None, []

    selected = candidates[0]
    selected["_selection_reason"] = selection_reason(
        selected=selected,
        target_region=target_region,
        target_component=target_component,
        target_class=target_class,
        target_det_id=target_det_id,
    )

    return selected, candidates


def ensure_2d_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """
    Ensure mask is a 2D array [H, W].
    """
    if mask is None:
        return None

    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        if mask.shape[2] == 1:
            return mask[:, :, 0]

        if mask.shape[2] >= 3:
            return cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)

    squeezed = np.squeeze(mask)

    if squeezed.ndim == 2:
        return squeezed

    raise ValueError(f"Expected 2D mask, got shape={getattr(mask, 'shape', None)}")


def ensure_binary_mask_2d(mask: np.ndarray | None) -> np.ndarray | None:
    """
    Convert mask to 2D uint8 binary mask with values 0 or 255.
    """
    mask_2d = ensure_2d_mask(mask)

    if mask_2d is None:
        return None

    return np.where(mask_2d > 0, 255, 0).astype(np.uint8)


def clip_bbox_xyxy_to_image(
    bbox_xyxy: list[int] | tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> list[int]:
    """
    Clip xyxy bbox to image boundary.
    """
    if len(bbox_xyxy) != 4:
        raise ValueError(f"bbox_xyxy must have 4 values, got: {bbox_xyxy}")

    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w, x2))
    y2 = max(0, min(image_h, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")

    return [x1, y1, x2, y2]


def create_full_size_region_mask(
    image_shape: tuple[int, int] | tuple[int, int, int],
    bbox_xyxy: list[int],
    mask_crop_path: str,
) -> np.ndarray:
    """
    Paste selected local mask crop back to full image canvas.
    """
    image_h, image_w = image_shape[:2]
    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)

    if not mask_crop_path:
        raise ValueError("mask_crop_path is empty.")

    mask_crop_file = Path(mask_crop_path)

    if not mask_crop_file.exists():
        raise FileNotFoundError(f"Mask crop file not found: {mask_crop_file}")

    mask_crop_raw = cv2.imread(str(mask_crop_file), cv2.IMREAD_UNCHANGED)

    if mask_crop_raw is None:
        raise FileNotFoundError(f"Failed to read mask crop: {mask_crop_file}")

    mask_crop = ensure_binary_mask_2d(mask_crop_raw)

    if mask_crop is None:
        raise ValueError(f"Failed to convert mask crop to 2D binary mask: {mask_crop_file}")

    x1, y1, x2, y2 = clip_bbox_xyxy_to_image(
        bbox_xyxy=bbox_xyxy,
        image_w=image_w,
        image_h=image_h,
    )

    target_w = x2 - x1
    target_h = y2 - y1

    if mask_crop.shape[:2] != (target_h, target_w):
        mask_crop = cv2.resize(
            mask_crop,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_crop = ensure_binary_mask_2d(mask_crop)

    if mask_crop is None:
        raise ValueError(f"Invalid mask crop after resize: {mask_crop_file}")

    if mask_crop.shape != (target_h, target_w):
        raise ValueError(
            f"Mask crop shape mismatch. "
            f"Expected {(target_h, target_w)}, got {mask_crop.shape}"
        )

    full_mask[y1:y2, x1:x2] = mask_crop

    return full_mask


def draw_selected_overlay(
    image_bgr: np.ndarray,
    full_mask: np.ndarray,
    bbox_xyxy: list[int],
    label: str,
    alpha: float = 0.45,
) -> np.ndarray:
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected image_bgr shape [H, W, 3], got {getattr(image_bgr, 'shape', None)}"
        )

    image_h, image_w = image_bgr.shape[:2]

    full_mask_bin = ensure_binary_mask_2d(full_mask)

    if full_mask_bin is None:
        raise ValueError("full_mask cannot be converted to 2D binary mask.")

    if full_mask_bin.shape[:2] != (image_h, image_w):
        full_mask_bin = cv2.resize(
            full_mask_bin,
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )
        full_mask_bin = ensure_binary_mask_2d(full_mask_bin)

        if full_mask_bin is None:
            raise ValueError("Resized full_mask cannot be converted to 2D binary mask.")

    bbox_clipped = clip_bbox_xyxy_to_image(
        bbox_xyxy=bbox_xyxy,
        image_w=image_w,
        image_h=image_h,
    )
    x1, y1, x2, y2 = bbox_clipped

    overlay = image_bgr.copy()

    color = np.array([0, 0, 255], dtype=np.uint8)
    color_layer = np.zeros_like(overlay, dtype=np.uint8)
    color_layer[:, :] = color

    mask_bool = full_mask_bin > 0
    blended = cv2.addWeighted(overlay, 1.0, color_layer, float(alpha), 0)
    overlay[mask_bool] = blended[mask_bool]

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    y_text_top = max(0, y1 - th - baseline - 4)
    y_text_bottom = min(image_h - 1, y_text_top + th + baseline + 4)

    cv2.rectangle(
        overlay,
        (x1, y_text_top),
        (min(image_w - 1, x1 + tw + 6), y_text_bottom),
        (0, 0, 255),
        -1,
    )
    cv2.putText(
        overlay,
        label,
        (x1 + 3, max(12, y_text_bottom - baseline - 2)),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return overlay


def copy_if_exists(src: str | None, dst: Path) -> str | None:
    if not src:
        return None

    src_path = Path(src)

    if not src_path.exists():
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)

    return str(dst)


def build_pipeline_config(args: argparse.Namespace, run_landmark_and_crops: bool = False) -> GarmentPipelineConfig:
    return GarmentPipelineConfig(
        yolo_weights=args.yolo_weights,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        landmark_checkpoint=args.landmark_checkpoint,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_device=args.yolo_device,
        sam_device=args.sam_device,
        landmark_device=args.landmark_device,
        save_yolo_vis=True,
        save_yolo_crops=False,
        save_landmark_visualizations=False,
        draw_landmark_index=False,
        draw_landmark_name=False,
        run_landmark_and_crops=run_landmark_and_crops,
    )


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    queries: list[str] = (
        [q.strip() for q in args.queries.split(",") if q.strip()]
        if args.queries
        else [args.query]
    )
    output_root = Path(args.output_dir)
    image_stem = image_path.stem

    # ── Pipeline: lazily built only when a query needs it ───────────────
    pipeline_dir = _resolve_pipeline_dir(args, image_path, output_root)
    pipeline_result: dict = {
        "status": "pending",
        "output_dir": str(pipeline_dir),
        "paths": {
            "segmentation_json": str(pipeline_dir / "02_samhq" / "segmentation_results.json"),
            "region_masked_crops_json": str(pipeline_dir / "05_region_masked_crops" / "region_masked_crops.json"),
        },
    }

    # ── Load models ──────────────────────────────────────────────────────
    fp_detector = _load_fp_detector(args)
    # DINO loaded lazily — only when a query actually needs it.
    _dino_cache: list = [None]

    # ── Per-query loop ──────────────────────────────────────────────────
    results: list[dict] = []
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    H, W = image_bgr.shape[:2]

    for query in queries:
        q_name = sanitize_filename_part(query)
        run_dir = output_root / f"{image_stem}_{q_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n{'='*60}\n[QUERY] {query!r}  →  {run_dir}\n{'='*60}")

        result = _run_one_query(
            query=query, image_bgr=image_bgr, H=H, W=W,
            pipeline_result=pipeline_result,
            locator=None, fp_detector=fp_detector, sam_wrapper=None,
            args=args, run_dir=run_dir, dino_cache=_dino_cache,
        )
        result["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
        results.append(result)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"3.1.2 Visual Check Summary — {image_stem}")
    print(f"{'='*60}")
    _print_timing_summary(results)
    print(f"\nResults saved to: {output_root}")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers (called from main)
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_pipeline_dir(
    args: argparse.Namespace, image_path: Path, output_root: Path,
) -> Path:
    """Return the pipeline cache directory for *image_path*."""
    if args.reuse_pipeline_dir:
        return Path(args.reuse_pipeline_dir)
    # Shared cache: one pipeline run per image.
    return output_root / ".pipeline_cache" / image_path.stem


def _ensure_pipeline(
    args: argparse.Namespace, image_path: Path, pipeline_dir: Path,
    needs_full: bool = False,
) -> tuple:
    """Run or reuse GarmentPipeline.  Lazy: lightweight (YOLO+SAM) by default,
    full pipeline only when *needs_full* is True (fast-path fallback).

    Returns (pipeline_instance, result_dict).
    """
    seg_json = pipeline_dir / "02_samhq" / "segmentation_results.json"
    full_json = pipeline_dir / "05_region_masked_crops" / "region_masked_crops.json"

    if needs_full:
        if full_json.exists():
            print("[CACHE] reused (full pipeline)")
        else:
            print("[CACHE] building full — fast-path fallback required")
            config = build_pipeline_config(args, run_landmark_and_crops=True)
            pipeline = GarmentPipeline(config)
            pipeline.run_image(image_path=str(image_path), output_dir=str(pipeline_dir))
    else:
        if seg_json.exists():
            print("[CACHE] reused (lightweight YOLO+SAM)")
        else:
            print("[CACHE] building lightweight (YOLO+SAM only)")
            config = build_pipeline_config(args, run_landmark_and_crops=False)
            pipeline = GarmentPipeline(config)
            pipeline.run_image(image_path=str(image_path), output_dir=str(pipeline_dir))

    return None, {
        "status": "cached",
        "output_dir": str(pipeline_dir),
        "paths": {
            "segmentation_json": str(seg_json),
            "region_masked_crops_json": str(full_json) if full_json.exists() else None,
        },
    }


def _load_sam(args: argparse.Namespace, pipeline) -> Any:
    if args.reuse_pipeline_dir:
        return None
    if pipeline is not None:
        try:
            return pipeline.get_sam_wrapper()
        except Exception as e:
            print(f"[WARN] Could not load SAM wrapper: {e}")
    return None


def _load_fp_detector(args: argparse.Namespace) -> Any:
    fp_model_path = Path(args.fp_model)
    if not fp_model_path.exists():
        print(f"[INFO] Fashionpedia model not found: {fp_model_path} — DINO-only mode")
        return None
    try:
        from fashion_vision.localization.fashionpedia_part_detector import (
            FashionpediaPartDetector,
        )
        det = FashionpediaPartDetector(str(fp_model_path), device=args.fp_device)
        print(f"[INFO] Fashionpedia detector loaded: {fp_model_path}")
        return det
    except Exception as e:
        print(f"[WARN] Could not load Fashionpedia detector: {e}")
        return None


def _load_dino_locator() -> Any:
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
    return GroundingDINOLocator()


def _print_timing_summary(results: list[dict]) -> None:
    """Print per-query timing broken down by backend."""
    from collections import Counter
    backend_counts: Counter = Counter()
    backend_times: dict[str, float] = {}
    for r in results:
        bk = str(r.get("backend") or "none")
        backend_counts[bk] += 1
        backend_times[bk] = backend_times.get(bk, 0) + r.get("elapsed_seconds", 0)

    print(f"  {'Query':<12} {'Backend':<28} {'Status':<14} {'Score':<8} {'Time(s)'}")
    print(f"  {'-'*12} {'-'*28} {'-'*14} {'-'*8} {'-'*8}")
    for r in results:
        q = str(r.get("query") or "-")
        bk = str(r.get("backend") or "-")
        st = str(r.get("status") or "-")
        sc = r.get("score")
        sc_s = f"{sc:.3f}" if isinstance(sc, (int, float)) else "-"
        et = r.get("elapsed_seconds", 0)
        print(f"  {q:<12} {bk:<28} {st:<14} {sc_s:<8} {et:.1f}s")

    print(f"\n  Backend breakdown:")
    for bk, n in backend_counts.most_common():
        total_t = backend_times.get(bk, 0)
        print(f"    {bk:<30} x{n}  total={total_t:.1f}s  avg={total_t/n:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Core: run a single query and save results
# ═══════════════════════════════════════════════════════════════════════════════


def _load_sam_from_pipeline_dir(pipeline_dir: Path, args: argparse.Namespace) -> Any:
    """Try to load SAM from a GarmentPipeline instance built from cache dir."""
    try:
        config = build_pipeline_config(args, run_landmark_and_crops=False)
        pipeline = GarmentPipeline(config)
        # Trigger SAM load without running full pipeline
        return pipeline.get_sam_wrapper()
    except Exception:
        return None


def _run_one_query(
    query: str,
    image_bgr: "np.ndarray",
    H: int, W: int,
    pipeline_result: dict,
    locator,
    fp_detector,
    sam_wrapper,
    args: argparse.Namespace,
    run_dir: Path,
    dino_cache: list | None = None,
) -> dict:
    """Run one region query and save focused output artifacts."""
    from fashion_vision.localization.region_localization_router import locate_region as _locate_region
    from fashion_vision.localization.garment_ref_filter import filter_instances

    intent = parse_intent(query)

    # ── Fast-path branch ────────────────────────────────────────────────
    if intent.is_fast_path:
        return _run_fast_path_query(
            query, intent, image_bgr, H, W, pipeline_result, args, run_dir,
            pipeline_dir=Path(pipeline_result["output_dir"]),
        )

    # ── Inner garment: SAM-based, no DINO needed ────────────────────────
    if intent.garment_ref == "inner":
        return _run_inner_garment_query(
            query, intent, image_bgr, pipeline_result, sam_wrapper, args, run_dir,
        )

    # ── FP-first: YOLO on full image for ALL Fashionpedia-core parts ────
    from fashion_vision.localization.fashionpedia_part_detector import PART_TO_FP_IDS
    _is_fp = intent.part in PART_TO_FP_IDS

    if fp_detector and _is_fp:
        fp_dets = fp_detector.detect(image_bgr, intent.part, conf=0.25)
        if fp_dets:
            result = _build_fp_direct_result(
                query, intent, image_bgr, fp_dets, sam_wrapper, run_dir,
            )
            save_json(result, run_dir / "result.json")
            print(f"[DONE] {query!r} → backend={result.get('backend')}  status={result['status']}  (FP direct)")
            return result
        # FP miss: for pocket/zipper/sleeve/hood/etc → not_detected, no pipeline.
        if intent.part not in ("neckline", "cuff"):
            result = _build_fp_direct_result(
                query, intent, image_bgr, [], sam_wrapper, run_dir,
            )
            save_json(result, run_dir / "result.json")
            print(f"[DONE] {query!r} → backend=fashionpedia_yolo  status=not_detected  (FP direct)")
            return result
        # neckline/cuff miss → need pipeline for fast-path fallback (below).
        print(f"[FALLBACK] {query!r} FP miss → building pipeline for fast-path fallback")

    # ── Pipeline-dependent path (only neckline/cuff miss, or non-FP DINO) ─
    _needs_full = intent.part in ("neckline", "cuff")
    seg_json = Path(pipeline_result["paths"]["segmentation_json"])
    if not seg_json.exists():
        pipe_dir = Path(pipeline_result["output_dir"])
        _ensure_pipeline(args, Path(args.image), pipe_dir, needs_full=_needs_full)
        pipeline_result = {
            "status": "cached", "output_dir": str(pipe_dir),
            "paths": {
                "segmentation_json": str(pipe_dir / "02_samhq" / "segmentation_results.json"),
                "region_masked_crops_json": str(pipe_dir / "05_region_masked_crops" / "region_masked_crops.json"),
            },
        }

    seg_data = load_json(Path(pipeline_result["paths"]["segmentation_json"]))
    instances = [
        {**seg, "pred_mask_path": seg.get("pred_mask_path") or seg.get("mask_path")}
        for img_item in seg_data.get("images", [])
        for seg in img_item.get("segments", [])
    ]
    filtered_instances = filter_instances(instances, intent.garment_ref)

    # Lazy-load DINO only for non-FP parts (button, placket, belt, etc.).
    # neckline/cuff never reach this point — they fall back to fast-path.
    _needs_dino = not _is_fp
    if locator is None and dino_cache is not None and _needs_dino:
        if dino_cache[0] is None:
            print("[DINO] lazy-loading (first non-FP query)")
            dino_cache[0] = _load_dino_locator()
        locator = dino_cache[0]

    best: dict | None = None
    for inst in filtered_instances:
        r = _locate_region(
            query, inst, image_bgr, W, H,
            locator=locator, sam_wrapper=sam_wrapper,
            fashionpedia_detector=fp_detector,
        )
        if r["status"] == "success" and (best is None or r.get("score", 0) > best.get("score", 0)):
            best = r

    # ── Save artifacts ──────────────────────────────────────────────────
    overlay_out = run_dir / "result_overlay.png"
    mask_out = run_dir / "mask.png"

    if best and best.get("bbox"):
        x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
        ov = image_bgr.copy()
        best_mask = best.get("mask")
        if best_mask is not None and isinstance(best_mask, np.ndarray):
            cv2.imwrite(str(mask_out), (best_mask > 0).astype(np.uint8) * 255)
            color_layer = np.zeros_like(ov)
            color_layer[:] = (0, 255, 0)
            mask_bool = best_mask > 0
            ov[mask_bool] = cv2.addWeighted(ov, 1.0, color_layer, 0.4, 0)[mask_bool]
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{intent.part or 'query'} {best['score']:.2f} [{best.get('backend', '?')}]"
        cv2.putText(ov, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(overlay_out), ov)

    # ── Build result dict ───────────────────────────────────────────────
    result: dict = {
        "query": query,
        "part": intent.part,
        "backend": best.get("backend") if best else None,
        "method": best.get("method") if best else None,
        "status": "success" if best else "not_detected",
        "score": best.get("score") if best else None,
        "bbox": best.get("bbox") if best else None,
        "mask_source": best.get("mask_source") if best else None,
        "garment_ref": intent.garment_ref,
        "garment_ref_matched": best.get("garment_ref_matched") if best else None,
        "is_zero_shot": intent.is_zero_shot,
        "direction": intent.direction,
        "outputs": {
            "overlay": str(overlay_out) if best else None,
            "mask": str(mask_out) if best and best.get("mask") is not None else None,
        },
    }
    if best:
        debug = best.get("debug", {})
        if isinstance(debug, dict) and args.save_debug_artifacts:
            result["debug"] = debug

    save_json(result, run_dir / "result.json")
    print(f"[DONE] {query!r} → backend={result.get('backend')}  status={result['status']}")
    return result


def _run_inner_garment_query(
    query: str, intent, image_bgr: np.ndarray,
    pipeline_result: dict, sam_wrapper, args: argparse.Namespace, run_dir: Path,
) -> dict:
    """Run inner garment detection (SAM-based, no DINO needed)."""
    from fashion_vision.localization.inner_garment_detector import (
        detect_inner_garment_from_sam,
    )

    # Build pipeline if needed + load SAM.
    seg_json = Path(pipeline_result["paths"]["segmentation_json"])
    pipe_dir = Path(pipeline_result["output_dir"])
    if not seg_json.exists():
        _ensure_pipeline(args, Path(args.image), pipe_dir, needs_full=False)
        seg_json = pipe_dir / "02_samhq" / "segmentation_results.json"
    if sam_wrapper is None:
        sam_wrapper = _load_sam_from_pipeline_dir(pipe_dir, args)

    seg_data = load_json(seg_json)
    instances = [
        {**seg, "pred_mask_path": seg.get("pred_mask_path") or seg.get("mask_path")}
        for img_item in seg_data.get("images", [])
        for seg in img_item.get("segments", [])
    ]

    # Outwear = coarse class "outerwear" OR fine class with "outwear" in name.
    _OUTWEAR_FINE = {"short sleeve outwear", "long sleeve outwear"}
    best: dict | None = None
    outer_count = 0
    print(f"  [inner] total instances: {len(instances)}", flush=True)
    for idx, inst in enumerate(instances):
        coarse = inst.get("coarse_class_name", "")
        fine = (inst.get("fine_class_name") or inst.get("class_name") or "").strip().lower()
        print(f"  [inner]   inst[{idx}]: coarse={coarse!r} fine={fine!r}", flush=True)
        is_outer = (coarse == "outerwear" or fine in _OUTWEAR_FINE)
        if not is_outer:
            continue
        outer_count += 1
        print(f"  [inner]   -> outerwear found, running SAM inner detection...", flush=True)
        if sam_wrapper is None:
            sam_wrapper = _load_sam_from_pipeline_dir(pipe_dir, args)
        if sam_wrapper is None:
            print(f"  [inner] SAM not available, skipping", flush=True)
            continue
        inner = detect_inner_garment_from_sam(image_bgr, inst, sam_wrapper)
        print(f"  [inner]   -> inner detection result: {'FOUND' if inner is not None else 'None'}", flush=True)
        if inner is not None:
            best = inner
            break
    print(f"  [inner] outerwear instances: {outer_count}, inner found: {best is not None}", flush=True)

    overlay_out = run_dir / "result_overlay.png"
    mask_out = run_dir / "mask.png"

    if best:
        inner_mask = best.get("mask")
        bbox = [int(v) for v in best["bbox_xyxy"]]
        overlay = image_bgr.copy()
        # Draw SAM mask as translucent colored overlay (面, not 框).
        if inner_mask is not None and isinstance(inner_mask, np.ndarray) and inner_mask.sum() > 0:
            cv2.imwrite(str(mask_out), (inner_mask > 0).astype(np.uint8) * 255)
            color_layer = np.zeros_like(overlay, dtype=np.uint8)
            color_layer[:] = (255, 128, 0)  # blue-orange for inner garment
            mask_bool = inner_mask > 0
            blended = cv2.addWeighted(overlay, 0.6, color_layer, 0.4, 0)
            overlay[mask_bool] = blended[mask_bool]
            # Thin bbox outline on top of mask.
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 128, 0), 2)
        else:
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        label = "inner_garment"
        cv2.putText(overlay, label, (bbox[0], max(bbox[1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
        cv2.imwrite(str(overlay_out), overlay)

    result: dict = {
        "query": query, "part": intent.part, "backend": "inner_garment",
        "status": "success" if best else "not_detected",
        "score": best.get("score") if best else None,
        "bbox": best.get("bbox_xyxy") if best else None,
        "mask_source": "sam_inner_garment" if best else None,
        "garment_ref": intent.garment_ref, "garment_ref_matched": True,
        "is_zero_shot": False, "direction": intent.direction,
        "num_outerwear_instances": outer_count,
        "outputs": {"overlay": str(overlay_out) if best else None,
                      "mask": str(mask_out) if best else None},
    }
    save_json(result, run_dir / "result.json")
    print(f"[DONE] {query!r} → backend=inner_garment  status={result['status']}")
    return result


def _build_fp_direct_result(
    query: str, intent, image_bgr: np.ndarray,
    fp_dets: list[dict], sam_wrapper, run_dir: Path,
) -> dict:
    """Build result dict from Fashionpedia YOLO detections on full image (no pipeline).

    Draws ALL detections, not just top-1.
    """
    overlay_out = run_dir / "result_overlay.png"
    mask_out = run_dir / "mask.png"

    if fp_dets:
        COLORS = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]
        ov = image_bgr.copy()
        all_bboxes = []
        all_scores = []
        for idx, det in enumerate(fp_dets):
            color = COLORS[idx % len(COLORS)]
            bbox = [int(v) for v in det["bbox_xyxy"]]
            all_bboxes.append(bbox)
            all_scores.append(det["score"])
            cv2.rectangle(ov, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"{intent.part or 'query'} #{idx+1} {det['score']:.2f} [fashionpedia_yolo]"
            cv2.putText(ov, label, (bbox[0], max(bbox[1] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imwrite(str(overlay_out), ov)

        return {
            "query": query, "part": intent.part, "backend": "fashionpedia_yolo",
            "status": "success", "score": all_scores[0], "bbox": all_bboxes[0],
            "all_bboxes": all_bboxes, "all_scores": all_scores,
            "num_detections": len(fp_dets),
            "mask_source": None, "garment_ref": intent.garment_ref,
            "garment_ref_matched": True, "is_zero_shot": False,
            "direction": intent.direction,
            "outputs": {"overlay": str(overlay_out), "mask": None},
        }
    else:
        return {
            "query": query, "part": intent.part, "backend": "fashionpedia_yolo",
            "status": "not_detected", "reason": "fashionpedia_no_detection",
            "score": None, "bbox": None, "mask_source": None,
            "garment_ref": intent.garment_ref, "garment_ref_matched": None,
            "is_zero_shot": False, "direction": intent.direction,
            "outputs": {"overlay": None, "mask": None},
        }


def _run_fast_path_query(
    query: str, intent, image_bgr: "np.ndarray", H: int, W: int,
    pipeline_result: dict, args: argparse.Namespace, run_dir: Path,
    pipeline_dir: Path,
) -> dict:
    """Run a fast-path (landmark) query and save focused artifacts."""
    target_region = intent.crop_region
    prefer_component = args.prefer_component.strip().lower() or intent.component or None
    from fashion_vision.localization.garment_ref_filter import garment_ref_to_target_class
    effective_target_class = args.target_class.strip() or garment_ref_to_target_class(intent.garment_ref)

    # Ensure full pipeline (landmarks + region crops) is built.
    full_json = pipeline_dir / "05_region_masked_crops" / "region_masked_crops.json"
    if not full_json.exists():
        _ensure_pipeline(args, Path(args.image), pipeline_dir, needs_full=True)
    masked_json = full_json
    if not masked_json.exists():
        raise FileNotFoundError(f"region_masked_crops.json not found: {masked_json}")
    masked_data = load_json(masked_json)

    selected, candidates = select_best_record(
        masked_crops_data=masked_data, target_region=target_region,
        target_component=prefer_component, target_class=effective_target_class,
        target_det_id=args.target_det_id,
    )

    if selected is None:
        result = {
            "query": query, "part": intent.part, "backend": "fast_path",
            "status": "failed", "reason": "no_matching_region_crop",
            "garment_ref": intent.garment_ref, "is_zero_shot": False,
        }
        save_json(result, run_dir / "result.json")
        print(f"[DONE] {query!r} → backend=fast_path  status=failed (no matching crop)")
        return result

    bbox_xyxy = clip_bbox_xyxy_to_image(
        [int(v) for v in selected["bbox_xyxy"]], image_w=W, image_h=H,
    )
    full_mask = create_full_size_region_mask(
        image_shape=image_bgr.shape, bbox_xyxy=bbox_xyxy,
        mask_crop_path=str(selected["mask_crop_path"]),
    )
    label = (
        f"{REGION_DISPLAY_NAME.get(target_region, target_region)} "
        f"det{selected.get('det_id')} {selected.get('class_name')}"
    )
    overlay = draw_selected_overlay(image_bgr, full_mask, bbox_xyxy, label, float(args.overlay_alpha))

    overlay_out = run_dir / "result_overlay.png"
    mask_out = run_dir / "mask.png"
    cv2.imwrite(str(overlay_out), overlay)
    cv2.imwrite(str(mask_out), full_mask)

    result: dict = {
        "query": query, "part": intent.part, "backend": "fast_path",
        "method": "fast_path", "status": "success",
        "score": None, "bbox": bbox_xyxy, "mask_source": "landmark_crop",
        "garment_ref": intent.garment_ref,
        "garment_ref_matched": True,
        "is_zero_shot": False,
        "outputs": {"overlay": str(overlay_out), "mask": str(mask_out)},
    }
    save_json(result, run_dir / "result.json")
    print(f"[DONE] {query!r} → backend=fast_path  status=success")
    return result


if __name__ == "__main__":
    main()
