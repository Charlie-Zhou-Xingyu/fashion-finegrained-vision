from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig


REGION_ALIASES = {
    "collar": [
        "collar",
        "neckline",
        "neck",
        "领口",
        "衣领",
        "领子",
        "领部",
        "脖子",
        "脖颈",
    ],
    "sleeve": [
        "sleeve",
        "sleeves",
        "袖子",
        "衣袖",
        "袖口",
        "左袖",
        "右袖",
        "袖部",
        "袖",
    ],
    "hem": [
        "hem",
        "下摆",
        "衣摆",
        "底边",
        "下边",
        "边缘",
    ],
    "waist": [
        "waist",
        "腰",
        "腰部",
        "腰线",
        "裤腰",
        "裙腰",
        "收腰",
        "腰围",
    ],
    "pant_leg": [
        "pant_leg",
        "pants leg",
        "trouser leg",
        "裤腿",
        "裤管",
        "腿部",
        "裤脚",
    ],
}


SPECIAL_QUERY_ALIASES = {
    "skirt_hem": [
        "裙摆",
        "裙子下摆",
        "半裙下摆",
        "裙底",
        "裙边",
    ],
    "dress_hem": [
        "连衣裙下摆",
        "连衣裙裙摆",
        "裙装下摆",
    ],
}


COMPONENT_ALIASES = {
    "left_sleeve": [
        "left sleeve",
        "左袖",
        "左边袖子",
        "左侧袖子",
        "左衣袖",
    ],
    "right_sleeve": [
        "right sleeve",
        "右袖",
        "右边袖子",
        "右侧袖子",
        "右衣袖",
    ],
}


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
    parser.add_argument("--query", type=str, required=True, help="Natural language query.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/query_region_online_demo",
        help="Output root directory.",
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
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for selected region mask.",
    )

    return parser.parse_args()


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


def infer_region_from_query(
    query: str,
) -> tuple[str | None, str | None, str, dict[str, Any]]:
    """
    Lightweight rule-based query parser.

    Returns:
        target_region:
            collar / sleeve / hem / waist / pant_leg / None
        target_component:
            left_sleeve / right_sleeve / None
        inferred_target_class:
            e.g. skirt for 裙摆, or empty string
        debug:
            matched info
    """
    q = str(query).strip().lower()

    debug: dict[str, Any] = {
        "query": query,
        "matched_region_alias": None,
        "matched_component_alias": None,
        "matched_special_alias": None,
        "inferred_target_class": "",
    }

    component = None
    for comp, aliases in COMPONENT_ALIASES.items():
        for alias in aliases:
            if alias.lower() in q:
                component = comp
                debug["matched_component_alias"] = alias
                break

        if component is not None:
            break

    # Special query first:
    # 裙摆 should not be treated as generic 下摆.
    inferred_target_class = ""
    for special_name, aliases in SPECIAL_QUERY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in q:
                debug["matched_special_alias"] = alias

                if special_name == "skirt_hem":
                    debug["matched_region_alias"] = alias
                    debug["inferred_target_class"] = "skirt"
                    return "hem", component, "skirt", debug

                if special_name == "dress_hem":
                    debug["matched_region_alias"] = alias
                    debug["inferred_target_class"] = "__dress__"
                    return "hem", component, "__dress__", debug

    region = None
    for region_name, aliases in REGION_ALIASES.items():
        for alias in aliases:
            if alias.lower() in q:
                region = region_name
                debug["matched_region_alias"] = alias
                break

        if region is not None:
            break

    if region is None and component in {"left_sleeve", "right_sleeve"}:
        region = "sleeve"

    return region, component, inferred_target_class, debug


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


def target_class_matches(class_name: Any, target_class: str) -> bool:
    """
    Match user-provided or query-inferred class hint.

    Special target_class:
        __dress__ means any dress class.
    """
    if not target_class:
        return True

    cls = str(class_name).strip().lower()
    target = str(target_class).strip().lower()

    if target == "__dress__":
        return cls in DRESS_CLASSES

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


def build_pipeline_config(args: argparse.Namespace) -> GarmentPipelineConfig:
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
    )


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    target_region, query_component, inferred_target_class, query_debug = infer_region_from_query(args.query)

    prefer_component = args.prefer_component.strip().lower() or query_component

    if prefer_component == "":
        prefer_component = None

    effective_target_class = args.target_class.strip()
    if not effective_target_class:
        effective_target_class = inferred_target_class

    if target_region is None:
        raise ValueError(
            f"Cannot infer target region from query: {args.query!r}. "
            f"Supported regions: {list(REGION_ALIASES.keys())}"
        )

    output_root = Path(args.output_dir)
    run_name = (
        f"{image_path.stem}_"
        f"{sanitize_filename_part(target_region)}_"
        f"{now_tag()}"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_dir = run_dir / "pipeline"

    if args.reuse_pipeline_dir:
        pipeline_dir = Path(args.reuse_pipeline_dir)
        masked_json_path = pipeline_dir / "05_region_masked_crops" / "region_masked_crops.json"

        if not masked_json_path.exists():
            raise FileNotFoundError(
                f"--reuse-pipeline-dir was set, but region_masked_crops.json "
                f"not found: {masked_json_path}"
            )

        pipeline_result = {
            "status": "reused",
            "output_dir": str(pipeline_dir),
            "paths": {
                "region_masked_crops_json": str(masked_json_path),
            },
        }
    else:
        config = build_pipeline_config(args)
        pipeline = GarmentPipeline(config)
        pipeline_result = pipeline.run_image(
            image_path=str(image_path),
            output_dir=str(pipeline_dir),
        )

    masked_json = Path(pipeline_result["paths"]["region_masked_crops_json"])

    if not masked_json.exists():
        raise FileNotFoundError(f"region_masked_crops.json not found: {masked_json}")

    masked_data = load_json(masked_json)

    selected, candidates = select_best_record(
        masked_crops_data=masked_data,
        target_region=target_region,
        target_component=prefer_component,
        target_class=effective_target_class,
        target_det_id=args.target_det_id,
    )

    if selected is None:
        result = {
            "status": "failed",
            "error": "no_matching_region_crop",
            "image": str(image_path),
            "query": args.query,
            "target_region": target_region,
            "target_component": prefer_component,
            "target_class": effective_target_class,
            "pipeline_result": pipeline_result,
            "query_debug": query_debug,
            "num_candidates": 0,
        }
        save_json(result, run_dir / "result.json")
        raise RuntimeError(
            f"No matching crop found for query={args.query!r}, "
            f"region={target_region!r}, class={effective_target_class!r}"
        )

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    bbox_xyxy = [int(v) for v in selected["bbox_xyxy"]]
    bbox_xyxy = clip_bbox_xyxy_to_image(
        bbox_xyxy=bbox_xyxy,
        image_w=image_bgr.shape[1],
        image_h=image_bgr.shape[0],
    )

    mask_crop_path = selected.get("mask_crop_path")

    if not mask_crop_path:
        raise RuntimeError("Selected record has no mask_crop_path.")

    full_mask = create_full_size_region_mask(
        image_shape=image_bgr.shape,
        bbox_xyxy=bbox_xyxy,
        mask_crop_path=str(mask_crop_path),
    )

    label = (
        f"{REGION_DISPLAY_NAME.get(target_region, target_region)} "
        f"det{selected.get('det_id')} "
        f"{selected.get('class_name')}"
    )

    overlay = draw_selected_overlay(
        image_bgr=image_bgr,
        full_mask=full_mask,
        bbox_xyxy=bbox_xyxy,
        label=label,
        alpha=float(args.overlay_alpha),
    )

    original_out = run_dir / "original.jpg"
    full_mask_out = run_dir / "region_mask_full.png"
    overlay_out = run_dir / "region_overlay.jpg"

    cv2.imwrite(str(original_out), image_bgr)
    cv2.imwrite(str(full_mask_out), full_mask)
    cv2.imwrite(str(overlay_out), overlay)

    selected_image_crop_out = copy_if_exists(
        selected.get("image_crop_path"),
        run_dir / "selected_image_crop.png",
    )
    selected_mask_crop_out = copy_if_exists(
        selected.get("mask_crop_path"),
        run_dir / "selected_mask_crop.png",
    )
    selected_masked_crop_out = copy_if_exists(
        selected.get("masked_crop_path"),
        run_dir / "selected_masked_crop.png",
    )

    result = {
        "status": "success",
        "image": str(image_path),
        "query": args.query,
        "target_region": target_region,
        "target_region_display": REGION_DISPLAY_NAME.get(target_region, target_region),
        "target_component": prefer_component,
        "target_class": effective_target_class,
        "query_debug": query_debug,
        "selection": {
            "rule": "deterministic_rule_based_selection",
            "reason": selected.get("_selection_reason"),
            "rank": selected.get("_selection_rank"),
        },
        "selected": {
            k: v
            for k, v in selected.items()
            if not str(k).startswith("_")
        },
        "num_candidates": len(candidates),
        "top_candidates": [
            {
                "rank": c.get("_selection_rank"),
                "image_path": c.get("image_path"),
                "class_name": c.get("class_name"),
                "garment_group": garment_group(c.get("class_name", "")),
                "det_id": c.get("det_id"),
                "region": c.get("region"),
                "component": c.get("component"),
                "bbox_xyxy": c.get("bbox_xyxy"),
                "mask_area_ratio": c.get("mask_area_ratio"),
                "masked_crop_path": c.get("masked_crop_path"),
            }
            for c in candidates[:10]
        ],
        "outputs": {
            "run_dir": str(run_dir),
            "original": str(original_out),
            "region_mask_full": str(full_mask_out),
            "region_overlay": str(overlay_out),
            "selected_image_crop": selected_image_crop_out,
            "selected_mask_crop": selected_mask_crop_out,
            "selected_masked_crop": selected_masked_crop_out,
            "result_json": str(run_dir / "result.json"),
            "pipeline_dir": str(pipeline_dir),
            "region_masked_crops_json": str(masked_json),
        },
        "pipeline_result": pipeline_result,
    }

    save_json(result, run_dir / "result.json")

    print("[INFO] Query region demo finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
