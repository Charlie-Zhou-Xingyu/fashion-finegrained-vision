from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(path: str | os.PathLike | None) -> None:
    if path is None:
        return

    path_str = str(path)
    if not path_str:
        return

    Path(path_str).mkdir(parents=True, exist_ok=True)


def load_json(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {path}, got {type(data).__name__}")

    return data


def save_json(obj: Any, path: str | os.PathLike) -> None:
    parent = os.path.dirname(str(path))
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_path(p: Any) -> str:
    return os.path.normpath(str(p))


def build_segment_index(segmentation_data: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """
    Build index:
      (image_path_norm, det_id) -> segment

    Also index by file_name for robustness.
    """
    index: dict[tuple[str, int], dict[str, Any]] = {}

    for img_item in segmentation_data.get("images", []):
        if not isinstance(img_item, dict):
            continue

        image_path = norm_path(img_item.get("image_path", ""))
        file_name = img_item.get("file_name", "")

        segments = img_item.get("segments", [])
        if not isinstance(segments, list):
            continue

        for seg in segments:
            if not isinstance(seg, dict):
                continue

            try:
                det_id = int(seg.get("det_id", -1))
            except Exception:
                continue

            index[(image_path, det_id)] = seg

            if file_name:
                index[(norm_path(file_name), det_id)] = seg

    return index


def find_segment(
    segment_index: dict[tuple[str, int], dict[str, Any]],
    image_path: Any,
    det_id: Any,
) -> dict[str, Any] | None:
    image_path_norm = norm_path(image_path)

    try:
        det_id_int = int(det_id)
    except Exception:
        return None

    if (image_path_norm, det_id_int) in segment_index:
        return segment_index[(image_path_norm, det_id_int)]

    file_name = os.path.basename(image_path_norm)
    if (norm_path(file_name), det_id_int) in segment_index:
        return segment_index[(norm_path(file_name), det_id_int)]

    return None


def clip_bbox_xyxy(
    bbox: Any,
    width: int,
    height: int,
) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = bbox
        x1 = int(round(float(x1)))
        y1 = int(round(float(y1)))
        x2 = int(round(float(x2)))
        y2 = int(round(float(y2)))
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)

    if y2 <= y1:
        y2 = min(height, y1 + 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def sanitize_name(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        text = default

    return (
        text.replace(" ", "_")
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


def make_output_name(record: dict[str, Any]) -> str:
    image_name = Path(str(record.get("image_path", "image"))).stem

    try:
        det_id = int(record.get("det_id", -1))
        det_text = f"det{det_id:03d}"
    except Exception:
        det_text = f"det{sanitize_name(record.get('det_id', 'NA'))}"

    class_name = sanitize_name(record.get("class_name", "unknown"))
    region = sanitize_name(record.get("region", "region"))
    component = sanitize_name(record.get("component", region))

    return f"{image_name}_{det_text}_{class_name}_{component}.png"


def ensure_2d_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """
    Ensure mask is a 2D array [H, W].

    Accepted inputs:
      - [H, W]
      - [H, W, 1]
      - [H, W, 3]
      - [H, W, 4]
      - squeeze-able arrays
    """
    if mask is None:
        return None

    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        channels = mask.shape[2]

        if channels == 1:
            return mask[:, :, 0]

        if channels >= 3:
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


def read_mask_as_binary(mask_path: str | os.PathLike | None) -> np.ndarray | None:
    """
    Read mask from disk and normalize to 2D uint8 binary mask, values 0 or 255.
    """
    if mask_path is None:
        return None

    mask_path_str = str(mask_path)
    if not mask_path_str:
        return None

    # IMREAD_UNCHANGED is more robust for grayscale / RGB / RGBA PNG.
    mask = cv2.imread(mask_path_str, cv2.IMREAD_UNCHANGED)

    if mask is None:
        return None

    return ensure_binary_mask_2d(mask)


def apply_mask_to_crop(
    image_crop: np.ndarray,
    mask_crop: np.ndarray,
    background: str = "white",
    transparent: bool = False,
) -> np.ndarray:
    """
    Apply a 2D binary mask to an image crop.

    Args:
        image_crop:
            BGR crop, shape [H, W, 3].
        mask_crop:
            Mask crop. Accepted shapes:
                - [H, W]
                - [H, W, 1]
                - [H, W, 3]
                - [H, W, 4]
        background:
            white / black / gray.
        transparent:
            If True, output BGRA with alpha from mask.

    Returns:
        Masked crop image.
    """
    if image_crop is None or not isinstance(image_crop, np.ndarray) or image_crop.size <= 0:
        raise ValueError("image_crop is empty or invalid.")

    if image_crop.ndim != 3 or image_crop.shape[2] != 3:
        raise ValueError(f"Expected image_crop shape [H, W, 3], got {image_crop.shape}")

    mask_bin = ensure_binary_mask_2d(mask_crop)
    if mask_bin is None:
        raise ValueError("mask_crop is None or cannot be converted to 2D mask.")

    if mask_bin.shape[:2] != image_crop.shape[:2]:
        mask_bin = cv2.resize(
            mask_bin,
            (image_crop.shape[1], image_crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_bool = mask_bin > 0

    if transparent:
        rgba = cv2.cvtColor(image_crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = np.where(mask_bool, 255, 0).astype(np.uint8)
        return rgba

    if background == "black":
        bg = np.zeros_like(image_crop)
    elif background == "gray":
        bg = np.full_like(image_crop, 127)
    else:
        bg = np.full_like(image_crop, 255)

    out = bg.copy()

    # mask_bool is [H, W], image_crop/out are [H, W, 3].
    # This selects foreground pixels and copies all channels.
    out[mask_bool] = image_crop[mask_bool]

    return out


def run_apply_samhq_mask(args: argparse.Namespace) -> dict[str, Any]:
    region_data = load_json(args.region_crops_json)
    segmentation_data = load_json(args.segmentation_json)

    output_dir = str(args.output_dir)
    image_crop_dir = os.path.join(output_dir, "image_crops")
    mask_crop_dir = os.path.join(output_dir, "mask_crops")
    masked_crop_dir = os.path.join(output_dir, "masked_crops")

    ensure_dir(image_crop_dir)
    ensure_dir(mask_crop_dir)
    ensure_dir(masked_crop_dir)

    segment_index = build_segment_index(segmentation_data)

    output_records: list[dict[str, Any]] = []

    num_total = 0
    num_success = 0
    num_failed = 0
    num_missing_segment = 0
    num_missing_image = 0
    num_missing_mask = 0
    num_empty_mask = 0
    num_invalid_bbox = 0
    num_invalid_crop = 0
    num_invalid_mask = 0

    region_counts: dict[str, int] = {}
    component_counts: dict[str, int] = {}

    crops = region_data.get("crops", [])
    if not isinstance(crops, list):
        raise ValueError("Invalid region crops JSON: 'crops' must be a list.")

    for record in tqdm(crops, desc="Apply SAM-HQ mask"):
        num_total += 1

        if not isinstance(record, dict):
            num_failed += 1
            output_records.append(
                {
                    "masked_success": False,
                    "masked_error": "invalid_record",
                    "raw_record": record,
                }
            )
            continue

        out_record = dict(record)
        out_record["masked_success"] = False
        out_record["mask_area_ratio"] = 0.0
        out_record["mask_crop_path"] = None
        out_record["masked_crop_path"] = None
        out_record["image_crop_path"] = None

        if not record.get("success", False):
            out_record["masked_error"] = "region_crop_not_success"
            output_records.append(out_record)
            num_failed += 1
            continue

        image_path = record.get("image_path")
        det_id = record.get("det_id", -1)

        segment = find_segment(segment_index, image_path, det_id)

        if segment is None:
            out_record["masked_error"] = "missing_segment"
            output_records.append(out_record)
            num_failed += 1
            num_missing_segment += 1
            continue

        mask_path = segment.get("mask_path")

        if not image_path:
            out_record["masked_error"] = "missing_image_path"
            output_records.append(out_record)
            num_failed += 1
            num_missing_image += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            out_record["masked_error"] = "missing_image"
            output_records.append(out_record)
            num_failed += 1
            num_missing_image += 1
            continue

        mask = read_mask_as_binary(mask_path)
        if mask is None:
            out_record["masked_error"] = "missing_or_invalid_mask"
            output_records.append(out_record)
            num_failed += 1
            num_missing_mask += 1
            continue

        h, w = image.shape[:2]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = ensure_binary_mask_2d(mask)

            if mask is None:
                out_record["masked_error"] = "invalid_resized_mask"
                output_records.append(out_record)
                num_failed += 1
                num_invalid_mask += 1
                continue

        bbox = clip_bbox_xyxy(record.get("bbox_xyxy"), w, h)

        if bbox is None:
            out_record["masked_error"] = "invalid_bbox"
            output_records.append(out_record)
            num_failed += 1
            num_invalid_bbox += 1
            continue

        x1, y1, x2, y2 = bbox

        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        if image_crop.size <= 0 or mask_crop.size <= 0:
            out_record["masked_error"] = "empty_crop"
            output_records.append(out_record)
            num_failed += 1
            num_invalid_crop += 1
            continue

        mask_bin = ensure_binary_mask_2d(mask_crop)

        if mask_bin is None:
            out_record["masked_error"] = "invalid_mask_crop"
            output_records.append(out_record)
            num_failed += 1
            num_invalid_mask += 1
            continue

        if mask_bin.shape[:2] != image_crop.shape[:2]:
            mask_bin = cv2.resize(
                mask_bin,
                (image_crop.shape[1], image_crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bin = ensure_binary_mask_2d(mask_bin)

            if mask_bin is None:
                out_record["masked_error"] = "invalid_resized_mask_crop"
                output_records.append(out_record)
                num_failed += 1
                num_invalid_mask += 1
                continue

        crop_area = max(1, int(mask_bin.shape[0] * mask_bin.shape[1]))
        mask_area = int((mask_bin > 0).sum())
        mask_area_ratio = float(mask_area / crop_area)

        if mask_area_ratio < float(args.min_mask_area_ratio):
            out_record["masked_error"] = "empty_or_too_small_mask"
            out_record["mask_area_ratio"] = mask_area_ratio
            output_records.append(out_record)
            num_failed += 1
            num_empty_mask += 1
            continue

        try:
            masked_crop = apply_mask_to_crop(
                image_crop=image_crop,
                mask_crop=mask_bin,
                background=str(args.background),
                transparent=bool(args.transparent),
            )
        except Exception as exc:
            out_record["masked_error"] = "apply_mask_failed"
            out_record["masked_exception"] = repr(exc)
            output_records.append(out_record)
            num_failed += 1
            num_invalid_mask += 1
            continue

        region = sanitize_name(record.get("region", "region"), default="region")
        component = sanitize_name(record.get("component", region), default=region)
        out_name = make_output_name(record)

        region_image_dir = os.path.join(image_crop_dir, region)
        region_mask_dir = os.path.join(mask_crop_dir, region)
        region_masked_dir = os.path.join(masked_crop_dir, region)

        ensure_dir(region_image_dir)
        ensure_dir(region_mask_dir)
        ensure_dir(region_masked_dir)

        image_crop_path = os.path.join(region_image_dir, out_name)
        mask_crop_path = os.path.join(region_mask_dir, out_name)
        masked_crop_path = os.path.join(region_masked_dir, out_name)

        cv2.imwrite(image_crop_path, image_crop)
        cv2.imwrite(mask_crop_path, mask_bin)
        cv2.imwrite(masked_crop_path, masked_crop)

        out_record["bbox_xyxy"] = bbox
        out_record["segment_mask_path"] = mask_path
        out_record["image_crop_path"] = image_crop_path
        out_record["mask_crop_path"] = mask_crop_path
        out_record["masked_crop_path"] = masked_crop_path
        out_record["mask_area_ratio"] = mask_area_ratio
        out_record["masked_success"] = True
        out_record["masked_error"] = None

        output_records.append(out_record)

        num_success += 1
        region_counts[region] = region_counts.get(region, 0) + 1
        component_counts[component] = component_counts.get(component, 0) + 1

    summary = {
        "region_crops_json": str(args.region_crops_json),
        "segmentation_json": str(args.segmentation_json),
        "output_dir": output_dir,
        "background": str(args.background),
        "transparent": bool(args.transparent),
        "min_mask_area_ratio": float(args.min_mask_area_ratio),
        "num_total": num_total,
        "num_success": num_success,
        "num_failed": num_failed,
        "num_missing_segment": num_missing_segment,
        "num_missing_image": num_missing_image,
        "num_missing_mask": num_missing_mask,
        "num_empty_mask": num_empty_mask,
        "num_invalid_bbox": num_invalid_bbox,
        "num_invalid_crop": num_invalid_crop,
        "num_invalid_mask": num_invalid_mask,
        "region_counts": region_counts,
        "component_counts": component_counts,
    }

    output = {
        "summary": summary,
        "crops": output_records,
    }

    output_json = os.path.join(output_dir, "region_masked_crops.json")
    save_json(output, output_json)

    print("[INFO] Apply SAM-HQ mask finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Output JSON: {output_json}")
    print(f"[INFO] Masked crop dir: {masked_crop_dir}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply SAM-HQ instance masks to semantic region crops."
    )
    parser.add_argument("--region-crops-json", required=True)
    parser.add_argument("--segmentation-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--background",
        default="white",
        choices=["white", "black", "gray"],
    )
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_apply_samhq_mask(args)


if __name__ == "__main__":
    main()
