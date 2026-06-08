import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_path(p):
    return os.path.normpath(str(p))


def build_segment_index(segmentation_data):
    """
    Build index:
      (image_path_norm, det_id) -> segment
    """
    index = {}

    for img_item in segmentation_data.get("images", []):
        image_path = norm_path(img_item.get("image_path", ""))
        file_name = img_item.get("file_name", "")

        for seg in img_item.get("segments", []):
            det_id = int(seg.get("det_id", -1))

            index[(image_path, det_id)] = seg

            # Also index by file_name to be more robust.
            if file_name:
                index[(norm_path(file_name), det_id)] = seg

    return index


def find_segment(segment_index, image_path, det_id):
    image_path_norm = norm_path(image_path)
    det_id = int(det_id)

    if (image_path_norm, det_id) in segment_index:
        return segment_index[(image_path_norm, det_id)]

    file_name = os.path.basename(image_path_norm)
    if (norm_path(file_name), det_id) in segment_index:
        return segment_index[(norm_path(file_name), det_id)]

    return None


def clip_bbox_xyxy(bbox, width, height):
    x1, y1, x2, y2 = bbox

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return [x1, y1, x2, y2]


def make_output_name(record):
    image_name = Path(record["image_path"]).stem
    det_id = int(record.get("det_id", -1))
    class_name = str(record.get("class_name", "unknown")).replace(" ", "_")
    region = str(record.get("region", "region"))
    component = str(record.get("component", region))

    return f"{image_name}_det{det_id:03d}_{class_name}_{component}.png"


def apply_mask_to_crop(image_crop, mask_crop, background="white", transparent=False):
    mask_bool = mask_crop > 0

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
    out[mask_bool] = image_crop[mask_bool]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-crops-json", required=True)
    parser.add_argument("--segmentation-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--background", default="white", choices=["white", "black", "gray"])
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.005)
    args = parser.parse_args()

    region_data = load_json(args.region_crops_json)
    segmentation_data = load_json(args.segmentation_json)

    output_dir = args.output_dir
    image_crop_dir = os.path.join(output_dir, "image_crops")
    mask_crop_dir = os.path.join(output_dir, "mask_crops")
    masked_crop_dir = os.path.join(output_dir, "masked_crops")

    ensure_dir(image_crop_dir)
    ensure_dir(mask_crop_dir)
    ensure_dir(masked_crop_dir)

    segment_index = build_segment_index(segmentation_data)

    output_records = []

    num_total = 0
    num_success = 0
    num_failed = 0
    num_missing_segment = 0
    num_missing_image = 0
    num_missing_mask = 0
    num_empty_mask = 0

    region_counts = {}
    component_counts = {}

    for record in tqdm(region_data.get("crops", []), desc="Apply SAM-HQ mask"):
        num_total += 1

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
        det_id = int(record.get("det_id", -1))
        segment = find_segment(segment_index, image_path, det_id)

        if segment is None:
            out_record["masked_error"] = "missing_segment"
            output_records.append(out_record)
            num_failed += 1
            num_missing_segment += 1
            continue

        mask_path = segment.get("mask_path")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            out_record["masked_error"] = "missing_image"
            output_records.append(out_record)
            num_failed += 1
            num_missing_image += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            out_record["masked_error"] = "missing_mask"
            output_records.append(out_record)
            num_failed += 1
            num_missing_mask += 1
            continue

        h, w = image.shape[:2]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        bbox = clip_bbox_xyxy(record.get("bbox_xyxy"), w, h)
        x1, y1, x2, y2 = bbox

        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        mask_bin = np.where(mask_crop > 0, 255, 0).astype(np.uint8)

        crop_area = max(1, int((x2 - x1) * (y2 - y1)))
        mask_area = int((mask_bin > 0).sum())
        mask_area_ratio = float(mask_area / crop_area)

        if mask_area_ratio < args.min_mask_area_ratio:
            out_record["masked_error"] = "empty_or_too_small_mask"
            out_record["mask_area_ratio"] = mask_area_ratio
            output_records.append(out_record)
            num_failed += 1
            num_empty_mask += 1
            continue

        masked_crop = apply_mask_to_crop(
            image_crop=image_crop,
            mask_crop=mask_bin,
            background=args.background,
            transparent=args.transparent,
        )

        region = str(record.get("region", "region"))
        component = str(record.get("component", region))
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
        "region_crops_json": args.region_crops_json,
        "segmentation_json": args.segmentation_json,
        "output_dir": output_dir,
        "background": args.background,
        "transparent": bool(args.transparent),
        "min_mask_area_ratio": args.min_mask_area_ratio,
        "num_total": num_total,
        "num_success": num_success,
        "num_failed": num_failed,
        "num_missing_segment": num_missing_segment,
        "num_missing_image": num_missing_image,
        "num_missing_mask": num_missing_mask,
        "num_empty_mask": num_empty_mask,
        "region_counts": region_counts,
        "component_counts": component_counts,
    }

    output_json = os.path.join(output_dir, "region_masked_crops.json")
    save_json(
        {
            "summary": summary,
            "crops": output_records,
        },
        output_json,
    )

    print("[INFO] Apply SAM-HQ mask finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Output JSON: {output_json}")
    print(f"[INFO] Masked crop dir: {masked_crop_dir}")


if __name__ == "__main__":
    main()
