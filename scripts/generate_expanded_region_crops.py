# scripts/generate_expanded_region_crops.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageOps


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_source_image_path(record: Dict[str, Any]) -> Optional[Path]:
    candidates = [
        "image_path",
        "source_image_path",
        "original_image_path",
        "full_image_path",
        "img_path",
    ]
    for key in candidates:
        value = record.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))
    return None


def open_rgb(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:
        return None


def clip_box(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(1, min(w, int(round(x2))))
    y2 = max(1, min(h, int(round(y2))))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(
    bbox,
    image_w: int,
    image_h: int,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    nx1 = x1 - expand_x * bw
    nx2 = x2 + expand_x * bw
    ny1 = y1 - expand_top * bh
    ny2 = y2 + expand_bottom * bh

    return clip_box((nx1, ny1, nx2, ny2), image_w, image_h)


def upper_bbox_from_region(
    bbox,
    image_w: int,
    image_h: int,
    expand_x: float,
    top_margin: float,
    bottom_scale: float,
):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    nx1 = x1 - expand_x * bw
    nx2 = x2 + expand_x * bw
    ny1 = y1 - top_margin * bh
    ny2 = y2 + bottom_scale * bh

    return clip_box((nx1, ny1, nx2, ny2), image_w, image_h)


def crop_and_save(img: Image.Image, box, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop = img.crop(box)
    crop.save(out_path)
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--region", default="collar", type=str)

    parser.add_argument("--expand-x", default=0.50, type=float)
    parser.add_argument("--expand-top", default=0.25, type=float)
    parser.add_argument("--expand-bottom", default=1.20, type=float)

    parser.add_argument("--upper-expand-x", default=0.80, type=float)
    parser.add_argument("--upper-top-margin", default=0.50, type=float)
    parser.add_argument("--upper-bottom-scale", default=2.20, type=float)

    parser.add_argument("--fallback-to-image-crop", action="store_true")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_dir = Path(args.output_dir)

    data = load_json(input_json)
    crops = data.get("crops", [])
    if not isinstance(crops, list):
        raise ValueError("Input JSON must contain list field: crops")

    n_region = 0
    n_expanded = 0
    n_upper = 0
    n_no_source = 0
    n_no_bbox = 0

    for idx, record in enumerate(crops):
        if not isinstance(record, dict):
            continue

        if str(record.get("region", "")) != args.region:
            continue

        n_region += 1

        bbox = record.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            n_no_bbox += 1
            continue

        source_path = find_source_image_path(record)

        if source_path is None and args.fallback_to_image_crop:
            fallback = record.get("image_crop_path") or record.get("crop_path")
            if fallback and Path(str(fallback)).exists():
                source_path = Path(str(fallback))
                # fallback crop 本身已经是 crop，所以 bbox 改成整图
                tmp_img = open_rgb(source_path)
                if tmp_img is None:
                    continue
                bbox = [0, 0, tmp_img.width, tmp_img.height]

        if source_path is None:
            n_no_source += 1
            continue

        img = open_rgb(source_path)
        if img is None:
            n_no_source += 1
            continue

        image_w, image_h = img.size

        exp_box = expand_bbox(
            bbox,
            image_w=image_w,
            image_h=image_h,
            expand_x=args.expand_x,
            expand_top=args.expand_top,
            expand_bottom=args.expand_bottom,
        )

        upper_box = upper_bbox_from_region(
            bbox,
            image_w=image_w,
            image_h=image_h,
            expand_x=args.upper_expand_x,
            top_margin=args.upper_top_margin,
            bottom_scale=args.upper_bottom_scale,
        )

        stem = f"{idx:06d}_{record.get('det_id', 0)}_{record.get('class_name', 'cls')}_{args.region}"
        stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")

        exp_path = output_dir / "expanded_crops" / args.region / f"{stem}_expanded.png"
        upper_path = output_dir / "upper_crops" / args.region / f"{stem}_upper.png"

        record["expanded_crop_path"] = crop_and_save(img, exp_box, exp_path)
        record["upper_crop_path"] = crop_and_save(img, upper_box, upper_path)
        record["expanded_bbox_xyxy"] = list(exp_box)
        record["upper_bbox_xyxy"] = list(upper_box)

        n_expanded += 1
        n_upper += 1

    save_json(data, output_json)

    print("[OK] Expanded region crops generated.")
    print(f"[OK] input_json: {input_json}")
    print(f"[OK] output_json: {output_json}")
    print(f"[OK] output_dir: {output_dir}")
    print(f"[OK] target region records: {n_region}")
    print(f"[OK] expanded crops: {n_expanded}")
    print(f"[OK] upper crops: {n_upper}")
    print(f"[WARN] no source image: {n_no_source}")
    print(f"[WARN] no bbox: {n_no_bbox}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
