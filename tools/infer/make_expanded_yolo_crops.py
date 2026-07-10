#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make expanded YOLO crops from an existing detections.json.

This script is designed for attributes that need context, such as pant_length.
It reads YOLO bbox coordinates from detections.json, expands the bbox, and
re-crops from the original image.

Example:
    python tools/infer/make_expanded_yolo_crops.py ^
      --detections-json outputs/pipeline_13cls_eval_balanced/01_yolo/detections.json ^
      --output-dir outputs/pipeline_13cls_eval_balanced/01_yolo/crops_expanded_pant_medium ^
      --classes shorts trousers ^
      --min-conf 0.0 ^
      --pad-left 0.15 ^
      --pad-right 0.15 ^
      --pad-top 0.15 ^
      --pad-bottom 0.30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_class_name(name: str) -> str:
    """
    Keep filename format consistent with existing crop parser:
    'short sleeve top' -> 'short_sleeve_top'
    """
    return str(name).strip().replace(" ", "_")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def expand_bbox_xyxy(
    bbox: List[float],
    image_width: int,
    image_height: int,
    pad_left: float,
    pad_right: float,
    pad_top: float,
    pad_bottom: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox)

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    nx1 = x1 - pad_left * w
    nx2 = x2 + pad_right * w
    ny1 = y1 - pad_top * h
    ny2 = y2 + pad_bottom * h

    nx1 = clamp(nx1, 0, image_width - 1)
    ny1 = clamp(ny1, 0, image_height - 1)
    nx2 = clamp(nx2, 1, image_width)
    ny2 = clamp(ny2, 1, image_height)

    # PIL crop box is (left, upper, right, lower), right/lower exclusive.
    ix1 = int(round(nx1))
    iy1 = int(round(ny1))
    ix2 = int(round(nx2))
    iy2 = int(round(ny2))

    if ix2 <= ix1:
        ix2 = min(image_width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(image_height, iy1 + 1)

    return ix1, iy1, ix2, iy2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-json", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["shorts", "trousers"],
        help="Class names to crop. Use names as stored in detections.json.",
    )
    parser.add_argument("--min-conf", default=0.0, type=float)
    parser.add_argument("--pad-left", default=0.15, type=float)
    parser.add_argument("--pad-right", default=0.15, type=float)
    parser.add_argument("--pad-top", default=0.15, type=float)
    parser.add_argument("--pad-bottom", default=0.30, type=float)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args()

    detections_json = Path(args.detections_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(detections_json)
    images = data.get("images", [])

    wanted_classes = set(args.classes)

    num_images = 0
    num_dets_total = 0
    num_selected = 0
    num_saved = 0
    num_missing_images = 0
    num_failed = 0

    manifest_rows: List[Dict[str, Any]] = []

    for image_record in images:
        image_path = Path(image_record["image_path"])
        file_name = image_record.get("file_name", image_path.name)
        image_stem = Path(file_name).stem

        detections = image_record.get("detections", [])
        num_images += 1
        num_dets_total += len(detections)

        if not image_path.exists():
            num_missing_images += 1
            print(f"[WARN] Missing image: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            num_failed += 1
            print(f"[WARN] Failed to open image: {image_path} error={repr(e)}")
            continue

        image_width, image_height = img.size

        for det in detections:
            class_name = str(det.get("class_name", ""))
            conf = float(det.get("confidence", 0.0))
            det_id = int(det.get("det_id", 0))

            if class_name not in wanted_classes:
                continue
            if conf < args.min_conf:
                continue

            bbox = det.get("bbox_xyxy")
            if bbox is None:
                print(f"[WARN] Missing bbox_xyxy for {image_path}, det_id={det_id}")
                continue

            num_selected += 1

            x1, y1, x2, y2 = expand_bbox_xyxy(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                pad_left=args.pad_left,
                pad_right=args.pad_right,
                pad_top=args.pad_top,
                pad_bottom=args.pad_bottom,
            )

            crop = img.crop((x1, y1, x2, y2))

            safe_cls = sanitize_class_name(class_name)
            conf_str = f"{conf:.2f}"
            out_name = f"{image_stem}_det{det_id:03d}_{safe_cls}_{conf_str}.jpg"
            out_path = output_dir / out_name

            if out_path.exists() and not args.overwrite:
                pass
            else:
                crop.save(out_path, quality=95)

            num_saved += 1

            manifest_rows.append({
                "source_image_path": str(image_path),
                "file_name": file_name,
                "image_stem": image_stem,
                "det_id": det_id,
                "class_id": det.get("class_id"),
                "class_name": class_name,
                "confidence": conf,
                "original_bbox_xyxy": bbox,
                "expanded_bbox_xyxy": [x1, y1, x2, y2],
                "image_width": image_width,
                "image_height": image_height,
                "output_path": str(out_path),
                "pad": {
                    "left": args.pad_left,
                    "right": args.pad_right,
                    "top": args.pad_top,
                    "bottom": args.pad_bottom,
                },
            })

    manifest_path = output_dir / "expanded_crops_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "detections_json": str(detections_json),
            "output_dir": str(output_dir),
            "classes": sorted(wanted_classes),
            "min_conf": args.min_conf,
            "pad": {
                "left": args.pad_left,
                "right": args.pad_right,
                "top": args.pad_top,
                "bottom": args.pad_bottom,
            },
            "num_images": num_images,
            "num_dets_total": num_dets_total,
            "num_selected": num_selected,
            "num_saved": num_saved,
            "num_missing_images": num_missing_images,
            "num_failed": num_failed,
            "rows": manifest_rows,
        }, f, ensure_ascii=False, indent=2)

    print("[OK] Expanded crops generated.")
    print(json.dumps({
        "detections_json": str(detections_json),
        "output_dir": str(output_dir),
        "classes": sorted(wanted_classes),
        "min_conf": args.min_conf,
        "pad_left": args.pad_left,
        "pad_right": args.pad_right,
        "pad_top": args.pad_top,
        "pad_bottom": args.pad_bottom,
        "num_images": num_images,
        "num_dets_total": num_dets_total,
        "num_selected": num_selected,
        "num_saved": num_saved,
        "num_missing_images": num_missing_images,
        "num_failed": num_failed,
        "manifest_path": str(manifest_path),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
