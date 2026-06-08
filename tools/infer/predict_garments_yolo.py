from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


CLASS_COLORS = {
    "short sleeve top": (0, 255, 0),
    "long sleeve top": (0, 200, 0),
    "short sleeve outwear": (0, 165, 255),
    "long sleeve outwear": (0, 120, 255),
    "vest": (255, 255, 0),
    "sling": (255, 200, 0),
    "shorts": (255, 128, 0),
    "trousers": (255, 80, 0),
    "skirt": (255, 0, 255),
    "short sleeve dress": (255, 0, 0),
    "long sleeve dress": (200, 0, 0),
    "vest dress": (180, 0, 255),
    "sling dress": (120, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO garment detection on an image or image directory and export JSON results."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
        help="Path to YOLO weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input image path or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/infer/yolo_garments",
        help="Directory to save visualizations and detections.json.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device, e.g. 0, cpu.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualization images with bounding boxes.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped garment regions.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )

    return parser.parse_args()


def collect_images(source: Path, max_images: int | None = None) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file extension: {source}")
        return [source]

    if source.is_dir():
        images = [
            p
            for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_images is not None:
            images = images[:max_images]
        return images

    raise FileNotFoundError(f"Source does not exist: {source}")


def safe_float(value: Any) -> float:
    return float(value)


def safe_int(value: Any) -> int:
    return int(value)


def sanitize_filename_part(text: str) -> str:
    return (
        text.strip()
        .replace(" ", "_")
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


def draw_detections(
    image,
    detections: list[dict[str, Any]],
) -> Any:
    vis = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        class_name = det["class_name"]
        conf = det["confidence"]

        color = CLASS_COLORS.get(class_name, (0, 255, 255))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        y_text_top = max(0, y1 - th - baseline - 4)
        y_text_bottom = y_text_top + th + baseline + 4

        cv2.rectangle(
            vis,
            (x1, y_text_top),
            (x1 + tw + 6, y_text_bottom),
            color,
            -1,
        )
        cv2.putText(
            vis,
            label,
            (x1 + 3, y_text_bottom - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return vis


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    ]


def clip_bbox_xyxy(
    bbox_xyxy: list[float],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return [x1, y1, x2, y2]


def save_crops(
    image,
    image_stem: str,
    detections: list[dict[str, Any]],
    crops_dir: Path,
) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        safe_class_name = sanitize_filename_part(det["class_name"])
        crop_name = (
            f"{image_stem}_det{det['det_id']:03d}_"
            f"{safe_class_name}_{det['confidence']:.2f}.jpg"
        )
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop)


def get_model_class_names(model: YOLO) -> dict[int, str]:
    return {int(k): str(v) for k, v in model.names.items()}


def run_inference(args: argparse.Namespace) -> None:
    weights = Path(args.weights)
    source = Path(args.source)
    output_dir = Path(args.output_dir)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "visualizations"
    crops_dir = output_dir / "crops"

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source, max_images=args.max_images)

    if len(images) == 0:
        raise RuntimeError(f"No images found in source: {source}")

    print(f"[INFO] weights: {weights}")
    print(f"[INFO] source: {source}")
    print(f"[INFO] images: {len(images)}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] imgsz: {args.imgsz}")
    print(f"[INFO] conf: {args.conf}")
    print(f"[INFO] iou: {args.iou}")
    print(f"[INFO] device: {args.device}")

    model = YOLO(str(weights))
    model_class_names = get_model_class_names(model)

    print(f"[INFO] model classes: {model_class_names}")

    all_results: dict[str, Any] = {
        "task": "predict_garments_yolo",
        "weights": str(weights),
        "source": str(source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "class_names": {str(k): v for k, v in model_class_names.items()},
        "images": [],
    }

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx:04d}/{len(images):04d}] {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        height, width = image.shape[:2]

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        if len(results) != 1:
            print(f"[WARN] Unexpected result count for {image_path}: {len(results)}")

        result = results[0]
        detections: list[dict[str, Any]] = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for det_id, (bbox, conf, class_id) in enumerate(
                zip(boxes_xyxy, confs, class_ids)
            ):
                bbox_xyxy = [
                    safe_float(bbox[0]),
                    safe_float(bbox[1]),
                    safe_float(bbox[2]),
                    safe_float(bbox[3]),
                ]
                bbox_xyxy = clip_bbox_xyxy(bbox_xyxy, width=width, height=height)
                bbox_xywh = xyxy_to_xywh(bbox_xyxy)

                class_id = safe_int(class_id)
                class_name = model_class_names.get(class_id, str(class_id))

                detection = {
                    "det_id": det_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": safe_float(conf),
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_xywh": bbox_xywh,
                    "bbox_format": "xyxy_abs_pixels",
                    "image_width": width,
                    "image_height": height,
                }
                detections.append(detection)

        image_record = {
            "image_id": idx - 1,
            "image_path": str(image_path),
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "num_detections": len(detections),
            "detections": detections,
        }

        all_results["images"].append(image_record)

        if args.save_vis:
            vis = draw_detections(image, detections)
            vis_path = vis_dir / f"{image_path.stem}_yolo_det.jpg"
            cv2.imwrite(str(vis_path), vis)

        if args.save_crops:
            save_crops(
                image=image,
                image_stem=image_path.stem,
                detections=detections,
                crops_dir=crops_dir,
            )

    json_path = output_dir / "detections.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    weights_copy_path = output_dir / "used_weights.txt"
    weights_copy_path.write_text(str(weights.resolve()), encoding="utf-8")

    print("[INFO] Inference finished.")
    print(f"[INFO] JSON saved to: {json_path}")

    if args.save_vis:
        print(f"[INFO] Visualizations saved to: {vis_dir}")

    if args.save_crops:
        print(f"[INFO] Crops saved to: {crops_dir}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
