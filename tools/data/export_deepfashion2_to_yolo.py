from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TARGET_CLASS_NAMES_5CLS = ["top", "pants", "skirt", "outwear", "dress"]
TARGET_CLASS_NAME_TO_ID_5CLS = {
    name: idx for idx, name in enumerate(TARGET_CLASS_NAMES_5CLS)
}

TARGET_CLASS_NAMES_13CLS = [
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear",
    "long sleeve outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress",
]
TARGET_CLASS_NAME_TO_ID_13CLS = {
    name: idx for idx, name in enumerate(TARGET_CLASS_NAMES_13CLS)
}

# Backward-compatible aliases.
TARGET_CLASS_NAMES = TARGET_CLASS_NAMES_5CLS
TARGET_CLASS_NAME_TO_ID = TARGET_CLASS_NAME_TO_ID_5CLS

TARGET_CLASS_ZH = {
    "top": "上衣",
    "pants": "裤装",
    "skirt": "半身裙",
    "outwear": "外套",
    "dress": "连衣裙",
}


# DeepFashion2 category_id:
# 1 short sleeve top
# 2 long sleeve top
# 3 short sleeve outwear
# 4 long sleeve outwear
# 5 vest
# 6 sling
# 7 shorts
# 8 trousers
# 9 skirt
# 10 short sleeve dress
# 11 long sleeve dress
# 12 vest dress
# 13 sling dress
DEEPFASHION2_CATEGORY_ID_TO_NAME = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}


DEEPFASHION2_CATEGORY_ID_TO_TARGET = {
    1: "top",
    2: "top",
    3: "outwear",
    4: "outwear",
    5: "outwear",   # vest -> outwear
    6: "top",
    7: "pants",
    8: "pants",
    9: "skirt",
    10: "dress",
    11: "dress",
    12: "dress",
    13: "dress",
}


def get_target_class_names(class_mode: str) -> List[str]:
    if class_mode == "5cls":
        return TARGET_CLASS_NAMES_5CLS

    if class_mode == "13cls":
        return TARGET_CLASS_NAMES_13CLS

    raise ValueError(f"Unsupported class_mode: {class_mode}")


def map_category_to_yolo_class(
    category_id: int,
    class_mode: str,
) -> Optional[Tuple[int, str]]:
    if class_mode == "5cls":
        target_name = DEEPFASHION2_CATEGORY_ID_TO_TARGET.get(category_id)
        if target_name is None:
            return None
        return TARGET_CLASS_NAME_TO_ID_5CLS[target_name], target_name

    if class_mode == "13cls":
        raw_name = DEEPFASHION2_CATEGORY_ID_TO_NAME.get(category_id)
        if raw_name is None:
            return None
        return category_id - 1, raw_name

    raise ValueError(f"Unsupported class_mode: {class_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DeepFashion2 annotations to YOLOv8 detection format."
    )
    parser.add_argument(
        "--deepfashion2-root",
        type=Path,
        required=True,
        help="Root directory of DeepFashion2 dataset, e.g. D:/Aliintern/fashion-ai-data/deepfashion2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "deepfashion2_yolo",
        help="Output directory for YOLO dataset.",
    )
    parser.add_argument(
        "--class-mode",
        choices=["5cls", "13cls"],
        default="5cls",
        help=(
            "YOLO class export mode. "
            "5cls merges DeepFashion2 into coarse garment classes. "
            "13cls preserves original DeepFashion2 categories."
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated DeepFashion2 splits to export. Default: train,validation",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "link"],
        default="copy",
        help="How to export images. copy: copy image files. link: create hard links when possible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Limit number of images per split for debugging. 0 means all.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip images without valid target boxes. Default keeps empty-label images.",
    )
    return parser.parse_args()


def normalize_split_name(split: str) -> str:
    split = split.strip()
    if split == "validation":
        return "val"
    return split


def find_image_dir(split_dir: Path) -> Path:
    candidates = [
        split_dir / "image",
        split_dir / "images",
        split_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find image directory under: {split_dir}")


def find_annos_dir(split_dir: Path) -> Path:
    candidates = [
        split_dir / "annos",
        split_dir / "annotations",
        split_dir / "anno",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find annotation directory under: {split_dir}")


def list_annotation_files(annos_dir: Path) -> List[Path]:
    return sorted(annos_dir.glob("*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image_path(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def get_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # width, height


def iter_deepfashion2_items(annotation: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key, value in annotation.items():
        if not key.startswith("item"):
            continue
        if not isinstance(value, dict):
            continue
        yield key, value


def clip_bbox_xyxy(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]

    x1 = max(0.0, min(x1, float(image_width - 1)))
    y1 = max(0.0, min(y1, float(image_height - 1)))
    x2 = max(0.0, min(x2, float(image_width - 1)))
    y2 = max(0.0, min(y2, float(image_height - 1)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def xyxy_to_yolo(
    bbox_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy

    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def format_yolo_line(class_id: int, box: Tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def safe_copy_or_link(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if copy_mode == "link":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    shutil.copy2(src, dst)


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Use --overwrite to remove and recreate it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_dir: Path, class_mode: str) -> None:
    class_names = get_target_class_names(class_mode)
    yaml_path = output_dir / f"deepfashion2_{class_mode}.yaml"

    content = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "",
            f"nc: {len(class_names)}",
            "names:",
        ]
        + [f"  {idx}: {name}" for idx, name in enumerate(class_names)]
        + [""]
    )

    yaml_path.write_text(content, encoding="utf-8")


def write_category_mapping(output_dir: Path, class_mode: str) -> None:
    class_names = get_target_class_names(class_mode)

    if class_mode == "5cls":
        target_classes = [
            {
                "id": idx,
                "name": name,
                "name_zh": TARGET_CLASS_ZH[name],
            }
            for idx, name in enumerate(class_names)
        ]

        deepfashion2_to_target = [
            {
                "deepfashion2_category_id": category_id,
                "deepfashion2_category_name": DEEPFASHION2_CATEGORY_ID_TO_NAME[category_id],
                "target_class_name": target_name,
                "target_class_id": TARGET_CLASS_NAME_TO_ID_5CLS[target_name],
                "note": "vest is mapped to outwear" if category_id == 5 else "",
            }
            for category_id, target_name in DEEPFASHION2_CATEGORY_ID_TO_TARGET.items()
        ]

    elif class_mode == "13cls":
        target_classes = [
            {
                "id": idx,
                "name": name,
                "name_zh": "",
            }
            for idx, name in enumerate(class_names)
        ]

        deepfashion2_to_target = [
            {
                "deepfashion2_category_id": category_id,
                "deepfashion2_category_name": raw_name,
                "target_class_name": raw_name,
                "target_class_id": category_id - 1,
                "note": "preserve original DeepFashion2 category",
            }
            for category_id, raw_name in DEEPFASHION2_CATEGORY_ID_TO_NAME.items()
        ]

    else:
        raise ValueError(f"Unsupported class_mode: {class_mode}")

    mapping = {
        "class_mode": class_mode,
        "target_classes": target_classes,
        "deepfashion2_to_target": deepfashion2_to_target,
    }

    path = output_dir / "category_mapping.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def export_split(
    deepfashion2_root: Path,
    output_dir: Path,
    split: str,
    copy_mode: str,
    limit_per_split: int,
    skip_empty: bool,
    class_mode: str,
) -> Dict[str, Any]:
    split_dir = deepfashion2_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    yolo_split = normalize_split_name(split)

    image_dir = find_image_dir(split_dir)
    annos_dir = find_annos_dir(split_dir)

    out_image_dir = output_dir / "images" / yolo_split
    out_label_dir = output_dir / "labels" / yolo_split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = list_annotation_files(annos_dir)
    if limit_per_split > 0:
        annotation_files = annotation_files[:limit_per_split]

    stats: Dict[str, Any] = {
        "split": split,
        "yolo_split": yolo_split,
        "image_dir": str(image_dir),
        "annos_dir": str(annos_dir),
        "class_mode": class_mode,
        "num_annotation_files": len(annotation_files),
        "num_images_exported": 0,
        "num_images_skipped_missing_image": 0,
        "num_images_skipped_empty": 0,
        "num_instances_exported": 0,
        "num_invalid_bbox": 0,
        "num_unknown_category": 0,
        "target_class_counts": Counter(),
        "deepfashion2_category_counts": Counter(),
        "empty_label_images": 0,
    }

    for idx, anno_path in enumerate(annotation_files, start=1):
        if idx % 1000 == 0:
            print(
                f"[INFO] split={split} progress: "
                f"{idx}/{len(annotation_files)} annotations, "
                f"exported_images={stats['num_images_exported']}, "
                f"instances={stats['num_instances_exported']}"
            )
        stem = anno_path.stem
        image_path = find_image_path(image_dir, stem)

        if image_path is None:
            stats["num_images_skipped_missing_image"] += 1
            continue

        annotation = load_json(anno_path)
        image_width, image_height = get_image_size(image_path)

        yolo_lines: List[str] = []

        for item_key, item in iter_deepfashion2_items(annotation):
            category_id = item.get("category_id")
            bbox = item.get("bounding_box")

            if category_id is None:
                stats["num_unknown_category"] += 1
                continue

            try:
                category_id_int = int(category_id)
            except ValueError:
                stats["num_unknown_category"] += 1
                continue

            mapped = map_category_to_yolo_class(
                category_id=category_id_int,
                class_mode=class_mode,
            )
            if mapped is None:
                stats["num_unknown_category"] += 1
                continue

            target_class_id, target_name = mapped

            if bbox is None:
                stats["num_invalid_bbox"] += 1
                continue

            clipped_bbox = clip_bbox_xyxy(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
            )
            if clipped_bbox is None:
                stats["num_invalid_bbox"] += 1
                continue

            yolo_box = xyxy_to_yolo(
                bbox_xyxy=clipped_bbox,
                image_width=image_width,
                image_height=image_height,
            )

            yolo_lines.append(format_yolo_line(target_class_id, yolo_box))

            stats["num_instances_exported"] += 1
            stats["target_class_counts"][target_name] += 1
            stats["deepfashion2_category_counts"][
                DEEPFASHION2_CATEGORY_ID_TO_NAME.get(category_id_int, str(category_id_int))
            ] += 1

        if skip_empty and len(yolo_lines) == 0:
            stats["num_images_skipped_empty"] += 1
            continue

        if len(yolo_lines) == 0:
            stats["empty_label_images"] += 1

        out_image_path = out_image_dir / image_path.name
        out_label_path = out_label_dir / f"{image_path.stem}.txt"

        safe_copy_or_link(image_path, out_image_path, copy_mode=copy_mode)
        out_label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )

        stats["num_images_exported"] += 1

    stats["target_class_counts"] = dict(stats["target_class_counts"])
    stats["deepfashion2_category_counts"] = dict(stats["deepfashion2_category_counts"])

    return stats


def main() -> None:
    args = parse_args()

    deepfashion2_root: Path = args.deepfashion2_root
    output_dir: Path = args.output_dir
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not deepfashion2_root.exists():
        raise FileNotFoundError(f"DeepFashion2 root does not exist: {deepfashion2_root}")

    reset_output_dir(output_dir, overwrite=args.overwrite)

    split_summaries: List[Dict[str, Any]] = []

    for split in splits:
        print(f"[INFO] Exporting split: {split}")
        split_summary = export_split(
            deepfashion2_root=deepfashion2_root,
            output_dir=output_dir,
            split=split,
            copy_mode=args.copy_mode,
            limit_per_split=args.limit_per_split,
            skip_empty=args.skip_empty,
            class_mode=args.class_mode,
        )
        split_summaries.append(split_summary)
        print(
            f"[INFO] Done split={split}: "
            f"images={split_summary['num_images_exported']}, "
            f"instances={split_summary['num_instances_exported']}"
        )

    write_dataset_yaml(output_dir, class_mode=args.class_mode)
    write_category_mapping(output_dir, class_mode=args.class_mode)

    total_class_counts = Counter()
    for split_summary in split_summaries:
        total_class_counts.update(split_summary["target_class_counts"])

    class_names = get_target_class_names(args.class_mode)
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    summary = {
        "task": "export_deepfashion2_to_yolo",
        "class_mode": args.class_mode,
        "deepfashion2_root": str(deepfashion2_root),
        "output_dir": str(output_dir),
        "target_classes": class_names,
        "target_class_name_to_id": class_name_to_id,
        "copy_mode": args.copy_mode,
        "splits": split_summaries,
        "total_target_class_counts": dict(total_class_counts),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Export finished.")
    print(f"[INFO] Class mode: {args.class_mode}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Dataset yaml: {output_dir / f'deepfashion2_{args.class_mode}.yaml'}")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
