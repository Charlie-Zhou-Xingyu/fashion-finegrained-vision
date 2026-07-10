from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a class-balanced train image list for an existing YOLO dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="YOLO dataset root, e.g. data/processed/deepfashion2_yolo_13cls",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Original YOLO dataset yaml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name. Default: train.",
    )
    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help="Class inverse-frequency power. 0.5 means inverse sqrt frequency.",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=5,
        help="Maximum repeat count per image.",
    )
    parser.add_argument(
        "--output-list-name",
        type=str,
        default="train_balanced.txt",
        help="Output balanced image list filename.",
    )
    parser.add_argument(
        "--output-yaml-name",
        type=str,
        default="deepfashion2_balanced.yaml",
        help="Output balanced yaml filename.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_yolo_label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []

    classes: list[int] = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue

        classes.append(cls)

    return classes


def find_image_files(image_dir: Path) -> list[Path]:
    image_files: list[Path] = []
    for ext in IMAGE_EXTS:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(image_files))


def parse_names_from_yaml_text(yaml_text: str) -> dict[int, str]:
    """
    Lightweight parser for current YOLO yaml style:

    names:
      0: short sleeve top
      1: long sleeve top
    """
    names: dict[int, str] = {}
    in_names = False

    for raw_line in yaml_text.splitlines():
        line = raw_line.rstrip()

        if line.strip() == "names:":
            in_names = True
            continue

        if in_names:
            if not line.startswith("  "):
                break

            stripped = line.strip()
            if ":" not in stripped:
                continue

            key, value = stripped.split(":", 1)
            try:
                idx = int(key.strip())
            except ValueError:
                continue

            names[idx] = value.strip()

    return names


def replace_train_line(yaml_text: str, new_train_value: str) -> str:
    lines = yaml_text.splitlines()
    out_lines = []

    replaced = False
    for line in lines:
        if line.strip().startswith("train:"):
            out_lines.append(f"train: {new_train_value}")
            replaced = True
        else:
            out_lines.append(line)

    if not replaced:
        out_lines.insert(1, f"train: {new_train_value}")

    return "\n".join(out_lines) + "\n"


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    split = args.split

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    if not args.yaml.exists():
        raise FileNotFoundError(f"YAML not found: {args.yaml}")

    yaml_text = read_text(args.yaml)
    class_names = parse_names_from_yaml_text(yaml_text)

    image_files = find_image_files(image_dir)

    image_to_classes: dict[str, list[int]] = {}
    image_to_unique_classes: dict[str, list[int]] = {}
    class_instance_counts: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()
    empty_images = 0

    for idx, image_path in enumerate(image_files, start=1):
        if idx % 5000 == 0:
            print(f"[INFO] scanning images: {idx}/{len(image_files)}")

        label_path = label_dir / f"{image_path.stem}.txt"
        classes = load_yolo_label_classes(label_path)
        unique_classes = sorted(set(classes))

        rel_image_path = image_path.relative_to(dataset_root).as_posix()

        image_to_classes[rel_image_path] = classes
        image_to_unique_classes[rel_image_path] = unique_classes

        if not classes:
            empty_images += 1
            continue

        class_instance_counts.update(classes)
        class_image_counts.update(unique_classes)

    if not class_image_counts:
        raise RuntimeError("No labeled classes found. Cannot build balanced list.")

    # Use image-level class counts for sampling.
    class_weights: dict[int, float] = {}
    for cls, count in class_image_counts.items():
        class_weights[cls] = 1.0 / (float(count) ** args.balance_power)

    min_weight = min(class_weights.values())

    balanced_lines: list[str] = []
    image_repeat_counts: dict[str, int] = {}

    for rel_image_path, unique_classes in image_to_unique_classes.items():
        if not unique_classes:
            repeat = 1
        else:
            image_weight = max(class_weights[c] for c in unique_classes)
            repeat = int(round(image_weight / min_weight))
            repeat = max(1, min(args.max_repeat, repeat))

        image_repeat_counts[rel_image_path] = repeat

        abs_image_path = (dataset_root / rel_image_path).resolve().as_posix()
        balanced_lines.extend([abs_image_path] * repeat)

    output_list_path = dataset_root / args.output_list_name
    write_text(output_list_path, "\n".join(balanced_lines) + "\n")

    balanced_yaml_text = replace_train_line(
        yaml_text=yaml_text,
        new_train_value=output_list_path.resolve().as_posix(),
    )

    output_yaml_path = dataset_root / args.output_yaml_name
    write_text(output_yaml_path, balanced_yaml_text)

    repeat_counter = Counter(image_repeat_counts.values())

    report: dict[str, Any] = {
        "task": "make_balanced_yolo_train_list",
        "dataset_root": str(dataset_root),
        "yaml": str(args.yaml),
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "num_images": len(image_files),
        "empty_images": empty_images,
        "num_balanced_lines": len(balanced_lines),
        "balance_power": args.balance_power,
        "max_repeat": args.max_repeat,
        "class_names": {str(k): v for k, v in sorted(class_names.items())},
        "class_instance_counts": {
            str(k): class_instance_counts[k] for k in sorted(class_instance_counts)
        },
        "class_image_counts": {
            str(k): class_image_counts[k] for k in sorted(class_image_counts)
        },
        "class_weights": {
            str(k): class_weights[k] for k in sorted(class_weights)
        },
        "repeat_count_distribution": {
            str(k): repeat_counter[k] for k in sorted(repeat_counter)
        },
        "output_list": str(output_list_path),
        "output_yaml": str(output_yaml_path),
    }

    report_json_path = dataset_root / "balance_report.json"
    report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# YOLO Balanced Train List Report",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Split: `{split}`",
        f"- Images: `{len(image_files)}`",
        f"- Empty images: `{empty_images}`",
        f"- Balanced train lines: `{len(balanced_lines)}`",
        f"- Balance power: `{args.balance_power}`",
        f"- Max repeat: `{args.max_repeat}`",
        "",
        "## Class Image Counts",
        "",
        "| Class ID | Class Name | Image Count | Instance Count | Weight |",
        "|---:|---|---:|---:|---:|",
    ]

    for cls in sorted(class_image_counts):
        md_lines.append(
            "| "
            f"{cls} | "
            f"{class_names.get(cls, str(cls))} | "
            f"{class_image_counts[cls]} | "
            f"{class_instance_counts[cls]} | "
            f"{class_weights[cls]:.8f} |"
        )

    md_lines.extend(
        [
            "",
            "## Repeat Count Distribution",
            "",
            "| Repeat | Num Images |",
            "|---:|---:|",
        ]
    )

    for repeat in sorted(repeat_counter):
        md_lines.append(f"| {repeat} | {repeat_counter[repeat]} |")

    md_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Balanced train list: `{output_list_path}`",
            f"- Balanced yaml: `{output_yaml_path}`",
            f"- JSON report: `{report_json_path}`",
            "",
        ]
    )

    report_md_path = dataset_root / "balance_report.md"
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("[INFO] Balanced train list generated.")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Num images: {len(image_files)}")
    print(f"[INFO] Balanced lines: {len(balanced_lines)}")
    print(f"[INFO] Output list: {output_list_path}")
    print(f"[INFO] Output yaml: {output_yaml_path}")
    print(f"[INFO] Report json: {report_json_path}")
    print(f"[INFO] Report md: {report_md_path}")


if __name__ == "__main__":
    main()
