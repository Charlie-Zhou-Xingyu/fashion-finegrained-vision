# scripts/build_fashionai_multiview_index_v1.py
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageOps


DEFAULT_SPLITS = ["train", "val", "test"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_image_path(row: Dict[str, Any]) -> Optional[str]:
    for key in ["image_path", "path", "img_path", "file_path", "filepath"]:
        value = row.get(key)
        if value and Path(str(value)).exists():
            return str(value)
    return None


def open_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:
        return None


def crop_upper(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    y2 = max(1, int(round(h * ratio)))
    return img.crop((0, 0, w, y2))


def crop_center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)
    return img.crop((x1, y1, x1 + side, y1 + side))


def crop_center_upper_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    x1 = max(0, (w - side) // 2)
    # upper-biased square: y starts at 0 when possible
    y1 = 0 if h >= side else max(0, (h - side) // 2)
    return img.crop((x1, y1, x1 + side, y1 + side))


def make_crop(img: Image.Image, view_type: str) -> Image.Image:
    if view_type == "upper60":
        return crop_upper(img, 0.60)
    if view_type == "upper75":
        return crop_upper(img, 0.75)
    if view_type == "center_square":
        return crop_center_square(img)
    if view_type == "center_upper_square":
        return crop_center_upper_square(img)
    raise ValueError(f"Unknown view_type: {view_type}")


def sanitize_stem(text: str) -> str:
    return (
        text.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def build_views_for_row(
    row: Dict[str, Any],
    split: str,
    task: str,
    out_root: Path,
    view_types: List[str],
    copy_original: bool,
    idx: int,
) -> List[Dict[str, Any]]:
    image_path = find_image_path(row)
    if not image_path:
        return []

    img = open_image(image_path)
    if img is None:
        return []

    source_path = Path(image_path)
    sample_id = row.get("sample_id") or row.get("id") or f"{split}_{idx:06d}"
    safe_id = sanitize_stem(str(sample_id))

    out_rows = []

    # original view
    original_row = dict(row)
    original_row["source_image_path"] = image_path
    original_row["view_type"] = "original"
    original_row["multi_view_source_id"] = sample_id

    if copy_original:
        out_path = out_root / "crops" / "original" / split / f"{safe_id}_original{source_path.suffix or '.jpg'}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, out_path)
        original_row["image_path"] = str(out_path)
    else:
        original_row["image_path"] = image_path

    out_rows.append(original_row)

    for view_type in view_types:
        crop = make_crop(img, view_type)
        out_path = out_root / "crops" / view_type / split / f"{safe_id}_{view_type}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=95)

        new_row = dict(row)
        new_row["image_path"] = str(out_path)
        new_row["source_image_path"] = image_path
        new_row["view_type"] = view_type
        new_row["multi_view_source_id"] = sample_id
        out_rows.append(new_row)

    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--input-index-dir", default="data/fashionai_attribute_index", type=str)
    parser.add_argument("--output-index-dir", required=True, type=str)
    parser.add_argument(
        "--view-types",
        nargs="+",
        default=["upper60", "upper75", "center_upper_square"],
    )
    parser.add_argument("--copy-label-map", action="store_true")
    parser.add_argument("--copy-original", action="store_true")
    parser.add_argument("--limit-per-split", default=0, type=int, help="0 means all.")
    args = parser.parse_args()

    input_dir = Path(args.input_index_dir)
    output_dir = Path(args.output_index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "task": args.task,
        "input_index_dir": str(input_dir),
        "output_index_dir": str(output_dir),
        "view_types": ["original"] + args.view_types,
        "splits": {},
    }

    for split in DEFAULT_SPLITS:
        input_jsonl = input_dir / f"{args.task}_{split}.jsonl"
        if not input_jsonl.exists():
            raise FileNotFoundError(input_jsonl)

        rows = read_jsonl(input_jsonl)
        if args.limit_per_split and args.limit_per_split > 0:
            rows = rows[: args.limit_per_split]

        out_rows = []
        skipped = 0

        for idx, row in enumerate(rows):
            new_rows = build_views_for_row(
                row=row,
                split=split,
                task=args.task,
                out_root=output_dir,
                view_types=args.view_types,
                copy_original=args.copy_original,
                idx=idx,
            )
            if not new_rows:
                skipped += 1
                continue
            out_rows.extend(new_rows)

        output_jsonl = output_dir / f"{args.task}_{split}.jsonl"
        write_jsonl(out_rows, output_jsonl)

        summary["splits"][split] = {
            "input_rows": len(rows),
            "output_rows": len(out_rows),
            "skipped": skipped,
            "output_jsonl": str(output_jsonl),
        }

        print(f"[OK] {split}: input={len(rows)}, output={len(out_rows)}, skipped={skipped}")

    label_map_src = input_dir / f"label_map_{args.task}.json"
    if label_map_src.exists():
        label_map_dst = output_dir / f"label_map_{args.task}.json"
        shutil.copy2(label_map_src, label_map_dst)
        summary["label_map"] = str(label_map_dst)

    stats_src = input_dir / f"stats_{args.task}.json"
    if stats_src.exists():
        old_stats = load_json(stats_src)
        summary["original_stats"] = old_stats

    save_json(summary, output_dir / f"multiview_v1_summary_{args.task}.json")

    print("[OK] FashionAI multiview V1 index generated.")
    print(f"[OK] output_dir: {output_dir}")
    print(f"[OK] summary: {output_dir / f'multiview_v1_summary_{args.task}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
