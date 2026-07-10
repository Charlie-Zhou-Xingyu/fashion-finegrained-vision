import argparse
import json
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s):
    return str(s).replace("\\", "/")


def stem_of(path_str):
    return Path(norm(path_str)).stem.lower()


def build_crop_index(crop_dir: Path):
    crop_files = []
    for p in crop_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            crop_files.append(p)

    index = {}

    for p in crop_files:
        name = p.name.lower()
        stem = p.stem.lower()

        # 常见 crop 命名：
        # originalstem_det000_class.jpg
        # originalstem_det001_xxx_pant_leg.jpg
        # 所以取 _det 前面的部分作为原图 stem
        if "_det" in stem:
            key = stem.split("_det")[0]
        else:
            # fallback：直接用整个 stem
            key = stem

        # 同一张图多个 crop 时，默认选文件大小最大的，通常更接近整衣/主体 crop
        size = p.stat().st_size
        old = index.get(key)
        if old is None or size > old[1]:
            index[key] = (p, size)

    return {k: str(v[0]) for k, v in index.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--input-index-dir", required=True)
    ap.add_argument("--crop-dir", required=True)
    ap.add_argument("--output-index-dir", required=True)
    ap.add_argument("--view-name", default="crop")
    ap.add_argument("--views", nargs="+", default=["original", "crop"])
    ap.add_argument("--copy-label-map", action="store_true")
    args = ap.parse_args()

    task = args.task
    input_index_dir = Path(args.input_index_dir)
    output_index_dir = Path(args.output_index_dir)
    crop_dir = Path(args.crop_dir)

    if not crop_dir.exists():
        raise FileNotFoundError(f"crop-dir not found: {crop_dir}")

    crop_index = build_crop_index(crop_dir)
    print(f"[INFO] crop files indexed: {len(crop_index)} from {crop_dir}")

    summary = {}

    for split in ["train", "val", "test"]:
        src_jsonl = input_index_dir / f"{task}_{split}.jsonl"
        rows = read_jsonl(src_jsonl)

        out = []
        with_crop = 0
        without_crop = 0

        for row in rows:
            img_path = row.get("image_path") or row.get("source_image_path") or row.get("image_relative_path")
            key = stem_of(img_path)
            crop_path = crop_index.get(key)

            if "original" in args.views:
                r0 = dict(row)
                r0["source_image_path"] = row.get("source_image_path") or row.get("image_path")
                r0["view_type"] = "original"
                r0["multi_view_source_id"] = row.get("sample_id")
                r0["crop_dir_aligned"] = True
                out.append(r0)

            if "crop" in args.views:
                if crop_path:
                    r1 = dict(row)
                    r1["source_image_path"] = row.get("source_image_path") or row.get("image_path")
                    r1["image_path"] = crop_path
                    r1["view_type"] = args.view_name
                    r1["multi_view_source_id"] = row.get("sample_id")
                    r1["crop_dir_aligned"] = True
                    out.append(r1)
                    with_crop += 1
                else:
                    without_crop += 1

        out_jsonl = output_index_dir / f"{task}_{split}.jsonl"
        write_jsonl(out_jsonl, out)

        summary[split] = {
            "input": len(rows),
            "output": len(out),
            "with_crop": with_crop,
            "without_crop": without_crop,
            "crop_dir": str(crop_dir),
            "view_name": args.view_name,
        }

        print(
            f"[OK] {task} {split}: input={len(rows)}, output={len(out)}, "
            f"with_crop={with_crop}, without_crop={without_crop}"
        )

    if args.copy_label_map:
        src_map = input_index_dir / f"label_map_{task}.json"
        dst_map = output_index_dir / f"label_map_{task}.json"
        if src_map.exists():
            shutil.copy2(src_map, dst_map)

    summary_path = output_index_dir / f"crop_dir_summary_{task}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
