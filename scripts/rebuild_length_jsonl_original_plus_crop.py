import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ["train", "val", "test"]

# 每个任务要保留的 crop view
TASK_CROP_POLICY = {
    "skirt_length": ["yolo_crop"],
    "pant_length": ["yolo_crop"],
    "sleeve_length": ["upper_crop", "yolo_crop"],
    "coat_length": ["yolo_crop", "upper_crop"],
}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_sid(row):
    return row.get("multi_view_source_id") or row.get("sample_id")


def make_original_row(base_row):
    r = dict(base_row)
    r["image_path"] = str(base_row.get("image_path"))
    r["source_image_path"] = str(base_row.get("image_path"))
    r["view_type"] = "original"
    r["multi_view_source_id"] = base_row.get("sample_id")
    r["v2_pipeline_aligned"] = True
    return r


def choose_crop_row(rows, preferred_views):
    """
    从当前已清洗 jsonl 里，为一个 sample 选一个 crop view。
    注意：如果当前只有 original，则不返回 crop。
    """
    for view in preferred_views:
        candidates = [r for r in rows if r.get("view_type") == view]
        if candidates:
            return candidates[0]

    return None


def merge_crop_with_base(base_row, crop_row):
    """
    以 base_row 的标签/样本信息为准，
    以 crop_row 的 image_path / view_type / pipeline 信息为准。
    """
    r = dict(base_row)

    # crop 路径
    r["image_path"] = crop_row.get("image_path")
    r["source_image_path"] = crop_row.get("source_image_path") or base_row.get("image_path")
    r["view_type"] = crop_row.get("view_type")
    r["multi_view_source_id"] = base_row.get("sample_id")
    r["v2_pipeline_aligned"] = True

    # 保留 pipeline 相关字段
    for k, v in crop_row.items():
        if k.startswith("pipeline_"):
            r[k] = v

    return r


def process_task_split(
    task: str,
    split: str,
    base_index_dir: Path,
    current_index_root: Path,
    backup: bool,
    dry_run: bool,
):
    base_jsonl = base_index_dir / f"{task}_{split}.jsonl"
    current_jsonl = current_index_root / task / f"{task}_{split}.jsonl"

    if not base_jsonl.exists():
        print(f"[SKIP] missing base jsonl: {base_jsonl}")
        return

    if not current_jsonl.exists():
        print(f"[SKIP] missing current jsonl: {current_jsonl}")
        return

    base_rows = read_jsonl(base_jsonl)
    current_rows = read_jsonl(current_jsonl)

    current_groups = defaultdict(list)
    for r in current_rows:
        sid = get_sid(r)
        if sid:
            current_groups[sid].append(r)

    preferred_views = TASK_CROP_POLICY[task]

    out_rows = []
    stats = Counter()

    for base in base_rows:
        sid = base.get("sample_id")
        if not sid:
            stats["base_missing_sample_id"] += 1
            continue

        # 1. 永远加入 original
        original = make_original_row(base)
        out_rows.append(original)
        stats["added_original"] += 1

        # 2. 加入最佳 crop
        crop = choose_crop_row(current_groups.get(sid, []), preferred_views)
        if crop is not None:
            crop_merged = merge_crop_with_base(base, crop)
            out_rows.append(crop_merged)
            stats[f"added_{crop.get('view_type')}"] += 1
        else:
            stats["no_crop_only_original"] += 1

    view_counter = Counter([r.get("view_type", "UNKNOWN") for r in out_rows])

    print(f"\n[FILE] {current_jsonl}")
    print(f"  base rows       : {len(base_rows)}")
    print(f"  current rows    : {len(current_rows)}")
    print(f"  output rows     : {len(out_rows)}")
    print(f"  stats           : {dict(stats)}")
    print(f"  views after     : {dict(view_counter)}")

    if dry_run:
        print("  [DRY-RUN] not written")
        return

    if backup:
        backup_path = current_jsonl.with_suffix(current_jsonl.suffix + ".bak_original_plus_crop")
        shutil.copy2(current_jsonl, backup_path)
        print(f"  [BACKUP] {backup_path}")

    write_jsonl(current_jsonl, out_rows)
    print("  [WRITE] overwritten")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-index-dir",
        default="data/fashionai_attribute_index",
    )
    parser.add_argument(
        "--current-index-root",
        default="outputs/fashionai_multiview_v2_pipeline",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["skirt_length", "pant_length", "sleeve_length", "coat_length"],
    )
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_index_dir = Path(args.base_index_dir)
    current_index_root = Path(args.current_index_root)

    for task in args.tasks:
        if task not in TASK_CROP_POLICY:
            raise ValueError(f"No crop policy for task: {task}")

        for split in SPLITS:
            process_task_split(
                task=task,
                split=split,
                base_index_dir=base_index_dir,
                current_index_root=current_index_root,
                backup=args.backup,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
