import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


POLICY = {
    "skirt_length": ["yolo_crop", "upper_crop", "original"],
    "pant_length": ["yolo_crop", "original"],
    "sleeve_length": ["upper_crop", "yolo_crop", "original"],
    "coat_length": ["yolo_crop", "upper_crop", "original"],

    # 如果后面也想一起修 neckline/collar，可以用这个策略
    "neckline_design": ["upper_crop", "expanded_collar", "yolo_crop", "original"],
    "neck_design": ["upper_crop", "expanded_collar", "yolo_crop", "original"],
    "collar_design": ["expanded_collar", "upper_crop", "yolo_crop", "original"],
    "lapel_design": ["upper_crop", "expanded_collar", "yolo_crop", "original"],
}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def choose_one(rows, preferred_views):
    """
    从同一个 multi_view_source_id 的多条 view 里，只选一个。
    """
    for view in preferred_views:
        candidates = [r for r in rows if r.get("view_type") == view]
        if candidates:
            return candidates[0]

    return rows[0] if rows else None


def process_file(path: Path, task: str, dry_run: bool):
    rows = read_jsonl(path)

    groups = defaultdict(list)
    for r in rows:
        sid = r.get("multi_view_source_id") or r.get("sample_id")
        groups[sid].append(r)

    preferred_views = POLICY[task]

    filtered = []
    before_views = Counter()
    after_views = Counter()

    for r in rows:
        before_views[r.get("view_type", "UNKNOWN")] += 1

    for sid, rs in groups.items():
        chosen = choose_one(rs, preferred_views)
        if chosen is None:
            continue
        filtered.append(chosen)
        after_views[chosen.get("view_type", "UNKNOWN")] += 1

    print(f"\n[FILE] {path}")
    print(f"  rows before   : {len(rows)}")
    print(f"  samples before: {len(groups)}")
    print(f"  rows after    : {len(filtered)}")
    print(f"  before views  : {dict(before_views)}")
    print(f"  after views   : {dict(after_views)}")

    if not dry_run:
        write_jsonl(path, filtered)
        print("  [WRITE] overwritten")
    else:
        print("  [DRY-RUN] not written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", default="outputs/fashionai_multiview_v2_pipeline")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    index_root = Path(args.index_root)

    for task in args.tasks:
        if task not in POLICY:
            raise ValueError(f"No policy defined for task: {task}")

        task_dir = index_root / task
        if not task_dir.exists():
            print(f"[SKIP] missing task dir: {task_dir}")
            continue

        for split in ["train", "val", "test"]:
            path = task_dir / f"{task}_{split}.jsonl"
            if not path.exists():
                print(f"[SKIP] missing jsonl: {path}")
                continue

            process_file(path, task, args.dry_run)


if __name__ == "__main__":
    main()
