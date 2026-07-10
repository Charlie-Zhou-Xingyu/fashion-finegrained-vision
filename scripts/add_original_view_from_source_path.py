import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_sid(r):
    return r.get("multi_view_source_id") or r.get("sample_id")


def make_original_row(r):
    source = r.get("source_image_path") or r.get("original_image_path")
    if not source:
        return None

    nr = dict(r)
    nr["image_path"] = source
    nr["source_image_path"] = source
    nr["view_type"] = "original"
    nr["multi_view_source_id"] = get_sid(r)

    # 原图不应该继承 crop 的检测信息，删掉容易误解的字段
    for k in [
        "pipeline_class_name",
        "pipeline_det_id",
        "pipeline_region",
        "pipeline_component",
        "pipeline_bbox_xyxy",
        "pipeline_expanded_bbox_xyxy",
        "pipeline_upper_bbox_xyxy",
        "pipeline_confidence",
    ]:
        nr.pop(k, None)

    return nr


def process_file(path: Path, dry_run: bool):
    rows = read_jsonl(path)

    groups = defaultdict(list)
    for r in rows:
        groups[get_sid(r)].append(r)

    out = list(rows)
    added = 0
    skipped_no_source = 0
    skipped_already_has_original = 0

    for sid, rs in groups.items():
        has_original = any(r.get("view_type") == "original" for r in rs)
        if has_original:
            skipped_already_has_original += 1
            continue

        # 优先从 yolo_crop/upper_crop 复制 source_image_path
        base = None
        for v in ["yolo_crop", "upper_crop", "image_crop", "expanded_collar"]:
            cand = [r for r in rs if r.get("view_type") == v]
            if cand:
                base = cand[0]
                break
        if base is None and rs:
            base = rs[0]

        nr = make_original_row(base)
        if nr is None:
            skipped_no_source += 1
            continue

        out.append(nr)
        added += 1

    before = Counter(r.get("view_type", "UNKNOWN") for r in rows)
    after = Counter(r.get("view_type", "UNKNOWN") for r in out)

    print(f"\n[FILE] {path}")
    print(f"  rows before: {len(rows)}")
    print(f"  rows after : {len(out)}")
    print(f"  added original: {added}")
    print(f"  skipped already has original: {skipped_already_has_original}")
    print(f"  skipped no source: {skipped_no_source}")
    print(f"  views before: {dict(before)}")
    print(f"  views after : {dict(after)}")

    if not dry_run:
        write_jsonl(path, out)
        print("  [WRITE] overwritten")
    else:
        print("  [DRY-RUN] not written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", default="outputs/fashionai_multiview_v2_pipeline")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.index_root)

    for task in args.tasks:
        for split in args.splits:
            path = root / task / f"{task}_{split}.jsonl"
            if not path.exists():
                print(f"[SKIP] missing {path}")
                continue
            process_file(path, args.dry_run)


if __name__ == "__main__":
    main()
