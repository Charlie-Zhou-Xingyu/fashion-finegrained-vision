import argparse
import json
from pathlib import Path
from collections import Counter
from PIL import Image


PANT_CLASS_NAMES = {
    "shorts",
    "trousers",
}

PANT_CLASS_IDS = {
    6,  # shorts
    7,  # trousers
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


def load_detections(detection_json: Path):
    data = json.loads(detection_json.read_text(encoding="utf-8"))
    images = data.get("images", [])

    det_by_filename = {}
    for item in images:
        fn = Path(str(item.get("file_name") or item.get("image_path"))).name.lower()
        det_by_filename[fn] = item

    return det_by_filename


def choose_best_pant_detection(dets):
    candidates = []

    for d in dets:
        class_name = str(d.get("class_name", "")).lower().strip()
        class_id = d.get("class_id", None)

        is_pant = False
        if class_name in PANT_CLASS_NAMES:
            is_pant = True
        try:
            if int(class_id) in PANT_CLASS_IDS:
                is_pant = True
        except Exception:
            pass

        if not is_pant:
            continue

        conf = float(d.get("confidence", d.get("conf", d.get("score", 0.0))))
        bbox = d.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(x) for x in bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        # 主要按置信度，其次按面积
        candidates.append((conf, area, d))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def expand_and_clip_bbox(bbox, w, h, pad_ratio=0.06):
    x1, y1, x2, y2 = [float(x) for x in bbox]

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w, int(round(x2 + pad_x)))
    y2 = min(h, int(round(y2 + pad_y)))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def make_crop_path(crop_root: Path, split: str, source_image_path: str, det):
    stem = Path(source_image_path).stem
    class_name = str(det.get("class_name", "pant")).replace(" ", "_")
    conf = float(det.get("confidence", det.get("conf", det.get("score", 0.0))))
    det_id = det.get("det_id", 0)

    filename = f"{stem}_det{int(det_id):03d}_{class_name}_{conf:.2f}.jpg"
    return crop_root / split / filename


def process_split(task_dir: Path, split: str, detections, crop_root: Path, dry_run: bool):
    jsonl_path = task_dir / f"pant_length_{split}.jsonl"
    if not jsonl_path.exists():
        print(f"[SKIP] missing: {jsonl_path}")
        return

    rows = read_jsonl(jsonl_path)

    out_rows = []
    stats = Counter()

    for r in rows:
        source_image_path = r.get("source_image_path") or r.get("image_path")
        filename = Path(str(source_image_path)).name.lower()

        det_item = detections.get(filename)
        if det_item is None:
            # detection 里没有这张图，保留原图
            nr = dict(r)
            nr["view_type"] = "original"
            out_rows.append(nr)
            stats["no_detection_record_keep_original"] += 1
            continue

        best = choose_best_pant_detection(det_item.get("detections", []))
        if best is None:
            nr = dict(r)
            nr["view_type"] = "original"
            out_rows.append(nr)
            stats["no_pant_box_keep_original"] += 1
            continue

        image_path = Path(str(source_image_path))
        if not image_path.exists():
            # 如果 jsonl 里是相对路径，但 detection 里有绝对路径，尝试用 detection 的 image_path
            alt = Path(str(det_item.get("image_path", "")))
            if alt.exists():
                image_path = alt
            else:
                nr = dict(r)
                nr["view_type"] = "original"
                out_rows.append(nr)
                stats["image_missing_keep_original"] += 1
                continue

        crop_path = make_crop_path(crop_root, split, str(image_path), best)

        if not dry_run:
            crop_path.parent.mkdir(parents=True, exist_ok=True)

            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = expand_and_clip_bbox(best["bbox_xyxy"], w, h)
            crop = img.crop((x1, y1, x2, y2))
            crop.save(crop_path, quality=95)

        nr = dict(r)
        nr["source_image_path"] = str(image_path)
        nr["image_path"] = str(crop_path)
        nr["view_type"] = "yolo_crop"
        nr["multi_view_source_id"] = r.get("multi_view_source_id") or r.get("sample_id")
        nr["pipeline_class_name"] = best.get("class_name")
        nr["pipeline_det_id"] = best.get("det_id")
        nr["pipeline_bbox_xyxy"] = best.get("bbox_xyxy")
        nr["pipeline_confidence"] = best.get("confidence", best.get("conf", best.get("score")))
        nr["v2_pipeline_aligned"] = True

        out_rows.append(nr)
        stats["yolo_crop"] += 1

    print(f"\n[FILE] {jsonl_path}")
    print(f"  rows: {len(rows)} -> {len(out_rows)}")
    print(f"  stats: {dict(stats)}")

    view_counter = Counter([r.get("view_type", "UNKNOWN") for r in out_rows])
    print(f"  views after: {dict(view_counter)}")

    if not dry_run:
        write_jsonl(jsonl_path, out_rows)
        print("  [WRITE] overwritten")
    else:
        print("  [DRY-RUN] not written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        default="outputs/fashionai_multiview_v2_pipeline/pant_length",
        help="pant_length index dir",
    )
    parser.add_argument(
        "--detections-json",
        required=True,
        help="YOLO detections.json",
    )
    parser.add_argument(
        "--crop-root",
        default="outputs/fashionai_multiview_v2_pipeline/pant_length/pipeline_all/01_yolo/pant_yolo_crops",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    det_json = Path(args.detections_json)
    crop_root = Path(args.crop_root)

    detections = load_detections(det_json)
    print(f"[OK] loaded detection records: {len(detections)}")

    for split in args.splits:
        process_split(
            task_dir=task_dir,
            split=split,
            detections=detections,
            crop_root=crop_root,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
