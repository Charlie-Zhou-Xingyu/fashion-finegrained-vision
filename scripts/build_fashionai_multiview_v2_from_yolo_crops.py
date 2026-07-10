import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_path(s):
    if s is None:
        return None
    return str(s).replace("\\", "/")


def filename_key(path_str):
    return Path(str(path_str).replace("\\", "/")).stem.lower()


def build_det_index(detections, prefer_classes=None):
    """
    Build filename -> best YOLO detection mapping.
    The script tries to choose the largest bbox/crop per image.
    If prefer_classes is provided, detections whose class_name contains one
    of the preferred keywords will be prioritized.
    """
    prefer_classes = [x.lower() for x in (prefer_classes or [])]
    index = {}

    if isinstance(detections, dict):
        if "results" in detections:
            det_list = detections["results"]
        elif "images" in detections:
            det_list = detections["images"]
        elif "detections" in detections:
            det_list = detections["detections"]
        else:
            det_list = []
            for v in detections.values():
                if isinstance(v, list):
                    det_list.extend(v)
    elif isinstance(detections, list):
        det_list = detections
    else:
        det_list = []

    # Flatten common nested formats
    flat = []
    for item in det_list:
        if isinstance(item, dict) and "detections" in item and isinstance(item["detections"], list):
            image_path = item.get("image_path") or item.get("path") or item.get("file_name") or item.get("image")
            for d in item["detections"]:
                if isinstance(d, dict):
                    dd = dict(d)
                    if image_path and not (dd.get("image_path") or dd.get("source_image_path") or dd.get("file_name")):
                        dd["image_path"] = image_path
                    flat.append(dd)
        elif isinstance(item, dict):
            flat.append(item)

    for d in flat:
        image_path = (
            d.get("image_path")
            or d.get("source_image_path")
            or d.get("file_name")
            or d.get("filename")
            or d.get("image")
            or d.get("path")
        )
        crop_path = (
            d.get("crop_path")
            or d.get("crop_image_path")
            or d.get("saved_crop_path")
            or d.get("crop")
        )
        bbox = d.get("bbox_xyxy") or d.get("xyxy") or d.get("bbox")
        class_name = str(d.get("class_name") or d.get("name") or d.get("label") or d.get("cls_name") or "").lower()
        conf = float(d.get("confidence") or d.get("conf") or d.get("score") or 0.0)
        det_id = d.get("det_id", d.get("id", d.get("index", None)))

        if not image_path:
            continue

        key = filename_key(image_path)

        area = 0.0
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = map(float, bbox[:4])
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            except Exception:
                area = 0.0

        prefer_score = 0
        if prefer_classes:
            prefer_score = 1 if any(k in class_name for k in prefer_classes) else 0

        score_tuple = (prefer_score, area, conf)

        current = index.get(key)
        if current is None or score_tuple > current["_score_tuple"]:
            nd = dict(d)
            nd["_score_tuple"] = score_tuple
            nd["_crop_path"] = crop_path
            nd["_bbox_xyxy"] = bbox
            nd["_class_name"] = class_name
            nd["_det_id"] = det_id
            index[key] = nd

    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--input-index-dir", required=True)
    ap.add_argument("--detections-json", required=True)
    ap.add_argument("--output-index-dir", required=True)
    ap.add_argument("--views", nargs="+", default=["original", "yolo_crop"])
    ap.add_argument("--prefer-classes", nargs="*", default=[])
    ap.add_argument("--copy-label-map", action="store_true")
    args = ap.parse_args()

    task = args.task
    input_index_dir = Path(args.input_index_dir)
    output_index_dir = Path(args.output_index_dir)
    output_index_dir.mkdir(parents=True, exist_ok=True)

    detections = load_json(Path(args.detections_json))
    det_index = build_det_index(detections, args.prefer_classes)

    summary = {}

    for split in ["train", "val", "test"]:
        in_path = input_index_dir / f"{task}_{split}.jsonl"
        rows = load_jsonl(in_path)

        out_rows = []
        with_crop = 0
        without_crop = 0

        for row in rows:
            src_path = row.get("image_path") or row.get("source_image_path") or row.get("image_relative_path")
            key = filename_key(src_path)
            det = det_index.get(key)

            if "original" in args.views:
                original = dict(row)
                original["source_image_path"] = row.get("source_image_path") or row.get("image_path")
                original["view_type"] = "original"
                original["multi_view_source_id"] = row.get("sample_id")
                original["v2_pipeline_aligned"] = True
                if det:
                    original["pipeline_class_name"] = det.get("_class_name")
                    original["pipeline_det_id"] = det.get("_det_id")
                    original["pipeline_bbox_xyxy"] = det.get("_bbox_xyxy")
                out_rows.append(original)

            if "yolo_crop" in args.views:
                crop_path = det.get("_crop_path") if det else None
                if crop_path:
                    crop_row = dict(row)
                    crop_row["source_image_path"] = row.get("source_image_path") or row.get("image_path")
                    crop_row["image_path"] = crop_path
                    crop_row["view_type"] = "yolo_crop"
                    crop_row["multi_view_source_id"] = row.get("sample_id")
                    crop_row["v2_pipeline_aligned"] = True
                    crop_row["pipeline_class_name"] = det.get("_class_name")
                    crop_row["pipeline_det_id"] = det.get("_det_id")
                    crop_row["pipeline_bbox_xyxy"] = det.get("_bbox_xyxy")
                    out_rows.append(crop_row)
                    with_crop += 1
                else:
                    without_crop += 1

        out_path = output_index_dir / f"{task}_{split}.jsonl"
        save_jsonl(out_path, out_rows)

        summary[split] = {
            "input": len(rows),
            "output": len(out_rows),
            "with_yolo_crop": with_crop,
            "without_yolo_crop": without_crop,
            "detections_index_size": len(det_index),
        }

        print(
            f"[OK] {split}: input={len(rows)}, output={len(out_rows)}, "
            f"with_yolo_crop={with_crop}, without_yolo_crop={without_crop}"
        )

    if args.copy_label_map:
        src = input_index_dir / f"label_map_{task}.json"
        dst = output_index_dir / f"label_map_{task}.json"
        if src.exists():
            shutil.copy2(src, dst)

    summary_path = output_index_dir / f"multiview_v2_yolo_summary_{task}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
