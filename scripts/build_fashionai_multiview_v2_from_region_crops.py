# scripts/build_fashionai_multiview_v2_from_region_crops.py
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SPLITS = ["train", "val", "test"]


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


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm_path_key(path: Any) -> str:
    if path is None:
        return ""
    try:
        return str(Path(str(path)).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path).replace("\\", "/").lower()


def filename_key(path: Any) -> str:
    if path is None:
        return ""
    return Path(str(path)).name.lower()


def get_score(record: Dict[str, Any]) -> float:
    for key in ["det_conf", "confidence", "score", "conf"]:
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return 0.0


def group_region_crops(crops: List[Dict[str, Any]], region: str) -> Dict[str, Dict[str, Any]]:
    """
    Return image_path_key -> best crop record.
    Uses full normalized path when possible and filename fallback.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for r in crops:
        if str(r.get("region", "")) != region:
            continue
        if not r.get("success", True):
            continue

        image_path = (
            r.get("image_path")
            or r.get("source_image_path")
            or r.get("original_image_path")
            or r.get("full_image_path")
            or r.get("img_path")
        )

        keys = [norm_path_key(image_path), filename_key(image_path)]
        for k in keys:
            if not k:
                continue
            grouped.setdefault(k, []).append(r)

    best: Dict[str, Dict[str, Any]] = {}
    for k, items in grouped.items():
        items_sorted = sorted(items, key=get_score, reverse=True)
        best[k] = items_sorted[0]

    return best


def find_crop_for_sample(
    sample: Dict[str, Any],
    crop_map: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    image_path = sample.get("image_path")
    keys = [norm_path_key(image_path), filename_key(image_path)]
    for k in keys:
        if k and k in crop_map:
            return crop_map[k]
    return None


def make_view_row(
    sample: Dict[str, Any],
    image_path: str,
    view_type: str,
    source_image_path: str,
    crop_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = dict(sample)
    row["image_path"] = image_path
    row["source_image_path"] = source_image_path
    row["view_type"] = view_type
    row["multi_view_source_id"] = sample.get("sample_id")
    row["v2_pipeline_aligned"] = True

    if crop_record is not None:
        row["pipeline_class_name"] = crop_record.get("class_name")
        row["pipeline_det_id"] = crop_record.get("det_id")
        row["pipeline_region"] = crop_record.get("region")
        row["pipeline_component"] = crop_record.get("component")
        row["pipeline_bbox_xyxy"] = crop_record.get("bbox_xyxy")
        row["pipeline_expanded_bbox_xyxy"] = crop_record.get("expanded_bbox_xyxy")
        row["pipeline_upper_bbox_xyxy"] = crop_record.get("upper_bbox_xyxy")

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--input-index-dir", default="data/fashionai_attribute_index", type=str)
    parser.add_argument("--region-crops-json", required=True, type=str)
    parser.add_argument("--output-index-dir", required=True, type=str)
    parser.add_argument("--region", default="collar", type=str)

    parser.add_argument(
        "--views",
        nargs="+",
        default=["original", "expanded_collar", "upper_crop"],
        choices=["original", "image_crop", "expanded_collar", "upper_crop"],
    )

    parser.add_argument("--copy-label-map", action="store_true")
    parser.add_argument("--limit-per-split", default=0, type=int)

    args = parser.parse_args()

    input_dir = Path(args.input_index_dir)
    output_dir = Path(args.output_index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(Path(args.region_crops_json))
    crops = data.get("crops", [])
    if not isinstance(crops, list):
        raise ValueError("region_crops_json must contain list field: crops")

    crop_map = group_region_crops(crops, region=args.region)

    summary = {
        "task": args.task,
        "region": args.region,
        "region_crops_json": args.region_crops_json,
        "output_index_dir": str(output_dir),
        "views": args.views,
        "num_region_crops": len(crops),
        "num_grouped_images": len(crop_map),
        "splits": {},
    }

    for split in SPLITS:
        input_jsonl = input_dir / f"{args.task}_{split}.jsonl"
        samples = read_jsonl(input_jsonl)
        if args.limit_per_split and args.limit_per_split > 0:
            samples = samples[: args.limit_per_split]

        out_rows: List[Dict[str, Any]] = []
        n_with_crop = 0
        n_without_crop = 0
        n_added_original = 0
        n_added_image_crop = 0
        n_added_expanded = 0
        n_added_upper = 0

        for sample in samples:
            source_image_path = str(sample.get("image_path"))
            crop_record = find_crop_for_sample(sample, crop_map)

            if "original" in args.views:
                out_rows.append(
                    make_view_row(
                        sample=sample,
                        image_path=source_image_path,
                        view_type="original",
                        source_image_path=source_image_path,
                        crop_record=crop_record,
                    )
                )
                n_added_original += 1

            if crop_record is None:
                n_without_crop += 1
                continue

            n_with_crop += 1

            if "image_crop" in args.views:
                p = crop_record.get("image_crop_path") or crop_record.get("crop_path")
                if p and Path(str(p)).exists():
                    out_rows.append(
                        make_view_row(sample, str(p), "image_crop", source_image_path, crop_record)
                    )
                    n_added_image_crop += 1

            if "expanded_collar" in args.views:
                p = crop_record.get("expanded_crop_path")
                if p and Path(str(p)).exists():
                    out_rows.append(
                        make_view_row(sample, str(p), "expanded_collar", source_image_path, crop_record)
                    )
                    n_added_expanded += 1

            if "upper_crop" in args.views:
                p = crop_record.get("upper_crop_path")
                if p and Path(str(p)).exists():
                    out_rows.append(
                        make_view_row(sample, str(p), "upper_crop", source_image_path, crop_record)
                    )
                    n_added_upper += 1

        output_jsonl = output_dir / f"{args.task}_{split}.jsonl"
        write_jsonl(out_rows, output_jsonl)

        summary["splits"][split] = {
            "input_samples": len(samples),
            "output_rows": len(out_rows),
            "samples_with_crop": n_with_crop,
            "samples_without_crop": n_without_crop,
            "added_original": n_added_original,
            "added_image_crop": n_added_image_crop,
            "added_expanded_collar": n_added_expanded,
            "added_upper_crop": n_added_upper,
            "output_jsonl": str(output_jsonl),
        }

        print(
            f"[OK] {split}: input={len(samples)}, output={len(out_rows)}, "
            f"with_crop={n_with_crop}, without_crop={n_without_crop}, "
            f"original={n_added_original}, expanded={n_added_expanded}, upper={n_added_upper}"
        )

    label_src = input_dir / f"label_map_{args.task}.json"
    if label_src.exists():
        label_dst = output_dir / f"label_map_{args.task}.json"
        shutil.copy2(label_src, label_dst)
        summary["label_map"] = str(label_dst)

    save_json(summary, output_dir / f"multiview_v2_summary_{args.task}.json")
    print(f"[OK] Saved summary: {output_dir / f'multiview_v2_summary_{args.task}.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
