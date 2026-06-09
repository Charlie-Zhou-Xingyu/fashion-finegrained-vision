import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEEPFASHION2_CATEGORIES = {
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

PRD_CATEGORIES = {
    1: "top",
    2: "pants",
    3: "skirt",
    4: "outerwear",
    5: "dress",
    6: "shoes",
    7: "bag",
    8: "accessory",
}

DEEPFASHION2_TO_PRD = {
    1: 1,
    2: 1,
    3: 4,
    4: 4,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 5,
    11: 5,
    12: 5,
    13: 5,
}


def find_image(image_dir: Path, stem: str):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def scan_split(split_name: str, image_dir: Path, anno_dir: Path):
    anno_files = sorted(anno_dir.glob("*.json"))

    raw_category_counter = Counter()
    prd_category_counter = Counter()
    instance_count_per_image = Counter()

    failed_annos = []
    missing_images = []
    empty_annos = []

    total_instances = 0
    valid_annos = 0

    for anno_path in anno_files:
        stem = anno_path.stem
        image_path = find_image(image_dir, stem)

        if image_path is None:
            missing_images.append(str(anno_path))

        try:
            with open(anno_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            failed_annos.append(
                {
                    "anno_path": str(anno_path),
                    "error": repr(e),
                }
            )
            continue

        image_instances = 0

        for key, value in data.items():
            if not key.startswith("item"):
                continue
            if not isinstance(value, dict):
                continue

            category_id = value.get("category_id")
            if category_id is None:
                continue

            try:
                category_id = int(category_id)
            except Exception:
                continue

            raw_category_counter[category_id] += 1

            prd_id = DEEPFASHION2_TO_PRD.get(category_id)
            if prd_id is not None:
                prd_category_counter[prd_id] += 1

            total_instances += 1
            image_instances += 1

        if image_instances == 0:
            empty_annos.append(str(anno_path))
        else:
            valid_annos += 1

        instance_count_per_image[image_instances] += 1

    result = {
        "split": split_name,
        "image_dir": str(image_dir),
        "anno_dir": str(anno_dir),
        "total_anno_files": len(anno_files),
        "valid_anno_files_with_instances": valid_annos,
        "empty_anno_files": len(empty_annos),
        "missing_image_files": len(missing_images),
        "failed_anno_files": len(failed_annos),
        "total_instances": total_instances,
        "raw_category_counts": {
            str(k): {
                "name": DEEPFASHION2_CATEGORIES.get(k, "unknown"),
                "count": v,
            }
            for k, v in sorted(raw_category_counter.items())
        },
        "prd_category_counts": {
            str(k): {
                "name": PRD_CATEGORIES.get(k, "unknown"),
                "count": prd_category_counter.get(k, 0),
            }
            for k in sorted(PRD_CATEGORIES)
        },
        "instance_count_per_image_distribution": {
            str(k): v for k, v in sorted(instance_count_per_image.items())
        },
    }

    return result, failed_annos, missing_images, empty_annos


def write_category_csv(path: Path, summary: dict):
    rows = []

    for raw_id, info in summary["raw_category_counts"].items():
        raw_id_int = int(raw_id)
        prd_id = DEEPFASHION2_TO_PRD.get(raw_id_int)
        rows.append(
            {
                "split": summary["split"],
                "raw_category_id": raw_id,
                "raw_category_name": info["name"],
                "raw_count": info["count"],
                "prd_category_id": prd_id,
                "prd_category_name": PRD_CATEGORIES.get(prd_id, "unknown"),
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "raw_category_id",
                "raw_category_name",
                "raw_count",
                "prd_category_id",
                "prd_category_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_list(path: Path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            if isinstance(item, dict):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                f.write(str(item) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\Aliintern\fashion-ai-data\deepfashion2",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"outputs\dataset_scan\deepfashion2",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": {
            "image_dir": root / "train" / "image",
            "anno_dir": root / "train" / "annos",
        },
        "validation": {
            "image_dir": root / "validation" / "image",
            "anno_dir": root / "validation" / "annos",
        },
    }

    all_summaries = {}

    for split_name, paths in splits.items():
        print(f"[INFO] Scanning split: {split_name}")
        summary, failed_annos, missing_images, empty_annos = scan_split(
            split_name=split_name,
            image_dir=paths["image_dir"],
            anno_dir=paths["anno_dir"],
        )

        all_summaries[split_name] = summary

        split_out = output_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        with open(split_out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        write_category_csv(split_out / "category_distribution.csv", summary)
        write_list(split_out / "failed_annos.jsonl", failed_annos)
        write_list(split_out / "missing_images.txt", missing_images)
        write_list(split_out / "empty_annos.txt", empty_annos)

        print(f"[INFO] {split_name} total instances: {summary['total_instances']}")
        print(f"[INFO] {split_name} failed annos: {summary['failed_anno_files']}")
        print(f"[INFO] {split_name} missing images: {summary['missing_image_files']}")

    with open(output_dir / "all_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
