import argparse
import csv
import json
import statistics
import time
from pathlib import Path


def find_image(image_dir: Path, stem: str):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def percentile(values, p):
    if not values:
        return None
    values_sorted = sorted(values)
    k = int(round((len(values_sorted) - 1) * p / 100.0))
    return values_sorted[k]


def parse_annotation(anno_path: Path):
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []

    for key, value in data.items():
        if not key.startswith("item"):
            continue
        if not isinstance(value, dict):
            continue

        category_id = value.get("category_id")
        bbox = value.get("bounding_box")

        if category_id is None:
            continue

        instances.append(
            {
                "category_id": int(category_id),
                "bbox": bbox,
                "has_bbox": bbox is not None,
                "has_segmentation": "segmentation" in value,
                "has_landmarks": "landmarks" in value,
            }
        )

    return instances


def benchmark_split(split_name: str, image_dir: Path, anno_dir: Path, output_dir: Path, max_files=None):
    anno_files = sorted(anno_dir.glob("*.json"))
    if max_files is not None:
        anno_files = anno_files[:max_files]

    per_image_csv = output_dir / f"{split_name}_per_image_times.csv"
    failed_path = output_dir / f"{split_name}_failed.jsonl"

    times_ms = []
    instance_counts = []
    failed = []
    missing_images = 0
    total_instances = 0

    start_all = time.perf_counter()

    with open(per_image_csv, "w", newline="", encoding="utf-8") as f_csv, \
            open(failed_path, "w", encoding="utf-8") as f_failed:

        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "split",
                "anno_path",
                "image_path",
                "num_instances",
                "time_ms",
            ],
        )
        writer.writeheader()

        for idx, anno_path in enumerate(anno_files):
            stem = anno_path.stem
            image_path = find_image(image_dir, stem)
            if image_path is None:
                missing_images += 1

            t0 = time.perf_counter()
            try:
                instances = parse_annotation(anno_path)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                num_instances = len(instances)
                total_instances += num_instances
                times_ms.append(elapsed_ms)
                instance_counts.append(num_instances)

                writer.writerow(
                    {
                        "split": split_name,
                        "anno_path": str(anno_path),
                        "image_path": str(image_path) if image_path else "",
                        "num_instances": num_instances,
                        "time_ms": f"{elapsed_ms:.6f}",
                    }
                )

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                item = {
                    "anno_path": str(anno_path),
                    "error": repr(e),
                    "time_ms": elapsed_ms,
                }
                failed.append(item)
                f_failed.write(json.dumps(item, ensure_ascii=False) + "\n")

            if (idx + 1) % 10000 == 0:
                print(f"[INFO] {split_name}: processed {idx + 1}/{len(anno_files)}")

    total_elapsed_s = time.perf_counter() - start_all

    summary = {
        "split": split_name,
        "num_annotation_files": len(anno_files),
        "total_instances": total_instances,
        "failed_files": len(failed),
        "missing_images": missing_images,
        "total_elapsed_seconds": total_elapsed_s,
        "files_per_second": len(anno_files) / total_elapsed_s if total_elapsed_s > 0 else None,
        "instances_per_second": total_instances / total_elapsed_s if total_elapsed_s > 0 else None,
        "time_ms_mean": statistics.mean(times_ms) if times_ms else None,
        "time_ms_median": statistics.median(times_ms) if times_ms else None,
        "time_ms_p90": percentile(times_ms, 90),
        "time_ms_p95": percentile(times_ms, 95),
        "time_ms_p99": percentile(times_ms, 99),
        "time_ms_min": min(times_ms) if times_ms else None,
        "time_ms_max": max(times_ms) if times_ms else None,
        "instances_per_file_mean": statistics.mean(instance_counts) if instance_counts else None,
        "instances_per_file_median": statistics.median(instance_counts) if instance_counts else None,
        "per_image_csv": str(per_image_csv),
        "failed_jsonl": str(failed_path),
    }

    with open(output_dir / f"{split_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


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
        default=r"outputs\benchmark_deepfashion2_gt",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="For quick test. If omitted, run full split.",
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

    all_summary = {}

    for split_name, paths in splits.items():
        print(f"\n[INFO] Benchmarking split: {split_name}")
        summary = benchmark_split(
            split_name=split_name,
            image_dir=paths["image_dir"],
            anno_dir=paths["anno_dir"],
            output_dir=output_dir,
            max_files=args.max_files,
        )
        all_summary[split_name] = summary

        print(f"[DONE] {split_name}")
        print(f"  files: {summary['num_annotation_files']}")
        print(f"  instances: {summary['total_instances']}")
        print(f"  total seconds: {summary['total_elapsed_seconds']:.3f}")
        print(f"  files/sec: {summary['files_per_second']:.2f}")
        print(f"  mean ms/file: {summary['time_ms_mean']:.4f}")
        print(f"  p95 ms/file: {summary['time_ms_p95']:.4f}")

    with open(output_dir / "all_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] All benchmark results saved to: {output_dir}")


if __name__ == "__main__":
    main()
