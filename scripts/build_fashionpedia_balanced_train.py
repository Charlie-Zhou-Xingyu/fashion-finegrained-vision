#!/usr/bin/env python3
"""
Build a class-balanced train.txt + YAML config for Fashionpedia part detection.

Algorithm:
    1. Scan labels/train/*.txt → per-class annotation counts + per-image class sets.
    2. target = median(class_counts)
    3. repeat_factor[cls] = min(max_repeat, (target / count[cls]) ** power)
    4. image_repeat = ceil(max(repeat_factor[cls] for cls in image_classes))
    5. Write train_balanced.txt (each image path repeated image_repeat times,
       shuffled with --seed).
    6. Auto-generate matched YAML config, CSVs with per-class stats.

Usage:
    # Preview stats (no files written):
    python scripts/build_fashionpedia_balanced_train.py --dry-run

    # Preview from parquet (no YOLO dataset needed):
    python scripts/build_fashionpedia_balanced_train.py --from-parquet E:/fashionpedia/data

    # Full generation:
    python scripts/build_fashionpedia_balanced_train.py \
        --labels-dir E:/fashionpedia_yolo_19cls/labels/train \
        --images-dir E:/fashionpedia_yolo_19cls/images/train \
        --base-yaml E:/fashionpedia_yolo_19cls/fashionpedia_parts.yaml \
        --out-dir E:/fashionpedia_yolo_19cls \
        --power 1.0 --max-repeat 12
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import yaml


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_label_stats(labels_dir: Path, images_dir: Path) -> dict:
    """Scan YOLO label files; return per-class + per-image annotation stats."""
    class_counts: Counter[int] = Counter()
    image_classes: dict[str, set[int]] = {}
    image_class_instances: dict[str, Counter[int]] = {}

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files in {labels_dir}")

    for label_path in label_files:
        stem = label_path.stem
        classes: set[int] = set()
        img_counter: Counter[int] = Counter()

        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cls_id = int(line.split()[0])
                except (ValueError, IndexError):
                    continue
                classes.add(cls_id)
                class_counts[cls_id] += 1
                img_counter[cls_id] += 1

        if classes:
            image_classes[stem] = classes
            image_class_instances[stem] = img_counter

    # Count images with missing image files
    missing = 0
    for stem in image_classes:
        if not (images_dir / f"{stem}.jpg").exists() and not (images_dir / f"{stem}.png").exists():
            missing += 1

    return {
        "class_counts": class_counts,
        "image_classes": image_classes,
        "image_class_instances": image_class_instances,
        "total_images": len(image_classes),
        "total_annotations": sum(class_counts.values()),
        "missing_images": missing,
    }


def _load_parquet_stats(parquet_dir: Path) -> dict:
    """Read Fashionpedia parquet files; return same stats dict as _load_label_stats."""
    import glob as glob_mod
    import pandas as pd

    PART_IDS = set(range(27, 46))  # Fashionpedia part categories

    class_counts: Counter[int] = Counter()
    image_classes: dict[str, set[int]] = {}
    image_class_instances: dict[str, Counter[int]] = {}

    for pq_path_str in sorted(glob_mod.glob(str(parquet_dir / "train-*.parquet"))):
        pq_path = Path(pq_path_str)
        print(f"  Reading {pq_path.name} ...")
        df = pd.read_parquet(pq_path)
        for _, row in df.iterrows():
            img_id = str(row["image_id"])
            classes: set[int] = set()
            img_counter: Counter[int] = Counter()
            for cat_id in row["objects"]["category"]:
                if cat_id in PART_IDS:
                    yolo_cls = cat_id - 27
                    classes.add(yolo_cls)
                    class_counts[yolo_cls] += 1
                    img_counter[yolo_cls] += 1
            if classes:
                image_classes[img_id] = classes
                image_class_instances[img_id] = img_counter

    print(f"  Parsed {len(image_classes)} images with parts, "
          f"{sum(class_counts.values())} annotations\n")
    return {
        "class_counts": class_counts,
        "image_classes": image_classes,
        "image_class_instances": image_class_instances,
        "total_images": len(image_classes),
        "total_annotations": sum(class_counts.values()),
        "missing_images": 0,
    }


def _compute_repeat_factors(
    class_counts: Counter[int], power: float, max_repeat: int,
) -> dict[int, float]:
    """Power-law repeat factors capped at max_repeat."""
    counts = sorted(class_counts.values())
    n = len(counts)
    if n == 0:
        return {}
    target = counts[n // 2] if n % 2 == 1 else (counts[n // 2 - 1] + counts[n // 2]) / 2.0

    factors: dict[int, float] = {}
    for cls_id, cnt in sorted(class_counts.items()):
        if cnt == 0:
            factors[cls_id] = float(max_repeat)
        else:
            factors[cls_id] = min(float(max_repeat), (target / cnt) ** power)
    return factors


def _compute_image_repeats(
    image_classes: dict[str, set[int]],
    repeat_factors: dict[int, float],
) -> dict[str, int]:
    """Per-image repeat = ceil(max factor over classes in image)."""
    result: dict[str, int] = {}
    for stem, classes in image_classes.items():
        max_factor = max(repeat_factors.get(c, 1.0) for c in classes)
        result[stem] = max(1, math.ceil(max_factor))
    return result


def _find_image_path(stem: str, images_dir: Path) -> Path:
    for ext in (".jpg", ".png"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return images_dir / f"{stem}.jpg"


def _get_class_names(yaml_path: Optional[Path]) -> dict[int, str]:
    """Read class names from YAML, or fall back to Fashionpedia 19-part defaults."""
    if yaml_path and yaml_path.exists():
        try:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            if isinstance(names, list):
                return dict(enumerate(names))
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
        except Exception:
            pass

    # Fallback
    return dict(enumerate([
        "hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline",
        "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe",
        "ribbon", "rivet", "ruffle", "sequin", "tassel",
    ]))


# ── Report / output writers ────────────────────────────────────────────────────

def _print_console_report(
    class_counts: Counter[int],
    repeat_factors: dict[int, float],
    balanced_counts: dict[int, int],
    image_repeats: dict[str, int],
    class_names: dict[int, str],
    power: float,
    max_repeat: int,
    source_label: str,
    missing_images: int,
    expansion_ratio: float,
    n_original: int,
    n_balanced: int,
) -> None:
    """Print the full console report (used by both --dry-run and real runs)."""
    sorted_cls = sorted(class_counts.keys())
    n_cls = len(sorted_cls)
    total_balanced = sum(balanced_counts.values())
    total_orig_ann = sum(class_counts.values())

    # Expansion warning
    warn = ""
    if expansion_ratio > 2.5:
        warn = "  *** HIGH-RISK: expansion_ratio > 2.5 — training time may be excessive ***"
    elif expansion_ratio > 1.5:
        warn = "  ** WARNING: expansion_ratio > 1.5 — training time may increase noticeably **"

    print("=" * 72)
    print("FASHIONPEDIA BALANCED TRAIN LIST BUILDER")
    print("=" * 72)
    print(f"  Source           : {source_label}")
    print(f"  Power (p)        : {power}")
    print(f"  Max repeat       : {max_repeat}")
    print(f"  Classes found    : {n_cls}")
    print(f"  unique_images    : {n_original}")
    print(f"  original_train_entries  : {n_original}")
    print(f"  balanced_train_entries  : {n_balanced}")
    print(f"  expansion_ratio  : {expansion_ratio:.2f}x")
    if warn:
        print(warn)
    if missing_images:
        print(f"  WARNING: {missing_images} label files have no matching image")
    print()

    # Per-class table
    print(f"{'Cls':>4s} {'Name':>16s} {'Before':>8s} {'After':>8s} "
          f"{'Factor':>8s} {'Images':>8s} {'Ratio':>8s}")
    print("-" * 72)

    image_count_per_cls: dict[int, int] = defaultdict(int)
    for stem, classes in image_repeats.items():
        # we need the original image_classes dict — not available here directly
        pass
    # We'll compute image_count_per_cls from the stats dict before calling this

    orig_vals = sorted(class_counts.values())
    before_max_min = orig_vals[-1] / max(1, orig_vals[0])
    bal_vals = sorted(balanced_counts.values())
    after_max_min = bal_vals[-1] / max(1, bal_vals[0])

    for cls_id in sorted_cls:
        name = class_names.get(cls_id, f"cls_{cls_id}")
        before = class_counts.get(cls_id, 0)
        after = balanced_counts.get(cls_id, 0)
        factor = repeat_factors.get(cls_id, 1.0)
        n_imgs = _image_count_for_cls(cls_id, image_repeats)  # computed below
        ratio = after / max(1, before)
        below = " ***" if before < orig_vals[n_cls // 2] else ""
        print(f"{cls_id:4d} {name:>16s} {before:8d} {after:8d} "
              f"{factor:8.3f} {n_imgs:8d} {ratio:7.2f}x{below}")

    print("-" * 72)
    print(f"{'TOTAL':>21s} {total_orig_ann:8d} {total_balanced:8d}")
    print()

    # Summary
    print("SUMMARY:")
    print(f"  before max:min ratio : {before_max_min:.1f}x")
    print(f"  after  max:min ratio : {after_max_min:.1f}x")
    print(f"  expansion_ratio      : {expansion_ratio:.2f}x")
    print()

    # Repeat distribution
    repeat_dist = Counter(image_repeats.values())
    print(f"repeat_histogram ({len(repeat_dist)} bins):")
    for r in sorted(repeat_dist):
        pct = 100.0 * repeat_dist[r] / max(1, n_original)
        bar = "█" * int(pct / 2)
        print(f"  repeat={r:2d}: {repeat_dist[r]:6d} images ({pct:5.1f}%) {bar}")
    print()

    # Top/bottom
    print("TOP 5 UNDER-REPRESENTED (highest repeat factor):")
    top5 = sorted(repeat_factors.items(), key=lambda x: -x[1])[:5]
    for cls_id, factor in top5:
        name = class_names.get(cls_id, f"cls_{cls_id}")
        print(f"  {cls_id:2d} {name:>16s}: factor={factor:.3f}, count={class_counts.get(cls_id, 0)}")
    print()
    print("TOP 5 OVER-REPRESENTED (lowest repeat factor):")
    bot5 = sorted(repeat_factors.items(), key=lambda x: x[1])[:5]
    for cls_id, factor in bot5:
        name = class_names.get(cls_id, f"cls_{cls_id}")
        print(f"  {cls_id:2d} {name:>16s}: factor={factor:.3f}, count={class_counts.get(cls_id, 0)}")
    print("=" * 72)


# ponytail: module-level cache to avoid recomputing per-class image counts
_image_count_cache: dict[int, int] = {}


def _image_count_for_cls(cls_id: int, image_repeats: dict[str, int]) -> int:
    """Count unique images containing cls_id.  Best-effort — uses cached data."""
    return _image_count_cache.get(cls_id, 0)


def _write_train_txt(
    image_classes: dict[str, set[int]],
    image_repeats: dict[str, int],
    images_dir: Path,
    output_path: Path,
    seed: int,
) -> int:
    """Write train_balanced.txt with shuffled repeated image paths. Returns line count."""
    entries: list[str] = []
    for stem in sorted(image_classes):
        img_path = _find_image_path(stem, images_dir)
        n = image_repeats.get(stem, 1)
        entries.extend([str(img_path.resolve())] * n)

    rng = random.Random(seed)
    rng.shuffle(entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in entries:
            f.write(line + "\n")
    return len(entries)


def _write_balanced_yaml(
    base_yaml_path: Path,
    balanced_txt_rel: str,
    output_path: Path,
) -> None:
    """Read base_yaml_path, replace `train:` field, write to output_path."""
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = balanced_txt_rel

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_balanced(
    *,
    labels_dir: Path,
    images_dir: Path,
    base_yaml: Path,
    out_dir: Path,
    power: float = 1.0,
    max_repeat: int = 12,
    seed: int = 42,
    dry_run: bool = False,
    from_parquet: Optional[Path] = None,
) -> dict:
    """Full pipeline: stats → repeat factors → write all output files. Returns summary dict."""

    # ── 1. Load stats ─────────────────────────────────────────────────────
    if from_parquet:
        source_label = f"Parquet: {from_parquet}"
        stats = _load_parquet_stats(from_parquet)
        class_names = _get_class_names(None)  # fallback names
    else:
        source_label = f"Labels: {labels_dir}"
        stats = _load_label_stats(labels_dir, images_dir)
        class_names = _get_class_names(base_yaml if base_yaml.exists() else None)

    class_counts = stats["class_counts"]
    image_classes = stats["image_classes"]
    image_class_instances = stats["image_class_instances"]
    n_original = stats["total_images"]

    # ── 2. Compute repeat factors ──────────────────────────────────────────
    repeat_factors = _compute_repeat_factors(class_counts, power, max_repeat)

    # ── 3. Compute image repeats ──────────────────────────────────────────
    image_repeats = _compute_image_repeats(image_classes, repeat_factors)

    # ── 4. Compute balanced class counts (annotation-accurate) ─────────────
    balanced_counts: dict[int, int] = defaultdict(int)
    image_count_per_cls: dict[int, set[str]] = defaultdict(set)

    for stem, n_repeat in image_repeats.items():
        inst_map = image_class_instances.get(stem, {})
        for cls_id, inst_count in inst_map.items():
            balanced_counts[cls_id] += inst_count * n_repeat
            image_count_per_cls[cls_id].add(stem)

    # Cache for _image_count_for_cls
    global _image_count_cache
    _image_count_cache = {k: len(v) for k, v in image_count_per_cls.items()}

    n_balanced = sum(image_repeats.values())
    expansion_ratio = n_balanced / max(1, n_original)

    # ── 5. Console report ─────────────────────────────────────────────────
    _print_console_report(
        class_counts=class_counts,
        repeat_factors=repeat_factors,
        balanced_counts=balanced_counts,
        image_repeats=image_repeats,
        class_names=class_names,
        power=power,
        max_repeat=max_repeat,
        source_label=source_label,
        missing_images=stats["missing_images"],
        expansion_ratio=expansion_ratio,
        n_original=n_original,
        n_balanced=n_balanced,
    )

    if dry_run:
        print("[DRY RUN] No files written. Remove --dry-run to generate outputs.")
        if from_parquet:
            print("Note: --from-parquet shows stats only. Run without it after YOLO dataset conversion.")
        return {"expansion_ratio": expansion_ratio, "n_original": n_original, "n_balanced": n_balanced}

    if from_parquet:
        print("Note: --from-parquet shows stats only. No files written.")
        return {"expansion_ratio": expansion_ratio, "n_original": n_original, "n_balanced": n_balanced}

    # ── 6. Write outputs ──────────────────────────────────────────────────
    tag = f"p{power}_r{max_repeat}"

    # 6a. train_balanced.txt
    train_txt_path = out_dir / f"train_balanced_{tag}.txt"
    lines_written = _write_train_txt(image_classes, image_repeats, images_dir, train_txt_path, seed)
    print(f"Written: {train_txt_path} ({lines_written:,} lines)")

    # 6b. balanced YAML
    yaml_path = out_dir / f"fashionpedia_parts_balanced_{tag}.yaml"
    _write_balanced_yaml(base_yaml, f"train_balanced_{tag}.txt", yaml_path)
    print(f"Written: {yaml_path}")

    # 6c. balance_report.csv
    orig_vals = sorted(class_counts.values())
    bal_vals = sorted(balanced_counts.values())
    n_cls = len(orig_vals)
    before_max_min = orig_vals[-1] / max(1, orig_vals[0])
    after_max_min = bal_vals[-1] / max(1, bal_vals[0])

    _write_csv(out_dir / "balance_report.csv", [{
        "metric": "unique_images", "value": n_original,
        "description": "Number of unique training images with at least one part annotation"},
        {"metric": "original_train_entries", "value": n_original,
         "description": "Entries in original train.txt (one per image)"},
        {"metric": "balanced_train_entries", "value": n_balanced,
         "description": "Entries in balanced train.txt (with repeats)"},
        {"metric": "expansion_ratio", "value": f"{expansion_ratio:.4f}",
         "description": "balanced_train_entries / original_train_entries"},
        {"metric": "before_max_min_ratio", "value": f"{before_max_min:.2f}",
         "description": "Max / min class annotation count before balancing"},
        {"metric": "after_max_min_ratio", "value": f"{after_max_min:.2f}",
         "description": "Max / min class annotation count after balancing"},
        {"metric": "power", "value": str(power),
         "description": "Power-law exponent"},
        {"metric": "max_repeat", "value": str(max_repeat),
         "description": "Maximum per-image repeat cap"},
        {"metric": "seed", "value": str(seed),
         "description": "Shuffle seed for train_balanced.txt"},
        {"metric": "n_classes", "value": str(n_cls),
         "description": "Number of target classes"},
        {"metric": "total_annotations", "value": str(stats["total_annotations"]),
         "description": "Total annotation instances in training set"},
        {"metric": "expansion_risk", "value":
         "HIGH" if expansion_ratio > 2.5 else ("WARNING" if expansion_ratio > 1.5 else "OK"),
         "description": "Risk level based on expansion_ratio"},
    ], ["metric", "value", "description"])
    print(f"Written: {out_dir / 'balance_report.csv'}")

    # 6d. class_distribution_before_after.csv
    dist_rows = []
    for cls_id in sorted(class_counts.keys()):
        dist_rows.append({
            "class_id": cls_id,
            "class_name": class_names.get(cls_id, f"cls_{cls_id}"),
            "before_count": class_counts[cls_id],
            "after_count": balanced_counts.get(cls_id, 0),
            "repeat_factor": round(repeat_factors.get(cls_id, 1.0), 4),
            "image_count": len(image_count_per_cls.get(cls_id, set())),
            "after_ratio": round(balanced_counts.get(cls_id, 0) / max(1, class_counts[cls_id]), 2),
        })
    _write_csv(out_dir / "class_distribution_before_after.csv", dist_rows,
               ["class_id", "class_name", "before_count", "after_count",
                "repeat_factor", "image_count", "after_ratio"])
    print(f"Written: {out_dir / 'class_distribution_before_after.csv'}")

    # 6e. repeat_histogram.csv
    repeat_dist = Counter(image_repeats.values())
    hist_rows = []
    for r in sorted(repeat_dist):
        hist_rows.append({
            "repeat_count": r,
            "n_images": repeat_dist[r],
            "pct_images": round(100.0 * repeat_dist[r] / max(1, n_original), 2),
        })
    _write_csv(out_dir / "repeat_histogram.csv", hist_rows,
               ["repeat_count", "n_images", "pct_images"])
    print(f"Written: {out_dir / 'repeat_histogram.csv'}")

    # ── 7. Training command hint ──────────────────────────────────────────
    print(f"\nTrain command:")
    print(f"  yolo detect train data={yaml_path} \\")
    print(f"      model=yolov8s.pt \\")
    print(f"      epochs=100 imgsz=640 batch=16 device=0 \\")
    print(f"      project=outputs/fashionpedia_19cls_yolov8s_balanced \\")
    print(f"      name={tag}")
    print("=" * 72)

    return {
        "expansion_ratio": expansion_ratio,
        "n_original": n_original,
        "n_balanced": n_balanced,
        "train_txt": str(train_txt_path),
        "yaml": str(yaml_path),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--labels-dir", type=Path,
                    default=Path("E:/fashionpedia_yolo_19cls/labels/train"),
                    help="YOLO label .txt directory [default: E:/fashionpedia_yolo_19cls/labels/train]")
    ap.add_argument("--images-dir", type=Path,
                    default=Path("E:/fashionpedia_yolo_19cls/images/train"),
                    help="Corresponding image directory [default: E:/fashionpedia_yolo_19cls/images/train]")
    ap.add_argument("--base-yaml", type=Path,
                    default=Path("E:/fashionpedia_yolo_19cls/fashionpedia_parts.yaml"),
                    help="Base YOLO dataset YAML to copy nc/names from [default: E:/fashionpedia_yolo_19cls/fashionpedia_parts.yaml]")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("E:/fashionpedia_yolo_19cls"),
                    help="Output directory for all generated files [default: E:/fashionpedia_yolo_19cls]")
    ap.add_argument("--power", "-p", type=float, default=1.0,
                    help="Power-law exponent (0=uniform, 1=inverse freq) [default: 1.0]")
    ap.add_argument("--max-repeat", "-r", type=int, default=12,
                    help="Maximum per-image repeat cap [default: 12]")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed for train_balanced.txt [default: 42]")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print statistics only; do not write any files")
    ap.add_argument("--from-parquet", type=Path, default=None, metavar="DIR",
                    help="Read directly from Fashionpedia parquet files (e.g. E:/fashionpedia/data). "
                         "Stats-only — no files written.")
    args = ap.parse_args()

    # Validation
    if not args.from_parquet and not args.dry_run:
        for p, name in [(args.labels_dir, "labels-dir"), (args.images_dir, "images-dir"),
                        (args.base_yaml, "base-yaml")]:
            if not p.exists():
                ap.error(f"{name} not found: {p}\n"
                         f"  Run convert_fashionpedia_to_yolo.py first, or use --from-parquet to preview.")

    build_balanced(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        base_yaml=args.base_yaml,
        out_dir=args.out_dir,
        power=args.power,
        max_repeat=args.max_repeat,
        seed=args.seed,
        dry_run=args.dry_run,
        from_parquet=args.from_parquet,
    )


if __name__ == "__main__":
    main()
