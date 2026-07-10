"""
Two-stage YOLO train-list generator with optional hard-class oversampling.

Stage 1 — Inverse-frequency balancing
--------------------------------------
Each image's repeat count is proportional to the rarest class it contains,
using inverse-frequency weighting with a configurable exponent and ceiling.

  balance_power=0.5  →  inverse-sqrt  (mild; default in make_balanced_yolo_train_list.py)
  balance_power=1.0  →  inverse-linear (strong; recommended for severe imbalance)

Stage 2 — Hard-class extra copies (optional)
---------------------------------------------
Images that contain at least one user-specified "hard" fine class receive
``--hard-repeat-bonus`` additional entries in the list, capped at
``--hard-max-repeat`` total.

Hard-class candidates from confusion matrix analysis:

  ID   Name                  Val recall  Notes
  ---  --------------------  ----------  --------------------------------
   1   long sleeve top       ~70 %       confused with long sleeve outwear
   2   short sleeve outwear  46.5 %      142 GT val instances — tiny class
   3   long sleeve outwear   ~65 %       confused with long sleeve top
   5   sling                 47.8 %      322 GT val instances
  10   long sleeve dress     60.5 %      confused with long sleeve top
  11   vest dress            ~60 %       confused with vest / sling dress
  12   sling dress           62.6 %      confused with vest dress / skirt

Stage 3 — Target-size cap (optional)
--------------------------------------
If ``--target-size N`` is given, all repeat counts are scaled down
proportionally so the total list length ≈ N, with each image keeping
at least 1 entry.  Relative class ratios are preserved.

Usage
-----
(a) Stronger oversampling only (balance_power=1.0, max_repeat=8)::

    python tools/data/make_hardclass_oversampled_train_list.py ^
        --dataset-root data\\processed\\deepfashion2_yolo_13cls ^
        --yaml         data\\processed\\deepfashion2_yolo_13cls\\deepfashion2_13cls.yaml ^
        --balance-power 1.0 ^
        --max-repeat 8 ^
        --output-list-name train_balanced_p10_r8.txt ^
        --output-yaml-name deepfashion2_13cls_balanced_p10_r8.yaml

(b) Stronger oversampling + hard-class extra copies + 30k cap::

    python tools/data/make_hardclass_oversampled_train_list.py ^
        --dataset-root data\\processed\\deepfashion2_yolo_13cls ^
        --yaml         data\\processed\\deepfashion2_yolo_13cls\\deepfashion2_13cls.yaml ^
        --balance-power 1.0 ^
        --max-repeat 8 ^
        --hard-class-ids 1,2,3,5,10,11,12 ^
        --hard-repeat-bonus 3 ^
        --hard-max-repeat 12 ^
        --target-size 30000 ^
        --output-list-name train_balanced_30k_bp10_r8_hard.txt ^
        --output-yaml-name deepfashion2_13cls_balanced_30k_bp10_r8_hard.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _find_image_files(image_dir: Path) -> list[Path]:
    """Return all image files under *image_dir*, sorted for reproducibility."""
    files: list[Path] = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


def _load_label_classes(label_path: Path) -> list[int]:
    """Return class IDs from a YOLO label file. Returns [] if file is absent."""
    if not label_path.exists():
        return []
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            classes.append(int(float(parts[0])))
        except ValueError:
            continue
    return classes


def _parse_names_from_yaml(yaml_text: str) -> dict[int, str]:
    """Parse the ``names:`` block from a YOLO dataset YAML string."""
    names: dict[int, str] = {}
    in_names = False
    for raw_line in yaml_text.splitlines():
        line = raw_line.rstrip()
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            if not line.startswith("  "):
                break
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            try:
                names[int(key.strip())] = value.strip()
            except ValueError:
                continue
    return names


def _replace_train_line(yaml_text: str, new_train_value: str) -> str:
    """Swap the ``train:`` line in a YOLO YAML string."""
    lines = yaml_text.splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if not replaced and line.strip().startswith("train:"):
            out.append(f"train: {new_train_value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.insert(1, f"train: {new_train_value}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------


def scan_dataset(
    image_dir: Path,
    label_dir: Path,
) -> tuple[
    dict[str, list[int]],   # rel_path → class ids (with repeats)
    dict[str, list[int]],   # rel_path → unique class ids
    Counter,                 # class-level instance counts
    Counter,                 # class-level image counts
    int,                     # empty image count
]:
    """Scan all images and their labels, returning class statistics.

    Args:
        image_dir: Directory containing training images.
        label_dir: Directory containing YOLO label txt files.

    Returns:
        Tuple of (image_to_classes, image_to_unique_classes,
                  class_instance_counts, class_image_counts, empty_image_count).
    """
    image_files = _find_image_files(image_dir)
    image_to_classes: dict[str, list[int]] = {}
    image_to_unique: dict[str, list[int]] = {}
    instance_counts: Counter = Counter()
    image_counts: Counter = Counter()
    empty = 0

    for i, img_path in enumerate(image_files, 1):
        if i % 5000 == 0:
            print(f"  [scan] {i}/{len(image_files)}", flush=True)

        label_path = label_dir / f"{img_path.stem}.txt"
        classes = _load_label_classes(label_path)
        unique = sorted(set(classes))

        rel = img_path.name  # use filename as key; abs path built later
        image_to_classes[rel] = classes
        image_to_unique[rel] = unique

        if not classes:
            empty += 1
            continue

        instance_counts.update(classes)
        image_counts.update(unique)

    return image_to_classes, image_to_unique, instance_counts, image_counts, empty


# ---------------------------------------------------------------------------
# Stage 1 — inverse-frequency balancing
# ---------------------------------------------------------------------------


def compute_base_repeats(
    image_to_unique: dict[str, list[int]],
    image_counts: Counter,
    balance_power: float,
    max_repeat: int,
) -> dict[str, int]:
    """Compute per-image repeat counts using inverse-frequency weighting.

    Args:
        image_to_unique: Mapping of image filename → list of unique class IDs.
        image_counts: Number of images containing each class (for weight calc).
        balance_power: Exponent applied to inverse frequency. 1.0 = inverse-linear.
        max_repeat: Maximum repeat count per image.

    Returns:
        Mapping of image filename → repeat count (>= 1).
    """
    if not image_counts:
        return {k: 1 for k in image_to_unique}

    class_weights: dict[int, float] = {
        cls: 1.0 / (float(count) ** balance_power)
        for cls, count in image_counts.items()
    }
    min_weight = min(class_weights.values())

    repeats: dict[str, int] = {}
    for rel, unique in image_to_unique.items():
        if not unique:
            repeats[rel] = 1
        else:
            img_weight = max(class_weights[c] for c in unique)
            r = int(round(img_weight / min_weight))
            repeats[rel] = max(1, min(max_repeat, r))

    return repeats


# ---------------------------------------------------------------------------
# Stage 2 — hard-class extra copies
# ---------------------------------------------------------------------------


def apply_hard_class_boost(
    repeats: dict[str, int],
    image_to_unique: dict[str, list[int]],
    hard_classes: set[int],
    hard_extra_copies: int,
    hard_max_repeat: int,
) -> dict[str, int]:
    """Add extra copies for images containing hard fine classes.

    Images whose unique class set intersects *hard_classes* receive
    ``hard_extra_copies`` additional entries, capped at ``hard_max_repeat``.

    Args:
        repeats: Base repeat counts from Stage 1.
        image_to_unique: Mapping of image filename → unique class IDs.
        hard_classes: Set of fine class IDs to boost.
        hard_extra_copies: Extra copies added per hard-class image.
        hard_max_repeat: Maximum total repeat count after boost.

    Returns:
        Updated repeat counts.
    """
    boosted = dict(repeats)
    for rel, unique in image_to_unique.items():
        if hard_classes.intersection(unique):
            boosted[rel] = min(boosted[rel] + hard_extra_copies, hard_max_repeat)
    return boosted


# ---------------------------------------------------------------------------
# Stage 3 — target-size cap
# ---------------------------------------------------------------------------


def apply_target_size_cap(
    repeats: dict[str, int],
    target_size: int,
) -> dict[str, int]:
    """Scale down repeat counts so total list entries ≈ target_size.

    Proportionally scales all repeat counts while preserving relative
    class balance ratios.

    When target_size >= number of source images, each image keeps at
    least 1 entry (pure oversampling regime).

    When target_size < number of source images, proportional scaling
    will round some repeats to 0, effectively excluding those images
    from the training list (subsampling regime).  A warning is printed
    in this case.  Rare/boosted images survive longer because their
    higher repeat counts round to non-zero later.

    Args:
        repeats: Per-image repeat counts (after Stages 1 and 2).
        target_size: Desired total number of training list entries.

    Returns:
        Scaled repeat counts (may include 0-repeat entries which are
        excluded from the final list by build_train_list).
    """
    total = sum(repeats.values())
    if total <= target_size:
        return dict(repeats)

    num_images = len(repeats)
    scale = target_size / total

    if target_size < num_images:
        print(
            f"  [WARN] --target-size {target_size:,} is less than the number of "
            f"source images ({num_images:,}).  Some images will be excluded "
            f"(subsampling regime).  Rare/boosted images are preferentially kept."
        )

    capped: dict[str, int] = {}
    for rel, r in repeats.items():
        new_r = round(r * scale)
        # In oversampling regime enforce min 1; in subsampling regime allow 0.
        capped[rel] = max(1, new_r) if target_size >= num_images else new_r
    return capped


# ---------------------------------------------------------------------------
# List / YAML generation
# ---------------------------------------------------------------------------


def build_train_list(
    dataset_root: Path,
    image_dir: Path,
    repeats: dict[str, int],
) -> list[str]:
    """Build the final list of absolute image paths with repetitions.

    Args:
        dataset_root: Root of the YOLO dataset (for resolving absolute paths).
        image_dir: Directory containing training images.
        repeats: Per-image repeat counts keyed by filename.

    Returns:
        List of absolute path strings (one entry per effective training sample).
    """
    lines: list[str] = []
    for rel, count in sorted(repeats.items()):
        if count <= 0:
            continue
        abs_path = (image_dir / rel).resolve().as_posix()
        lines.extend([abs_path] * count)
    return lines


def write_train_list(lines: list[str], path: Path) -> None:
    """Write the training list to a text file.

    Args:
        lines: Absolute image paths (with repetitions).
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_yaml(
    source_yaml_text: str,
    train_list_path: Path,
    output_path: Path,
) -> None:
    """Write a YOLO dataset YAML pointing to the new train list.

    Args:
        source_yaml_text: Original YAML text (val path, names block reused).
        train_list_path: Absolute path to the new train image list.
        output_path: Destination YAML path.
    """
    new_yaml = _replace_train_line(source_yaml_text, train_list_path.as_posix())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_yaml, encoding="utf-8")


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------


def compute_effective_distribution(
    image_to_unique: dict[str, list[int]],
    repeats: dict[str, int],
    class_names: dict[int, str],
) -> dict[int, int]:
    """Compute effective image count per class after applying repeat counts.

    Args:
        image_to_unique: Mapping of image filename → unique class IDs.
        repeats: Per-image repeat counts.
        class_names: Class ID → name mapping.

    Returns:
        Mapping of class ID → effective image count.
    """
    effective: Counter = Counter()
    for rel, unique in image_to_unique.items():
        r = repeats.get(rel, 1)
        effective.update({c: r for c in unique})
    return dict(effective)


def print_distribution_table(
    label: str,
    raw_image_counts: Counter,
    effective_counts: dict[int, int],
    class_names: dict[int, str],
) -> None:
    """Print a side-by-side before/after class distribution table.

    Args:
        label: Header label for this table.
        raw_image_counts: Raw per-class image counts (before balancing).
        effective_counts: Effective per-class image counts (after balancing).
        class_names: Class ID → name mapping.
    """
    all_classes = sorted(set(raw_image_counts) | set(effective_counts))

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  {'ID':<4} {'Name':<24} {'Raw imgs':>10} {'Effective':>10} {'×':>6}")
    print(f"  {'-' * 58}")
    for cls in all_classes:
        name = class_names.get(cls, str(cls))
        raw = raw_image_counts.get(cls, 0)
        eff = effective_counts.get(cls, 0)
        mult = f"{eff / raw:.1f}" if raw > 0 else "—"
        print(f"  {cls:<4} {name:<24} {raw:>10,} {eff:>10,} {mult:>6}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def verify_outputs(
    yaml_path: Path,
    list_path: Path,
    label_dir: Path,
    label_count_before: int,
) -> bool:
    """Run post-generation sanity checks and print a report.

    Args:
        yaml_path: Expected output YAML path.
        list_path: Expected output train list path.
        label_dir: Label directory (used to confirm no label files changed).
        label_count_before: Number of label files counted before generation.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("\n--- Sanity check ---")
    passed = True

    if yaml_path.exists():
        print(f"  [OK] YAML exists: {yaml_path}")
    else:
        print(f"  [FAIL] YAML missing: {yaml_path}")
        passed = False

    if list_path.exists():
        line_count = sum(1 for _ in list_path.open(encoding="utf-8") if _.strip())
        print(f"  [OK] Train list exists: {list_path}")
        print(f"       → {line_count:,} total training entries")
    else:
        print(f"  [FAIL] Train list missing: {list_path}")
        passed = False

    label_count_after = sum(1 for _ in label_dir.glob("*.txt"))
    if label_count_after == label_count_before:
        print(f"  [OK] Label files unchanged: {label_count_after:,} files")
    else:
        print(
            f"  [WARN] Label file count changed: {label_count_before} → {label_count_after}"
        )
        passed = False

    print(f"\n  {'All checks passed.' if passed else 'Some checks FAILED — see above.'}")
    return passed


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------


def save_report(
    report: dict,
    dataset_root: Path,
    output_list_name: str,
) -> None:
    """Save a JSON report alongside the generated list.

    Args:
        report: Report dictionary to serialise.
        dataset_root: Dataset root directory.
        output_list_name: Base name used to derive the report filename.
    """
    stem = Path(output_list_name).stem
    report_path = dataset_root / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Report JSON: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a class-balanced YOLO train list with optional "
            "hard-class extra oversampling and target-size cap "
            "(data-list level only — no image/label files are modified)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="YOLO dataset root (e.g. data/processed/deepfashion2_yolo_13cls).",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Source YOLO dataset YAML (val path and names block will be reused).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to balance. Default: train.",
    )
    # Stage 1
    parser.add_argument(
        "--balance-power",
        type=float,
        default=1.0,
        help=(
            "Exponent for inverse-frequency class weight. "
            "0.5=sqrt (mild), 1.0=linear (strong). Default: 1.0."
        ),
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=8,
        help="Maximum repeat count per image in Stage 1. Default: 8.",
    )
    # Stage 2 — hard-class boost
    parser.add_argument(
        "--hard-class-ids",
        "--hard-classes",          # backward-compatible alias
        dest="hard_class_ids",
        type=str,
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated fine class IDs to boost (Stage 2). "
            "Example: --hard-class-ids 1,2,3,5,10,11,12. "
            "Omit to skip Stage 2."
        ),
    )
    parser.add_argument(
        "--hard-repeat-bonus",
        "--hard-extra-copies",     # backward-compatible alias
        dest="hard_repeat_bonus",
        type=int,
        default=3,
        metavar="N",
        help="Extra copies added per hard-class image (Stage 2). Default: 3.",
    )
    parser.add_argument(
        "--hard-max-repeat",
        type=int,
        default=12,
        help="Maximum total repeat count after Stage 2 boost. Default: 12.",
    )
    # Stage 3 — target-size cap
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap total training list to approximately N entries (Stage 3). "
            "Scales all repeat counts down proportionally; each image keeps "
            "at least 1 entry. Omit to skip Stage 3."
        ),
    )
    # Output
    parser.add_argument(
        "--output-list-name",
        type=str,
        default="train_balanced_p10_r8.txt",
        help="Output train list filename (inside --dataset-root).",
    )
    parser.add_argument(
        "--out-yaml",
        "--output-yaml-name",      # backward-compatible alias
        dest="out_yaml",
        type=str,
        default="deepfashion2_13cls_balanced_p10_r8.yaml",
        metavar="NAME",
        help="Output YAML filename (inside --dataset-root).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: run all stages, write outputs, verify."""
    args = parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split

    for p, label in [
        (dataset_root, "dataset root"),
        (image_dir, "image dir"),
        (label_dir, "label dir"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")
    if not args.yaml.exists():
        raise FileNotFoundError(f"Source YAML not found: {args.yaml}")

    yaml_text = args.yaml.read_text(encoding="utf-8")
    class_names = _parse_names_from_yaml(yaml_text)

    label_count_before = sum(1 for _ in label_dir.glob("*.txt"))

    hard_classes: set[int] = set()
    if args.hard_class_ids:
        for token in args.hard_class_ids.split(","):
            token = token.strip()
            if token:
                hard_classes.add(int(token))

    print(f"\nScanning {image_dir} ...")
    image_to_classes, image_to_unique, instance_counts, image_counts, empty = scan_dataset(
        image_dir, label_dir
    )
    print(f"  {len(image_to_classes):,} images found, {empty:,} without labels.")

    # Stage 1
    print(
        f"\nStage 1: inverse-frequency balancing "
        f"(power={args.balance_power}, max_repeat={args.max_repeat})"
    )
    base_repeats = compute_base_repeats(
        image_to_unique, image_counts, args.balance_power, args.max_repeat
    )

    # Stage 2 (optional)
    final_repeats = base_repeats
    if hard_classes:
        print(
            f"Stage 2: hard-class boost  classes={sorted(hard_classes)}, "
            f"bonus={args.hard_repeat_bonus}, max={args.hard_max_repeat}"
        )
        final_repeats = apply_hard_class_boost(
            base_repeats,
            image_to_unique,
            hard_classes,
            args.hard_repeat_bonus,
            args.hard_max_repeat,
        )
    else:
        print("Stage 2: skipped (no --hard-class-ids specified).")

    # Stage 3 (optional)
    if args.target_size is not None:
        total_before_cap = sum(final_repeats.values())
        print(
            f"Stage 3: target-size cap  "
            f"target={args.target_size:,}, current={total_before_cap:,}"
        )
        final_repeats = apply_target_size_cap(final_repeats, args.target_size)
        total_after_cap = sum(final_repeats.values())
        print(f"         → scaled to {total_after_cap:,} entries")
    else:
        print("Stage 3: skipped (no --target-size specified).")

    final_eff = compute_effective_distribution(image_to_unique, final_repeats, class_names)

    print_distribution_table(
        "Class distribution — before vs. after balancing",
        image_counts,
        final_eff,
        class_names,
    )

    train_lines = build_train_list(dataset_root, image_dir, final_repeats)
    list_path = dataset_root / args.output_list_name
    yaml_path = dataset_root / args.out_yaml

    print(f"Writing train list → {list_path}")
    write_train_list(train_lines, list_path)

    print(f"Writing dataset YAML → {yaml_path}")
    write_dataset_yaml(yaml_text, list_path, yaml_path)

    repeat_dist = Counter(final_repeats.values())
    report = {
        "balance_power": args.balance_power,
        "max_repeat": args.max_repeat,
        "hard_classes": sorted(hard_classes),
        "hard_repeat_bonus": args.hard_repeat_bonus if hard_classes else None,
        "hard_max_repeat": args.hard_max_repeat if hard_classes else None,
        "target_size": args.target_size,
        "num_source_images": len(image_to_classes),
        "empty_images": empty,
        "num_train_entries": len(train_lines),
        "repeat_distribution": {str(k): v for k, v in sorted(repeat_dist.items())},
        "raw_image_counts": {
            str(k): image_counts[k] for k in sorted(image_counts)
        },
        "effective_image_counts": {
            str(k): final_eff.get(k, 0) for k in sorted(class_names)
        },
        "class_names": {str(k): v for k, v in sorted(class_names.items())},
        "output_list": str(list_path),
        "output_yaml": str(yaml_path),
    }
    save_report(report, dataset_root, args.output_list_name)

    verify_outputs(yaml_path, list_path, label_dir, label_count_before)


if __name__ == "__main__":
    main()
