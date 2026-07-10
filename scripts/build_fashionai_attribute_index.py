#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build JSONL index files for FashionAI attribute classification.

This script converts the original FashionAI attribute answer CSV into clean,
validated, task-specific JSONL dataset indexes.

Current supported behavior:
1. Read FashionAI answer CSV as headerless 3-column file.
2. Parse each row as:
   - image_relative_path
   - task_name
   - label_string
3. Filter rows by target task, e.g. sleeve_length_labels.
4. Validate image existence.
5. Parse y/n vector label.
6. Keep only clean single-positive y/n labels by default.
7. Preserve original FashionAI raw label id.
8. Remap raw label ids to continuous training label ids.
9. Save train/val/test JSONL indexes, label map, invalid samples, and reports.

Example:
    python scripts/build_fashionai_attribute_index.py ^
      --dataset-root D:\Aliintern\fashion-ai-data\fashionai_attributes\round1_fashionAI_attributes_test_a ^
      --answer-csv D:\Aliintern\fashion-ai-data\fashionai_attributes\round1_fashionAI_attributes_test_a\Tests\round1_fashionAI_attributes_answer_a.csv ^
      --task sleeve_length_labels ^
      --output-dir data\fashionai_attribute_index
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple


IGNORED_DIR_NAMES = {
    "__MACOSX",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
}

IGNORED_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

IGNORED_FILE_PREFIXES = (
    "._",
)

VALID_LABEL_CHARS = {"y", "n"}


FASHIONAI_ATTR_VALUES: Dict[str, List[str]] = {
    "skirt_length_labels": [
        "Invisible",
        "Short Length",
        "Knee Length",
        "Midi Length",
        "Ankle Length",
        "Floor Length",
    ],
    "coat_length_labels": [
        "Invisible",
        "High Waist Length",
        "Regular Length",
        "Long Length",
        "Micro Length",
        "Knee Length",
        "Midi Length",
        "Ankle&Floor Length",
    ],
    "collar_design_labels": [
        "Invisible",
        "Shirt Collar",
        "Peter Pan",
        "Puritan Collar",
        "Rib Collar",
    ],
    "lapel_design_labels": [
        "Invisible",
        "Notched",
        "Collarless",
        "Shawl Collar",
        "Plus Size Shawl",
    ],
    "neck_design_labels": [
        "Invisible",
        "Turtle Neck",
        "Ruffle Semi-High Collar",
        "Low Turtle Neck",
        "Draped Collar",
    ],
    "neckline_design_labels": [
        "Invisible",
        "Strapless Neck",
        "Deep V Neckline",
        "Straight Neck",
        "V Neckline",
        "Square Neckline",
        "Off Shoulder",
        "Round Neckline",
        "Sweat Heart Neck",
        "One Shoulder Neckline",
    ],
    "pant_length_labels": [
        "Invisible",
        "Short Pant",
        "Mid Length",
        "3/4 Length",
        "Cropped Pant",
        "Full Length",
    ],
    "sleeve_length_labels": [
        "Invisible",
        "Sleeveless",
        "Cup Sleeves",
        "Short Sleeves",
        "Elbow Sleeves",
        "3/4 Sleeves",
        "Wrist Length",
        "Long Sleeves",
        "Extra Long Sleeves",
    ],
}


@dataclass
class RawAttributeRow:
    """One raw row from FashionAI answer CSV."""

    row_index: int
    image_relative_path: str
    task_name: str
    label_string: str


@dataclass
class IndexedSample:
    """One validated sample for JSONL index."""

    sample_id: str
    image_relative_path: str
    image_path: str
    task: str
    source_task: str
    raw_label: str
    raw_label_id: int
    label_id: int
    label_name: str
    split: str


@dataclass
class InvalidSample:
    """One invalid or skipped row."""

    row_index: int
    image_relative_path: str
    task_name: str
    label_string: str
    reason: str


@dataclass
class BuildStats:
    """Statistics for index building."""

    dataset_root: str
    answer_csv: str
    task: str
    task_short_name: str
    total_csv_rows: int
    task_rows: int
    valid_samples: int
    invalid_samples: int
    missing_images: int
    raw_label_vector_length_distribution: Dict[str, int]
    raw_label_distribution: Dict[str, int]
    raw_label_id_distribution: Dict[str, int]
    label_id_distribution: Dict[str, int]
    split_distribution: Dict[str, int]
    split_label_distribution: Dict[str, Dict[str, int]]
    raw_id_to_label_id: Dict[str, int]
    label_id_to_raw_id: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FashionAI attribute JSONL index."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory of one FashionAI round folder.",
    )
    parser.add_argument(
        "--answer-csv",
        type=str,
        required=True,
        help="Path to FashionAI answer CSV.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sleeve_length_labels",
        help="Target FashionAI task name, e.g. sleeve_length_labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fashionai_attribute_index",
        help="Output directory for generated JSONL and report files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--invalid-report-limit",
        type=int,
        default=5000,
        help="Maximum number of invalid samples saved to invalid JSONL.",
    )
    return parser.parse_args()


def is_ignored_path(path: Path) -> bool:
    """
    Check whether a path should be ignored.

    Args:
        path: File or directory path.

    Returns:
        True if path belongs to metadata or temporary files.
    """
    for part in path.parts:
        if part in IGNORED_DIR_NAMES:
            return True

    name = path.name
    if name in IGNORED_FILE_NAMES:
        return True

    return any(name.startswith(prefix) for prefix in IGNORED_FILE_PREFIXES)


def normalize_relative_path(path_str: str) -> str:
    """
    Normalize relative path from CSV.

    Args:
        path_str: Raw path string.

    Returns:
        POSIX-style relative path.
    """
    return path_str.strip().replace("\\", "/")


def detect_csv_encoding(csv_path: Path) -> str:
    """
    Detect CSV encoding from common encodings.

    Args:
        csv_path: CSV file path.

    Returns:
        Encoding name.

    Raises:
        UnicodeDecodeError: If all encoding attempts fail.
    """
    candidate_encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    last_error: Optional[UnicodeDecodeError] = None

    for encoding in candidate_encodings:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                file.read(4096)
            return encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return "utf-8-sig"


def read_headerless_answer_csv(csv_path: Path) -> List[RawAttributeRow]:
    """
    Read headerless FashionAI answer CSV.

    Args:
        csv_path: Answer CSV path.

    Returns:
        Parsed raw rows.

    Raises:
        ValueError: If any non-empty row has fewer than 3 columns.
    """
    encoding = detect_csv_encoding(csv_path)
    rows: List[RawAttributeRow] = []

    with csv_path.open("r", encoding=encoding, newline="") as file:
        reader = csv.reader(file)

        for row_index, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) < 3:
                raise ValueError(
                    f"CSV row {row_index} has fewer than 3 columns: {row}"
                )

            rows.append(
                RawAttributeRow(
                    row_index=row_index,
                    image_relative_path=normalize_relative_path(row[0]),
                    task_name=row[1].strip(),
                    label_string=row[2].strip().lower(),
                )
            )

    return rows


def parse_single_positive_raw_label_id(
    label_string: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse a clean single-positive y/n label vector.

    Args:
        label_string: Raw label vector, e.g. "nnynnnnnn".

    Returns:
        A tuple of raw label id and invalid reason.
        If valid, returns (raw_label_id, None).
        If invalid, returns (None, reason).
    """
    if not label_string:
        return None, "empty_label"

    chars = set(label_string)
    if not chars.issubset(VALID_LABEL_CHARS):
        return None, "invalid_label_chars"

    positive_count = label_string.count("y")
    if positive_count == 0:
        return None, "no_positive_label"

    if positive_count > 1:
        return None, "multiple_positive_labels"

    return label_string.index("y"), None


def strip_labels_suffix(task: str) -> str:
    """
    Convert FashionAI source task name to short task name.

    Args:
        task: Source task name, e.g. sleeve_length_labels.

    Returns:
        Short task name, e.g. sleeve_length.
    """
    suffix = "_labels"
    if task.endswith(suffix):
        return task[: -len(suffix)]
    return task


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """
    Validate train/val/test split ratios.

    Args:
        train_ratio: Train ratio.
        val_ratio: Validation ratio.
        test_ratio: Test ratio.

    Raises:
        ValueError: If ratios are invalid.
    """
    ratios = [train_ratio, val_ratio, test_ratio]

    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {total:.6f}."
        )


def build_contiguous_label_mapping(
    raw_label_ids: Sequence[int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build mapping from raw FashionAI label ids to continuous training label ids.

    Args:
        raw_label_ids: Raw label ids observed in valid samples.

    Returns:
        raw_id_to_label_id and label_id_to_raw_id.
    """
    unique_raw_ids = sorted(set(raw_label_ids))
    raw_id_to_label_id = {
        raw_id: label_id for label_id, raw_id in enumerate(unique_raw_ids)
    }
    label_id_to_raw_id = {
        label_id: raw_id for raw_id, label_id in raw_id_to_label_id.items()
    }

    return raw_id_to_label_id, label_id_to_raw_id


def get_label_name(task: str, raw_label_id: int) -> str:
    """
    Get semantic label name from official FashionAI attr values.

    Args:
        task: Source task name.
        raw_label_id: Raw FashionAI label position.

    Returns:
        Human-readable label name if known, otherwise class_{raw_label_id}.
    """
    attr_values = FASHIONAI_ATTR_VALUES.get(task)
    if attr_values is None:
        return f"class_{raw_label_id}"

    if raw_label_id < 0 or raw_label_id >= len(attr_values):
        return f"class_{raw_label_id}"

    return attr_values[raw_label_id]


def stratified_split(
    samples: Sequence[IndexedSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[IndexedSample], List[IndexedSample], List[IndexedSample], List[str]]:
    """
    Split samples into train/val/test in a label-aware way.

    Args:
        samples: Valid indexed samples.
        train_ratio: Train ratio.
        val_ratio: Validation ratio.
        test_ratio: Test ratio.
        seed: Random seed.

    Returns:
        Train, validation, test samples and warnings.
    """
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    rng = random.Random(seed)
    warnings: List[str] = []

    label_to_samples: DefaultDict[int, List[IndexedSample]] = defaultdict(list)
    for sample in samples:
        label_to_samples[sample.label_id].append(sample)

    train_samples: List[IndexedSample] = []
    val_samples: List[IndexedSample] = []
    test_samples: List[IndexedSample] = []

    for label_id, class_samples in sorted(label_to_samples.items()):
        class_samples = list(class_samples)
        rng.shuffle(class_samples)

        num_samples = len(class_samples)
        if num_samples < 3:
            warnings.append(
                f"Label {label_id} has only {num_samples} samples; "
                "all assigned to train split."
            )
            for sample in class_samples:
                sample.split = "train"
            train_samples.extend(class_samples)
            continue

        num_train = int(round(num_samples * train_ratio))
        num_val = int(round(num_samples * val_ratio))

        if val_ratio > 0 and num_val == 0:
            num_val = 1

        if test_ratio > 0 and num_samples - num_train - num_val <= 0:
            if num_train > 1:
                num_train -= 1

        num_test = max(0, num_samples - num_train - num_val)

        train_part = class_samples[:num_train]
        val_part = class_samples[num_train:num_train + num_val]
        test_part = class_samples[num_train + num_val:num_train + num_val + num_test]

        for sample in train_part:
            sample.split = "train"
        for sample in val_part:
            sample.split = "val"
        for sample in test_part:
            sample.split = "test"

        train_samples.extend(train_part)
        val_samples.extend(val_part)
        test_samples.extend(test_part)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples, warnings


def write_jsonl(samples: Sequence[object], output_path: Path) -> None:
    """
    Write dataclass objects or dictionaries to JSONL.

    Args:
        samples: Sequence of dataclass objects or dictionaries.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            if hasattr(sample, "__dataclass_fields__"):
                payload = asdict(sample)
            else:
                payload = sample
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(data: object, output_path: Path) -> None:
    """
    Write object as JSON.

    Args:
        data: JSON-serializable object.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def count_split_distribution(samples: Sequence[IndexedSample]) -> Dict[str, int]:
    """
    Count samples per split.

    Args:
        samples: Indexed samples.

    Returns:
        Split distribution.
    """
    return dict(Counter(sample.split for sample in samples))


def count_split_label_distribution(
    samples: Sequence[IndexedSample],
) -> Dict[str, Dict[str, int]]:
    """
    Count label distribution within each split.

    Args:
        samples: Indexed samples.

    Returns:
        Nested split-label distribution.
    """
    result: DefaultDict[str, Counter[str]] = defaultdict(Counter)

    for sample in samples:
        result[sample.split][str(sample.label_id)] += 1

    return {
        split: dict(counter)
        for split, counter in sorted(result.items())
    }


def build_label_map(
    samples: Sequence[IndexedSample],
    source_task: str,
    task_short_name: str,
    raw_id_to_label_id: Dict[int, int],
    label_id_to_raw_id: Dict[int, int],
) -> Dict[str, object]:
    """
    Build label map for training and inference.

    Args:
        samples: Valid samples.
        source_task: FashionAI source task name.
        task_short_name: Short task name.
        raw_id_to_label_id: Mapping from raw ids to continuous label ids.
        label_id_to_raw_id: Mapping from continuous label ids to raw ids.

    Returns:
        Label map dictionary.
    """
    id_to_label: Dict[str, str] = {}
    label_to_id: Dict[str, int] = {}

    for label_id in sorted(label_id_to_raw_id):
        raw_id = label_id_to_raw_id[label_id]
        label_name = get_label_name(source_task, raw_id)
        id_to_label[str(label_id)] = label_name
        label_to_id[label_name] = label_id

    raw_attr_values = {
        str(index): value
        for index, value in enumerate(FASHIONAI_ATTR_VALUES.get(source_task, []))
    }

    observed_raw_label_ids = sorted({sample.raw_label_id for sample in samples})

    return {
        "task": task_short_name,
        "source_task": source_task,
        "num_classes": len(id_to_label),
        "raw_attr_values": raw_attr_values,
        "observed_raw_label_ids": observed_raw_label_ids,
        "raw_id_to_label_id": {
            str(raw_id): label_id
            for raw_id, label_id in sorted(raw_id_to_label_id.items())
        },
        "label_id_to_raw_id": {
            str(label_id): raw_id
            for label_id, raw_id in sorted(label_id_to_raw_id.items())
        },
        "id_to_label": id_to_label,
        "label_to_id": label_to_id,
        "note": (
            "raw_label_id is the original FashionAI y-position. "
            "label_id is the contiguous training target id."
        ),
    }


def build_index(
    args: argparse.Namespace,
) -> Tuple[List[IndexedSample], List[InvalidSample], BuildStats, Dict[str, object]]:
    """
    Build task-specific indexes.

    Args:
        args: Parsed arguments.

    Returns:
        Valid indexed samples, invalid samples, stats, and label map.
    """
    dataset_root = Path(args.dataset_root).resolve()
    answer_csv = Path(args.answer_csv).resolve()
    source_task = args.task.strip()
    task_short_name = strip_labels_suffix(source_task)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    if not answer_csv.exists():
        raise FileNotFoundError(f"Answer CSV does not exist: {answer_csv}")

    raw_rows = read_headerless_answer_csv(answer_csv)
    task_rows = [row for row in raw_rows if row.task_name == source_task]

    invalid_samples: List[InvalidSample] = []
    pre_samples: List[Tuple[RawAttributeRow, Path, int]] = []

    raw_label_vector_length_counter: Counter[str] = Counter()
    raw_label_counter: Counter[str] = Counter()
    raw_label_id_counter: Counter[str] = Counter()
    missing_images_count = 0

    for row in task_rows:
        raw_label_vector_length_counter[str(len(row.label_string))] += 1
        raw_label_counter[row.label_string] += 1

        image_path = dataset_root / row.image_relative_path

        if is_ignored_path(image_path):
            invalid_samples.append(
                InvalidSample(
                    row_index=row.row_index,
                    image_relative_path=row.image_relative_path,
                    task_name=row.task_name,
                    label_string=row.label_string,
                    reason="ignored_metadata_path",
                )
            )
            continue

        if not image_path.exists():
            missing_images_count += 1
            invalid_samples.append(
                InvalidSample(
                    row_index=row.row_index,
                    image_relative_path=row.image_relative_path,
                    task_name=row.task_name,
                    label_string=row.label_string,
                    reason="missing_image",
                )
            )
            continue

        raw_label_id, invalid_reason = parse_single_positive_raw_label_id(
            row.label_string
        )

        if invalid_reason is not None or raw_label_id is None:
            invalid_samples.append(
                InvalidSample(
                    row_index=row.row_index,
                    image_relative_path=row.image_relative_path,
                    task_name=row.task_name,
                    label_string=row.label_string,
                    reason=invalid_reason or "unknown_label_parse_error",
                )
            )
            continue

        raw_label_id_counter[str(raw_label_id)] += 1
        pre_samples.append((row, image_path, raw_label_id))

    raw_label_ids = [item[2] for item in pre_samples]
    raw_id_to_label_id, label_id_to_raw_id = build_contiguous_label_mapping(
        raw_label_ids
    )

    valid_samples: List[IndexedSample] = []
    label_id_counter: Counter[str] = Counter()

    for sample_index, (row, image_path, raw_label_id) in enumerate(pre_samples):
        label_id = raw_id_to_label_id[raw_label_id]
        label_name = get_label_name(source_task, raw_label_id)
        label_id_counter[str(label_id)] += 1

        valid_samples.append(
            IndexedSample(
                sample_id=f"{task_short_name}_{sample_index:06d}",
                image_relative_path=row.image_relative_path,
                image_path=image_path.as_posix(),
                task=task_short_name,
                source_task=source_task,
                raw_label=row.label_string,
                raw_label_id=raw_label_id,
                label_id=label_id,
                label_name=label_name,
                split="",
            )
        )

    train_samples, val_samples, test_samples, split_warnings = stratified_split(
        samples=valid_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    all_split_samples = train_samples + val_samples + test_samples

    label_map = build_label_map(
        samples=all_split_samples,
        source_task=source_task,
        task_short_name=task_short_name,
        raw_id_to_label_id=raw_id_to_label_id,
        label_id_to_raw_id=label_id_to_raw_id,
    )

    stats = BuildStats(
        dataset_root=dataset_root.as_posix(),
        answer_csv=answer_csv.as_posix(),
        task=source_task,
        task_short_name=task_short_name,
        total_csv_rows=len(raw_rows),
        task_rows=len(task_rows),
        valid_samples=len(valid_samples),
        invalid_samples=len(invalid_samples),
        missing_images=missing_images_count,
        raw_label_vector_length_distribution=dict(
            sorted(raw_label_vector_length_counter.items())
        ),
        raw_label_distribution=dict(raw_label_counter.most_common()),
        raw_label_id_distribution=dict(sorted(raw_label_id_counter.items())),
        label_id_distribution=dict(sorted(label_id_counter.items())),
        split_distribution=count_split_distribution(all_split_samples),
        split_label_distribution=count_split_label_distribution(all_split_samples),
        raw_id_to_label_id={
            str(raw_id): label_id
            for raw_id, label_id in sorted(raw_id_to_label_id.items())
        },
        label_id_to_raw_id={
            str(label_id): raw_id
            for label_id, raw_id in sorted(label_id_to_raw_id.items())
        },
        warnings=split_warnings,
    )

    return all_split_samples, invalid_samples, stats, label_map


def write_markdown_summary(
    stats: BuildStats,
    label_map: Dict[str, object],
    output_path: Path,
) -> None:
    """
    Write human-readable build summary.

    Args:
        stats: Build statistics.
        label_map: Label mapping dictionary.
        output_path: Markdown output path.
    """
    lines: List[str] = []

    lines.append("# FashionAI Attribute Index Build Report")
    lines.append("")
    lines.append("## 1. Basic Information")
    lines.append("")
    lines.append(f"- Dataset root: `{stats.dataset_root}`")
    lines.append(f"- Answer CSV: `{stats.answer_csv}`")
    lines.append(f"- Task: `{stats.task}`")
    lines.append(f"- Short task name: `{stats.task_short_name}`")
    lines.append("")
    lines.append("## 2. Row Statistics")
    lines.append("")
    lines.append(f"- Total CSV rows: `{stats.total_csv_rows}`")
    lines.append(f"- Task rows: `{stats.task_rows}`")
    lines.append(f"- Valid samples: `{stats.valid_samples}`")
    lines.append(f"- Invalid samples: `{stats.invalid_samples}`")
    lines.append(f"- Missing images: `{stats.missing_images}`")
    lines.append("")
    lines.append("## 3. Raw Label Vector Length Distribution")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            stats.raw_label_vector_length_distribution,
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("## 4. Raw Label ID Distribution")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(stats.raw_label_id_distribution, ensure_ascii=False, indent=2)
    )
    lines.append("```")
    lines.append("")
    lines.append("## 5. Training Label ID Distribution")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats.label_id_distribution, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 6. Raw ID to Training Label ID Mapping")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats.raw_id_to_label_id, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 7. Label Map")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(label_map, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 8. Split Distribution")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats.split_distribution, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 9. Split Label Distribution")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(stats.split_label_distribution, ensure_ascii=False, indent=2)
    )
    lines.append("```")
    lines.append("")
    lines.append("## 10. Raw Label Distribution")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats.raw_label_distribution, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    if stats.warnings:
        lines.append("## 11. Warnings")
        lines.append("")
        for warning in stats.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## 12. Notes")
    lines.append("")
    lines.append("- This report contains dataset index statistics only.")
    lines.append("- No model training accuracy or F1 score is reported here.")
    lines.append("- `raw_label_id` is the original FashionAI y-position.")
    lines.append("- `label_id` is the contiguous training target id.")
    lines.append("- Samples containing labels outside clean y/n single-positive format are excluded.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run index building."""
    args = parse_args()

    try:
        samples, invalid_samples, stats, label_map = build_index(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_short_name = strip_labels_suffix(args.task)

    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    test_samples = [sample for sample in samples if sample.split == "test"]

    all_jsonl = output_dir / f"{task_short_name}_all.jsonl"
    train_jsonl = output_dir / f"{task_short_name}_train.jsonl"
    val_jsonl = output_dir / f"{task_short_name}_val.jsonl"
    test_jsonl = output_dir / f"{task_short_name}_test.jsonl"
    label_map_json = output_dir / f"label_map_{task_short_name}.json"
    stats_json = output_dir / f"stats_{task_short_name}.json"
    invalid_jsonl = output_dir / f"invalid_{task_short_name}.jsonl"
    report_md = output_dir / f"build_report_{task_short_name}.md"

    write_jsonl(samples, all_jsonl)
    write_jsonl(train_samples, train_jsonl)
    write_jsonl(val_samples, val_jsonl)
    write_jsonl(test_samples, test_jsonl)
    write_json(label_map, label_map_json)
    write_json(asdict(stats), stats_json)

    invalid_limit = max(0, int(args.invalid_report_limit))
    write_jsonl(invalid_samples[:invalid_limit], invalid_jsonl)

    write_markdown_summary(stats, label_map, report_md)

    print("[OK] FashionAI attribute index build completed.")
    print(f"[OK] Task: {args.task}")
    print(f"[OK] All JSONL: {all_jsonl}")
    print(f"[OK] Train JSONL: {train_jsonl}")
    print(f"[OK] Val JSONL: {val_jsonl}")
    print(f"[OK] Test JSONL: {test_jsonl}")
    print(f"[OK] Label map: {label_map_json}")
    print(f"[OK] Stats: {stats_json}")
    print(f"[OK] Invalid report: {invalid_jsonl}")
    print(f"[OK] Markdown report: {report_md}")

    print("")
    print("Summary:")
    print(f"  Total CSV rows: {stats.total_csv_rows}")
    print(f"  Task rows: {stats.task_rows}")
    print(f"  Valid samples: {stats.valid_samples}")
    print(f"  Invalid samples: {stats.invalid_samples}")
    print(f"  Missing images: {stats.missing_images}")
    print(f"  Raw label ID distribution: {stats.raw_label_id_distribution}")
    print(f"  Training label ID distribution: {stats.label_id_distribution}")
    print(f"  Raw ID to label ID: {stats.raw_id_to_label_id}")
    print(f"  Split distribution: {stats.split_distribution}")

    if stats.warnings:
        print("")
        print("Warnings:")
        for warning in stats.warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
