#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan FashionAI attribute dataset structure.

This script performs a safe, read-only scan over the FashionAI attribute dataset.
It does not train models, does not modify source data, and does not assume a fixed
CSV schema.

Main functions:
1. Ignore macOS metadata files and directories.
2. Discover CSV / JSON / TXT / Markdown files.
3. Inspect CSV schemas, row counts, and sample rows.
4. Discover attribute image folders under Images/.
5. Count image files for each attribute task.
6. Try to infer possible image columns and label columns conservatively.
7. Generate machine-readable JSON report and human-readable Markdown report.

Example:
    python scripts/scan_fashionai_attributes.py ^
      --data-root D:\Aliintern\fashion-ai-data\fashionai_attributes ^
      --output-dir outputs\p2_fashionai_scan
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IGNORED_DIR_NAMES = {
    "__MACOSX",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
}

IGNORED_FILE_PREFIXES = (
    "._",
)

IGNORED_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".md",
}


@dataclass
class CsvColumnProfile:
    """Profile of a single CSV column."""

    name: str
    non_empty_count: int
    empty_count: int
    unique_count_in_sample: int
    top_values_in_sample: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class CsvFileReport:
    """Summary report for one CSV file."""

    path: str
    relative_path: str
    exists: bool
    readable: bool
    encoding: Optional[str]
    num_rows: Optional[int]
    num_columns: Optional[int]
    columns: List[str] = field(default_factory=list)
    possible_image_columns: List[str] = field(default_factory=list)
    possible_label_columns: List[str] = field(default_factory=list)
    column_profiles: List[CsvColumnProfile] = field(default_factory=list)
    sample_rows: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ImageFolderReport:
    """Summary report for one image folder."""

    path: str
    relative_path: str
    folder_name: str
    num_images: int
    extension_distribution: Dict[str, int]
    sample_images: List[str] = field(default_factory=list)


@dataclass
class DatasetScanReport:
    """Full scan report for the FashionAI attribute dataset."""

    data_root: str
    output_dir: str
    csv_files: List[CsvFileReport]
    text_files: List[str]
    image_folders: List[ImageFolderReport]
    attribute_image_folders: List[ImageFolderReport]
    warnings: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan FashionAI attribute dataset safely."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root path of fashionai_attributes directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/p2_fashionai_scan",
        help="Directory to save scan reports.",
    )
    parser.add_argument(
        "--csv-sample-rows",
        type=int,
        default=20,
        help="Number of sample rows to store for each CSV.",
    )
    parser.add_argument(
        "--profile-max-rows",
        type=int,
        default=50000,
        help="Maximum rows used for column value profiling.",
    )
    parser.add_argument(
        "--image-sample-count",
        type=int,
        default=10,
        help="Number of sample image paths to store for each image folder.",
    )
    return parser.parse_args()


def is_ignored_path(path: Path) -> bool:
    """
    Return whether a path should be ignored.

    Args:
        path: File or directory path.

    Returns:
        True if the path is macOS metadata, IDE metadata, cache, or hidden resource.
    """
    for part in path.parts:
        if part in IGNORED_DIR_NAMES:
            return True

    name = path.name
    if name in IGNORED_FILE_NAMES:
        return True

    return any(name.startswith(prefix) for prefix in IGNORED_FILE_PREFIXES)


def iter_files(root: Path) -> Iterable[Path]:
    """
    Recursively iterate files under root while ignoring metadata paths.

    Args:
        root: Root directory.

    Yields:
        Valid file paths.
    """
    for path in root.rglob("*"):
        if is_ignored_path(path):
            continue
        if path.is_file():
            yield path


def safe_relative_path(path: Path, root: Path) -> str:
    """Return path relative to root using POSIX-style separators."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def detect_csv_encoding(csv_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Try several common encodings for CSV files.

    Args:
        csv_path: CSV file path.

    Returns:
        A tuple of detected encoding and error message.
    """
    candidate_encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]

    for encoding in candidate_encodings:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                file.read(4096)
            return encoding, None
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            return None, str(exc)

    return None, "Failed to decode CSV with utf-8-sig, utf-8, gbk, or latin1."


def looks_like_image_column(column_name: str) -> bool:
    """Heuristically determine whether a CSV column may contain image names."""
    name = column_name.lower()
    keywords = ["image", "img", "file", "filename", "filepath", "path", "id"]
    return any(keyword in name for keyword in keywords)


def looks_like_label_column(column_name: str) -> bool:
    """Heuristically determine whether a CSV column may contain labels."""
    name = column_name.lower()
    keywords = ["label", "attribute", "answer", "class", "category"]
    return any(keyword in name for keyword in keywords)


def read_csv_report(
    csv_path: Path,
    root: Path,
    sample_rows_limit: int,
    profile_max_rows: int,
) -> CsvFileReport:
    """
    Read and summarize one CSV file safely.

    Args:
        csv_path: Path to CSV file.
        root: Dataset root.
        sample_rows_limit: Number of sample rows to store.
        profile_max_rows: Maximum number of rows used for profiling.

    Returns:
        CsvFileReport.
    """
    encoding, encoding_error = detect_csv_encoding(csv_path)

    report = CsvFileReport(
        path=csv_path.as_posix(),
        relative_path=safe_relative_path(csv_path, root),
        exists=csv_path.exists(),
        readable=False,
        encoding=encoding,
        num_rows=None,
        num_columns=None,
    )

    if encoding is None:
        report.error = encoding_error
        return report

    try:
        with csv_path.open("r", encoding=encoding, newline="") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames or []
            report.columns = columns
            report.num_columns = len(columns)
            report.possible_image_columns = [
                column for column in columns if looks_like_image_column(column)
            ]
            report.possible_label_columns = [
                column for column in columns if looks_like_label_column(column)
            ]

            value_counters: Dict[str, Counter[str]] = {
                column: Counter() for column in columns
            }
            non_empty_counts: Dict[str, int] = {column: 0 for column in columns}
            empty_counts: Dict[str, int] = {column: 0 for column in columns}

            num_rows = 0
            sample_rows: List[Dict[str, str]] = []

            for row in reader:
                num_rows += 1

                if len(sample_rows) < sample_rows_limit:
                    sample_rows.append({key: row.get(key, "") for key in columns})

                if num_rows <= profile_max_rows:
                    for column in columns:
                        value = str(row.get(column, "")).strip()
                        if value:
                            non_empty_counts[column] += 1
                            value_counters[column][value] += 1
                        else:
                            empty_counts[column] += 1

            report.num_rows = num_rows
            report.sample_rows = sample_rows
            report.column_profiles = [
                CsvColumnProfile(
                    name=column,
                    non_empty_count=non_empty_counts[column],
                    empty_count=empty_counts[column],
                    unique_count_in_sample=len(value_counters[column]),
                    top_values_in_sample=value_counters[column].most_common(20),
                )
                for column in columns
            ]
            report.readable = True

    except Exception as exc:  # noqa: BLE001
        report.error = f"{type(exc).__name__}: {exc}"

    return report


def collect_text_files(root: Path) -> List[str]:
    """
    Collect text-like files such as CSV, JSON, TXT, and Markdown.

    Args:
        root: Dataset root.

    Returns:
        Relative paths of text files.
    """
    text_files = []
    for file_path in iter_files(root):
        if file_path.suffix.lower() in TEXT_EXTENSIONS:
            text_files.append(safe_relative_path(file_path, root))
    return sorted(text_files)


def collect_image_files(folder: Path) -> List[Path]:
    """
    Collect image files directly under a folder, non-recursively.

    Args:
        folder: Image folder.

    Returns:
        Image file paths.
    """
    if not folder.exists() or not folder.is_dir():
        return []

    image_files = []
    for path in folder.iterdir():
        if is_ignored_path(path):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(path)
    return sorted(image_files)


def scan_image_folders(
    root: Path,
    image_sample_count: int,
) -> Tuple[List[ImageFolderReport], List[ImageFolderReport]]:
    """
    Scan all folders that contain images.

    Args:
        root: Dataset root.
        image_sample_count: Number of sample image paths.

    Returns:
        A tuple of all image folder reports and attribute image folder reports.
    """
    all_image_folders: List[ImageFolderReport] = []
    attribute_image_folders: List[ImageFolderReport] = []

    for folder in root.rglob("*"):
        if is_ignored_path(folder):
            continue
        if not folder.is_dir():
            continue

        image_files = collect_image_files(folder)
        if not image_files:
            continue

        ext_counter = Counter(path.suffix.lower() for path in image_files)

        report = ImageFolderReport(
            path=folder.as_posix(),
            relative_path=safe_relative_path(folder, root),
            folder_name=folder.name,
            num_images=len(image_files),
            extension_distribution=dict(sorted(ext_counter.items())),
            sample_images=[
                safe_relative_path(path, root)
                for path in image_files[:image_sample_count]
            ],
        )
        all_image_folders.append(report)

        if folder.name.endswith("_labels"):
            attribute_image_folders.append(report)

    all_image_folders.sort(key=lambda item: item.relative_path)
    attribute_image_folders.sort(key=lambda item: item.relative_path)

    return all_image_folders, attribute_image_folders


def write_json_report(report: DatasetScanReport, output_path: Path) -> None:
    """
    Write scan report as JSON.

    Args:
        report: DatasetScanReport instance.
        output_path: Output JSON path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(report), file, ensure_ascii=False, indent=2)


def write_markdown_report(report: DatasetScanReport, output_path: Path) -> None:
    """
    Write scan report as Markdown.

    Args:
        report: DatasetScanReport instance.
        output_path: Output Markdown path.
    """
    lines: List[str] = []

    lines.append("# FashionAI Attribute Dataset Scan Report")
    lines.append("")
    lines.append("## 1. Basic Information")
    lines.append("")
    lines.append(f"- Data root: `{report.data_root}`")
    lines.append(f"- Output dir: `{report.output_dir}`")
    lines.append(f"- CSV files found: `{len(report.csv_files)}`")
    lines.append(f"- Text-like files found: `{len(report.text_files)}`")
    lines.append(f"- Image folders found: `{len(report.image_folders)}`")
    lines.append(f"- Attribute image folders found: `{len(report.attribute_image_folders)}`")
    lines.append("")

    if report.warnings:
        lines.append("## 2. Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## 3. Attribute Image Folders")
    lines.append("")
    if report.attribute_image_folders:
        lines.append("| Folder | Images | Extensions |")
        lines.append("|---|---:|---|")
        for folder in report.attribute_image_folders:
            lines.append(
                f"| `{folder.relative_path}` | {folder.num_images} | "
                f"`{json.dumps(folder.extension_distribution, ensure_ascii=False)}` |"
            )
    else:
        lines.append("No attribute image folders found.")
    lines.append("")

    lines.append("## 4. CSV Files")
    lines.append("")
    if report.csv_files:
        for csv_report in report.csv_files:
            lines.append(f"### `{csv_report.relative_path}`")
            lines.append("")
            lines.append(f"- Readable: `{csv_report.readable}`")
            lines.append(f"- Encoding: `{csv_report.encoding}`")
            lines.append(f"- Rows: `{csv_report.num_rows}`")
            lines.append(f"- Columns: `{csv_report.num_columns}`")
            if csv_report.error:
                lines.append(f"- Error: `{csv_report.error}`")

            lines.append("")
            lines.append("Columns:")
            lines.append("")
            for column in csv_report.columns:
                lines.append(f"- `{column}`")
            lines.append("")

            lines.append("Possible image columns:")
            lines.append("")
            if csv_report.possible_image_columns:
                for column in csv_report.possible_image_columns:
                    lines.append(f"- `{column}`")
            else:
                lines.append("- None detected")
            lines.append("")

            lines.append("Possible label columns:")
            lines.append("")
            if csv_report.possible_label_columns:
                for column in csv_report.possible_label_columns:
                    lines.append(f"- `{column}`")
            else:
                lines.append("- None detected")
            lines.append("")

            if csv_report.column_profiles:
                lines.append("Column profiles based on sampled rows:")
                lines.append("")
                lines.append("| Column | Non-empty | Empty | Unique in sample | Top values in sample |")
                lines.append("|---|---:|---:|---:|---|")
                for profile in csv_report.column_profiles:
                    top_values = json.dumps(
                        profile.top_values_in_sample[:10],
                        ensure_ascii=False,
                    )
                    lines.append(
                        f"| `{profile.name}` | {profile.non_empty_count} | "
                        f"{profile.empty_count} | {profile.unique_count_in_sample} | "
                        f"`{top_values}` |"
                    )
                lines.append("")

            if csv_report.sample_rows:
                lines.append("Sample rows:")
                lines.append("")
                lines.append("```json")
                lines.append(
                    json.dumps(
                        csv_report.sample_rows[:5],
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                lines.append("```")
                lines.append("")
    else:
        lines.append("No CSV files found.")
        lines.append("")

    lines.append("## 5. All Text-like Files")
    lines.append("")
    for text_file in report.text_files:
        lines.append(f"- `{text_file}`")
    lines.append("")

    lines.append("## 6. Notes")
    lines.append("")
    lines.append("- This report is generated from local filesystem scanning.")
    lines.append("- No accuracy, F1, or model result is included in this scan.")
    lines.append("- `__MACOSX`, `.DS_Store`, and `._*` metadata files are ignored.")
    lines.append("- The script does not assume which CSV is the ground-truth label file.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> DatasetScanReport:
    """
    Build full dataset scan report.

    Args:
        args: Parsed command line arguments.

    Returns:
        DatasetScanReport.
    """
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    warnings: List[str] = []

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    text_files = collect_text_files(data_root)

    csv_paths = [
        data_root / relative_path
        for relative_path in text_files
        if Path(relative_path).suffix.lower() == ".csv"
    ]

    csv_reports = [
        read_csv_report(
            csv_path=csv_path,
            root=data_root,
            sample_rows_limit=args.csv_sample_rows,
            profile_max_rows=args.profile_max_rows,
        )
        for csv_path in csv_paths
    ]

    image_folders, attribute_image_folders = scan_image_folders(
        root=data_root,
        image_sample_count=args.image_sample_count,
    )

    if not csv_reports:
        warnings.append("No CSV files were found after filtering ignored paths.")

    if not attribute_image_folders:
        warnings.append("No attribute image folders ending with '_labels' were found.")

    sleeve_folders = [
        folder for folder in attribute_image_folders
        if folder.folder_name == "sleeve_length_labels"
    ]
    if not sleeve_folders:
        warnings.append("No sleeve_length_labels image folder was found.")

    return DatasetScanReport(
        data_root=data_root.as_posix(),
        output_dir=output_dir.as_posix(),
        csv_files=csv_reports,
        text_files=text_files,
        image_folders=image_folders,
        attribute_image_folders=attribute_image_folders,
        warnings=warnings,
    )


def main() -> int:
    """Run dataset scan."""
    args = parse_args()

    try:
        report = build_report(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "fashionai_attribute_scan_report.json"
    markdown_path = output_dir / "fashionai_attribute_scan_report.md"

    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path)

    print("[OK] FashionAI attribute dataset scan completed.")
    print(f"[OK] JSON report: {json_path}")
    print(f"[OK] Markdown report: {markdown_path}")

    print("")
    print("Summary:")
    print(f"  CSV files: {len(report.csv_files)}")
    print(f"  Text-like files: {len(report.text_files)}")
    print(f"  Image folders: {len(report.image_folders)}")
    print(f"  Attribute image folders: {len(report.attribute_image_folders)}")

    sleeve_folders = [
        folder for folder in report.attribute_image_folders
        if folder.folder_name == "sleeve_length_labels"
    ]
    for folder in sleeve_folders:
        print(f"  sleeve_length_labels images: {folder.num_images}")

    if report.warnings:
        print("")
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
