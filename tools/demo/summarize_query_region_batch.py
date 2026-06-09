from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize query_region_online_demo batch outputs. "
            "This script scans result.json files and generates CSV/JSON summaries."
        )
    )

    parser.add_argument(
        "--batch-dir",
        type=str,
        required=True,
        help="Batch output directory, e.g. outputs/query_region_online_demo_batch60.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Output CSV path. Default: <batch-dir>/batch_results.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output summary JSON path. Default: <batch-dir>/query_success_summary.json.",
    )
    parser.add_argument(
        "--failed-csv",
        type=str,
        default="",
        help="Output failed cases CSV path. Default: <batch-dir>/failed_cases.csv.",
    )
    parser.add_argument(
        "--include-pipeline-result",
        action="store_true",
        help=(
            "Include pipeline_result field in output rows. "
            "Disabled by default because it can make CSV too large."
        ),
    )
    parser.add_argument(
        "--dedupe-latest",
        action="store_true",
        help=(
            "Deduplicate rows and keep the latest result.json according to file "
            "modification time. Default dedupe key is image + query."
        ),
    )
    parser.add_argument(
        "--dedupe-key",
        type=str,
        default="image_query",
        choices=[
            "image_query",
            "image_query_target",
            "image_query_target_component",
            "image_query_target_component_class",
        ],
        help=(
            "Deduplication key. Only used when --dedupe-latest is enabled. "
            "Default: image_query."
        ),
    )

    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def json_dumps_compact(value: Any) -> str:
    """Convert a value to a compact JSON string for CSV storage."""
    if value is None:
        return ""

    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def find_result_json_files(batch_dir: Path) -> list[Path]:
    """Find all result.json files under a batch directory."""
    result_files: list[Path] = []

    for path in batch_dir.rglob("result.json"):
        if path.is_file():
            result_files.append(path)

    result_files.sort(key=lambda x: str(x))

    return result_files


def normalize_query(query: Any) -> str:
    """Normalize query value."""
    return str(query or "").strip()


def normalize_status(status: Any) -> str:
    """Normalize status value."""
    value = str(status or "").strip().lower()

    if value in {"success", "failed"}:
        return value

    return value or "unknown"


def path_mtime(path: str | Path) -> float:
    """Return file modification time. Return 0.0 if unavailable."""
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0


def mtime_to_iso(mtime: float) -> str:
    """Convert timestamp to readable local datetime string."""
    if mtime <= 0:
        return ""

    try:
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def extract_row(
    result: dict[str, Any],
    result_json_path: Path,
    batch_dir: Path,
    include_pipeline_result: bool = False,
) -> dict[str, Any]:
    """Extract one flat CSV row from a result.json object."""
    status = normalize_status(result.get("status"))
    query = normalize_query(result.get("query"))

    selected = result.get("selected")
    if not isinstance(selected, dict):
        selected = {}

    selection = result.get("selection")
    if not isinstance(selection, dict):
        selection = {}

    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}

    query_debug = result.get("query_debug")
    if not isinstance(query_debug, dict):
        query_debug = {}

    pipeline_result = result.get("pipeline_result")
    if not isinstance(pipeline_result, dict):
        pipeline_result = {}

    pipeline_timing = pipeline_result.get("timing")
    if not isinstance(pipeline_timing, dict):
        pipeline_timing = {}

    image_path = str(result.get("image", ""))
    image_name = Path(image_path).name if image_path else ""

    run_dir = str(outputs.get("run_dir", "") or "")
    run_dir_name = Path(run_dir).name if run_dir else ""

    overlay_path = outputs.get("region_overlay", "")
    result_json_str = str(result_json_path)

    try:
        relative_result_json = str(result_json_path.relative_to(batch_dir))
    except Exception:
        relative_result_json = result_json_str

    result_mtime = path_mtime(result_json_path)

    row: dict[str, Any] = {
        "status": status,
        "error": result.get("error", ""),
        "image": image_path,
        "image_name": image_name,
        "query": query,
        "target_region": result.get("target_region", ""),
        "target_region_display": result.get("target_region_display", ""),
        "target_component": result.get("target_component", ""),
        "target_class": result.get("target_class", ""),
        "matched_region_alias": query_debug.get("matched_region_alias", ""),
        "matched_component_alias": query_debug.get("matched_component_alias", ""),
        "matched_special_alias": query_debug.get("matched_special_alias", ""),
        "inferred_target_class": query_debug.get("inferred_target_class", ""),
        "selection_rule": selection.get("rule", ""),
        "selection_reason": selection.get("reason", ""),
        "selection_rank": selection.get("rank", ""),
        "selected_class_name": selected.get("class_name", ""),
        "selected_det_id": selected.get("det_id", ""),
        "selected_region": selected.get("region", ""),
        "selected_component": selected.get("component", ""),
        "selected_source": selected.get("source", ""),
        "selected_fallback": selected.get("fallback", ""),
        "selected_bbox_xyxy": json_dumps_compact(selected.get("bbox_xyxy", "")),
        "selected_mask_area_ratio": selected.get("mask_area_ratio", ""),
        "selected_num_landmarks": selected.get("num_landmarks", ""),
        "selected_num_reliable_landmarks": selected.get("num_reliable_landmarks", ""),
        "num_candidates": result.get("num_candidates", ""),
        "run_dir": run_dir,
        "run_dir_name": run_dir_name,
        "overlay_path": overlay_path,
        "selected_image_crop": outputs.get("selected_image_crop", ""),
        "selected_mask_crop": outputs.get("selected_mask_crop", ""),
        "selected_masked_crop": outputs.get("selected_masked_crop", ""),
        "region_mask_full": outputs.get("region_mask_full", ""),
        "pipeline_dir": outputs.get("pipeline_dir", ""),
        "region_masked_crops_json": outputs.get("region_masked_crops_json", ""),
        "result_json": result_json_str,
        "relative_result_json": relative_result_json,
        "result_json_mtime": result_mtime,
        "result_json_mtime_iso": mtime_to_iso(result_mtime),
        "pipeline_status": pipeline_result.get("status", ""),
        "pipeline_total_seconds": pipeline_timing.get("total_seconds", ""),
        "pipeline_yolo_seconds": pipeline_timing.get("yolo_seconds", ""),
        "pipeline_sam_hq_seconds": pipeline_timing.get("sam_hq_seconds", ""),
        "pipeline_landmarks_seconds": pipeline_timing.get("landmarks_seconds", ""),
        "pipeline_region_crops_seconds": pipeline_timing.get("region_crops_seconds", ""),
        "pipeline_masked_crops_seconds": pipeline_timing.get("masked_crops_seconds", ""),
    }

    if include_pipeline_result:
        row["pipeline_result"] = json_dumps_compact(pipeline_result)

    return row


def make_parse_failed_row(result_json_path: Path, batch_dir: Path, error: Exception) -> dict[str, Any]:
    """Create a row when result.json cannot be parsed."""
    result_mtime = path_mtime(result_json_path)

    try:
        relative_result_json = str(result_json_path.relative_to(batch_dir))
    except Exception:
        relative_result_json = str(result_json_path)

    return {
        "status": "parse_failed",
        "error": str(error),
        "image": "",
        "image_name": "",
        "query": "",
        "target_region": "",
        "target_component": "",
        "target_class": "",
        "result_json": str(result_json_path),
        "relative_result_json": relative_result_json,
        "result_json_mtime": result_mtime,
        "result_json_mtime_iso": mtime_to_iso(result_mtime),
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write rows to a UTF-8-SIG CSV file for Excel compatibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count rows by one field."""
    counter: Counter[str] = Counter()

    for row in rows:
        value = str(row.get(key, "") or "").strip()
        if not value:
            value = "<empty>"
        counter[value] += 1

    return dict(counter)


def count_success_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Count total/success/failed by one field."""
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "success": 0, "failed": 0}
    )

    for row in rows:
        group_value = str(row.get(key, "") or "").strip()
        if not group_value:
            group_value = "<empty>"

        status = str(row.get("status", "") or "").strip().lower()

        grouped[group_value]["total"] += 1

        if status == "success":
            grouped[group_value]["success"] += 1
        elif status == "failed":
            grouped[group_value]["failed"] += 1

    result: dict[str, dict[str, Any]] = {}

    for group_value, counts in grouped.items():
        total = counts["total"]
        success = counts["success"]
        failed = counts["failed"]
        success_rate = success / total if total > 0 else 0.0

        result[group_value] = {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": round(success_rate, 6),
        }

    return result


def get_dedupe_key(row: dict[str, Any], dedupe_key: str) -> tuple[str, ...]:
    """Build deduplication key from a row."""
    image = str(row.get("image", "") or "")
    query = str(row.get("query", "") or "")
    target_region = str(row.get("target_region", "") or "")
    target_component = str(row.get("target_component", "") or "")
    target_class = str(row.get("target_class", "") or "")

    if dedupe_key == "image_query":
        return image, query

    if dedupe_key == "image_query_target":
        return image, query, target_region

    if dedupe_key == "image_query_target_component":
        return image, query, target_region, target_component

    if dedupe_key == "image_query_target_component_class":
        return image, query, target_region, target_component, target_class

    return image, query


def dedupe_rows_latest(
    rows: list[dict[str, Any]],
    dedupe_key: str = "image_query",
) -> list[dict[str, Any]]:
    """
    Deduplicate rows and keep the latest result.json.

    The default dedupe key is image + query. This is suitable for the batch60
    experiment because each image should have one result for each query.
    """
    best: dict[tuple[str, ...], dict[str, Any]] = {}

    for row in rows:
        key = get_dedupe_key(row, dedupe_key)
        mtime = float(row.get("result_json_mtime", 0.0) or 0.0)

        if key not in best:
            best[key] = row
            continue

        old_mtime = float(best[key].get("result_json_mtime", 0.0) or 0.0)
        if mtime >= old_mtime:
            best[key] = row

    deduped = list(best.values())
    deduped.sort(
        key=lambda x: (
            str(x.get("image_name", "")),
            str(x.get("query", "")),
            str(x.get("target_region", "")),
            str(x.get("target_component", "")),
            str(x.get("result_json", "")),
        )
    )

    return deduped


def build_summary(
    rows: list[dict[str, Any]],
    result_files: list[Path],
    batch_dir: Path,
    raw_num_rows_before_deduplication: int | None = None,
    dedupe_latest: bool = False,
    dedupe_key: str = "",
) -> dict[str, Any]:
    """Build JSON summary from flattened rows."""
    total = len(rows)
    success = sum(1 for row in rows if row.get("status") == "success")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    parse_failed = sum(1 for row in rows if row.get("status") == "parse_failed")
    unknown = total - success - failed - parse_failed

    success_rate = success / total if total > 0 else 0.0

    unique_images = sorted({str(row.get("image", "")) for row in rows if row.get("image")})
    unique_queries = sorted({str(row.get("query", "")) for row in rows if row.get("query")})

    summary: dict[str, Any] = {
        "batch_dir": str(batch_dir),
        "num_result_json_files": len(result_files),
        "raw_num_rows_before_deduplication": (
            raw_num_rows_before_deduplication
            if raw_num_rows_before_deduplication is not None
            else total
        ),
        "num_rows": total,
        "num_rows_after_deduplication": total,
        "dedupe_latest": dedupe_latest,
        "dedupe_key": dedupe_key if dedupe_latest else "",
        "num_unique_images": len(unique_images),
        "queries": unique_queries,
        "status_counts": {
            "success": success,
            "failed": failed,
            "parse_failed": parse_failed,
            "unknown": unknown,
        },
        "success_rate": round(success_rate, 6),
        "success_rate_percent": round(success_rate * 100.0, 2),
        "by_query": count_success_by_key(rows, "query"),
        "by_target_region": count_success_by_key(rows, "target_region"),
        "by_target_component": count_success_by_key(rows, "target_component"),
        "by_target_class": count_success_by_key(rows, "target_class"),
        "selected_class_counts": count_by(
            [row for row in rows if row.get("status") == "success"],
            "selected_class_name",
        ),
        "selected_region_counts": count_by(
            [row for row in rows if row.get("status") == "success"],
            "selected_region",
        ),
        "selected_component_counts": count_by(
            [row for row in rows if row.get("status") == "success"],
            "selected_component",
        ),
        "error_counts": count_by(
            [row for row in rows if row.get("status") != "success"],
            "error",
        ),
    }

    return summary


def main() -> None:
    args = parse_args()

    batch_dir = Path(args.batch_dir)

    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    output_csv = Path(args.output_csv) if args.output_csv else batch_dir / "batch_results.csv"
    output_json = Path(args.output_json) if args.output_json else batch_dir / "query_success_summary.json"
    failed_csv = Path(args.failed_csv) if args.failed_csv else batch_dir / "failed_cases.csv"

    result_files = find_result_json_files(batch_dir)

    if not result_files:
        raise FileNotFoundError(f"No result.json files found under: {batch_dir}")

    rows: list[dict[str, Any]] = []

    for result_json_path in result_files:
        try:
            result = load_json(result_json_path)
            row = extract_row(
                result=result,
                result_json_path=result_json_path,
                batch_dir=batch_dir,
                include_pipeline_result=args.include_pipeline_result,
            )
            rows.append(row)
        except Exception as exc:
            rows.append(
                make_parse_failed_row(
                    result_json_path=result_json_path,
                    batch_dir=batch_dir,
                    error=exc,
                )
            )

    raw_num_rows = len(rows)

    if args.dedupe_latest:
        rows = dedupe_rows_latest(rows, dedupe_key=args.dedupe_key)

    failed_rows = [
        row for row in rows if str(row.get("status", "")).lower() != "success"
    ]

    summary = build_summary(
        rows=rows,
        result_files=result_files,
        batch_dir=batch_dir,
        raw_num_rows_before_deduplication=raw_num_rows,
        dedupe_latest=bool(args.dedupe_latest),
        dedupe_key=args.dedupe_key,
    )

    write_csv(rows, output_csv)
    write_csv(failed_rows, failed_csv)
    save_json(summary, output_json)

    print("[INFO] Batch summarization finished.")
    print(f"[INFO] Batch dir: {batch_dir}")
    print(f"[INFO] Result JSON files found: {len(result_files)}")
    print(f"[INFO] Raw rows before deduplication: {raw_num_rows}")
    print(f"[INFO] Dedupe latest: {bool(args.dedupe_latest)}")
    if args.dedupe_latest:
        print(f"[INFO] Dedupe key: {args.dedupe_key}")
    print(f"[INFO] Rows written: {len(rows)}")
    print(f"[INFO] Success: {summary['status_counts']['success']}")
    print(f"[INFO] Failed: {summary['status_counts']['failed']}")
    print(f"[INFO] Parse failed: {summary['status_counts']['parse_failed']}")
    print(f"[INFO] Unknown: {summary['status_counts']['unknown']}")
    print(f"[INFO] Success rate: {summary['success_rate_percent']}%")
    print(f"[INFO] CSV saved to: {output_csv}")
    print(f"[INFO] Failed cases CSV saved to: {failed_csv}")
    print(f"[INFO] Summary JSON saved to: {output_json}")


if __name__ == "__main__":
    main()
