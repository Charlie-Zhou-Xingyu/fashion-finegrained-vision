from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEEPFASHION2_ID_TO_CATEGORY = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export category-to-valid-landmark-index schema from "
            "DeepFashion2 landmark JSONL files."
        )
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        nargs="+",
        required=True,
        help="One or more landmark JSONL files, e.g. train.jsonl validation.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/landmark_schema",
        help="Output directory.",
    )
    parser.add_argument(
        "--present-threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum present ratio for a landmark index to be considered valid "
            "for a category."
        ),
    )
    parser.add_argument(
        "--visible-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional minimum visible ratio. 0.0 means visibility ratio is not "
            "used for filtering."
        ),
    )
    parser.add_argument(
        "--max-landmarks",
        type=int,
        default=39,
        help="Max landmark index count used by current model.",
    )
    parser.add_argument(
        "--preview-records",
        type=int,
        default=3,
        help="Print first N parsed records for debugging.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception as exc:
                print(f"[WARN] Failed to parse {path}:{line_idx}: {exc}")

    return records


def normalize_category_name(value: Any) -> str:
    if value is None:
        return "unknown"

    name = str(value).strip().lower()
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name if name else "unknown"


def get_category_name(record: Dict[str, Any]) -> str:
    """
    Get category name from record.

    Supports:
        category_name
        class_name
        category
        label
        category_id
        class_id
    """
    candidate_keys = [
        "category_name",
        "class_name",
        "category",
        "label",
    ]

    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return normalize_category_name(value)

    for key in ["category_id", "class_id"]:
        value = record.get(key)
        if value is None:
            continue

        try:
            category_id = int(value)
        except Exception:
            continue

        if category_id in DEEPFASHION2_ID_TO_CATEGORY:
            return DEEPFASHION2_ID_TO_CATEGORY[category_id]

        return normalize_category_name(f"category_{category_id}")

    return "unknown"


def extract_landmarks(record: Dict[str, Any]) -> List[Any]:
    """
    Extract landmark list from record.

    Supports keys:
        landmarks
        keypoints
        joints
    """
    candidate_keys = [
        "landmarks",
        "keypoints",
        "joints",
    ]

    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, list):
            return value

    return []


def extract_visibility_list(record: Dict[str, Any]) -> List[Any]:
    """
    Extract separate visibility list if present.

    Some datasets store:
        landmarks: [[x, y], ...]
        visibility: [0/1/2, ...]
    """
    candidate_keys = [
        "visibility",
        "visibilities",
        "landmark_visibility",
        "landmark_visibilities",
        "valid",
    ]

    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, list):
            return value

    return []


def is_zero_or_none(value: Any) -> bool:
    if value is None:
        return True

    try:
        return abs(float(value)) < 1e-12
    except Exception:
        return False


def parse_landmark_item(
    item: Any,
    separate_visibility: Any = None,
) -> Tuple[bool, bool]:
    """
    Parse one landmark item.

    Returns:
        present: Whether landmark exists.
        visible: Whether landmark is visible.

    Supports dict:
        {"x": ..., "y": ..., "visibility": 2}
        {"x": ..., "y": ..., "v": 2}

    Supports list:
        [x, y, visibility]
        [x, y]

    Also supports separate visibility list.
    """
    if separate_visibility is not None:
        try:
            v = int(separate_visibility)
        except Exception:
            v = 0

        return v > 0, v == 2

    if isinstance(item, dict):
        visibility = item.get("visibility", item.get("v", None))
        x = item.get("x", None)
        y = item.get("y", None)

        if visibility is not None:
            try:
                v = int(visibility)
            except Exception:
                v = 0

            return v > 0, v == 2

        present = x is not None and y is not None and not (
            is_zero_or_none(x) and is_zero_or_none(y)
        )
        return bool(present), bool(present)

    if isinstance(item, list):
        if len(item) >= 3:
            try:
                v = int(item[2])
            except Exception:
                v = 0

            return v > 0, v == 2

        if len(item) >= 2:
            x, y = item[0], item[1]
            present = not (is_zero_or_none(x) and is_zero_or_none(y))
            return bool(present), bool(present)

    return False, False


def preview_records(records: List[Dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return

    print("[INFO] Preview records:")

    for idx, record in enumerate(records[:limit], start=1):
        category_name = get_category_name(record)
        landmarks = extract_landmarks(record)
        visibility = extract_visibility_list(record)

        print(f"  record {idx}:")
        print(f"    keys: {sorted(record.keys())}")
        print(f"    category: {category_name}")
        print(f"    num_landmarks: {len(landmarks)}")
        print(f"    num_visibility: {len(visibility)}")
        if landmarks:
            print(f"    first_landmark: {landmarks[0]}")
        if visibility:
            print(f"    first_visibility: {visibility[0]}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_total = Counter()
    category_index_present = defaultdict(Counter)
    category_index_visible = defaultdict(Counter)

    total_records = 0
    did_preview = False

    for jsonl_path_str in args.jsonl:
        jsonl_path = Path(jsonl_path_str)

        if not jsonl_path.exists():
            print(f"[WARN] JSONL not found: {jsonl_path}")
            continue

        records = load_jsonl(jsonl_path)
        print(f"[INFO] Loaded {len(records)} records from {jsonl_path}")

        if not did_preview:
            preview_records(records, int(args.preview_records))
            did_preview = True

        for record in records:
            total_records += 1
            category_name = get_category_name(record)
            category_total[category_name] += 1

            landmarks = extract_landmarks(record)
            visibility_list = extract_visibility_list(record)

            for idx in range(1, int(args.max_landmarks) + 1):
                item_idx = idx - 1

                if item_idx >= len(landmarks):
                    continue

                separate_visibility = None
                if item_idx < len(visibility_list):
                    separate_visibility = visibility_list[item_idx]

                present, visible = parse_landmark_item(
                    landmarks[item_idx],
                    separate_visibility=separate_visibility,
                )

                if present:
                    category_index_present[category_name][idx] += 1

                if visible:
                    category_index_visible[category_name][idx] += 1

    stats: Dict[str, Any] = {
        "total_records": total_records,
        "max_landmarks": int(args.max_landmarks),
        "present_threshold": float(args.present_threshold),
        "visible_threshold": float(args.visible_threshold),
        "categories": {},
    }

    category_to_valid_indices: Dict[str, List[int]] = {}

    for category_name in sorted(category_total.keys()):
        total = int(category_total[category_name])
        valid_indices: List[int] = []

        index_stats: Dict[str, Any] = {}

        for idx in range(1, int(args.max_landmarks) + 1):
            present_count = int(category_index_present[category_name][idx])
            visible_count = int(category_index_visible[category_name][idx])

            present_ratio = present_count / total if total > 0 else 0.0
            visible_ratio = visible_count / total if total > 0 else 0.0

            is_valid = (
                present_ratio >= float(args.present_threshold)
                and visible_ratio >= float(args.visible_threshold)
            )

            if is_valid:
                valid_indices.append(idx)

            index_stats[str(idx)] = {
                "present_count": present_count,
                "visible_count": visible_count,
                "present_ratio": present_ratio,
                "visible_ratio": visible_ratio,
                "valid": bool(is_valid),
            }

        category_to_valid_indices[category_name] = valid_indices

        stats["categories"][category_name] = {
            "num_records": total,
            "valid_indices": valid_indices,
            "num_valid_indices": len(valid_indices),
            "indices": index_stats,
        }

    stats_path = output_dir / "category_landmark_stats.json"
    mapping_path = output_dir / "category_to_valid_indices.json"
    py_path = output_dir / "category_to_valid_indices.py"

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(category_to_valid_indices, f, ensure_ascii=False, indent=2)

    with py_path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated category-to-valid-landmark-index mapping.\n")
        f.write("CATEGORY_TO_VALID_LANDMARK_INDICES = {\n")
        for category_name in sorted(category_to_valid_indices.keys()):
            indices = category_to_valid_indices[category_name]
            f.write(f"    {category_name!r}: {indices!r},\n")
        f.write("}\n")

    print("[INFO] Export finished.")
    print(f"[INFO] Total records: {total_records}")
    print(f"[INFO] Stats: {stats_path}")
    print(f"[INFO] Mapping JSON: {mapping_path}")
    print(f"[INFO] Mapping Python: {py_path}")

    print("[INFO] Summary:")
    for category_name in sorted(category_to_valid_indices.keys()):
        print(
            f"  {category_name}: "
            f"{len(category_to_valid_indices[category_name])} valid indices "
            f"{category_to_valid_indices[category_name]}"
        )


if __name__ == "__main__":
    main()
