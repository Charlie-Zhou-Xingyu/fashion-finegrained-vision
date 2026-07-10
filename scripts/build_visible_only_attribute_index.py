import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_label_map(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input-index-dir", required=True)
    parser.add_argument("--input-label-map", required=True)
    parser.add_argument("--output-index-dir", required=True)
    parser.add_argument("--exclude-labels", nargs="+", default=["Invisible"])
    args = parser.parse_args()

    input_dir = Path(args.input_index_dir)
    output_dir = Path(args.output_index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_label_map = load_label_map(Path(args.input_label_map))

    # Support both possible label-map formats.
    if "id_to_label" in old_label_map:
        old_id_to_label = {int(k): v for k, v in old_label_map["id_to_label"].items()}
    elif "idx_to_label" in old_label_map:
        old_id_to_label = {int(k): v for k, v in old_label_map["idx_to_label"].items()}
    elif "classes" in old_label_map:
        old_id_to_label = {i: v for i, v in enumerate(old_label_map["classes"])}
    else:
        # fallback: label_map may be label -> id
        old_id_to_label = {int(v): k for k, v in old_label_map.items() if isinstance(v, int)}

    visible_labels = [
        label for _, label in sorted(old_id_to_label.items(), key=lambda x: x[0])
        if label not in set(args.exclude_labels)
    ]

    new_label_to_id = {label: i for i, label in enumerate(visible_labels)}
    new_id_to_label = {str(i): label for label, i in new_label_to_id.items()}

    new_label_map = {
        "task": args.task,
        "num_classes": len(visible_labels),
        "label_to_id": new_label_to_id,
        "id_to_label": new_id_to_label,
        "classes": visible_labels,
        "excluded_labels": args.exclude_labels,
        "source_label_map": str(args.input_label_map),
    }

    summary = {
        "task": args.task,
        "input_index_dir": str(input_dir),
        "output_index_dir": str(output_dir),
        "exclude_labels": args.exclude_labels,
        "visible_labels": visible_labels,
        "splits": {},
    }

    for split in SPLITS:
        in_path = input_dir / f"{args.task}_{split}.jsonl"
        rows = read_jsonl(in_path)

        out_rows = []
        excluded = 0
        missing = 0

        for r in rows:
            label_name = r.get("label_name")
            if label_name in args.exclude_labels:
                excluded += 1
                continue
            if label_name not in new_label_to_id:
                missing += 1
                continue

            nr = dict(r)
            nr["old_label_id"] = r.get("label_id")
            nr["old_label_name"] = label_name
            nr["label_id"] = new_label_to_id[label_name]
            nr["label_name"] = label_name
            nr["visible_only"] = True
            nr["excluded_from_original_labels"] = args.exclude_labels
            out_rows.append(nr)

        out_path = output_dir / f"{args.task}_{split}.jsonl"
        write_jsonl(out_rows, out_path)

        summary["splits"][split] = {
            "input_rows": len(rows),
            "output_rows": len(out_rows),
            "excluded_rows": excluded,
            "missing_label_rows": missing,
            "output_jsonl": str(out_path),
        }

        print(
            f"[OK] {split}: input={len(rows)}, output={len(out_rows)}, "
            f"excluded={excluded}, missing={missing}"
        )

    save_json(new_label_map, output_dir / f"label_map_{args.task}.json")
    save_json(summary, output_dir / f"visible_only_summary_{args.task}.json")

    print(f"[OK] Saved label map: {output_dir / f'label_map_{args.task}.json'}")
    print(f"[OK] Saved summary: {output_dir / f'visible_only_summary_{args.task}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
