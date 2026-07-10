# scripts/inspect_region_crops_json.py
import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=str)
    parser.add_argument("--show", default=3, type=int)
    args = parser.parse_args()

    path = Path(args.json)
    data = load_json(path)

    crops = data.get("crops", [])
    print(f"[INFO] json: {path}")
    print(f"[INFO] top-level keys: {list(data.keys())}")
    print(f"[INFO] num crops: {len(crops)}")

    region_counter = Counter()
    component_counter = Counter()
    class_counter = Counter()
    success_counter = Counter()
    masked_success_counter = Counter()
    source_counter = Counter()

    for r in crops:
        region_counter[str(r.get("region", ""))] += 1
        component_counter[str(r.get("component", ""))] += 1
        class_counter[str(r.get("class_name", ""))] += 1
        success_counter[str(r.get("success", ""))] += 1
        masked_success_counter[str(r.get("masked_success", ""))] += 1
        source_counter[str(r.get("source", ""))] += 1

    print("\n[REGION COUNTS]")
    for k, v in region_counter.most_common():
        print(f"{k}: {v}")

    print("\n[COMPONENT COUNTS]")
    for k, v in component_counter.most_common(30):
        print(f"{k}: {v}")

    print("\n[CLASS COUNTS]")
    for k, v in class_counter.most_common(30):
        print(f"{k}: {v}")

    print("\n[SUCCESS COUNTS]")
    for k, v in success_counter.most_common():
        print(f"{k}: {v}")

    print("\n[MASKED SUCCESS COUNTS]")
    for k, v in masked_success_counter.most_common():
        print(f"{k}: {v}")

    print("\n[SOURCE COUNTS]")
    for k, v in source_counter.most_common():
        print(f"{k}: {v}")

    print("\n[SAMPLE RECORDS]")
    for i, r in enumerate(crops[: args.show]):
        print(f"\n--- sample {i} ---")
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
