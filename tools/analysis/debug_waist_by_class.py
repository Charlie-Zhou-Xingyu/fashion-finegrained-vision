import json
import argparse
import collections
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-json", required=True)
    args = parser.parse_args()

    path = Path(args.landmarks_json)
    data = json.load(open(path, "r", encoding="utf-8"))

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = (
            data.get("results")
            or data.get("records")
            or data.get("predictions")
            or data.get("items")
            or []
        )
    else:
        raise TypeError(type(data))

    region_counter = collections.Counter()
    by_class_region = collections.defaultdict(collections.Counter)
    by_class_total_instances = collections.Counter()
    by_class_waist_instances = collections.Counter()

    for rec in records:
        cls = (
            rec.get("class_name")
            or rec.get("category")
            or rec.get("label")
            or rec.get("det_class_name")
            or "unknown"
        )

        by_class_total_instances[cls] += 1

        landmarks = rec.get("landmarks") or rec.get("keypoints") or []
        has_waist = False

        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            region = lm.get("region", "unknown")
            region_counter[region] += 1
            by_class_region[cls][region] += 1
            if region == "waist":
                has_waist = True

        if has_waist:
            by_class_waist_instances[cls] += 1

    print("=== Overall region counts ===")
    for k, v in region_counter.most_common():
        print(f"{k}: {v}")

    print()
    print("=== Waist landmarks by class ===")
    for cls in sorted(by_class_region.keys()):
        waist = by_class_region[cls].get("waist", 0)
        total = sum(by_class_region[cls].values())
        inst_total = by_class_total_instances[cls]
        inst_waist = by_class_waist_instances[cls]
        if waist > 0 or cls != "unknown":
            ratio = waist / total if total else 0
            print(
                f"{cls}: waist_landmarks={waist}, "
                f"all_landmarks={total}, "
                f"waist_lm_ratio={ratio:.4f}, "
                f"instances_with_waist={inst_waist}/{inst_total}"
            )

    print()
    print("=== Full by-class region table ===")
    for cls in sorted(by_class_region.keys()):
        print(cls, dict(by_class_region[cls]))


if __name__ == "__main__":
    main()
