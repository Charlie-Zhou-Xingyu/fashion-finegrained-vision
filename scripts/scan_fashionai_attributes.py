import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ATTRIBUTE_TAXONOMY = {
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


def parse_onehot_label(label: str):
    label = str(label).strip()
    y_positions = [i for i, ch in enumerate(label) if ch.lower() == "y"]

    if len(y_positions) != 1:
        return None, f"invalid_onehot:{label}"

    return y_positions[0], ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\Aliintern\fashion-ai-data\fashionai_attributes\round1_fashionAI_attributes_test_a",
    )
    parser.add_argument(
        "--answer-csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"outputs\dataset_scan\fashionai_attributes",
    )
    args = parser.parse_args()

    root = Path(args.root)
    images_root = root / "Images"

    if args.answer_csv is None:
        answer_csv = root / "Tests" / "round1_fashionAI_attributes_answer_a.csv"
    else:
        answer_csv = Path(args.answer_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Answer CSV: {answer_csv}")

    df = pd.read_csv(
        answer_csv,
        header=None,
        names=["image_path", "attr_key", "label"],
    )

    total_rows = len(df)
    attr_key_counter = Counter()
    label_counter_by_attr = defaultdict(Counter)
    missing_images = []
    invalid_labels = []
    records = []

    for idx, row in df.iterrows():
        rel_image_path = str(row["image_path"]).strip()
        attr_key = str(row["attr_key"]).strip()
        label = str(row["label"]).strip()

        attr_key_counter[attr_key] += 1

        attr_id, error = parse_onehot_label(label)

        attr_value = None
        if error:
            invalid_labels.append(
                {
                    "row_index": int(idx),
                    "image_path": rel_image_path,
                    "attr_key": attr_key,
                    "label": label,
                    "error": error,
                }
            )
        else:
            values = ATTRIBUTE_TAXONOMY.get(attr_key)
            if values is None:
                invalid_labels.append(
                    {
                        "row_index": int(idx),
                        "image_path": rel_image_path,
                        "attr_key": attr_key,
                        "label": label,
                        "error": "unknown_attr_key",
                    }
                )
            elif attr_id >= len(values):
                invalid_labels.append(
                    {
                        "row_index": int(idx),
                        "image_path": rel_image_path,
                        "attr_key": attr_key,
                        "label": label,
                        "error": f"attr_id_out_of_range:{attr_id}",
                    }
                )
            else:
                attr_value = values[attr_id]
                label_counter_by_attr[attr_key][attr_id] += 1

        abs_image_path = root / rel_image_path
        if not abs_image_path.exists():
            missing_images.append(
                {
                    "row_index": int(idx),
                    "image_path": str(abs_image_path),
                    "attr_key": attr_key,
                    "label": label,
                }
            )

        records.append(
            {
                "row_index": int(idx),
                "image_path": str(abs_image_path),
                "relative_image_path": rel_image_path,
                "attr_key": attr_key,
                "raw_label": label,
                "attr_id": attr_id,
                "attr_value": attr_value,
            }
        )

    # Write normalized index
    index_path = output_dir / "fashionai_attributes_index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write label distribution CSV
    dist_path = output_dir / "attribute_label_distribution.csv"
    with open(dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attr_key",
                "attr_id",
                "attr_value",
                "count",
            ],
        )
        writer.writeheader()

        for attr_key, values in ATTRIBUTE_TAXONOMY.items():
            counter = label_counter_by_attr[attr_key]
            for attr_id, attr_value in enumerate(values):
                writer.writerow(
                    {
                        "attr_key": attr_key,
                        "attr_id": attr_id,
                        "attr_value": attr_value,
                        "count": counter.get(attr_id, 0),
                    }
                )

    # Write invalid and missing logs
    with open(output_dir / "missing_images.jsonl", "w", encoding="utf-8") as f:
        for item in missing_images:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_dir / "invalid_labels.jsonl", "w", encoding="utf-8") as f:
        for item in invalid_labels:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "root": str(root),
        "answer_csv": str(answer_csv),
        "total_rows": total_rows,
        "attr_key_counts": dict(attr_key_counter),
        "missing_images": len(missing_images),
        "invalid_labels": len(invalid_labels),
        "output_index": str(index_path),
        "distribution_csv": str(dist_path),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== FashionAI Attribute Summary ==========")
    print(f"Total rows: {total_rows}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Invalid labels: {len(invalid_labels)}")

    print("\nAttr key counts:")
    for k, v in sorted(attr_key_counter.items()):
        print(f"  {k}: {v}")

    print(f"\nSaved index: {index_path}")
    print(f"Saved distribution: {dist_path}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
