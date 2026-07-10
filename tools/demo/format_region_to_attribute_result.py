# tools/demo/format_region_to_attribute_result.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def choose_best_result(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = []
    for row in rows:
        if row.get("error") is not None:
            continue
        pred = row.get("prediction")
        if not isinstance(pred, dict):
            continue
        if "confidence" not in pred:
            continue
        valid.append(row)

    if not valid:
        return None

    # First version: choose highest confidence prediction.
    # Later this can be replaced by garment area, detection confidence, or user-selected instance.
    return max(valid, key=lambda r: float(r["prediction"]["confidence"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", required=True, type=str)
    parser.add_argument("--query", default="领口", type=str)
    parser.add_argument("--region", default="neckline", type=str)
    parser.add_argument("--attribute-task", default="neckline_design", type=str)
    parser.add_argument("--output-json", required=True, type=str)
    args = parser.parse_args()

    pred_jsonl = Path(args.pred_jsonl)
    output_json = Path(args.output_json)

    rows = read_jsonl(pred_jsonl)
    best = choose_best_result(rows)

    if best is None:
        result = {
            "query": args.query,
            "status": "failed",
            "error": "no_valid_attribute_prediction",
            "region": args.region,
            "attribute_task": args.attribute_task,
            "num_candidates": len(rows),
        }
        write_json(result, output_json)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    pred = best["prediction"]

    result = {
        "query": args.query,
        "status": "ok",
        "region": {
            "region_name": best.get("region", args.region),
            "component": best.get("component"),
            "crop_path": best.get("crop_path"),
            "image_crop_path": best.get("image_crop_path"),
            "masked_crop_path": best.get("masked_crop_path"),
            "class_name": best.get("class_name"),
            "det_id": best.get("det_id"),
            "bbox_xyxy": best.get("bbox_xyxy"),
        },
        "attribute": {
            "task": args.attribute_task,
            "pred_label_id": pred.get("pred_label_id"),
            "pred_label_name": pred.get("pred_label_name"),
            "confidence": pred.get("confidence"),
            "topk": pred.get("topk"),
        },
        "compact": {
            "query": args.query,
            "region": best.get("region", args.region),
            "attribute_task": args.attribute_task,
            "prediction": pred.get("pred_label_name"),
            "confidence": pred.get("confidence"),
        },
        "num_candidates": len(rows),
    }

    write_json(result, output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
