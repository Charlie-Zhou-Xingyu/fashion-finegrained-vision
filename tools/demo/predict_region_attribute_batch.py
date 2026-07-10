import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.attribute_classifier import build_attribute_classifier


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_label_map(label_map_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    data = load_json(label_map_path)

    # Compatible with several common formats.
    if "id_to_label" in data:
        raw_id_to_label = data["id_to_label"]
    elif "idx_to_label" in data:
        raw_id_to_label = data["idx_to_label"]
    elif "classes" in data and isinstance(data["classes"], list):
        raw_id_to_label = {str(i): name for i, name in enumerate(data["classes"])}
    elif all(str(k).isdigit() for k in data.keys()):
        raw_id_to_label = data
    elif "label_to_id" in data:
        label_to_id = {str(k): int(v) for k, v in data["label_to_id"].items()}
        id_to_label = {v: k for k, v in label_to_id.items()}
        return id_to_label, label_to_id
    else:
        raise ValueError(f"Unsupported label map format: {label_map_path}")

    id_to_label = {int(k): str(v) for k, v in raw_id_to_label.items()}
    label_to_id = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id


def build_infer_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            # Maybe the checkpoint itself is a state dict.
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Remove possible 'module.' prefix from DataParallel.
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[len("module."):]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict, strict=True)
    return model


def predict_one(
    model: torch.nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    id_to_label: Dict[int, str],
    device: torch.device,
    topk: int,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    k = min(topk, probs.numel())
    confs, ids = torch.topk(probs, k=k)

    topk_rows = []
    for label_id, conf in zip(ids.cpu().tolist(), confs.cpu().tolist()):
        topk_rows.append({
            "label_id": int(label_id),
            "label_name": id_to_label[int(label_id)],
            "confidence": float(conf),
        })

    return {
        "pred_label_id": topk_rows[0]["label_id"],
        "pred_label_name": topk_rows[0]["label_name"],
        "confidence": topk_rows[0]["confidence"],
        "topk": topk_rows,
    }


def get_crop_path(record: Dict[str, Any], crop_input_type: str) -> str:
    if crop_input_type == "image_crop":
        return record.get("image_crop_path") or record.get("crop_path")

    if crop_input_type == "masked_crop":
        return record.get("masked_crop_path")

    if crop_input_type == "raw_region_crop":
        return record.get("crop_path")

    if crop_input_type == "expanded_crop":
        return (
            record.get("expanded_crop_path")
            or record.get("image_crop_path")
            or record.get("crop_path")
        )

    if crop_input_type == "upper_crop":
        return (
            record.get("upper_crop_path")
            or record.get("expanded_crop_path")
            or record.get("image_crop_path")
            or record.get("crop_path")
        )

    raise ValueError(f"Unsupported crop_input_type: {crop_input_type}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch predict fine-grained attributes from existing region crops."
    )

    parser.add_argument("--region-crops-json", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--label-map", required=True, type=str)
    parser.add_argument("--task", default="sleeve_length", type=str)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--topk", default=3, type=int)
    parser.add_argument("--device", default="auto", type=str)

    parser.add_argument(
        "--region",
        default="sleeve",
        type=str,
        help="Filter records by region. Use 'all' to disable.",
    )
    parser.add_argument(
        "--component-contains",
        default="",
        type=str,
        help="Optional substring filter for component, e.g. left_sleeve.",
    )
    parser.add_argument(
        "--class-contains",
        default="",
        type=str,
        help="Optional substring filter for class_name.",
    )
    parser.add_argument(
        "--crop-input-type",
        default="image_crop",
        choices=[
            "image_crop",
            "masked_crop",
            "raw_region_crop",
            "expanded_crop",
            "upper_crop",
        ],
        help="Which crop path to use as classifier input.",
    )

    parser.add_argument(
        "--require-masked-success",
        action="store_true",
        help="Require masked_success=True even when using image_crop or raw_region_crop.",
    )

    parser.add_argument("--max-samples", default=0, type=int)
    parser.add_argument("--output-jsonl", required=True, type=str)
    parser.add_argument("--output-summary", required=True, type=str)

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    region_crops_json = Path(args.region_crops_json)
    checkpoint_path = Path(args.checkpoint)
    label_map_path = Path(args.label_map)
    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)

    id_to_label, label_to_id = load_label_map(label_map_path)
    num_classes = len(id_to_label)

    # IMPORTANT:
    # If your build_attribute_classifier signature differs,
    # adjust this line after checking models/attribute_classifier.py.
    model = build_attribute_classifier(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=False,
    )

    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    transform = build_infer_transform(args.img_size)

    data = load_json(region_crops_json)
    records = data.get("crops", [])
    if not isinstance(records, list):
        raise ValueError("Invalid input JSON: expected key 'crops' as a list.")

    selected = []
    for record in records:
        if not record.get("success", False):
            continue

        if args.crop_input_type == "masked_crop" or args.require_masked_success:
            if "masked_success" in record and not record.get("masked_success", False):
                continue




        region = str(record.get("region", ""))
        component = str(record.get("component", ""))
        class_name = str(record.get("class_name", ""))

        if args.region != "all" and region != args.region:
            continue

        if args.component_contains and args.component_contains not in component:
            continue

        if args.class_contains and args.class_contains not in class_name:
            continue

        crop_path_str = get_crop_path(record, args.crop_input_type)
        if not crop_path_str:
            continue

        crop_path = Path(crop_path_str)
        if not crop_path.exists():
            continue

        selected.append(record)

        if args.max_samples > 0 and len(selected) >= args.max_samples:
            break

    results = []
    num_failed = 0
    pred_counter: Dict[str, int] = {}

    for record in tqdm(selected, desc="Predict attributes"):
        crop_path_str = get_crop_path(record, args.crop_input_type)
        crop_path = Path(crop_path_str)

        try:
            pred = predict_one(
                model=model,
                image_path=crop_path,
                transform=transform,
                id_to_label=id_to_label,
                device=device,
                topk=args.topk,
            )

            pred_name = pred["pred_label_name"]
            pred_counter[pred_name] = pred_counter.get(pred_name, 0) + 1

            out = {
                "task": args.task,
                "image_path": record.get("image_path"),
                "class_name": record.get("class_name"),
                "det_id": record.get("det_id"),
                "region": record.get("region"),
                "component": record.get("component"),
                "source": record.get("source"),
                "fallback": record.get("fallback"),

                "bbox_xyxy": record.get("bbox_xyxy"),
                "expanded_bbox_xyxy": record.get("expanded_bbox_xyxy"),
                "upper_bbox_xyxy": record.get("upper_bbox_xyxy"),

                "crop_input_type": args.crop_input_type,
                "crop_path": str(crop_path),

                "raw_region_crop_path": record.get("crop_path"),
                "image_crop_path": record.get("image_crop_path"),
                "masked_crop_path": record.get("masked_crop_path"),
                "mask_crop_path": record.get("mask_crop_path"),
                "expanded_crop_path": record.get("expanded_crop_path"),
                "upper_crop_path": record.get("upper_crop_path"),
                "segment_mask_path": record.get("segment_mask_path"),

                "prediction": pred,
                "error": None,
            }


        except Exception as e:
            num_failed += 1
            out = {
                "task": args.task,
                "image_path": record.get("image_path"),
                "class_name": record.get("class_name"),
                "det_id": record.get("det_id"),
                "region": record.get("region"),
                "component": record.get("component"),
                "crop_input_type": args.crop_input_type,
                "crop_path": str(crop_path),

                "raw_region_crop_path": record.get("crop_path"),
                "image_crop_path": record.get("image_crop_path"),
                "masked_crop_path": record.get("masked_crop_path"),
                "expanded_crop_path": record.get("expanded_crop_path"),
                "upper_crop_path": record.get("upper_crop_path"),

                "prediction": None,
                "error": repr(e),
            }


        results.append(out)

    summary = {
        "task": args.task,
        "region_crops_json": str(region_crops_json),
        "checkpoint": str(checkpoint_path),
        "label_map": str(label_map_path),
        "arch": args.arch,
        "img_size": args.img_size,
        "topk": args.topk,
        "device": str(device),
        "filters": {
            "region": args.region,
            "component_contains": args.component_contains,
            "class_contains": args.class_contains,
            "crop_input_type": args.crop_input_type,
            "require_masked_success": args.require_masked_success,
            "max_samples": args.max_samples,
        },
        "num_input_records": len(records),
        "num_selected": len(selected),
        "num_results": len(results),
        "num_failed": num_failed,
        "prediction_counts": pred_counter,
        "output_jsonl": str(output_jsonl),
    }

    write_jsonl(results, output_jsonl)
    write_json(summary, output_summary)

    print("[OK] Batch region attribute prediction completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
