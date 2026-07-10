#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict garment-level fine-grained attributes from YOLO garment crops.

Use cases:
    YOLO garment crop -> sleeve_length classifier -> top-k prediction
    YOLO garment crop -> pant_length classifier -> top-k prediction

This script validates whether FashionAI-trained attribute classifiers work
on garment-level YOLO crops.

Example sleeve_length:
    python tools/demo/predict_garment_attribute_from_yolo_crops.py ^
      --crops-dir outputs/pipeline_13cls_eval_balanced/01_yolo/crops ^
      --checkpoint outputs/p2_sleeve_length_resnet18_seed2/best.pt ^
      --label-map data/fashionai_attribute_index/label_map_sleeve_length.json ^
      --task sleeve_length ^
      --arch resnet18 ^
      --attribute-mode sleeve_length ^
      --topk 3 ^
      --output-jsonl outputs/p2_region_attribute_demo/sleeve_length_garment_crop_all_v2.jsonl ^
      --output-summary outputs/p2_region_attribute_demo/sleeve_length_garment_crop_all_v2_summary.json

Example pant_length:
    python tools/demo/predict_garment_attribute_from_yolo_crops.py ^
      --crops-dir outputs/pipeline_13cls_eval_balanced/01_yolo/crops ^
      --checkpoint outputs/p2_pant_length_resnet18_seed2/best.pt ^
      --label-map data/fashionai_attribute_index/label_map_pant_length.json ^
      --task pant_length ^
      --arch resnet18 ^
      --attribute-mode pant_length ^
      --topk 3 ^
      --output-jsonl outputs/p2_region_attribute_demo/pant_length_garment_crop_all.jsonl ^
      --output-summary outputs/p2_region_attribute_demo/pant_length_garment_crop_all_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.attribute_classifier import build_attribute_classifier


# ---------------------------------------------------------------------
# YOLO classes for sleeve_length weak evaluation
# ---------------------------------------------------------------------
SLEEVE_RELATED_CLASSES = {
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "short_sleeve_dress",
    "long_sleeve_dress",
}

SHORT_SLEEVE_CLASSES = {
    "short_sleeve_top",
    "short_sleeve_outwear",
    "short_sleeve_dress",
}

LONG_SLEEVE_CLASSES = {
    "long_sleeve_top",
    "long_sleeve_outwear",
    "long_sleeve_dress",
}


# ---------------------------------------------------------------------
# YOLO classes for pant_length weak evaluation
# ---------------------------------------------------------------------
PANT_RELATED_CLASSES = {
    "shorts",
    "trousers",
}

PANT_SHORT_CLASSES = {
    "shorts",
}

PANT_LONG_CLASSES = {
    "trousers",
}


# ---------------------------------------------------------------------
# FashionAI sleeve_length labels
# ---------------------------------------------------------------------
SLEEVE_LENGTH_SHORT_PRED_LABELS = {
    "Cup Sleeves",
    "Short Sleeves",
    "Elbow Sleeves",
}

SLEEVE_LENGTH_LONG_PRED_LABELS = {
    "3/4 Sleeves",
    "Wrist Length",
    "Long Sleeves",
    "Extra Long Sleeves",
}

SLEEVE_LENGTH_NO_SLEEVE_LABELS = {
    "Sleeveless",
}


# ---------------------------------------------------------------------
# FashionAI pant_length labels
# ---------------------------------------------------------------------
PANT_LENGTH_INVISIBLE_LABELS = {
    "Invisible",
}

PANT_LENGTH_SHORT_PRED_LABELS = {
    "Short Pant",
}

PANT_LENGTH_MID_PRED_LABELS = {
    "Mid Length",
}

PANT_LENGTH_LONG_PRED_LABELS = {
    "3/4 Length",
    "Cropped Pant",
    "Full Length",
}


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


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[len("module."):]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict, strict=True)
    return model


def parse_yolo_crop_filename(path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse YOLO crop filename.

    Expected examples:
        000002_det000_short_sleeve_top_0.86.jpg
        000034_det000_long_sleeve_outwear_0.80.jpg
        000001_det001_skirt_0.42.jpg
        000003_det000_shorts_0.91.jpg
        000004_det000_trousers_0.88.jpg

    Returns:
        {
            "image_stem": "000002",
            "det_id": 0,
            "class_name": "short_sleeve_top",
            "yolo_confidence": 0.86
        }
    """
    stem = path.stem

    # Pattern:
    # image_stem_detXXX_class_name_conf
    # class_name may contain underscores.
    pattern = r"^(?P<image_stem>\d+)_det(?P<det_id>\d+)_(?P<class_name>.+)_(?P<conf>\d+(?:\.\d+)?)$"
    match = re.match(pattern, stem)
    if not match:
        return None

    image_stem = match.group("image_stem")
    det_id = int(match.group("det_id"))
    class_name = match.group("class_name")
    yolo_confidence = float(match.group("conf"))

    return {
        "image_stem": image_stem,
        "det_id": det_id,
        "class_name": class_name,
        "yolo_confidence": yolo_confidence,
    }


def list_crop_files(crops_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for path in crops_dir.iterdir():
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    files.sort(key=lambda p: p.name)
    return files


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


# ---------------------------------------------------------------------
# Sleeve weak evaluation
# ---------------------------------------------------------------------
def get_weak_gt_group_for_sleeve_length(class_name: str) -> str:
    if class_name in SHORT_SLEEVE_CLASSES:
        return "short"
    if class_name in LONG_SLEEVE_CLASSES:
        return "long"
    return "unknown"


def get_pred_group_for_sleeve_length(pred_label_name: str) -> str:
    if pred_label_name in SLEEVE_LENGTH_SHORT_PRED_LABELS:
        return "short"
    if pred_label_name in SLEEVE_LENGTH_LONG_PRED_LABELS:
        return "long"
    if pred_label_name in SLEEVE_LENGTH_NO_SLEEVE_LABELS:
        return "no_sleeve"
    return "unknown"


def coarse_match_sleeve_length(class_name: str, pred_label_name: str) -> Optional[bool]:
    gt_group = get_weak_gt_group_for_sleeve_length(class_name)
    pred_group = get_pred_group_for_sleeve_length(pred_label_name)

    if gt_group == "unknown":
        return None

    if gt_group == "short":
        return pred_group == "short"

    if gt_group == "long":
        return pred_group == "long"

    return None


def coarse_match_sleeve_length_relaxed(class_name: str, pred_label_name: str) -> Optional[bool]:
    gt_group = get_weak_gt_group_for_sleeve_length(class_name)
    pred_group = get_pred_group_for_sleeve_length(pred_label_name)

    if gt_group == "unknown":
        return None

    if gt_group == "short":
        return pred_group in {"short", "no_sleeve"}

    if gt_group == "long":
        return pred_group == "long"

    return None


# ---------------------------------------------------------------------
# Pant weak evaluation
# ---------------------------------------------------------------------
def get_weak_gt_group_for_pant_length(class_name: str) -> str:
    if class_name in PANT_SHORT_CLASSES:
        return "short"
    if class_name in PANT_LONG_CLASSES:
        return "long"
    return "unknown"


def get_pred_group_for_pant_length(pred_label_name: str) -> str:
    if pred_label_name in PANT_LENGTH_INVISIBLE_LABELS:
        return "invisible"
    if pred_label_name in PANT_LENGTH_SHORT_PRED_LABELS:
        return "short"
    if pred_label_name in PANT_LENGTH_MID_PRED_LABELS:
        return "mid"
    if pred_label_name in PANT_LENGTH_LONG_PRED_LABELS:
        return "long"
    return "unknown"


def coarse_match_pant_length(class_name: str, pred_label_name: str) -> Optional[bool]:
    """
    Strict weak coarse match:
        YOLO shorts expects Short Pant.
        YOLO trousers expects 3/4 Length, Cropped Pant, or Full Length.
    """
    gt_group = get_weak_gt_group_for_pant_length(class_name)
    pred_group = get_pred_group_for_pant_length(pred_label_name)

    if gt_group == "unknown":
        return None

    if gt_group == "short":
        return pred_group == "short"

    if gt_group == "long":
        return pred_group == "long"

    return None


def coarse_match_pant_length_relaxed(class_name: str, pred_label_name: str) -> Optional[bool]:
    """
    Relaxed weak coarse match:
        YOLO shorts expects Short Pant or Mid Length.
        YOLO trousers expects 3/4 Length, Cropped Pant, or Full Length.
    """
    gt_group = get_weak_gt_group_for_pant_length(class_name)
    pred_group = get_pred_group_for_pant_length(pred_label_name)

    if gt_group == "unknown":
        return None

    if gt_group == "short":
        return pred_group in {"short", "mid"}

    if gt_group == "long":
        return pred_group == "long"

    return None


# ---------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------
def get_related_classes(attribute_mode: str) -> set[str]:
    if attribute_mode == "sleeve_length":
        return SLEEVE_RELATED_CLASSES
    if attribute_mode == "pant_length":
        return PANT_RELATED_CLASSES
    raise ValueError(f"Unsupported attribute_mode: {attribute_mode}")


def get_weak_gt_group(attribute_mode: str, class_name: str) -> str:
    if attribute_mode == "sleeve_length":
        return get_weak_gt_group_for_sleeve_length(class_name)
    if attribute_mode == "pant_length":
        return get_weak_gt_group_for_pant_length(class_name)
    return "unknown"


def get_pred_group(attribute_mode: str, pred_label_name: str) -> str:
    if attribute_mode == "sleeve_length":
        return get_pred_group_for_sleeve_length(pred_label_name)
    if attribute_mode == "pant_length":
        return get_pred_group_for_pant_length(pred_label_name)
    return "unknown"


def coarse_match(attribute_mode: str, class_name: str, pred_label_name: str) -> Optional[bool]:
    if attribute_mode == "sleeve_length":
        return coarse_match_sleeve_length(class_name, pred_label_name)
    if attribute_mode == "pant_length":
        return coarse_match_pant_length(class_name, pred_label_name)
    return None


def coarse_match_relaxed(attribute_mode: str, class_name: str, pred_label_name: str) -> Optional[bool]:
    if attribute_mode == "sleeve_length":
        return coarse_match_sleeve_length_relaxed(class_name, pred_label_name)
    if attribute_mode == "pant_length":
        return coarse_match_pant_length_relaxed(class_name, pred_label_name)
    return None


def get_coarse_eval_descriptions(attribute_mode: str) -> Dict[str, Dict[str, Any]]:
    if attribute_mode == "sleeve_length":
        return {
            "strict": {
                "description": (
                    "Strict weak directional evaluation using YOLO class names. "
                    "short_sleeve_* expects Cup/Short/Elbow. "
                    "long_sleeve_* expects 3/4/Wrist/Long/Extra Long."
                ),
                "short_expected_pred_labels": sorted(SLEEVE_LENGTH_SHORT_PRED_LABELS),
                "long_expected_pred_labels": sorted(SLEEVE_LENGTH_LONG_PRED_LABELS),
            },
            "relaxed": {
                "description": (
                    "Relaxed weak directional evaluation. "
                    "short_sleeve_* expects Sleeveless/Cup/Short/Elbow, "
                    "long_sleeve_* expects 3/4/Wrist/Long/Extra Long."
                ),
                "short_expected_pred_labels": sorted(
                    SLEEVE_LENGTH_SHORT_PRED_LABELS | SLEEVE_LENGTH_NO_SLEEVE_LABELS
                ),
                "long_expected_pred_labels": sorted(SLEEVE_LENGTH_LONG_PRED_LABELS),
            },
        }

    if attribute_mode == "pant_length":
        return {
            "strict": {
                "description": (
                    "Strict weak directional evaluation using YOLO class names. "
                    "shorts expects Short Pant. "
                    "trousers expects 3/4 Length/Cropped Pant/Full Length."
                ),
                "short_expected_pred_labels": sorted(PANT_LENGTH_SHORT_PRED_LABELS),
                "long_expected_pred_labels": sorted(PANT_LENGTH_LONG_PRED_LABELS),
            },
            "relaxed": {
                "description": (
                    "Relaxed weak directional evaluation. "
                    "shorts expects Short Pant/Mid Length. "
                    "trousers expects 3/4 Length/Cropped Pant/Full Length."
                ),
                "short_expected_pred_labels": sorted(
                    PANT_LENGTH_SHORT_PRED_LABELS | PANT_LENGTH_MID_PRED_LABELS
                ),
                "long_expected_pred_labels": sorted(PANT_LENGTH_LONG_PRED_LABELS),
            },
        }

    raise ValueError(f"Unsupported attribute_mode: {attribute_mode}")


def summarize_by_class(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_class: Dict[str, Dict[str, Any]] = {}

    tmp_pred_counts: Dict[str, Counter] = defaultdict(Counter)
    tmp_pred_group_counts: Dict[str, Counter] = defaultdict(Counter)
    tmp_total: Counter = Counter()
    tmp_correct_strict: Counter = Counter()
    tmp_correct_relaxed: Counter = Counter()
    tmp_failed: Counter = Counter()

    for row in results:
        class_name = str(row.get("class_name", "unknown"))
        tmp_total[class_name] += 1

        if row.get("error"):
            tmp_failed[class_name] += 1
            continue

        pred = row.get("prediction") or {}
        pred_name = pred.get("pred_label_name", "unknown")
        pred_group = row.get("pred_group", "unknown")

        tmp_pred_counts[class_name][pred_name] += 1
        tmp_pred_group_counts[class_name][pred_group] += 1

        if row.get("coarse_match") is True:
            tmp_correct_strict[class_name] += 1

        if row.get("coarse_match_relaxed") is True:
            tmp_correct_relaxed[class_name] += 1

    for class_name in sorted(tmp_total.keys()):
        total = tmp_total[class_name]
        correct_strict = tmp_correct_strict[class_name]
        correct_relaxed = tmp_correct_relaxed[class_name]
        failed = tmp_failed[class_name]

        by_class[class_name] = {
            "num_total": int(total),
            "num_failed": int(failed),
            "num_coarse_correct_strict": int(correct_strict),
            "coarse_accuracy_strict": float(correct_strict / total) if total > 0 else None,
            "num_coarse_correct_relaxed": int(correct_relaxed),
            "coarse_accuracy_relaxed": float(correct_relaxed / total) if total > 0 else None,
            "prediction_counts": dict(tmp_pred_counts[class_name]),
            "pred_group_counts": dict(tmp_pred_group_counts[class_name]),
        }

    return by_class


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict attributes from YOLO garment crop folder."
    )

    parser.add_argument("--crops-dir", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--label-map", required=True, type=str)
    parser.add_argument("--task", default="sleeve_length", type=str)
    parser.add_argument(
        "--attribute-mode",
        default="sleeve_length",
        choices=["sleeve_length", "pant_length"],
        help="Mode for class filtering and weak evaluation.",
    )
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--topk", default=3, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument(
        "--min-yolo-conf",
        default=0.0,
        type=float,
        help="Filter YOLO crop files by confidence parsed from filename.",
    )
    parser.add_argument(
        "--max-samples",
        default=0,
        type=int,
        help="Max selected samples. 0 means all.",
    )
    parser.add_argument("--output-jsonl", required=True, type=str)
    parser.add_argument("--output-summary", required=True, type=str)

    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    checkpoint_path = Path(args.checkpoint)
    label_map_path = Path(args.label_map)
    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)

    if not crops_dir.exists():
        raise FileNotFoundError(f"crops_dir not found: {crops_dir}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    id_to_label, _ = load_label_map(label_map_path)
    num_classes = len(id_to_label)

    model = build_attribute_classifier(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=False,
    )
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    transform = build_infer_transform(args.img_size)

    all_files = list_crop_files(crops_dir)
    related_classes = get_related_classes(args.attribute_mode)

    selected_records = []
    skipped_parse = 0
    skipped_class = 0
    skipped_conf = 0

    for crop_path in all_files:
        meta = parse_yolo_crop_filename(crop_path)
        if meta is None:
            skipped_parse += 1
            continue

        class_name = meta["class_name"]
        yolo_confidence = float(meta["yolo_confidence"])

        if class_name not in related_classes:
            skipped_class += 1
            continue

        if yolo_confidence < args.min_yolo_conf:
            skipped_conf += 1
            continue

        selected_records.append({
            "crop_path": crop_path,
            **meta,
        })

        if args.max_samples > 0 and len(selected_records) >= args.max_samples:
            break

    results: List[Dict[str, Any]] = []
    pred_counter: Counter = Counter()
    pred_group_counter: Counter = Counter()
    class_counter: Counter = Counter()
    weak_gt_group_counter: Counter = Counter()

    num_failed = 0

    num_coarse_eval_strict = 0
    num_coarse_correct_strict = 0

    num_coarse_eval_relaxed = 0
    num_coarse_correct_relaxed = 0

    for record in tqdm(selected_records, desc="Predict garment attributes"):
        crop_path = Path(record["crop_path"])
        class_name = str(record["class_name"])
        class_counter[class_name] += 1

        weak_gt_group = None
        pred_group = None
        strict_match = None
        relaxed_match = None

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
            pred_counter[pred_name] += 1

            weak_gt_group = get_weak_gt_group(args.attribute_mode, class_name)
            pred_group = get_pred_group(args.attribute_mode, pred_name)

            strict_match = coarse_match(args.attribute_mode, class_name, pred_name)
            relaxed_match = coarse_match_relaxed(args.attribute_mode, class_name, pred_name)

            weak_gt_group_counter[weak_gt_group] += 1
            pred_group_counter[pred_group] += 1

            if strict_match is not None:
                num_coarse_eval_strict += 1
                if strict_match:
                    num_coarse_correct_strict += 1

            if relaxed_match is not None:
                num_coarse_eval_relaxed += 1
                if relaxed_match:
                    num_coarse_correct_relaxed += 1

            out = {
                "task": args.task,
                "attribute_mode": args.attribute_mode,
                "crop_path": str(crop_path),
                "image_stem": record["image_stem"],
                "det_id": record["det_id"],
                "class_name": class_name,
                "yolo_confidence": record["yolo_confidence"],
                "weak_gt_group": weak_gt_group,
                "pred_group": pred_group,
                "coarse_match": strict_match,
                "coarse_match_relaxed": relaxed_match,
                "prediction": pred,
                "error": None,
            }

        except Exception as e:
            num_failed += 1
            out = {
                "task": args.task,
                "attribute_mode": args.attribute_mode,
                "crop_path": str(crop_path),
                "image_stem": record.get("image_stem"),
                "det_id": record.get("det_id"),
                "class_name": class_name,
                "yolo_confidence": record.get("yolo_confidence"),
                "weak_gt_group": weak_gt_group,
                "pred_group": pred_group,
                "coarse_match": None,
                "coarse_match_relaxed": None,
                "prediction": None,
                "error": repr(e),
            }

        results.append(out)

    by_class = summarize_by_class(results)
    eval_desc = get_coarse_eval_descriptions(args.attribute_mode)

    summary = {
        "task": args.task,
        "attribute_mode": args.attribute_mode,
        "crops_dir": str(crops_dir),
        "checkpoint": str(checkpoint_path),
        "label_map": str(label_map_path),
        "arch": args.arch,
        "img_size": args.img_size,
        "topk": args.topk,
        "device": str(device),
        "filters": {
            "min_yolo_conf": args.min_yolo_conf,
            "max_samples": args.max_samples,
            "related_classes": sorted(related_classes),
        },
        "num_all_crop_files": len(all_files),
        "num_selected": len(selected_records),
        "num_results": len(results),
        "num_failed": num_failed,
        "num_skipped_parse": skipped_parse,
        "num_skipped_class": skipped_class,
        "num_skipped_conf": skipped_conf,
        "class_counts": dict(class_counter),
        "prediction_counts": dict(pred_counter),
        "weak_gt_group_counts": dict(weak_gt_group_counter),
        "pred_group_counts": dict(pred_group_counter),
        "coarse_eval_strict": {
            **eval_desc["strict"],
            "num_eval": num_coarse_eval_strict,
            "num_correct": num_coarse_correct_strict,
            "coarse_accuracy": (
                float(num_coarse_correct_strict / num_coarse_eval_strict)
                if num_coarse_eval_strict > 0
                else None
            ),
        },
        "coarse_eval_relaxed": {
            **eval_desc["relaxed"],
            "num_eval": num_coarse_eval_relaxed,
            "num_correct": num_coarse_correct_relaxed,
            "coarse_accuracy": (
                float(num_coarse_correct_relaxed / num_coarse_eval_relaxed)
                if num_coarse_eval_relaxed > 0
                else None
            ),
        },
        "by_class": by_class,
        "output_jsonl": str(output_jsonl),
    }

    write_jsonl(results, output_jsonl)
    write_json(summary, output_summary)

    print("[OK] Garment-level attribute prediction completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
