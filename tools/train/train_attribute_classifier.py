#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train FashionAI attribute classifier.

Example:
    python tools/train/train_attribute_classifier.py ^
      --train-jsonl data\fashionai_attribute_index\sleeve_length_train.jsonl ^
      --val-jsonl data\fashionai_attribute_index\sleeve_length_val.jsonl ^
      --test-jsonl data\fashionai_attribute_index\sleeve_length_test.jsonl ^
      --label-map data\fashionai_attribute_index\label_map_sleeve_length.json ^
      --arch resnet18 ^
      --epochs 20 ^
      --batch-size 32 ^
      --img-size 224 ^
      --lr 0.0003 ^
      --output-dir outputs\p2_sleeve_length_resnet18
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.fashionai_attribute_dataset import FashionAIAttributeDataset
from models.attribute_classifier import (
    build_attribute_classifier,
    count_trainable_parameters,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FashionAI attribute classifier."
    )
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--test-jsonl", type=str, default="")
    parser.add_argument("--label-map", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help="Use inverse-frequency class weights from train set.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """
    Resolve training device.

    Args:
        device_arg: auto, cpu, cuda, or cuda device string.

    Returns:
        torch.device.
    """
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_arg)


def load_label_map(label_map_path: str) -> Dict[str, Any]:
    """
    Load label map JSON.

    Args:
        label_map_path: Label map path.

    Returns:
        Label map dictionary.
    """
    path = Path(label_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        label_map = json.load(file)

    if "num_classes" not in label_map:
        raise ValueError("label_map must contain num_classes")

    if "id_to_label" not in label_map:
        raise ValueError("label_map must contain id_to_label")

    return label_map


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train and eval transforms.

    Args:
        img_size: Input image size.

    Returns:
        Train transform and eval transform.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def collate_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch with metadata.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)

    metadata_keys = [
        "sample_id",
        "image_path",
        "image_relative_path",
        "task",
        "source_task",
        "raw_label",
        "raw_label_id",
        "label_id",
        "label_name",
        "split",
    ]

    metadata = {
        key: [item.get(key, "") for item in batch]
        for key in metadata_keys
    }

    return {
        "image": images,
        "label": labels,
        "metadata": metadata,
    }


def build_dataloaders(
    train_jsonl: str,
    val_jsonl: str,
    test_jsonl: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Build train, validation, and optional test dataloaders.

    Args:
        train_jsonl: Train JSONL path.
        val_jsonl: Validation JSONL path.
        test_jsonl: Test JSONL path.
        img_size: Image size.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        Train loader, validation loader, and test loader.
    """
    train_transform, eval_transform = build_transforms(img_size)

    train_dataset = FashionAIAttributeDataset(
        jsonl_path=train_jsonl,
        transform=train_transform,
    )
    val_dataset = FashionAIAttributeDataset(
        jsonl_path=val_jsonl,
        transform=eval_transform,
    )

    test_loader = None
    if test_jsonl:
        test_dataset = FashionAIAttributeDataset(
            jsonl_path=test_jsonl,
            transform=eval_transform,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_metadata,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_metadata,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_metadata,
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(
    train_jsonl: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from train JSONL.

    Args:
        train_jsonl: Train JSONL path.
        num_classes: Number of classes.
        device: Target device.

    Returns:
        Class weight tensor.
    """
    counts = np.zeros(num_classes, dtype=np.float64)

    with Path(train_jsonl).open("r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            label_id = int(sample["label_id"])
            counts[label_id] += 1

    if np.any(counts == 0):
        raise ValueError(f"Some classes have zero training samples: {counts.tolist()}")

    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model.
        loader: Train DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.

    Returns:
        Training metrics.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)

        all_targets.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = accuracy_score(all_targets, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": float(accuracy),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    id_to_label: Dict[str, str],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], np.ndarray]:
    """
    Evaluate model.

    Args:
        model: Model.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: Device.
        id_to_label: Mapping from label id string to label name.

    Returns:
        Metrics, prediction rows, and confusion matrix.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_confidences: List[float] = []
    prediction_rows: List[Dict[str, Any]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        metadata = batch["metadata"]

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        labels_list = labels.cpu().tolist()
        preds_list = preds.cpu().tolist()
        confidences_list = confidences.cpu().tolist()

        all_targets.extend(labels_list)
        all_preds.extend(preds_list)
        all_confidences.extend(confidences_list)

        for index in range(batch_size):
            gt_id = int(labels_list[index])
            pred_id = int(preds_list[index])

            prediction_rows.append(
                {
                    "sample_id": metadata["sample_id"][index],
                    "image_path": metadata["image_path"][index],
                    "image_relative_path": metadata["image_relative_path"][index],
                    "gt_label_id": gt_id,
                    "gt_label_name": id_to_label.get(str(gt_id), str(gt_id)),
                    "pred_label_id": pred_id,
                    "pred_label_name": id_to_label.get(str(pred_id), str(pred_id)),
                    "confidence": float(confidences_list[index]),
                    "correct": int(gt_id == pred_id),
                    "raw_label": metadata["raw_label"][index],
                    "raw_label_id": metadata["raw_label_id"][index],
                }
            )

    avg_loss = total_loss / max(total_samples, 1)

    labels_sorted = sorted(int(key) for key in id_to_label.keys())

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_precision": float(
            precision_score(
                all_targets,
                all_preds,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                all_targets,
                all_preds,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                all_targets,
                all_preds,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                all_targets,
                all_preds,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    matrix = confusion_matrix(
        all_targets,
        all_preds,
        labels=labels_sorted,
    )

    return metrics, prediction_rows, matrix


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write list of dictionaries to CSV.

    Args:
        rows: Rows.
        output_path: CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_csv(
    matrix: np.ndarray,
    id_to_label: Dict[str, str],
    output_path: Path,
) -> None:
    """
    Write confusion matrix to CSV.

    Args:
        matrix: Confusion matrix.
        id_to_label: Label id to label name mapping.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_ids = sorted(int(key) for key in id_to_label.keys())
    headers = ["gt/pred"] + [
        f"{label_id}:{id_to_label[str(label_id)]}" for label_id in label_ids
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for row_index, label_id in enumerate(label_ids):
            row_name = f"{label_id}:{id_to_label[str(label_id)]}"
            writer.writerow([row_name] + matrix[row_index].tolist())


def write_json(data: Any, output_path: Path) -> None:
    """
    Write JSON file.

    Args:
        data: JSON-serializable data.
        output_path: Output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_config_dict(args: argparse.Namespace, label_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build config dictionary for checkpoint and logging.

    Args:
        args: Parsed args.
        label_map: Label map.

    Returns:
        Config dictionary.
    """
    return {
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "test_jsonl": args.test_jsonl,
        "label_map": args.label_map,
        "arch": args.arch,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "use_class_weight": args.use_class_weight,
        "task": label_map.get("task", ""),
        "num_classes": label_map.get("num_classes", None),
    }


def main() -> int:
    """Run training."""
    args = parse_args()
    set_random_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        label_map = load_label_map(args.label_map)
        id_to_label = {
            str(key): value
            for key, value in label_map["id_to_label"].items()
        }
        num_classes = int(label_map["num_classes"])

        device = resolve_device(args.device)

        config = build_config_dict(args, label_map)
        write_json(config, output_dir / "train_config.json")

        train_loader, val_loader, test_loader = build_dataloaders(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            test_jsonl=args.test_jsonl,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = build_attribute_classifier(
            arch=args.arch,
            num_classes=num_classes,
            pretrained=args.pretrained,
        )
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.use_class_weight:
            class_weights = compute_class_weights(
                train_jsonl=args.train_jsonl,
                num_classes=num_classes,
                device=device,
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
        )

        print("[INFO] Training started.")
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Architecture: {args.arch}")
        print(f"[INFO] Pretrained: {args.pretrained}")
        print(f"[INFO] Num classes: {num_classes}")
        print(f"[INFO] Train batches: {len(train_loader)}")
        print(f"[INFO] Val batches: {len(val_loader)}")
        print(f"[INFO] Trainable parameters: {count_trainable_parameters(model)}")
        print(f"[INFO] Output dir: {output_dir}")

        best_macro_f1 = -1.0
        history: List[Dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_metrics, val_predictions, val_matrix = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                id_to_label=id_to_label,
            )

            scheduler.step()

            epoch_time = time.time() - epoch_start

            epoch_record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(epoch_record)

            print(
                "[EPOCH] "
                f"{epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_acc={train_metrics['accuracy']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_acc={val_metrics['accuracy']:.6f} "
                f"val_macro_f1={val_metrics['macro_f1']:.6f} "
                f"time={epoch_time:.2f}s"
            )

            write_json(history, output_dir / "metrics_history.json")
            write_csv(val_predictions, output_dir / "val_predictions.csv")
            write_csv(
                [row for row in val_predictions if int(row["correct"]) == 0],
                output_dir / "error_cases_val.csv",
            )
            write_confusion_matrix_csv(
                val_matrix,
                id_to_label,
                output_dir / "confusion_matrix_val.csv",
            )

            current_macro_f1 = float(val_metrics["macro_f1"])
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                save_checkpoint(
                    path=str(output_dir / "best.pt"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_macro_f1,
                    label_map=label_map,
                    config=config,
                )
                write_json(
                    {
                        "best_epoch": epoch,
                        "best_val_macro_f1": best_macro_f1,
                        "val_metrics": val_metrics,
                    },
                    output_dir / "best_metrics.json",
                )
                print(f"[OK] New best checkpoint saved at epoch {epoch}.")

        save_checkpoint(
            path=str(output_dir / "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=args.epochs,
            best_metric=best_macro_f1,
            label_map=label_map,
            config=config,
        )

        best_checkpoint_path = output_dir / "best.pt"
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[INFO] Loaded best checkpoint for test: {best_checkpoint_path}")


        if test_loader is not None:
            test_metrics, test_predictions, test_matrix = evaluate(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                id_to_label=id_to_label,
            )
            write_json(test_metrics, output_dir / "test_metrics.json")
            write_csv(test_predictions, output_dir / "test_predictions.csv")
            write_csv(
                [row for row in test_predictions if int(row["correct"]) == 0],
                output_dir / "error_cases_test.csv",
            )
            write_confusion_matrix_csv(
                test_matrix,
                id_to_label,
                output_dir / "confusion_matrix_test.csv",
            )

            print("[TEST] Final test metrics:")
            print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

        print("[OK] Training completed.")
        print(f"[OK] Best val macro-F1: {best_macro_f1:.6f}")
        print(f"[OK] Output dir: {output_dir}")

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
