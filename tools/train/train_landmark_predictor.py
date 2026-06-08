"""
Train garment landmark predictor.

Example:
    python tools/train_landmark_predictor.py ^
      --train-jsonl data/processed/deepfashion2_landmarks/train.jsonl ^
      --val-jsonl data/processed/deepfashion2_landmarks/validation.jsonl ^
      --output-dir outputs/landmark_predictor_resnet18 ^
      --epochs 5 ^
      --batch-size 64 ^
      --image-size 256 ^
      --lr 1e-4 ^
      --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from fashion_vision.landmarks.dataset import DeepFashion2LandmarkDataset
from fashion_vision.landmarks.model import (
    build_landmark_model,
    compute_landmark_metrics,
    masked_smooth_l1_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train garment landmark predictor.")

    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-landmarks", type=int, default=39)
    parser.add_argument("--pad-ratio", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=1)

    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Use only first N train samples for quick debugging. 0 means all.",
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=0,
        help="Use only first N val samples for quick debugging. 0 means all.",
    )

    return parser.parse_args()


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move tensor fields to device.
    """
    moved = dict(batch)

    for key in ["image", "landmarks", "valid", "visibility"]:
        if key in moved and torch.is_tensor(moved[key]):
            moved[key] = moved[key].to(device, non_blocking=True)

    return moved


def make_dataloader(
    jsonl_path: str,
    image_size: int,
    max_landmarks: int,
    pad_ratio: float,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    limit: int = 0,
) -> DataLoader:
    dataset = DeepFashion2LandmarkDataset(
        jsonl_path=jsonl_path,
        image_size=image_size,
        max_landmarks=max_landmarks,
        pad_ratio=pad_ratio,
    )

    if limit > 0:
        dataset.records = dataset.records[:limit]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_pck_005 = 0.0
    total_pck_010 = 0.0
    total_batches = 0

    start_time = time.time()

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        images = batch["image"]
        targets = batch["landmarks"]
        valid = batch["valid"]

        preds = model(images)

        loss = masked_smooth_l1_loss(
            pred_landmarks=preds,
            target_landmarks=targets,
            valid_mask=valid,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        metrics = compute_landmark_metrics(
            pred_landmarks=preds.detach(),
            target_landmarks=targets,
            valid_mask=valid,
        )

        total_loss += float(loss.item())
        total_mae += metrics["mae"]
        total_pck_005 += metrics["pck_005"]
        total_pck_010 += metrics["pck_010"]
        total_batches += 1

        if step % 50 == 0:
            print(
                f"[TRAIN] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.5f} "
                f"mae={metrics['mae']:.5f} "
                f"pck@0.05={metrics['pck_005']:.4f} "
                f"pck@0.10={metrics['pck_010']:.4f}"
            )

    elapsed = time.time() - start_time

    return {
        "loss": total_loss / max(1, total_batches),
        "mae": total_mae / max(1, total_batches),
        "pck_005": total_pck_005 / max(1, total_batches),
        "pck_010": total_pck_010 / max(1, total_batches),
        "elapsed_sec": elapsed,
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_pck_005 = 0.0
    total_pck_010 = 0.0
    total_batches = 0

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        images = batch["image"]
        targets = batch["landmarks"]
        valid = batch["valid"]

        preds = model(images)

        loss = masked_smooth_l1_loss(
            pred_landmarks=preds,
            target_landmarks=targets,
            valid_mask=valid,
        )

        metrics = compute_landmark_metrics(
            pred_landmarks=preds,
            target_landmarks=targets,
            valid_mask=valid,
        )

        total_loss += float(loss.item())
        total_mae += metrics["mae"]
        total_rmse += metrics["rmse"]
        total_pck_005 += metrics["pck_005"]
        total_pck_010 += metrics["pck_010"]
        total_batches += 1

    result = {
        "loss": total_loss / max(1, total_batches),
        "mae": total_mae / max(1, total_batches),
        "rmse": total_rmse / max(1, total_batches),
        "pck_005": total_pck_005 / max(1, total_batches),
        "pck_010": total_pck_010 / max(1, total_batches),
    }

    print(
        f"[VAL] epoch={epoch} "
        f"loss={result['loss']:.5f} "
        f"mae={result['mae']:.5f} "
        f"rmse={result['rmse']:.5f} "
        f"pck@0.05={result['pck_005']:.4f} "
        f"pck@0.10={result['pck_010']:.4f}"
    )

    return result


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }

    torch.save(checkpoint, output_path)
    print(f"[INFO] Saved checkpoint: {output_path}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    save_json(vars(args), config_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Device: {device}")

    train_loader = make_dataloader(
        jsonl_path=args.train_jsonl,
        image_size=args.image_size,
        max_landmarks=args.max_landmarks,
        pad_ratio=args.pad_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        limit=args.limit_train,
    )

    val_loader = make_dataloader(
        jsonl_path=args.val_jsonl,
        image_size=args.image_size,
        max_landmarks=args.max_landmarks,
        pad_ratio=args.pad_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        limit=args.limit_val,
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches: {len(val_loader)}")

    model = build_landmark_model(
        model_name=args.model,
        max_landmarks=args.max_landmarks,
        pretrained=not bool(args.no_pretrained),
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history = {
        "train": [],
        "val": [],
    }

    best_pck_005 = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        history["train"].append({"epoch": epoch, **train_metrics})

        val_metrics = {}
        if epoch % int(args.val_interval) == 0:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
            )
            history["val"].append({"epoch": epoch, **val_metrics})

            if val_metrics["pck_005"] > best_pck_005:
                best_pck_005 = val_metrics["pck_005"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    metrics={"train": train_metrics, "val": val_metrics},
                    output_path=output_dir / "best.pt",
                )

        if epoch % int(args.save_every) == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                metrics={"train": train_metrics, "val": val_metrics},
                output_path=output_dir / f"epoch_{epoch:03d}.pt",
            )

        save_json(history, output_dir / "history.json")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        args=args,
        metrics={"history": history},
        output_path=output_dir / "last.pt",
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Best pck@0.05: {best_pck_005:.4f}")


if __name__ == "__main__":
    main()
