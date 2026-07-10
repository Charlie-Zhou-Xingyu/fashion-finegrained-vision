import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_label_map(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "id_to_label" in data:
        id_to_label = {int(k): v for k, v in data["id_to_label"].items()}
    elif "idx_to_label" in data:
        id_to_label = {int(k): v for k, v in data["idx_to_label"].items()}
    elif "classes" in data:
        id_to_label = {i: v for i, v in enumerate(data["classes"])}
    else:
        # fallback: label -> id
        id_to_label = {int(v): k for k, v in data.items() if isinstance(v, int)}

    label_to_id = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id


class JsonlImageDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        img_size: int,
    ):
        self.rows = rows
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p = Path(str(r["image_path"]))
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        y = int(r["label_id"])
        return x, y, idx


def build_model(arch: str, num_classes: int):
    arch = arch.lower()

    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported arch: {arch}")


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    if missing:
        print("[WARN] Missing keys sample:", missing[:10])
    if unexpected:
        print("[WARN] Unexpected keys sample:", unexpected[:10])

    return model


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "weighted_f1": float(wf1),
    }


def save_confusion_matrix(
    path: Path,
    y_true: List[int],
    y_pred: List[int],
    id_to_label: Dict[int, str],
):
    labels = sorted(id_to_label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        header = ["gt/pred"] + [f"{i}:{id_to_label[i]}" for i in labels]
        writer.writerow(header)
        for i, row in zip(labels, cm):
            writer.writerow([f"{i}:{id_to_label[i]}"] + list(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--view-filter",
        default="",
        help="Optional filter by row['view_type'], e.g. original, expanded_collar, upper_crop",
    )
    parser.add_argument("--save-predictions", action="store_true")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_to_label, label_to_id = load_label_map(Path(args.label_map))
    num_classes = len(id_to_label)

    rows = read_jsonl(Path(args.jsonl))
    original_n = len(rows)

    if args.view_filter:
        rows = [r for r in rows if str(r.get("view_type", "")) == args.view_filter]

    print(f"[INFO] jsonl: {args.jsonl}")
    print(f"[INFO] original rows: {original_n}")
    print(f"[INFO] eval rows: {len(rows)}")
    print(f"[INFO] view_filter: {args.view_filter or '<none>'}")
    print(f"[INFO] num_classes: {num_classes}")
    print(f"[INFO] device: {device}")

    ds = JsonlImageDataset(rows, img_size=args.img_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.arch, num_classes=num_classes)
    model = load_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    pred_rows = []

    with torch.no_grad():
        for x, y, idxs in dl:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

            if args.save_predictions:
                for j, idx in enumerate(idxs.cpu().tolist()):
                    r = dict(rows[idx])
                    pi = int(pred[j].cpu().item())
                    gi = int(y[j].cpu().item())
                    r["pred_label_id"] = pi
                    r["pred_label_name"] = id_to_label.get(pi, str(pi))
                    r["gt_label_id"] = gi
                    r["gt_label_name"] = id_to_label.get(gi, str(gi))
                    r["confidence"] = float(conf[j].cpu().item())
                    r["correct"] = bool(pi == gi)
                    pred_rows.append(r)

    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "jsonl": args.jsonl,
        "checkpoint": args.checkpoint,
        "label_map": args.label_map,
        "arch": args.arch,
        "img_size": args.img_size,
        "view_filter": args.view_filter,
        "num_rows": len(rows),
        "num_classes": num_classes,
    })

    print("[RESULT]")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        y_true,
        y_pred,
        id_to_label,
    )

    if args.save_predictions:
        with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for r in pred_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Saved metrics: {output_dir / 'metrics.json'}")
    print(f"[OK] Saved confusion matrix: {output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
