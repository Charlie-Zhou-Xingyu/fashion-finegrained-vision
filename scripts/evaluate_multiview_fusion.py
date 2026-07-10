import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class JsonlImageDataset(Dataset):
    def __init__(self, jsonl_path, transform):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = r["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        sid = r.get("multi_view_source_id") or r.get("sample_id") or Path(img_path).stem
        if "label" in r:
            label = int(r["label"])
        elif "label_id" in r:
            label = int(r["label_id"])
        elif "raw_label_id" in r:
            label = int(r["raw_label_id"])
        else:
            raise KeyError(f"No label field in row. keys={list(r.keys())}")


        return {
            "image": img,
            "label": label,
            "sid": sid,
            "view_type": r.get("view_type", "unknown"),
            "image_path": img_path,
        }


def build_model(arch, num_classes, pretrained=False):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported arch: {arch}")


def load_label_map(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        # 常见格式：{"label_to_id": {...}}
        if "label_to_id" in data:
            label_to_id = data["label_to_id"]
            id_to_label = {int(v): k for k, v in label_to_id.items()}
            return id_to_label

        # 也可能是 {"Invisible":0, ...}
        if all(isinstance(v, int) for v in data.values()):
            return {int(v): k for k, v in data.items()}

        # 也可能是 {"0":"xxx"}
        if all(str(k).isdigit() for k in data.keys()):
            return {int(k): v for k, v in data.items()}

    raise ValueError(f"Unknown label map format: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    id_to_label = load_label_map(args.label_map)
    num_classes = len(id_to_label)

    print(f"[INFO] device={device}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] checkpoint={args.checkpoint}")

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    ds = JsonlImageDataset(args.test_jsonl, tfm)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.arch, num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)

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

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    grouped_logits = defaultdict(list)
    grouped_labels = {}
    grouped_views = defaultdict(list)

    row_preds = []
    row_labels = []

    with torch.no_grad():
        for batch in dl:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().tolist()
            sids = batch["sid"]
            views = batch["view_type"]

            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()

            preds = probs.argmax(dim=1).tolist()

            for i, sid in enumerate(sids):
                grouped_logits[sid].append(probs[i])
                grouped_labels[sid] = labels[i]
                grouped_views[sid].append(views[i])

                row_preds.append(preds[i])
                row_labels.append(labels[i])

    # row-level
    row_acc = accuracy_score(row_labels, row_preds)
    row_p, row_r, row_f1, _ = precision_recall_fscore_support(
        row_labels,
        row_preds,
        average="macro",
        zero_division=0,
    )

    # sample-level fusion
    fused_preds = []
    fused_labels = []

    for sid, probs_list in grouped_logits.items():
        mean_prob = torch.stack(probs_list, dim=0).mean(dim=0)
        pred = int(mean_prob.argmax().item())
        label = int(grouped_labels[sid])

        fused_preds.append(pred)
        fused_labels.append(label)

    fused_acc = accuracy_score(fused_labels, fused_preds)
    fused_p, fused_r, fused_f1, _ = precision_recall_fscore_support(
        fused_labels,
        fused_preds,
        average="macro",
        zero_division=0,
    )

    result = {
        "row_level": {
            "num_rows": len(row_labels),
            "accuracy": row_acc,
            "macro_precision": row_p,
            "macro_recall": row_r,
            "macro_f1": row_f1,
        },
        "fusion_sample_level": {
            "num_samples": len(fused_labels),
            "accuracy": fused_acc,
            "macro_precision": fused_p,
            "macro_recall": fused_r,
            "macro_f1": fused_f1,
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[OK] saved: {args.output_json}")


if __name__ == "__main__":
    main()
