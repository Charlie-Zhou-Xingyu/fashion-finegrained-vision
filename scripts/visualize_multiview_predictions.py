import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_label_map(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        if "label_to_id" in data:
            label_to_id = data["label_to_id"]
            return {int(v): k for k, v in label_to_id.items()}

        if all(isinstance(v, int) for v in data.values()):
            return {int(v): k for k, v in data.items()}

        if all(str(k).isdigit() for k in data.keys()):
            return {int(k): v for k, v in data.items()}

    raise ValueError(f"Unknown label map format: {path}")


def get_label_id(row):
    if "label" in row:
        return int(row["label"])
    if "label_id" in row:
        return int(row["label_id"])
    if "raw_label_id" in row:
        return int(row["raw_label_id"])
    raise KeyError(f"No label field. keys={list(row.keys())}")


def get_sid(row):
    return row.get("multi_view_source_id") or row.get("sample_id") or Path(row["image_path"]).stem


def build_model(arch, num_classes):
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


def load_checkpoint(model, checkpoint_path, device):
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

    model.load_state_dict(state, strict=True)
    return model


def get_font(size=18):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def resize_keep_ratio(img, target_w, target_h, bg=(255, 255, 255)):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    img2 = img.resize((nw, nh), Image.BICUBIC)

    canvas = Image.new("RGB", (target_w, target_h), bg)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas.paste(img2, (x, y))
    return canvas


def draw_multiline(draw, xy, text, font, fill=(0, 0, 0), line_gap=4):
    x, y = xy
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_gap


def predict_one(model, img_path, transform, device):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred = int(prob.argmax().item())
    conf = float(prob[pred].item())
    return pred, conf, prob


def safe_label(id_to_label, idx):
    return str(id_to_label.get(int(idx), f"class_{idx}"))


def make_sample_card(sample, id_to_label, card_w=760, card_h=360):
    """
    sample:
      {
        sid, label_id, rows, view_results, fusion_pred, fusion_conf, correct
      }
    """
    font_title = get_font(20)
    font = get_font(16)
    font_small = get_font(14)

    card = Image.new("RGB", (card_w, card_h), (245, 245, 245))
    draw = ImageDraw.Draw(card)

    sid = sample["sid"]
    gt_id = sample["label_id"]
    gt_name = safe_label(id_to_label, gt_id)

    fusion_pred = sample["fusion_pred"]
    fusion_conf = sample["fusion_conf"]
    fusion_name = safe_label(id_to_label, fusion_pred)

    correct = sample["correct"]
    border_color = (0, 150, 0) if correct else (210, 30, 30)

    # border
    for i in range(4):
        draw.rectangle([i, i, card_w - 1 - i, card_h - 1 - i], outline=border_color)

    # image area
    img_w, img_h = 250, 250
    left_x = 20
    top_y = 55

    # 找 original / crop
    original_row = None
    crop_row = None
    for r in sample["rows"]:
        vt = r.get("view_type", "unknown")
        if vt == "original":
            original_row = r
        elif crop_row is None:
            crop_row = r

    if original_row is not None and Path(original_row["image_path"]).exists():
        img = Image.open(original_row["image_path"]).convert("RGB")
        img = resize_keep_ratio(img, img_w, img_h)
    else:
        img = Image.new("RGB", (img_w, img_h), (230, 230, 230))
    card.paste(img, (left_x, top_y))
    draw.text((left_x, top_y + img_h + 5), "original", font=font_small, fill=(0, 0, 0))

    crop_x = left_x + img_w + 20
    if crop_row is not None and Path(crop_row["image_path"]).exists():
        img = Image.open(crop_row["image_path"]).convert("RGB")
        img = resize_keep_ratio(img, img_w, img_h)
        crop_title = crop_row.get("view_type", "crop")
    else:
        img = Image.new("RGB", (img_w, img_h), (230, 230, 230))
        crop_title = "no_crop"
    card.paste(img, (crop_x, top_y))
    draw.text((crop_x, top_y + img_h + 5), crop_title, font=font_small, fill=(0, 0, 0))

    # text area
    text_x = crop_x + img_w + 25
    text_y = 20

    title = "CORRECT" if correct else "WRONG"
    draw.text((20, 15), f"{title} | {sid}", font=font_title, fill=border_color)

    lines = []
    lines.append(f"GT: {gt_id} | {gt_name}")
    lines.append(f"Fusion: {fusion_pred} | {fusion_name}")
    lines.append(f"Fusion conf: {fusion_conf:.3f}")
    lines.append("")

    for vt, vr in sample["view_results"].items():
        pred = vr["pred"]
        conf = vr["conf"]
        name = safe_label(id_to_label, pred)
        mark = "✓" if pred == gt_id else "x"
        lines.append(f"{vt}: {pred} | {name}")
        lines.append(f"  conf={conf:.3f} {mark}")

    draw_multiline(
        draw,
        (text_x, text_y),
        "\n".join(lines),
        font=font,
        fill=(0, 0, 0),
    )

    return card


def make_contact_sheet(samples, id_to_label, output_path, cols=2, rows=3):
    card_w, card_h = 760, 360
    gap = 16
    margin = 16

    page_w = margin * 2 + cols * card_w + (cols - 1) * gap
    page_h = margin * 2 + rows * card_h + (rows - 1) * gap

    page = Image.new("RGB", (page_w, page_h), (255, 255, 255))

    for i, sample in enumerate(samples):
        r = i // cols
        c = i % cols
        x = margin + c * (card_w + gap)
        y = margin + r * (card_h + gap)
        card = make_sample_card(sample, id_to_label, card_w=card_w, card_h=card_h)
        page.paste(card, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--samples-per-page", type=int, default=6)
    parser.add_argument("--sort-by", default="wrong_first", choices=["wrong_first", "conf_low", "original_order"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_to_label = load_label_map(args.label_map)
    num_classes = len(id_to_label)

    print(f"[INFO] device={device}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] output_dir={output_dir}")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model = build_model(args.arch, num_classes)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    rows = read_jsonl(args.test_jsonl)

    groups = defaultdict(list)
    for r in rows:
        groups[get_sid(r)].append(r)

    samples = []

    for sid, rs in groups.items():
        label_id = get_label_id(rs[0])

        probs = []
        view_results = {}

        for r in rs:
            img_path = r["image_path"]
            vt = r.get("view_type", "unknown")
            pred, conf, prob = predict_one(model, img_path, transform, device)
            probs.append(prob)

            view_results[vt] = {
                "pred": pred,
                "conf": conf,
                "pred_name": safe_label(id_to_label, pred),
                "image_path": img_path,
            }

        mean_prob = torch.stack(probs, dim=0).mean(dim=0)
        fusion_pred = int(mean_prob.argmax().item())
        fusion_conf = float(mean_prob[fusion_pred].item())

        samples.append({
            "sid": sid,
            "label_id": label_id,
            "label_name": safe_label(id_to_label, label_id),
            "rows": rs,
            "view_results": view_results,
            "fusion_pred": fusion_pred,
            "fusion_name": safe_label(id_to_label, fusion_pred),
            "fusion_conf": fusion_conf,
            "correct": fusion_pred == label_id,
        })

    # 保存详细预测 jsonl
    detail_path = output_dir / "predictions_detail.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for s in samples:
            out = {
                "sid": s["sid"],
                "gt_id": s["label_id"],
                "gt_name": s["label_name"],
                "fusion_pred": s["fusion_pred"],
                "fusion_name": s["fusion_name"],
                "fusion_conf": s["fusion_conf"],
                "correct": s["correct"],
                "view_results": s["view_results"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[OK] saved details: {detail_path}")

    correct_samples = [s for s in samples if s["correct"]]
    wrong_samples = [s for s in samples if not s["correct"]]

    print(f"[INFO] total samples={len(samples)} correct={len(correct_samples)} wrong={len(wrong_samples)}")

    # 排序
    if args.sort_by == "wrong_first":
        samples_sorted = sorted(samples, key=lambda x: (x["correct"], -x["fusion_conf"]))
    elif args.sort_by == "conf_low":
        samples_sorted = sorted(samples, key=lambda x: x["fusion_conf"])
    else:
        samples_sorted = samples

    # 分别输出 wrong / correct / all
    sets = [
        ("wrong", wrong_samples),
        ("correct", correct_samples),
        ("all", samples_sorted),
    ]

    for prefix, subset in sets:
        max_items = args.max_pages * args.samples_per_page
        subset = subset[:max_items]

        if not subset:
            continue

        pages = math.ceil(len(subset) / args.samples_per_page)
        for page_idx in range(pages):
            start = page_idx * args.samples_per_page
            end = start + args.samples_per_page
            page_samples = subset[start:end]

            out_path = output_dir / f"{prefix}_samples_page_{page_idx + 1:03d}.jpg"
            make_contact_sheet(
                page_samples,
                id_to_label,
                out_path,
                cols=2,
                rows=math.ceil(args.samples_per_page / 2),
            )
            print(f"[OK] saved: {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
