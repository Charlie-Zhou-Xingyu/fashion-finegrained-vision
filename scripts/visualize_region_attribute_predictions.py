# scripts/visualize_region_attribute_predictions.py
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return ImageFont.load_default()


def safe_text(value: Any, max_len: int = 34) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def open_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:
        return None


def get_image_path(row: Dict[str, Any], image_field: str) -> Optional[Path]:
    if image_field == "auto":
        for key in ["crop_path", "image_crop_path", "masked_crop_path"]:
            value = row.get(key)
            if value and Path(str(value)).exists():
                return Path(str(value))
        return None

    value = row.get(image_field)
    if not value:
        return None
    return Path(str(value))


def get_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    pred = row.get("prediction")
    if isinstance(pred, dict):
        return pred
    return {}


def topk_to_text(pred: Dict[str, Any], idx: int) -> str:
    topk = pred.get("topk")
    if not isinstance(topk, list) or idx >= len(topk):
        return ""
    item = topk[idx]
    label = item.get("label_name", "")
    conf = item.get("confidence", None)
    if conf is None:
        return safe_text(label, 22)
    return f"{safe_text(label, 18)} {float(conf):.2f}"


def confidence_color(conf: float) -> tuple:
    if conf >= 0.85:
        return (0, 130, 0)
    if conf >= 0.60:
        return (230, 140, 0)
    return (200, 0, 0)


def make_page(
    rows: List[Dict[str, Any]],
    output_path: Path,
    cols: int,
    thumb_size: int,
    text_height: int,
    font: ImageFont.ImageFont,
    image_field: str,
) -> None:
    page_rows = math.ceil(len(rows) / cols)
    cell_w = thumb_size
    cell_h = thumb_size + text_height

    canvas = Image.new("RGB", (cols * cell_w, page_rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    for i, row in enumerate(rows):
        col = i % cols
        row_idx = i // cols
        x = col * cell_w
        y = row_idx * cell_h

        img_path = get_image_path(row, image_field)
        img = open_image(img_path) if img_path else None

        if img is None:
            draw.rectangle([x, y, x + thumb_size - 1, y + thumb_size - 1], outline=(220, 0, 0), width=3)
            draw.text((x + 8, y + 8), "IMAGE NOT FOUND", fill=(220, 0, 0), font=font)
            if img_path:
                draw.text((x + 8, y + 30), safe_text(str(img_path), 28), fill=(220, 0, 0), font=font)
        else:
            img.thumbnail((thumb_size, thumb_size))
            px = x + (thumb_size - img.width) // 2
            py = y + (thumb_size - img.height) // 2
            canvas.paste(img, (px, py))
            draw.rectangle([x, y, x + thumb_size - 1, y + thumb_size - 1], outline=(220, 220, 220), width=1)

        pred = get_prediction(row)
        pred_name = pred.get("pred_label_name", "N/A")
        conf = pred.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0

        color = confidence_color(conf_f)

        class_name = row.get("class_name", "")
        region = row.get("region", "")
        component = row.get("component", "")
        det_id = row.get("det_id", "")

        lines = [
            (f"Pred: {safe_text(pred_name, 24)}", color),
            (f"Conf: {conf_f:.3f}", (0, 0, 180)),
            (f"T2: {topk_to_text(pred, 1)}", (80, 80, 80)),
            (f"T3: {topk_to_text(pred, 2)}", (80, 80, 80)),
            (f"Cls: {safe_text(class_name, 24)}", (0, 0, 0)),
            (f"Reg: {safe_text(region, 10)} / {safe_text(component, 12)}", (0, 0, 0)),
            (f"Det: {det_id}", (80, 80, 80)),
        ]

        text_x = x + 5
        text_y = y + thumb_size + 4
        line_gap = 18

        for j, (text, fill) in enumerate(lines):
            draw.text((text_x, text_y + j * line_gap), text, fill=fill, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize region attribute prediction JSONL as contact sheets."
    )
    parser.add_argument("--pred-jsonl", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--image-field",
        default="crop_path",
        choices=["auto", "crop_path", "image_crop_path", "masked_crop_path"],
        help="Which image path field to visualize.",
    )
    parser.add_argument("--limit", default=100, type=int, help="0 means all.")
    parser.add_argument("--cols", default=5, type=int)
    parser.add_argument("--page-size", default=40, type=int)
    parser.add_argument("--thumb-size", default=220, type=int)
    parser.add_argument("--text-height", default=132, type=int)
    parser.add_argument("--font-size", default=14, type=int)
    parser.add_argument(
        "--sort-by-confidence",
        action="store_true",
        help="Sort by prediction confidence descending.",
    )
    parser.add_argument(
        "--filter-pred",
        default="",
        type=str,
        help="Optional filter by predicted label substring, e.g. Invisible.",
    )
    parser.add_argument(
        "--min-confidence",
        default=-1.0,
        type=float,
        help="Optional minimum confidence filter.",
    )
    parser.add_argument(
        "--max-confidence",
        default=2.0,
        type=float,
        help="Optional maximum confidence filter.",
    )
    args = parser.parse_args()

    pred_jsonl = Path(args.pred_jsonl)
    output_dir = Path(args.output_dir)

    rows = read_jsonl(pred_jsonl)

    filtered = []
    for row in rows:
        if row.get("error") is not None:
            continue
        pred = get_prediction(row)
        pred_name = str(pred.get("pred_label_name", ""))
        conf = float(pred.get("confidence", 0.0))

        if args.filter_pred and args.filter_pred.lower() not in pred_name.lower():
            continue
        if conf < args.min_confidence:
            continue
        if conf > args.max_confidence:
            continue

        filtered.append(row)

    if args.sort_by_confidence:
        filtered.sort(key=lambda r: float(get_prediction(r).get("confidence", 0.0)), reverse=True)

    if args.limit and args.limit > 0:
        filtered = filtered[: args.limit]

    if not filtered:
        print("[WARN] No rows to visualize.")
        return 0

    font = load_font(args.font_size)

    page_size = max(1, args.page_size)
    total_pages = math.ceil(len(filtered) / page_size)

    for page_idx in range(total_pages):
        start = page_idx * page_size
        end = min(len(filtered), start + page_size)
        page = filtered[start:end]

        output_path = output_dir / f"page_{page_idx:03d}.jpg"
        make_page(
            rows=page,
            output_path=output_path,
            cols=args.cols,
            thumb_size=args.thumb_size,
            text_height=args.text_height,
            font=font,
            image_field=args.image_field,
        )
        print(f"[OK] Saved: {output_path}")

    print("[OK] Visualization completed.")
    print(f"[OK] Input rows: {len(rows)}")
    print(f"[OK] Visualized rows: {len(filtered)}")
    print(f"[OK] Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
