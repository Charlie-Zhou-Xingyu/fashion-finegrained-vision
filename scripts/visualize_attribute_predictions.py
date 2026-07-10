#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize FashionAI attribute predictions as contact sheets.

Example:
    python scripts/visualize_attribute_predictions.py ^
      --csv outputs/p2_neckline_design_resnet18_seed2/error_cases_test.csv ^
      --output-dir outputs/p2_neckline_design_resnet18_seed2/vis_errors_test ^
      --limit 80 ^
      --cols 5 ^
      --thumb-size 220 ^
      --sort-by confidence ^
      --descending
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create contact sheets for attribute prediction CSV files."
    )
    parser.add_argument("--csv", required=True, type=str, help="Prediction/error CSV path.")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory.")
    parser.add_argument("--limit", default=80, type=int, help="Max rows to visualize. 0 means all.")
    parser.add_argument("--cols", default=5, type=int, help="Number of columns per page.")
    parser.add_argument("--thumb-size", default=220, type=int, help="Thumbnail square size.")
    parser.add_argument("--text-height", default=92, type=int, help="Text area height.")
    parser.add_argument("--page-size", default=40, type=int, help="Images per page.")
    parser.add_argument("--sort-by", default="", type=str, help="Optional column to sort by.")
    parser.add_argument("--descending", action="store_true", help="Sort descending.")
    parser.add_argument("--only-errors", action="store_true", help="Keep only correct == 0 rows if column exists.")
    parser.add_argument("--font-size", default=14, type=int, help="Font size.")
    return parser.parse_args()


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


def safe_text(value: object, max_len: int = 36) -> str:
    text = "" if pd.isna(value) else str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def open_image(image_path: str) -> Optional[Image.Image]:
    if not image_path:
        return None

    path = Path(str(image_path))
    if not path.exists():
        return None

    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:
        return None


def draw_multiline_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[tuple[str, tuple[int, int, int]]],
    font: ImageFont.ImageFont,
    line_gap: int = 19,
) -> None:
    x, y = xy
    for idx, (text, color) in enumerate(lines):
        draw.text((x, y + idx * line_gap), text, fill=color, font=font)


def make_page(
    df: pd.DataFrame,
    output_path: Path,
    cols: int,
    thumb_size: int,
    text_height: int,
    font: ImageFont.ImageFont,
) -> None:
    rows = math.ceil(len(df) / cols)
    cell_w = thumb_size
    cell_h = thumb_size + text_height

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    for local_idx, (_, row) in enumerate(df.iterrows()):
        col = local_idx % cols
        row_idx = local_idx // cols
        x = col * cell_w
        y = row_idx * cell_h

        image_path = str(row.get("image_path", ""))
        img = open_image(image_path)

        if img is None:
            draw.rectangle([x, y, x + thumb_size - 1, y + thumb_size - 1], outline=(220, 0, 0), width=2)
            draw.text((x + 8, y + 8), "IMAGE NOT FOUND", fill=(220, 0, 0), font=font)
        else:
            img.thumbnail((thumb_size, thumb_size))
            px = x + (thumb_size - img.width) // 2
            py = y + (thumb_size - img.height) // 2
            canvas.paste(img, (px, py))
            draw.rectangle([x, y, x + thumb_size - 1, y + thumb_size - 1], outline=(220, 220, 220), width=1)

        sample_id = safe_text(row.get("sample_id", ""), 28)
        gt = safe_text(row.get("gt_label_name", ""), 30)
        pred = safe_text(row.get("pred_label_name", ""), 30)
        conf = row.get("confidence", "")
        try:
            conf_text = f"{float(conf):.3f}"
        except Exception:
            conf_text = safe_text(conf, 12)

        correct = row.get("correct", "")
        try:
            is_correct = int(correct) == 1
        except Exception:
            is_correct = False

        pred_color = (0, 130, 0) if is_correct else (210, 0, 0)

        text_lines = [
            (f"ID: {sample_id}", (70, 70, 70)),
            (f"GT: {gt}", (0, 0, 0)),
            (f"Pred: {pred}", pred_color),
            (f"Conf: {conf_text}", (0, 0, 180)),
        ]

        draw_multiline_text(
            draw,
            (x + 5, y + thumb_size + 4),
            text_lines,
            font=font,
            line_gap=20,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.only_errors and "correct" in df.columns:
        df = df[df["correct"] == 0].copy()

    if args.sort_by:
        if args.sort_by not in df.columns:
            raise ValueError(f"--sort-by column not found: {args.sort_by}")
        df = df.sort_values(args.sort_by, ascending=not args.descending).copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    if len(df) == 0:
        print("[WARN] No rows to visualize.")
        return 0

    font = load_font(args.font_size)

    page_size = max(1, args.page_size)
    total_pages = math.ceil(len(df) / page_size)

    for page_idx in range(total_pages):
        start = page_idx * page_size
        end = min(len(df), start + page_size)
        page_df = df.iloc[start:end].copy()
        output_path = output_dir / f"errors_page_{page_idx:03d}.jpg"

        make_page(
            df=page_df,
            output_path=output_path,
            cols=args.cols,
            thumb_size=args.thumb_size,
            text_height=args.text_height,
            font=font,
        )

        print(f"[OK] Saved: {output_path}")

    print(f"[OK] Visualized rows: {len(df)}")
    print(f"[OK] Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
