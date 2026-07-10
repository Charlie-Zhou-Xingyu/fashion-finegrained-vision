"""Smoke-test visualizer for garment_attribute_pipeline.py JSONL output.

Reads per-instance attribute predictions produced by
``src/fashion_vision/attributes/garment_attribute_pipeline.py`` and renders one
tile per garment instance.  Each tile shows the best available crop image
alongside every predicted attribute task (label + confidence score + top-k
alternatives).

Crop images are looked up from an optional ``--region-crops-json`` file using
the same instance-key formula as the pipeline
(``{image_stem}__det{raw_det_id}`` when image_path is present, else the raw
det_id).  If no region-crops JSON is provided, tiles are rendered text-only.

Usage::

    python scripts/visualize_attribute_pipeline_output.py \\
        --pred-jsonl   outputs/smoke_test_p3/predictions.jsonl \\
        --region-crops-json outputs/.../region_crops_with_expanded.json \\
        --output-dir   outputs/smoke_test_p3/vis
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIDENCE_GREEN = (0, 140, 0)
_CONFIDENCE_ORANGE = (200, 120, 0)
_CONFIDENCE_RED = (190, 0, 0)
_GRAY = (100, 100, 100)
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_LIGHT_GRAY_BG = (245, 245, 245)
_BORDER_COLOR = (200, 200, 200)


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------


def _load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Crop index from region_crops.json
# ---------------------------------------------------------------------------


def _build_crop_index(region_crops_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Build instance_key → [crop_records] index using same key formula as pipeline."""
    data = json.loads(region_crops_path.read_text(encoding="utf-8"))
    crops: list[dict[str, Any]] = data.get("crops", [])

    index: dict[str, list[dict[str, Any]]] = {}
    for crop in crops:
        image_path = str(crop.get("image_path", "") or "")
        image_stem = Path(image_path).stem if image_path else ""
        raw_det_id = str(crop.get("det_id", ""))
        instance_key = f"{image_stem}__det{raw_det_id}" if image_stem else raw_det_id

        if instance_key not in index:
            index[instance_key] = []
        index[instance_key].append(crop)

    return index


def _best_crop_path(crops: list[dict[str, Any]], root: Path | None) -> Path | None:
    """Return the best available crop image path from a list of crop records."""
    priority = ["masked_crop_path", "image_crop_path", "expanded_crop_path", "crop_path"]
    for record in crops:
        # prefer records with masked_success=True for masked_crop_path
        if record.get("masked_success") and record.get("masked_crop_path"):
            p = _resolve_path(record["masked_crop_path"], root)
            if p and p.exists():
                return p
    for record in crops:
        for key in priority:
            val = record.get(key)
            if val:
                p = _resolve_path(str(val), root)
                if p and p.exists():
                    return p
    return None


def _resolve_path(path_str: str, root: Path | None) -> Path | None:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if root is not None:
        candidate = root / p
        if candidate.exists():
            return candidate
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _confidence_color(score: float) -> tuple[int, int, int]:
    if score >= 0.80:
        return _CONFIDENCE_GREEN
    if score >= 0.55:
        return _CONFIDENCE_ORANGE
    return _CONFIDENCE_RED


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------


def _render_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    pred: dict[str, Any],
    crops: list[dict[str, Any]],
    root: Path | None,
    thumb_size: int,
    text_height: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    line_h = int(font.size) + 4 if hasattr(font, "size") else 18

    # ---- crop image ----
    crop_path = _best_crop_path(crops, root) if crops else None
    img: Image.Image | None = None
    if crop_path:
        try:
            img = Image.open(crop_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
        except Exception:
            img = None

    if img is not None:
        img.thumbnail((thumb_size, thumb_size))
        px = x + (thumb_size - img.width) // 2
        py = y + (thumb_size - img.height) // 2
        canvas.paste(img, (px, py))
    else:
        draw.rectangle(
            [x, y, x + thumb_size - 1, y + thumb_size - 1],
            fill=_LIGHT_GRAY_BG,
            outline=_CONFIDENCE_RED,
            width=2,
        )
        draw.text((x + 6, y + thumb_size // 2 - 8), "NO IMAGE", fill=_CONFIDENCE_RED, font=font)

    draw.rectangle(
        [x, y, x + thumb_size - 1, y + thumb_size - 1],
        outline=_BORDER_COLOR,
        width=1,
    )

    # ---- text panel ----
    tx = x + 4
    ty = y + thumb_size + 3

    # Instance header
    det_id = str(pred.get("det_id", ""))
    fine_cls = str(pred.get("fine_class_name", ""))
    coarse_cls = str(pred.get("coarse_class_name") or "?")
    draw.text((tx, ty), _truncate(det_id, 28), fill=_GRAY, font=small_font)
    ty += line_h - 2
    draw.text((tx, ty), _truncate(f"{fine_cls} → {coarse_cls}", 30), fill=_BLACK, font=small_font)
    ty += line_h

    # Error state
    error = pred.get("error")
    if error:
        draw.text((tx, ty), _truncate(f"ERR: {error}", 30), fill=_CONFIDENCE_RED, font=small_font)
        return

    # Per-task predictions
    attributes: dict[str, Any] = pred.get("attributes") or {}
    if not attributes:
        draw.text((tx, ty), "(no attributes)", fill=_GRAY, font=small_font)
        return

    for task_name, attr in attributes.items():
        if ty + line_h > y + thumb_size + text_height - 2:
            draw.text((tx, ty), "…", fill=_GRAY, font=small_font)
            break

        label = str(attr.get("label", "?"))
        score = float(attr.get("score", 0.0))
        color = _confidence_color(score)

        task_short = task_name.replace("_design", "").replace("_length", "_len")
        line1 = f"{_truncate(task_short, 12)}: {_truncate(label, 14)} {score:.2f}"
        draw.text((tx, ty), line1, fill=color, font=small_font)
        ty += line_h - 2

        # Top-k alternatives (top2 and top3) in gray
        topk: list[dict[str, Any]] = attr.get("topk", [])
        alts = [
            f"{_truncate(str(t.get('label', '')), 10)}({float(t.get('score', 0)):.2f})"
            for t in topk[1:3]
        ]
        if alts and ty + line_h <= y + thumb_size + text_height - 2:
            draw.text((tx + 8, ty), " ".join(alts), fill=_GRAY, font=small_font)
            ty += line_h - 2


# ---------------------------------------------------------------------------
# Page builder
# ---------------------------------------------------------------------------


def _make_page(
    rows: list[dict[str, Any]],
    crop_index: dict[str, list[dict[str, Any]]],
    root: Path | None,
    output_path: Path,
    cols: int,
    thumb_size: int,
    text_height: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    cell_w = thumb_size
    cell_h = thumb_size + text_height
    num_rows = math.ceil(len(rows) / cols)

    canvas = Image.new("RGB", (cols * cell_w, num_rows * cell_h), _WHITE)
    draw = ImageDraw.Draw(canvas)

    for i, pred in enumerate(rows):
        col = i % cols
        row_idx = i // cols
        x = col * cell_w
        y = row_idx * cell_h

        instance_key = str(pred.get("det_id", ""))
        crops = crop_index.get(instance_key, [])

        _render_tile(
            canvas=canvas,
            draw=draw,
            x=x,
            y=y,
            pred=pred,
            crops=crops,
            root=root,
            thumb_size=thumb_size,
            text_height=text_height,
            font=font,
            small_font=small_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f"[OK] Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize garment_attribute_pipeline.py JSONL output as contact sheets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred-jsonl", required=True, type=str,
                        help="JSONL output from garment_attribute_pipeline.py.")
    parser.add_argument("--region-crops-json", default=None, type=str,
                        help="region_crops.json for crop image lookup (optional).")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Directory for output JPEG contact sheets.")
    parser.add_argument("--cols", default=3, type=int, help="Tiles per row.")
    parser.add_argument("--page-size", default=12, type=int,
                        help="Instances per contact sheet page (0 = all).")
    parser.add_argument("--thumb-size", default=200, type=int,
                        help="Crop thumbnail size in pixels.")
    parser.add_argument("--text-height", default=180, type=int,
                        help="Pixels reserved for text below each thumbnail.")
    parser.add_argument("--font-size", default=13, type=int)
    args = parser.parse_args()

    pred_jsonl = Path(args.pred_jsonl)
    output_dir = Path(args.output_dir)

    if not pred_jsonl.exists():
        print(f"[ERROR] pred-jsonl not found: {pred_jsonl}")
        return 1

    preds = _read_jsonl(pred_jsonl)
    print(f"[INFO] Loaded {len(preds)} prediction records from {pred_jsonl}")

    crop_index: dict[str, list[dict[str, Any]]] = {}
    # Use cwd as the project root for resolving relative crop paths.
    # Run this script from the project root directory (where outputs/ lives).
    root: Path = Path.cwd()
    if args.region_crops_json:
        rc_path = Path(args.region_crops_json)
        if not rc_path.is_absolute():
            rc_path = root / rc_path
        if rc_path.exists():
            crop_index = _build_crop_index(rc_path)
            print(f"[INFO] Crop index built: {len(crop_index)} unique instances")
        else:
            print(f"[WARN] region-crops-json not found: {rc_path} — images will be skipped")

    font = _load_font(args.font_size)
    small_font = _load_font(max(10, args.font_size - 1))

    page_size = max(1, args.page_size) if args.page_size > 0 else len(preds)
    total_pages = math.ceil(len(preds) / page_size) if preds else 0

    if total_pages == 0:
        print("[WARN] No prediction records to visualize.")
        return 0

    for page_idx in range(total_pages):
        start = page_idx * page_size
        end = min(len(preds), start + page_size)
        page_rows = preds[start:end]

        out_path = output_dir / f"page_{page_idx:03d}.jpg"
        _make_page(
            rows=page_rows,
            crop_index=crop_index,
            root=root,
            output_path=out_path,
            cols=args.cols,
            thumb_size=args.thumb_size,
            text_height=args.text_height,
            font=font,
            small_font=small_font,
        )

    print(f"[OK] Visualization complete.")
    print(f"[OK] Instances: {len(preds)}, Pages: {total_pages}")
    print(f"[OK] Output dir: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
