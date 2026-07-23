#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a self-contained HTML product demo report from full 3.1.x pipeline outputs.

Reads per-image pipeline output directories produced by
``scripts/run_full_31x_pipeline.py`` and generates a single HTML file with
embedded (base64) images showing:

- Original image with YOLO garment detection overlay
- Per-garment masked region crops
- Fine-grained attribute predictions with confidence bars
- Batch summary statistics

Usage::

    conda activate fashion-demo2
    python scripts/build_31x_product_demo.py \
        --pipeline-output-dir outputs/full_31x_demo \
        --output-html outputs/full_31x_demo/product_demo.html

The output HTML is fully self-contained (no external CSS/fonts/images) and
opens in any modern browser.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence thresholds for color coding
_CONF_GREEN = 0.80
_CONF_ORANGE = 0.55

# Max dimensions for embedded images
_HERO_MAX_WIDTH = 600
_THUMB_MAX_WIDTH = 200
_REGION_THUMB_MAX_WIDTH = 100

# JPEG encoding quality (1-100)
_JPEG_QUALITY = 85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, return list of records."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _img_to_b64(img_path: Path, max_width: int = 600) -> str:
    """Load, resize, and base64-encode a JPEG image.

    Args:
        img_path: Path to the image file.
        max_width: Maximum width in pixels (aspect ratio preserved).

    Returns:
        ``data:image/jpeg;base64,...`` string, or a placeholder SVG string
        if the image cannot be loaded.
    """
    if not img_path or not img_path.exists():
        return _placeholder_svg(max_width, 200, "IMAGE NOT FOUND")

    try:
        from PIL import Image, ImageOps
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)

        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            new_size = (max_width, int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return _placeholder_svg(max_width, 200, "LOAD ERROR")


def _placeholder_svg(w: int, h: int, text: str) -> str:
    """Return a data-URI SVG placeholder."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        f'<rect width="{w}" height="{h}" fill="#1a1a2e"/>'
        f'<text x="{w//2}" y="{h//2}" text-anchor="middle" dy=".3em" '
        f'fill="#666" font-family="sans-serif" font-size="14">{text}</text>'
        f'</svg>'
    )
    b64 = base64.b64encode(svg.encode()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _confidence_color(score: float) -> str:
    """Return CSS color for a confidence score."""
    if score >= _CONF_GREEN:
        return "#22c55e"
    if score >= _CONF_ORANGE:
        return "#f59e0b"
    return "#ef4444"


def _confidence_bar(score: float) -> str:
    """Return an inline CSS bar representing confidence."""
    color = _confidence_color(score)
    pct = int(score * 100)
    return (
        f'<span style="display:inline-block;width:80px;height:8px;'
        f'background:#1e293b;border-radius:4px;vertical-align:middle;margin:0 6px;">'
        f'<span style="display:block;width:{pct}%;height:100%;'
        f'background:{color};border-radius:4px;"></span></span>'
    )


def _truncate(text: str, max_len: int = 30) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_FINE_TO_COARSE: dict[str, str] = {
    "short sleeve top": "top", "long sleeve top": "top",
    "vest": "top", "sling": "top",
    "short sleeve outwear": "outerwear", "long sleeve outwear": "outerwear",
    "shorts": "pants", "trousers": "pants",
    "skirt": "skirt",
    "short sleeve dress": "dress", "long sleeve dress": "dress",
    "vest dress": "dress", "sling dress": "dress",
}


def _infer_coarse(fine: str) -> str:
    return _FINE_TO_COARSE.get(fine, "unknown")


def _build_data_model(pipeline_root: Path) -> list[dict[str, Any]]:
    """Scan pipeline output directories and build unified data model.

    Returns a list of image-section dicts, each containing:
    - image_name, image_stem
    - hero_b64: YOLO overlay visualization as base64
    - timing_s: total pipeline time
    - garments: list of per-garment dicts with crops, attributes
    """
    batch_summary = _load_json(pipeline_root / "batch_summary.json")
    per_image = batch_summary.get("per_image", []) if batch_summary else []

    image_sections = []

    for entry in per_image:
        if entry.get("status") != "success":
            continue

        img_stem = Path(entry["image"]).stem
        img_out = Path(entry["output_dir"])
        if not img_out.exists():
            continue

        # YOLO visualization (hero image).
        yolo_vis_dir = img_out / "01_yolo" / "visualizations"
        hero_b64 = ""
        if yolo_vis_dir.exists():
            vis_files = sorted(yolo_vis_dir.glob("*.jpg"))
            if vis_files:
                hero_b64 = _img_to_b64(vis_files[0], _HERO_MAX_WIDTH)

        # SAM overlay (fallback hero).
        if not hero_b64:
            sam_overlay_dir = img_out / "02_samhq" / "overlays"
            if sam_overlay_dir.exists():
                overlay_files = sorted(sam_overlay_dir.glob("*.jpg"))
                if overlay_files:
                    hero_b64 = _img_to_b64(overlay_files[0], _HERO_MAX_WIDTH)

        # Region crops index.
        region_json = _load_json(img_out / "04_region_crops" / "region_crops.json")
        crop_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if region_json:
            for crop in region_json.get("crops", []):
                image_path = str(crop.get("image_path", "") or "")
                image_stem_crop = Path(image_path).stem if image_path else ""
                raw_det_id = str(crop.get("det_id", ""))
                instance_key = (
                    f"{image_stem_crop}__det{raw_det_id}"
                    if image_stem_crop
                    else raw_det_id
                )
                crop_index[instance_key].append(crop)

        # Masked crops: build an image lookup from region_masked_crops.json.
        masked_json = _load_json(img_out / "05_region_masked_crops" / "region_masked_crops.json")
        masked_lookup: dict[str, list[str]] = defaultdict(list)
        if masked_json:
            for mc in masked_json.get("crops", []):
                det_id = str(mc.get("det_id", ""))
                mcp = mc.get("masked_crop_path", "")
                if mcp:
                    masked_lookup[det_id].append(str(mcp))

        # Attribute predictions.
        attr_records = _load_jsonl(img_out / "06_attributes" / "predictions.jsonl")
        attr_index: dict[str, dict[str, Any]] = {}
        for rec in attr_records:
            det_id = rec.get("det_id", "")
            attr_index[det_id] = rec

        # YOLO detections for class names.
        detections_json = _load_json(img_out / "01_yolo" / "detections.json")
        det_lookup: dict[str, dict[str, Any]] = {}
        if detections_json:
            for img_rec in detections_json.get("images", []):
                for det in img_rec.get("detections", []):
                    did = str(det.get("instance_id", ""))
                    det_lookup[did] = det

        # Build garment records.
        garments = []
        # Collect unique instance keys from attributes + crops.
        all_keys = set(attr_index.keys()) | set(crop_index.keys())
        for instance_key in sorted(all_keys):
            attr_rec = attr_index.get(instance_key, {})
            raw_det_id = instance_key.split("__det")[-1] if "__det" in instance_key else instance_key
            det_info = det_lookup.get(raw_det_id, {})

            fine_class = attr_rec.get("fine_class_name") or det_info.get("class_name", "")
            coarse_class = attr_rec.get("coarse_class_name") or _infer_coarse(fine_class)
            confidence = det_info.get("confidence", 0)

            # Best crop image.
            crops = crop_index.get(instance_key, [])
            crop_b64 = ""
            # Priority: masked_crop -> image_crop -> expanded_crop -> crop
            for priority_key in ("masked_crop_path", "image_crop_path",
                                  "expanded_crop_path", "crop_path"):
                for c in crops:
                    val = c.get(priority_key, "")
                    if val:
                        crop_path = Path(str(val))
                        if crop_path.exists():
                            crop_b64 = _img_to_b64(crop_path, _THUMB_MAX_WIDTH)
                            break
                if crop_b64:
                    break

            # Region crop thumbnails (one per region type).
            region_thumbs = []
            seen_regions = set()
            for c in crops:
                region = c.get("region", "")
                if region in seen_regions:
                    continue
                seen_regions.add(region)
                for key in ("masked_crop_path", "image_crop_path", "crop_path"):
                    val = c.get(key, "")
                    if val:
                        p = Path(str(val))
                        if p.exists():
                            region_thumbs.append({
                                "region": region,
                                "b64": _img_to_b64(p, _REGION_THUMB_MAX_WIDTH),
                            })
                            break

            # Attribute predictions.
            attrs = attr_rec.get("attributes", {}) or {}
            attr_rows = []
            for task_name, task_pred in attrs.items():
                label = task_pred.get("label", "?")
                score = float(task_pred.get("score", 0))
                topk = task_pred.get("topk", [])
                alts = [
                    f'{t.get("label","")}({float(t.get("score",0)):.2f})'
                    for t in topk[1:3]
                ]
                attr_rows.append({
                    "task": task_name,
                    "label": label,
                    "score": score,
                    "score_pct": int(score * 100),
                    "color": _confidence_color(score),
                    "bar": _confidence_bar(score),
                    "alts": " | ".join(alts) if alts else "",
                })

            if crop_b64 or attr_rows:
                garments.append({
                    "instance_key": instance_key,
                    "fine_class": fine_class,
                    "coarse_class": coarse_class,
                    "confidence": confidence,
                    "crop_b64": crop_b64,
                    "region_thumbs": region_thumbs,
                    "attributes": attr_rows,
                    "num_attrs": len(attr_rows),
                })

        image_sections.append({
            "image_name": entry["image"],
            "image_stem": img_stem,
            "hero_b64": hero_b64,
            "timing_s": entry.get("elapsed_s", 0),
            "garments": garments,
            "num_garments": len(garments),
        })

    return image_sections


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CSS = r"""
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
  background:#0b1120;color:#e2e8f0;line-height:1.5;min-height:100vh
}
.container{max-width:1440px;margin:0 auto;padding:20px 24px}

/* Header */
.header{
  background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
  border-bottom:2px solid #00b4d8;padding:32px 24px;margin-bottom:24px;
  border-radius:0 0 16px 16px
}
.header h1{font-size:28px;font-weight:700;color:#fff;margin-bottom:6px}
.header h1 span{color:#00b4d8}
.header .subtitle{font-size:14px;color:#94a3b8}

/* Summary bar */
.summary-bar{
  display:flex;flex-wrap:wrap;gap:16px;margin-bottom:32px
}
.stat-card{
  flex:1;min-width:160px;background:#1a2736;border:1px solid #2a3f5f;
  border-radius:10px;padding:20px;text-align:center
}
.stat-card .value{font-size:32px;font-weight:700;color:#fff}
.stat-card .label{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px}
.stat-card.accent{border-color:#00b4d8}
.stat-card.accent .value{color:#00b4d8}

/* Image section */
.image-section{
  background:#1a2736;border:1px solid #2a3f5f;border-radius:12px;
  margin-bottom:28px;overflow:hidden
}
.image-header{
  display:flex;justify-content:space-between;align-items:center;
  padding:18px 24px;background:#0f1a2e;border-bottom:1px solid #2a3f5f
}
.image-header .image-title{font-size:18px;font-weight:600;color:#fff}
.image-header .image-meta{font-size:13px;color:#64748b}
.image-header .image-meta span{color:#00b4d8;font-weight:600;margin:0 4px}

.hero-row{
  display:flex;gap:20px;padding:20px 24px;border-bottom:1px solid #2a3f5f;
  align-items:flex-start
}
.hero-image{flex-shrink:0;border-radius:8px;overflow:hidden;border:1px solid #334155}
.hero-image img{display:block;max-width:100%;height:auto}
.hero-info{flex:1;min-width:200px}
.hero-info h3{font-size:16px;color:#fff;margin-bottom:10px}
.hero-info table{width:100%;border-collapse:collapse}
.hero-info td{padding:4px 8px;font-size:13px;color:#94a3b8}
.hero-info td:first-child{color:#64748b;width:80px}

/* Garment cards grid */
.garments-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(420px,1fr));
  gap:16px;padding:20px 24px
}
.garment-card{
  background:#0f1a2e;border:1px solid #2a3f5f;border-radius:10px;
  overflow:hidden;transition:border-color 0.2s
}
.garment-card:hover{border-color:#00b4d8}
.garment-card-header{
  display:flex;justify-content:space-between;align-items:center;
  padding:12px 16px;background:#0d1525;border-bottom:1px solid #2a3f5f
}
.garment-card-header .gc-title{font-size:14px;font-weight:600;color:#fff}
.garment-card-header .gc-badge{
  font-size:11px;padding:3px 10px;border-radius:12px;
  background:#1e3a5f;color:#93c5fd;font-weight:500
}
.garment-card-body{display:flex;gap:14px;padding:14px 16px}
.gc-crop{flex-shrink:0;width:200px}
.gc-crop img{border-radius:6px;max-width:100%;height:auto;border:1px solid #334155}
.gc-thumbs{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.gc-thumb{
  width:50px;height:50px;border-radius:4px;overflow:hidden;
  border:1px solid #334155;position:relative
}
.gc-thumb img{width:100%;height:100%;object-fit:cover}
.gc-thumb .region-label{
  position:absolute;bottom:0;left:0;right:0;
  font-size:8px;text-align:center;background:rgba(0,0,0,0.7);
  color:#94a3b8;padding:1px 0
}
.gc-attrs{flex:1;min-width:0}
.gc-attrs table{width:100%;border-collapse:collapse}
.gc-attrs td{padding:3px 6px;font-size:12px;border-bottom:1px solid #1a2736}
.gc-attrs .task-name{color:#64748b;width:80px;white-space:nowrap}
.gc-attrs .pred-label{color:#e2e8f0;font-weight:500}
.gc-attrs .pred-score{text-align:right;white-space:nowrap}
.gc-attrs .pred-alts{font-size:10px;color:#475569;padding-top:1px}

.no-garments{padding:24px;text-align:center;color:#64748b;font-size:14px}

/* Footer */
.footer{
  margin-top:40px;padding:20px 24px;text-align:center;
  color:#475569;font-size:12px;border-top:1px solid #1e293b
}
.footer span{color:#00b4d8}
"""


def _build_html(image_sections: list[dict[str, Any]], title: str) -> str:
    """Generate the complete self-contained HTML document."""

    # ---- Summary statistics ----
    total_images = len(image_sections)
    total_garments = sum(s["num_garments"] for s in image_sections)
    total_attrs = sum(
        g["num_attrs"] for s in image_sections for g in s["garments"]
    )
    avg_time = (
        sum(s["timing_s"] for s in image_sections) / total_images
        if total_images else 0
    )
    images_with_attrs = sum(
        1 for s in image_sections
        if any(g["num_attrs"] > 0 for g in s["garments"])
    )

    # Coarse class distribution.
    coarse_counts: dict[str, int] = defaultdict(int)
    for s in image_sections:
        for g in s["garments"]:
            coarse_counts[g["coarse_class"]] += 1

    # ---- Build summary bar ----
    stat_cards = f"""
    <div class="stat-card accent"><div class="value">{total_images}</div><div class="label">Images</div></div>
    <div class="stat-card"><div class="value">{total_garments}</div><div class="label">Garments Detected</div></div>
    <div class="stat-card"><div class="value">{total_attrs}</div><div class="label">Attribute Predictions</div></div>
    <div class="stat-card"><div class="value">{images_with_attrs}/{total_images}</div><div class="label">Images with Attributes</div></div>
    <div class="stat-card"><div class="value">{avg_time:.1f}s</div><div class="label">Avg Time / Image</div></div>
    """

    # Class distribution.
    dist_parts = " · ".join(
        f'{c}: <span style="color:#00b4d8;font-weight:600">{n}</span>'
        for c, n in sorted(coarse_counts.items())
    )

    # ---- Build image sections ----
    sections_html = []
    for idx, section in enumerate(image_sections):
        # Garment cards.
        cards_html = []
        for g in section["garments"]:
            # Region thumbnails.
            thumbs_html = ""
            if g["region_thumbs"]:
                thumb_divs = []
                for rt in g["region_thumbs"]:
                    thumb_divs.append(
                        f'<div class="gc-thumb">'
                        f'<img src="{rt["b64"]}" loading="lazy" alt="{rt["region"]}">'
                        f'<div class="region-label">{_truncate(rt["region"], 10)}</div>'
                        f'</div>'
                    )
                thumbs_html = f'<div class="gc-thumbs">{"".join(thumb_divs)}</div>'

            # Attribute rows.
            attr_rows_html = ""
            if g["attributes"]:
                rows = []
                for a in g["attributes"]:
                    task_short = a["task"].replace("_design", "").replace("_length", "_len")
                    rows.append(
                        f'<tr>'
                        f'<td class="task-name">{task_short}</td>'
                        f'<td class="pred-label">{a["label"]}</td>'
                        f'<td class="pred-score">'
                        f'<span style="color:{a["color"]};font-weight:600">{a["score"]:.3f}</span>'
                        f'{a["bar"]}{a["score_pct"]}%</td>'
                        f'</tr>'
                    )
                    if a["alts"]:
                        rows.append(
                            f'<tr><td></td><td class="pred-alts" colspan="2">'
                            f'alt: {a["alts"]}</td></tr>'
                        )
                attr_rows_html = f'<table>{"".join(rows)}</table>'

            # Build card.
            cards_html.append(f"""
            <div class="garment-card">
              <div class="garment-card-header">
                <span class="gc-title">{g["fine_class"]} → {g["coarse_class"]}</span>
                <span class="gc-badge">{g["num_attrs"]} attributes</span>
              </div>
              <div class="garment-card-body">
                <div class="gc-crop">
                  <img src="{g["crop_b64"] or _placeholder_svg(200,200,'NO CROP')}" loading="lazy" alt="crop">
                  {thumbs_html}
                </div>
                <div class="gc-attrs">{attr_rows_html or '<span style="color:#64748b;font-size:12px">No attributes predicted</span>'}</div>
              </div>
            </div>""")

        # Build image section.
        garments_html = "".join(cards_html) if cards_html else '<div class="no-garments">No garments detected</div>'
        sections_html.append(f"""
        <div class="image-section">
          <div class="image-header">
            <div class="image-title">📷 {section["image_name"]}</div>
            <div class="image-meta">
              <span>{section["num_garments"]}</span> garments · <span>{section["timing_s"]:.1f}s</span>
            </div>
          </div>
          <div class="hero-row">
            <div class="hero-image">
              <img src="{section["hero_b64"] or _placeholder_svg(400,400,'NO IMAGE')}" loading="lazy" alt="hero">
            </div>
            <div class="hero-info">
              <h3>Pipeline Analysis</h3>
              <table>
                <tr><td>Garments</td><td style="color:#e2e8f0">{section["num_garments"]} detected</td></tr>
                <tr><td>Attributes</td><td style="color:#e2e8f0">{sum(g["num_attrs"] for g in section["garments"])} predicted</td></tr>
                <tr><td>Time</td><td style="color:#e2e8f0">{section["timing_s"]:.1f}s total</td></tr>
                <tr><td>Pipeline</td><td style="color:#22c55e;font-size:11px">
                  ✓ YOLO → SAM-HQ → Landmark → Crops → Attributes (6 stages)
                </td></tr>
              </table>
            </div>
          </div>
          <div class="garments-grid">{garments_html}</div>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="header">
  <h1>🛍️ Fashion <span>Fine-Grained Vision</span> — Product Demo</h1>
  <div class="subtitle">
    Full 6‑stage pipeline: YOLO garment detection → SAM‑HQ segmentation → Landmark prediction → Region crops → Masked crops → Attribute inference
  </div>
  <div class="subtitle" style="margin-top:8px">
    8 fine-grained attribute classifiers: neckline, collar, neck, lapel, sleeve, coat, pant, skirt
    &nbsp;|&nbsp; Category distribution: {dist_parts}
  </div>
</div>

<div class="container">
  <div class="summary-bar">{stat_cards}</div>
  {"".join(sections_html)}
</div>

<div class="footer">
  Generated by <span>scripts/build_31x_product_demo.py</span> · Fashion Fine-Grained Vision System · All images embedded as base64 · Open in any browser
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build self-contained HTML product demo from 3.1.x pipeline outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline-output-dir",
        type=Path,
        required=True,
        help="Root directory containing per-image pipeline outputs (from run_full_31x_pipeline.py).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML file path. Default: <pipeline-output-dir>/product_demo.html",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Fashion Fine-Grained Vision — 3.1.x Pipeline Demo",
        help="HTML page title.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images in the report (0 = all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pipeline_root = args.pipeline_output_dir.resolve()
    if not pipeline_root.is_dir():
        print(f"ERROR: Pipeline output dir not found: {pipeline_root}", file=sys.stderr)
        return 1

    output_html = args.output_html or (pipeline_root / "product_demo.html")

    print(f"Building data model from: {pipeline_root}")
    image_sections = _build_data_model(pipeline_root)

    if args.max_images > 0:
        image_sections = image_sections[:args.max_images]

    if not image_sections:
        print("ERROR: No successful image sections found.", file=sys.stderr)
        return 1

    # Stats.
    total_garments = sum(s["num_garments"] for s in image_sections)
    total_attrs = sum(
        g["num_attrs"] for s in image_sections for g in s["garments"]
    )

    print(f"  Images: {len(image_sections)}")
    print(f"  Garments: {total_garments}")
    print(f"  Attribute predictions: {total_attrs}")

    print(f"Generating HTML: {output_html}")
    html = _build_html(image_sections, args.title)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    file_size_mb = output_html.stat().st_size / (1024 * 1024)
    print(f"[OK] Done!  File: {output_html}  ({file_size_mb:.1f} MB)")
    print(f"  Open in browser: file:///{output_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
