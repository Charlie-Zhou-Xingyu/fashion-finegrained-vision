#!/usr/bin/env python3
"""
Generate interactive HTML visualization report for 3.1.2 evaluation results.

Reads ``per_result.jsonl``, generates cropped comparison panels, and writes an
interactive HTML page with per-part grids, filtering, and summary metrics.

Usage::

    conda activate fashion-demo2
    python scripts/generate_html_report.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_JSONL = PROJECT_ROOT / "data/validation/eval_v2/per_result.jsonl"
METRICS_JSON = PROJECT_ROOT / "data/validation/eval_v2/metrics.json"
PANELS_DIR = PROJECT_ROOT / "data/validation/eval_v2/panels"
HTML_OUT = PROJECT_ROOT / "data/validation/eval_v2/report.html"

N_PER_PART = 25  # max panels per part
PANEL_SIZE = 320  # max dimension of each panel image


# ── Panel rendering ──────────────────────────────────────────────────────────────

def make_comparison_panel(
    image_bgr: np.ndarray,
    gt_bbox_xyxy: List[float],
    pred_bboxes: List[List[float]],
    part: str,
    best_iou: float,
    is_hit: bool,
    backend: str,
    garment_bbox: List[float] | None = None,
    crop_offset: List[int] | None = None,
) -> np.ndarray:
    """Create a compact comparison overlay: crop around GT bbox, draw GT+Pred.

    Returns BGR image of size at most PANEL_SIZE × PANEL_SIZE.
    """
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in gt_bbox_xyxy]

    # Expand crop region around GT with 60% padding
    gw, gh = x2 - x1, y2 - y1
    pad_w, pad_h = int(gw * 0.6), int(gh * 0.6)
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(W, x2 + pad_w)
    cy2 = min(H, y2 + pad_h)

    crop = image_bgr[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]

    # Remap bboxes to crop coords
    def remap(b):
        return [b[0] - cx1, b[1] - cy1, b[2] - cx1, b[3] - cy1]

    gt_crop = remap(gt_bbox_xyxy)
    if garment_bbox:
        gb_crop = remap(garment_bbox)
    else:
        gb_crop = None

    GREEN = (0, 190, 0)
    RED = (0, 0, 230)
    BLUE = (180, 120, 0)
    LGRAY = (220, 220, 220)

    # Draw garment bbox (thin, dashed style via thinner line)
    if gb_crop:
        gx1, gy1, gx2, gy2 = [int(v) for v in gb_crop]
        cv2.rectangle(crop, (gx1, gy1), (gx2, gy2), BLUE, 1, cv2.LINE_AA)

    # Draw predictions in red
    for pb in pred_bboxes:
        px1, py1, px2, py2 = [int(v) for v in remap(pb)]
        cv2.rectangle(crop, (px1, py1), (px2, py2), RED, 2, cv2.LINE_AA)

    # Draw GT in green (on top, thicker)
    gx1_c, gy1_c, gx2_c, gy2_c = [int(v) for v in gt_crop]
    cv2.rectangle(crop, (gx1_c, gy1_c), (gx2_c, gy2_c), GREEN, 3, cv2.LINE_AA)

    # Status banner at top
    status = "HIT" if is_hit else "MISS"
    status_color = (0, 170, 0) if is_hit else (0, 50, 220)
    banner_h = 28
    banner = np.full((banner_h, cw, 3), (240, 240, 240), dtype=np.uint8)
    cv2.putText(banner, f"{part} | {status} | IoU={best_iou:.3f} | {backend}",
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

    crop = np.vstack([banner, crop])

    # Scale to fit PANEL_SIZE
    final_h, final_w = crop.shape[:2]
    scale = PANEL_SIZE / max(final_h, final_w)
    if scale < 1:
        nw, nh = int(final_w * scale), int(final_h * scale)
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # Add colored border
    border_color = (0, 180, 0) if is_hit else (50, 50, 240)
    Hf, Wf = crop.shape[:2]
    border = 3
    out = np.full((Hf + border * 2, Wf + border * 2, 3), border_color, dtype=np.uint8)
    out[border:border + Hf, border:border + Wf] = crop

    return out


# ── HTML generation ──────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { text-align: center; color: #e94560; margin-bottom: 10px; font-size: 28px; }
.subtitle { text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }

/* Summary table */
.summary-table { width: 100%; border-collapse: collapse; margin: 0 auto 30px;
                 max-width: 900px; font-size: 13px; }
.summary-table th { background: #16213e; padding: 8px 10px; text-align: left;
                    border-bottom: 2px solid #e94560; position: sticky; top: 0; z-index: 2; }
.summary-table td { padding: 5px 10px; border-bottom: 1px solid #333; }
.summary-table tr:hover td { background: #1f3050; }
.bar-cell { position: relative; min-width: 120px; }
.bar-fill { position: absolute; left: 0; top: 2px; bottom: 2px;
            background: #e94560; border-radius: 2px; opacity: 0.6;
            transition: width 0.3s; }
.bar-value { position: relative; z-index: 1; padding-left: 4px; font-size: 12px; }
.acc-good { color: #2ecc71; font-weight: bold; }
.acc-ok { color: #f39c12; }
.acc-bad { color: #e74c3c; }

/* Controls */
.controls { max-width: 900px; margin: 0 auto 15px; display: flex; gap: 8px;
            flex-wrap: wrap; justify-content: center; }
.controls button { padding: 6px 16px; border: 1px solid #444; background: #16213e;
                   color: #ccc; cursor: pointer; border-radius: 4px; font-size: 13px; }
.controls button:hover { background: #1f3050; }
.controls button.active { background: #e94560; border-color: #e94560; color: white; }

/* Part sections */
.part-section { max-width: 100%; margin: 0 auto 30px; }
.part-header { background: #16213e; padding: 12px 18px; cursor: pointer;
               border-radius: 6px 6px 0 0; display: flex; justify-content: space-between;
               align-items: center; border-left: 4px solid #e94560; }
.part-header h2 { font-size: 16px; color: #fff; }
.part-header .stats { font-size: 13px; color: #aaa; }
.part-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
             gap: 8px; padding: 12px; background: #0f0f23; border-radius: 0 0 6px 6px; }
.part-grid.collapsed { display: none; }
.panel-card { background: #1a1a2e; border-radius: 4px; overflow: hidden;
              transition: transform 0.15s; }
.panel-card:hover { transform: scale(1.02); z-index: 1; }
.panel-card img { width: 100%; height: auto; display: block; }
.panel-card .meta { padding: 4px 8px; font-size: 11px; color: #888; }
.panel-card.hit { border: 2px solid #2ecc71; }
.panel-card.miss { border: 2px solid #e74c3c; }
.no-data { color: #666; text-align: center; padding: 30px; font-style: italic; }

@media (max-width: 640px) {
  .part-grid { grid-template-columns: 1fr; }
  body { padding: 8px; }
}
"""


def _acc_class(v: float) -> str:
    if v >= 80: return "acc-good"
    if v >= 40: return "acc-ok"
    return "acc-bad"


def build_html(
    per_part_data: Dict[str, List[Dict]],
    metrics: Dict,
    panels_dir: Path,
) -> str:
    """Build the complete HTML report string."""
    iou_thresholds = metrics.get("iou_thresholds", [0.01, 0.3, 0.5])
    parts_sorted = sorted(per_part_data.keys())
    now_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')

    rel_panels = os.path.relpath(panels_dir, HTML_OUT.parent).replace("\\", "/")
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3.1.2 Region Localization — Evaluation Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>3.1.2 Region Localization Accuracy</h1>
<p class="subtitle">
  {metrics.get('num_images', '?')} images · {metrics.get('num_annotations', '?')} annotations
  · {metrics.get('num_unmapped', 0)} unmapped skipped
  · Generated {now_str}
</p>

<!-- Summary table -->
<table class="summary-table">
<thead><tr>
  <th>Part</th>
  <th>Backend</th>
  <th>N</th>
"""
    for iou_t in iou_thresholds:
        html += f"<th>IoU > {iou_t}</th>"
    html += "</tr></thead><tbody>\n"

    results_map = metrics.get("results", {})
    for part in parts_sorted:
        data = per_part_data[part]
        backend = data[0].get("backend", "?") if data else "?"
        n_total = len(data)
        html += f"<tr><td>{part}</td><td style='font-size:11px;color:#888'>{backend}</td><td>{n_total}</td>"
        for iou_t in iou_thresholds:
            key = str(iou_t)
            m = results_map.get(key, {}).get("per_part", {}).get(part, {})
            acc = m.get("accuracy", 0)
            hits = m.get("hits", 0)
            html += (
                f"<td class='bar-cell'>"
                f"<div class='bar-fill' style='width:{acc}%'></div>"
                f"<span class='bar-value {_acc_class(acc)}'>{acc:.1f}% ({hits}/{n_total})</span>"
                f"</td>"
            )
        html += "</tr>\n"

    html += "</tbody></table>\n"

    # Per-part sections
    for part in parts_sorted:
        data = per_part_data[part]
        n_total = len(data)
        n_hits = sum(1 for d in data if d.get("is_hit_iou_0.3"))
        backend = data[0].get("backend", "?") if data else "?"

        html += f"""
<div class="part-section" id="section-{part}">
<div class="part-header" onclick="togglePart('{part}')">
  <h2>{part} <span style="color:#888;font-weight:normal;font-size:13px">[{backend}]</span></h2>
  <span class="stats">{n_total} annotations · {n_hits} hits ({n_hits/max(n_total,1)*100:.1f}%) @ IoU>0.3</span>
</div>
<div class="controls" style="margin:8px 0">
  <button class="active" onclick="filterPart('{part}', 'all', this)">All</button>
  <button onclick="filterPart('{part}', 'hit', this)">Hit (IoU>0.3)</button>
  <button onclick="filterPart('{part}', 'miss', this)">Miss</button>
</div>
<div class="part-grid" id="grid-{part}">
"""
        for d in data:
            is_hit = d.get("is_hit_iou_0.3", False)
            cls = "hit" if is_hit else "miss"
            img_rel = d.get("panel_file", "")
            if img_rel:
                img_src = f"{rel_panels}/{img_rel}"
                biou = d.get("best_iou", 0)
                gt_bbox = d.get("gt_bbox", [0, 0, 0, 0])
                npred = len(d.get("pred_bboxes", []))
                html += (
                    f"<div class='panel-card {cls}' data-hit='{str(is_hit).lower()}'>"
                    f"<a href='{img_src}' target='_blank'><img src='{img_src}' loading='lazy' "
                    f"alt='{part} IoU={biou:.3f}'></a>"
                    f"<div class='meta'>IoU={biou:.3f} | "
                    f"GT=[{gt_bbox[0]:.0f}..{gt_bbox[2]:.0f}] | "
                    f"{npred} pred(s)</div>"
                    f"</div>\n"
                )
        html += "</div></div>\n"

    html += f"""
<script>
function togglePart(part) {{
  const grid = document.getElementById('grid-' + part);
  if (grid) grid.classList.toggle('collapsed');
}}
function filterPart(part, mode, btn) {{
  const grid = document.getElementById('grid-' + part);
  if (!grid) return;
  const cards = grid.querySelectorAll('.panel-card');
  cards.forEach(c => {{
    if (mode === 'all') c.style.display = '';
    else if (mode === 'hit') c.style.display = c.dataset.hit === 'true' ? '' : 'none';
    else c.style.display = c.dataset.hit === 'false' ? '' : 'none';
  }});
  btn.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}}
// Lazy load images
document.addEventListener('DOMContentLoaded', () => {{
  const imgs = document.querySelectorAll('img[loading="lazy"]');
  const observer = new IntersectionObserver((entries) => {{
    entries.forEach(e => {{ if (e.isIntersecting) {{ e.target.src = e.target.dataset.src || e.target.src; }} }});
  }});
  imgs.forEach(img => observer.observe(img));
}});
</script>
</body></html>"""

    return html


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("Loading results ...")
    with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"  {len(results)} annotations loaded")

    PANELS_DIR.mkdir(parents=True, exist_ok=True)

    # Group by part
    by_part: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_part[r["part"]].append(r)

    print(f"\nGenerating panels (up to {N_PER_PART} per part) ...")
    random.seed(42)
    total_panels = 0

    for part in sorted(by_part.keys()):
        samples = by_part[part]
        # Prioritize misses + random sample
        hits = [s for s in samples if s.get("is_hit_iou_0.3")]
        misses = [s for s in samples if not s.get("is_hit_iou_0.3")]

        selected = []
        # Take up to N_PER_PART/2 misses (or all if fewer)
        selected.extend(random.sample(misses, min(N_PER_PART // 2, len(misses))))
        # Fill remaining with hits
        remaining = N_PER_PART - len(selected)
        selected.extend(random.sample(hits, min(remaining, len(hits))))
        # If still not enough, add more misses
        sel_paths = {s["image_path"] for s in selected}
        extras = [s for s in samples if s["image_path"] not in sel_paths]
        need = N_PER_PART - len(selected)
        if need > 0 and extras:
            selected.extend(random.sample(extras, min(need, len(extras))))

        for si, rec in enumerate(selected):
            img_path = rec["image_path"]
            image = cv2.imread(img_path)
            if image is None:
                continue

            panel = make_comparison_panel(
                image,
                rec["gt_bbox"],
                rec.get("pred_bboxes", []),
                rec["part"],
                rec.get("best_iou", 0),
                rec.get("is_hit_iou_0.3", False),
                rec.get("backend", "?"),
                rec.get("garment_bbox"),
                rec.get("crop_offset"),
            )

            stem = Path(img_path).stem
            status = "H" if rec.get("is_hit_iou_0.3") else "M"
            out_name = f"{part}_{status}_iou{rec['best_iou']:.2f}_{stem}.jpg"
            out_path = PANELS_DIR / out_name
            cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 82])

            # Store relative panel path in the record
            rec["panel_file"] = out_name
            total_panels += 1

        print(f"  {part}: {len(selected)} panels")

    print(f"\n  Total: {total_panels} panels saved to {PANELS_DIR}")

    # Build HTML
    print(f"\nBuilding HTML report ...")
    html = build_html(by_part, metrics, PANELS_DIR)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"  Report saved to: {HTML_OUT}")
    print(f"\n  Open in browser: file:///{HTML_OUT.as_posix()}")


if __name__ == "__main__":
    main()
