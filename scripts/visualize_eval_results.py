#!/usr/bin/env python3
"""
Visualize 3.1.2 region localization accuracy — focus on prediction vs GT comparison.

Reads per_result.jsonl and produces an HTML gallery with clear overlay of
predicted boxes vs ground-truth, organized by part.

Usage::

    python scripts/visualize_eval_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGES_DIR = PROJECT_ROOT / "data/validation/to_annotate"
RESULTS_JSONL = PROJECT_ROOT / "data/validation/eval_312/per_result.jsonl"
METRICS_JSON = PROJECT_ROOT / "data/validation/eval_312/metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs/eval_312_region_accuracy"


GREEN = (80, 200, 80)
BLUE = (255, 140, 30)
RED = (60, 60, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
YELLOW = (50, 210, 230)


def draw_box(img, bbox, color, thickness=3, label=None, dashed=False):
    """Draw xyxy box with label banner at top-left corner."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    if dashed:
        for i in range(0, max(x2 - x1, y2 - y1), 12):
            cv2.line(img, (x1 + i, y1), (min(x1 + i + 6, x2), y1), color, 1)
            cv2.line(img, (x1 + i, y2), (min(x1 + i + 6, x2), y2), color, 1)
            cv2.line(img, (x1, y1 + i), (x1, min(y1 + i + 6, y2)), color, 1)
            cv2.line(img, (x2, y1 + i), (x2, min(y1 + i + 6, y2)), color, 1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)


def build_comparison_image(
    image_path: Path,
    record: dict,
    max_dim: int = 700,
) -> np.ndarray:
    """Build a clear overlay image showing GT vs prediction."""
    image = cv2.imread(str(image_path))
    if image is None:
        image = np.full((480, 640, 3), 80, dtype=np.uint8)

    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h))

    part = record["part"]
    gt_bbox = [v * scale for v in record["gt_bbox"]]
    pred_bboxes = [[v * scale for v in pb] for pb in record.get("pred_bboxes", [])]
    is_hit = record.get("is_hit", False)
    best_iou = record.get("best_iou", 0)
    backend = record.get("backend", "?")
    garment_bbox = record.get("garment_bbox")

    vis = image.copy()

    # Garment crop region — thin yellow dashed
    if garment_bbox:
        gb = [v * scale for v in garment_bbox]
        draw_box(vis, gb, YELLOW, 1, dashed=True)

    # GT box — thick green solid
    draw_box(vis, gt_bbox, GREEN, 3, "GT")

    # Pred box(es) — blue solid for hit, red dashed for miss
    for pb in pred_bboxes:
        color = BLUE if is_hit else RED
        label = f"{'HIT' if is_hit else 'MISS'} IoU={best_iou:.3f}"
        draw_box(vis, pb, color, 3, label)

    # Status bar at top
    status_text = "HIT" if is_hit else "MISS"
    bar_color = BLUE if is_hit else RED
    bar_h = 32
    bar = np.full((bar_h, vis.shape[1], 3), bar_color, dtype=np.uint8)
    cv2.putText(bar, f"{status_text} | {part} | IoU={best_iou:.3f} | {backend} | {record.get('image','')}",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # If not_detected, add "NO DETECTION" text on image
    if record.get("status") == "not_detected" and not pred_bboxes:
        cv2.putText(vis, "NO DETECTION", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)

    vis = np.vstack([bar, vis])
    return vis


def build_html(results: List[dict], metrics: dict, images_dir: Path, output_dir: Path,
               max_per_part: int = 0) -> str:
    """Build self-contained HTML gallery."""
    by_part: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_part[r["part"]].append(r)

    per_part = metrics.get("per_part", {})
    # Sort by accuracy: worst first
    part_order = sorted(by_part.keys(),
                        key=lambda p: per_part.get(p, {}).get("accuracy", 0))

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    html = []
    html.append("""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>3.1.2 Region Accuracy — Pred vs GT</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,sans-serif; background:#111; color:#ddd; }
h1 { padding:14px 24px; background:#1a1a2e; font-size:1.2em; position:sticky; top:0; z-index:10; }
.summary { display:flex; flex-wrap:wrap; gap:10px; padding:12px 24px; background:#16162a; }
.card { background:#1e1e3a; border-radius:6px; padding:8px 14px; min-width:120px; }
.card .n { font-size:0.8em; color:#999; }
.card .a { font-size:1.4em; font-weight:700; }
.card .d { font-size:0.7em; color:#777; }
.g { color:#4ecca3; } .o { color:#f9ed69; } .b { color:#f38181; }
.section { margin:0 20px 20px; }
.section h2 { padding:8px 14px; background:#1a1a3e; border-radius:6px;
  font-size:1em; margin:16px 0 8px; position:sticky; top:46px; z-index:5; }
.gallery { display:flex; flex-wrap:wrap; gap:10px; }
.gallery a { text-decoration:none; display:block; }
.gallery img { border-radius:4px; border:2px solid #333; max-width:320px; height:auto; }
.gallery img.hit { border-color:#4ecca3; }
.gallery img.miss { border-color:#f38181; }
</style></head><body>
<h1>3.1.2 Region Localization Accuracy — Pred vs GT (IoU &gt; 0 = hit)</h1>
""")

    overall = metrics.get("overall_accuracy", 0)
    acc_class = "g" if overall >= 92 else "o" if overall >= 50 else "b"
    html.append(f"""<div class="summary">
<div class="card"><div class="n">Overall</div><div class="a {acc_class}">{overall:.1f}%</div>
<div class="d">{metrics.get('num_annotations',0)} annotations</div></div>""")
    for p in part_order:
        pm = per_part.get(p, {})
        acc = pm.get("accuracy", 0)
        c = "g" if acc >= 92 else "o" if acc >= 50 else "b"
        html.append(f"""<div class="card"><div class="n">{p}</div>
<div class="a {c}">{acc:.1f}%</div><div class="d">{pm.get('hits',0)}/{pm.get('total',0)}</div></div>""")
    html.append("</div>")

    for part in part_order:
        items = by_part[part]
        # Hits first (blue), then misses (red)
        items.sort(key=lambda r: (not r.get("is_hit", False), -r.get("best_iou", 0)))
        if max_per_part > 0:
            items = items[:max_per_part]

        pm = per_part.get(part, {})
        acc = pm.get("accuracy", 0)

        html.append(f"""<div class="section">
<h2>{part} — {acc:.1f}% ({pm.get('hits',0)}/{pm.get('total',0)})</h2>
<div class="gallery">""")

        for idx, r in enumerate(items):
            img_path = images_dir / r["image"]
            vis = build_comparison_image(img_path, r)
            name = f"{part}_{r['image'].replace('.jpg','')}_{idx}.jpg"
            cv2.imwrite(str(viz_dir / name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

            cls = "hit" if r.get("is_hit") else "miss"
            html.append(f'<a href="viz/{name}" target="_blank">'
                        f'<img class="{cls}" src="viz/{name}" loading="lazy"></a>')

        html.append("</div></div>")

    html.append("</body></html>")
    return "\n".join(html)


def main():
    p = argparse.ArgumentParser(description="Visualize 3.1.2 accuracy")
    p.add_argument("--results-jsonl", default=str(RESULTS_JSONL))
    p.add_argument("--metrics-json", default=str(METRICS_JSON))
    p.add_argument("--images-dir", default=str(IMAGES_DIR))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--max-per-part", type=int, default=30)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with open(args.results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"{len(results)} results loaded")
    html = build_html(results, metrics, Path(args.images_dir), output_dir, args.max_per_part)
    hp = output_dir / "index.html"
    hp.write_text(html, encoding="utf-8")
    print(f"Done: {hp}")


if __name__ == "__main__":
    main()
