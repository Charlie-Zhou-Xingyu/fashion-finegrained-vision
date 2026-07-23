#!/usr/bin/env python
"""
P1.4d — Visualize 3.1.2 region localization results.

Produces an HTML gallery with predicted bboxes overlaid on source images.
Supports full312 (Fashionpedia + DINO) and fashionpedia-only backends.

Usage (full312, CUDA)::

    $env:VISION_REGION_BACKEND="full312"
    $env:VISION_REGION_ENABLE_REAL="true"
    $env:VISION_REGION_DEVICE="cuda"
    python scripts/visualize_312_region_results.py `
        --images-dir artifacts/p13_visual_qa/annotated `
        --output-dir outputs/vis_312_regions `
        --backend full312 `
        --samples-per-part 5

Usage (CPU, single image)::

    python scripts/visualize_312_region_results.py `
        --image path/to/image.jpg `
        --output-dir outputs/vis_312_regions `
        --backend full312

Outputs:
    outputs/vis_312_regions/
        index.html       — gallery page
        summary.json     — per-part statistics
        images/           — annotated images

Does NOT require GPU or real models in dry-run mode (--dry-run).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Per-part colours (BGR for OpenCV).
_PART_COLORS = {
    "neckline": (0, 255, 0), "collar": (0, 200, 0), "lapel": (0, 180, 0),
    "sleeve": (255, 0, 0), "cuff": (200, 0, 0),
    "pocket": (0, 0, 255),
    "zipper": (255, 255, 0), "button": (0, 255, 255), "buckle": (200, 200, 0),
    "hem": (255, 0, 255), "waist": (200, 100, 200),
    "shoulder": (100, 200, 200), "epaulette": (80, 180, 180),
    "hood": (128, 128, 0), "bow": (0, 128, 128),
    "sequin": (128, 0, 128), "bead": (100, 100, 0),
    "applique": (0, 100, 100), "flower": (100, 0, 100),
    "ribbon": (150, 150, 0), "rivet": (0, 150, 150),
    "fringe": (150, 0, 150), "ruffle": (200, 50, 200),
    "tassel": (50, 200, 50), "strap": (50, 50, 200),
    "pattern": (200, 200, 100), "decoration": (100, 200, 100),
}


def _draw_bbox(img, bbox, label, confidence, backend, color):
    """Draw a single bbox with label on the image (mutates in-place)."""
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label}"
    if confidence is not None:
        text += f" {confidence:.2f}"
    if backend:
        text += f" [{backend}]"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)


def _img_to_b64(img) -> str:
    """Encode BGR numpy image to base64 JPEG for HTML embedding."""
    import cv2
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("ascii")


def _build_html(entries: list[dict], output_dir: Path) -> str:
    """Build a self-contained HTML gallery page."""
    parts = []
    for e in entries:
        parts.append(f"""
        <div class="card">
            <img src="data:image/jpeg;base64,{e['b64']}" loading="lazy"/>
            <div class="label">{e['part_type']}</div>
            <div class="meta">
                conf={e.get('confidence', 'N/A')}<br>
                backend={e.get('backend', '?')}<br>
                file={e.get('filename', '')}
            </div>
        </div>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>3.1.2 Region Visualization</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #111; color: #eee; }}
h1 {{ color: #fff; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.card {{ background: #222; border-radius: 6px; overflow: hidden; width: 320px; }}
.card img {{ width: 100%; display: block; }}
.card .label {{ padding: 6px 10px 0; font-weight: bold; font-size: 14px; }}
.card .meta {{ padding: 2px 10px 8px; font-size: 11px; color: #999; }}
</style></head><body>
<h1>3.1.2 Region Localization — {len(entries)} samples</h1>
<div class="gallery">{''.join(parts)}</div>
</body></html>"""


def run_visualization(args) -> int:
    """Main visualization logic."""
    import cv2

    # Resolve backend.
    device = args.device or os.environ.get("VISION_REGION_DEVICE", "cpu")

    if args.dry_run:
        print(f"DRY RUN: backend={args.backend} device={device}")
        print("Would visualize region predictions without running real models.")
        # Create minimal summary and HTML.
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)
        summary = {
            "backend": args.backend, "device": device,
            "mode": "dry_run", "num_visualized": 0,
            "note": "Run without --dry-run to execute real backend.",
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(output_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(_build_html([], output_dir))
        print(f"Output: {output_dir}")
        return 0

    # Real backend.
    from inference.serving.region_backend import build_region_backend, reset_region_backend
    reset_region_backend()
    backend = build_region_backend(
        args.backend,
        model_path=args.fp_model or None,
        device=device,
        confidence_threshold=args.conf,
    )
    print(f"Backend: {backend.backend_name}  device={device}  enabled={backend.enabled}")
    if not backend.enabled:
        print("ERROR: backend not enabled.", file=sys.stderr)
        return 1

    # Collect images.
    image_files: list[Path] = []
    if args.image:
        image_files = [Path(args.image)]
    elif args.images_dir:
        img_dir = Path(args.images_dir)
        image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        image_files = image_files[:args.max_images]
    else:
        print("ERROR: --image or --images-dir required.", file=sys.stderr)
        return 1

    if not image_files:
        print("No images found.")
        return 0

    # Resolve parts.
    if args.parts and args.parts != "all":
        parts_to_query = [p.strip() for p in args.parts.split(",")]
    else:
        parts_to_query = [
            "neckline", "collar", "lapel", "pocket", "zipper",
            "sleeve", "cuff", "button", "buckle", "bow",
            "sequin", "hood", "epaulette", "fringe", "ruffle",
        ]

    # Output dirs.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = output_dir / "images"
    img_out_dir.mkdir(exist_ok=True)

    entries: list[dict] = []
    per_part: dict[str, dict] = {}
    total_time = 0.0

    for part in parts_to_query:
        per_part.setdefault(part, {"num_samples": 0, "num_predictions": 0})
        count = 0

        for img_path in image_files:
            if count >= args.samples_per_part:
                break

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            t0 = time.perf_counter()
            regions = backend.locate_regions(image=img, requested_part=part)
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            per_part[part]["num_samples"] += 1
            vis_img = img.copy()

            for r in regions:
                per_part[part]["num_predictions"] += 1
                color = _PART_COLORS.get(r["part_type"], (128, 128, 128))
                _draw_bbox(vis_img, r["bbox"], r["part_type"],
                           r["confidence"], r["backend"], color)

            out_name = f"{part}_{count:03d}.jpg"
            out_path = img_out_dir / out_name
            cv2.imwrite(str(out_path), vis_img)

            entries.append({
                "b64": _img_to_b64(vis_img),
                "part_type": part,
                "filename": img_path.name,
                "confidence": regions[0]["confidence"] if regions else None,
                "backend": regions[0]["backend"] if regions else None,
                "num_regions": len(regions),
            })
            count += 1

        print(f"  {part}: {count} samples, {per_part[part]['num_predictions']} preds")

    # Summary.
    summary = {
        "backend": args.backend, "device": device,
        "samples_per_part": args.samples_per_part,
        "num_parts": len(parts_to_query),
        "num_visualized": len(entries),
        "total_time_s": round(total_time, 1),
        "parts": per_part,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # HTML gallery.
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(_build_html(entries, output_dir))

    print(f"\nDone: {len(entries)} samples, {len(parts_to_query)} parts")
    print(f"Output: {output_dir}")
    print(f"  index.html, summary.json, images/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P1.4d 3.1.2 region visualization")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--image", help="Single input image")
    g.add_argument("--images-dir", help="Directory of input images")
    parser.add_argument("--output-dir", default="outputs/vis_312_regions",
                        help="Output directory (default: outputs/vis_312_regions)")
    parser.add_argument("--backend", default="full312",
                        choices=["full312", "fashionpedia"],
                        help="Backend to use (default: full312)")
    parser.add_argument("--device", default=None,
                        help="Device override (cpu/cuda, default: $VISION_REGION_DEVICE or cpu)")
    parser.add_argument("--fp-model", help="Fashionpedia model path override")
    parser.add_argument("--parts", default="all",
                        help="Comma-separated parts or 'all' (default: all)")
    parser.add_argument("--samples-per-part", type=int, default=5,
                        help="Max samples per part (default: 5)")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Max images to load from directory (default: 100)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create empty HTML/summary without running models")
    args = parser.parse_args()

    try:
        return run_visualization(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
