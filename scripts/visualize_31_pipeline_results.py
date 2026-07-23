#!/usr/bin/env python
"""
P1.4e — Visualize combined 3.1.1 + 3.1.2 pipeline results.

Shows 3.1.1 garment instance bboxes (blue) and 3.1.2 localized region
bboxes (red) on the same image.  Proves the integrated chain works.

Usage (full312 + 3.1.1 instances, CUDA)::

    $env:VISION_REGION_BACKEND="full312"
    $env:VISION_REGION_ENABLE_REAL="true"
    $env:VISION_REGION_DEVICE="cuda"
    python scripts/visualize_31_pipeline_results.py `
        --image path/to/image.jpg `
        --output-dir outputs/vis_31_pipeline `
        --backend full312

Output:
    outputs/vis_31_pipeline/
        index.html
        summary.json
        images/
"""

from __future__ import annotations

import argparse, base64, json, os, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Colors: blue for 3.1.1 garment instances, red for 3.1.2 regions.
_GARMENT_COLOR = (255, 0, 0)   # Blue (BGR)
_REGION_COLORS = {
    "neckline": (0, 255, 0), "collar": (0, 200, 0), "lapel": (0, 180, 0),
    "sleeve": (0, 0, 255), "cuff": (0, 0, 200),
    "pocket": (255, 0, 0), "zipper": (0, 255, 255),
    "button": (255, 255, 0), "sequin": (128, 0, 128),
    "hood": (128, 128, 0), "bow": (0, 128, 128),
    "hem": (255, 0, 255), "waist": (200, 100, 200),
    "shoulder": (100, 200, 200), "epaulette": (80, 180, 180),
}


def _draw_bbox(img, bbox, label, confidence, backend, color, offset_y=0):
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    y_text = y1 - 6 - offset_y
    text = f"{label}"
    if confidence is not None:
        text += f" {confidence:.2f}"
    if backend: text += f" [{backend}]"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
    cv2.putText(img, text, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def _img_to_b64(img) -> str:
    import cv2
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("ascii")


def _build_html(entries, title="3.1.1 + 3.1.2 Pipeline Results") -> str:
    cards = []
    for e in entries:
        extra = ""
        if e.get("garment_count"): extra += f"garments={e['garment_count']} "
        cards.append(f"""
        <div class="card">
            <img src="data:image/jpeg;base64,{e['b64']}" loading="lazy"/>
            <div class="label">{e.get('filename','')}</div>
            <div class="meta">{extra}regions={e.get('region_count',0)}</div>
        </div>""")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:sans-serif;margin:20px;background:#111;color:#eee}}
h1{{color:#fff}}.gallery{{display:flex;flex-wrap:wrap;gap:12px}}
.card{{background:#222;border-radius:6px;overflow:hidden;width:400px}}
.card img{{width:100%}}.card .label{{padding:4px 10px 0;font-size:13px;font-weight:bold}}
.card .meta{{padding:2px 10px 8px;font-size:11px;color:#999}}
</style></head><body><h1>{title} — {len(entries)} samples</h1>
<div class="gallery">{''.join(cards)}</div></body></html>"""


def run(args) -> int:
    import cv2, numpy as np

    device = args.device or os.environ.get("VISION_REGION_DEVICE", "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_out = output_dir / "images"; img_out.mkdir(exist_ok=True)

    # Dry-run mode.
    if args.dry_run:
        summary = {"mode": "dry_run", "backend": args.backend, "device": device}
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(output_dir / "index.html", "w") as f:
            f.write(_build_html([], "3.1.1+3.1.2 Dry Run"))
        print(f"Dry run complete: {output_dir}")
        return 0

    # Load image(s).
    image_paths = []
    if args.image:
        image_paths = [Path(args.image)]
    elif args.images_dir:
        img_dir = Path(args.images_dir)
        image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:args.max_images]
    else:
        print("ERROR: --image or --images-dir required", file=sys.stderr); return 1

    # Resolve 3.1.1 garment instances if available.
    garment_instances_all = None
    if args.garment_instances_json:
        with open(args.garment_instances_json) as f:
            gi_data = json.load(f)
        garment_instances_all = gi_data if isinstance(gi_data, list) else gi_data.get("instances", [])

    # Load backend.
    from inference.serving.region_backend import build_region_backend, reset_region_backend
    reset_region_backend()
    backend = build_region_backend(args.backend, model_path=args.fp_model or None,
                                   device=device, confidence_threshold=args.conf)
    print(f"Backend: {backend.backend_name} device={device} enabled={backend.enabled}")
    if not backend.enabled:
        print("ERROR: backend not enabled", file=sys.stderr); return 1

    parts = args.parts.split(",") if args.parts != "all" else [
        "neckline", "collar", "lapel", "pocket", "zipper",
        "sleeve", "cuff", "button", "sequin", "hood", "bow",
    ]

    entries, total_regions, total_time = [], 0, 0.0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        # Resolve garment instances for this image.
        instances_for_img = None
        if garment_instances_all:
            instances_for_img = [
                gi for gi in garment_instances_all
                if gi.get("image") == img_path.name or "_" in str(img_path.stem)
            ]
            if not instances_for_img:
                instances_for_img = garment_instances_all  # fallback: all

        t0 = time.perf_counter()

        # Query all parts with available instances.
        regions = backend.locate_regions(
            image=img, query_all_parts=True,
            garment_instances=instances_for_img,
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        total_regions += len(regions)

        vis = img.copy()

        # Draw 3.1.1 garment instances (blue).
        if instances_for_img:
            for gi in instances_for_img:
                bbox = gi.get("bbox")
                if bbox and len(bbox) == 4:
                    iid = gi.get("instance_id", "?")
                    cat = gi.get("category", "")
                    _draw_bbox(vis, bbox, f"3.1.1 {cat}", None, iid, _GARMENT_COLOR, offset_y=18)

        # Draw 3.1.2 regions (red/colored).
        for r in regions:
            color = _REGION_COLORS.get(r["part_type"], (0, 0, 255))
            label = r["part_type"]
            if r.get("instance_id"):
                label += f"@{r['instance_id']}"
            _draw_bbox(vis, r["bbox"], label, r["confidence"], r.get("backend"), color)

        out_name = f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(img_out / out_name), vis)
        entries.append({
            "b64": _img_to_b64(vis), "filename": img_path.name,
            "garment_count": len(instances_for_img) if instances_for_img else 0,
            "region_count": len(regions),
        })
        print(f"  {img_path.name}: {len(regions)} regions ({elapsed:.1f}s)")

    summary = {"backend": args.backend, "device": device,
               "num_images": len(image_paths), "total_regions": total_regions,
               "total_time_s": round(total_time, 1)}
    with open(output_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    with open(output_dir / "index.html", "w") as f: f.write(_build_html(entries))
    print(f"\nDone: {len(entries)} images, {total_regions} regions. Output: {output_dir}")
    return 0


def main():
    p = argparse.ArgumentParser(description="P1.4e 3.1.1+3.1.2 combined viz")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--image"); g.add_argument("--images-dir")
    p.add_argument("--output-dir", default="outputs/vis_31_pipeline")
    p.add_argument("--backend", default="full312", choices=["full312", "fashionpedia"])
    p.add_argument("--device", default=None)
    p.add_argument("--fp-model", default=None)
    p.add_argument("--parts", default="all")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--garment-instances-json", help="JSON file with 3.1.1 instances")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try: return run(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if os.environ.get("DEBUG"): import traceback; traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
