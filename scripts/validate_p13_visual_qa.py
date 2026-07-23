"""P1.3 validation — run real 3.1.1 segmentation through the QA orchestrator
on random DeepFashion2 test images and produce an HTML gallery for review.

Usage (explicitly approved heavy run — loads YOLO + SAM-HQ per image):

    python scripts/validate_p13_visual_qa.py --num-images 20 --device cuda

Output: artifacts/p13_visual_qa/index.html (+ annotated jpgs + results.json)

This is a dev-side validation tool: it draws bboxes from the SERVING response
(``meta.garment_instances_summary``), so what you see is exactly what the QA
layer exposes — no direct pipeline output is read.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGE_DIR = Path(r"D:\Aliintern\fashion-ai-data\deepfashion2\test\test\image")
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "p13_visual_qa"

QUERIES = [
    "图里有几件衣服？",
    "图中检测到了什么？",
    "有没有上衣？",
    "检测框在哪里？",
    "有没有分割结果？",
]

_CATEGORY_COLORS = {
    "top": (46, 134, 222),        # blue
    "pants": (39, 174, 96),       # green
    "skirt": (155, 89, 182),      # purple
    "outerwear": (230, 126, 34),  # orange
    "dress": (231, 76, 60),       # red
}


class CachingProvider:
    """Wrap a vision provider so each unique image runs segmentation ONCE,
    even though the orchestrator calls extract() per query."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: Dict[str, Any] = {}

    def extract(self, **kwargs: Any) -> Any:
        img = kwargs.get("image_bytes") or b""
        key = hashlib.md5(img if isinstance(img, bytes) else str(img).encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._inner.extract(**kwargs)
        return self._cache[key]


def build_orchestrator(device: str) -> Any:
    from inference.serving.attribute_service import AttributeService
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.qa_orchestrator import QaOrchestrator
    from inference.serving.rag_service import RagService
    from inference.serving.real_vision_provider import RealVisionAttributeProvider

    provider = RealVisionAttributeProvider(
        backend="fashion_vision_3_1",
        mode="segmentation_only",
        timeout_ms=300_000,  # real pipeline load+run per image; generous budget
        yolo_device=device,
        sam_device=device,
    )
    return QaOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        attribute_service=AttributeService(),
        rag_service=RagService(),
        vision_provider=CachingProvider(provider),
    )


def annotate(image_path: Path, instances: List[Dict[str, Any]], out_path: Path) -> None:
    """Draw bboxes + category labels from garment_instances_summary."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for inst in instances:
        bbox = inst.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        cat = inst.get("category", "?")
        color = _CATEGORY_COLORS.get(cat, (128, 128, 128))
        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        conf = inst.get("confidence")
        label = f"{cat} {conf:.2f}" if isinstance(conf, (int, float)) else str(cat)
        mask_tag = " +mask" if inst.get("mask_present") else ""
        draw.rectangle([x1, max(0, y1 - 16), x1 + 8 * len(label + mask_tag), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 15)), label + mask_tag, fill=(255, 255, 255))
    img.save(out_path, quality=90)


def render_html(records: List[Dict[str, Any]], out_path: Path) -> None:
    rows = []
    for rec in records:
        qa_html = "".join(
            f"<div class='qa'><b>Q:</b> {html.escape(q)}<br><b>A:</b> {html.escape(a)}</div>"
            for q, a in rec["qa"]
        )
        rows.append(f"""
        <div class="card">
          <img src="annotated/{rec['annotated']}" loading="lazy">
          <div class="info">
            <div class="fname">{html.escape(rec['image'])}
              — {rec['num_instances']} instance(s), seg {rec['latency_ms']:.0f} ms</div>
            {qa_html}
          </div>
        </div>""")
    out_path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>P1.3 Visual Instance QA — validation</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 20px; background:#fafafa; }}
 .card {{ display:flex; gap:16px; background:#fff; border:1px solid #ddd;
          border-radius:8px; padding:12px; margin-bottom:16px; }}
 .card img {{ max-width:360px; max-height:420px; object-fit:contain; }}
 .fname {{ font-weight:600; margin-bottom:8px; color:#333; }}
 .qa {{ margin:6px 0; padding:6px 8px; background:#f4f7fb; border-radius:4px; font-size:14px; }}
</style>
<h1>P1.3 Visual Instance QA — {len(records)} DeepFashion2 test images</h1>
<p>Bboxes drawn from <code>meta.garment_instances_summary</code> of the serving
response (what the QA layer actually exposes). "+mask" = mask_present.</p>
{''.join(rows)}""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise SystemExit(f"Image dir not found: {args.image_dir}")

    all_images = sorted(args.image_dir.glob("*.jpg"))
    if not all_images:
        raise SystemExit(f"No .jpg files in {args.image_dir}")
    random.seed(args.seed)
    sample = random.sample(all_images, min(args.num_images, len(all_images)))

    annotated_dir = OUTPUT_DIR / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    orch = build_orchestrator(args.device)
    records: List[Dict[str, Any]] = []

    for i, image_path in enumerate(sample):
        img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        t0 = time.perf_counter()
        qa_pairs: List[List[str]] = []
        summary: List[Dict[str, Any]] = []
        for q in QUERIES:
            r = orch.answer(query=q, image_bytes=img_b64)
            qa_pairs.append([q, r.answer])
            if not summary:
                summary = r.meta.get("garment_instances_summary", [])
        latency_ms = (time.perf_counter() - t0) * 1000

        annotated_name = f"{image_path.stem}_annotated.jpg"
        annotate(image_path, summary, annotated_dir / annotated_name)
        records.append({
            "image": image_path.name,
            "annotated": annotated_name,
            "num_instances": len(summary),
            "latency_ms": latency_ms,
            "qa": qa_pairs,
            "garment_instances_summary": summary,
        })
        print(f"[{i + 1}/{len(sample)}] {image_path.name}: "
              f"{len(summary)} instance(s), {latency_ms:.0f} ms")

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(records, OUTPUT_DIR / "index.html")
    print(f"\nDone. Open: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
