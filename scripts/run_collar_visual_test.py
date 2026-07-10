"""
Visual test: collar localization on 100 sampled FashionAI collar_design images.

Samples 100 images with a fixed seed, runs the full garment pipeline on each,
extracts the collar region, and writes an HTML gallery for manual inspection.

Usage:
    python scripts/run_collar_visual_test.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig
from tools.demo.query_region_online_demo import (
    create_full_size_region_mask,
    draw_selected_overlay,
    select_best_record,
)

# ── config ─────────────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\collar_design_labels"
)
OUTPUT_ROOT = Path("outputs/visual_tests/collar_design_sample100")
SEED = 42
N_SAMPLES = 100
QUERY = "领口"
TARGET_REGION = "collar"


# ── sampling ───────────────────────────────────────────────────────────────────

def sample_images() -> list[Path]:
    all_imgs = sorted(SAMPLE_DIR.glob("*.jpg")) + sorted(SAMPLE_DIR.glob("*.png"))
    rng = random.Random(SEED)
    chosen = rng.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))
    return sorted(chosen, key=lambda p: p.name)


# ── per-image processing ───────────────────────────────────────────────────────

def _write_fail_overlay(img_bgr: Any, per_dir: Path, msg: str) -> None:
    if img_bgr is None:
        return
    out = img_bgr.copy()
    cv2.putText(out, msg[:50], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
    cv2.imwrite(str(per_dir / "overlay.jpg"), out)


def process_one(
    pipeline: GarmentPipeline,
    img_path: Path,
    per_dir: Path,
) -> dict[str, Any]:
    """Run pipeline on one image and produce original.jpg + overlay.jpg."""
    pipeline_dir = per_dir / "pipeline"
    shutil.copy(img_path, per_dir / "original.jpg")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return {"status": "error", "error": "cv2.imread returned None"}

    # Stage 1–5
    try:
        result = pipeline.run_image(str(img_path), str(pipeline_dir))
    except Exception as exc:
        _write_fail_overlay(img_bgr, per_dir, f"pipeline: {exc}")
        return {"status": "error", "error": str(exc)}

    masked_json_path = Path(result["paths"]["region_masked_crops_json"])
    if not masked_json_path.exists():
        _write_fail_overlay(img_bgr, per_dir, "no region_masked_crops.json")
        return {"status": "failed", "reason": "no crops json"}

    with masked_json_path.open(encoding="utf-8") as fh:
        masked_data = json.load(fh)

    selected, _ = select_best_record(masked_data, target_region=TARGET_REGION)

    if selected is None:
        _write_fail_overlay(img_bgr, per_dir, "no collar crop found")
        return {"status": "failed", "reason": "no collar crop"}

    bbox = selected.get("bbox_xyxy") or []
    try:
        full_mask = create_full_size_region_mask(
            img_bgr.shape, bbox, selected["mask_crop_path"]
        )
        label = f"collar [{selected.get('class_name', '')}]"
        overlay = draw_selected_overlay(img_bgr, full_mask, bbox, label)
    except Exception as exc:
        overlay = img_bgr.copy()
        cv2.putText(
            overlay, f"overlay err: {str(exc)[:40]}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 220), 1,
        )

    cv2.imwrite(str(per_dir / "overlay.jpg"), overlay)
    return {
        "status": "success",
        "bbox": bbox,
        "class_name": selected.get("class_name", ""),
        "det_id": selected.get("det_id"),
    }


# ── HTML gallery ───────────────────────────────────────────────────────────────

def generate_html(results: list[dict[str, Any]]) -> None:
    n_ok = sum(1 for r in results if r["status"] == "success")
    cards = []
    for r in results:
        ok = r["status"] == "success"
        badge_cls = "ok" if ok else "fail"
        badge_txt = f"{'✓' if ok else '✗'} {r['status']}"
        detail = f"bbox={r['bbox']}" if ok else (r.get("reason") or r.get("error") or "")
        cls_txt = r.get("class_name", "")
        cards.append(
            f'<div class="card">'
            f'<div class="imgs">'
            f'<figure><img src="per_image/{r["id"]}/original.jpg" loading="lazy">'
            f"<figcaption>original</figcaption></figure>"
            f'<figure><img src="per_image/{r["id"]}/overlay.jpg" loading="lazy">'
            f"<figcaption>overlay</figcaption></figure>"
            f"</div>"
            f'<div class="meta">'
            f'<span class="{badge_cls}">{badge_txt}</span>'
            f' <span class="cls">{cls_txt}</span><br>'
            f'<span class="detail">{detail}</span><br>'
            f'<span class="name">{r["filename"]}</span>'
            f"</div></div>"
        )

    html = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='UTF-8'>\n"
        f"<title>Collar Visual Test (n={len(results)}, seed={SEED})</title>\n"
        "<style>\n"
        "body{font-family:monospace;background:#111;color:#ccc;margin:0;padding:12px}\n"
        "h1{font-size:1.1em;margin:0 0 4px}\n"
        ".summary{color:#aaa;margin-bottom:10px;font-size:.85em}\n"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:10px}\n"
        ".card{background:#1e1e1e;border-radius:6px;padding:8px}\n"
        ".imgs{display:flex;gap:4px;margin-bottom:5px}\n"
        ".imgs figure{margin:0;flex:1;overflow:hidden;border-radius:3px}\n"
        ".imgs img{width:100%;height:190px;object-fit:cover;display:block}\n"
        ".imgs figcaption{font-size:.65em;color:#555;text-align:center;padding:1px 0}\n"
        ".meta{font-size:.7em;line-height:1.5}\n"
        ".ok{color:#4caf50;font-weight:bold}\n"
        ".fail{color:#f44336;font-weight:bold}\n"
        ".cls{color:#90caf9}\n"
        ".detail{color:#888}\n"
        ".name{color:#555;word-break:break-all}\n"
        "</style>\n</head>\n<body>\n"
        f'<h1>Collar Design Visual Test — query: "{QUERY}"</h1>\n'
        f'<div class="summary">'
        f"n={len(results)} images &nbsp;|&nbsp; seed={SEED} &nbsp;|&nbsp; "
        f"success: {n_ok}/{len(results)} ({100 * n_ok // max(len(results), 1)}%)"
        f"</div>\n"
        f'<div class="grid">{"".join(cards)}</div>\n'
        "</body>\n</html>"
    )
    (OUTPUT_ROOT / "index.html").write_text(html, encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sample = sample_images()
    manifest: dict[str, Any] = {
        "seed": SEED,
        "n_samples": len(sample),
        "query": QUERY,
        "target_region": TARGET_REGION,
        "source_dir": str(SAMPLE_DIR),
        "images": [str(p) for p in sample],
    }
    (OUTPUT_ROOT / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Sampled {len(sample)} images → {OUTPUT_ROOT}\n")

    config = GarmentPipelineConfig()
    pipeline = GarmentPipeline(config)

    results: list[dict[str, Any]] = []
    t_total = time.time()

    for idx, img_path in enumerate(sample):
        img_id = f"{idx + 1:03d}"
        per_dir = OUTPUT_ROOT / "per_image" / img_id
        per_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{img_id}/{len(sample)}] {img_path.name} ...", end=" ", flush=True)
        t0 = time.time()

        rec = process_one(pipeline, img_path, per_dir)
        elapsed = time.time() - t0

        rec.update(id=img_id, filename=img_path.name)
        (per_dir / "result.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{rec['status']}  ({elapsed:.1f}s)")
        results.append(rec)

    total = time.time() - t_total
    n_ok = sum(1 for r in results if r["status"] == "success")

    print(f"\n{'=' * 50}")
    print(
        f"Done: {n_ok}/{len(results)} success  "
        f"total={total:.0f}s  avg={total / len(results):.1f}s/img"
    )

    summary: dict[str, Any] = {
        "n_success": n_ok,
        "n_failed": len(results) - n_ok,
        "total_sec": round(total, 1),
        "avg_sec_per_image": round(total / len(results), 1),
        "results": results,
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    generate_html(results)
    print(f"Gallery:  {OUTPUT_ROOT / 'index.html'}")
    print(f"Manifest: {OUTPUT_ROOT / 'sample_manifest.json'}")


if __name__ == "__main__":
    main()
