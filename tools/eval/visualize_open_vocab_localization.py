#!/usr/bin/env python
"""
Visual evaluation of Phase 1/2 open-vocabulary localization (3.1.2).

Randomly samples images from a directory, runs locate_region() for each
Chinese query, then saves:
  - one visualization PNG per (image, query) pair
  - outputs/index.html — browsable HTML with tabs per query
  - outputs/results.jsonl — one JSON record per (image, query)

GPU is recommended but not required; the script falls back to CPU with a
warning. Fast-path queries (neckline/cuff/hem) that need landmark data will
fail gracefully and appear as "failed" in the report.

Usage:
    python tools/eval/visualize_open_vocab_localization.py \
        --image-dir "D:\\..." \
        --output-dir outputs/vis_10 \
        --sample-size 10 --seed 42

    # explicit queries:
    python tools/eval/visualize_open_vocab_localization.py \
        --image-dir "D:\\..." \
        --output-dir outputs/vis_100 \
        --sample-size 100 --seed 42 \
        --queries "这件外套的口袋" "这件外套的拉链" "这件外套的扣子"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── project root wiring ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from fashion_vision.localization.region_localization_router import locate_region  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

DEFAULT_QUERIES: list[str] = [
    "这件外套的口袋",
    "这件外套的拉链",
    "这件外套的扣子",
    "这件外套的腰带",
    "这件外套的图案",
    "这件外套的领口",
    "这件外套的袖口",
    "这件外套的下摆",
]

# BGR colours
_CLR_OK = (0, 210, 80)
_CLR_FAIL = (60, 60, 220)
_PANEL_BG = (24, 24, 36)  # dark navy
_PANEL_FG_OK = (170, 240, 170)
_PANEL_FG_FAIL = (240, 170, 170)


# ── drawing helpers ────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try CJK fonts first (for query strings), then ASCII fallbacks."""
    for path, kwargs in [
        ("C:/Windows/Fonts/msyh.ttc", {"index": 0}),
        ("C:/Windows/Fonts/simsun.ttc", {"index": 0}),
        ("C:/Windows/Fonts/simhei.ttf", {}),
        ("C:/Windows/Fonts/arial.ttf", {}),
        ("C:/Windows/Fonts/consola.ttf", {}),
        ("C:/Windows/Fonts/cour.ttf", {}),
    ]:
        try:
            return ImageFont.truetype(path, size, **kwargs)
        except Exception:
            pass
    return ImageFont.load_default()


def _overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    colour_bgr: tuple[int, int, int] = (0, 140, 255),
    alpha: float = 0.35,
) -> np.ndarray:
    """Blend semi-transparent colour over mask-positive pixels."""
    out = image_bgr.copy()
    colour_layer = np.zeros_like(out)
    colour_layer[:] = colour_bgr
    blended = cv2.addWeighted(colour_layer, alpha, out, 1.0 - alpha, 0)
    out[mask > 0] = blended[mask > 0]
    return out


def _build_vis(
    image_bgr: np.ndarray,
    result: dict[str, Any],
    query: str,
    img_name: str,
    font: Any,
    max_width: int = 640,
) -> Image.Image:
    """
    Build a combined PIL Image: [photo with overlays] stacked above [text panel].

    The mask overlay uses a lower alpha for bbox_fill (it is redundant with the
    bbox rectangle) and a stronger alpha for actual SAM masks.
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_width / w)
    tw, th = int(w * scale), int(h * scale)
    vis = cv2.resize(image_bgr, (tw, th), interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr.copy()

    status = result.get("status", "unknown")
    mask_source = result.get("mask_source", "")

    # Mask overlay — less prominent for bbox_fill (just the bbox anyway)
    mask = result.get("mask")
    if mask is not None and isinstance(mask, np.ndarray) and mask.any():
        scaled_mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask
        alpha = 0.20 if mask_source == "bbox_fill" else 0.40
        colour = (180, 100, 0) if mask_source == "bbox_fill" else (0, 140, 255)
        vis = _overlay_mask(vis, scaled_mask, colour, alpha)

    # Bounding box
    bbox = result.get("bbox")
    clr = _CLR_OK if status == "success" else _CLR_FAIL
    if bbox:
        x1, y1, x2, y2 = (int(round(v * scale)) for v in bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), clr, 2)

    # Small failure label on the photo itself
    if status != "success":
        reason_short = (result.get("reason") or result.get("error") or "no detection")[:42]
        cv2.putText(vis, f"FAIL: {reason_short}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1, cv2.LINE_AA)

    pil_photo = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    # ── text panel ─────────────────────────────────────────────────────────
    debug = result.get("debug") or {}
    score = result.get("score")
    score_str = f"{score:.3f}" if score is not None else "-"
    prompts = debug.get("prompts_used") or result.get("prompts_used") or []
    thr = debug.get("thresholds_used") or {}
    n_raw = debug.get("candidate_count_before_nms")
    n_nms = debug.get("candidate_count_after_nms")
    n_sb = debug.get("candidate_count_before_shape_filter")
    n_sa = debug.get("candidate_count_after_shape_filter")
    sp = debug.get("shape_prior_status") or "-"

    lines: list[str] = [
        f"File:      {img_name}",
        f"Query:     {query}",
        f"Status:    {status}    Score: {score_str}    Mask: {mask_source or '-'}",
        f"Backend:   {result.get('backend') or '-'}    ShapePrior: {sp}",
    ]
    if prompts:
        lines.append(f"Prompts:   {' | '.join(str(p) for p in prompts[:3])[:80]}")
    if thr:
        lines.append(
            f"Thresholds: box={thr.get('box_threshold', '-')}  "
            f"text={thr.get('text_threshold', '-')}"
        )
    if n_raw is not None:
        lines.append(
            f"Counts:    raw={n_raw} →nms={n_nms} →shape_b={n_sb} →shape_a={n_sa}"
        )
    reason = result.get("reason") or result.get("error")
    if reason:
        lines.append(f"Reason:    {str(reason)[:90]}")

    line_h = 17
    pad = 7
    panel_h = pad + len(lines) * line_h + pad
    panel = Image.new("RGB", (pil_photo.width, panel_h), _PANEL_BG)
    draw = ImageDraw.Draw(panel)
    fg = _PANEL_FG_OK if status == "success" else _PANEL_FG_FAIL
    for i, line in enumerate(lines):
        try:
            draw.text((pad, pad + i * line_h), line, fill=fg, font=font)
        except Exception:
            draw.text((pad, pad + i * line_h),
                      line.encode("ascii", "replace").decode("ascii"),
                      fill=fg, font=font)

    combined = Image.new("RGB", (pil_photo.width, pil_photo.height + panel_h))
    combined.paste(pil_photo, (0, 0))
    combined.paste(panel, (0, pil_photo.height))
    return combined


# ── HTML generation ────────────────────────────────────────────────────────────

def _write_html(
    html_path: Path,
    results: list[dict],
    queries: list[str],
) -> None:
    by_query: dict[str, list[dict]] = {q: [] for q in queries}
    for r in results:
        if r["query"] in by_query:
            by_query[r["query"]].append(r)

    tabs_html = ""
    content_html = ""
    for i, q in enumerate(queries):
        recs = by_query.get(q, [])
        ok = sum(1 for r in recs if r["status"] == "success")
        active = "active" if i == 0 else ""
        tabs_html += (
            f'<button class="tab {active}" onclick="showTab({i})">'
            f'{q}&nbsp;{ok}/{len(recs)}</button>\n'
        )

        cards = ""
        for r in recs:
            vis_rel = "vis/" + Path(r["output_visualization_path"]).name
            st = r["status"]
            border = "#2a6e3a" if st == "success" else "#6e2a2a"
            sc = f'{r["score"]:.3f}' if r["score"] is not None else "-"
            tip = f'{r["image_name"]} | {st} | score={sc} | {r.get("backend") or ""}'
            sp = r.get("shape_prior_status") or ""
            cards += (
                f'\n  <div class="card" style="border-color:{border}">'
                f'<a href="{vis_rel}" target="_blank">'
                f'<img src="{vis_rel}" title="{tip}" loading="lazy"></a>'
                f'<div class="cap"><b>{r["image_name"][:24]}</b><br>'
                f'{st} &bull; {sc}<br>'
                f'{r.get("backend") or ""}<br>{sp}</div></div>'
            )

        disp = "block" if i == 0 else "none"
        content_html += (
            f'<div class="tc" id="tc{i}" style="display:{disp}">'
            f'<div class="grid">{cards}\n</div></div>\n'
        )

    n_imgs = len({r["image_name"] for r in results})
    total = len(results)
    ok_total = sum(1 for r in results if r["status"] == "success")

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<title>Open-Vocab Localization Eval</title>
<style>
body{{font-family:Arial,sans-serif;margin:18px;background:#111120;color:#ddd}}
h1{{color:#7ec8e8;margin-bottom:4px}}
p{{color:#999;margin:4px 0 12px}}
.tabs{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}}
.tab{{padding:7px 12px;background:#1c1c2e;border:1px solid #444;color:#aaa;
      cursor:pointer;border-radius:4px;font-size:13px;white-space:nowrap}}
.tab.active{{background:#273f6e;border-color:#7ec8e8;color:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}}
.card{{border:2px solid #444;border-radius:5px;overflow:hidden;background:#1a1a2c}}
.card img{{width:100%;height:210px;object-fit:cover;display:block}}
.cap{{padding:5px 8px;font-size:11px;color:#999;line-height:1.5}}
</style></head>
<body>
<h1>Open-Vocabulary Localization Evaluation</h1>
<p>Images: {n_imgs} &nbsp;&bull;&nbsp; Total cases: {total}
   &nbsp;&bull;&nbsp; Success: {ok_total}/{total}
   &nbsp;&bull;&nbsp; Queries: {len(queries)}</p>
<div class="tabs">
{tabs_html}</div>
{content_html}
<script>
function showTab(n){{
  document.querySelectorAll('.tc').forEach((e,i)=>e.style.display=i===n?'block':'none');
  document.querySelectorAll('.tab').forEach((e,i)=>e.classList.toggle('active',i===n));
}}
</script>
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visual evaluation of Phase 1/2 open-vocab localization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            10-image sanity check:
              python tools/eval/visualize_open_vocab_localization.py \\
                --image-dir "D:\\Aliintern\\fashion-ai-data\\fashionai_attributes\\ \\
                  round1_fashionAI_attributes_test_a\\Images\\coat_length_labels" \\
                --output-dir outputs/open_vocab_vis_coat_10 \\
                --sample-size 10 --seed 42 \\
                --queries "这件外套的口袋" "这件外套的拉链" "这件外套的扣子" "这件外套的下摆"

            100-image run:
              python tools/eval/visualize_open_vocab_localization.py \\
                --image-dir "D:\\..." \\
                --output-dir outputs/open_vocab_vis_coat_100 \\
                --sample-size 100 --seed 42
        """),
    )
    p.add_argument(
        "--image-dir", required=True,
        help="Directory containing images to evaluate.",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Output directory for vis PNGs, index.html, results.jsonl.",
    )
    p.add_argument("--sample-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--queries", nargs="+", default=None,
        help="Query strings. Defaults to 8 coat queries if omitted.",
    )
    p.add_argument(
        "--device", default="cuda",
        help="'cuda' or 'cpu'. Falls back to CPU with a warning if CUDA unavailable.",
    )
    p.add_argument(
        "--dino-threshold", type=float, default=0.3,
        help="Detection score threshold passed to GroundingDINO.",
    )
    p.add_argument(
        "--max-vis-width", type=int, default=640,
        help="Max width (px) for saved visualization images.",
    )
    p.add_argument(
        "--dino-model-id", default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model ID for Grounding DINO.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    queries = args.queries or DEFAULT_QUERIES

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── find + sample images ──────────────────────────────────────────────────
    img_dir = Path(args.image_dir)
    if not img_dir.exists():
        print(f"ERROR: image directory not found: {img_dir}", file=sys.stderr)
        sys.exit(1)

    all_imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not all_imgs:
        print(f"ERROR: no images found in {img_dir}", file=sys.stderr)
        sys.exit(1)

    sample_n = min(args.sample_size, len(all_imgs))
    sampled = sorted(random.Random(args.seed).sample(all_imgs, sample_n))

    print(f"[INFO] Found {len(all_imgs)} images, sampling {sample_n}.")
    print(f"[INFO] {len(queries)} queries × {sample_n} images = {sample_n * len(queries)} cases.")

    # ── load Grounding DINO ───────────────────────────────────────────────────
    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable — using CPU (inference will be slow).")
        device = "cpu"

    print(f"[INFO] Loading {args.dino_model_id} on {device} …")
    try:
        from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
        locator = GroundingDINOLocator(model_id=args.dino_model_id, device=device)
    except Exception as exc:
        print(f"ERROR: Grounding DINO load failed: {exc}", file=sys.stderr)
        print("Install 'transformers' and ensure network/cache access.", file=sys.stderr)
        sys.exit(1)

    font = _load_font(13)
    results: list[dict[str, Any]] = []

    # ── per-image × per-query loop ────────────────────────────────────────────
    for img_idx, img_path in enumerate(sampled):
        print(f"\n[{img_idx + 1}/{sample_n}] {img_path.name}")

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print("  [WARN] Cannot read — skipping.")
            continue

        img_h, img_w = image_bgr.shape[:2]

        # Full-image stub instance: coat images fill most of the frame,
        # so [0,0,W,H] is a reasonable garment bbox for shape-prior filtering.
        instance: dict[str, Any] = {
            "instance_id": "stub_full_image",
            "fine_class_name": "long sleeve outwear",
            "class_name": "long sleeve outwear",
            "coarse_class_name": "outerwear",
            "bbox": [0, 0, img_w, img_h],
        }

        for q_idx, query in enumerate(queries):
            # Filename: keep ASCII only to avoid filesystem issues
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in img_path.stem)[:24]
            vis_fname = f"{img_idx:04d}_{safe}__q{q_idx:02d}.jpg"
            vis_path = vis_dir / vis_fname

            try:
                result = locate_region(
                    query=query,
                    instance=instance,
                    image=image_bgr,
                    image_width=img_w,
                    image_height=img_h,
                    locator=locator,
                    dino_threshold=args.dino_threshold,
                    prefer_pred_mask=False,   # no pre-saved masks
                    sam_wrapper=None,          # bbox_fill fallback
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": str(exc),
                    "query": query,
                    "backend": "error",
                }

            status = result.get("status", "unknown")
            score = result.get("score")
            sc_str = f"{score:.3f}" if score is not None else "-"
            print(
                f"  q{q_idx}: {query!r:22s} → {status:<8s}  "
                f"score={sc_str:6s}  backend={result.get('backend') or '-'}"
            )

            # Save visualization (never crash the loop on draw failure)
            try:
                vis_img = _build_vis(
                    image_bgr=image_bgr,
                    result=result,
                    query=query,
                    img_name=img_path.name,
                    font=font,
                    max_width=args.max_vis_width,
                )
                vis_img.save(str(vis_path), quality=88)
            except Exception as exc:
                print(f"    [WARN] Visualization failed ({exc}); saving raw image.")
                cv2.imwrite(str(vis_path), image_bgr)

            debug = result.get("debug") or {}
            results.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "query": query,
                "query_idx": q_idx,
                "status": status,
                "bbox": result.get("bbox"),
                "score": score,
                "mask_source": result.get("mask_source"),
                "backend": result.get("backend"),
                "prompts_used": debug.get("prompts_used") or result.get("prompts_used"),
                "thresholds_used": debug.get("thresholds_used"),
                "candidate_count_before_nms": debug.get("candidate_count_before_nms"),
                "candidate_count_after_nms": debug.get("candidate_count_after_nms"),
                "candidate_count_before_shape_filter": debug.get("candidate_count_before_shape_filter"),
                "candidate_count_after_shape_filter": debug.get("candidate_count_after_shape_filter"),
                "shape_prior_status": debug.get("shape_prior_status"),
                "reason": result.get("reason") or result.get("error"),
                "output_visualization_path": str(vis_path),
            })

    # ── persist ───────────────────────────────────────────────────────────────
    jsonl_path = output_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_html(output_dir / "index.html", results, queries)

    # ── summary ───────────────────────────────────────────────────────────────
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*62}")
    print(f"Done: {ok}/{total} success  ({total - ok} failed/error)")
    print(f"Output:  {output_dir.resolve()}")
    print(f"HTML:    {(output_dir / 'index.html').resolve()}")
    print(f"JSONL:   {jsonl_path.resolve()}")
    print("\nPer-query breakdown:")
    for q in queries:
        qr = [r for r in results if r["query"] == q]
        q_ok = sum(1 for r in qr if r["status"] == "success")
        print(f"  {q:32s}  {q_ok:3d}/{len(qr)}")


if __name__ == "__main__":
    main()
