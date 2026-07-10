"""
Visual test: open-vocab Grounding DINO localization on 50 sampled coat images.

Improvements vs v1:
- Multi-prompt ensemble via open_vocab_prompt_map (e.g. zipper + clothing zipper + zip)
- NMS deduplication across prompts
- Threshold raised to ACCEPT_THRESHOLD (0.40); everything above WATERMARK (0.20)
  is still shown in grey so the user can calibrate the threshold manually
- min_bbox_area_ratio=0.003 drops tiny jewellery / noise detections

Skips the full garment pipeline (no YOLO, no SAM-HQ) — tests DINO component only.

Usage:
    cd D:\\Aliintern\\fashion-finegrained-vision
    python scripts/run_open_vocab_visual_test.py
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
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
from fashion_vision.localization.intent_parser import parse_intent
from fashion_vision.localization.open_vocab_prompt_map import get_prompts_for_region

# ── config ─────────────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\coat_length_labels"
)
OUTPUT_ROOT = Path("outputs/visual_tests/open_vocab_coat_length_sample50")
SEED = 42
N_SAMPLES = 50

ACCEPT_THRESHOLD = 0.40    # detection is "success" above this
WATERMARK_THRESHOLD = 0.20 # show-but-grey below accept, down to this

MIN_BBOX_AREA_RATIO = 0.003   # drop tiny detections (jewellery, noise)
NMS_IOU_THRESHOLD = 0.50

QUERIES = ["拉链", "口袋", "扣子", "腰带", "门襟"]

QUERY_KEY: dict[str, str] = {
    "拉链": "zipper",
    "口袋": "pocket",
    "扣子": "button",
    "腰带": "belt",
    "门襟": "placket",
}

# BGR: accepted detections drawn in these colours per query
QUERY_COLOR: dict[str, tuple[int, int, int]] = {
    "拉链": (200, 200, 0),
    "口袋": (0, 200, 200),
    "扣子": (0, 140, 255),
    "腰带": (200, 0, 200),
    "门襟": (255, 120, 0),
}
WATERMARK_COLOR = (100, 100, 100)  # grey for below-threshold detections


# ── sampling ───────────────────────────────────────────────────────────────────

def sample_images() -> list[Path]:
    all_imgs = sorted(SAMPLE_DIR.glob("*.jpg")) + sorted(SAMPLE_DIR.glob("*.png"))
    rng = random.Random(SEED)
    chosen = rng.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))
    return sorted(chosen, key=lambda p: p.name)


# ── overlay drawing ────────────────────────────────────────────────────────────

def draw_two_level(
    img_bgr: np.ndarray,
    accepted: list[dict],
    watermark: list[dict],
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    """Draw accepted boxes in colour and below-threshold boxes in grey."""
    out = img_bgr.copy()

    # watermark boxes (grey, thin)
    for det in watermark[:5]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), WATERMARK_COLOR, 1)
        cv2.putText(
            out, f"{det['score']:.2f}", (x1, max(y1 - 2, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, WATERMARK_COLOR, 1,
        )

    # accepted boxes (colour, thick)
    for i, det in enumerate(accepted[:3]):
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if i == 0:
            prompt_short = det.get("prompt", label)[:20]
            txt = f"{prompt_short} {det['score']:.2f}"
            cv2.putText(
                out, txt, (x1, max(y1 - 4, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1,
            )

    if not accepted and not watermark:
        cv2.putText(
            out, "no detection", (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 60, 180), 1,
        )

    return out


# ── per-image processing ───────────────────────────────────────────────────────

def process_one(
    locator: GroundingDINOLocator,
    img_path: Path,
    per_dir: Path,
) -> dict[str, Any]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return {"status": "error", "error": "cv2.imread returned None", "queries": {}}

    shutil.copy(img_path, per_dir / "original.jpg")
    query_results: dict[str, Any] = {}

    for q in QUERIES:
        intent = parse_intent(q)
        key = QUERY_KEY[q]
        overlay_path = per_dir / f"{key}_overlay.jpg"

        # multi-prompt list from prompt map (e.g. ["clothing zipper", "zip", ...])
        prompts = get_prompts_for_region(intent.part or key)

        try:
            # run with low watermark threshold to capture everything worth showing
            all_dets = locator.detect_multi_prompt(
                img_bgr, prompts,
                garment_mask=None,
                threshold=WATERMARK_THRESHOLD,
                min_bbox_area_ratio=MIN_BBOX_AREA_RATIO,
                nms_iou_threshold=NMS_IOU_THRESHOLD,
            )
        except Exception as exc:
            cv2.imwrite(str(overlay_path), img_bgr)
            query_results[key] = {
                "query": q, "prompts": prompts, "part": intent.part,
                "backend": "open_vocab_grounding_dino",
                "status": "error", "error": str(exc),
                "top_score": None, "n_accepted": 0, "detections": [],
            }
            continue

        accepted = [d for d in all_dets if d["score"] >= ACCEPT_THRESHOLD]
        watermark = [d for d in all_dets if d["score"] < ACCEPT_THRESHOLD]

        status = "success" if accepted else (
            "below_threshold" if watermark else "no_detection"
        )
        top_score = all_dets[0]["score"] if all_dets else None

        overlay = draw_two_level(img_bgr, accepted, watermark, QUERY_COLOR[q], key)
        cv2.imwrite(str(overlay_path), overlay)

        query_results[key] = {
            "query": q,
            "prompts": prompts,
            "part": intent.part,
            "is_fast_path": intent.is_fast_path,
            "backend": "open_vocab_grounding_dino",
            "accept_threshold": ACCEPT_THRESHOLD,
            "status": status,
            "top_score": round(top_score, 3) if top_score is not None else None,
            "n_accepted": len(accepted),
            "n_watermark": len(watermark),
            "detections": [
                {**d, "score": round(d["score"], 3), "accepted": d["score"] >= ACCEPT_THRESHOLD}
                for d in all_dets[:5]
            ],
        }

    return {"status": "ok", "queries": query_results}


# ── HTML gallery ───────────────────────────────────────────────────────────────

_CSS = """
body{font-family:monospace;background:#111;color:#ccc;margin:0;padding:10px}
h1{font-size:1.05em;margin:0 0 3px}
.meta{color:#888;font-size:.75em;margin-bottom:8px}
.legend{font-size:.72em;margin-bottom:10px;color:#aaa}
.legend span{padding:1px 6px;border-radius:3px;margin-right:6px}
.qtable{border-collapse:collapse;margin-bottom:12px;font-size:.72em}
.qtable th,.qtable td{border:1px solid #2a2a2a;padding:2px 8px;text-align:center}
.qtable th{background:#1c1c1c}
.card{background:#1a1a1a;border-radius:6px;margin-bottom:7px;padding:6px 8px}
.card-fn{font-size:.62em;color:#444;margin-bottom:3px;word-break:break-all}
.img-row{display:flex;gap:4px;overflow-x:auto}
.cell{min-width:135px;max-width:150px;flex-shrink:0;text-align:center}
.cell img{width:100%;height:165px;object-fit:cover;border-radius:3px;display:block}
.ci{font-size:.58em;line-height:1.35;padding:2px 0;color:#666}
.ci-q{color:#ddd;font-weight:bold}
.ci-pt{color:#90caf9;font-size:.9em}
.ok{color:#4caf50}.bt{color:#ff9800}.nd{color:#666}.er{color:#f44336}
"""


def _status_cls(status: str) -> str:
    return {"success": "ok", "below_threshold": "bt", "no_detection": "nd",
            "error": "er"}.get(status, "nd")


def generate_html(results: list[dict[str, Any]]) -> None:
    keys = [QUERY_KEY[q] for q in QUERIES]
    n = len(results)

    # summary counts
    counts = {
        k: {
            "ok": sum(1 for r in results if r["queries"].get(k, {}).get("status") == "success"),
            "bt": sum(1 for r in results if r["queries"].get(k, {}).get("status") == "below_threshold"),
        }
        for k in keys
    }
    q_headers = "".join(
        f'<th>{q}<br><span style="color:#90caf9">{k}</span></th>'
        for q, k in zip(QUERIES, keys)
    )
    count_row = "".join(
        f'<td><span class="ok">✓{counts[k]["ok"]}</span> '
        f'<span class="bt">~{counts[k]["bt"]}</span> '
        f'/ {n}</td>'
        for k in keys
    )
    table = (
        f'<table class="qtable"><tr><th></th>{q_headers}</tr>'
        f'<tr><td>success/near/total</td>{count_row}</tr></table>'
    )

    cards = []
    for r in results:
        cells = [
            f'<div class="cell"><img src="per_image/{r["id"]}/original.jpg" loading="lazy">'
            f'<div class="ci"><span style="color:#555">original</span></div></div>'
        ]
        for q, k in zip(QUERIES, keys):
            qr = r["queries"].get(k, {})
            status = qr.get("status", "unknown")
            sc = qr.get("top_score")
            score_txt = f"{sc:.2f}" if sc is not None else "—"
            n_acc = qr.get("n_accepted", 0)
            n_wm = qr.get("n_watermark", 0)
            sc_cls = _status_cls(status)
            badge = "✓" if status == "success" else ("~" if status == "below_threshold" else "✗")
            prompts = qr.get("prompts", [])
            pt_short = " | ".join(prompts[:2])
            cells.append(
                f'<div class="cell">'
                f'<img src="per_image/{r["id"]}/{k}_overlay.jpg" loading="lazy">'
                f'<div class="ci">'
                f'<div class="ci-q">{q}</div>'
                f'<div class="ci-pt">{pt_short}</div>'
                f'<div class="{sc_cls}">{badge} top:{score_txt} acc:{n_acc} wm:{n_wm}</div>'
                f'<div style="color:#555">GDINO</div>'
                f'</div></div>'
            )
        cards.append(
            f'<div class="card">'
            f'<div class="card-fn">{r["filename"]}</div>'
            f'<div class="img-row">{"".join(cells)}</div>'
            f'</div>'
        )

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>Open-Vocab DINO v2</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Open-Vocab Grounding DINO v2 — coat_length_labels</h1>"
        f'<div class="meta">'
        f"n={n} images &nbsp;|&nbsp; seed={SEED} &nbsp;|&nbsp; "
        f"accept≥{ACCEPT_THRESHOLD} &nbsp;|&nbsp; watermark≥{WATERMARK_THRESHOLD} &nbsp;|&nbsp; "
        f"min_area={MIN_BBOX_AREA_RATIO} &nbsp;|&nbsp; multi-prompt+NMS</div>"
        f'<div class="legend">'
        f'<span style="border:2px solid #4caf50">colour box</span> accepted (≥{ACCEPT_THRESHOLD})'
        f'&nbsp;&nbsp;'
        f'<span style="border:1px solid #666">grey box</span> below threshold ({WATERMARK_THRESHOLD}–{ACCEPT_THRESHOLD})'
        f'&nbsp;&nbsp;'
        f'<span class="ok">✓</span>=success &nbsp;'
        f'<span class="bt">~</span>=below_threshold &nbsp;'
        f'<span class="nd">✗</span>=no_detection</div>'
        f"{table}"
        + "\n".join(cards)
        + "</body></html>"
    )
    (OUTPUT_ROOT / "index.html").write_text(html, encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SAMPLE_DIR.exists():
        print(f"ERROR: source folder not found: {SAMPLE_DIR}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sample = sample_images()
    manifest: dict[str, Any] = {
        "seed": SEED, "n_samples": len(sample),
        "queries": QUERIES, "query_key_map": QUERY_KEY,
        "accept_threshold": ACCEPT_THRESHOLD,
        "watermark_threshold": WATERMARK_THRESHOLD,
        "min_bbox_area_ratio": MIN_BBOX_AREA_RATIO,
        "source_dir": str(SAMPLE_DIR),
        "images": [str(p) for p in sample],
    }
    (OUTPUT_ROOT / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Sampled {len(sample)} images | queries: {QUERIES}")
    print(f"accept≥{ACCEPT_THRESHOLD}  watermark≥{WATERMARK_THRESHOLD}  multi-prompt+NMS")
    print("Loading Grounding DINO model...", flush=True)

    locator = GroundingDINOLocator()
    print("Model loaded.\n")

    results: list[dict[str, Any]] = []
    t_total = time.time()

    for idx, img_path in enumerate(sample):
        img_id = f"{idx + 1:03d}"
        per_dir = OUTPUT_ROOT / "per_image" / img_id
        per_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{img_id}/{len(sample)}] {img_path.name}", end="  ", flush=True)
        t0 = time.time()

        rec = process_one(locator, img_path, per_dir)
        elapsed = time.time() - t0

        rec.update(id=img_id, filename=img_path.name)
        (per_dir / "result.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        q_line = "  ".join(
            f"{QUERY_KEY[q]}="
            + {"success": "✓", "below_threshold": "~", "no_detection": "✗", "error": "!"}.get(
                rec["queries"].get(QUERY_KEY[q], {}).get("status", ""), "?"
            )
            for q in QUERIES
        )
        print(f"({elapsed:.1f}s)  {q_line}")
        results.append(rec)

    total = time.time() - t_total
    keys = [QUERY_KEY[q] for q in QUERIES]

    print(f"\n{'='*55}")
    print(f"Done: {len(results)} images  total={total:.0f}s  avg={total/len(results):.1f}s/img\n")
    print(f"{'Query':<10} {'Key':<10} {'✓success':>8} {'~below':>8} {'✗no-det':>8} {'err':>5}")
    print("-" * 55)
    for q, k in zip(QUERIES, keys):
        n_ok  = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "success")
        n_bt  = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "below_threshold")
        n_nd  = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "no_detection")
        n_er  = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "error")
        print(f"{q:<10} {k:<10} {n_ok:>8} {n_bt:>8} {n_nd:>8} {n_er:>5}")

    summary: dict[str, Any] = {
        "n_images": len(results),
        "queries": QUERIES,
        "accept_threshold": ACCEPT_THRESHOLD,
        "watermark_threshold": WATERMARK_THRESHOLD,
        "min_bbox_area_ratio": MIN_BBOX_AREA_RATIO,
        "backend": "open_vocab_grounding_dino",
        "total_sec": round(total, 1),
        "avg_sec_per_image": round(total / len(results), 1),
        "per_query": {
            k: {
                "query_cn": q,
                "n_success": sum(1 for r in results if r["queries"].get(k, {}).get("status") == "success"),
                "n_below_threshold": sum(1 for r in results if r["queries"].get(k, {}).get("status") == "below_threshold"),
                "n_no_detection": sum(1 for r in results if r["queries"].get(k, {}).get("status") == "no_detection"),
                "n_error": sum(1 for r in results if r["queries"].get(k, {}).get("status") == "error"),
            }
            for q, k in zip(QUERIES, keys)
        },
        "per_image": results,
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    generate_html(results)

    print(f"\nGallery:  {OUTPUT_ROOT / 'index.html'}")
    print(f"Summary:  {OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
