"""
Open-vocab Grounding DINO visual test — YOLO-crop mode.

Runs YOLO (Stage 1 only) to obtain garment bounding boxes, crops each garment
with padding, runs Grounding DINO inside each crop, then maps detections back to
original image coordinates.  Compares directly to the full-image test by reusing
the same 50-image sample manifest when available.

Status labels (no ground truth, so no "success"):
  detected       — passes score threshold + crop-relative area filter
  low_confidence — passes watermark threshold only
  no_detection   — nothing above watermark
  error          — runtime failure

Usage:
    cd D:\\Aliintern\\fashion-finegrained-vision
    python scripts/run_open_vocab_yolo_crop_test.py
    python scripts/run_open_vocab_yolo_crop_test.py --manifest path/to/manifest.json
"""
from __future__ import annotations

import argparse
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

# ── constants ──────────────────────────────────────────────────────────────────
DF2_NAMES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear",
    "long sleeve outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress",
]

PREV_MANIFEST = Path(
    "outputs/visual_tests/open_vocab_coat_length_sample50/sample_manifest.json"
)
SAMPLE_DIR = Path(
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\coat_length_labels"
)
OUTPUT_ROOT = Path("outputs/visual_tests/open_vocab_coat_length_sample50_yolo_crop")
YOLO_WEIGHTS = ROOT / "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
SEED = 42
N_SAMPLES = 50
YOLO_CONF = 0.25
CROP_PAD_RATIO = 0.07          # 7% padding around each garment box
NMS_IOU_THRESHOLD = 0.50

# ── per-query config ───────────────────────────────────────────────────────────
# min/max_crop_area_ratio: relative to the garment crop, not the full image.
# accept_threshold: box counted as "detected" only above this.
# watermark_threshold: box shown in grey (low_confidence) above this.
QUERY_CONFIG: dict[str, dict[str, Any]] = {
    "zipper": {
        "query_cn": "拉链",
        "prompts": ["clothing zipper", "zip"],
        "accept_threshold": 0.40,
        "watermark_threshold": 0.20,
        "min_crop_area_ratio": 0.003,
        "max_crop_area_ratio": 0.30,   # zipper is a narrow strip, not the whole garment
        "enabled": True,
    },
    "pocket": {
        "query_cn": "口袋",
        "prompts": ["clothing pocket", "garment pocket"],
        "accept_threshold": 0.35,
        "watermark_threshold": 0.20,
        "min_crop_area_ratio": 0.005,
        "max_crop_area_ratio": 0.55,
        "enabled": True,
    },
    "button": {
        "query_cn": "扣子",
        "prompts": ["clothing button", "coat button"],
        "accept_threshold": 0.40,
        "watermark_threshold": 0.20,
        "min_crop_area_ratio": 0.002,
        "max_crop_area_ratio": 0.25,
        "enabled": True,
    },
    "belt": {
        "query_cn": "腰带",
        # "waist belt" intentionally excluded — triggers waist-region false positives
        "prompts": ["clothing belt", "coat belt", "fabric belt"],
        "accept_threshold": 0.40,
        "watermark_threshold": 0.20,
        "min_crop_area_ratio": 0.005,
        "max_crop_area_ratio": 0.60,
        "enabled": True,
    },
    "placket": {
        "query_cn": "门襟",
        "prompts": ["front placket", "shirt placket", "front opening of coat"],
        "accept_threshold": 0.45,   # raised — placket is the noisiest query
        "watermark_threshold": 0.25,
        "min_crop_area_ratio": 0.010,
        "max_crop_area_ratio": 0.60,
        "enabled": True,            # set False here to disable in gallery
    },
}

WATERMARK_COLOR = (80, 80, 80)
QUERY_COLOR: dict[str, tuple[int, int, int]] = {
    "zipper":  (200, 200,   0),
    "pocket":  (  0, 200, 200),
    "button":  (  0, 140, 255),
    "belt":    (200,   0, 200),
    "placket": (255, 120,   0),
}
YOLO_BOX_COLOR = (40, 220, 40)


# ── sampling ───────────────────────────────────────────────────────────────────

def load_or_sample(manifest_path: Path) -> tuple[list[Path], dict[str, Any]]:
    """Reuse existing manifest for fair comparison; sample fresh if absent."""
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        images = [Path(p) for p in data["images"]]
        print(f"Reusing manifest: {manifest_path}  ({len(images)} images)")
        return images, data
    all_imgs = sorted(SAMPLE_DIR.glob("*.jpg")) + sorted(SAMPLE_DIR.glob("*.png"))
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(all_imgs, min(N_SAMPLES, len(all_imgs))), key=lambda p: p.name)
    data = {
        "seed": SEED, "n_samples": len(chosen),
        "source_dir": str(SAMPLE_DIR),
        "images": [str(p) for p in chosen],
    }
    print(f"No existing manifest — sampled {len(chosen)} images (seed={SEED})")
    return chosen, data


# ── YOLO helpers ───────────────────────────────────────────────────────────────

def get_garment_crops(
    img_bgr: np.ndarray,
    yolo_model: Any,
    img_path: Path,
) -> list[dict[str, Any]]:
    """
    Run YOLO on img_path, return padded garment crops with original-image coords.

    Returns list of:
        {crop, cx1, cy1, cx2, cy2, class_name, yolo_conf}
    """
    results = yolo_model.predict(str(img_path), conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    H, W = img_bgr.shape[:2]
    crops: list[dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        cls_name = DF2_NAMES[cls_id] if 0 <= cls_id < len(DF2_NAMES) else f"cls{cls_id}"

        pw = int((x2 - x1) * CROP_PAD_RATIO)
        ph = int((y2 - y1) * CROP_PAD_RATIO)
        cx1, cy1 = max(0, int(x1) - pw), max(0, int(y1) - ph)
        cx2, cy2 = min(W, int(x2) + pw), min(H, int(y2) + ph)

        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        crops.append({
            "crop": crop,
            "cx1": cx1, "cy1": cy1, "cx2": cx2, "cy2": cy2,
            "class_name": cls_name, "yolo_conf": round(conf, 3),
        })
    return crops


def draw_yolo_boxes(img_bgr: np.ndarray, crops: list[dict[str, Any]]) -> np.ndarray:
    vis = img_bgr.copy()
    for i, c in enumerate(crops):
        cv2.rectangle(vis, (c["cx1"], c["cy1"]), (c["cx2"], c["cy2"]), YOLO_BOX_COLOR, 2)
        label = f"[{i}] {c['class_name'][:14]}  {c['yolo_conf']:.2f}"
        cv2.putText(vis, label, (c["cx1"], max(c["cy1"] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, YOLO_BOX_COLOR, 1)
    if not crops:
        cv2.putText(vis, "no garment detected", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 60, 180), 1)
    return vis


# ── detection + crop-relative filtering ───────────────────────────────────────

def detect_across_crops(
    locator: GroundingDINOLocator,
    crops: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Run detect_multi_prompt on every garment crop, apply crop-relative area
    filter, then remap accepted detections to original image coordinates.

    Returns all passing detections sorted descending by score.
    """
    prompts      = cfg["prompts"]
    min_cr_ratio = cfg["min_crop_area_ratio"]
    max_cr_ratio = cfg["max_crop_area_ratio"]
    watermark    = cfg["watermark_threshold"]

    all_dets: list[dict[str, Any]] = []
    for crop_idx, c in enumerate(crops):
        crop_img = c["crop"]
        cH, cW   = crop_img.shape[:2]
        crop_area = cW * cH
        if crop_area == 0:
            continue

        # min_bbox_area_ratio=0 → no internal filter; we apply crop-relative below
        raw = locator.detect_multi_prompt(
            crop_img, prompts,
            garment_mask=None,
            threshold=watermark,
            min_bbox_area_ratio=0.0,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )

        for d in raw:
            dx1, dy1, dx2, dy2 = d["bbox_xyxy"]
            bbox_area = max(0, dx2 - dx1) * max(0, dy2 - dy1)
            cr_ratio  = bbox_area / crop_area

            if cr_ratio < min_cr_ratio or cr_ratio > max_cr_ratio:
                continue

            # remap to original-image coordinates
            all_dets.append({
                **d,
                "bbox_xyxy": [c["cx1"] + dx1, c["cy1"] + dy1,
                               c["cx1"] + dx2, c["cy1"] + dy2],
                "crop_area_ratio": round(cr_ratio, 4),
                "crop_idx": crop_idx,
                "crop_class": c["class_name"],
            })

    all_dets.sort(key=lambda d: d["score"], reverse=True)
    return all_dets


# ── overlay drawing ────────────────────────────────────────────────────────────

def draw_two_level(
    img_bgr: np.ndarray,
    accepted: list[dict[str, Any]],
    watermark: list[dict[str, Any]],
    color: tuple[int, int, int],
) -> np.ndarray:
    out = img_bgr.copy()
    for det in watermark[:5]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), WATERMARK_COLOR, 1)
        cv2.putText(out, f"{det['score']:.2f}", (x1, max(y1 - 2, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, WATERMARK_COLOR, 1)
    for i, det in enumerate(accepted[:3]):
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if i == 0:
            prompt_short = det.get("prompt", "")[:18]
            cv2.putText(out, f"{prompt_short} {det['score']:.2f}",
                        (x1, max(y1 - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
    if not accepted and not watermark:
        cv2.putText(out, "no detection", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 160), 1)
    return out


# ── per-image processing ───────────────────────────────────────────────────────

def process_one(
    locator: GroundingDINOLocator,
    yolo_model: Any,
    img_path: Path,
    per_dir: Path,
) -> dict[str, Any]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return {"status": "error", "error": "imread failed",
                "n_garment_crops": 0, "queries": {}}

    shutil.copy(img_path, per_dir / "original.jpg")

    crops = get_garment_crops(img_bgr, yolo_model, img_path)
    cv2.imwrite(str(per_dir / "yolo_boxes.jpg"), draw_yolo_boxes(img_bgr, crops))

    query_results: dict[str, Any] = {}

    for key, cfg in QUERY_CONFIG.items():
        if not cfg.get("enabled", True):
            query_results[key] = {
                "query_cn": cfg["query_cn"], "status": "disabled",
                "top_score": None, "n_accepted": 0, "n_watermark": 0,
            }
            cv2.imwrite(str(per_dir / f"{key}_overlay.jpg"), img_bgr)
            continue

        if not crops:
            query_results[key] = {
                "query_cn": cfg["query_cn"], "prompts": cfg["prompts"],
                "backend": "GDINO+yolo_crop", "status": "no_detection",
                "reason": "no_garment_detected", "top_score": None,
                "n_accepted": 0, "n_watermark": 0, "detections": [],
            }
            cv2.imwrite(str(per_dir / f"{key}_overlay.jpg"), img_bgr)
            continue

        try:
            all_dets = detect_across_crops(locator, crops, cfg)
        except Exception as exc:
            query_results[key] = {
                "query_cn": cfg["query_cn"], "prompts": cfg["prompts"],
                "backend": "GDINO+yolo_crop", "status": "error", "error": str(exc),
                "top_score": None, "n_accepted": 0, "n_watermark": 0, "detections": [],
            }
            cv2.imwrite(str(per_dir / f"{key}_overlay.jpg"), img_bgr)
            continue

        acc_th   = cfg["accept_threshold"]
        accepted = [d for d in all_dets if d["score"] >= acc_th]
        wm_dets  = [d for d in all_dets if d["score"] <  acc_th]

        status = ("detected" if accepted else
                  "low_confidence" if wm_dets else
                  "no_detection")

        top_score = all_dets[0]["score"] if all_dets else None
        overlay = draw_two_level(img_bgr, accepted, wm_dets, QUERY_COLOR.get(key, (180, 180, 180)))
        cv2.imwrite(str(per_dir / f"{key}_overlay.jpg"), overlay)

        query_results[key] = {
            "query_cn": cfg["query_cn"],
            "prompts": cfg["prompts"],
            "accept_threshold": acc_th,
            "backend": "GDINO+yolo_crop",
            "status": status,
            "top_score": round(top_score, 3) if top_score is not None else None,
            "n_accepted": len(accepted),
            "n_watermark": len(wm_dets),
            "detections": [
                {**d, "score": round(d["score"], 3),
                 "accepted": d["score"] >= acc_th}
                for d in all_dets[:4]
            ],
        }

    return {"status": "ok", "n_garment_crops": len(crops), "queries": query_results}


# ── HTML gallery ───────────────────────────────────────────────────────────────

_CSS = """
body{font-family:monospace;background:#111;color:#ccc;margin:0;padding:10px}
h1{font-size:1.05em;margin:0 0 3px}
.meta{color:#888;font-size:.74em;margin-bottom:8px}
.legend{font-size:.70em;margin-bottom:10px;color:#999}
.qtable{border-collapse:collapse;margin-bottom:12px;font-size:.70em}
.qtable th,.qtable td{border:1px solid #2a2a2a;padding:2px 8px;text-align:center}
.qtable th{background:#1c1c1c}
.card{background:#1a1a1a;border-radius:6px;margin-bottom:7px;padding:6px 8px}
.fn{font-size:.60em;color:#444;margin-bottom:3px;word-break:break-all}
.row{display:flex;gap:4px;overflow-x:auto}
.cell{min-width:130px;max-width:148px;flex-shrink:0;text-align:center}
.cell img{width:100%;height:162px;object-fit:cover;border-radius:3px;display:block}
.ci{font-size:.57em;line-height:1.35;padding:2px 0}
.ci-hd{color:#aaa;font-weight:bold}
.ci-pt{color:#90caf9}
.ci-be{color:#555}
.dt{color:#4caf50}.lc{color:#ff9800}.nd{color:#555}.er{color:#f44336}.di{color:#444}
"""


def _sc(status: str) -> str:
    return {"detected": "dt", "low_confidence": "lc", "no_detection": "nd",
            "error": "er", "disabled": "di"}.get(status, "nd")


def _badge(status: str) -> str:
    return {"detected": "✓ detected", "low_confidence": "~ low_conf",
            "no_detection": "✗ no_det", "error": "! error",
            "disabled": "— disabled"}.get(status, status)


def generate_html(results: list[dict[str, Any]]) -> None:
    keys = list(QUERY_CONFIG.keys())
    n = len(results)

    # summary table
    q_headers = "".join(
        f'<th>{cfg["query_cn"]}<br><span style="color:#90caf9">{k}</span></th>'
        for k, cfg in QUERY_CONFIG.items()
    )
    count_row = "".join(
        f'<td>'
        f'<span class="dt">✓{sum(1 for r in results if r["queries"].get(k, {}).get("status")=="detected")}</span> '
        f'<span class="lc">~{sum(1 for r in results if r["queries"].get(k, {}).get("status")=="low_confidence")}</span>'
        f'/{n}</td>'
        for k in keys
    )
    table = (
        f'<table class="qtable"><tr><th></th>{q_headers}</tr>'
        f'<tr><td>det/low_conf/total</td>{count_row}</tr></table>'
    )

    cards = []
    for r in results:
        n_crops = r.get("n_garment_crops", 0)

        # cell 1: original
        c_orig = (
            f'<div class="cell"><img src="per_image/{r["id"]}/original.jpg" loading="lazy">'
            f'<div class="ci"><span class="ci-hd">original</span></div></div>'
        )
        # cell 2: YOLO boxes
        c_yolo = (
            f'<div class="cell"><img src="per_image/{r["id"]}/yolo_boxes.jpg" loading="lazy">'
            f'<div class="ci"><span class="ci-hd">YOLO</span><br>'
            f'<span style="color:#4caf50">{n_crops} garment{"s" if n_crops!=1 else ""}</span></div></div>'
        )

        q_cells = []
        for k, cfg in QUERY_CONFIG.items():
            qr = r["queries"].get(k, {})
            status = qr.get("status", "unknown")
            sc_val = qr.get("top_score")
            score_txt = f"{sc_val:.2f}" if sc_val is not None else "—"
            n_acc = qr.get("n_accepted", 0)
            n_wm  = qr.get("n_watermark", 0)
            prompts = qr.get("prompts", cfg.get("prompts", []))
            pt_txt = " | ".join(prompts[:2])
            acc_th = cfg.get("accept_threshold", "—")
            q_cells.append(
                f'<div class="cell">'
                f'<img src="per_image/{r["id"]}/{k}_overlay.jpg" loading="lazy">'
                f'<div class="ci">'
                f'<div class="ci-hd">{cfg["query_cn"]}</div>'
                f'<div class="ci-pt">{pt_txt}</div>'
                f'<div class="ci-be">GDINO+yolo_crop  th≥{acc_th}</div>'
                f'<div class="{_sc(status)}">{_badge(status)}</div>'
                f'<div style="color:#666">top:{score_txt} acc:{n_acc} wm:{n_wm}</div>'
                f'</div></div>'
            )

        cards.append(
            f'<div class="card"><div class="fn">{r["filename"]}</div>'
            f'<div class="row">{c_orig}{c_yolo}{"".join(q_cells)}</div></div>'
        )

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>Open-Vocab DINO — YOLO-crop mode</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Open-Vocab Grounding DINO — YOLO-crop mode</h1>"
        f'<div class="meta">'
        f"n={n} images &nbsp;|&nbsp; seed={SEED} &nbsp;|&nbsp; "
        f"YOLO conf≥{YOLO_CONF} &nbsp;|&nbsp; crop_pad={int(CROP_PAD_RATIO*100)}% &nbsp;|&nbsp; "
        f"per-query thresholds + crop-relative area filter &nbsp;|&nbsp; "
        f'compare: <a href="../open_vocab_coat_length_sample50/index.html" '
        f'style="color:#90caf9">full-image baseline</a></div>'
        f'<div class="legend">'
        f'<span class="dt">✓ detected</span> ≥ accept_threshold &nbsp;'
        f'<span class="lc">~ low_confidence</span> ≥ watermark &nbsp;'
        f'<span class="nd">✗ no_detection</span> &nbsp;'
        f'<span style="border:2px solid #4caf50;padding:0 4px">colour box</span> accepted &nbsp;'
        f'<span style="border:1px solid #555;padding:0 4px">grey box</span> watermark</div>'
        f"{table}"
        + "\n".join(cards)
        + "</body></html>"
    )
    (OUTPUT_ROOT / "index.html").write_text(html, encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest", default=str(PREV_MANIFEST),
        help="Path to sample_manifest.json from the full-image test (reuses same images).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not YOLO_WEIGHTS.exists():
        print(f"ERROR: YOLO weights not found: {YOLO_WEIGHTS}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sample, manifest_data = load_or_sample(Path(args.manifest))
    (OUTPUT_ROOT / "sample_manifest.json").write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    enabled_keys = [k for k, cfg in QUERY_CONFIG.items() if cfg.get("enabled", True)]
    print(f"Queries: {enabled_keys}")
    print(f"Loading YOLO from {YOLO_WEIGHTS.name}...", flush=True)
    from ultralytics import YOLO as UltralyticsYOLO
    yolo_model = UltralyticsYOLO(str(YOLO_WEIGHTS))

    print("Loading Grounding DINO model...", flush=True)
    locator = GroundingDINOLocator()
    print("Models loaded.\n")

    results: list[dict[str, Any]] = []
    t_total = time.time()

    for idx, img_path in enumerate(sample):
        img_id = f"{idx + 1:03d}"
        per_dir = OUTPUT_ROOT / "per_image" / img_id
        per_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{img_id}/{len(sample)}] {img_path.name}", end="  ", flush=True)
        t0 = time.time()

        rec = process_one(locator, yolo_model, img_path, per_dir)
        elapsed = time.time() - t0

        rec.update(id=img_id, filename=img_path.name)
        (per_dir / "result.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        n_crops = rec.get("n_garment_crops", 0)
        q_line = "  ".join(
            f'{k}=' + {
                "detected": "✓", "low_confidence": "~",
                "no_detection": "✗", "error": "!", "disabled": "-",
            }.get(rec["queries"].get(k, {}).get("status", ""), "?")
            for k in QUERY_CONFIG
        )
        print(f"crops={n_crops}  ({elapsed:.1f}s)  {q_line}")
        results.append(rec)

    total = time.time() - t_total

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done: {len(results)} images  total={total:.0f}s  avg={total/len(results):.1f}s/img\n")
    header = f"{'Key':<10} {'CN':<6} {'✓det':>5} {'~low':>5} {'✗no':>5} {'!err':>5}"
    print(header)
    print("-" * 45)
    for k, cfg in QUERY_CONFIG.items():
        n_det = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "detected")
        n_lc  = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "low_confidence")
        n_nd  = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "no_detection")
        n_er  = sum(1 for r in results if r["queries"].get(k, {}).get("status") == "error")
        print(f"{k:<10} {cfg['query_cn']:<6} {n_det:>5} {n_lc:>5} {n_nd:>5} {n_er:>5}")

    summary: dict[str, Any] = {
        "n_images": len(results),
        "yolo_weights": str(YOLO_WEIGHTS),
        "yolo_conf": YOLO_CONF,
        "crop_pad_ratio": CROP_PAD_RATIO,
        "total_sec": round(total, 1),
        "avg_sec_per_image": round(total / len(results), 1),
        "query_config": {k: {ck: cv for ck, cv in cfg.items() if ck != "prompts"}
                         for k, cfg in QUERY_CONFIG.items()},
        "per_query": {
            k: {
                "n_detected":       sum(1 for r in results if r["queries"].get(k, {}).get("status") == "detected"),
                "n_low_confidence": sum(1 for r in results if r["queries"].get(k, {}).get("status") == "low_confidence"),
                "n_no_detection":   sum(1 for r in results if r["queries"].get(k, {}).get("status") == "no_detection"),
                "n_error":          sum(1 for r in results if r["queries"].get(k, {}).get("status") == "error"),
            }
            for k in QUERY_CONFIG
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
