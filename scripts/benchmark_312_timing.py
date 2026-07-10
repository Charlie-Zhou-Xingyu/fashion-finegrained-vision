"""
Timing benchmark for 3.1.2 region localization — three pathways.

Pathways measured:
  1. Fashionpedia YOLO part detection  (口袋/袖子/拉链/领口)
  2. Grounding-DINO fallback           (胸 — zero-shot, not in FP coverage)
  3. Inner garment detection           (内搭 — SAM-based, no YOLO/DINO)

Usage::

    conda activate fashion-demo2
    python scripts/benchmark_312_timing.py

Outputs: ``outputs/benchmark_312_timing/``
  - timing_details.csv   per-image × per-query timings
  - timing_summary.csv   aggregated by pathway
  - timing_summary.json  full report
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────

IMAGES_DIR = Path(
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\lapel_design_labels"
)
NUM_IMAGES = 50
RANDOM_SEED = 42

QUERIES: List[Tuple[str, str]] = [
    # (query_text, pathway_label)
    ("口袋", "fashionpedia_yolo"),
    ("袖子", "fashionpedia_yolo"),
    ("拉链", "fashionpedia_yolo"),
    ("领口", "fashionpedia_yolo"),
    ("内搭", "inner_garment"),
    ("胸",   "grounding_dino"),
]

# Model paths
YOLO_WEIGHTS = "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
SAM_CHECKPOINT = "checkpoints/sam_hq/sam_hq_vit_b.pth"
FP_MODEL = "models/detectors/fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
DINO_MODEL = "models/grounding_dino_tiny"  # local path

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark_312_timing"


# ── Helpers ────────────────────────────────────────────────────────────────────

def sample_images(image_dir: Path, n: int, seed: int) -> List[Path]:
    """Randomly sample *n* image files from *image_dir*."""
    all_images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    rng = random.Random(seed)
    return rng.sample(all_images, min(n, len(all_images)))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mask_area(inst: dict) -> float:
    """Best-effort mask area for instance sorting (inner garment heuristic)."""
    mask = inst.get("mask")
    if mask is not None and isinstance(mask, np.ndarray):
        return float((mask > 0).sum())
    h = inst.get("bbox_h") or (inst.get("bbox_xyxy", [0, 0, 0, 0])[3] - inst.get("bbox_xyxy", [0, 0, 0, 0])[1])
    w = inst.get("bbox_w") or (inst.get("bbox_xyxy", [0, 0, 0, 0])[2] - inst.get("bbox_xyxy", [0, 0, 0, 0])[0])
    return float(h * w)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_garment_pipeline():
    """Load GarmentPipeline (YOLO garment detector + SAM-HQ). Skip landmarks/crops."""
    from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig

    config = GarmentPipelineConfig(
        yolo_weights=YOLO_WEIGHTS,
        sam_checkpoint=SAM_CHECKPOINT,
        sam_model_type="vit_b",
        yolo_device="0",
        sam_device="cuda",
        run_landmark_and_crops=False,
        run_attribute_inference=False,
        save_yolo_vis=False,
        save_yolo_crops=False,
    )
    pipeline = GarmentPipeline(config)
    return pipeline


def load_fp_detector():
    """Load Fashionpedia YOLOv8s 19-class part detector."""
    from fashion_vision.localization.fashionpedia_part_detector import (
        FashionpediaPartDetector,
    )
    fp_path = str(PROJECT_ROOT / FP_MODEL)
    if not Path(fp_path).exists():
        print(f"[WARN] Fashionpedia model not found: {fp_path}")
        return None
    det = FashionpediaPartDetector(fp_path, device="cuda")
    print(f"[INFO] Fashionpedia detector loaded: {fp_path}")
    return det


def load_dino_locator():
    """Load Grounding DINO from local weights."""
    from fashion_vision.localization.grounding_dino_locator import (
        GroundingDINOLocator,
    )
    dino_path = str(PROJECT_ROOT / DINO_MODEL)
    if not Path(dino_path).exists():
        print(f"[WARN] DINO model not found: {dino_path}")
        return None
    locator = GroundingDINOLocator(model_id=dino_path, device="cuda")
    print(f"[INFO] DINO locator loaded: {dino_path}")
    return locator


# ── Per-query runners ─────────────────────────────────────────────────────────

def run_fp_query(
    query: str,
    instance: dict,
    image_bgr: np.ndarray,
    W: int, H: int,
    fp_detector,
    sam_wrapper,
) -> dict:
    """Run a Fashionpedia YOLO query via locate_region."""
    from fashion_vision.localization.region_localization_router import (
        locate_region,
    )

    t0 = time.perf_counter()
    result = locate_region(
        query, instance, image_bgr, W, H,
        locator=None,  # no DINO — FP YOLO covers this part
        fashionpedia_detector=fp_detector,
        sam_wrapper=sam_wrapper,
    )
    elapsed = time.perf_counter() - t0
    result["_elapsed"] = elapsed
    return result


def run_dino_query(
    query: str,
    instance: dict,
    image_bgr: np.ndarray,
    W: int, H: int,
    fp_detector,
    dino_locator,
    sam_wrapper,
) -> dict:
    """Run a Grounding-DINO query via locate_region (for parts not in FP)."""
    from fashion_vision.localization.region_localization_router import (
        locate_region,
    )

    t0 = time.perf_counter()
    result = locate_region(
        query, instance, image_bgr, W, H,
        locator=dino_locator,
        fashionpedia_detector=fp_detector,
        sam_wrapper=sam_wrapper,
    )
    elapsed = time.perf_counter() - t0
    result["_elapsed"] = elapsed
    return result


def run_inner_query(
    image_bgr: np.ndarray,
    instances: list,
    sam_wrapper,
) -> dict:
    """Run inner garment detection on outerwear instances."""
    from fashion_vision.localization.inner_garment_detector import (
        detect_inner_garment_from_sam,
    )

    _OUTWEAR_FINE = {"short sleeve outwear", "long sleeve outwear"}
    best: Optional[dict] = None
    inner_elapsed = 0.0

    for inst in instances:
        coarse = inst.get("coarse_class_name", "")
        fine = (inst.get("fine_class_name") or inst.get("class_name") or "").strip().lower()
        is_outer = (coarse == "outerwear" or fine in _OUTWEAR_FINE)
        if not is_outer:
            continue

        t0 = time.perf_counter()
        inner = detect_inner_garment_from_sam(image_bgr, inst, sam_wrapper)
        inner_elapsed = time.perf_counter() - t0

        if inner is not None:
            best = inner
            break

    return {
        "status": "success" if best else "not_detected",
        "backend": "inner_garment",
        "bbox": best.get("bbox_xyxy") if best else None,
        "score": best.get("score") if best else None,
        "_elapsed": inner_elapsed,
        "_inner_found": best is not None,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("3.1.2 Region Localization Timing Benchmark")
    print(f"Images dir : {IMAGES_DIR}")
    print(f"Sample size: {NUM_IMAGES}")
    print(f"Queries    : {[q for q, _ in QUERIES]}")
    print("=" * 70)

    # ── Sample images ──────────────────────────────────────────────────────
    images = sample_images(IMAGES_DIR, NUM_IMAGES, RANDOM_SEED)
    print(f"\nSampled {len(images)} images.")

    # ── Load models ────────────────────────────────────────────────────────
    print("\n--- Loading models ---")
    t0 = time.perf_counter()
    pipeline = load_garment_pipeline()
    print(f"  GarmentPipeline loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    fp_detector = load_fp_detector()
    print(f"  Fashionpedia detector loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    dino_locator = load_dino_locator()
    print(f"  DINO locator loaded in {time.perf_counter() - t0:.1f}s")

    sam_wrapper = pipeline.get_sam_wrapper()
    print(f"  SAM wrapper ready: {sam_wrapper is not None}")

    # ── Benchmark loop ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details_csv = OUTPUT_DIR / "timing_details.csv"
    rows: List[Dict[str, Any]] = []

    import tempfile

    with tempfile.TemporaryDirectory(prefix="bench312_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for img_idx, img_path in enumerate(images):
            print(f"\n[{img_idx + 1}/{len(images)}] {img_path.name}")
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                print(f"  [SKIP] Could not read image")
                continue
            H, W = image_bgr.shape[:2]

            # ── Shared: run garment pipeline ────────────────────────────
            pipe_out = tmpdir_path / f"pipe_{img_idx}"
            t_pipe_start = time.perf_counter()
            try:
                pipe_result = pipeline.run_image(str(img_path), str(pipe_out))
            except Exception as e:
                print(f"  [FAIL] GarmentPipeline error: {e}")
                continue
            pipe_elapsed = time.perf_counter() - t_pipe_start
            print(f"  Pipeline: {pipe_elapsed:.1f}s (YOLO={pipe_result['timing'].get('yolo_seconds',0):.1f}s, SAM={pipe_result['timing'].get('sam_hq_seconds',0):.1f}s)")

            # ── Load instances ──────────────────────────────────────────
            seg_json_path = Path(pipe_result["paths"]["segmentation_json"])
            if not seg_json_path.exists():
                print(f"  [SKIP] No segmentation JSON at {seg_json_path}")
                continue

            seg_data = load_json(seg_json_path)
            instances: List[dict] = []
            for img_item in seg_data.get("images", []):
                for seg in img_item.get("segments", []):
                    inst = dict(seg)
                    inst.setdefault("pred_mask_path", inst.get("pred_mask_path") or inst.get("mask_path"))
                    instances.append(inst)

            if not instances:
                print(f"  [SKIP] No garment instances detected")
                continue
            print(f"  Instances: {len(instances)}")

            # Pick the best instance for Fashionpedia/DINO queries
            # (largest mask area = main garment)
            primary_inst = max(instances, key=_mask_area)

            # ── Run each query ──────────────────────────────────────────
            for query, expected_pathway in QUERIES:
                row = {
                    "image": img_path.name,
                    "image_idx": img_idx,
                    "query": query,
                    "expected_pathway": expected_pathway,
                    "pipeline_sec": pipe_elapsed,
                }

                try:
                    if expected_pathway == "inner_garment":
                        result = run_inner_query(image_bgr, instances, sam_wrapper)
                    elif expected_pathway == "grounding_dino":
                        result = run_dino_query(
                            query, primary_inst, image_bgr, W, H,
                            fp_detector, dino_locator, sam_wrapper,
                        )
                    else:  # fashionpedia_yolo
                        result = run_fp_query(
                            query, primary_inst, image_bgr, W, H,
                            fp_detector, sam_wrapper,
                        )

                    row["backend"] = result.get("backend", "?")
                    row["status"] = result.get("status", "?")
                    row["elapsed_sec"] = round(result.get("_elapsed", 0), 4)
                    row["score"] = result.get("score")
                    row["num_detections"] = (
                        len(result.get("debug", {}).get("dino_detections", []))
                        or len(result.get("debug", {}).get("fp_detections", []))
                        or (1 if result.get("status") == "success" else 0)
                    )
                    row["backend_ok"] = (
                        row["backend"] == expected_pathway
                        or (expected_pathway == "fashionpedia_yolo" and row["backend"] in ("fashionpedia_yolo", "fast_path"))
                        or (expected_pathway == "grounding_dino" and row["backend"] in ("open_vocab_grounding_dino", "zero_shot_grounding_dino", "grounding_dino"))
                    )
                except Exception as e:
                    row["backend"] = "error"
                    row["status"] = "error"
                    row["elapsed_sec"] = 0
                    row["error"] = str(e)[:200]
                    row["backend_ok"] = False

                rows.append(row)
                status_icon = "✓" if row.get("backend_ok") else "✗"
                print(f"  {status_icon} {query:<6} → backend={row['backend']:<32} status={row['status']:<14} {row['elapsed_sec']:.3f}s")

    # ── Save CSV ───────────────────────────────────────────────────────────
    fieldnames = [
        "image", "image_idx", "query", "expected_pathway", "pipeline_sec",
        "backend", "status", "elapsed_sec", "score", "num_detections",
        "backend_ok", "error",
    ]
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetails saved to: {details_csv}")

    # ── Summary stats ──────────────────────────────────────────────────────
    pathway_times: Dict[str, List[float]] = {}
    for r in rows:
        if r["status"] == "error":
            continue
        pathway = r["expected_pathway"]
        pathway_times.setdefault(pathway, []).append(r["elapsed_sec"])

    summary_rows = []
    for pathway in ["fashionpedia_yolo", "grounding_dino", "inner_garment"]:
        times = pathway_times.get(pathway, [])
        if not times:
            continue
        times_sorted = sorted(times)
        n = len(times_sorted)
        mean_t = sum(times_sorted) / n
        std_t = (sum((t - mean_t) ** 2 for t in times_sorted) / n) ** 0.5
        median_t = times_sorted[n // 2]
        p95_t = times_sorted[int(n * 0.95)]

        success_n = sum(
            1 for r in rows
            if r["expected_pathway"] == pathway and r["status"] == "success"
        )
        backend_ok_n = sum(
            1 for r in rows
            if r["expected_pathway"] == pathway and r.get("backend_ok")
        )

        summary_rows.append({
            "pathway": pathway,
            "count": n,
            "mean_sec": round(mean_t, 4),
            "std_sec": round(std_t, 4),
            "median_sec": round(median_t, 4),
            "p95_sec": round(p95_t, 4),
            "min_sec": round(times_sorted[0], 4),
            "max_sec": round(times_sorted[-1], 4),
            "success_rate": round(success_n / n, 3) if n else 0,
            "backend_ok_rate": round(backend_ok_n / n, 3) if n else 0,
        })

    summary_csv = OUTPUT_DIR / "timing_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV saved to: {summary_csv}")

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Pathway':<25} {'N':>4} {'Mean(s)':>8} {'Std(s)':>8} {'Median(s)':>9} {'P95(s)':>8} {'Succ%':>6}")
    print(f"{'-'*25} {'-'*4} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*6}")
    for s in summary_rows:
        print(
            f"{s['pathway']:<25} {s['count']:>4} "
            f"{s['mean_sec']:>8.3f} {s['std_sec']:>8.3f} "
            f"{s['median_sec']:>9.3f} {s['p95_sec']:>8.3f} "
            f"{s['success_rate']:>5.1%}"
        )

    # Backend routing check
    print(f"\n--- Backend routing check ---")
    routing_errors = [r for r in rows if not r.get("backend_ok")]
    if routing_errors:
        print(f"  ✗ {len(routing_errors)} routing errors (expected != actual backend):")
        for r in routing_errors[:10]:
            print(f"    {r['image']} | {r['query']} | expected={r['expected_pathway']} actual={r['backend']}")
    else:
        print(f"  ✓ All queries routed to expected backends")

    # ── Save JSON ──────────────────────────────────────────────────────────
    summary_json = OUTPUT_DIR / "timing_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "images_dir": str(IMAGES_DIR),
                "num_images": NUM_IMAGES,
                "queries": [q for q, _ in QUERIES],
            },
            "summary": summary_rows,
            "details": [
                {k: str(v) if isinstance(v, (Path,)) else v for k, v in r.items()}
                for r in rows
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"Full report saved to: {summary_json}")
    print("Done.")


if __name__ == "__main__":
    main()
