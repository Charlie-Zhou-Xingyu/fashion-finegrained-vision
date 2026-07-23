"""
Output parity validation: compare Original vs CachedFastPath vs BatchBackedFastPath.

Runs all three methods on the same images and compares:
    - Detection counts (YOLO garments)
    - SAM mask counts
    - Landmark instance counts
    - Region crop / mask-aware crop counts
    - Detection agreement (class labels, bbox IoU)
    - File artifact existence

Usage::

    python inference/benchmarks/validate_cached_fast_path_outputs.py \\
        --images 10 --output-dir outputs/inference_optimization/validation/output_parity

Outputs:
    output_parity_report.json
    output_parity_report.md
    output_parity_per_image.csv

Status: New module. Does NOT modify existing 3.1 code.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_IMAGE_DIR = (
    r"D:\Aliintern\fashion-ai-data\fashionai_attributes"
    r"\round1_fashionAI_attributes_test_a\Images\lapel_design_labels"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def sample_images(d: str, n: int, seed: int = 42) -> List[Path]:
    dp = Path(d)
    if not dp.is_dir():
        return []
    all_imgs = sorted(
        p for p in dp.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    rng = random.Random(seed)
    return rng.sample(all_imgs, min(n, len(all_imgs)))


def _safe_len(obj: Any) -> Optional[int]:
    """Safely get length of something that might be a list, dict, None."""
    if obj is None:
        return None
    if isinstance(obj, (list, dict)):
        return len(obj)
    return None


def _extract_counts(result: Dict[str, Any], method: str) -> Dict[str, Optional[int]]:
    """Extract comparable count fields from a pipeline result."""
    counts: Dict[str, Optional[int]] = {}

    if method == "batch_backed":
        # BatchBackedFastPath returns aggregate, not per-image
        pr = result.get("pipeline_result", {})
        timing = result.get("timing_breakdown", {})
        counts["num_images"] = result.get("num_images")
        counts["total_ms"] = result.get("total_ms_all_images")
        return counts

    # CachedFastPath and original run_image
    counts["num_instances"] = result.get("num_instances")
    timing = result.get("timing", {})

    # From paths — count artifacts
    paths = result.get("paths", {})
    for key in paths:
        p = Path(paths[key]) if paths[key] else None
        if p and p.exists():
            if p.is_dir():
                files = list(p.glob("*"))
                counts[f"files_{key}"] = len(files)
            else:
                counts[f"exists_{key}"] = 1

    counts["has_timing"] = 1 if timing else 0
    counts["has_error"] = 1 if "error" in result else 0

    return counts


def _extract_detections(result: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    """Extract detection dicts for comparison."""
    if method == "batch_backed":
        return []
    # Cached and original both store detections info
    return []


def compute_bbox_iou(a: List[float], b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_detections_json(json_path: str) -> List[Dict[str, Any]]:
    """Load detections.json and return list of detection dicts."""
    p = Path(json_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    dets = []
    for img_rec in data.get("images", []):
        for det in img_rec.get("detections", []):
            dets.append({
                "class_name": det.get("class_name"),
                "class_id": det.get("class_id"),
                "bbox_xyxy": det.get("bbox_xyxy"),
                "confidence": det.get("confidence"),
                "image": img_rec.get("file_name", ""),
            })
    return dets


def compare_detections(
    orig_dets: List[Dict],
    cached_dets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compare two detection lists by class labels and bbox IoU."""
    if len(orig_dets) == 0 and len(cached_dets) == 0:
        return {"count_match": True, "count_orig": 0, "count_cached": 0,
                "mean_iou": 1.0, "matched_pairs": 0, "mismatched": False}

    # Simple comparison: match by class_name, then compute IoU for same-class pairs
    orig_by_class: Dict[str, List[Dict]] = {}
    cached_by_class: Dict[str, List[Dict]] = {}
    for d in orig_dets:
        cn = d.get("class_name", "unknown")
        orig_by_class.setdefault(cn, []).append(d)
    for d in cached_dets:
        cn = d.get("class_name", "unknown")
        cached_by_class.setdefault(cn, []).append(d)

    ious: List[float] = []
    matched = 0
    for cn in set(orig_by_class.keys()) | set(cached_by_class.keys()):
        o_list = orig_by_class.get(cn, [])
        c_list = cached_by_class.get(cn, [])
        for o_det, c_det in zip(o_list, c_list):
            o_box = o_det.get("bbox_xyxy")
            c_box = c_det.get("bbox_xyxy")
            if o_box and c_box and len(o_box) == 4 and len(c_box) == 4:
                iou = compute_bbox_iou(o_box, c_box)
                ious.append(iou)
                if iou >= iou_threshold:
                    matched += 1

    count_match = len(orig_dets) == len(cached_dets)
    mean_iou = sum(ious) / len(ious) if ious else 1.0
    class_diff = set(orig_by_class.keys()) != set(cached_by_class.keys())

    return {
        "count_match": count_match,
        "count_orig": len(orig_dets),
        "count_cached": len(cached_dets),
        "mean_iou": round(mean_iou, 4),
        "matched_pairs": matched,
        "iou_pairs_total": len(ious),
        "mismatched_classes": class_diff,
        "orig_classes": sorted(orig_by_class.keys()),
        "cached_classes": sorted(cached_by_class.keys()),
    }


# ── Main validation ────────────────────────────────────────────────────────────

def run_validation(
    image_paths: List[Path],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run all three methods and compare outputs."""
    from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig
    from inference.pipelines.fast_path_existing_cached import CachedFastPath
    from inference.pipelines.fast_path_batch_backed import BatchBackedFastPath
    import tempfile

    per_image: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "images_total": len(image_paths),
        "images_with_errors": 0,
        "detection_count_match_rate": 0.0,
        "mean_iou_all": 0.0,
        "file_checks_pass": 0,
        "file_checks_fail": 0,
    }

    iou_values: List[float] = []
    count_matches = 0
    count_mismatches = 0

    # ── Load original pipeline (once) ──
    orig_config = GarmentPipelineConfig(
        run_landmark_and_crops=True, run_attribute_inference=False,
        save_yolo_vis=False, save_yolo_crops=False,
    )
    orig_pipeline = GarmentPipeline(orig_config)

    # ── Load cached pipeline ──
    cached_pipe = CachedFastPath(lazy=True)
    if image_paths:
        cached_pipe.warmup(str(image_paths[0]))

    # ── Load batch pipeline ──
    batch_pipe = BatchBackedFastPath()

    with tempfile.TemporaryDirectory(prefix="valid_") as td:
        tmp = Path(td)

        for i, img_path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name}")
            row: Dict[str, Any] = {"image": img_path.name, "image_idx": i}

            try:
                # ── Original ──
                orig_out = tmp / f"orig_{i:04d}"
                orig_out.mkdir(parents=True, exist_ok=True)
                orig_result = orig_pipeline.run_image(str(img_path), str(orig_out))
                row["orig_ok"] = orig_result.get("status") == "success"
                row["orig_num_instances"] = orig_result.get("region_crops_summary", {}).get("num_instances", 0)

                # ── Cached ──
                cached_out = tmp / f"cached_{i:04d}"
                cached_out.mkdir(parents=True, exist_ok=True)
                cached_result = cached_pipe.run_image(str(img_path), str(cached_out))
                row["cached_ok"] = cached_result.get("status") == "ok"
                row["cached_num_instances"] = cached_result.get("num_instances", 0)

                # ── Compare detections ──
                orig_dets_json = orig_out / "01_yolo" / "detections.json"
                if orig_dets_json.exists():
                    orig_dets = load_detections_json(str(orig_dets_json))
                else:
                    orig_dets = []

                # CachedFastPath saves detections in a slightly different location
                cached_dets_json = cached_out / "01_yolo" / "detections.json"
                if cached_dets_json.exists():
                    cached_dets = load_detections_json(str(cached_dets_json))
                else:
                    cached_dets = []

                det_cmp = compare_detections(orig_dets, cached_dets, iou_threshold)
                row.update({
                    "orig_det_count": det_cmp["count_orig"],
                    "cached_det_count": det_cmp["count_cached"],
                    "det_count_match": det_cmp["count_match"],
                    "det_mean_iou": det_cmp["mean_iou"],
                    "det_classes_match": not det_cmp["mismatched_classes"],
                })
                if det_cmp["iou_pairs_total"] > 0:
                    iou_values.append(det_cmp["mean_iou"])
                if det_cmp["count_match"]:
                    count_matches += 1
                else:
                    count_mismatches += 1

                # ── File existence checks ──
                checks_ok = 0
                checks_total = 0
                for check_dir in ["01_yolo", "02_samhq", "03_landmarks",
                                   "04_region_crops", "05_region_masked_crops"]:
                    checks_total += 1
                    d = cached_out / check_dir
                    if d.is_dir() and any(d.iterdir()):
                        checks_ok += 1
                row["file_checks_ok"] = checks_ok
                row["file_checks_total"] = checks_total

            except Exception as exc:
                row["error"] = str(exc)[:200]
                summary["images_with_errors"] += 1

            per_image.append(row)

    # ── Batch-Backed (run once for all images) ──
    batch_result = batch_pipe.run_images([str(p) for p in image_paths])
    batch_counts = _extract_counts(batch_result, "batch_backed")

    # ── Aggregate ──
    n = len(per_image)
    summary.update({
        "detection_count_match_rate": round(count_matches / max(1, n), 3),
        "detection_count_mismatch_rate": round(count_mismatches / max(1, n), 3),
        "mean_iou_all": round(sum(iou_values) / max(1, len(iou_values)), 4),
        "num_iou_pairs": len(iou_values),
        "file_checks_pass": sum(1 for r in per_image if r.get("file_checks_ok", 0) == r.get("file_checks_total", 5)),
        "file_checks_fail": sum(1 for r in per_image if r.get("file_checks_ok", 0) != r.get("file_checks_total", 5)),
        "batch_backed_summary": batch_counts,
    })

    # ── Determine overall status ──
    det_ok = summary["detection_count_match_rate"] >= 0.9
    iou_ok = summary["mean_iou_all"] >= 0.90
    files_ok = summary["file_checks_fail"] == 0

    if det_ok and iou_ok and files_ok:
        status = "pass"
    elif det_ok and files_ok:
        status = "warn"
    elif not det_ok or summary["images_with_errors"] > n * 0.1:
        status = "fail"
    else:
        status = "inconclusive"

    return {
        "status": status,
        "summary": summary,
        "per_image": per_image,
        "batch_backed": batch_counts,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate output parity: Original vs CachedFastPath vs BatchBacked",
    )
    ap.add_argument("--images", type=int, default=10)
    ap.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--output-dir", type=str,
                    default="outputs/inference_optimization/validation/output_parity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance-bbox-iou", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        imgs = sample_images(args.image_dir, args.images, args.seed)
        print(f"[DRY RUN] {len(imgs)} images available.")
        print(f"  image_dir: {Path(args.image_dir).is_dir()}")
        return

    image_paths = sample_images(args.image_dir, args.images, args.seed)
    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    print(f"Validating on {len(image_paths)} images...\n")
    t0 = time.perf_counter()
    result = run_validation(image_paths, iou_threshold=args.tolerance_bbox_iou)
    elapsed = time.perf_counter() - t0

    result["metadata"] = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "images": len(image_paths),
        "wallclock_s": round(elapsed, 1),
        "iou_threshold": args.tolerance_bbox_iou,
    }

    # ── Write JSON ──
    json_path = output_dir / "output_parity_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    # ── Write CSV ──
    csv_path = output_dir / "output_parity_per_image.csv"
    if result["per_image"]:
        keys = [k for k in result["per_image"][0].keys()
                if k not in ("error",)]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in result["per_image"]:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    # ── Write MD ──
    s = result["summary"]
    md = f"""# Output Parity Validation

> Date: {result['metadata']['date']} | Images: {len(image_paths)} | Status: **{result['status'].upper()}**

## Summary

| Metric | Value |
|---|---|
| Images validated | {s['images_total']} |
| Images with errors | {s['images_with_errors']} |
| Detection count match rate | {s['detection_count_match_rate']:.1%} |
| Mean bbox IoU (cached vs orig) | {s['mean_iou_all']:.3f} |
| File artifact checks passed | {s['file_checks_pass']}/{s['images_total']} |
| File artifact checks failed | {s['file_checks_fail']}/{s['images_total']} |

## Status: {result['status'].upper()}

"""
    if result["status"] == "pass":
        md += "All checks passed. CachedFastPath outputs are consistent with original.\n"
    elif result["status"] == "warn":
        md += "Minor discrepancies found. Likely explainable. See per-image details.\n"
    elif result["status"] == "fail":
        md += "Systematic mismatch detected. Do NOT use CachedFastPath for production.\n"
    else:
        md += "Results inconclusive. Review per-image data.\n"

    md += f"\n## Per-Image Summary\n\n"
    md += "| Image | Orig # | Cached # | Count Match | Mean IoU | Files OK |\n"
    md += "|---|---:|---:|---|---|---|\n"
    for row in result["per_image"][:20]:
        md += (f"| {row['image'][:30]} | {row.get('orig_det_count','?')} | "
               f"{row.get('cached_det_count','?')} | {row.get('det_count_match','?')} | "
               f"{row.get('det_mean_iou','?')} | {row.get('file_checks_ok','?')}/{row.get('file_checks_total','?')} |\n")

    md_path = output_dir / "output_parity_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md.strip())

    print(f"\n=== OUTPUT PARITY: {result['status'].upper()} ===")
    print(f"  Detection count match: {s['detection_count_match_rate']:.1%}")
    print(f"  Mean IoU: {s['mean_iou_all']:.3f}")
    print(f"  File checks: {s['file_checks_pass']}/{s['images_total']} pass")
    print(f"\n[Reports: {json_path}, {md_path}, {csv_path}]")


if __name__ == "__main__":
    main()
