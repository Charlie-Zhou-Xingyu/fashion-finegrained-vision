#!/usr/bin/env python3
"""
Calibrate part shape-prior thresholds from eval_v2 true-positive detection data.

Reads ``data/validation/eval_v2/per_result.jsonl``, extracts geometric
statistics from all TP detections per part, and outputs calibrated thresholds
(μ ± 2σ or P5/P95) alongside the current author-estimated values.

Usage::

    python scripts/calibrate_shape_priors_from_eval.py
    python scripts/calibrate_shape_priors_from_eval.py --min-samples 5 --output configs/shape_priors_calibrated_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EVAL_JSONL = PROJECT_ROOT / "data/validation/eval_v2/per_result.jsonl"
OUTPUT_YAML = PROJECT_ROOT / "configs/shape_priors_calibrated_v1.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_stats(bbox_xyxy: List[float]) -> dict:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "w": w, "h": h, "area": w * h,
        "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
        "aspect_hw": h / w, "aspect_wh": w / h,
    }


def _shape_stats_for_pred(
    pred_bbox_xyxy: List[float],
    garment_bbox_xyxy: Optional[List[float]],
) -> Dict[str, float]:
    """Compute shape-prior metrics for one detection relative to garment bbox."""
    p = _bbox_stats(pred_bbox_xyxy)
    stats: Dict[str, float] = {
        "area_abs": p["area"],
        "aspect_hw": p["aspect_hw"],
        "aspect_wh": p["aspect_wh"],
    }
    if garment_bbox_xyxy:
        g = _bbox_stats(garment_bbox_xyxy)
        stats["area_ratio"] = p["area"] / max(1.0, g["area"])
        stats["center_x_offset"] = abs(p["cx"] - g["cx"]) / max(1.0, g["w"])
        stats["center_y_offset"] = abs(p["cy"] - g["cy"]) / max(1.0, g["h"])
        # Normalised coords within garment bbox (for y_band / x_band)
        stats["cy_norm"] = (p["cy"] - g["y1"]) / max(1.0, g["h"])
        stats["cx_norm"] = (p["cx"] - g["x1"]) / max(1.0, g["w"])
    return stats


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy)."""
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sv[int(k)]
    return sv[f] * (c - k) + sv[c] * (k - f)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    return mu, math.sqrt(var)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_current_config() -> Dict[str, dict]:
    """Load current shape-prior configs from part_detection_config.py."""
    from fashion_vision.localization.part_detection_config import PART_DETECTION_CONFIG

    out: Dict[str, dict] = {}
    for part, cfg in PART_DETECTION_CONFIG.items():
        shape = cfg.get("shape", {})
        if shape:
            out[part] = dict(shape)
    return out


def collect_tp_stats(
    jsonl_path: Path, min_iou: float = 0.01,
) -> Dict[str, List[Dict[str, float]]]:
    """Collect per-part TP detection geometry statistics.

    Returns ``{part: [stat_dict, ...]}`` where each stat_dict contains
    area_ratio, aspect_hw, aspect_wh, center_x_offset, cy_norm, cx_norm.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"per_result.jsonl not found at {jsonl_path}")

    per_part: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            # Only consider TP records (at the loosest IoU threshold — 0.01)
            if not rec.get("is_hit_iou_0.01"):
                continue
            part = rec["part"]
            gt_bbox = rec.get("gt_bbox")
            garment_bbox = rec.get("garment_bbox")
            pred_bboxes = rec.get("pred_bboxes", [])

            for pb in pred_bboxes:
                # Only use the prediction that matched (best IoU > min_iou)
                # Since we can't compute per-pred IoU without the GT, use all
                # predictions from hit records — they contributed to the hit.
                stats = _shape_stats_for_pred(pb, garment_bbox)
                per_part[part].append(stats)

    return dict(per_part)


def derive_thresholds(
    samples: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Derive calibrated thresholds from a list of per-detection stat dicts."""
    if len(samples) < 3:
        return {"n_samples": len(samples), "warning": "too_few_samples"}

    keys = ["area_ratio", "aspect_hw", "aspect_wh", "center_x_offset",
            "cy_norm", "cx_norm"]
    result: Dict[str, Any] = {"n_samples": len(samples)}

    for key in keys:
        vals = [s[key] for s in samples if key in s]
        if not vals:
            continue
        mu, std = _mean_std(vals)
        p5 = _percentile(vals, 5)
        p95 = _percentile(vals, 95)
        lo_2sig = max(0.0, mu - 2 * std)
        hi_2sig = mu + 2 * std

        result[f"{key}"] = {
            "mean": round(mu, 4),
            "std": round(std, 4),
            "p5": round(p5, 4),
            "p95": round(p95, 4),
            "lo_2sigma": round(lo_2sig, 4),
            "hi_2sigma": round(hi_2sig, 4),
        }

    return result


def build_calibrated_shape_config(
    per_part_stats: Dict[str, Dict[str, Any]],
    current_config: Dict[str, dict],
) -> Dict[str, dict]:
    """Build calibrated shape-prior config dict, annotated with old values."""
    calibrated: Dict[str, dict] = {}

    for part, derived in sorted(per_part_stats.items()):
        n = derived.get("n_samples", 0)
        if n < 5:
            calibrated[part] = {
                "_n_tp_samples": n,
                "_status": "insufficient_data",
                "_note": "Keep current thresholds; not enough TP data to calibrate.",
            }
            continue

        entry: Dict[str, Any] = {"_n_tp_samples": n, "_status": "calibrated"}

        # --- area_ratio ---
        ar = derived.get("area_ratio", {})
        old_cfg = current_config.get(part, {})
        entry["min_area_ratio"] = round(ar.get("p5", 0.0), 4)
        entry["max_area_ratio"] = round(ar.get("p95", 1.0), 4)
        if "min_area_ratio" in old_cfg:
            entry["_old_min_area_ratio"] = old_cfg["min_area_ratio"]
        if "max_area_ratio" in old_cfg:
            entry["_old_max_area_ratio"] = old_cfg["max_area_ratio"]

        # --- aspect_hw (verticality) ---
        ahw = derived.get("aspect_hw", {})
        if ahw:
            entry["min_aspect_ratio_h_over_w"] = round(ahw.get("p5", 0.0), 2)
            entry["max_aspect_ratio_h_over_w"] = round(ahw.get("p95", 100.0), 2)
            if "min_aspect_ratio_h_over_w" in old_cfg:
                entry["_old_min_aspect_ratio_h_over_w"] = old_cfg["min_aspect_ratio_h_over_w"]
            if "max_aspect_ratio_h_over_w" in old_cfg:
                entry["_old_max_aspect_ratio_h_over_w"] = old_cfg["max_aspect_ratio_h_over_w"]

        # --- aspect_wh (horizontality) ---
        awh = derived.get("aspect_wh", {})
        if awh:
            entry["min_aspect_ratio_w_over_h"] = round(awh.get("p5", 0.0), 2)
            entry["max_aspect_ratio_w_over_h"] = round(awh.get("p95", 100.0), 2)
            if "min_aspect_ratio_w_over_h" in old_cfg:
                entry["_old_min_aspect_ratio_w_over_h"] = old_cfg["min_aspect_ratio_w_over_h"]
            if "max_aspect_ratio_w_over_h" in old_cfg:
                entry["_old_max_aspect_ratio_w_over_h"] = old_cfg["max_aspect_ratio_w_over_h"]

        # --- center_x_offset ---
        cxo = derived.get("center_x_offset", {})
        if cxo and "prefer_center_x" in old_cfg:
            # Set center_x_tolerance at 95th percentile of TP center offsets
            p95_offset = round(cxo.get("p95", 0.35), 2)
            entry["prefer_center_x"] = True
            entry["center_x_tolerance"] = max(p95_offset, 0.10)  # at least 10% tolerance
            entry["_old_center_x_tolerance"] = old_cfg.get("center_x_tolerance")

        # --- y_band ---
        cyn = derived.get("cy_norm", {})
        if cyn and "y_band" in old_cfg:
            # Expand y_band by 0.05 margin on each side relative to TP range
            lo = max(0.0, cyn.get("p5", 0.0) - 0.05)
            hi = min(1.0, cyn.get("p95", 1.0) + 0.05)
            entry["y_band"] = [round(lo, 2), round(hi, 2)]
            entry["_old_y_band"] = old_cfg["y_band"]

        # --- x_band ---
        cxn = derived.get("cx_norm", {})
        if cxn and "x_band" in old_cfg:
            lo = max(0.0, cxn.get("p5", 0.0) - 0.05)
            hi = min(1.0, cxn.get("p95", 1.0) + 0.05)
            entry["x_band"] = [round(lo, 2), round(hi, 2)]
            entry["_old_x_band"] = old_cfg["x_band"]

        # --- mask_dilation_px ---
        if "mask_dilation_px" in old_cfg:
            entry["mask_dilation_px"] = old_cfg["mask_dilation_px"]

        calibrated[part] = entry

    return calibrated


def print_comparison(
    calibrated: Dict[str, dict],
    current: Dict[str, dict],
):
    """Print a side-by-side comparison table."""
    print(f"\n{'=' * 90}")
    print("SHAPE PRIOR CALIBRATION REPORT")
    print(f"{'=' * 90}")

    for part in sorted(set(list(calibrated) + list(current))):
        cal = calibrated.get(part, {})
        cur = current.get(part, {})
        n = cal.get("_n_tp_samples", 0)
        status = cal.get("_status", "not_calibrated")

        print(f"\n── {part}  (n_TP={n}, status={status}) ──")

        def _cmp(key: str, fmt: str = ".3f", label: str = None):
            old_v = cur.get(key)
            new_v = cal.get(key)
            if label is None:
                label = key
            if old_v is not None and new_v is not None:
                if isinstance(old_v, list):
                    old_s = f"[{old_v[0]:.2f}, {old_v[1]:.2f}]"
                    new_s = f"[{new_v[0]:.2f}, {new_v[1]:.2f}]"
                else:
                    old_s = f"{old_v:{fmt}}"
                    new_s = f"{new_v:{fmt}}"
                print(f"  {label:<30s}  {old_s:<12s} → {new_s}")
            elif old_v is not None:
                print(f"  {label:<30s}  {old_v}  (no calibration data)")
            elif new_v is not None:
                print(f"  {label:<30s}  (new) → {new_v}")

        _cmp("min_area_ratio", ".4f")
        _cmp("max_area_ratio", ".4f")
        _cmp("min_aspect_ratio_h_over_w", ".2f")
        _cmp("max_aspect_ratio_h_over_w", ".2f")
        _cmp("min_aspect_ratio_w_over_h", ".2f")
        _cmp("max_aspect_ratio_w_over_h", ".2f")
        _cmp("center_x_tolerance", ".2f")
        _cmp("y_band")
        _cmp("x_band")

    # Summary line
    n_calibrated = sum(1 for v in calibrated.values() if v.get("_status") == "calibrated")
    n_insufficient = sum(1 for v in calibrated.values() if v.get("_status") == "insufficient_data")
    print(f"\n{'=' * 90}")
    print(f"SUMMARY: {n_calibrated} parts calibrated, {n_insufficient} with insufficient data")
    print(f"Review the YAML output before applying thresholds.")
    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate shape-prior thresholds from eval_v2 TP data")
    parser.add_argument("--eval-jsonl", type=str, default=str(EVAL_JSONL),
                        help="Path to per_result.jsonl")
    parser.add_argument("--output", type=str, default=str(OUTPUT_YAML),
                        help="Output YAML path for calibrated config")
    parser.add_argument("--min-iou", type=float, default=0.01,
                        help="Minimum IoU to consider a detection as TP")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Minimum TP samples required to calibrate a part")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print comparison without writing YAML")
    args = parser.parse_args()

    jsonl_path = Path(args.eval_jsonl)
    output_path = Path(args.output)

    # 1. Collect TP detection stats
    print(f"Reading TP detection stats from: {jsonl_path}")
    per_part_stats_raw = collect_tp_stats(jsonl_path, args.min_iou)

    # 2. Derive thresholds per part
    per_part_derived = {}
    for part, samples in sorted(per_part_stats_raw.items()):
        derived = derive_thresholds(samples)
        if derived["n_samples"] >= args.min_samples:
            per_part_derived[part] = derived
        else:
            per_part_derived[part] = {
                "n_samples": derived["n_samples"],
                "warning": "too_few_samples",
            }

    # 3. Load current config
    current_config = load_current_config()

    # 4. Build calibrated shape config
    calibrated = build_calibrated_shape_config(per_part_derived, current_config)

    # 5. Print comparison
    print_comparison(calibrated, current_config)

    # 6. Write YAML
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            yaml.dump(
                calibrated,
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        print(f"\nCalibrated config written to: {output_path}")
        print("Review before applying to part_detection_config.py.")


if __name__ == "__main__":
    main()
