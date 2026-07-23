"""
Visualize FP16 mask drift for low-IoU cases from sam_fp16 benchmark.

Generates overlay_fp32.png, overlay_fp16.png, and diff.png for each case
where mask IoU < 0.995.

Usage:
    python inference/benchmarks/visualize_sam_fp16_drift.py
    python inference/benchmarks/visualize_sam_fp16_drift.py \
        --benchmark-json outputs/benchmarks/sam_fp16_20260723_115011.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from inference.wrappers.sam_wrapper import SamHqWrapper, mask_iou

IOU_THRESHOLD = 0.995
DEFAULT_OUTPUT_DIR = "outputs/benchmarks/sam_fp16_vis"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize SAM FP16 mask drift")
    p.add_argument(
        "--benchmark-json", type=Path,
        help="Path to sam_fp16 benchmark JSON (default: latest)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
    )
    p.add_argument(
        "--iou-threshold", type=float, default=IOU_THRESHOLD,
        help="Only visualize cases with IoU below this threshold",
    )
    return p.parse_args()


def find_latest_benchmark() -> Path:
    """Find the most recent sam_fp16 benchmark JSON."""
    pattern = str(_PROJECT_ROOT / "outputs" / "benchmarks" / "sam_fp16_*.json")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No benchmark JSON found matching: {pattern}")
    return Path(files[-1])


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    """Overlay mask on image with semi-transparent color.

    Parameters
    ----------
    image_rgb : HxWx3 uint8.
    mask : HxW bool.
    color : (B, G, R) tuple for overlay.
    """
    overlay = image_rgb.copy()
    mask_vis = np.asarray(mask, dtype=bool)
    alpha = 0.4
    for c in range(3):
        overlay[mask_vis, c] = (
            overlay[mask_vis, c] * (1 - alpha) + color[c] * alpha
        ).astype(np.uint8)
    return overlay


def make_diff(
    image_rgb: np.ndarray,
    mask_fp32: np.ndarray,
    mask_fp16: np.ndarray,
) -> np.ndarray:
    """Create diff viz: red=fp32-only, blue=fp16-only, green=overlap, grey bg."""
    m32 = np.asarray(mask_fp32, dtype=bool)
    m16 = np.asarray(mask_fp16, dtype=bool)

    # Background: desaturated original.
    grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB).astype(np.float32)
    bg = (bg * 0.3).astype(np.uint8)  # darken

    diff = bg.copy().astype(np.int32)

    overlap = m32 & m16
    fp32_only = m32 & ~m16
    fp16_only = m16 & ~m32

    # Green = overlap (B=0, G=200, R=0)
    diff[overlap, 1] = np.clip(diff[overlap, 1] + 200, 0, 255)
    # Blue = FP16-only (B=200, G=0, R=0)
    diff[fp16_only, 0] = np.clip(diff[fp16_only, 0] + 200, 0, 255)
    # Red = FP32-only (B=0, G=0, R=200)
    diff[fp32_only, 2] = np.clip(diff[fp32_only, 2] + 200, 0, 255)

    return np.clip(diff, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()

    # Locate benchmark JSON.
    bench_path = args.benchmark_json or find_latest_benchmark()
    print(f"Benchmark JSON: {bench_path}")
    bench = json.loads(bench_path.read_text(encoding="utf-8"))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = bench.get("checkpoint", "checkpoints/sam_hq/sam_hq_vit_b.pth")
    model_type = bench.get("model", "vit_b").replace("SAM-HQ ", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary: dict = {
        "source_benchmark": str(bench_path),
        "timestamp": datetime.now().isoformat(),
        "iou_threshold": args.iou_threshold,
        "cases": [],
    }

    for img_data in bench.get("images", []):
        iou = img_data.get("mask_iou_fp32_vs_fp16", 1.0)
        if iou >= args.iou_threshold:
            print(f"  {img_data['image_id']}: IoU={iou:.4f} >= {args.iou_threshold} — skip")
            continue

        img_id = img_data["image_id"]
        img_path = img_data.get("image_path", "")
        bboxes = img_data.get("bboxes", [])

        print(f"  {img_id}: IoU={iou:.4f} — generating viz")

        # Read image.
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"    WARN: cannot read {img_path}")
            summary["cases"].append({
                "image_id": img_id, "iou": iou, "error": "image_read_failed",
            })
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate FP32 masks.
        w32 = SamHqWrapper(
            checkpoint=checkpoint, model_type=model_type,
            device=device, use_fp16=False,
        )
        w32.set_image(image_rgb)
        masks_fp32, _, _ = w32.predict_boxes(np.array(bboxes, dtype=np.float32))
        del w32
        torch.cuda.empty_cache()

        # Generate FP16 masks.
        w16 = SamHqWrapper(
            checkpoint=checkpoint, model_type=model_type,
            device=device, use_fp16=True,
        )
        w16.set_image(image_rgb)
        masks_fp16, _, _ = w16.predict_boxes(np.array(bboxes, dtype=np.float32))
        del w16
        torch.cuda.empty_cache()

        # Per-box masks — combine for overlay, ensure 2D.
        combined_fp32 = np.squeeze(np.any(masks_fp32, axis=0))
        combined_fp16 = np.squeeze(np.any(masks_fp16, axis=0))
        if combined_fp32.ndim != 2 or combined_fp16.ndim != 2:
            combined_fp32 = combined_fp32.reshape(image_rgb.shape[:2])
            combined_fp16 = combined_fp16.reshape(image_rgb.shape[:2])

        # Compute statistics.
        inter = np.logical_and(combined_fp32, combined_fp16).sum()
        union = np.logical_or(combined_fp32, combined_fp16).sum()
        fp32_only = np.logical_and(combined_fp32, ~combined_fp16).sum()
        fp16_only = np.logical_and(combined_fp16, ~combined_fp32).sum()

        # Generate images.
        overlay_fp32 = make_overlay(image_rgb, combined_fp32, color=(0, 200, 0))
        overlay_fp16 = make_overlay(image_rgb, combined_fp16, color=(200, 100, 0))
        diff_img = make_diff(image_rgb, combined_fp32, combined_fp16)

        # BGR for cv2.imwrite.
        ov32_path = out_dir / f"{img_id}_overlay_fp32.png"
        ov16_path = out_dir / f"{img_id}_overlay_fp16.png"
        diff_path = out_dir / f"{img_id}_diff.png"

        cv2.imwrite(str(ov32_path), cv2.cvtColor(overlay_fp32, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(ov16_path), cv2.cvtColor(overlay_fp16, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(diff_path), diff_img)

        summary["cases"].append({
            "image_id": img_id,
            "iou": iou,
            "fp32_area": int(combined_fp32.sum()),
            "fp16_area": int(combined_fp16.sum()),
            "intersection_area": int(inter),
            "union_area": int(union),
            "fp32_only_area": int(fp32_only),
            "fp16_only_area": int(fp16_only),
            "diff_ratio": round(float(fp32_only + fp16_only) / max(union, 1), 6),
            "files": {
                "overlay_fp32": str(ov32_path),
                "overlay_fp16": str(ov16_path),
                "diff": str(diff_path),
            },
        })
        print(f"    fp32_only={fp32_only}px  fp16_only={fp16_only}px  "
              f"diff_ratio={summary['cases'][-1]['diff_ratio']:.4f}")

    # Write summary.
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")
    print(f"Cases visualized: {len(summary['cases'])}")
    print("[DONE]")


if __name__ == "__main__":
    main()
