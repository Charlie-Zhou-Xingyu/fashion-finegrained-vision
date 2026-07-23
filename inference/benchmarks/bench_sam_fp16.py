"""
SAM-HQ FP32 vs FP16 autocast benchmark.

Measures set_image + predict timing with full statistics (p50/p95/max).
Outputs structured JSON to outputs/benchmarks/.

Usage:
    python inference/benchmarks/bench_sam_fp16.py
    python inference/benchmarks/bench_sam_fp16.py --n-bench 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from inference.wrappers.sam_wrapper import SamHqWrapper, mask_iou, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM-HQ FP32 vs FP16 benchmark")
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-bench", type=int, default=10)
    p.add_argument("--checkpoint", default="checkpoints/sam_hq/sam_hq_vit_b.pth")
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--image-dir", default="D:/Aliintern/fashion-ai-data/deepfashion2/validation/image")
    p.add_argument("--num-images", type=int, default=5, help="Number of test images")
    p.add_argument("--output-dir", default="outputs/benchmarks")
    return p.parse_args()


def load_test_cases(args) -> list[dict]:
    """Collect test images and their bboxes from YOLO detections."""
    img_dir = Path(args.image_dir)
    cases = []

    # Try to get bbox from existing pipeline outputs first.
    demo_root = _PROJECT_ROOT / "outputs" / "full_31x_demo"
    for img_path in sorted(img_dir.glob("*.jpg"))[: args.num_images * 3]:
        img_id = img_path.stem
        # Look for pre-computed YOLO detections.
        det_path = demo_root / img_id / "01_yolo" / "detections.json"
        bboxes = []
        if det_path.exists():
            data = json.loads(det_path.read_text(encoding="utf-8"))
            for img_rec in data.get("images", []):
                for det in img_rec.get("detections", []):
                    bboxes.append(det["bbox_xyxy"])
        if not bboxes:
            # No pre-computed — use a synthetic center crop box.
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            bboxes.append([w * 0.2, h * 0.2, w * 0.8, h * 0.8])

        cases.append({
            "image_id": img_id,
            "image_path": str(img_path),
            "bboxes": bboxes,
        })
        if len(cases) >= args.num_images:
            break

    return cases


def bench_one_config(
    wrapper: SamHqWrapper,
    image_rgb: np.ndarray,
    boxes: list,
    n_warmup: int,
    n_bench: int,
) -> dict:
    """Benchmark one FP config on one image. Returns timing arrays."""
    box_np = np.array(boxes, dtype=np.float32)

    # Warmup.
    for _ in range(n_warmup):
        wrapper.set_image(image_rgb)
        wrapper.predict_boxes(box_np)

    torch.cuda.synchronize()

    set_ms: list[float] = []
    pred_ms: list[float] = []

    for _ in range(n_bench):
        wrapper.set_image(image_rgb)
        set_ms.append(wrapper.last_timing.get("set_image_ms", 0))

        wrapper.predict_boxes(box_np)
        pred_ms.append(wrapper.last_timing.get("predict_boxes_ms", 0))

    return {
        "set_image_ms": stats(set_ms),
        "predict_ms": stats(pred_ms),
        "total_ms": stats([s + p for s, p in zip(set_ms, pred_ms)]),
    }


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("[ERROR] CUDA not available. Use --device cpu")
        sys.exit(1)

    device_name = (torch.cuda.get_device_name(0)
                   if args.device == "cuda" and torch.cuda.is_available()
                   else "CPU")
    print(f"Device: {device_name}")

    cases = load_test_cases(args)
    print(f"Test images: {len(cases)}")

    results: dict = {
        "benchmark": "sam_fp16",
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "gpu": device_name,
        "model": f"SAM-HQ {args.model_type}",
        "checkpoint": args.checkpoint,
        "n_warmup": args.n_warmup,
        "n_bench": args.n_bench,
        "images": [],
    }

    fp32_totals: list[float] = []
    fp16_totals: list[float] = []

    for case in cases:
        img_id = case["image_id"]
        bboxes = case["bboxes"]
        print(f"\n--- {img_id} ({len(bboxes)} box(es)) ---")

        image_bgr = cv2.imread(case["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ── FP32 ──────────────────────────────────────────────────
        w32 = SamHqWrapper(
            checkpoint=args.checkpoint, model_type=args.model_type,
            device=args.device, use_fp16=False,
        )
        fp32_result = bench_one_config(
            w32, image_rgb, bboxes, args.n_warmup, args.n_bench,
        )
        print(f"  FP32:  set={fp32_result['set_image_ms']['mean']:.1f}ms  "
              f"pred={fp32_result['predict_ms']['mean']:.1f}ms  "
              f"total={fp32_result['total_ms']['mean']:.1f}ms")

        # Get FP32 reference masks for IoU comparison.
        w32.set_image(image_rgb)
        ref_masks, _, _ = w32.predict_boxes(np.array(bboxes, dtype=np.float32))
        del w32
        torch.cuda.empty_cache()

        # ── FP16 autocast ─────────────────────────────────────────
        w16 = SamHqWrapper(
            checkpoint=args.checkpoint, model_type=args.model_type,
            device=args.device, use_fp16=True,
        )
        fp16_result = bench_one_config(
            w16, image_rgb, bboxes, args.n_warmup, args.n_bench,
        )
        print(f"  FP16:  set={fp16_result['set_image_ms']['mean']:.1f}ms  "
              f"pred={fp16_result['predict_ms']['mean']:.1f}ms  "
              f"total={fp16_result['total_ms']['mean']:.1f}ms")

        # IoU check.
        w16.set_image(image_rgb)
        fp16_masks, _, _ = w16.predict_boxes(np.array(bboxes, dtype=np.float32))
        del w16
        torch.cuda.empty_cache()

        ious: list[float] = []
        for m_ref, m_fp16 in zip(ref_masks, fp16_masks):
            iou = mask_iou(m_ref, m_fp16)
            ious.append(iou)

        iou_mean = float(np.mean(ious))
        if iou_mean < 0.995:
            print(f"  WARN: IoU(fp32,fp16)={iou_mean:.6f} < 0.995 - mask drift detected")
        else:
            print(f"  IoU(fp32,fp16)={iou_mean:.6f}")

        speedup = (fp32_result["total_ms"]["mean"] /
                   fp16_result["total_ms"]["mean"]
                   if fp16_result["total_ms"]["mean"] > 0 else 0)
        print(f"  Speedup: {speedup:.2f}x")

        fp32_totals.append(fp32_result["total_ms"]["mean"])
        fp16_totals.append(fp16_result["total_ms"]["mean"])

        results["images"].append({
            "image_id": img_id,
            "image_path": case["image_path"],
            "num_boxes": len(bboxes),
            "bboxes": bboxes,
            "fp32": fp32_result,
            "fp16_autocast": fp16_result,
            "mask_iou_fp32_vs_fp16": iou_mean,
            "speedup": round(speedup, 3),
        })

    # ── Summary ──────────────────────────────────────────────────────
    results["summary"] = {
        "fp32_total_mean_ms": round(float(np.mean(fp32_totals)), 2),
        "fp16_total_mean_ms": round(float(np.mean(fp16_totals)), 2),
        "speedup": round(
            float(np.mean(fp32_totals) / np.mean(fp16_totals))
            if np.mean(fp16_totals) > 0 else 0, 3,
        ),
        "set_image_speedup": round(
            float(np.mean([
                i["fp32"]["set_image_ms"]["mean"] /
                max(i["fp16_autocast"]["set_image_ms"]["mean"], 0.1)
                for i in results["images"]
            ])), 3,
        ),
        "min_mask_iou": round(float(min(
            i["mask_iou_fp32_vs_fp16"] for i in results["images"]
        )), 6),
    }

    # ── Write JSON ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sam_fp16_{ts}.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON → {out_path}")

    # ── Print summary ────────────────────────────────────────────────
    s = results["summary"]
    print(f"\n=== SUMMARY ===")
    print(f"  FP32 total mean:  {s['fp32_total_mean_ms']:.1f}ms")
    print(f"  FP16 total mean:  {s['fp16_total_mean_ms']:.1f}ms")
    print(f"  Speedup:          {s['speedup']:.2f}x")
    print(f"  set_image speedup:{s['set_image_speedup']:.2f}x")
    print(f"  Min mask IoU:     {s['min_mask_iou']:.6f}")
    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
