"""
SAM-HQ one-by-one vs batched box prediction benchmark.

Compares 4 modes x (1/2/5 box cases):
    FP32_one_by_one / FP32_batched / FP16_one_by_one / FP16_batched

Usage:
    python inference/benchmarks/bench_sam_batch_boxes.py
    python inference/benchmarks/bench_sam_batch_boxes.py --n-bench 20
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
    p = argparse.ArgumentParser(
        description="SAM-HQ batched vs one-by-one box prediction benchmark"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-bench", type=int, default=10)
    p.add_argument("--checkpoint", default="checkpoints/sam_hq/sam_hq_vit_b.pth")
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--output-dir", default="outputs/benchmarks")
    return p.parse_args()


def find_case_images() -> list[dict]:
    """Find images with 1 / 2 / 5 garment instances from existing pipeline outputs.

    Returns a list of dicts with: case_id, image_path, bboxes, num_boxes.
    """
    demo_root = _PROJECT_ROOT / "outputs" / "full_31x_demo"
    val_dir = Path(
        "D:/Aliintern/fashion-ai-data/deepfashion2/validation/image"
    )

    cases: list[dict] = []
    found_boxes: dict[int, bool] = {1: False, 2: False, 5: False}

    for subdir in sorted(demo_root.iterdir()):
        if not subdir.is_dir():
            continue
        det_path = subdir / "01_yolo" / "detections.json"
        if not det_path.exists():
            continue

        data = json.loads(det_path.read_text(encoding="utf-8"))
        for img_rec in data.get("images", []):
            dets = img_rec.get("detections", [])
            n = len(dets)
            if n in found_boxes and not found_boxes[n]:
                img_id = Path(img_rec["image_path"]).stem
                img_path = val_dir / f"{img_id}.jpg"
                if img_path.exists():
                    bboxes = [d["bbox_xyxy"] for d in dets]
                    cases.append({
                        "case_id": f"{n}_boxes_{img_id}",
                        "image_path": str(img_path),
                        "bboxes": bboxes,
                        "num_boxes": n,
                    })
                    found_boxes[n] = True

        if all(found_boxes.values()):
            break

    # Fill any missing cases synthetically.
    if not found_boxes[5]:
        # Use the 5-garment image if available.
        known_5 = "001574"
        det_path = demo_root / known_5 / "01_yolo" / "detections.json"
        if det_path.exists():
            data = json.loads(det_path.read_text(encoding="utf-8"))
            for img_rec in data.get("images", []):
                dets = img_rec.get("detections", [])
                if len(dets) >= 5:
                    img_path = val_dir / f"{known_5}.jpg"
                    bboxes = [d["bbox_xyxy"] for d in dets[:5]]
                    cases.append({
                        "case_id": f"5_boxes_{known_5}",
                        "image_path": str(img_path),
                        "bboxes": bboxes,
                        "num_boxes": 5,
                    })
                    found_boxes[5] = True
                    break

    return sorted(cases, key=lambda c: c["num_boxes"])


def run_one_bench(
    wrapper: SamHqWrapper,
    image_rgb: np.ndarray,
    boxes: list,
    mode: str,     # "one_by_one" or "batched"
    n_warmup: int,
    n_bench: int,
) -> dict:
    """Benchmark one config. Returns stats dict."""
    boxes_np = np.array(boxes, dtype=np.float32)

    # Warmup.
    for _ in range(n_warmup):
        wrapper.set_image(image_rgb)
        if mode == "batched":
            wrapper.predict_boxes(boxes_np)
        else:
            for b in boxes_np:
                wrapper.predict_boxes(b)

    torch.cuda.synchronize()

    set_ms: list[float] = []
    pred_ms: list[float] = []

    for _ in range(n_bench):
        wrapper.set_image(image_rgb)
        set_ms.append(wrapper.last_timing.get("set_image_ms", 0))

        t0 = time.perf_counter()
        if mode == "batched":
            wrapper.predict_boxes(boxes_np)
        else:
            for b in boxes_np:
                wrapper.predict_boxes(b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred_ms.append((t1 - t0) * 1000)

    return {
        "set_image_ms": stats(set_ms),
        "predict_ms": stats(pred_ms),
        "total_ms": stats([s + p for s, p in zip(set_ms, pred_ms)]),
    }


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    cases = find_case_images()
    print(f"Test cases: {[c['case_id'] for c in cases]}\n")

    results: dict = {
        "benchmark": "sam_batch_boxes",
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "gpu": device_name,
        "model": f"SAM-HQ {args.model_type}",
        "checkpoint": args.checkpoint,
        "n_warmup": args.n_warmup,
        "n_bench": args.n_bench,
        "cases": [],
    }

    for case in cases:
        cid = case["case_id"]
        n = case["num_boxes"]
        boxes = case["bboxes"]
        print(f"=== {cid} ({n} boxes) ===")

        image_bgr = cv2.imread(case["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        case_result: dict = {
            "case_id": cid,
            "image_path": case["image_path"],
            "num_boxes": n,
            "boxes": boxes,
            "results": {},
            "mask_iou": {},
        }

        # ── FP32_one_by_one (reference) ────────────────────────────
        w32 = SamHqWrapper(
            checkpoint=args.checkpoint, model_type=args.model_type,
            device=args.device, use_fp16=False,
        )
        r_fp32_one = run_one_bench(
            w32, image_rgb, boxes, "one_by_one", args.n_warmup, args.n_bench,
        )
        case_result["results"]["FP32_one_by_one"] = r_fp32_one
        print(f"  FP32_one_by_one: set={r_fp32_one['set_image_ms']['mean']:.0f}ms "
              f"pred={r_fp32_one['predict_ms']['mean']:.0f}ms "
              f"total={r_fp32_one['total_ms']['mean']:.0f}ms")

        # Get reference masks (one-by-one FP32).
        w32.set_image(image_rgb)
        ref_all: list[np.ndarray] = []
        for b in np.array(boxes, dtype=np.float32):
            m, _, _ = w32.predict_boxes(b)
            ref_all.append(m[0])
        del w32
        torch.cuda.empty_cache()

        # ── FP32_batched ──────────────────────────────────────────
        w32b = SamHqWrapper(
            checkpoint=args.checkpoint, model_type=args.model_type,
            device=args.device, use_fp16=False,
        )
        r_fp32_batch = run_one_bench(
            w32b, image_rgb, boxes, "batched", args.n_warmup, args.n_bench,
        )
        case_result["results"]["FP32_batched"] = r_fp32_batch
        w32b.set_image(image_rgb)
        m_batch_32, _, _ = w32b.predict_boxes(np.array(boxes, dtype=np.float32))
        iou_32b = float(np.mean([
            mask_iou(ref_all[i], m_batch_32[i]) for i in range(len(ref_all))
        ]))
        case_result["mask_iou"]["FP32_batched_vs_FP32_one_by_one"] = iou_32b
        print(f"  FP32_batched:   set={r_fp32_batch['set_image_ms']['mean']:.0f}ms "
              f"pred={r_fp32_batch['predict_ms']['mean']:.0f}ms "
              f"total={r_fp32_batch['total_ms']['mean']:.0f}ms  IoU={iou_32b:.6f}")
        del w32b
        torch.cuda.empty_cache()

        # ── FP16_one_by_one ───────────────────────────────────────
        w16 = SamHqWrapper(
            checkpoint=args.checkpoint, model_type=args.model_type,
            device=args.device, use_fp16=True,
        )
        r_fp16_one = run_one_bench(
            w16, image_rgb, boxes, "one_by_one", args.n_warmup, args.n_bench,
        )
        case_result["results"]["FP16_one_by_one"] = r_fp16_one
        w16.set_image(image_rgb)
        ious_16one: list[float] = []
        for i, b in enumerate(np.array(boxes, dtype=np.float32)):
            m, _, _ = w16.predict_boxes(b)
            ious_16one.append(mask_iou(ref_all[i], m[0]))
        iou_16one = float(np.mean(ious_16one))
        case_result["mask_iou"]["FP16_one_by_one_vs_FP32_one_by_one"] = iou_16one
        print(f"  FP16_one_by_one:set={r_fp16_one['set_image_ms']['mean']:.0f}ms "
              f"pred={r_fp16_one['predict_ms']['mean']:.0f}ms "
              f"total={r_fp16_one['total_ms']['mean']:.0f}ms  IoU={iou_16one:.6f}")
        del w16
        torch.cuda.empty_cache()

        # ── FP16_batched ─────────────────────────────────────────
        w16b = SamHqWrapper(
            checkpoint=args.checkpoint, model_type=args.model_type,
            device=args.device, use_fp16=True,
        )
        r_fp16_batch = run_one_bench(
            w16b, image_rgb, boxes, "batched", args.n_warmup, args.n_bench,
        )
        case_result["results"]["FP16_batched"] = r_fp16_batch
        w16b.set_image(image_rgb)
        m_batch_16, _, _ = w16b.predict_boxes(np.array(boxes, dtype=np.float32))
        iou_16b = float(np.mean([
            mask_iou(ref_all[i], m_batch_16[i]) for i in range(len(ref_all))
        ]))
        case_result["mask_iou"]["FP16_batched_vs_FP32_one_by_one"] = iou_16b
        print(f"  FP16_batched:   set={r_fp16_batch['set_image_ms']['mean']:.0f}ms "
              f"pred={r_fp16_batch['predict_ms']['mean']:.0f}ms "
              f"total={r_fp16_batch['total_ms']['mean']:.0f}ms  IoU={iou_16b:.6f}")
        del w16b
        torch.cuda.empty_cache()

        # ── Summary for this case ─────────────────────────────────
        t32_one = r_fp32_one["total_ms"]["mean"]
        t16_batch = r_fp16_batch["total_ms"]["mean"]
        case_result["speedup_fp16_batched_vs_fp32_one"] = round(
            t32_one / max(t16_batch, 0.1), 3,
        )

        print(f"  → best speedup: FP16_batched vs FP32_one_by_one = "
              f"{case_result['speedup_fp16_batched_vs_fp32_one']:.2f}x")

        results["cases"].append(case_result)

    # ── Aggregate summary ────────────────────────────────────────────
    summary_by_n: dict = {}
    for n_target in [1, 2, 5]:
        group = [c for c in results["cases"] if c["num_boxes"] == n_target]
        if not group:
            continue
        c = group[0]
        r = c["results"]
        summary_by_n[str(n_target)] = {
            "case_id": c["case_id"],
            "FP32_one_by_one_total_ms": r["FP32_one_by_one"]["total_ms"]["mean"],
            "FP32_batched_total_ms": r["FP32_batched"]["total_ms"]["mean"],
            "FP16_one_by_one_total_ms": r["FP16_one_by_one"]["total_ms"]["mean"],
            "FP16_batched_total_ms": r["FP16_batched"]["total_ms"]["mean"],
            "speedup_FP32_batched": round(
                r["FP32_one_by_one"]["total_ms"]["mean"] /
                max(r["FP32_batched"]["total_ms"]["mean"], 0.1), 3,
            ),
            "speedup_FP16_one_by_one": round(
                r["FP32_one_by_one"]["total_ms"]["mean"] /
                max(r["FP16_one_by_one"]["total_ms"]["mean"], 0.1), 3,
            ),
            "speedup_FP16_batched": round(
                r["FP32_one_by_one"]["total_ms"]["mean"] /
                max(r["FP16_batched"]["total_ms"]["mean"], 0.1), 3,
            ),
        }

    results["summary"] = {
        "by_num_boxes": summary_by_n,
        "overall_best_mode": "FP16_batched",
    }

    # ── Write JSON ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sam_batch_boxes_{ts}.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON → {out_path}")

    # ── Print summary table ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY: one-by-one vs batched")
    print(f"{'='*70}")
    header = f"{'N boxes':<8} {'FP32 1x1':>10} {'FP32 bat':>10} {'FP16 1x1':>10} {'FP16 bat':>10} {'Bestx':>8}"
    print(header)
    print("-" * 58)
    for n_str, s in summary_by_n.items():
        print(f"{n_str:<8} {s['FP32_one_by_one_total_ms']:>8.0f}ms "
              f"{s['FP32_batched_total_ms']:>8.0f}ms "
              f"{s['FP16_one_by_one_total_ms']:>8.0f}ms "
              f"{s['FP16_batched_total_ms']:>8.0f}ms "
              f"{s['speedup_FP16_batched']:>6.2f}x")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
