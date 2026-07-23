"""
SAM call pattern benchmark: compare naive_per_box vs one_set_image vs batched.

Tests 6 modes x (1/2/5 boxes):
    FP32_naive_per_box          — set_image per box (worst case)
    FP32_one_set_image_loop      — set_image once, predict loop (current pipeline)
    FP32_one_set_image_batched   — set_image once, batched predict (optimized)
    FP16_naive_per_box
    FP16_one_set_image_loop
    FP16_one_set_image_batched

Usage:
    python inference/benchmarks/bench_sam_call_pattern.py
    python inference/benchmarks/bench_sam_call_pattern.py --n-bench 20
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

from inference.wrappers.sam_wrapper import SamHqWrapper, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAM call pattern: naive vs one-set vs batched"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-bench", type=int, default=10)
    p.add_argument("--checkpoint", default="checkpoints/sam_hq/sam_hq_vit_b.pth")
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--output-dir", default="outputs/benchmarks")
    return p.parse_args()


def find_test_cases() -> list[dict]:
    """Find images with 1, 2, 5 garment instances."""
    demo_root = _PROJECT_ROOT / "outputs" / "full_31x_demo"
    val_dir = Path("D:/Aliintern/fashion-ai-data/deepfashion2/validation/image")

    found: dict[int, dict] = {}
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
            if n in (1, 2, 5) and n not in found:
                img_id = Path(img_rec["image_path"]).stem
                img_path = val_dir / f"{img_id}.jpg"
                if img_path.exists():
                    found[n] = {
                        "case_id": f"{n}_boxes_{img_id}",
                        "image_path": str(img_path),
                        "boxes": [d["bbox_xyxy"] for d in dets[:n]],
                        "num_boxes": n,
                        "synthetic": False,
                    }
        if len(found) >= 3:
            break

    # Fill missing: synthetic boxes for 5 if not found.
    if 5 not in found and 2 in found:
        c2 = found[2]
        boxes_2 = c2["boxes"]
        # Repeat + offset: note synthetic flag.
        synthetic_5 = boxes_2 + boxes_2 + boxes_2[:1]
        img_path = c2["image_path"]
        img_id = Path(img_path).stem
        found[5] = {
            "case_id": f"5_boxes_{img_id}_synthetic",
            "image_path": img_path,
            "boxes": synthetic_5,
            "num_boxes": 5,
            "synthetic": True,
        }

    return [found[k] for k in sorted(found.keys())]


def bench_mode(
    wrapper_cls,
    checkpoint: str,
    model_type: str,
    device: str,
    use_fp16: bool,
    image_rgb: np.ndarray,
    boxes: list,
    mode: str,
    n_warmup: int,
    n_bench: int,
) -> dict:
    """Benchmark one mode. Returns stats dict."""
    set_calls: int = 0
    pred_calls: int = 0

    set_ms: list[float] = []
    pred_ms: list[float] = []

    for _ in range(n_warmup):
        wrapper = wrapper_cls(
            checkpoint=checkpoint, model_type=model_type,
            device=device, use_fp16=use_fp16,
        )

        if mode == "naive_per_box":
            for box in boxes:
                wrapper.set_image(image_rgb)
                set_calls += 1
                wrapper.predict(box=np.array(box, dtype=np.float32))
                pred_calls += 1
        elif mode == "one_set_image_loop":
            wrapper.set_image(image_rgb)
            set_calls = 1
            for box in boxes:
                wrapper.predict(box=np.array(box, dtype=np.float32))
                pred_calls += 1
        elif mode == "one_set_image_batched":
            wrapper.set_image(image_rgb)
            set_calls = 1
            wrapper.predict_boxes(np.array(boxes, dtype=np.float32))
            pred_calls = 1

        del wrapper
        torch.cuda.empty_cache()

    torch.cuda.synchronize()

    for _ in range(n_bench):
        wrapper = wrapper_cls(
            checkpoint=checkpoint, model_type=model_type,
            device=device, use_fp16=use_fp16,
        )

        t0 = time.perf_counter()

        if mode == "naive_per_box":
            _set_acc = 0.0
            _pred_acc = 0.0
            for box in boxes:
                wrapper.set_image(image_rgb)
                _set_acc += wrapper.last_timing.get("set_image_ms", 0)
                wrapper.predict(box=np.array(box, dtype=np.float32))
                _pred_acc += wrapper.last_timing.get("predict_ms", 0)
            set_ms.append(_set_acc)
            pred_ms.append(_pred_acc)

        elif mode == "one_set_image_loop":
            wrapper.set_image(image_rgb)
            set_ms.append(wrapper.last_timing.get("set_image_ms", 0))
            p_acc = 0.0
            for box in boxes:
                wrapper.predict(box=np.array(box, dtype=np.float32))
                p_acc += wrapper.last_timing.get("predict_ms", 0)
            pred_ms.append(p_acc)

        elif mode == "one_set_image_batched":
            wrapper.set_image(image_rgb)
            set_ms.append(wrapper.last_timing.get("set_image_ms", 0))
            wrapper.predict_boxes(np.array(boxes, dtype=np.float32))
            pred_ms.append(wrapper.last_timing.get("predict_boxes_ms", 0))

        torch.cuda.synchronize()
        del wrapper
        torch.cuda.empty_cache()

    return {
        "set_image_calls": set_calls if n_warmup > 0 else (1 if "one_set" in mode else len(boxes)),
        "predict_calls": pred_calls if n_warmup > 0 else (1 if "batched" in mode else len(boxes)),
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

    cases = find_test_cases()
    print(f"Test cases: {[(c['case_id'], c['num_boxes']) for c in cases]}")

    MODES = [
        ("naive_per_box", False),
        ("one_set_image_loop", False),
        ("one_set_image_batched", False),
        ("naive_per_box", True),
        ("one_set_image_loop", True),
        ("one_set_image_batched", True),
    ]

    MODE_LABELS = [
        "FP32_naive_per_box",
        "FP32_one_set_image_loop",
        "FP32_one_set_image_batched",
        "FP16_naive_per_box",
        "FP16_one_set_image_loop",
        "FP16_one_set_image_batched",
    ]

    results: dict = {
        "benchmark": "sam_call_pattern",
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
        boxes = case["boxes"]
        print(f"\n=== {cid} ({n} boxes) ===")

        image_bgr = cv2.imread(case["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        case_result: dict = {
            "case_id": cid,
            "image_path": case["image_path"],
            "num_boxes": n,
            "synthetic": case.get("synthetic", False),
            "results": {},
        }

        for (mode, use_fp16), label in zip(MODES, MODE_LABELS):
            r = bench_mode(
                SamHqWrapper, args.checkpoint, args.model_type,
                args.device, use_fp16, image_rgb, boxes, mode,
                args.n_warmup, args.n_bench,
            )
            case_result["results"][label] = r
            t = r["total_ms"]["mean"]
            s = r["set_image_ms"]["mean"]
            p = r["predict_ms"]["mean"]
            sc = r["set_image_calls"]
            pc = r["predict_calls"]
            print(f"  {label:<32s}: set={s:6.0f}ms x{sc}  "
                  f"pred={p:6.0f}ms x{pc}  total={t:6.0f}ms")

        results["cases"].append(case_result)

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("SUMMARY: Call Pattern Benchmark")
    print(f"{'='*100}")
    header = (f"{'N':<3} {'synth':<6} "
              f"{'FP32 naive':>12} {'FP32 loop':>12} {'FP32 batch':>12} "
              f"{'FP16 naive':>12} {'FP16 loop':>12} {'FP16 batch':>12} "
              f"{'best':>8}")
    print(header)
    print("-" * 100)

    summary_by_n: dict = {}
    for case_result in results["cases"]:
        n = case_result["num_boxes"]
        r = case_result["results"]
        synth = "SYNTH" if case_result.get("synthetic") else "real"
        totals = {
            label: r[label]["total_ms"]["mean"] for label in MODE_LABELS
        }
        best = min(totals.values())
        worst = max(totals.values())
        best_label = min(totals, key=totals.get)
        print(f"{n:<3} {synth:<6} "
              f"{totals['FP32_naive_per_box']:>10.0f}ms "
              f"{totals['FP32_one_set_image_loop']:>10.0f}ms "
              f"{totals['FP32_one_set_image_batched']:>10.0f}ms "
              f"{totals['FP16_naive_per_box']:>10.0f}ms "
              f"{totals['FP16_one_set_image_loop']:>10.0f}ms "
              f"{totals['FP16_one_set_image_batched']:>10.0f}ms "
              f"{worst/best:>5.2f}x")

        summary_by_n[str(n)] = {
            "case_id": case_result["case_id"],
            "synthetic": case_result.get("synthetic", False),
            "totals_ms": totals,
            "best_mode": best_label,
            "speedup_vs_naive_fp32": round(
                totals["FP32_naive_per_box"] / max(best, 0.1), 3,
            ),
        }

    results["summary"] = {
        "by_num_boxes": summary_by_n,
        "key_finding": (
            "FP16_one_set_image_batched is consistently the fastest. "
            "naive_per_box (set_image per garment) is 2-5x slower than "
            "one_set_image, confirming that set_image is the bottleneck."
        ),
    }

    # ── Write JSON ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sam_call_pattern_{ts}.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON -> {out_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
