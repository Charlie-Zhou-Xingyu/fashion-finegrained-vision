"""Evaluate YOLO + SAM-HQ segmentation quality against DeepFashion2 GT masks.

Reads an existing ``segmentation_results.json`` (produced by the SAM-HQ stage of
``tools/infer/garment_pipeline.py``) and compares each predicted mask to the
corresponding DeepFashion2 ground-truth polygon annotation.

Matching strategy (per image)
------------------------------
1. Map every predicted segment and every GT instance to their **target category**
   (top / pants / skirt / outwear / dress) using
   ``src/fashion_vision/data/class_mapping.py``.
2. Within each category group, greedily match by descending bbox IoU.
3. A match is only accepted when ``bbox_iou >= --bbox-iou-threshold`` (default 0.5).
4. Compute mask IoU for each accepted pair using the existing
   ``fashion_vision.evaluation.iou.compute_mask_iou()``.

If the annotation directory does not exist the script exits immediately with an
actionable error message so results are never fabricated.

Usage
-----
::

    python tools/eval/eval_segmentation_iou.py \\
        --segmentation-json outputs/test_garment_pipeline_core/000367/02_samhq/segmentation_results.json \\
        --ann-dir D:/Aliintern/fashion-ai-data/deepfashion2/train/annos \\
        --output-dir outputs/benchmarks/segmentation_iou

Outputs
-------
``outputs/benchmarks/segmentation_iou/iou_report.json``
    Full per-instance results plus aggregate statistics.

``outputs/benchmarks/segmentation_iou/iou_report.md``
    Human-readable Markdown report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# sys.path — allow running as a plain script from the project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fashion_vision.data.class_mapping import DEEPFASHION2_TO_TARGET_CATEGORY  # noqa: E402
from fashion_vision.data.deepfashion2_parser import DeepFashion2Parser  # noqa: E402
from fashion_vision.evaluation.iou import compute_mask_iou  # noqa: E402

logger = logging.getLogger(__name__)

_PRD_IOU_TARGET: float = 0.85

# YOLO model uses 0-based class IDs; DeepFashion2 GT uses 1-based IDs.
# Shift by +1 to use the same mapping table.
_YOLO_CLASS_ID_TO_TARGET: dict[int, str] = {
    k - 1: v for k, v in DEEPFASHION2_TO_TARGET_CATEGORY.items()
}


# ---------------------------------------------------------------------------
# Pure geometry helpers (fully testable without model weights or GT files)
# ---------------------------------------------------------------------------


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute axis-aligned bounding-box IoU.

    Args:
        box_a: ``[x1, y1, x2, y2]``.
        box_b: ``[x1, y1, x2, y2]``.

    Returns:
        IoU in ``[0, 1]``.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_predictions_to_gt(
    preds: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
    bbox_iou_threshold: float = 0.5,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """Greedily match predicted segments to GT instances by category and bbox IoU.

    Args:
        preds: List of predicted segment dicts (must have ``target_category``
            and ``bbox_xyxy`` keys).
        gt_instances: List of GT instance dicts (must have ``target_category``
            and ``bbox`` keys).
        bbox_iou_threshold: Minimum bbox IoU to accept a match.

    Returns:
        List of ``(pred, gt, bbox_iou_score)`` tuples for each matched pair.
    """
    if not preds or not gt_instances:
        return []

    # Collect all candidate pairs: same category, bbox IoU >= threshold.
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gt_instances):
            if pred["target_category"] != gt["target_category"]:
                continue
            iou_score = bbox_iou(pred["bbox_xyxy"], gt["bbox"])
            if iou_score >= bbox_iou_threshold:
                candidates.append((iou_score, pi, gi))

    # Greedy assignment: highest IoU first; each pred and GT used at most once.
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches: list[tuple[dict, dict, float]] = []

    for iou_score, pi, gi in candidates:
        if pi in used_preds or gi in used_gts:
            continue
        matches.append((preds[pi], gt_instances[gi], iou_score))
        used_preds.add(pi)
        used_gts.add(gi)

    return matches


# ---------------------------------------------------------------------------
# Report formatting helpers (testable without real data)
# ---------------------------------------------------------------------------


def compute_aggregate_stats(iou_values: list[float]) -> dict[str, Any]:
    """Compute mean, median, and PRD-target pass rate from a list of IoU values.

    Args:
        iou_values: Per-instance mask IoU values.

    Returns:
        Dict with ``mean``, ``median``, ``min``, ``max``, ``count``,
        ``pct_above_prd_target``.  Returns zero-filled dict when empty.
    """
    if not iou_values:
        return {
            "count": 0, "mean": None, "median": None,
            "min": None, "max": None,
            "pct_above_prd_target": None,
            "prd_iou_target": _PRD_IOU_TARGET,
        }
    arr = np.array(iou_values, dtype=np.float64)
    return {
        "count": len(iou_values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "pct_above_prd_target": float(np.mean(arr >= _PRD_IOU_TARGET) * 100),
        "prd_iou_target": _PRD_IOU_TARGET,
    }


def format_iou_report_md(report: dict[str, Any]) -> str:
    """Render an IoU evaluation report as a Markdown string.

    Args:
        report: Dict produced by :func:`run_eval`.

    Returns:
        Markdown text (no trailing newline).
    """
    meta = report.get("meta", {})
    overall = report.get("overall", {})
    per_class = report.get("per_class", {})

    lines: list[str] = [
        "# Segmentation IoU Evaluation Report",
        "",
        f"> Generated: {meta.get('timestamp', 'unknown')}",
        f"> Segmentation JSON: `{meta.get('segmentation_json', 'unknown')}`",
        f"> Annotation dir: `{meta.get('ann_dir', 'unknown')}`",
        f"> Bbox IoU match threshold: {meta.get('bbox_iou_threshold', 0.5)}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        "|---|---:|",
        f"| Images evaluated | {meta.get('images_evaluated', 0)} |",
        f"| GT instances loaded | {meta.get('gt_instances_total', 0)} |",
        f"| Predicted segments | {meta.get('pred_segments_total', 0)} |",
        f"| Matched pairs (bbox IoU ≥ threshold) | {meta.get('matched_pairs', 0)} |",
        f"| Unmatched GT (missed) | {meta.get('unmatched_gt', 0)} |",
        f"| Unmatched preds (false positives) | {meta.get('unmatched_preds', 0)} |",
        "",
        "## Overall Mask IoU",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    if overall.get("count", 0) == 0:
        lines.append("| (no matched pairs) | — |")
    else:
        lines += [
            f"| Matched instances | {overall['count']} |",
            f"| Mean mask IoU | {overall['mean']:.4f} |",
            f"| Median mask IoU | {overall['median']:.4f} |",
            f"| Min mask IoU | {overall['min']:.4f} |",
            f"| Max mask IoU | {overall['max']:.4f} |",
            f"| PRD target (IoU ≥ {_PRD_IOU_TARGET}) | "
            f"**{overall['pct_above_prd_target']:.1f}%** |",
        ]

    prd_mean = overall.get("mean")
    if prd_mean is not None:
        status = "PASS" if prd_mean >= _PRD_IOU_TARGET else "FAIL"
        lines += [
            "",
            f"**PRD 3.1.1 target (mean IoU ≥ {_PRD_IOU_TARGET}): [{status}]**",
            f"Mean IoU = {prd_mean:.4f}",
        ]

    if per_class:
        lines += ["", "## Per-class Breakdown", "",
                  "| Category | Matched | Mean IoU | Median IoU | % ≥ 0.85 |",
                  "|---|---:|---:|---:|---:|"]
        for cat, stats in sorted(per_class.items()):
            if stats["count"] == 0:
                lines.append(f"| {cat} | 0 | — | — | — |")
            else:
                lines.append(
                    f"| {cat} | {stats['count']} | "
                    f"{stats['mean']:.4f} | "
                    f"{stats['median']:.4f} | "
                    f"{stats['pct_above_prd_target']:.1f}% |"
                )

    lines += ["", "## PRD 3.1.1 Target Summary", "",
              "| PRD Requirement | Status |",
              "|---|---|"]
    mean = overall.get("mean")
    if mean is None:
        lines.append("| Mean IoU ≥ 0.85 | UNKNOWN (no matched pairs) |")
    elif mean >= _PRD_IOU_TARGET:
        lines.append(f"| Mean IoU ≥ 0.85 | ✓ PASS ({mean:.4f}) |")
    else:
        lines.append(f"| Mean IoU ≥ 0.85 | ✗ FAIL ({mean:.4f}) |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GT annotation loading
# ---------------------------------------------------------------------------


def load_gt_annotation(
    image_stem: str,
    ann_dir: Path,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Load GT instances for one image from a DeepFashion2 annotation JSON.

    Args:
        image_stem: Image file stem (e.g. ``"000004"``).
        ann_dir: Directory containing annotation JSON files.
        image_width: Image width in pixels (from segmentation JSON).
        image_height: Image height in pixels (from segmentation JSON).

    Returns:
        List of GT instance dicts, each with keys:
        ``gt_category_id``, ``target_category``, ``bbox``, ``gt_mask``.
        Empty list if annotation file is missing (logged as warning).
    """
    ann_path = ann_dir / f"{image_stem}.json"
    if not ann_path.exists():
        logger.warning("Annotation not found: %s", ann_path)
        return []

    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse annotation %s: %s", ann_path, exc)
        return []

    instances: list[dict[str, Any]] = []
    for key, item in ann.items():
        if not key.startswith("item") or not isinstance(item, dict):
            continue
        try:
            cat_id = int(item["category_id"])
            bbox_raw = item.get("bounding_box", [])
            if len(bbox_raw) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox_raw]
            x1 = max(0.0, min(x1, image_width - 1))
            y1 = max(0.0, min(y1, image_height - 1))
            x2 = max(0.0, min(x2, image_width - 1))
            y2 = max(0.0, min(y2, image_height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            seg = item.get("segmentation", [])
            gt_mask = DeepFashion2Parser.polygon_to_mask(seg, image_height, image_width)

            if gt_mask.sum() == 0:
                continue

            target_cat = DEEPFASHION2_TO_TARGET_CATEGORY.get(cat_id)
            if target_cat is None:
                continue

            instances.append({
                "gt_category_id": cat_id,
                "target_category": target_cat,
                "bbox": [x1, y1, x2, y2],
                "gt_mask": gt_mask,
            })
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping GT item %s: %s", key, exc)
            continue

    return instances


# ---------------------------------------------------------------------------
# Predicted mask loading
# ---------------------------------------------------------------------------


def load_pred_mask(mask_path: str | Path) -> np.ndarray | None:
    """Load a predicted mask PNG produced by SAM-HQ.

    Args:
        mask_path: Path to the saved mask PNG.

    Returns:
        Boolean numpy array of shape ``(H, W)``, or ``None`` on failure.
    """
    p = Path(mask_path)
    if not p.exists():
        logger.warning("Predicted mask not found: %s", p)
        return None
    mask_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        logger.warning("Failed to read mask PNG: %s", p)
        return None
    return mask_gray > 0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def run_eval(
    segmentation_json: Path,
    ann_dir: Path,
    bbox_iou_threshold: float = 0.5,
    max_images: int = 0,
) -> dict[str, Any]:
    """Run IoU evaluation over all images in a segmentation results JSON.

    Args:
        segmentation_json: Path to ``segmentation_results.json``.
        ann_dir: DeepFashion2 annotation directory.
        bbox_iou_threshold: Minimum bbox IoU to accept a prediction-GT match.
        max_images: Process at most this many images (0 = all).

    Returns:
        Full evaluation report dict suitable for JSON serialisation.

    Raises:
        FileNotFoundError: If *segmentation_json* does not exist.
        FileNotFoundError: If *ann_dir* does not exist.
    """
    if not segmentation_json.exists():
        raise FileNotFoundError(
            f"Segmentation JSON not found: {segmentation_json}\n"
            "Run the YOLO + SAM-HQ pipeline first, e.g.:\n"
            "  python tools/infer/run_garment_pipeline.py ..."
        )

    if not ann_dir.exists():
        raise FileNotFoundError(
            f"DeepFashion2 annotation directory not found: {ann_dir}\n"
            "Expected path: <dataset_root>/train/annos  or  <dataset_root>/validation/annos\n"
            "Set --ann-dir to the directory containing the per-image annotation JSON files\n"
            "  (e.g. D:/Aliintern/fashion-ai-data/deepfashion2/train/annos)."
        )

    data = json.loads(segmentation_json.read_text(encoding="utf-8"))
    images: list[dict[str, Any]] = data.get("images", [])

    if not images:
        raise ValueError(f"No images found in segmentation JSON: {segmentation_json}")

    if max_images > 0:
        images = images[:max_images]

    all_iou: list[float] = []
    per_class_iou: dict[str, list[float]] = defaultdict(list)
    instance_records: list[dict[str, Any]] = []

    images_evaluated = 0
    gt_instances_total = 0
    pred_segments_total = 0
    matched_pairs = 0
    unmatched_gt = 0
    unmatched_preds = 0

    for img_rec in images:
        image_path = img_rec.get("image_path", "")
        image_stem = Path(image_path).stem
        width = int(img_rec.get("width", 0))
        height = int(img_rec.get("height", 0))
        segments = img_rec.get("segments", [])

        if not segments:
            logger.debug("No segments for image %s — skipping.", image_stem)
            continue

        # Attach target category to predictions.
        preds: list[dict[str, Any]] = []
        for seg in segments:
            class_id = int(seg.get("class_id", -1))
            target_cat = _YOLO_CLASS_ID_TO_TARGET.get(class_id)
            if target_cat is None:
                logger.warning(
                    "Unknown YOLO class_id %d in image %s — skipping segment.",
                    class_id, image_stem,
                )
                continue
            preds.append({**seg, "target_category": target_cat})

        gt_instances = load_gt_annotation(image_stem, ann_dir, width, height)
        if not gt_instances:
            logger.debug("No GT instances for %s — skipping.", image_stem)
            continue

        images_evaluated += 1
        gt_instances_total += len(gt_instances)
        pred_segments_total += len(preds)

        matches = match_predictions_to_gt(preds, gt_instances, bbox_iou_threshold)

        matched_gt_indices: set[int] = set()
        matched_pred_indices: set[int] = set()

        for pred_seg, gt_inst, b_iou in matches:
            pred_mask = load_pred_mask(pred_seg.get("mask_path", ""))
            if pred_mask is None:
                continue

            gt_mask = gt_inst["gt_mask"]

            # Resize prediction mask to GT dimensions if needed.
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            mask_iou = compute_mask_iou(
                pred_mask.astype(np.uint8),
                gt_mask.astype(np.uint8),
            )

            all_iou.append(mask_iou)
            per_class_iou[pred_seg["target_category"]].append(mask_iou)
            matched_pairs += 1

            matched_gt_indices.add(id(gt_inst))
            matched_pred_indices.add(id(pred_seg))

            instance_records.append({
                "image_stem": image_stem,
                "det_id": pred_seg.get("det_id"),
                "target_category": pred_seg["target_category"],
                "class_name": pred_seg.get("class_name"),
                "bbox_iou": round(b_iou, 4),
                "mask_iou": round(mask_iou, 4),
                "meets_prd_target": mask_iou >= _PRD_IOU_TARGET,
            })

        unmatched_gt += len(gt_instances) - len(matched_gt_indices)
        unmatched_preds += len(preds) - len(matched_pred_indices)

    import datetime
    overall = compute_aggregate_stats(all_iou)
    per_class_stats = {
        cat: compute_aggregate_stats(vals)
        for cat, vals in per_class_iou.items()
    }

    report = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "segmentation_json": str(segmentation_json),
            "ann_dir": str(ann_dir),
            "bbox_iou_threshold": bbox_iou_threshold,
            "images_evaluated": images_evaluated,
            "gt_instances_total": gt_instances_total,
            "pred_segments_total": pred_segments_total,
            "matched_pairs": matched_pairs,
            "unmatched_gt": unmatched_gt,
            "unmatched_preds": unmatched_preds,
        },
        "overall": overall,
        "per_class": per_class_stats,
        "instances": instance_records,
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YOLO+SAM-HQ segmentation IoU against DeepFashion2 GT annotations. "
            "Reads a pre-computed segmentation_results.json — does NOT run YOLO or SAM-HQ."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--segmentation-json", required=True, type=Path,
        help="Path to segmentation_results.json from the SAM-HQ pipeline stage.",
    )
    parser.add_argument(
        "--ann-dir", required=True, type=Path,
        help=(
            "DeepFashion2 annotation directory containing per-image JSON files. "
            "Typically <dataset_root>/train/annos or <dataset_root>/validation/annos."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/benchmarks/segmentation_iou"),
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--bbox-iou-threshold", type=float, default=0.5,
        help="Minimum bbox IoU to accept a prediction-GT match.",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Evaluate at most N images (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    logger.info("Segmentation JSON : %s", args.segmentation_json)
    logger.info("Annotation dir    : %s", args.ann_dir)
    logger.info("Output dir        : %s", args.output_dir)

    report = run_eval(
        segmentation_json=args.segmentation_json,
        ann_dir=args.ann_dir,
        bbox_iou_threshold=args.bbox_iou_threshold,
        max_images=args.max_images,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "iou_report.json"
    md_path = args.output_dir / "iou_report.md"

    # Strip large gt_mask arrays before serialising to JSON.
    report_json = {
        k: v for k, v in report.items() if k != "instances"
    }
    report_json["instances"] = [
        {ik: iv for ik, iv in inst.items()}
        for inst in report.get("instances", [])
    ]
    json_path.write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(
        format_iou_report_md(report) + "\n",
        encoding="utf-8",
    )

    logger.info("JSON report : %s", json_path)
    logger.info("MD report   : %s", md_path)

    meta = report["meta"]
    overall = report["overall"]
    print(f"\n{'='*60}")
    print(f"  Images evaluated : {meta['images_evaluated']}")
    print(f"  Matched pairs    : {meta['matched_pairs']}")
    print(f"  Unmatched GT     : {meta['unmatched_gt']}")
    print(f"  Unmatched preds  : {meta['unmatched_preds']}")
    print(f"{'='*60}")
    if overall.get("mean") is None:
        print("  No matched pairs — cannot compute IoU statistics.")
        print(f"  Check --ann-dir and that the images match the annotation split.")
    else:
        print(f"  Mean mask IoU    : {overall['mean']:.4f}")
        print(f"  Median mask IoU  : {overall['median']:.4f}")
        print(f"  % IoU ≥ 0.85     : {overall['pct_above_prd_target']:.1f}%")
        target_met = overall["mean"] >= _PRD_IOU_TARGET
        print(f"  PRD target (≥0.85): {'PASS' if target_met else 'FAIL'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
