"""
Batch-Backed Fast Path — zero code replication, wraps existing GarmentPipeline.

Internally creates a temporary directory of symlinks (or copies) and calls
``GarmentPipeline.run_source()`` once per batch.  This avoids the per-image
model reloading overhead that ``run_image()`` incurs because ``run_source()``
loads YOLO + SAM-HQ once and processes all images in a loop.

Does NOT modify any existing 3.1 code.  Uses only public APIs of
``GarmentPipeline``.

Usage::

    from inference.pipelines.fast_path_batch_backed import BatchBackedFastPath

    pipe = BatchBackedFastPath()
    results = pipe.run_images(["img1.jpg", "img2.jpg", "img3.jpg"])
    # results[i]["timing"] → per-image timing (from pipeline summary JSON)

Limitations:
    - Designed for batch/offline processing, not single-image interactive API.
    - All images in a batch share one output directory (run_source semantics).
    - Per-image timing is approximate (total / N) unless pipeline writes per-image
      breakdowns.

Status: New module.  Does NOT modify any existing 3.1 code.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BatchBackedFastPath:
    """Thin wrapper around GarmentPipeline.run_source() for batch processing.

    Zero line of inference logic is replicated.  All processing is delegated
    to the existing, tested pipeline.  The only value-add is creating a
    temporary batch directory so that multiple images are processed in one
    ``run_source()`` call, avoiding per-image YOLO + SAM-HQ reloading.
    """

    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        landmark_checkpoint: Optional[str] = None,
    ) -> None:
        self._yolo_weights = yolo_weights or (
            "models/detectors/yolov8n_deepfashion2_13cls_best.pt"
        )
        self._sam_checkpoint = sam_checkpoint or (
            "checkpoints/sam_hq/sam_hq_vit_b.pth"
        )
        self._landmark_checkpoint = landmark_checkpoint or (
            "outputs/landmark_predictor_resnet18/best.pt"
        )

    def run_images(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the Fast Path on a batch of images using one run_source() call.

        Args:
            image_paths: List of image paths to process together.
            output_dir: If provided, intermediate and final outputs are saved
                        here.  If None, a temporary directory is used.

        Returns:
            Dict with keys: status, timing, total_wallclock_s, num_images,
            per_image_ms_avg, pipeline_result (the raw GarmentPipeline output).
        """
        from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig

        images = [Path(p) for p in image_paths]

        # ── Create temp batch directory ─────────────────────────────────
        _temp_ctx: Any = None
        if output_dir is not None:
            batch_dir = Path(output_dir) / "batch_images"
            batch_dir.mkdir(parents=True, exist_ok=True)
            out_root = Path(output_dir)
        else:
            _temp_ctx = tempfile.TemporaryDirectory(prefix="bbfp_")
            batch_dir = Path(_temp_ctx.name) / "batch_images"
            batch_dir.mkdir(parents=True, exist_ok=True)
            out_root = Path(_temp_ctx.name)

        try:
            # Symlink or copy images into batch directory
            for img in images:
                dest = batch_dir / img.name
                if not dest.exists():
                    try:
                        dest.symlink_to(img.resolve())
                    except OSError:
                        shutil.copy2(str(img), str(dest))

            # ── Run existing pipeline (ZERO replicated logic) ───────────
            config = GarmentPipelineConfig(
                yolo_weights=self._yolo_weights,
                sam_checkpoint=self._sam_checkpoint,
                sam_model_type="vit_b",
                landmark_checkpoint=self._landmark_checkpoint,
                yolo_device="0",
                sam_device="cuda",
                landmark_device="cuda",
                run_landmark_and_crops=True,
                run_attribute_inference=False,
                save_yolo_vis=False,
                save_yolo_crops=False,
            )

            t0 = time.perf_counter()
            pipeline = GarmentPipeline(config)
            result = pipeline.run_source(
                source=str(batch_dir),
                output_dir=str(out_root / "pipeline_output"),
                max_images=len(images),
            )
            total_s = time.perf_counter() - t0

            timing = result.get("timing", {})
            total_ms = timing.get("total_seconds", 0) * 1000
            n = len(images)

            return {
                "status": "ok",
                "num_images": n,
                "total_wallclock_s": round(total_s, 2),
                "total_ms_all_images": round(total_ms, 2),
                "per_image_ms_avg": round(total_ms / max(1, n), 2),
                "throughput_qps": round(n / max(1e-9, total_s), 2),
                "timing_breakdown": {
                    "yolo_ms_total": round(timing.get("yolo_seconds", 0) * 1000, 2),
                    "sam_hq_ms_total": round(timing.get("sam_hq_seconds", 0) * 1000, 2),
                    "landmarks_ms_total": round(timing.get("landmarks_seconds", 0) * 1000, 2),
                    "region_crops_ms_total": round(timing.get("region_crops_seconds", 0) * 1000, 2),
                    "masked_crops_ms_total": round(timing.get("masked_crops_seconds", 0) * 1000, 2),
                },
                "pipeline_result": result,
            }

        finally:
            if _temp_ctx is not None:
                _temp_ctx.cleanup()


# ── Self-check ─────────────────────────────────────────────────────────────────

def _demo() -> None:
    pipe = BatchBackedFastPath()
    assert pipe._yolo_weights is not None
    print("[BatchBackedFastPath] Constructor OK (no models loaded, no old code touched).")


if __name__ == "__main__":
    _demo()
