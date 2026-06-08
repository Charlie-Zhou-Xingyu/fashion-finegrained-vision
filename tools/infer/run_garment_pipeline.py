from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-click garment fine-grained region crop pipeline: "
            "YOLO detection + SAM-HQ segmentation + landmark inference "
            "+ semantic region crops + SAM-HQ mask-aware crops."
        )
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input image path or image directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/infer/garment_region_crop_pipeline",
        help="Root output directory.",
    )

    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
        help="YOLO detector weights.",
    )

    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="checkpoints/sam_hq/sam_hq_vit_b.pth",
        help="SAM-HQ checkpoint.",
    )

    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM-HQ model type.",
    )

    parser.add_argument(
        "--landmark-checkpoint",
        type=str,
        default="outputs/landmark_predictor_resnet18/best.pt",
        help="Landmark predictor checkpoint.",
    )

    parser.add_argument(
        "--landmark-model",
        type=str,
        default="resnet18",
        help="Landmark predictor model name.",
    )

    parser.add_argument(
        "--landmark-image-size",
        type=int,
        default=256,
        help="Landmark predictor input image size.",
    )

    parser.add_argument(
        "--landmark-max-landmarks",
        type=int,
        default=39,
        help="Maximum number of garment landmarks.",
    )

    parser.add_argument(
        "--landmark-pad-ratio",
        type=float,
        default=0.05,
        help="Padding ratio for landmark crop bbox.",
    )

    parser.add_argument(
        "--landmark-device",
        type=str,
        default="cuda",
        help="Landmark inference device, e.g. cuda or cpu.",
    )

    parser.add_argument(
        "--save-landmark-visualizations",
        action="store_true",
        help="Save landmark visualization images.",
    )

    parser.add_argument(
        "--draw-landmark-index",
        action="store_true",
        help="Draw landmark index on landmark visualization images.",
    )

    parser.add_argument(
        "--draw-landmark-name",
        action="store_true",
        help="Draw landmark name on landmark visualization images if available.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO image size.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="YOLO NMS IoU threshold.",
    )

    parser.add_argument(
        "--yolo-device",
        type=str,
        default="0",
        help="YOLO device, e.g. 0 or cpu.",
    )

    parser.add_argument(
        "--sam-device",
        type=str,
        default="cuda",
        help="SAM-HQ device, e.g. cuda or cpu.",
    )

    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save YOLO crop images.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )

    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO detection and reuse existing 01_yolo/detections.json.",
    )

    parser.add_argument(
        "--skip-sam",
        action="store_true",
        help="Skip SAM-HQ segmentation and reuse existing 02_samhq/segmentation_results.json.",
    )

    parser.add_argument(
        "--skip-landmarks",
        action="store_true",
        help="Skip landmark inference and reuse existing 03_landmarks/landmarks_results.json.",
    )

    parser.add_argument(
        "--skip-region-crops",
        action="store_true",
        help="Skip region crop generation and reuse existing 04_region_crops/region_crops.json.",
    )

    parser.add_argument(
        "--skip-masked-crops",
        action="store_true",
        help="Skip SAM-HQ mask-aware region crop generation.",
    )

    parser.add_argument(
        "--region-crop-regions",
        type=str,
        nargs="+",
        default=["collar", "sleeve", "hem", "waist", "pant_leg"],
        help="Regions to crop from landmarks.",
    )

    parser.add_argument(
        "--region-max-outside-distance",
        type=float,
        default=5.0,
        help="Max outside-mask distance for reliable landmarks in region crop.",
    )

    parser.add_argument(
        "--region-min-points",
        type=int,
        default=2,
        help="Minimum reliable landmarks for multi-point region crop.",
    )

    parser.add_argument(
        "--region-pad-ratio",
        type=float,
        default=0.35,
        help="Padding ratio for region crop.",
    )

    parser.add_argument(
        "--region-single-point-box-ratio",
        type=float,
        default=0.18,
        help="Single-point box ratio for one-point region crop.",
    )

    parser.add_argument(
        "--no-region-fallback",
        action="store_true",
        help="Disable bbox fallback for region crop.",
    )

    parser.add_argument(
        "--no-category-regions",
        action="store_true",
        help="Disable category-aware region filtering.",
    )

    parser.add_argument(
        "--masked-crop-background",
        type=str,
        default="white",
        choices=["white", "black", "gray"],
        help="Background color for masked region crops.",
    )

    parser.add_argument(
        "--masked-crop-transparent",
        action="store_true",
        help="Save masked region crops with transparent background.",
    )

    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=0.005,
        help="Minimum SAM-HQ mask area ratio inside region crop.",
    )

    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n[COMMAND]")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def write_pipeline_summary(
    output_dir: Path,
    args: argparse.Namespace,
    yolo_output_dir: Path,
    sam_output_dir: Path,
    landmark_output_dir: Path,
    region_crop_output_dir: Path,
    masked_crop_output_dir: Path,
) -> None:
    summary = {
        "task": "run_garment_region_crop_pipeline",
        "source": args.source,
        "output_dir": str(output_dir),
        "steps": {
            "yolo_enabled": not args.skip_yolo,
            "sam_hq_enabled": not args.skip_sam,
            "landmarks_enabled": not args.skip_landmarks,
            "region_crops_enabled": not args.skip_region_crops,
            "masked_crops_enabled": not args.skip_masked_crops,
        },
        "yolo": {
            "weights": args.yolo_weights,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.yolo_device,
            "output_dir": str(yolo_output_dir),
            "detections_json": str(yolo_output_dir / "detections.json"),
            "visualizations_dir": str(yolo_output_dir / "visualizations"),
        },
        "sam_hq": {
            "checkpoint": args.sam_checkpoint,
            "model_type": args.sam_model_type,
            "device": args.sam_device,
            "output_dir": str(sam_output_dir),
            "segmentation_results_json": str(
                sam_output_dir / "segmentation_results.json"
            ),
            "masks_dir": str(sam_output_dir / "masks"),
            "overlays_dir": str(sam_output_dir / "overlays"),
        },
        "landmarks": {
            "checkpoint": args.landmark_checkpoint,
            "model": args.landmark_model,
            "image_size": args.landmark_image_size,
            "max_landmarks": args.landmark_max_landmarks,
            "pad_ratio": args.landmark_pad_ratio,
            "device": args.landmark_device,
            "add_landmark_schema": True,
            "use_mask_quality": True,
            "output_dir": str(landmark_output_dir),
            "landmarks_results_json": str(
                landmark_output_dir / "landmarks_results.json"
            ),
            "summary_json": str(landmark_output_dir / "summary.json"),
            "visualizations_dir": str(landmark_output_dir / "visualizations"),
        },
        "region_crops": {
            "output_dir": str(region_crop_output_dir),
            "region_crops_json": str(region_crop_output_dir / "region_crops.json"),
            "crops_dir": str(region_crop_output_dir / "crops"),
            "regions": args.region_crop_regions,
            "use_category_regions": not args.no_category_regions,
            "max_outside_distance": args.region_max_outside_distance,
            "min_points": args.region_min_points,
            "pad_ratio": args.region_pad_ratio,
            "single_point_box_ratio": args.region_single_point_box_ratio,
            "fallback": not args.no_region_fallback,
        },
        "masked_crops": {
            "output_dir": str(masked_crop_output_dir),
            "region_masked_crops_json": str(
                masked_crop_output_dir / "region_masked_crops.json"
            ),
            "image_crops_dir": str(masked_crop_output_dir / "image_crops"),
            "mask_crops_dir": str(masked_crop_output_dir / "mask_crops"),
            "masked_crops_dir": str(masked_crop_output_dir / "masked_crops"),
            "background": args.masked_crop_background,
            "transparent": bool(args.masked_crop_transparent),
            "min_mask_area_ratio": args.min_mask_area_ratio,
        },
    }

    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Pipeline summary saved to: {summary_path}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    yolo_output_dir = output_dir / "01_yolo"
    sam_output_dir = output_dir / "02_samhq"
    landmark_output_dir = output_dir / "03_landmarks"
    region_crop_output_dir = output_dir / "04_region_crops"
    masked_crop_output_dir = output_dir / "05_region_masked_crops"

    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_script = Path("tools/infer/predict_garments_yolo.py")
    sam_script = Path("tools/infer/segment_garments_samhq.py")
    landmark_script = Path("tools/infer/infer_landmarks_for_predictions.py")
    region_crop_script = Path("tools/crop/crop_garment_regions_from_landmarks.py")
    masked_crop_script = Path("tools/crop/apply_samhq_mask_to_region_crops.py")

    if not yolo_script.exists():
        raise FileNotFoundError(f"YOLO script not found: {yolo_script}")

    if not sam_script.exists():
        raise FileNotFoundError(f"SAM-HQ script not found: {sam_script}")

    if not landmark_script.exists():
        raise FileNotFoundError(f"Landmark script not found: {landmark_script}")

    if not region_crop_script.exists():
        raise FileNotFoundError(f"Region crop script not found: {region_crop_script}")

    if not masked_crop_script.exists():
        raise FileNotFoundError(f"Masked crop script not found: {masked_crop_script}")

    yolo_weights = Path(args.yolo_weights)
    sam_checkpoint = Path(args.sam_checkpoint)
    landmark_checkpoint = Path(args.landmark_checkpoint)

    if not yolo_weights.exists() and not args.skip_yolo:
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    if not sam_checkpoint.exists() and not args.skip_sam:
        raise FileNotFoundError(f"SAM-HQ checkpoint not found: {sam_checkpoint}")

    if not landmark_checkpoint.exists() and not args.skip_landmarks:
        raise FileNotFoundError(
            f"Landmark checkpoint not found: {landmark_checkpoint}"
        )

    detections_json = yolo_output_dir / "detections.json"
    segmentation_json = sam_output_dir / "segmentation_results.json"
    landmarks_json = landmark_output_dir / "landmarks_results.json"
    region_crops_json = region_crop_output_dir / "region_crops.json"

    # -------------------------------------------------------------------------
    # Step 1: YOLO detection
    # -------------------------------------------------------------------------
    if not args.skip_yolo:
        yolo_cmd = [
            sys.executable,
            str(yolo_script),
            "--weights",
            args.yolo_weights,
            "--source",
            args.source,
            "--output-dir",
            str(yolo_output_dir),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--device",
            args.yolo_device,
            "--save-vis",
        ]

        if args.save_crops:
            yolo_cmd.append("--save-crops")

        if args.max_images is not None:
            yolo_cmd.extend(["--max-images", str(args.max_images)])

        print("[INFO] Step 1/5: Running YOLO garment detection...")
        run_command(yolo_cmd)
    else:
        print("[INFO] Step 1/5: Skip YOLO detection.")
        if not detections_json.exists():
            raise FileNotFoundError(
                f"--skip-yolo was set, but detections.json not found: {detections_json}"
            )

    # -------------------------------------------------------------------------
    # Step 2: SAM-HQ segmentation
    # -------------------------------------------------------------------------
    if not args.skip_sam:
        if not detections_json.exists():
            raise FileNotFoundError(f"detections.json not found: {detections_json}")

        sam_cmd = [
            sys.executable,
            str(sam_script),
            "--detections-json",
            str(detections_json),
            "--sam-checkpoint",
            args.sam_checkpoint,
            "--model-type",
            args.sam_model_type,
            "--output-dir",
            str(sam_output_dir),
            "--device",
            args.sam_device,
        ]

        print("[INFO] Step 2/5: Running SAM-HQ garment segmentation...")
        run_command(sam_cmd)
    else:
        print("[INFO] Step 2/5: Skip SAM-HQ segmentation.")
        if not segmentation_json.exists():
            raise FileNotFoundError(
                f"--skip-sam was set, but segmentation_results.json not found: "
                f"{segmentation_json}"
            )

    # -------------------------------------------------------------------------
    # Step 3: Landmark inference
    # -------------------------------------------------------------------------
    if not args.skip_landmarks:
        if not segmentation_json.exists():
            raise FileNotFoundError(
                f"segmentation_results.json not found: {segmentation_json}"
            )

        landmark_cmd = [
            sys.executable,
            str(landmark_script),
            "--segmentation-json",
            str(segmentation_json),
            "--output-dir",
            str(landmark_output_dir),
            "--checkpoint",
            args.landmark_checkpoint,
            "--model",
            args.landmark_model,
            "--image-size",
            str(args.landmark_image_size),
            "--max-landmarks",
            str(args.landmark_max_landmarks),
            "--pad-ratio",
            str(args.landmark_pad_ratio),
            "--device",
            args.landmark_device,
            "--add-landmark-schema",
            "--use-mask-quality",
        ]

        if args.save_landmark_visualizations:
            landmark_cmd.append("--save-visualizations")

        if args.draw_landmark_index:
            landmark_cmd.append("--draw-index")

        if args.draw_landmark_name:
            landmark_cmd.append("--draw-name")

        print("[INFO] Step 3/5: Running garment landmark inference...")
        run_command(landmark_cmd)
    else:
        print("[INFO] Step 3/5: Skip landmark inference.")
        if not landmarks_json.exists():
            raise FileNotFoundError(
                f"--skip-landmarks was set, but landmarks_results.json not found: "
                f"{landmarks_json}"
            )

    # -------------------------------------------------------------------------
    # Step 4: Semantic region crops from landmarks
    # -------------------------------------------------------------------------
    if not args.skip_region_crops:
        if not landmarks_json.exists():
            raise FileNotFoundError(
                f"landmarks_results.json not found: {landmarks_json}"
            )

        region_crop_cmd = [
            sys.executable,
            str(region_crop_script),
            "--landmarks-json",
            str(landmarks_json),
            "--output-dir",
            str(region_crop_output_dir),
            "--regions",
            *args.region_crop_regions,
            "--max-outside-distance",
            str(args.region_max_outside_distance),
            "--min-points",
            str(args.region_min_points),
            "--pad-ratio",
            str(args.region_pad_ratio),
            "--single-point-box-ratio",
            str(args.region_single_point_box_ratio),
        ]

        if not args.no_category_regions:
            region_crop_cmd.append("--use-category-regions")

        if not args.no_region_fallback:
            region_crop_cmd.append("--fallback")

        print("[INFO] Step 4/5: Cropping garment semantic local regions...")
        run_command(region_crop_cmd)
    else:
        print("[INFO] Step 4/5: Skip region crop generation.")
        if not region_crops_json.exists():
            raise FileNotFoundError(
                f"--skip-region-crops was set, but region_crops.json not found: "
                f"{region_crops_json}"
            )

    # -------------------------------------------------------------------------
    # Step 5: SAM-HQ mask-aware region crops
    # -------------------------------------------------------------------------
    if not args.skip_masked_crops:
        if not region_crops_json.exists():
            raise FileNotFoundError(
                f"region_crops.json not found: {region_crops_json}"
            )

        if not segmentation_json.exists():
            raise FileNotFoundError(
                f"segmentation_results.json not found: {segmentation_json}"
            )

        masked_crop_cmd = [
            sys.executable,
            str(masked_crop_script),
            "--region-crops-json",
            str(region_crops_json),
            "--segmentation-json",
            str(segmentation_json),
            "--output-dir",
            str(masked_crop_output_dir),
            "--background",
            args.masked_crop_background,
            "--min-mask-area-ratio",
            str(args.min_mask_area_ratio),
        ]

        if args.masked_crop_transparent:
            masked_crop_cmd.append("--transparent")

        print("[INFO] Step 5/5: Applying SAM-HQ masks to region crops...")
        run_command(masked_crop_cmd)
    else:
        print("[INFO] Step 5/5: Skip masked crop generation.")

    write_pipeline_summary(
        output_dir=output_dir,
        args=args,
        yolo_output_dir=yolo_output_dir,
        sam_output_dir=sam_output_dir,
        landmark_output_dir=landmark_output_dir,
        region_crop_output_dir=region_crop_output_dir,
        masked_crop_output_dir=masked_crop_output_dir,
    )

    print("\n[INFO] Garment fine-grained region crop pipeline finished.")
    print(f"[INFO] Output dir: {output_dir}")

    print("\n[INFO] Step outputs:")
    print(f"[INFO] YOLO detections: {yolo_output_dir / 'detections.json'}")
    print(f"[INFO] YOLO visualizations: {yolo_output_dir / 'visualizations'}")

    print(f"[INFO] SAM-HQ results: {sam_output_dir / 'segmentation_results.json'}")
    print(f"[INFO] SAM-HQ overlays: {sam_output_dir / 'overlays'}")
    print(f"[INFO] SAM-HQ masks: {sam_output_dir / 'masks'}")

    print(f"[INFO] Landmark results: {landmark_output_dir / 'landmarks_results.json'}")
    print(f"[INFO] Landmark visualizations: {landmark_output_dir / 'visualizations'}")

    print(f"[INFO] Region crops JSON: {region_crop_output_dir / 'region_crops.json'}")
    print(f"[INFO] Region crop images: {region_crop_output_dir / 'crops'}")

    print(
        f"[INFO] Masked region crops JSON: "
        f"{masked_crop_output_dir / 'region_masked_crops.json'}"
    )
    print(f"[INFO] Masked crop images: {masked_crop_output_dir / 'masked_crops'}")

    print(f"\n[INFO] Pipeline summary: {output_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
