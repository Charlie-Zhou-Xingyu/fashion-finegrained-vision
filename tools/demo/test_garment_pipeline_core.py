from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig


def main():
    config = GarmentPipelineConfig(
        yolo_weights="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
        sam_checkpoint="checkpoints/sam_hq/sam_hq_vit_b.pth",
        sam_model_type="vit_b",
        landmark_checkpoint="outputs/landmark_predictor_resnet18/best.pt",
        yolo_device="0",
        sam_device="cuda",
        landmark_device="cuda",
    )

    pipeline = GarmentPipeline(config)

    result = pipeline.run_image(
        image_path="assets/random_train500/images/000367.jpg",
        output_dir="outputs/test_garment_pipeline_core/000367",
    )

    print(result)


if __name__ == "__main__":
    main()
