# DeepFashion2 GT Processing Benchmark

## 1. Benchmark Scope

This benchmark evaluates the lightweight ground-truth annotation processing pipeline for DeepFashion2.

The benchmark includes:

- loading annotation JSON files
- parsing garment instances
- reading category ids
- reading bounding boxes
- counting instances
- measuring per-file processing latency

The benchmark does **not** include:

- image decoding
- polygon-to-mask conversion
- crop generation
- model inference
- GPU acceleration

Therefore, this benchmark measures the data processing layer rather than the final model inference speed.

---

## 2. Dataset Scale

| Split | Annotation Files | Instances |
|---|---:|---:|
| Train | 103,822 | 169,045 |
| Validation | 32,153 | 52,490 |
| Total | 135,975 | 221,535 |

---

## 3. Benchmark Environment

Current environment:

```text
OS: Windows
Conda env: fashion-demo2
Dataset root: D:\Aliintern\fashion-ai-data\deepfashion2
Project root: D:\Aliintern\fashion-finegrained-vision
```

---

## 4. Full Benchmark Result

### 4.1 Train Split

| Metric | Value |
|---|---:|
| Annotation files | 103,822 |
| Instances | 169,045 |
| Total time | 22.556 s |
| Throughput | 4,602.93 files/s |
| Mean latency | 0.1250 ms/file |
| P95 latency | 0.1879 ms/file |

### 4.2 Validation Split

| Metric | Value |
|---|---:|
| Annotation files | 32,153 |
| Instances | 52,490 |
| Total time | 8.763 s |
| Throughput | 3,669.30 files/s |
| Mean latency | 0.1782 ms/file |
| P95 latency | 0.2782 ms/file |

### 4.3 Overall

| Metric | Value |
|---|---:|
| Total annotation files | 135,975 |
| Total instances | 221,535 |
| Total time | 31.319 s |
| Overall throughput | ~4,342 files/s |
| Overall instance throughput | ~7,073 instances/s |

---

## 5. Interpretation

The current data processing pipeline can parse the full DeepFashion2 annotation set in approximately **31.3 seconds**.

This indicates that the basic annotation processing layer is not a bottleneck for the current stage.

The next benchmark will include:

- polygon-to-mask conversion
- image loading
- region crop generation
- optional visualization output

Model inference speed will be evaluated separately after training the YOLO-seg baseline.

---

## 6. Next Steps

1. Build unified DeepFashion2 instance index.
2. Add mask and crop benchmark.
3. Convert DeepFashion2 annotations to YOLO-seg format.
4. Train a 5-class garment instance segmentation baseline.
5. Evaluate YOLO-seg inference speed and segmentation accuracy.

# End-to-End Garment Pipeline Benchmark

## 1. Benchmark Setting

Input:

```text
500 randomly sampled images from DeepFashion2 train/image
```

Pipeline:

1. YOLO garment detection
2. SAM-HQ garment segmentation
3. Garment landmark prediction
4. Semantic local region crop generation
5. SAM-HQ mask-aware region crop generation

Hardware:

```text
GPU device: cuda / device 0
```

---

## 2. End-to-End Timing Result

| Metric | Value |
|---|---:|
| Number of images | 500 |
| Total time | 210.06 s |
| Average time per image | 420.1 ms/image |
| Throughput | 2.38 images/s |
| Throughput per minute | 142.8 images/min |

---

## 3. Stage-wise Timing

| Stage | Total Time | Avg Time / Image | Percentage |
|---|---:|---:|---:|
| YOLO detection | 12.14 s | 24.3 ms | 5.78% |
| SAM-HQ segmentation | 146.25 s | 292.5 ms | 69.62% |
| Landmark prediction | 27.20 s | 54.4 ms | 12.95% |
| Region crop generation | 3.40 s | 6.8 ms | 1.62% |
| Masked crop generation | 21.07 s | 42.1 ms | 10.03% |
| **Total** | **210.06 s** | **420.1 ms** | **100%** |

---

## 4. Bottleneck Analysis

The main bottleneck is the SAM-HQ segmentation stage, which takes 146.25 seconds for 500 images and accounts for approximately 69.62% of the total end-to-end runtime.

The YOLO detection stage is relatively efficient, taking only 24.3 ms per image.

This indicates that the current detector is not the primary bottleneck. Future optimization should focus on:

1. SAM-HQ model acceleration
2. Replacing SAM-HQ with a lighter segmentation model for production
3. ONNXRuntime/TensorRT acceleration
4. Reducing unnecessary mask/overlay file IO
5. Batch processing where applicable

---

## 5. PRD Alignment

The current prototype validates the feasibility of the full visual processing pipeline.

Current status:

- YOLO detection: efficient, about 24.3 ms/image.
- SAM-HQ segmentation: functional but slow, about 292.5 ms/image.
- Landmark prediction: functional, about 54.4 ms/image.
- Region crop generation: lightweight, about 6.8 ms/image.
- Mask-aware crop generation: functional, about 42.1 ms/image.

The full pipeline currently reaches approximately 420.1 ms/image. Further inference optimization is required to meet strict real-time requirements.

The measured bottleneck provides evidence for the PRD optimization plan involving TensorRT, ONNXRuntime, and model-level acceleration.
