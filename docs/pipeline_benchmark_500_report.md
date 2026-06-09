# 500-Image End-to-End Garment Pipeline Benchmark

## 1. Experiment Setting

This benchmark evaluates the end-to-end fine-grained garment processing pipeline on 500 randomly sampled images from the DeepFashion2 training set.

The pipeline consists of the following stages:

1. YOLO garment detection
2. SAM-HQ garment segmentation
3. Garment landmark prediction
4. Semantic region crop generation
5. Mask-aware semantic region crop generation

## 2. Processing Scale and Success Rate

| Metric | Value |
|---|---:|
| Input images | 500 |
| Successfully processed images | 500 |
| Failed images | 0 |
| Processed garment instances | 913 |
| Failed garment instances in landmark stage | 0 |
| Average processed instances per image | 1.83 |
| Region crop records | 3220 |
| Successful region crops | 3220 |
| Failed region crops | 0 |
| Mask-aware crop records | 3220 |
| Successful mask-aware crops | 3212 |
| Failed mask-aware crops | 8 |

The image-level processing success rate is 100%. The landmark stage processed 913 garment instances with no failed instances. The region crop stage generated 3220 semantic local crops with a 100% success rate. The mask-aware crop stage successfully generated 3212 out of 3220 crops, achieving a success rate of 99.75%.

## 3. Region Crop Statistics

| Region | Region Crops | Mask-aware Successful Crops |
|---|---:|---:|
| waist | 913 | 908 |
| sleeve | 838 | 838 |
| hem | 649 | 647 |
| collar | 556 | 555 |
| pant_leg | 264 | 264 |
| **Total** | **3220** | **3212** |

The region crop stage generated 2917 crops from landmark-based regions and 303 crops using bounding-box fallback. The fallback ratio is approximately 9.41%.

## 4. Timing Results

| Stage | Total Time | Average Time per Image | Percentage |
|---|---:|---:|---:|
| YOLO detection | 12.14 s | 24.27 ms/image | 5.78% |
| SAM-HQ segmentation | 146.25 s | 292.51 ms/image | 69.62% |
| Landmark prediction | 27.20 s | 54.40 ms/image | 12.95% |
| Region crop generation | 3.40 s | 6.80 ms/image | 1.62% |
| Mask-aware crop generation | 21.07 s | 42.14 ms/image | 10.03% |
| **Total** | **210.06 s** | **420.13 ms/image** | **100%** |

Overall throughput:

| Metric | Value |
|---|---:|
| Total time | 210.06 s |
| Average time per image | 420.13 ms/image |
| Throughput | 2.38 images/s |
| Throughput per minute | 142.8 images/min |

## 5. Instance-Level Timing

The benchmark processed 913 garment instances. Based on this number:

| Stage | Average Time |
|---|---:|
| SAM-HQ segmentation | 160.2 ms/instance |
| Landmark prediction | 29.8 ms/instance |

On average, each garment instance generated approximately 3.53 semantic region crops.

## 6. Bottleneck Analysis

The main runtime bottleneck is the SAM-HQ segmentation stage. It takes 146.25 seconds out of the total 210.06 seconds, accounting for approximately 69.62% of the end-to-end runtime.

The YOLO detection stage is relatively efficient, requiring only 24.27 ms per image. The region crop generation stage is also lightweight, requiring only 6.80 ms per image.

The mask-aware crop stage accounts for about 10.03% of the total runtime. Since this stage mainly involves image reading, mask reading, local mask intersection, background filling, and image writing, future optimization can focus on reducing unnecessary disk I/O and making visualization or crop saving optional.

## 7. Conclusion

The benchmark shows that the current prototype pipeline is functionally stable and can process large image batches end to end. On 500 randomly sampled DeepFashion2 images, the pipeline successfully processed all images and all detected garment instances. It generated 3220 semantic region crops and 3212 valid mask-aware crops.

The current end-to-end latency is approximately 420.13 ms per image, with a throughput of 2.38 images/s. The main optimization target is the SAM-HQ segmentation stage, which contributes nearly 70% of the total runtime.

Future work should focus on:

1. Accelerating or replacing SAM-HQ with a lightweight segmentation model.
2. Exporting suitable modules to ONNXRuntime or TensorRT.
3. Reducing disk I/O from masks, overlays, and crop outputs.
4. Adding optional switches to disable heavy visualization outputs during benchmark or production inference.
5. Evaluating batched inference for detector and landmark prediction.
