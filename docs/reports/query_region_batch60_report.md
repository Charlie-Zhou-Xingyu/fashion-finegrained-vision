# Query Region Batch60 Report

## 1. Objective

This experiment evaluates the current rule-based Chinese query-to-region demo on randomly sampled DeepFashion2 images.

The goal is to verify:

1. Whether the full garment pipeline can run stably on random images.
2. Whether common Chinese garment part queries can return valid local region results.
3. Whether pipeline reuse works correctly for multiple queries on the same image.
4. Whether the current demo is ready to be used as the first prototype of PRD 3.1.2.

---

## 2. Experiment Setting

### 2.1 Image Source

```text
D:\Aliintern\fashion-ai-data\deepfashion2\train\image
```

### 2.2 Output Directory

```text
outputs\query_region_online_demo_batch60
```

### 2.3 Sampling

| Item | Value |
|---|---:|
| Number of sampled images | 60 |
| Random seed | 20260609 |
| Image format | JPG |

### 2.4 Queries

Each sampled image was queried with the following five queries:

```text
领口
左袖子
右袖子
下摆
腰部
```

Total query runs:

```text
60 images × 5 queries = 300 query runs
```

---

## 3. Execution Strategy

For each image:

1. Run the full pipeline once using query `腰部`.
2. Reuse the generated pipeline directory for all five queries.
3. Record success/failure for each query command.
4. Save summary statistics to `batch_summary.json`.

This avoids repeatedly running YOLO, SAM-HQ, landmark prediction, and crop generation for each query.

---

## 4. Batch Summary

The raw batch directory contained 398 `result.json` files because some queries were rerun during debugging.

To obtain a clean controlled evaluation, results were deduplicated by:

```text
image + query
```

For duplicate records, the latest result was kept.

After deduplication, the final number of rows is:

```text
60 images × 5 queries = 300 rows
```

Raw and deduplicated summary:

| Metric | Value |
|---|---:|
| Raw result.json files | 398 |
| Rows before deduplication | 398 |
| Rows after deduplication | 300 |
| Deduplication key | image + query |
| Unique images | 60 |
| Queries | 5 |
| Final query runs | 300 |

---

## 5. Final Result

| Metric | Value |
|---|---:|
| Query runs | 300 |
| Success | 276 |
| Failed | 24 |
| Parse failed | 0 |
| Unknown | 0 |
| Valid response rate | 92.0% |

The valid response rate is calculated as:

```text
276 / 300 = 92.0%
```

This number is a valid response rate, not strict localization accuracy, because no manual correctness annotation was used in this batch.

---

## 6. Per-query Result

| Query | Total | Success | Failed | Success Rate |
|---|---:|---:|---:|---:|
| 下摆 | 60 | 57 | 3 | 95.0% |
| 右袖子 | 60 | 51 | 9 | 85.0% |
| 左袖子 | 60 | 51 | 9 | 85.0% |
| 腰部 | 60 | 60 | 0 | 100.0% |
| 领口 | 60 | 57 | 3 | 95.0% |
| **Total** | **300** | **276** | **24** | **92.0%** |

---

## 7. Per-region Result

| Target Region | Total | Success | Failed | Success Rate |
|---|---:|---:|---:|---:|
| hem | 60 | 57 | 3 | 95.0% |
| sleeve | 120 | 102 | 18 | 85.0% |
| waist | 60 | 60 | 0 | 100.0% |
| collar | 60 | 57 | 3 | 95.0% |

The sleeve region has the lowest valid response rate in this batch, mainly because left/right sleeve queries require component-level matching.

---

## 8. Selected Class Distribution

| Selected Class | Count |
|---|---:|
| short sleeve top | 146 |
| long sleeve top | 58 |
| skirt | 27 |
| long sleeve outwear | 20 |
| vest | 13 |
| long sleeve dress | 5 |
| short sleeve dress | 4 |
| trousers | 3 |

The majority of selected regions come from upper-body garments, especially `short sleeve top` and `long sleeve top`.

---

## 9. Failure Analysis

All failed cases share the same error type:

| Error Type | Count |
|---|---:|
| no_matching_region_crop | 24 |

This means the query parser worked, but no valid matching region crop was found in the generated `region_masked_crops.json`.

Possible reasons include:

1. The queried part does not exist in the image.
2. The detected garment category does not support the queried region.
3. The landmark stage did not generate the required semantic region.
4. Left/right sleeve component matching failed because only one sleeve component was generated.
5. The mask-aware crop stage did not produce a valid crop for the corresponding region.

These are candidate-missing failures rather than runtime crashes.

---

## 10. Conclusion

The rule-based Chinese query-to-region demo is stable on the 60-image DeepFashion2 batch.

Final controlled result:

```text
300 query runs
276 successful results
24 failed results
92.0% valid response rate
```

The current module can be considered a completed P1 prototype for PRD 3.1.2 language-guided local garment region localization.

The next recommended step is to connect localized regions with FashionAI attribute classification, starting from the `sleeve_length_labels` task.


## 10. Supported PRD Functionality

This experiment supports the prototype validation of PRD 3.1.2:

> Language-guided local garment region localization.

Current supported local regions:

| Region | Status |
|---|---|
| collar | Supported |
| sleeve | Supported |
| left_sleeve | Supported through component filtering |
| right_sleeve | Supported through component filtering |
| hem | Supported |
| waist | Supported |
| pant_leg | Supported if generated by upstream pipeline |
| skirt hem | Supported through query-specific class constraint |

---

## 11. Known Limitations

1. The query parser is rule-based and not a learned language model.
2. The system only selects from pre-generated region crops.
3. If no candidate exists in `region_masked_crops.json`, the query fails.
4. The current batch did not include manual correctness labeling.
5. The query set does not yet cover pockets, shoulders, patterns, decorations, fabric texture, or craftsmanship.
6. Shoes, bags, and accessories are not covered by the current pipeline.

---

## 12. Deliverables Generated

The experiment generated:

```text
outputs\query_region_online_demo_batch60\batch_summary.json
outputs\query_region_online_demo_batch60\sampled_60_images.txt
```

For each query result, the demo generated a separate output directory containing:

```text
result.json
region_overlay.jpg
region_mask_full.png
selected_image_crop.png
selected_mask_crop.png
selected_masked_crop.png
```

---

## 13. Conclusion

The rule-based Chinese query-to-region demo is functional and stable enough to be treated as the first deliverable prototype for text-guided local garment region localization.

The current module can be considered a completed P1 prototype with the following status:

| Item | Status |
|---|---|
| Single-image query demo | Completed |
| Pipeline reuse | Completed |
| Batch validation | Completed |
| Waist priority rule | Completed |
| Skirt hem vs generic hem separation | Completed |
| JSON output | Completed |
| Overlay visualization | Completed |
| Manual inspection | Partially completed |
| Strict accuracy evaluation | Not completed |

Next recommended step:

1. Summarize all `result.json` files into a CSV table.
2. Start the FashionAI attribute classification baseline.
3. Connect localized regions with attribute classifiers to build the first region-to-attribute demo.
