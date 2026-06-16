# Data Scan Report

## 1. Dataset Overview

This project currently uses two main datasets:

1. **DeepFashion2** for garment instance segmentation.
2. **FashionAI Attributes** for fine-grained fashion attribute classification.

Raw datasets are stored under:

```text
D:\Aliintern\fashion-ai-data
```

---

## 2. DeepFashion2 Scan Result

### 2.1 File-level Statistics

| Split | Image Files | Annotation Files |
|---|---:|---:|
| Train | 191,961 | 103,822 |
| Validation | 32,153 | 32,153 |

Note: In the training split, the number of image files is larger than the number of annotation files. Therefore, the supervised training subset is based on images with corresponding annotations.

### 2.2 Annotation-level Statistics

| Split | Valid Instances | Failed Annotations | Missing Images |
|---|---:|---:|---:|
| Train | 169,045 | 0 | 0 |
| Validation | 52,490 | 0 | 0 |

### 2.3 Raw Category Distribution

#### Train

| Raw Category | Count |
|---|---:|
| short sleeve top | 39,021 |
| long sleeve top | 19,467 |
| short sleeve outwear | 344 |
| long sleeve outwear | 7,077 |
| vest | 8,784 |
| sling | 1,105 |
| shorts | 19,558 |
| trousers | 30,020 |
| skirt | 16,990 |
| short sleeve dress | 9,799 |
| long sleeve dress | 4,149 |
| vest dress | 9,216 |
| sling dress | 3,515 |

#### Validation

| Raw Category | Count |
|---|---:|
| short sleeve top | 12,556 |
| long sleeve top | 5,966 |
| short sleeve outwear | 142 |
| long sleeve outwear | 2,011 |
| vest | 2,113 |
| sling | 322 |
| shorts | 4,167 |
| trousers | 9,586 |
| skirt | 6,522 |
| short sleeve dress | 3,127 |
| long sleeve dress | 1,477 |
| vest dress | 3,352 |
| sling dress | 1,149 |

### 2.4 PRD Category Mapping

DeepFashion2 categories are mapped to the PRD garment categories as follows:

| PRD Category | Train Instances | Validation Instances |
|---|---:|---:|
| top | 68,377 | 20,957 |
| pants | 49,578 | 13,753 |
| skirt | 16,990 | 6,522 |
| outerwear | 7,421 | 2,153 |
| dress | 26,679 | 9,105 |
| shoes | 0 | 0 |
| bag | 0 | 0 |
| accessory | 0 | 0 |

### 2.5 DeepFashion2 Conclusion

DeepFashion2 can support 5 out of 8 PRD garment categories:

- top
- pants
- skirt
- outerwear
- dress

The following PRD categories are not covered by current DeepFashion2 annotations:

- shoes
- bag
- accessory

These missing categories need to be supplemented using external datasets, pseudo-labeling, or manual annotation.

---

## 3. FashionAI Attributes Scan Result

### 3.1 Overall Statistics

| Metric | Value |
|---|---:|
| Total records | 10,080 |
| Missing images | 0 |
| Invalid labels | 0 |

### 3.2 Attribute Task Distribution

| Attribute Key | Count |
|---|---:|
| neckline_design_labels | 2,095 |
| sleeve_length_labels | 1,740 |
| coat_length_labels | 1,453 |
| skirt_length_labels | 1,153 |
| collar_design_labels | 1,082 |
| pant_length_labels | 949 |
| lapel_design_labels | 900 |
| neck_design_labels | 708 |

### 3.3 Supported Attribute Tasks

The FashionAI dataset supports the following fine-grained fashion attribute tasks:

- coat length
- collar design
- lapel design
- neck design
- neckline design
- pant length
- skirt length
- sleeve length

### 3.4 FashionAI Conclusion

FashionAI Attributes can support the first-stage baseline of PRD 3.1.3 fine-grained attribute extraction.

Currently covered attributes include:

- coat length
- skirt length
- pant length
- sleeve length
- collar design
- lapel design
- neck design
- neckline design

However, it does not cover all PRD attribute categories, especially:

- fabric/material attributes
- craftsmanship attributes
- pattern attributes
- style attributes

Additional datasets or annotations are required for full PRD coverage.

---

## 4. Current PRD Coverage

| PRD Module | Dataset / Method | Current Coverage |
|---|---|---|
| 3.1.1 Garment Instance Segmentation | DeepFashion2 + YOLO + SAM-HQ | 5/8 PRD garment categories covered |
| 3.1.2 Language-guided Local Region Grounding | DeepFashion2 landmarks + rule-based Chinese query parser | Prototype ready for collar, sleeve, hem, waist, pant_leg |
| 3.1.3 Fine-grained Attribute Extraction | FashionAI Attributes | 8 attribute tasks covered, training baseline pending |

Notes:

- The current 3.1.2 implementation is a rule-based prototype rather than open-vocabulary visual grounding.
- It supports Chinese query parsing and local region selection from pre-generated semantic region crops.
- It has been validated on 60 randomly sampled DeepFashion2 images with a 92.0% valid response rate.

---

## 5. Updated Next Steps

1. Finalize existing reports:
   - DeepFashion2 GT benchmark
   - 500-image end-to-end pipeline benchmark
   - Query-region batch60 validation report

2. Build the FashionAI attribute classification baseline:
   - start with `sleeve_length_labels`
   - train a ResNet18 single-task classifier
   - output accuracy, per-class metrics, and confusion matrix

3. Connect the query-region demo with the attribute classifier:
   - localize the queried region
   - crop the selected region
   - run attribute classification
   - output structured region-level semantic attributes

4. Convert DeepFashion2 annotations to YOLO-seg format:
   - start with 5 PRD garment classes: top, pants, skirt, outerwear, dress
   - train YOLO-seg as a potential replacement for YOLO + SAM-HQ

5. Plan missing PRD categories:
   - shoes
   - bags
   - accessories

The missing categories should not block the current clothing-focused prototype. They will be handled later using external datasets, pseudo-labeling, or manual annotation.


In the current query-region prototype, `vest` and `sling` are treated as upper-body garments for local region selection. They are not forced into the outerwear category.

For future 5-class YOLO-seg training, category mapping should be explicitly defined in a separate configuration file to avoid ambiguity between top and outerwear.
