# Current Progress Summary

## 1. Completed Work

### 1.1 DeepFashion2 Ground-truth Processing Benchmark

The DeepFashion2 ground-truth annotation parsing benchmark has been completed.

Results:

- Parsed 135,975 annotation files.
- Processed 221,535 garment instances.
- Full parsing time: 31.319 seconds.
- Overall throughput: approximately 4,342 files/s.
- Overall instance throughput: approximately 7,073 instances/s.

Conclusion:

- DeepFashion2 annotation parsing is lightweight.
- Basic JSON annotation processing is not the bottleneck of the current system.

---

### 1.2 End-to-End Garment Pipeline Benchmark

A 500-image end-to-end benchmark has been completed on randomly sampled DeepFashion2 training images.

Pipeline:

1. YOLO garment detection
2. SAM-HQ garment segmentation
3. Garment landmark prediction
4. Semantic local region crop generation
5. Mask-aware semantic region crop generation

Results:

| Metric | Value |
|---|---:|
| Images | 500 |
| Processed garment instances | 913 |
| Region crops | 3220 |
| Successful mask-aware crops | 3212 |
| Total time | 210.06 s |
| Average latency | 420.13 ms/image |
| Throughput | 2.38 images/s |

Stage-wise timing:

| Stage | Time | Percentage |
|---|---:|---:|
| YOLO detection | 12.14 s | 5.78% |
| SAM-HQ segmentation | 146.25 s | 69.62% |
| Landmark prediction | 27.20 s | 12.95% |
| Region crop generation | 3.40 s | 1.62% |
| Mask-aware crop generation | 21.07 s | 10.03% |

Conclusion:

- The full visual pipeline is functional and stable.
- SAM-HQ is the main runtime bottleneck.
- Region crop generation is lightweight.
- The next optimization direction is replacing or accelerating SAM-HQ, e.g. YOLO-seg, ONNXRuntime, or TensorRT.

---

### 1.3 Text-guided Local Region Demo

A rule-based Chinese query-to-region demo has been implemented and validated.

Supported queries:

- 领口
- 左袖子
- 右袖子
- 下摆
- 腰部
- 裙摆
- 裤腿

Batch60 validation:

| Metric | Value |
|---|---:|
| Sampled images | 60 |
| Queries per image | 5 |
| Query runs | 300 |
| Success | 276 |
| Failed | 24 |
| Valid response rate | 92.0% |

Per-query result:

| Query | Success Rate |
|---|---:|
| 腰部 | 100.0% |
| 领口 | 95.0% |
| 下摆 | 95.0% |
| 左袖子 | 85.0% |
| 右袖子 | 85.0% |

Conclusion:

- The rule-based Chinese text-to-region prototype is functional and stable.
- Failure cases are all `no_matching_region_crop`.
- The module is ready to be treated as the P1 deliverable prototype.

---

## 2. Current Dataset Coverage

| Dataset | Supported Module | Status |
|---|---|---|
| DeepFashion2 | Garment detection, segmentation, landmarks, local region crops | Integrated |
| FashionAI Attributes | Fine-grained attribute classification | Dataset scanned, baseline pending |
| Shoes/Bags/Accessories | PRD extended classes | Not covered by current main datasets |

DeepFashion2 currently supports five PRD garment categories:

- top
- pants
- skirt
- outerwear
- dress

The following PRD categories are not covered by DeepFashion2:

- shoes
- bag
- accessory

FashionAI Attributes currently supports several first-stage attribute tasks:

- sleeve length
- skirt length
- pant length
- coat length
- collar design
- lapel design
- neck design
- neckline design

---

## 3. Current PRD Alignment

| PRD Module | Requirement | Current Status |
|---|---|---|
| 3.1.1 Garment instance segmentation | Detect and segment garment instances | Prototype ready using YOLO + SAM-HQ |
| 3.1.2 Language-guided local region localization | Locate garment parts from natural language query | Rule-based Chinese query prototype ready |
| 3.1.3 Fine-grained attribute extraction | Extract local garment attributes | Dataset ready, baseline pending |
| 3.2 Multimodal QA | Generate professional answers | Not started |
| 3.3 Agent/RAG | Intent recognition and fashion knowledge retrieval | Not started |

---

## 4. Main Limitations

1. The current segmentation pipeline depends on SAM-HQ, which is accurate but slow.
2. The text-guided region demo is rule-based, not open-vocabulary grounding.
3. The current local regions are limited to collar, sleeve, hem, waist, and pant_leg.
4. FashionAI attribute classification baseline has not been trained yet.
5. Shoes, bags, and accessories are not covered by the current main datasets.
6. Strict PRD performance targets are not yet addressed in the prototype stage.

---

## 5. Updated Next Steps

The current project should move from feasibility validation to deliverable modules.

Recommended order:

1. Finalize reports for the existing DeepFashion2 pipeline and query-region demo.
2. Build the FashionAI attribute classification baseline, starting with sleeve_length_labels.
3. Connect the region localization demo with the attribute classifier to form a region-to-attribute prototype.
4. Convert DeepFashion2 to YOLO-seg format and train a 5-class segmentation baseline.
5. Plan external data or manual annotation for shoes, bags, and accessories.

The immediate next engineering task is:

- P2: FashionAI attribute classification baseline.



Current Progress Summary
0. Executive Summary
The current project has completed the main feasibility validation for the visual garment analysis pipeline and has started producing deliverable modules aligned with the PRD.

Completed modules include:

DeepFashion2 annotation parsing benchmark
End-to-end garment detection / segmentation / landmark / region crop pipeline benchmark
Rule-based Chinese text-guided local region localization demo
FashionAI fine-grained attribute classification baselines for sleeve length, pant length, and neckline design
The project has now moved from pure feasibility validation to region-to-attribute prototype integration, where local garment regions such as neckline, sleeve, pant leg, and hem can be connected to fine-grained attribute classifiers.

Current main conclusions:

DeepFashion2 JSON annotation parsing is lightweight and not a system bottleneck.
The full garment visual pipeline is functional and stable.
SAM-HQ segmentation is the dominant runtime bottleneck.
Rule-based Chinese query-to-region localization is already usable as a P1 prototype.
FashionAI attribute classifiers are feasible and have produced usable baselines.
The next key engineering goal is to connect local region localization with attribute classification.
1. Completed Work
1.1 DeepFashion2 Ground-truth Processing Benchmark
The DeepFashion2 ground-truth annotation parsing benchmark has been completed.

Results
Metric	Value
Parsed annotation files	135,975
Processed garment instances	221,535
Full parsing time	31.319 s
File throughput	~4,342 files/s
Instance throughput	~7,073 instances/s
Conclusion
DeepFashion2 annotation parsing is lightweight.
Basic JSON annotation processing is not the bottleneck of the current system.
Future optimization should focus on model inference stages rather than annotation loading.
1.2 End-to-End Garment Pipeline Benchmark
A 500-image end-to-end benchmark has been completed on randomly sampled DeepFashion2 training images.

Pipeline
The benchmark covers the following stages:

YOLO garment detection
SAM-HQ garment segmentation
Garment landmark prediction
Semantic local region crop generation
Mask-aware semantic region crop generation
Overall Results
Metric	Value
Images	500
Processed garment instances	913
Region crops	3,220
Successful mask-aware crops	3,212
Total time	210.06 s
Average latency	420.13 ms/image
Throughput	2.38 images/s
Stage-wise Timing
Stage	Time	Percentage
YOLO detection	12.14 s	5.78%
SAM-HQ segmentation	146.25 s	69.62%
Landmark prediction	27.20 s	12.95%
Region crop generation	3.40 s	1.62%
Mask-aware crop generation	21.07 s	10.03%
Conclusion
The full visual pipeline is functional and stable.
SAM-HQ is the main runtime bottleneck.
Region crop generation itself is lightweight.
Mask-aware crop generation is not the primary bottleneck, but still contributes around 10% of total runtime.
The next runtime optimization direction is replacing or accelerating SAM-HQ, for example:
YOLO-seg
ONNXRuntime
TensorRT
lighter segmentation backbones
caching or batch-level optimization
1.3 Text-guided Local Region Demo
A rule-based Chinese query-to-region demo has been implemented and validated.

Supported Chinese Queries
Current supported queries include:

领口
左袖子
右袖子
下摆
腰部
裙摆
裤腿
Batch60 Validation Results
Metric	Value
Sampled images	60
Queries per image	5
Query runs	300
Success	276
Failed	24
Valid response rate	92.0%
Per-query Result
Query	Success Rate
腰部	100.0%
领口	95.0%
下摆	95.0%
左袖子	85.0%
右袖子	85.0%
Conclusion
The rule-based Chinese text-to-region prototype is functional and stable.
The module can map common Chinese garment-part queries to semantic local regions.
Failure cases are all caused by no_matching_region_crop, meaning the queried region was not available for the detected garment instance.
The current module is suitable to be treated as the P1 deliverable prototype.
Although it is not open-vocabulary grounding, it provides a stable engineering bridge from natural language query to local visual region.
2. FashionAI Attribute Classification Progress
2.1 Dataset Index Construction
The project already contains a reusable FashionAI attribute index builder:

scripts/build_fashionai_attribute_index.py
The script supports the following FashionAI attribute tasks:

Source Task	Short Task
sleeve_length_labels	sleeve_length
pant_length_labels	pant_length
skirt_length_labels	skirt_length
coat_length_labels	coat_length
collar_design_labels	collar_design
lapel_design_labels	lapel_design
neck_design_labels	neck_design
neckline_design_labels	neckline_design
The script converts FashionAI answer CSV rows into task-specific JSONL indexes, including:

train / val / test JSONL files
label map JSON
stats JSON
invalid sample report
markdown build report
Example generated files:

data/fashionai_attribute_index/sleeve_length_train.jsonl
data/fashionai_attribute_index/pant_length_train.jsonl
data/fashionai_attribute_index/neckline_design_train.jsonl
data/fashionai_attribute_index/label_map_neckline_design.json
data/fashionai_attribute_index/stats_neckline_design.json
2.2 Sleeve Length Classification
sleeve_length has completed the full baseline workflow:

FashionAI index construction
ResNet18 attribute classifier training
Evaluation
YOLO crop based weak evaluation
Crop strategy comparison
Current Status
Item	Status
Dataset index	Completed
Classifier training	Completed
Evaluation	Completed
YOLO crop inference	Completed
Weak evaluation	Completed
Weak Evaluation Result
The current best sleeve-length crop strategy achieved approximately:

Metric	Value
Strict weak accuracy	72.83%
Relaxed weak accuracy	80.35%
Conclusion
FashionAI-trained sleeve-length classifier transfers reasonably well to YOLO garment crops.
Crop strategy is important.
Class-aware or region-aware cropping improves weak evaluation quality.
Sleeve length is currently one of the more mature P2 attribute modules.
2.3 Pant Length Classification
pant_length has also completed the full baseline workflow.

Current Status
Item	Status
Dataset index	Completed
Classifier training	Completed
Evaluation	Completed
YOLO crop inference	Completed
Weak evaluation	Completed
Weak Evaluation Result
Metric	Value
Strict weak accuracy	69.89%
Relaxed weak accuracy	72.04%
Conclusion
The FashionAI-trained pant-length classifier is usable on YOLO garment crops.
Current weak accuracy is slightly lower than sleeve length.
Possible reasons include:
cropped-pant vs full-length ambiguity
detection box truncation
pose and occlusion
weak label mismatch between YOLO coarse garment classes and FashionAI fine-grained pant-length labels
2.4 Neckline Design Classification
neckline_design has been added as the next fine-grained attribute task.

Dataset Index Statistics
The neckline index was built from:

D:/Aliintern/fashion-ai-data/fashionai_attributes/round1_fashionAI_attributes_test_a
with answer CSV:

Tests/round1_fashionAI_attributes_answer_a.csv
Task:

neckline_design_labels
Dataset statistics:

Metric	Value
Total CSV rows	10,080
Task rows	2,095
Valid samples	2,058
Invalid samples	37
Missing images	0
Train samples	1,647
Val samples	206
Test samples	205
Number of classes	10
Unlike sleeve_length and pant_length, the generated neckline_design index includes Invisible as a valid class because raw label id 0 appears in the data.

Neckline Classes
Label ID	Label Name
0	Invisible
1	Strapless Neck
2	Deep V Neckline
3	Straight Neck
4	V Neckline
5	Square Neckline
6	Off Shoulder
7	Round Neckline
8	Sweat Heart Neck
9	One Shoulder Neckline
Test Metrics
A ResNet18 classifier was trained with pretrained weights.

Output directory:

outputs/p2_neckline_design_resnet18_seed2
Final test metrics:

Metric	Value
Test accuracy	67.80%
Macro precision	69.27%
Macro recall	66.16%
Macro F1	66.50%
Weighted F1	67.57%
Best val macro-F1	67.93%
Correct vs Incorrect Prediction Count
Item	Count	Percentage
Test samples	205	100.00%
Correct predictions	139	67.80%
Incorrect predictions	66	32.20%
Per-class Recall
Ground-truth Class	Test Count	Correct	Recall
Invisible	22	18	81.82%
Strapless Neck	22	14	63.64%
Deep V Neckline	32	23	71.88%
Straight Neck	12	4	33.33%
V Neckline	23	14	60.87%
Square Neckline	20	17	85.00%
Off Shoulder	19	14	73.68%
Round Neckline	18	12	66.67%
Sweat Heart Neck	20	12	60.00%
One Shoulder Neckline	17	11	64.71%
Per-class Precision
Predicted Class	Prediction Count	Correct	Precision
Invisible	31	18	58.06%
Strapless Neck	21	14	66.67%
Deep V Neckline	33	23	69.70%
Straight Neck	6	4	66.67%
V Neckline	20	14	70.00%
Square Neckline	26	17	65.38%
Off Shoulder	16	14	87.50%
Round Neckline	14	12	85.71%
Sweat Heart Neck	17	12	70.59%
One Shoulder Neckline	21	11	52.38%
Main Confusion Patterns
The main error patterns are visually reasonable:

V Neckline vs Deep V Neckline

V Neckline -> Deep V Neckline: 7 cases
Deep V Neckline -> V Neckline: 3 cases
This is expected because the difference is often a matter of depth rather than shape.
Strapless Neck vs Sweat Heart Neck

Strapless Neck -> Sweat Heart Neck: 4 cases
Sweat Heart Neck -> Strapless Neck: 6 cases
This is expected because sweetheart necklines are visually close to strapless necklines.
Off Shoulder vs One Shoulder Neckline

Off Shoulder -> One Shoulder Neckline: 4 cases
This can happen due to pose, occlusion, hair coverage, or asymmetric crop boundaries.
Straight Neck is the weakest class

Test count: 12
Correct: 4
Recall: 33.33%
It is confused with Invisible, Strapless, Square, Off Shoulder, and One Shoulder.
This class has both low sample count and ambiguous visual boundaries.
Visible neckline predicted as Invisible

Non-Invisible ground-truth samples: 183
Predicted as Invisible: 13
Rate: 7.10%
This is not the dominant error type, but should be monitored.
Neckline Baseline Conclusion
The neckline_design baseline is functional and usable.
Most errors occur between visually similar neckline categories.
Square Neckline and Invisible have strong recall.
Round Neckline has high precision but moderate recall.
Straight Neck is the weakest class.
This 10-class classifier is suitable as the first neckline attribute module for region-to-attribute integration.
Future ablations may include:
visible-only 9-class training without Invisible
class-weighted training
longer training
coarse-group relaxed evaluation
expanded upper-neck crop inference
3. Current Dataset Coverage
Dataset	Supported Module	Current Status
DeepFashion2	Garment detection, segmentation, landmarks, local region crops	Integrated
FashionAI Attributes	Fine-grained attribute classification	Sleeve, pant, and neckline baselines completed
Shoes/Bags/Accessories	PRD extended classes	Not covered by current main datasets
3.1 DeepFashion2 Coverage
DeepFashion2 currently supports five PRD garment categories:

top
pants
skirt
outerwear
dress
The following PRD categories are not covered by DeepFashion2:

shoes
bag
accessory
3.2 FashionAI Attribute Coverage
FashionAI Attributes supports several first-stage attribute tasks:

sleeve length
skirt length
pant length
coat length
collar design
lapel design
neck design
neckline design
Current trained / evaluated tasks:

Attribute Task	Status
sleeve_length	Baseline completed
pant_length	Baseline completed
neckline_design	Baseline completed
skirt_length	Index supported, training pending
coat_length	Index supported, training pending
collar_design	Index supported, training pending
lapel_design	Index supported, training pending
neck_design	Index supported, training pending
4. Current PRD Alignment
PRD Module	Requirement	Current Status
3.1.1 Garment instance segmentation	Detect and segment garment instances	Prototype ready using YOLO + SAM-HQ
3.1.2 Language-guided local region localization	Locate garment parts from natural language query	Rule-based Chinese query prototype ready
3.1.3 Fine-grained attribute extraction	Extract local garment attributes	Sleeve, pant, and neckline baselines completed
3.2 Multimodal QA	Generate professional answers	Not started
3.3 Agent/RAG	Intent recognition and fashion knowledge retrieval	Not started
4.1 Completed PRD-aligned Capabilities
The current system can already demonstrate:

Garment instance detection

Detect major garment categories using YOLO.
Garment instance segmentation

Segment garment instances using SAM-HQ.
Landmark and region processing

Generate semantic local regions such as neckline, sleeve, waist, hem, skirt hem, and pant leg.
Chinese query-to-region prototype

Convert Chinese queries such as 领口 and 左袖子 into local region crops.
Fine-grained attribute classification

Classify sleeve length, pant length, and neckline design using FashionAI-trained classifiers.
4.2 Partially Completed PRD Capabilities
The following capabilities are partially completed and require integration:

Region-to-attribute inference

The region localization module is ready.
Attribute classifiers are ready for several tasks.
The next step is to connect local region crops to the corresponding attribute classifiers.
Text-guided fine-grained attribute extraction

Current query-to-region is rule-based.
Attribute extraction can be attached to selected regions.
A full natural-language-to-attribute pipeline is not yet complete.
4.3 Not Yet Started
The following PRD modules have not started:

Multimodal QA answer generation
Agent-based intent recognition
Fashion knowledge RAG
Open-vocabulary grounding
Shoes / bags / accessories analysis
5. Main Limitations
SAM-HQ runtime bottleneck

SAM-HQ accounts for about 69.62% of the end-to-end runtime.
This is the most important performance bottleneck.
Rule-based query-to-region localization

The current text-guided localization module is rule-based.
It is stable for supported Chinese queries but not open-vocabulary.
Limited local region vocabulary

Current local regions are mainly:
collar / neckline
sleeve
hem
waist
skirt hem
pant leg
Attribute classifiers are still baseline models

Current FashionAI models use ResNet18 baselines.
More advanced models or crop strategies may improve performance.
Domain gap between FashionAI and YOLO pipeline crops

FashionAI classifiers are trained on FashionAI attribute images.
Deployment uses YOLO / region crops.
Crop alignment and domain transfer remain important evaluation points.
Shoes, bags, and accessories are not covered

DeepFashion2 does not cover these PRD categories.
External data or manual annotation will be required.
Strict PRD production targets are not yet addressed

The current work is still prototype-oriented.
Production-level latency, robustness, and open-domain coverage require further work.
6. Current Engineering Assets
6.1 Important Scripts
Script	Purpose
scripts/build_fashionai_attribute_index.py	Build FashionAI JSONL indexes
tools/train/train_attribute_classifier.py	Train FashionAI attribute classifiers
tools/demo/predict_garment_attribute_from_yolo_crops.py	Run attribute classifiers on YOLO garment crops for sleeve/pant
tools/infer/make_expanded_yolo_crops.py	Generate expanded YOLO crops
tools/infer/run_garment_pipeline.py	Run garment analysis pipeline
tools/infer/garment_pipeline.py	Core garment pipeline logic
tools/infer/predict_garments_yolo.py	YOLO garment detection
tools/infer/segment_garments_samhq.py	SAM-HQ garment segmentation
tools/infer/infer_landmarks_for_predictions.py	Landmark inference
scripts/scan_fashionai_attributes.py	Scan FashionAI dataset structure
scripts/quick_count_data.py	Quick dataset count utility
6.2 Important Output Directories
Directory	Content
data/fashionai_attribute_index	FashionAI train/val/test JSONL and label maps
outputs/p2_neckline_design_resnet18_seed2	Neckline classifier output
outputs/pipeline_13cls_eval_balanced	YOLO / pipeline evaluation outputs
outputs/p2_fashionai_scan	FashionAI scan reports
7. Updated Next Steps
The current project should continue moving from feasibility validation to integrated deliverable modules.

7.1 Immediate Next Tasks
Task 1: Visualize and review neckline baseline errors
Generate contact sheets for:

error_cases_test.csv
test_predictions.csv
Purpose:

Validate whether high-confidence errors are visually reasonable.
Inspect confusion cases such as:
V Neckline vs Deep V Neckline
Strapless Neck vs Sweat Heart Neck
Off Shoulder vs One Shoulder Neckline
Straight Neck vs Square Neckline
visible neckline predicted as Invisible
Task 2: Extend crop-based attribute inference to neckline_design
Current script:

tools/demo/predict_garment_attribute_from_yolo_crops.py
currently supports:

sleeve_length
pant_length
It should be extended to support:

neckline_design
or a generic attribute inference mode.

The first goal is not weak accuracy, but:

prediction JSONL
prediction summary
class distribution
visual inspection
Task 3: Run neckline_design on YOLO garment crops
Use:

outputs/p2_neckline_design_resnet18_seed2/best.pt
data/fashionai_attribute_index/label_map_neckline_design.json
Run inference on YOLO garment crops and inspect qualitative behavior.

Task 4: Run neckline_design on expanded / upper-neck crops
Compare several crop strategies:

Crop Strategy	Purpose
YOLO tight garment crop	Baseline
Expanded garment crop	More context
Upper60 expanded crop	Recommended first deployment crop for neckline
Expanded neckline region crop	Final PRD-aligned local-region crop
Recommended neckline crop direction:

expanded upper-neck crop
rather than overly tight local crop.

7.2 Medium-term Tasks
Train additional FashionAI attribute classifiers:

collar_design
neck_design
lapel_design
skirt_length
coat_length
Build a region-to-attribute prototype:

Query: 领口
Region: neckline / upper-neck crop
Attribute: neckline_design classifier
Output: structured attribute result
Convert DeepFashion2 to YOLO-seg format and train a 5-class segmentation baseline:

top
pants
skirt
outerwear
dress
Investigate segmentation acceleration:

YOLO-seg
ONNXRuntime
TensorRT
batch inference
model distillation
Plan external data or manual annotation for unsupported PRD categories:

shoes
bags
accessories
8. Recommended Delivery Milestones
Milestone P1: Local Region Localization Prototype
Status: Completed

Deliverables:

YOLO + SAM-HQ garment pipeline
Local semantic region crop generation
Chinese query-to-region demo
Batch60 validation report
Milestone P2: Fine-grained Attribute Classification Baseline
Status: Partially completed

Completed:

sleeve_length
pant_length
neckline_design
Pending:

collar_design
neck_design
lapel_design
skirt_length
coat_length
Milestone P3: Region-to-Attribute Prototype
Status: Next priority

Target demo:

Input image + Chinese query "领口"
-> garment detection
-> local neckline region crop
-> neckline_design classifier
-> output fine-grained neckline attribute
Example expected output:

{
  "query": "领口",
  "region": "neckline",
  "attribute_task": "neckline_design",
  "prediction": "V Neckline",
  "confidence": 0.73
}
Milestone P4: Runtime Optimization
Status: Planned

Main target:

Reduce or replace SAM-HQ dependency.
Build a faster segmentation pipeline.
Milestone P5: PRD Extended Categories
Status: Planned

Required categories:

shoes
bags
accessories
Data strategy is still pending.

9. Final Current Status
The project currently has a working prototype stack:

image
-> garment detection
-> segmentation
-> landmarks
-> local region crops
-> Chinese query-to-region mapping
-> FashionAI attribute classifiers
Completed proof points:

DeepFashion2 parsing is fast.
End-to-end garment pipeline is functional.
Query-to-region localization works with 92.0% valid response rate on Batch60.
Sleeve-length and pant-length attribute classifiers have completed crop-based weak evaluation.
Neckline-design classifier has completed FashionAI in-domain baseline with 67.80% test accuracy and 66.50% macro-F1.
The immediate next step is to connect the neckline local region pipeline with the neckline_design classifier and evaluate different crop strategies, especially:

YOLO garment crop
expanded garment crop
upper60 expanded crop
expanded neckline region crop


task	experiment	best_epoch	val_acc	val_macro_f1	test_acc	test_macro_f1	test_weighted_f1
neckline_design	baseline/original	7	0.684466	0.679304	0.678049	0.665043	0.675721
neck_design	multiview_v2_pipeline	6	0.682692	0.680770	0.632850	0.623992	0.631733
collar_design	multiview_v2_pipeline	8	0.716981	0.709467	0.772586	0.764234	0.768946
lapel_design	multiview_v2_pipeline	8	0.637037	0.635711	0.674074	0.679597	0.671422
sleeve_length	multiview_v2_pipeline	6	0.658643	0.641720	0.633262	0.611810	0.637866
coat_length	multiview_v2_pipeline	10	0.724014	0.691766	0.689286	0.617544	0.671381
pant_length	multiview_v2_pipeline	5	0.706522	0.688248	0.760870	0.739689	0.759625
skirt_length	multiview_v2_pipeline	16	0.654867	0.639894	0.611607	0.592879	0.609096