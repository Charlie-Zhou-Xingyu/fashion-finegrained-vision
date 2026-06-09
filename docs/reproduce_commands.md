# Reproduce Commands

This document records the main commands used in the current project stage.

Environment:

```text
Conda env: fashion-demo2
Project root: D:\Aliintern\fashion-finegrained-vision
Dataset root: D:\Aliintern\fashion-ai-data
```

Before running commands, enter the project root:

```bat
cd /d D:\Aliintern\fashion-finegrained-vision
conda activate fashion-demo2
```

---

## 1. Single-image Query Region Demo

Example image:

```text
D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg
```

### 1.1 Query Waist

```bat
python -m tools.demo.query_region_online_demo ^
  --image D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg ^
  --query "腰部" ^
  --output-dir outputs\query_region_online_demo
```

Expected behavior:

- Run full pipeline.
- Generate selected waist region.
- If both upper-body waist and skirt/pants waist exist, select upper-body waist first.

---

### 1.2 Reuse Pipeline for Other Queries

After the first command, find the generated pipeline directory, for example:

```text
outputs\query_region_online_demo\000252_waist_20260609_182447\pipeline
```

Then reuse it.

#### Collar

```bat
python -m tools.demo.query_region_online_demo ^
  --image D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg ^
  --query "领口" ^
  --output-dir outputs\query_region_online_demo ^
  --reuse-pipeline-dir outputs\query_region_online_demo\000252_waist_20260609_182447\pipeline
```

#### Left Sleeve

```bat
python -m tools.demo.query_region_online_demo ^
  --image D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg ^
  --query "左袖子" ^
  --output-dir outputs\query_region_online_demo ^
  --reuse-pipeline-dir outputs\query_region_online_demo\000252_waist_20260609_182447\pipeline
```

#### Right Sleeve

```bat
python -m tools.demo.query_region_online_demo ^
  --image D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg ^
  --query "右袖子" ^
  --output-dir outputs\query_region_online_demo ^
  --reuse-pipeline-dir outputs\query_region_online_demo\000252_waist_20260609_182447\pipeline
```

#### Generic Hem

```bat
python -m tools.demo.query_region_online_demo ^
  --image D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg ^
  --query "下摆" ^
  --output-dir outputs\query_region_online_demo ^
  --reuse-pipeline-dir outputs\query_region_online_demo\000252_waist_20260609_182447\pipeline
```

#### Skirt Hem

```bat
python -m tools.demo.query_region_online_demo ^
  --image D:\Aliintern\fashion-ai-data\deepfashion2\train\image\000252.jpg ^
  --query "裙摆" ^
  --output-dir outputs\query_region_online_demo ^
  --reuse-pipeline-dir outputs\query_region_online_demo\000252_waist_20260609_182447\pipeline
```

Expected behavior:

- `下摆` is treated as generic hem.
- `裙摆` is treated as skirt hem and should only select `class_name=skirt`.

---

## 2. Batch60 Query Region Validation

This command randomly samples 60 images and runs five queries per image.

Queries:

```text
领口
左袖子
右袖子
下摆
腰部
```

Command:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -Command "$BatchStart=Get-Date; $ImageDir='D:\Aliintern\fashion-ai-data\deepfashion2\train\image'; $OutDir='outputs\query_region_online_demo_batch60'; $Seed=20260609; $N=60; $Queries=@('领口','左袖子','右袖子','下摆','腰部'); $PipelineOk=0; $PipelineFail=0; $QueryOk=0; $QueryFail=0; New-Item -ItemType Directory -Force -Path $OutDir | Out-Null; $images=Get-ChildItem $ImageDir -Filter *.jpg | Sort-Object { Get-Random -SetSeed $Seed } | Select-Object -First $N; $listPath=Join-Path $OutDir 'sampled_60_images.txt'; $images.FullName | Set-Content -Encoding UTF8 $listPath; Write-Host '[INFO] sampled images saved to' $listPath; foreach($img in $images){ $stem=[IO.Path]::GetFileNameWithoutExtension($img.FullName); Write-Host ''; Write-Host '============================================================'; Write-Host '[IMAGE]' $img.FullName; Write-Host '============================================================'; Write-Host '[STEP] Run full pipeline once with query=腰部'; $pStart=Get-Date; python -m tools.demo.query_region_online_demo --image $img.FullName --query '腰部' --output-dir $OutDir; $pExit=$LASTEXITCODE; $pEnd=Get-Date; $pSec=($pEnd-$pStart).TotalSeconds; Write-Host ('[TIME] full pipeline command seconds: {0:N2}' -f $pSec); if($pExit -ne 0){ $PipelineFail++; Write-Host '[WARN] pipeline command failed for' $stem; continue } else { $PipelineOk++ }; $latest=Get-ChildItem $OutDir -Directory -Filter ($stem + '_waist_*') | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($null -eq $latest){ $PipelineFail++; Write-Host '[WARN] no output dir found for' $stem; continue }; $pipeline=Join-Path $latest.FullName 'pipeline'; if(!(Test-Path (Join-Path $pipeline '05_region_masked_crops\region_masked_crops.json'))){ $PipelineFail++; Write-Host '[WARN] pipeline json not found:' $pipeline; continue }; foreach($q in $Queries){ Write-Host '[QUERY]' $q; $qStart=Get-Date; python -m tools.demo.query_region_online_demo --image $img.FullName --query $q --output-dir $OutDir --reuse-pipeline-dir $pipeline; $qExit=$LASTEXITCODE; $qEnd=Get-Date; $qSec=($qEnd-$qStart).TotalSeconds; Write-Host ('[TIME] query={0}, seconds={1:N2}, exit={2}' -f $q,$qSec,$qExit); if($qExit -eq 0){ $QueryOk++ } else { $QueryFail++ } } }; $BatchEnd=Get-Date; $TotalSec=($BatchEnd-$BatchStart).TotalSeconds; $summary=[ordered]@{ batch_start=$BatchStart.ToString('yyyy-MM-dd HH:mm:ss'); batch_end=$BatchEnd.ToString('yyyy-MM-dd HH:mm:ss'); total_seconds=[math]::Round($TotalSec,2); total_minutes=[math]::Round($TotalSec/60,2); image_dir=$ImageDir; output_dir=$OutDir; sampled_images=$images.Count; queries=$Queries; expected_query_runs=$images.Count*$Queries.Count; pipeline_ok=$PipelineOk; pipeline_fail=$PipelineFail; query_ok=$QueryOk; query_fail=$QueryFail }; $summaryPath=Join-Path $OutDir 'batch_summary.json'; $summary | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $summaryPath; Write-Host ''; Write-Host '==================== BATCH SUMMARY ===================='; $summary | ConvertTo-Json -Depth 5; Write-Host '[DONE] summary saved to' $summaryPath"
```

Generated files:

```text
outputs\query_region_online_demo_batch60\batch_summary.json
outputs\query_region_online_demo_batch60\sampled_60_images.txt
```

---

## 3. Batch60 Result

The completed batch60 result was:

```json
{
  "batch_start": "2026-06-09 18:50:54",
  "batch_end": "2026-06-09 19:13:34",
  "total_seconds": 1359.44,
  "total_minutes": 22.66,
  "sampled_images": 60,
  "expected_query_runs": 300,
  "pipeline_ok": 60,
  "pipeline_fail": 0,
  "query_ok": 276,
  "query_fail": 24
}
```

---

## 4. Suggested Next Reproduction Commands

### 4.1 Summarize Batch Results

To be added after implementing:

```text
tools/demo/summarize_query_region_batch.py
```

Expected command:

```bat
python -m tools.demo.summarize_query_region_batch ^
  --batch-dir outputs\query_region_online_demo_batch60 ^
  --output-csv outputs\query_region_online_demo_batch60\batch_results.csv ^
  --output-json outputs\query_region_online_demo_batch60\query_success_summary.json ^
  --failed-csv outputs\query_region_online_demo_batch60\failed_cases.csv
```

---

### 4.2 FashionAI Attribute Baseline

To be added after implementing the FashionAI attribute baseline scripts.

Planned first task:

```text
sleeve_length_labels
```

Planned outputs:

```text
outputs\attribute_baseline\sleeve_length\best.pt
outputs\attribute_baseline\sleeve_length\metrics.json
outputs\attribute_baseline\sleeve_length\confusion_matrix.png
```
