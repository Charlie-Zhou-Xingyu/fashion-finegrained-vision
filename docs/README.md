# Documentation Index

This directory contains all project documentation for the fashion fine-grained vision system.

---

## Canonical Status Documents

These two files are the authoritative references for the current state of the project.

- [Current Project Status](current_project_status.md) — PRD module status, pipeline stages, immediate next steps, and safety constraints.
- [Attribute Pipeline Step 1–10 Engineering Summary](attribute_pipeline_step_1_10_summary.md) — Architecture overview, step-by-step implementation log, test results, and remaining work for PRD 3.1.3.

---

## Architecture

Design decisions and cleanup plans for the codebase structure.

- [Codebase Cleanup Plan](architecture/codebase_cleanup_plan.md) — Stage-by-stage cleanup plan; Stage 1 complete, Stages 2–3 deferred.
- [Code Quality Refactor Plan](architecture/code_quality_refactor_plan.md) — Planned refactoring of demo scripts into reusable modules.

---

## Plans

Implementation and training plans for upcoming pipeline work.

- [3.1.1 SAM Instance Segmentation Plan](plans/3_1_1_sam_instance_segmentation_plan.md) — PRD 3.1.1 functional specification and development plan.
- [YOLO Balanced Retraining Plan](plans/yolo_balanced_training_plan.md) — Class-balanced YOLOv8 retraining strategy for the 13-class garment detector.

---

## Reports

Benchmark results and experiment reports produced during P1/P2.

- [DeepFashion2 GT Processing Benchmark](reports/benchmark_report.md) — Annotation parsing throughput benchmark (135,975 files, 221,535 instances).
- [Pipeline Benchmark — 500 Images](reports/pipeline_benchmark_500_report.md) — End-to-end YOLO+SAM-HQ+landmark+crop pipeline benchmarked on 500 images (420 ms/image).
- [Query-Region Batch60 Report](reports/query_region_batch60_report.md) — Rule-based query-to-region demo evaluation (92% valid response rate).
- [Text-Guided Region Demo Report](reports/text_guided_region_demo_report.md) — Text-guided local region localization demo results.
- [Data Scan Report](reports/data_scan_report.md) — FashionAI attribute dataset scan summary.

---

## Logs

Experiment logs and reproduction command references.

- [Experiment Log](logs/experiment_log.md) — Chronological record of experiments and outcomes.
- [Reproduce Commands](logs/reproduce_commands.md) — Shell commands to reproduce key pipeline runs and training jobs.

---

## Archive

Older documents superseded by newer canonical versions.

- [Current Progress Summary (archived)](archive/current_progress_summary.md) — Earlier progress summary; superseded by `current_project_status.md`.
