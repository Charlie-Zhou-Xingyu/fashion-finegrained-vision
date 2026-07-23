"""
Performance benchmark harness for inference pipeline.

Sub-modules:
- microbench_model.py: Model-only benchmarks (synthetic tensor, no I/O)
- bench_stage.py: Stage-level benchmarks (real input, pre+model+post)
- bench_pipeline.py: End-to-end pipeline benchmarks (real images from disk)
- benchmark_runner.py: Unified CLI entry point

Output format follows the schema in docs/inference_optimization_plan_v2.md Appendix B.
"""
