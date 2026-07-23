"""
Inference optimization module for fashion fine-grained vision pipeline.

This package provides:
- TensorRT-accelerated model wrappers (wrappers/)
- Per-path pipeline definitions (pipelines/)
- Benchmark harness with latency taxonomy (benchmarks/)
- LLM client abstraction layer (llm/)
- FastAPI serving infrastructure (serving/)
- ONNX/TensorRT export scripts (export/)

Status: Pre-implementation / Planning phase.
No TensorRT engines exist yet. All wrappers default to PyTorch fallback.
"""
