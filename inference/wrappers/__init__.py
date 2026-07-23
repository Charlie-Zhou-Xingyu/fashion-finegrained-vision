"""
Model wrappers with TensorRT acceleration and automatic PyTorch fallback.

Each wrapper exposes a common interface:
    - __init__(engine_path, pt_path, use_fallback=False, ...)
    - infer(inputs) -> outputs

When use_fallback=True or the engine file is missing, the wrapper silently
falls back to the original PyTorch model. This is the primary rollback mechanism.

No TensorRT engines exist yet — all wrappers will initially run in fallback mode.
"""
