"""
Per-path pipeline definitions.

Each pipeline path is a self-contained module that chains model wrappers:
    fast_path.py  — YOLO + SAM + Landmark + Crop (high-throughput)
    query_path.py — Fast Path + DINO/FP part localization (query-dependent)
    full_analysis_path.py — Query Path + Attributes + Inner + LLM fallback

Status: Pre-implementation. No pipeline files exist yet.
These will be written during Week 4-7 of the optimization roadmap.
"""
