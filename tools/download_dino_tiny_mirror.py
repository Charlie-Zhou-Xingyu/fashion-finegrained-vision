#!/usr/bin/env python3
"""Download DINO Tiny via HF mirror (for users in China)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    "IDEA-Research/grounding-dino-tiny",
    local_dir="models/grounding_dino_tiny",
)
print("DONE")
