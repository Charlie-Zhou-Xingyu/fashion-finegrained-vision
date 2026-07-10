#!/usr/bin/env python3
"""One-time download script for Grounding DINO Base model.

Downloads from HuggingFace Hub and saves to a local directory.
DINO-base uses Swin-Base backbone (~100M params) vs Swin-Tiny (~28M) for tiny.

Usage:
    python tools/download_dino_base.py [--output models/grounding_dino_base]
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Grounding DINO Base to a local directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/grounding_dino_base",
        help="Output directory (default: models/grounding_dino_base)",
    )
    args = parser.parse_args()

    model_id = "IDEA-Research/grounding-dino-base"
    output_dir = Path(args.output)

    # Check if already downloaded
    required_files = ["config.json", "pytorch_model.bin", "preprocessor_config.json"]
    if output_dir.exists() and all(
        (output_dir / f).exists() for f in required_files
    ):
        print(f"Model already exists at {output_dir.resolve()}")
        _verify(output_dir)
        return

    print(f"Downloading {model_id} ...")
    print("This requires internet access (one-time only).")
    print(f"Target: {output_dir.resolve()}")
    print(f"Expected size: ~400-500 MB (Swin-Base backbone)")
    print("-" * 60)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except ImportError:
        print("huggingface_hub not available, trying transformers...")
        try:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection

            print("Loading model from HuggingFace Hub...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = GroundingDinoForObjectDetection.from_pretrained(model_id)

            print(f"Saving to {output_dir} ...")
            output_dir.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(str(output_dir))
            model.save_pretrained(str(output_dir))
        except ImportError:
            print(
                "ERROR: Neither huggingface_hub nor transformers is installed.",
                file=sys.stderr,
            )
            print(
                "Install with: pip install transformers huggingface_hub", file=sys.stderr
            )
            sys.exit(1)

    print("-" * 60)
    print("Download complete. Verifying...")
    _verify(output_dir)


def _verify(path: Path) -> None:
    """Verify the downloaded model can be loaded locally."""
    try:
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        print(f"  Loading processor from {path} ...")
        AutoProcessor.from_pretrained(str(path), local_files_only=True)
        print("  Processor OK")

        print(f"  Loading model from {path} ...")
        model = GroundingDinoForObjectDetection.from_pretrained(
            str(path), local_files_only=True
        )
        n_params = sum(p.numel() for p in model.parameters())
        # Estimate VRAM: fp32 params * 4 bytes
        est_vram_mb = n_params * 4 / (1024 * 1024)
        print(f"  Model OK ({n_params:,} parameters, ~{est_vram_mb:.0f} MB fp32)")

        # Check GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
            model = model.to("cuda")
            mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  GPU VRAM used: {mem_after - mem_before:.0f} MB")
            model = model.to("cpu")
            torch.cuda.empty_cache()

        print("\nVerification passed. DINO Base is ready for local use.")
        print(f"  Use: GroundingDINOLocator(model_id='{path}')")
    except Exception as e:
        print(f"Verification FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
