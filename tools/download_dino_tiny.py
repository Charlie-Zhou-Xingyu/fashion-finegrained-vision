#!/usr/bin/env python3
"""One-time download script for Grounding DINO Tiny model.

Downloads from HuggingFace Hub and saves to a local directory so the
3.1.2 benchmark and inference can run without internet access.

Usage:
    python tools/download_dino_tiny.py [--output models/grounding_dino_tiny]
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Grounding DINO Tiny to a local directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/grounding_dino_tiny",
        help="Output directory (default: models/grounding_dino_tiny)",
    )
    args = parser.parse_args()

    model_id = "IDEA-Research/grounding-dino-tiny"
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
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        print(f"  Loading processor from {path} ...")
        AutoProcessor.from_pretrained(str(path), local_files_only=True)
        print("  Processor OK")

        print(f"  Loading model from {path} ...")
        model = GroundingDinoForObjectDetection.from_pretrained(
            str(path), local_files_only=True
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model OK ({n_params:,} parameters)")

        print("\nVerification passed. DINO Tiny is ready for local use.")
        print(f"  Use: GroundingDINOLocator(model_id='{path}')")
    except Exception as e:
        print(f"Verification FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
