"""Thin CLI smoke-test wrapper for mask_attribute_pipeline.

Delegates all logic to
``src/fashion_vision/attributes/mask_attribute_pipeline.py``.

Example::

    python scripts/run_attribute_from_mask_smoke.py \\
        --image assets/random_train60/images/000004.jpg \\
        --mask "outputs/test_pipeline_smoke/02_samhq/masks/000004_det000_long sleeve top_mask.png" \\
        --garment-category top \\
        --component-type collar \\
        --output-dir outputs/smoke_test_attr_from_mask \\
        --device cpu \\
        --topk 3
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path when run as a plain script.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fashion_vision.attributes.mask_attribute_pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
