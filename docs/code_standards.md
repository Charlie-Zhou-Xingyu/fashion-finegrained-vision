# Code Standards — Fashion Fine-Grained Vision

> Applies to all code under `src/`, `tools/`, `scripts/`, and `tests/`.  
> Last updated: 2026-06-23

---

## 1. Modular Design

Split code by responsibility. One file = one coherent concern.

**Rule:** No file should contain both business logic and I/O, or both model inference and result formatting.

**Preferred structure:**

```
src/fashion_vision/
  localization/
    intent_parser.py          ← NLP parsing only
    grounding_dino_locator.py ← DINO inference only
    part_shape_priors.py      ← geometric filtering only
    region_localization_router.py ← routing + orchestration only
  models/
    sam_hq_wrapper.py         ← SAM inference only
  data/
    class_mapping.py          ← category mapping only
```

**Anti-patterns to avoid:**
- A 500-line script that does detection, filtering, visualization, and file I/O in sequence.
- Putting model loading inside a utility function used by multiple callers.
- Business logic inside `__main__` blocks (use `__main__` only for smoke tests).

---

## 2. Configuration Files

All tunable parameters live in config files. Code reads them; code does not define them.

**Formats in this project:**
- `configs/*.yaml` — pipeline parameters, category mappings, per-part thresholds.
- `src/fashion_vision/localization/part_detection_config.py` — per-part DINO prompts and shape configs (Python dict, not YAML, because it is imported by other modules).

**Rule:** A parameter that may change between runs, environments, or experiments is a config value, not a constant.

**Examples of what belongs in config:**
```yaml
# configs/localization.yaml
dino_model_id: "IDEA-Research/grounding-dino-tiny"
default_box_threshold: 0.30
default_text_threshold: 0.25
mask_fill_mode: "grey"          # grey | black | white
mask_dilation_px_default: 0
translation_server_url: ""      # empty = disabled
translation_timeout_s: 3.0
```

**Anti-patterns to avoid:**
```python
# BAD — hardcoded inside business logic
threshold = 0.35
model_id = "IDEA-Research/grounding-dino-tiny"

# GOOD — read from config at startup
threshold = config["box_threshold"]
model_id = config["dino_model_id"]
```

---

## 3. Exception Handling

Handle exceptions at system boundaries. Trust internal code; validate external inputs.

**Boundaries that require explicit handling:**
- File I/O (images, masks, JSON, YAML)
- External HTTP calls (translation service, model servers)
- Model inference calls (DINO, SAM, landmark predictor)
- User-supplied query strings

**Pattern:**

```python
def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a binary mask from a PNG file."""
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cv2.imread returned None for: {path}")
    return (mask > 0).astype(np.uint8) * 255
```

**For non-fatal degraded paths, log and continue — do not silently swallow:**

```python
try:
    garment_mask = load_binary_mask(mask_path)
except FileNotFoundError:
    logger.warning("Garment mask not found at %s — running without mask", mask_path)
    garment_mask = None
```

**Rules:**
- Never use a bare `except:` or `except Exception:` without logging.
- Never return a default value from a failed load without logging what failed.
- Raise to the caller when the failure is unrecoverable (image not readable, required model not loaded).
- Use `logger.warning` for degraded but continued execution; `logger.error` for failures that affect output correctness.

---

## 4. Docstrings and Comments

### Docstrings

Every public function and class gets a docstring. Use Google style.

```python
def filter_by_shape_priors(
    detections: list[dict],
    part: str | None,
    garment_bbox: list | None = None,
    shape_config: dict | None = None,
) -> list[dict]:
    """
    Filter DINO detections using per-part geometric priors.

    Adds ``_shape_prior_status`` to each detection dict. Returns only
    detections that pass all configured checks. Returns an empty list
    (not a fallback candidate) when all candidates are rejected.

    Args:
        detections: List of detection dicts with ``bbox_xyxy`` and ``score``.
            Modified in-place to add ``_shape_prior_status``.
        part: Canonical part name for config lookup (e.g. ``"zipper"``).
            Ignored when ``shape_config`` is supplied explicitly.
        garment_bbox: Parent garment ``[x1, y1, x2, y2]`` in full-image coords.
            Area-ratio, y_band, x_band, and center_x checks are skipped when None.
        shape_config: Explicit config dict. Pass ``{}`` to disable all checks.
            ``None`` → look up from ``part`` in ``PART_DETECTION_CONFIG``.

    Returns:
        Subset of detections that passed all checks, preserving original order.
        Empty list if all candidates are rejected.
    """
```

### Comments

Write comments for non-obvious *why*, not the obvious *what*. Delete comments that just restate the code.

```python
# BAD — restates what the code does
# Check if score is above threshold
if score > threshold:
    ...

# GOOD — explains why
# GDINO requires a trailing period for reliable token-level scoring
if not text_query.rstrip().endswith("."):
    text_query = text_query.rstrip() + "."

# GOOD — marks a deliberate simplification with its known ceiling
# ponytail: bbox aspect ratio only — true circularity requires the SAM mask
```

**Inline `ponytail:` comments** mark deliberate shortcuts. Include the limitation and upgrade path.

---

## 5. Type Hints

All function signatures use type hints. Use `from __future__ import annotations` at the top of every file so forward references work without quotes.

```python
from __future__ import annotations

from typing import Optional
import numpy as np


def select_side_detection(
    detections: list[dict],
    side: str,
) -> list[dict]:
    ...


def compute_containment(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> ContainmentResult:
    ...


def locate_region(
    query: str,
    instance: dict[str, Any],
    image: np.ndarray,
    image_width: int,
    image_height: int,
    locator: Optional[GroundingDINOLocator] = None,
    dino_threshold: float = 0.3,
) -> dict[str, Any]:
    ...
```

**Rules:**
- Use `Optional[X]` (or `X | None` in Python ≥ 3.10) for nullable parameters.
- Use `TYPE_CHECKING` guards for heavyweight imports that are only needed for type hints:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fashion_vision.localization.grounding_dino_locator import GroundingDINOLocator
```

- Avoid `Any` where a more specific type is feasible. Use it for `dict` values from JSON where the schema varies.
- Dataclasses are preferred over raw `dict` for structured return values.

---

## 6. Code Reuse

Encapsulate repeated logic. Three identical code blocks = extract a function.

**Common patterns in this project that must not be duplicated:**

| Logic | Canonical location |
|---|---|
| Load binary mask from PNG | `region_locator.load_binary_mask()` |
| Compute bbox IoU | `grounding_dino_locator._box_iou()` |
| Parse query intent | `intent_parser.parse_intent()` |
| Crop image to instance bbox | `region_localization_router._crop_image_and_mask()` |
| Get per-part DINO prompts | `part_detection_config.get_part_prompts()` |
| Get per-part thresholds | `part_detection_config.get_part_thresholds()` |
| Category mapping (13→5) | `class_mapping.py` + `configs/category_mapping.yaml` |

**Anti-patterns:**
```python
# BAD — bbox IoU reimplemented inline in a new script
def my_iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * ...

# GOOD — import and reuse
from fashion_vision.localization.grounding_dino_locator import _box_iou
```

If a utility is used in more than one module, it belongs in a shared utility module, not duplicated.

---

## 7. PEP 8 and Naming Conventions

Follow PEP 8. Use the project's existing naming style, summarised here:

| Entity | Convention | Example |
|---|---|---|
| Module / file | `lowercase_underscore` | `intent_parser.py` |
| Class | `CamelCase` | `GroundingDINOLocator`, `QueryIntent` |
| Function / method | `lowercase_underscore` | `parse_intent()`, `filter_by_shape_priors()` |
| Constant (module-level) | `UPPER_UNDERSCORE` | `FAST_PATH_PARTS`, `DEFAULT_BOX_THRESHOLD` |
| Private helper | leading underscore | `_box_iou()`, `_crop_image_and_mask()` |
| Type alias | `CamelCase` or `UPPER_UNDERSCORE` | `BboxXYXY` |

**Line length:** 100 characters max (project default; enforced by linting if configured).

**Imports:** stdlib → third-party → local, separated by blank lines:
```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from fashion_vision.localization.intent_parser import parse_intent
from fashion_vision.localization.part_detection_config import get_part_prompts
```

**Avoid:**
- Single-letter variable names outside comprehensions and loop indices.
- Abbreviations that are not obvious: `cfg` is acceptable; `lclztn` is not.
- Mutable default arguments (`def f(x=[])` is a Python bug trap).

---

## 8. Module-Level Smoke Tests

Every non-trivial module includes a minimal `if __name__ == "__main__":` self-check. It must:
- Run without GPU if possible (use `device="cpu"`).
- Exercise the main code path.
- Assert at least one result property.
- Print a clear pass message.

```python
if __name__ == "__main__":
    # Smoke test: verify shape prior rejects a wide box for zipper.
    wide_box = {"bbox_xyxy": [100, 100, 300, 160], "score": 0.5}
    garment = [0, 0, 400, 600]
    result = filter_by_shape_priors([wide_box], "zipper", garment)
    assert result == [], f"Expected empty list, got {result}"
    print("part_shape_priors smoke test passed.")
```

Formal tests live in `tests/`. The `__main__` block is a quick sanity check for contributors who run the file directly during development.

---

## Quick Reference Checklist

Before submitting any new file or PR:

- [ ] File has a single coherent responsibility.
- [ ] All tunable parameters come from config files, not hardcoded literals.
- [ ] All public functions have Google-style docstrings with Args and Returns.
- [ ] All function signatures have type hints.
- [ ] Exception handling at every I/O and external-call boundary.
- [ ] No logic duplicated from an existing module.
- [ ] Imports sorted: stdlib → third-party → local.
- [ ] Names follow PEP 8 and the table above.
- [ ] `if __name__ == "__main__":` smoke test included for non-trivial modules.
