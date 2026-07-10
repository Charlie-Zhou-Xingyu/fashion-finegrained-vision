"""
Extract the raw 14x14 confusion matrix from a YOLO validation run and save
it as JSON compatible with ``eval_13cls_confusion_as_5cls.py`` and
``aggregate_yolo_val_to_5cls.py``.

YOLO validation generates a confusion matrix internally but does not save
the raw counts to disk.  This script runs ``model.val()`` via the ultralytics
Python API, intercepts the confusion matrix via a callback, and writes it to
a JSON file.

Usage
-----
::

    python tools/eval/extract_yolo_confusion_matrix.py \\
        --weights runs/detect/yolov8n_df2_13cls_balanced/weights/best.pt \\
        --data data/processed/deepfashion2_yolo_13cls/deepfashion2_13cls_balanced.yaml \\
        --out outputs/confusion_matrix_14x14.json \\
        --imgsz 640 \\
        --device 0

The output JSON format is::

    {
      "labels": ["short sleeve top", "long sleeve top", ..., "background"],
      "matrix": [[...], ...]   // shape (14, 14), rows=predicted, cols=GT
    }

Pass the output path to ``--confusion-json`` in ``aggregate_yolo_val_to_5cls.py``
or to ``--matrix-json`` in ``eval_13cls_confusion_as_5cls.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.eval.category_mapping import load_category_mapping  # noqa: E402

_DEFAULT_MAPPING_YAML = _PROJECT_ROOT / "configs" / "category_mapping.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO val via Python API and save the 14x14 confusion matrix as JSON."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to YOLO model weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to YOLO data YAML used during validation.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the confusion matrix JSON file.",
    )
    parser.add_argument(
        "--mapping-yaml",
        type=Path,
        default=_DEFAULT_MAPPING_YAML,
        help=(
            "Category mapping YAML (used to embed class labels in the JSON). "
            f"Default: {_DEFAULT_MAPPING_YAML}"
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device string passed to YOLO val, e.g. '0' or 'cpu' (default: '0').",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Validation batch size (default: 32).",
    )
    return parser.parse_args()


def _extract_confusion_matrix(validator: Any) -> np.ndarray | None:
    """Try to extract the confusion matrix numpy array from a validator object.

    Ultralytics stores the confusion matrix on the validator as
    ``validator.confusion_matrix.matrix``.  This helper probes multiple known
    attribute paths to remain robust across minor ultralytics version changes.

    Args:
        validator: The ultralytics DetectionValidator object passed to callbacks.

    Returns:
        Numpy array of shape (nc+1, nc+1) with raw counts, or None if not found.
    """
    # Primary path: DetectionValidator.confusion_matrix.matrix
    cm_obj = getattr(validator, "confusion_matrix", None)
    if cm_obj is not None:
        matrix = getattr(cm_obj, "matrix", None)
        if matrix is not None and isinstance(matrix, np.ndarray):
            return matrix.copy()

    # Secondary path: some versions nest it differently
    for attr_chain in ("validator.confusion_matrix.matrix",):
        obj: Any = validator
        found = True
        for attr in attr_chain.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                found = False
                break
        if found and isinstance(obj, np.ndarray):
            return obj.copy()

    return None


def run_val_and_extract(
    weights: Path,
    data: Path,
    imgsz: int,
    device: str,
    batch: int,
) -> np.ndarray:
    """Run YOLO val and return the confusion matrix.

    Args:
        weights: Path to model weights.
        data: Path to YOLO data YAML.
        imgsz: Inference image size.
        device: Device string.
        batch: Validation batch size.

    Returns:
        Confusion matrix as a numpy array of shape (nc+1, nc+1).

    Raises:
        ImportError: If ultralytics is not installed.
        RuntimeError: If the confusion matrix could not be captured.
        FileNotFoundError: If weights or data paths do not exist.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required.  Install it with: pip install ultralytics"
        ) from exc

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Data YAML not found: {data}")

    model = YOLO(str(weights))

    _captured: list[np.ndarray] = []

    def _on_val_end(validator: Any) -> None:
        """Callback to capture confusion matrix after validation ends."""
        matrix = _extract_confusion_matrix(validator)
        if matrix is not None:
            _captured.append(matrix)

    model.add_callback("on_val_end", _on_val_end)

    print(f"Running YOLO val: weights={weights}  data={data}  imgsz={imgsz}")
    model.val(
        data=str(data),
        imgsz=imgsz,
        device=device,
        batch=batch,
        plots=True,
        verbose=True,
    )

    if not _captured:
        raise RuntimeError(
            "Confusion matrix was not captured via the 'on_val_end' callback.  "
            "This may indicate an incompatible ultralytics version.  "
            "Check that 'confusion_matrix' is an attribute of the validator object."
        )

    return _captured[0]


def build_label_list(mapping_yaml: Path, nc: int) -> list[str]:
    """Build a list of class labels including the background label.

    Args:
        mapping_yaml: Path to category mapping YAML.
        nc: Number of foreground classes (13 for DeepFashion2).

    Returns:
        List of class name strings with 'background' appended as the last entry.
    """
    mapping = load_category_mapping(mapping_yaml)
    labels = [mapping.deepfashion2_13cls[i] for i in range(nc)]
    labels.append("background")
    return labels


def save_confusion_matrix_json(
    matrix: np.ndarray,
    labels: list[str],
    out_path: Path,
) -> None:
    """Save confusion matrix as a JSON file.

    Args:
        matrix: Confusion matrix array of shape (nc+1, nc+1).
        labels: Class label strings (length nc+1, last entry is background).
        out_path: Output JSON path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "labels": labels,
        "matrix": matrix.astype(int).tolist(),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Run YOLO val, capture confusion matrix, and save as JSON."""
    args = parse_args()

    matrix = run_val_and_extract(
        weights=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
    )

    nc = matrix.shape[0] - 1  # last row/col is background
    print(f"Captured confusion matrix: shape={matrix.shape}, nc={nc}")

    labels = build_label_list(args.mapping_yaml, nc)
    save_confusion_matrix_json(matrix, labels, args.out)

    print(f"Saved confusion matrix JSON: {args.out}")
    print(
        "Run aggregate_yolo_val_to_5cls.py with "
        f"--confusion-json {args.out} to include a 5x5 matrix in the report."
    )


if __name__ == "__main__":
    main()
