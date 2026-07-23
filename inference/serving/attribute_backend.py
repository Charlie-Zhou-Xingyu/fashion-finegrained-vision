"""
P1.4f — 3.1.3 Fine-Grained Attribute Extraction Backend.

Serving-safe adapter around the existing FashionAI ResNet18 classifiers.
Each task predicts one attribute from a garment/region crop image.

Backends:
    - ``DisabledAttributeBackend`` — always returns placeholder (default).
    - ``FashionAttributeBackend`` — loads 3.1.3 checkpoints and runs inference.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Task definitions (mirrors attribute_inference.yaml) ─────────────────────

# (task_name, checkpoint_rel_path, label_map_rel_path, region_filter)
_ATTRIBUTE_TASKS: List[Dict[str, str]] = [
    {"task": "neckline_design", "checkpoint": "outputs/p2_neckline_design_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_neckline_design.json", "region": "collar"},
    {"task": "collar_design", "checkpoint": "outputs/p2_collar_design_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_collar_design.json", "region": "collar"},
    {"task": "neck_design", "checkpoint": "outputs/p2_neck_design_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_neck_design.json", "region": "collar"},
    {"task": "lapel_design", "checkpoint": "outputs/p2_lapel_design_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_lapel_design.json", "region": "collar"},
    {"task": "sleeve_length", "checkpoint": "outputs/p2_sleeve_length_multiview_v2_pipeline_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_sleeve_length.json", "region": "all"},
    {"task": "coat_length", "checkpoint": "outputs/p2_coat_length_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_coat_length.json", "region": "all"},
    {"task": "pant_length", "checkpoint": "outputs/p2_pant_length_multiview_v2_pipeline_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_pant_length.json", "region": "all"},
    {"task": "skirt_length", "checkpoint": "outputs/p2_skirt_length_multiview_v2_pipeline_resnet18_seed2/best.pt",
     "label_map": "data/fashionai_attribute_index/label_map_skirt_length.json", "region": "all"},
]

# Map region part_type → relevant attribute task(s).
_PART_TO_ATTR_TASKS: Dict[str, List[str]] = {
    "neckline": ["neckline_design", "neck_design"],
    "collar": ["collar_design", "neckline_design", "neck_design", "lapel_design"],
    "lapel": ["lapel_design"],
    "sleeve": ["sleeve_length"],
    "cuff": ["sleeve_length"],
}

# ── Normalised output schema ────────────────────────────────────────────────


def _normalize_attribute_result(
    task_name: str,
    raw: Dict[str, Any],
    region_id: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert raw classifier output to a safe attribute dict."""
    return {
        "task": task_name,
        "value": raw.get("label"),
        "attribute_confidence": raw.get("score"),
        "topk": raw.get("topk", [])[:3],
        "region_id": region_id,
        "instance_id": instance_id,
    }


# ── Backend interface ───────────────────────────────────────────────────────


class AttributeExtractionBackend(ABC):
    """Interface for 3.1.3 fine-grained attribute extraction."""

    @abstractmethod
    def extract_attributes(
        self,
        image: Any,
        region_bbox: Optional[List[float]] = None,
        task_names: Optional[List[str]] = None,
        region_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run attribute classifiers on *image* (or a crop at *region_bbox*).

        Args:
            image: BGR numpy array (full image).
            region_bbox: [x1,y1,x2,y2] crop region, or None for full image.
            task_names: Specific task names to run, or None for all relevant.
            region_id: Region identifier for provenance.
            instance_id: Garment instance identifier for provenance.

        Returns:
            List of attribute result dicts. Never raises.
        """
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def enabled(self) -> bool: ...


# ── Disabled backend ────────────────────────────────────────────────────────


class DisabledAttributeBackend(AttributeExtractionBackend):
    """Always returns empty — safe default when 3.1.3 is not enabled."""

    @property
    def backend_name(self) -> str:
        return "disabled"

    @property
    def enabled(self) -> bool:
        return False

    def extract_attributes(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return []


# ── Real FashionAI attribute backend ────────────────────────────────────────


class FashionAttributeBackend(AttributeExtractionBackend):
    """Wraps existing 3.1.3 FashionAI ResNet18 classifiers.

    Loads checkpoints lazily on first call.  When checkpoints are missing,
    returns empty results with structured warnings — does not crash.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._tasks: Dict[str, Any] = {}   # task_name -> LoadedTask
        self._id_to_label: Dict[str, Dict[int, str]] = {}
        self._loaded = False
        self._load_error: Optional[str] = None
        self._missing_checkpoints: List[str] = []

    @property
    def backend_name(self) -> str:
        return "fashionai_313"

    @property
    def enabled(self) -> bool:
        return self._load_error is None

    # -- lazy load -----------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_error is not None:
            return False

        import sys, json as _json
        root = str(_PROJECT_ROOT)
        src_root = str(_PROJECT_ROOT / "src")
        for p in (root, src_root):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Check heavy deps.
        try:
            from importlib import util as _iu
            for mod in ("torch", "torchvision", "PIL"):
                if _iu.find_spec(mod) is None:
                    self._load_error = f"Missing dependency: {mod}"
                    return False
        except Exception:
            pass

        import torch

        try:
            from models.attribute_classifier import build_attribute_classifier
        except ImportError:
            self._load_error = "Cannot import models.attribute_classifier"
            return False

        device = torch.device("cuda" if self._device == "cuda"
                              and torch.cuda.is_available() else "cpu")

        for task_def in _ATTRIBUTE_TASKS:
            task_name = task_def["task"]
            ckpt_path = _PROJECT_ROOT / task_def["checkpoint"]
            label_path = _PROJECT_ROOT / task_def["label_map"]

            if not ckpt_path.exists():
                self._missing_checkpoints.append(str(task_def["checkpoint"]))
                continue

            try:
                # Load label map.
                with open(label_path, "r", encoding="utf-8") as f:
                    label_data = _json.load(f)
                if isinstance(label_data, dict) and "id_to_label" in label_data:
                    id_to_label = {int(k): v for k, v in label_data["id_to_label"].items()}
                elif isinstance(label_data, dict):
                    id_to_label = {int(k): v for k, v in label_data.items()}
                else:
                    id_to_label = {i: str(v) for i, v in enumerate(label_data)}

                # Load model.
                num_classes = len(id_to_label)
                model = build_attribute_classifier(arch="resnet18", num_classes=num_classes)
                state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state, strict=False)
                model.to(device).eval()

                # Transform (ImageNet normalisation).
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

                self._tasks[task_name] = {
                    "model": model, "transform": transform,
                    "id_to_label": id_to_label, "device": device,
                }
                self._id_to_label[task_name] = id_to_label
                logger.info("FashionAttributeBackend: loaded %s (%d classes)", task_name, num_classes)
            except Exception:
                logger.exception("FashionAttributeBackend: failed to load task %s", task_name)
                continue

        if self._missing_checkpoints:
            logger.warning(
                "FashionAttributeBackend: %d/%d checkpoints missing: %s",
                len(self._missing_checkpoints), len(_ATTRIBUTE_TASKS),
                ", ".join(self._missing_checkpoints[:3]),
            )

        self._loaded = True
        logger.info("FashionAttributeBackend: %d/%d tasks loaded",
                     len(self._tasks), len(_ATTRIBUTE_TASKS))
        return len(self._tasks) > 0

    # -- crop helper ---------------------------------------------------------

    @staticmethod
    def _crop_region(image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Crop *image* to *bbox*, returning an RGB numpy array or None."""
        if bbox is None or len(bbox) != 4:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        return crop[..., ::-1]  # BGR → RGB

    # -- single-task inference -----------------------------------------------

    def _run_task(
        self, task_name: str, image_rgb: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Run one classifier on an RGB image crop. Returns {label, score, topk} or None."""
        task = self._tasks.get(task_name)
        if task is None:
            return None

        import torch
        from PIL import Image
        pil_img = Image.fromarray(image_rgb)
        x = task["transform"](pil_img).unsqueeze(0).to(task["device"])

        with torch.no_grad():
            logits = task["model"](x)
            probs = torch.softmax(logits, dim=1)[0]

        k = min(3, probs.numel())
        confs, ids = torch.topk(probs, k=k)
        id_to_label = task["id_to_label"]

        return {
            "label": id_to_label.get(int(ids[0]), f"cls_{int(ids[0])}"),
            "score": round(float(confs[0]), 4),
            "topk": [
                {"label": id_to_label.get(int(ids[i]), f"cls_{int(ids[i])}"),
                 "score": round(float(confs[i]), 4)}
                for i in range(k)
            ],
        }

    # -- public API ----------------------------------------------------------

    def extract_attributes(
        self,
        image: Any,
        region_bbox: Optional[List[float]] = None,
        task_names: Optional[List[str]] = None,
        region_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run 3.1.3 classifiers and return normalised results.

        If *region_bbox* is provided, only that crop is used.  Otherwise the
        full image is used (for garment-level tasks like sleeve_length).
        """
        if not self._ensure_loaded():
            return []

        # Determine which tasks to run.
        if task_names:
            tasks_to_run = [t for t in task_names if t in self._tasks]
        else:
            tasks_to_run = list(self._tasks.keys())

        if not tasks_to_run:
            return []

        # Crop if needed.
        if region_bbox is not None:
            crop = self._crop_region(image, region_bbox)
            if crop is None:
                return []
            infer_img = crop
        else:
            infer_img = image[..., ::-1] if image.shape[-1] == 3 else image  # BGR→RGB

        results: List[Dict[str, Any]] = []
        for task_name in tasks_to_run:
            raw = self._run_task(task_name, infer_img)
            if raw is None:
                continue
            results.append(_normalize_attribute_result(
                task_name, raw, region_id=region_id, instance_id=instance_id,
            ))

        return results


# ── Factory & singleton ─────────────────────────────────────────────────────


def build_attribute_backend(name: str = "disabled", device: str = "cpu") -> AttributeExtractionBackend:
    name = name.strip().lower()
    if name in ("fashionai", "313", "real"):
        return FashionAttributeBackend(device=device)
    return DisabledAttributeBackend()


_attr_backend: Optional[AttributeExtractionBackend] = None


def _resolve_attribute_settings() -> Dict[str, Any]:
    import os
    defaults = {"backend": "disabled", "enable_real": False, "device": "cpu"}
    try:
        from inference.serving.deps import get_config
        cfg = (get_config().get("attribute_backend") or {})
        settings = dict(defaults)
        settings["backend"] = str(cfg.get("backend", "disabled"))
        settings["enable_real"] = bool(cfg.get("enable_real", False))
        settings["device"] = str(cfg.get("device", "cpu"))
        for key in ("backend", "enable_real", "device"):
            env = os.getenv(f"VISION_ATTR_{key.upper()}")
            if env is not None:
                if key == "enable_real":
                    settings[key] = env.strip().lower() in ("1", "true", "yes")
                else:
                    settings[key] = env.strip().lower()
        return settings
    except Exception:
        return defaults


def get_attribute_backend() -> AttributeExtractionBackend:
    global _attr_backend
    if _attr_backend is None:
        settings = _resolve_attribute_settings()
        if not settings["enable_real"]:
            _attr_backend = DisabledAttributeBackend()
        else:
            _attr_backend = build_attribute_backend(settings["backend"], settings["device"])
        logger.info("Attribute backend: %s (enabled=%s)", _attr_backend.backend_name, _attr_backend.enabled)
    return _attr_backend


def reset_attribute_backend() -> None:
    global _attr_backend
    _attr_backend = None
