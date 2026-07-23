"""
P0 — SubprocessVisionProvider: runs 3.1 pipeline via subprocess and reads outputs.

Design (ponytail: subprocess is laziest way to avoid loading heavy models):
    1. If pipeline output already exists for the image → read directly (fast path).
    2. Otherwise → ``python tools/infer/run_garment_pipeline.py`` via subprocess,
       then read the output JSONs.

Implements :class:`VisionAttributeProvider` interface for QaOrchestrator wiring.
Also exposes ``extract_from_path()`` for direct file-path usage.

Usage::

    from inference.serving.subprocess_vision_provider import SubprocessVisionProvider
    provider = SubprocessVisionProvider(output_root="outputs/full_31x_demo")

    # QaOrchestrator-compatible (VisionAttributeProvider interface):
    result = provider.extract(image="D:/.../000088.jpg")

    # Direct path usage:
    result = provider.extract_from_path("D:/.../000088.jpg")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from inference.serving.schemas import WarningItem, WarningSeverity
from inference.serving.vision_provider import (
    VisionAttributeProvider,
    VisionAttributeResult,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PIPELINE_SCRIPT = _PROJECT_ROOT / "tools" / "infer" / "run_garment_pipeline.py"


class SubprocessVisionProvider(VisionAttributeProvider):
    """Reads (or runs) the PRD 3.1 pipeline and returns structured results.

    Implements :class:`VisionAttributeProvider` for QaOrchestrator wiring.
    """

    def __init__(
        self,
        output_root: str = "outputs/full_31x_demo",
        pipeline_script: Optional[str] = None,
        conda_env: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._output_root = Path(output_root)
        self._pipeline_script = str(pipeline_script or _DEFAULT_PIPELINE_SCRIPT)
        self._conda_env = conda_env

    # ── VisionAttributeProvider interface ──────────────────────────────────

    def extract(
        self,
        *,
        image: Any = None,
        image_url: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        query: Optional[str] = None,
        garment_category: Optional[str] = None,
        regions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        provided_attributes: Optional[Dict[str, Any]] = None,
    ) -> VisionAttributeResult:
        """QaOrchestrator-compatible entry point.

        Accepts *image* as a file path string, or *image_bytes* as raw bytes.
        ``image_url`` is not supported (returns empty result with warning).
        """
        # ponytail: provided_attributes take priority (same as MockVisionAttributeProvider).
        if provided_attributes:
            return VisionAttributeResult(
                attributes={},
                meta={"provider": "subprocess_vision", "reason": "provided_attrs_exist"},
            )

        # Resolve image source.
        image_path: Optional[str] = None
        cleanup_temp = False

        if isinstance(image, str) and image:
            # File path directly.
            image_path = image
        elif isinstance(image, Path):
            image_path = str(image)
        elif image_bytes:
            # Save to temp file.
            try:
                if isinstance(image_bytes, str):
                    # Assume base64-encoded string.
                    raw = base64.b64decode(image_bytes)
                else:
                    raw = image_bytes
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False, dir=self._output_root,
                )
                tmp.write(raw)
                tmp.flush()
                image_path = tmp.name
                cleanup_temp = True
                logger.info("SubprocessVisionProvider: saved temp image %s", image_path)
            except Exception:
                logger.exception("Failed to decode image_bytes")
                return VisionAttributeResult(
                    warnings=[WarningItem(
                        code="image_decode_error", scope="vision",
                        message="Failed to decode image_bytes.",
                        severity=WarningSeverity.warn,
                    )],
                )
        elif image_url:
            # Not supported — would require network access.
            return VisionAttributeResult(
                warnings=[WarningItem(
                    code="image_url_not_supported", scope="vision",
                    message="image_url is not supported by SubprocessVisionProvider.",
                    severity=WarningSeverity.info,
                )],
            )

        if not image_path:
            return VisionAttributeResult(
                warnings=[WarningItem(
                    code="vision_input_missing", scope="vision",
                    message="No image source provided.",
                    severity=WarningSeverity.info,
                )],
            )

        try:
            return self.extract_from_path(image_path)
        finally:
            if cleanup_temp:
                try:
                    os.unlink(image_path)
                except OSError:
                    pass

    # ── Direct path API ────────────────────────────────────────────────────

    def extract_from_path(
        self,
        image_path: str,
        *,
        output_subdir: Optional[str] = None,
        force_rerun: bool = False,
    ) -> VisionAttributeResult:
        """Extract attributes for *image_path* (file path on disk).

        Returns a :class:`VisionAttributeResult` with:
        - attributes: dict[task_name] = {label, score, topk, region}
        - garment_instances: list from YOLO detections
        - sources: evidence paths (crops, JSONs)
        """
        image_path = str(image_path)
        image_stem = Path(image_path).stem

        if output_subdir is None:
            output_subdir = image_stem

        out_dir = self._output_root / output_subdir
        summary_path = out_dir / "pipeline_summary.json"

        # Fast path: output already exists.
        if summary_path.exists() and not force_rerun:
            logger.info("Reading existing pipeline output: %s", out_dir)
            return self._read_existing_output(out_dir, image_stem)

        # Slow path: run pipeline.
        logger.info("Running pipeline for: %s", image_path)
        self._run_pipeline(image_path, out_dir)

        if summary_path.exists():
            return self._read_existing_output(out_dir, image_stem)

        return VisionAttributeResult(
            warnings=[WarningItem(
                code="pipeline_failed", scope="vision",
                message=f"Pipeline did not produce output for {image_stem}.",
                severity=WarningSeverity.error,
            )],
        )

    # ── Internal ────────────────────────────────────────────────────────────

    def _run_pipeline(self, image_path: str, out_dir: Path) -> None:
        """Run the 3.1 pipeline via subprocess (inline Python script)."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # ponytail: use inline Python to call GarmentPipeline.run_image()
        # directly.  This is a subprocess so models are isolated from the
        # serving process.
        script = f'''
import sys
from pathlib import Path
sys.path.insert(0, r"{_PROJECT_ROOT}")
sys.path.insert(0, r"{_PROJECT_ROOT / 'tools' / 'infer'}")
sys.path.insert(0, r"{_PROJECT_ROOT / 'src'}")

from tools.infer.garment_pipeline import GarmentPipeline, GarmentPipelineConfig

config = GarmentPipelineConfig(
    yolo_weights="models/detectors/yolov8n_deepfashion2_13cls_best.pt",
    yolo_device="cuda",
    sam_device="cuda",
    landmark_device="cuda",
    attribute_device="cuda",
    run_landmark_and_crops=True,
    run_attribute_inference=True,
    attribute_topk=3,
    yolo_conf=0.25,
    yolo_iou=0.7,
    save_yolo_vis=True,
    draw_landmark_index=False,
    draw_landmark_name=False,
    save_landmark_visualizations=False,
    region_crop_regions=["collar", "sleeve", "hem", "waist", "pant_leg"],
    use_category_regions=True,
    region_fallback=True,
    masked_crop_background="white",
    masked_crop_transparent=False,
)

pipeline = GarmentPipeline(config)
result = pipeline.run_image(
    image_path=r"{image_path}",
    output_dir=r"{out_dir}",
)
print(f"PIPELINE_OK: garments={{len(result.get('detections', []))}}")
'''

        cmd = [sys.executable, "-c", script]
        logger.info("Running pipeline for %s", Path(image_path).stem)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(_PROJECT_ROOT),
            )
            if result.returncode != 0:
                logger.error("Pipeline failed (rc=%d):\n%s\n%s",
                             result.returncode, result.stdout[-1000:], result.stderr[-2000:])
            else:
                logger.info("Pipeline stdout: %s", result.stdout.strip()[-500:])
        except subprocess.TimeoutExpired:
            logger.error("Pipeline timed out for %s", image_path)
        except Exception:
            logger.exception("Pipeline subprocess failed")

    def _read_existing_output(
        self, out_dir: Path, image_stem: str
    ) -> VisionAttributeResult:
        """Parse pipeline output JSONs into VisionAttributeResult."""
        warnings: List[WarningItem] = []
        attributes: Dict[str, Any] = {}
        garment_instances: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        # 1. Read detections (YOLO).
        det_path = out_dir / "01_yolo" / "detections.json"
        if det_path.exists():
            try:
                det_data = json.loads(det_path.read_text(encoding="utf-8"))
                for img in det_data.get("images", []):
                    for det in img.get("detections", []):
                        inst = {
                            "instance_id": f"{image_stem}__det{det['det_id']}",
                            "category": det.get("coarse_class_name", "unknown"),
                            "fine_class_name": det.get("fine_class_name", ""),
                            "confidence": det.get("confidence"),
                            "bbox": det.get("bbox_xyxy"),
                            "mask_present": True,
                        }
                        garment_instances.append(inst)
                meta["num_garments"] = len(garment_instances)
                meta["garment_classes"] = list(set(
                    inst["category"] for inst in garment_instances
                ))
                sources.append({
                    "type": "detection_json",
                    "path": str(det_path.relative_to(self._output_root)),
                    "id": "yolo_detections",
                })
            except Exception:
                logger.exception("Failed to read detections.json")
                warnings.append(WarningItem(
                    code="detection_parse_error", scope="vision",
                    message="YOLO detections JSON parse failed.",
                    severity=WarningSeverity.warn,
                ))

        # 2. Read attributes (predictions.jsonl).
        attr_path = out_dir / "06_attributes" / "predictions.jsonl"
        if attr_path.exists():
            try:
                for line in attr_path.read_text(encoding="utf-8").strip().split("\n"):
                    if not line.strip():
                        continue
                    pred = json.loads(line)
                    det_id = pred.get("det_id", "")
                    for task_name, task_data in pred.get("attributes", {}).items():
                        key = f"{det_id}__{task_name}"
                        attributes[key] = {
                            "label": task_data.get("label", ""),
                            "score": task_data.get("score"),
                            "topk": task_data.get("topk", []),
                            "task": task_name,
                            "fine_class": pred.get("fine_class_name", ""),
                            "coarse_class": pred.get("coarse_class_name", ""),
                        }
                meta["num_attribute_predictions"] = len(attributes)
                sources.append({
                    "type": "attribute_jsonl",
                    "path": str(attr_path.relative_to(self._output_root)),
                    "id": "attribute_predictions",
                })
            except Exception:
                logger.exception("Failed to read predictions.jsonl")
                warnings.append(WarningItem(
                    code="attribute_parse_error", scope="vision",
                    message="Attribute predictions JSONL parse failed.",
                    severity=WarningSeverity.warn,
                ))

        # 3. Read region crops summary for evidence paths.
        crops_path = out_dir / "04_region_crops" / "region_crops.json"
        collar_crops: List[Dict[str, str]] = []
        if crops_path.exists():
            try:
                crops_data = json.loads(crops_path.read_text(encoding="utf-8"))
                for crop in crops_data.get("crops", []):
                    if crop.get("region") == "collar":
                        collar_crops.append({
                            "crop_path": crop.get("crop_path", ""),
                            "component": crop.get("component", ""),
                            "instance_id": crop.get("instance_id", ""),
                        })
                meta["num_collar_crops"] = len(collar_crops)
            except Exception:
                logger.exception("Failed to read region_crops.json")

        # 4. Read masked crops for evidence.
        masked_path = out_dir / "05_region_masked_crops" / "region_masked_crops.json"
        collar_masked: List[Dict[str, str]] = []
        if masked_path.exists():
            try:
                masked_data = json.loads(masked_path.read_text(encoding="utf-8"))
                for crop in masked_data.get("masked_crops", []):
                    if crop.get("region") == "collar":
                        collar_masked.append({
                            "masked_path": crop.get("masked_crop_path", ""),
                            "component": crop.get("component", ""),
                            "instance_id": crop.get("instance_id", ""),
                        })
            except Exception:
                logger.exception("Failed to read region_masked_crops.json")

        # Add evidence paths to sources.
        for cc in collar_crops:
            sources.append({
                "type": "region_crop",
                "path": cc["crop_path"],
                "region": "collar",
                "component": cc["component"],
                "instance_id": cc["instance_id"],
            })
        for cm in collar_masked:
            sources.append({
                "type": "masked_crop",
                "path": cm["masked_path"],
                "region": "collar",
                "component": cm["component"],
                "instance_id": cm["instance_id"],
            })

        # Add pipeline summary as source.
        summary_path = out_dir / "pipeline_summary.json"
        if summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                meta["pipeline_timing"] = summary_data.get("timing", {})
                meta["pipeline_status"] = summary_data.get("status", "unknown")
            except Exception:
                pass
            sources.append({
                "type": "pipeline_summary_json",
                "path": str(summary_path.relative_to(self._output_root)),
                "id": "pipeline_summary",
            })

        # Build attribute dict keyed by task_name for QaOrchestrator compatibility.
        # Merge all instances' predictions into per-task dicts.
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for key, attr in attributes.items():
            task = attr["task"]
            if task not in by_task:
                by_task[task] = []
            by_task[task].append({
                "value": attr["label"],
                "attribute_confidence": attr["score"],
                "source": "model_prediction",
                "topk": attr["topk"],
                "fine_class": attr["fine_class"],
                "coarse_class": attr["coarse_class"],
                "instance_id": key.split("__")[0] if "__" in key else "",
            })

        # For single-attribute queries, provide the best prediction per task.
        qa_attributes: Dict[str, Any] = {}
        for task, preds in by_task.items():
            best = max(preds, key=lambda p: p["attribute_confidence"] or 0)
            qa_attributes[task] = best

        return VisionAttributeResult(
            attributes=qa_attributes,
            garment_instances=garment_instances,
            sources=sources,
            warnings=warnings,
            meta={
                **meta,
                "vision_backend": "subprocess_3_1_pipeline",
                "output_dir": str(out_dir),
                "by_task_predictions": by_task,
                "collar_crops": collar_crops,
                "collar_masked": collar_masked,
                "raw_attributes": attributes,
            },
        )


# ── Collar QA answer generator ───────────────────────────────────────────────

# ponytail: rich Chinese NL templates for collar QA, with confidence tiers.
_COLLAR_TEMPLATES = {
    "high": (
        "这件{garment_cn}采用 **{label}** 领型设计。"
        "从领口区域裁剪图来看，{extra_context}。"
        "该识别结果置信度为 {conf_pct}。"
    ),
    "medium": (
        "这件{garment_cn}的领型可能为 **{label}**，置信度 {conf_pct}。"
        "同时检测到其他领口相关特征：{related_attrs}。"
        "由于置信度中等，建议结合实物确认。"
    ),
    "low": (
        "领型识别置信度较低（{conf_pct}），系统不能给出确定性结论。"
        "模型 top-3 候选为：{topk_list}。"
        "建议以实物标签或商品详情页为准。"
    ),
    "unavailable": (
        "暂未获取到这件{garment_cn}的领型信息。"
        "可能原因：图像中领口区域不清晰，或该服饰类型不支持领型识别。"
    ),
}

# Collar-related tasks and their Chinese display names.
_COLLAR_TASK_CN = {
    "collar_design": "领型",
    "neckline_design": "领口线",
    "neck_design": "领部设计",
    "lapel_design": "翻领",
}


def generate_collar_qa(
    result: VisionAttributeResult,
    query: str = "这件衣服的领口是什么设计？",
) -> Dict[str, Any]:
    """Generate a rich collar QA answer from pipeline results.

    Returns a dict with: answer, confidence, attribute_label, evidence_crops,
    source_json_paths, garment_info, all_collar_attributes.
    """
    attrs = result.attributes
    by_task = result.meta.get("by_task_predictions", {})
    collar_crops = result.meta.get("collar_crops", [])
    collar_masked = result.meta.get("collar_masked", [])
    garments = result.garment_instances

    # Determine garment type for display.
    garment_cn = "服饰"
    if garments:
        from inference.serving.qa_orchestrator import _CATEGORY_CN
        cats = [inst["category"] for inst in garments if inst.get("category")]
        if cats:
            garment_cn = _CATEGORY_CN.get(cats[0], cats[0])

    # Find collar_design (primary) and related attributes.
    collar_design = attrs.get("collar_design", {})
    collar_label = collar_design.get("value", "")
    collar_score = collar_design.get("attribute_confidence")
    collar_topk = collar_design.get("topk", [])

    # Collect related collar attributes.
    related_attr_strs = []
    all_collar_attrs = {}
    for task in ("collar_design", "neckline_design", "neck_design", "lapel_design"):
        preds = by_task.get(task, [])
        if preds:
            best = max(preds, key=lambda p: p.get("attribute_confidence") or 0)
            all_collar_attrs[task] = best
            cn = _COLLAR_TASK_CN.get(task, task)
            related_attr_strs.append(
                f"{cn}={best['value']}({best['attribute_confidence']:.2f})"
            )

    # Evidence crop paths.
    evidence_crops = []
    for cc in collar_crops[:4]:  # max 4 crop images
        for src in result.sources:
            if src.get("type") == "region_crop" and src.get("instance_id") == cc.get("instance_id"):
                evidence_crops.append(src.get("path", ""))
                break

    # Source JSON paths.
    source_paths = []
    for src in result.sources:
        if src.get("type") in ("attribute_jsonl", "pipeline_summary_json"):
            source_paths.append(src.get("path", ""))

    # ── Generate answer ──────────────────────────────────────────────────
    if not collar_label:
        answer = _COLLAR_TEMPLATES["unavailable"].format(garment_cn=garment_cn)
        confidence = None
    elif collar_score is not None and collar_score >= 0.7:
        extra = _build_extra_context(collar_label)
        answer = _COLLAR_TEMPLATES["high"].format(
            garment_cn=garment_cn, label=collar_label,
            extra_context=extra, conf_pct=f"{collar_score:.0%}",
        )
        confidence = collar_score
    elif collar_score is not None and collar_score >= 0.4:
        answer = _COLLAR_TEMPLATES["medium"].format(
            garment_cn=garment_cn, label=collar_label,
            conf_pct=f"{collar_score:.0%}",
            related_attrs="；".join(related_attr_strs) if related_attr_strs else "无",
        )
        confidence = collar_score
    elif collar_score is not None:
        topk_str = ", ".join(
            f"{t.get('label', '?')}({t.get('score', 0):.0%})"
            for t in (collar_topk or [])[:3]
        )
        answer = _COLLAR_TEMPLATES["low"].format(
            conf_pct=f"{collar_score:.0%}", topk_list=topk_str or "无",
        )
        confidence = collar_score
    else:
        answer = _COLLAR_TEMPLATES["unavailable"].format(garment_cn=garment_cn)
        confidence = None

    return {
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "attribute_label": collar_label,
        "attribute_task": "collar_design",
        "evidence_crops": evidence_crops,
        "source_json_paths": source_paths,
        "garment_info": {
            "garment_cn": garment_cn,
            "num_garments": len(garments),
            "garment_classes": list(set(g.get("category") for g in garments)),
        },
        "all_collar_attributes": {
            task: {
                "label": preds[0]["value"] if preds else "",
                "score": preds[0]["attribute_confidence"] if preds else None,
                "topk": preds[0].get("topk", []) if preds else [],
            }
            for task, preds in by_task.items()
            if task in _COLLAR_TASK_CN
        },
        "warnings": [w.message for w in result.warnings],
        "meta": {
            "output_dir": result.meta.get("output_dir", ""),
            "pipeline_timing": result.meta.get("pipeline_timing", {}),
        },
    }


def _build_extra_context(label: str) -> str:
    """Provide a brief design insight for a collar type (ponytail: small lookup)."""
    _COLLAR_CONTEXT: Dict[str, str] = {
        "Shirt Collar": "衬衫领线条利落，是经典的正装和商务休闲选择",
        "Puritan Collar": "清教徒领型较为简洁，领尖通常较短，适合简约风格",
        "Peter Pan": "彼得潘领（娃娃领）呈圆形平贴，常见于甜美或复古风格的服饰",
        "V Neckline": "V领设计从领口向下延伸，视觉上可拉长颈部线条",
        "Turtle Neck": "高领包覆颈部，保暖且具有时尚感，常见于秋冬服饰",
        "Invisible": "隐形式领部设计，领口简洁无多余装饰",
        "Stand Collar": "立领向上竖起，常见于中式服饰和现代简约设计",
        "Ruffle Semi-High Collar": "荷叶边半高领设计，增添柔美气质",
        "Low Turtle Neck": "低领座设计，比传统高领更轻便舒适",
    }
    return _COLLAR_CONTEXT.get(label, f"{label}是一种常见的领型设计，具体风格取决于服饰整体搭配")


# ── Convenience function ────────────────────────────────────────────────────


def demo_collar_qa_for_image(
    image_id: str,
    output_root: str = "outputs/full_31x_demo",
    query: str = "这件衣服的领口是什么设计？",
) -> Dict[str, Any]:
    """One-call collar QA for a pre-processed image ID."""
    provider = SubprocessVisionProvider(output_root=output_root)
    # The image path is derived from the pipeline output data.
    out_dir = Path(output_root) / image_id
    summary_path = out_dir / "pipeline_summary.json"
    if not summary_path.exists():
        return {"error": f"No pipeline output for {image_id} at {out_dir}"}

    # Read the source image path from the pipeline summary.
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    image_path = summary.get("source", "")
    if not image_path:
        image_path = str(out_dir)  # fallback

    result = provider.extract_from_path(image_path, output_subdir=image_id)
    return generate_collar_qa(result, query=query)
