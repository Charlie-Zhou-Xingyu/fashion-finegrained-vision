"""
P0 — QaOrchestrator CLI wrapper with SubprocessVisionProvider.

Runs the full QA chain: IntentClassifier → SubprocessVisionProvider (3.1
pipeline) → AttributeService → template answer generation.

Supports two modes:
    --fast  (default): reads existing pipeline outputs (fast path)
    --slow: re-runs the pipeline via subprocess (slow path)

Usage::

    # Fast path (pre-processed image)
    python scripts/run_qa_orchestrator.py --image-id 000088 \\
        --query "这件衣服的领口是什么设计？"

    # Slow path (new image — runs pipeline from scratch)
    python scripts/run_qa_orchestrator.py \\
        --image D:/Aliintern/fashion-ai-data/deepfashion2/validation/image/000001.jpg \\
        --slow \\
        --query "这件衣服的领口是什么设计？"

    # Output JSON to file
    python scripts/run_qa_orchestrator.py --image-id 000088 \\
        --query "这件衣服的领口是什么设计？" \\
        --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P0 QaOrchestrator — Collar QA with SubprocessVisionProvider",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-id", type=str, help="Pre-processed image ID (fast path)")
    src.add_argument("--image", type=str, help="Image file path (slow path)")

    p.add_argument("--slow", action="store_true", help="Force subprocess re-run")
    p.add_argument(
        "--query", type=str,
        default="这件衣服的领口是什么设计？",
        help="User query (Chinese)",
    )
    p.add_argument(
        "--output-root", type=str, default="outputs/full_31x_demo",
        help="Pipeline output root directory",
    )
    p.add_argument("--output", "-o", type=str, help="Write result JSON to file")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return p.parse_args()


def run_fast_path(
    image_id: str,
    query: str,
    output_root: str,
) -> Dict[str, Any]:
    """Use existing pipeline output (fast path)."""
    from inference.serving.subprocess_vision_provider import (
        SubprocessVisionProvider,
        generate_collar_qa,
    )

    out_dir = Path(output_root) / image_id
    summary_path = out_dir / "pipeline_summary.json"
    if not summary_path.exists():
        return {"error": f"No pipeline output at {out_dir}", "path": "fast"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_image = summary.get("source", str(out_dir))

    provider = SubprocessVisionProvider(output_root=output_root)
    vision_result = provider.extract_from_path(source_image, output_subdir=image_id)
    qa = generate_collar_qa(vision_result, query=query)
    qa["path"] = "fast"
    qa["image_id"] = image_id
    qa["source_image"] = source_image
    qa["pipeline_rerun"] = False
    return qa


def run_slow_path(
    image_path: str,
    query: str,
    output_root: str,
) -> Dict[str, Any]:
    """Run the full 3.1 pipeline via subprocess (slow path)."""
    from inference.serving.subprocess_vision_provider import (
        SubprocessVisionProvider,
        generate_collar_qa,
    )

    image_stem = Path(image_path).stem
    print(f"[SLOW PATH] Running pipeline for {image_stem}...")
    t0 = time.time()

    provider = SubprocessVisionProvider(output_root=output_root)
    vision_result = provider.extract_from_path(
        image_path, output_subdir=image_stem, force_rerun=True,
    )

    elapsed = time.time() - t0
    print(f"[SLOW PATH] Pipeline completed in {elapsed:.1f}s")

    qa = generate_collar_qa(vision_result, query=query)
    qa["path"] = "slow"
    qa["image_id"] = image_stem
    qa["source_image"] = image_path
    qa["pipeline_rerun"] = True
    qa["pipeline_wall_time_s"] = round(elapsed, 1)
    return qa


def run_orchestrator_chain(
    image_path: str,
    query: str,
    vision_result: Any,
) -> Dict[str, Any]:
    """Demonstrate the full QaOrchestrator chain.

    This shows how SubprocessVisionProvider plugs into QaOrchestrator.answer().
    """
    from inference.serving.intent_classifier import RuleIntentClassifier
    from inference.serving.attribute_service import AttributeService
    from inference.serving.qa_orchestrator import QaOrchestrator, QAOrchestratorResult

    # Build orchestrator with a real vision provider (not mock).
    intent_cls = RuleIntentClassifier()
    attr_svc = AttributeService()
    from inference.serving.rag_service import RagService
    rag_svc = RagService()

    # ponytail: create a thin wrapper that returns our pre-computed result.
    class _CachedVisionProvider:
        def __init__(self, result):
            self._result = result
        def extract(self, **kwargs):
            return self._result

    cached_vp = _CachedVisionProvider(vision_result)
    orchestrator = QaOrchestrator(
        intent_classifier=intent_cls,
        attribute_service=attr_svc,
        rag_service=rag_svc,
        vision_provider=cached_vp,
    )

    # Call the orchestrator with image path as "image".
    orchestrator_result = orchestrator.answer(
        query=query,
        image=str(image_path),
    )

    return {
        "orchestrator_route": orchestrator_result.meta.get("route", "unknown"),
        "orchestrator_answer_type": orchestrator_result.answer_type,
        "orchestrator_answer": orchestrator_result.answer,
        "orchestrator_confidence": orchestrator_result.answer_confidence,
        "orchestrator_sources": orchestrator_result.sources[:3],
        "orchestrator_used_tools": orchestrator_result.used_tools,
        "intent": orchestrator_result.intent,
    }


def main() -> None:
    args = parse_args()

    if args.image_id and not args.slow:
        # ── Fast path ──────────────────────────────────────────────────
        print(f"[FAST PATH] Reading existing output for {args.image_id}")
        qa_result = run_fast_path(args.image_id, args.query, args.output_root)
    elif args.image:
        # ── Slow path ─────────────────────────────────────────────────
        qa_result = run_slow_path(args.image, args.query, args.output_root)
    else:
        print("ERROR: --image-id requires --slow for re-run, or use --image for new image")
        sys.exit(1)

    if "error" in qa_result:
        print(f"ERROR: {qa_result['error']}")
        sys.exit(1)

    # ── Orchestrator chain demo ────────────────────────────────────────
    print("\n--- QaOrchestrator Chain ---")
    try:
        from inference.serving.subprocess_vision_provider import SubprocessVisionProvider
        provider = SubprocessVisionProvider(output_root=args.output_root)
        source_img = qa_result.get("source_image", "")
        if source_img and Path(source_img).exists():
            image_id = qa_result.get("image_id", "")
            vision_result = provider.extract_from_path(
                source_img, output_subdir=image_id,
            )
            orch_result = run_orchestrator_chain(
                source_img, args.query, vision_result,
            )
            qa_result["orchestrator_chain"] = orch_result
            print(f"  Route: {orch_result['orchestrator_route']}")
            print(f"  Intent: {orch_result['intent']}")
            print(f"  Tools: {orch_result['orchestrator_used_tools']}")
            print(f"  Answer: {orch_result['orchestrator_answer'][:120]}...")
        else:
            qa_result["orchestrator_chain"] = {"error": "source_image not found"}
    except Exception as e:
        qa_result["orchestrator_chain"] = {"error": str(e)}
        print(f"  Orchestrator chain error: {e}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Path:       {qa_result.get('path', '?')}")
    print(f"Image:      {qa_result.get('image_id', '?')}")
    print(f"Label:      {qa_result.get('attribute_label', '?')}")
    print(f"Confidence: {qa_result.get('confidence', '?')}")
    print(f"Crops:      {len(qa_result.get('evidence_crops', []))}")
    if qa_result.get("pipeline_wall_time_s"):
        print(f"Wall time:  {qa_result['pipeline_wall_time_s']}s (pipeline re-run)")
    print(f"{'='*60}")

    # ── Output ─────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(qa_result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nResult written to: {output_path}")

    # Always dump to stdout when no output file specified.
    if not args.output or args.verbose:
        print(f"\nFull Answer:\n{qa_result['answer']}")


if __name__ == "__main__":
    main()
