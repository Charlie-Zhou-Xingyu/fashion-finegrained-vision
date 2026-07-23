"""
P0 — Collar QA Golden Path Demo.

Tests end-to-end collar QA on 3 pre-processed images, outputs structured JSON
results for integration into the product demo HTML.

Usage::

    conda activate fashion-demo2
    python scripts/demo_collar_qa.py

Outputs:
    outputs/full_31x_demo/collar_qa_results.json  — structured QA results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inference.serving.subprocess_vision_provider import (
    SubprocessVisionProvider,
    generate_collar_qa,
)

# ── Demo configuration ──────────────────────────────────────────────────────

OUTPUT_ROOT = "outputs/full_31x_demo"
RESULT_FILE = f"{OUTPUT_ROOT}/collar_qa_results.json"

# 3 test images: diverse garment types with collar attributes.
# 000088: 2x short sleeve top (collar_design scores 0.88 / 0.75)
# 000579: short sleeve top + skirt (collar_design score 0.48)
# 001574: 5 garments, collar_design scores 0.78 / 0.74 / 0.37 / 0.53
DEMO_IMAGES = ["000088", "000579", "001574"]

QUERY = "这件衣服的领口是什么设计？"


def main() -> None:
    provider = SubprocessVisionProvider(output_root=OUTPUT_ROOT)

    results = {
        "title": "Collar QA Golden Path — Demo Results",
        "query": QUERY,
        "pipeline": "3.1.1 YOLO → SAM-HQ → Landmark → 3.1.3 Attribute",
        "provider": "SubprocessVisionProvider (reads existing pipeline outputs)",
        "answer_generator": "Template-based NL generation with confidence tiers",
        "images": {},
    }

    for image_id in DEMO_IMAGES:
        print(f"\n{'='*60}")
        print(f"Processing: {image_id}")
        print(f"{'='*60}")

        out_dir = Path(OUTPUT_ROOT) / image_id
        summary_path = out_dir / "pipeline_summary.json"

        if not summary_path.exists():
            results["images"][image_id] = {
                "error": f"No pipeline output at {out_dir}",
                "status": "SKIPPED",
            }
            print(f"  SKIPPED: no pipeline output at {out_dir}")
            continue

        # Read source image path from pipeline summary.
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        image_path = summary.get("source", str(out_dir))

        # Extract vision data.
        vision_result = provider.extract_from_path(image_path, output_subdir=image_id)

        # Generate collar QA.
        qa_result = generate_collar_qa(vision_result, query=QUERY)

        # Trim for output (remove large internal structures).
        qa_result.pop("all_collar_attributes", None)

        results["images"][image_id] = {
            "status": "success",
            "source_image": image_path,
            "num_garments": vision_result.meta.get("num_garments", 0),
            "garment_classes": vision_result.meta.get("garment_classes", []),
            "pipeline_timing_s": vision_result.meta.get(
                "pipeline_timing", {}
            ).get("total_seconds", "N/A"),
            "qa": {
                "answer": qa_result["answer"],
                "confidence": qa_result["confidence"],
                "attribute_label": qa_result["attribute_label"],
                "evidence_crops": qa_result["evidence_crops"],
                "source_json_paths": qa_result["source_json_paths"],
                "garment_info": qa_result["garment_info"],
                "warnings": qa_result["warnings"],
            },
        }

        print(f"  Garments: {results['images'][image_id]['num_garments']}")
        print(f"  Collar label: {qa_result['attribute_label']}")
        print(f"  Confidence: {qa_result['confidence']}")
        print(f"  Answer: {qa_result['answer'][:120]}...")

    # Write results.
    output_path = Path(RESULT_FILE)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n{'='*60}")
    print(f"Results written to: {output_path}")
    print(f"{'='*60}")

    # Summary.
    success = sum(
        1 for v in results["images"].values() if v.get("status") == "success"
    )
    print(f"\nSummary: {success}/{len(DEMO_IMAGES)} images processed successfully.")
    for image_id, data in results["images"].items():
        if data.get("status") == "success":
            qa = data["qa"]
            print(f"  {image_id}: label='{qa['attribute_label']}' "
                  f"conf={qa['confidence']} crops={len(qa['evidence_crops'])}")


if __name__ == "__main__":
    main()
