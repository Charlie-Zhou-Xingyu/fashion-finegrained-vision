#!/usr/bin/env python
"""
P1.4b/c — Smoke test for real 3.1.2 region backend.

Two modes:
  backend-direct — calls FashionpediaRegionBackend directly.
  service-qa     — sends request to running /v1/mm/qa endpoint.

Usage (backend-direct)::

    VISION_REGION_BACKEND=fashionpedia VISION_REGION_ENABLE_REAL=true \\
    python scripts/smoke_test_312_region_backend.py \\
        --image path/to/image.jpg \\
        --query "领口在哪里？" \\
        --mode backend-direct

Usage (service-qa)::

    python scripts/smoke_test_312_region_backend.py \\
        --image path/to/image.jpg \\
        --query "领口在哪里？" \\
        --mode service-qa \\
        --url http://127.0.0.1:8000/v1/mm/qa

PowerShell::

    $env:VISION_REGION_BACKEND="fashionpedia"
    $env:VISION_REGION_ENABLE_REAL="true"
    $env:VISION_REGION_DEVICE="cpu"
    python scripts/smoke_test_312_region_backend.py --image test.jpg --query "领口在哪里？"

Does NOT run in CI — requires real model weights.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Forbidden response keys (safety check).
_FORBIDDEN = frozenset({
    "mask_bitmap", "mask_data", "mask_path",
    "crop_path", "crop_data", "crop_bytes",
    "temp_path", "file_path", "local_path",
    "tensor", "checkpoint", "raw_output",
})


def _check_no_leak(data: dict, label: str) -> list[str]:
    """Scan payload for forbidden keys. Returns list of violations."""
    payload = json.dumps(data, ensure_ascii=False)
    violations = []
    for key in _FORBIDDEN:
        if key in payload:
            violations.append(key)
    for path_indicator in ("D:\\\\", "/tmp/", "\\temp", "outputs/", "checkpoints/"):
        if path_indicator in payload:
            violations.append(f"path_like:{path_indicator}")
    return violations


def _print_safe_json(data: dict) -> None:
    """Print JSON without risk of mojibake from terminal encoding."""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run_backend_direct(args) -> int:
    """Mode A: call FashionpediaRegionBackend directly."""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: cannot decode image: {args.image}", file=sys.stderr)
        return 1
    print(f"Image: {image_path}  shape={img.shape}  dtype={img.dtype}")

    from inference.serving.region_query_mapper import extract_requested_region_part
    requested_part = args.part or extract_requested_region_part(args.query)
    print(f"Query: {args.query!r}  ->  requested_part: {requested_part!r}")

    from inference.serving.region_backend import build_region_backend, reset_region_backend
    reset_region_backend()

    # Override backend via --backend flag or env.
    backend_name = args.backend or "fashionpedia"
    device = args.device or "cpu"
    print(f"Backend config: name={backend_name} device={device}")

    backend = build_region_backend(
        backend_name,
        model_path=args.fp_model or None,
        device=device,
        confidence_threshold=args.conf,
    )
    print(f"Backend: {backend.backend_name}  enabled: {backend.enabled}")

    if not backend.enabled:
        print("ERROR: region backend not enabled. Set VISION_REGION_BACKEND + VISION_REGION_ENABLE_REAL.",
              file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    regions = backend.locate_regions(image=img, query=args.query, requested_part=requested_part)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    output = {
        "query": args.query,
        "requested_part": requested_part,
        "backend": backend.backend_name,
        "elapsed_ms": elapsed_ms,
        "num_regions": len(regions),
        "localized_regions": regions,
    }
    print(f"\n--- Result ({elapsed_ms} ms, {len(regions)} regions) ---")
    _print_safe_json(output)

    violations = _check_no_leak(output, "backend-direct")
    if violations:
        print(f"\nSAFETY VIOLATIONS: {violations}", file=sys.stderr)
        return 1
    print("\nSafety check: OK (no forbidden fields)")
    return 0


def run_service_qa(args) -> int:
    """Mode B: call running /v1/mm/qa endpoint."""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    print(f"Image: {image_path}  ({len(image_bytes)} bytes, base64: {len(image_b64)} chars)")

    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' library required for service-qa mode. pip install requests",
              file=sys.stderr)
        return 1

    url = args.url or "http://127.0.0.1:8000/v1/mm/qa"
    payload = {"query": args.query, "image_bytes": image_b64}

    print(f"POST {url}")
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=60)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: cannot connect to {url}. Is the service running?", file=sys.stderr)
        print("Start with:  uvicorn inference.serving.app:app --reload", file=sys.stderr)
        return 1

    data = resp.json()
    print(f"Status: {resp.status_code}  ({elapsed_ms} ms)")
    print(f"Response status: {data.get('status')}")

    qa_data = data.get("data", {})
    print(f"Answer: {qa_data.get('answer', 'N/A')}")
    print(f"Answer type: {qa_data.get('answer_type', 'N/A')}")

    meta = qa_data.get("meta", {})
    summary = meta.get("localized_regions_summary", [])
    if summary:
        print(f"\nLocalized regions ({len(summary)}):")
        for s in summary:
            print(f"  {s.get('part_type')}  bbox={s.get('bbox')}  conf={s.get('confidence')}")
    else:
        print("No localized_regions_summary in response.")

    warnings = data.get("warnings", [])
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [{w.get('code')}] {w.get('message')}")

    # Safety check.
    violations = _check_no_leak(data, "service-qa")
    if violations:
        print(f"\nSAFETY VIOLATIONS: {violations}", file=sys.stderr)
        return 1
    print("\nSafety check: OK (no forbidden fields)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P1.4b/c region backend smoke test")
    parser.add_argument("--image", required=True, help="Path to input image (jpg/png)")
    parser.add_argument("--query", default="领口在哪里？", help="Chinese region query")
    parser.add_argument("--part", default=None, help="Override requested part_type")
    parser.add_argument("--mode", default="backend-direct",
                        choices=["backend-direct", "service-qa"],
                        help="Test mode (default: backend-direct)")
    parser.add_argument("--backend", default="fashionpedia",
                        choices=["fashionpedia", "full312"],
                        help="Region backend (default: fashionpedia)")
    parser.add_argument("--device", default=None,
                        help="Device (cpu/cuda, default: $VISION_REGION_DEVICE or cpu)")
    parser.add_argument("--fp-model", default=None,
                        help="Fashionpedia model path override")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/mm/qa",
                        help="Service URL (service-qa mode)")
    parser.add_argument("--debug", action="store_true", help="Print full exception on error")
    args = parser.parse_args()

    try:
        if args.mode == "service-qa":
            return run_service_qa(args)
        else:
            return run_backend_direct(args)
    except Exception as exc:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
