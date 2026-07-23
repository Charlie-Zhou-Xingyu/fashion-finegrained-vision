#!/usr/bin/env python
"""P0a.8 — Lightweight serving latency benchmark.

Uses FastAPI TestClient (no real server, no network, no image downloads).
Default: report-only.  Use ``--fail-on-budget`` for CI-hard-fail mode.

Usage::

    python scripts/bench_serving.py
    python scripts/bench_serving.py --iterations 100 --warmup 10
    python scripts/bench_serving.py --output artifacts/serving_latency_report.json
    python scripts/bench_serving.py --fail-on-budget
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from inference.serving.app import app


# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_BUDGETS_MS: Dict[str, float] = {
    "health": 10,
    "intent_attribute": 10,
    "intent_knowledge": 10,
    "rag_retrieve_fiber": 20,
    "mm_qa_attribute_request": 30,
    "mm_qa_attribute_image_url_mock": 50,
    "mm_qa_knowledge": 50,
    "mm_qa_image_bytes_no_leak": 50,
    "merchant_selling_points_basic": 30,
    "merchant_title_basic": 30,
    "merchant_policy_blocked": 30,
}

BENCH_CASES: List[Dict[str, Any]] = [
    {"id": "health", "method": "GET", "endpoint": "/v1/health", "body": None},
    {"id": "intent_attribute", "method": "POST", "endpoint": "/v1/intent/classify",
     "body": {"query": "这是什么面料"}},
    {"id": "intent_knowledge", "method": "POST", "endpoint": "/v1/intent/classify",
     "body": {"query": "纤维是什么"}},
    {"id": "rag_retrieve_fiber", "method": "POST", "endpoint": "/v1/rag/retrieve",
     "body": {"query": "纤维是什么", "top_k": 3}},
    {"id": "mm_qa_attribute_request", "method": "POST", "endpoint": "/v1/mm/qa",
     "body": {"query": "这件衣服是什么面料？",
              "attributes": {"fabric": {"value": "棉", "attribute_confidence": 0.86, "source": "request_raw"}}}},
    {"id": "mm_qa_attribute_image_url_mock", "method": "POST", "endpoint": "/v1/mm/qa",
     "body": {"query": "这件衣服是什么面料？", "image_url": "http://example.com/img.jpg"}},
    {"id": "mm_qa_knowledge", "method": "POST", "endpoint": "/v1/mm/qa",
     "body": {"query": "纤维是什么"}},
    {"id": "mm_qa_image_bytes_no_leak", "method": "POST", "endpoint": "/v1/mm/qa",
     "body": {"query": "这件衣服是什么面料？",
              "image_bytes": "SECRET_BASE64_DO_NOT_LEAK_bench"}},
    {"id": "merchant_selling_points_basic", "method": "POST", "endpoint": "/v1/merchant/content/generate",
     "body": {"content_type": "selling_points",
              "attributes": {"fabric": {"value": "棉"}, "color": {"value": "白色"}, "style": {"value": "通勤"}}}},
    {"id": "merchant_title_basic", "method": "POST", "endpoint": "/v1/merchant/content/generate",
     "body": {"content_type": "title", "garment_category": "衬衫",
              "attributes": {"fabric": {"value": "棉"}, "color": {"value": "白色"}}}},
    {"id": "merchant_policy_blocked", "method": "POST", "endpoint": "/v1/merchant/content/generate",
     "body": {"content_type": "selling_points",
              "attributes": {"fabric": "棉", "function": "抗菌"}}},
]


# ── Helpers ────────────────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    case_id: str
    iterations: int
    success_count: int
    error_count: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    budget_ms: float
    within_budget: bool


def _pct(values: List[float], pct: float) -> float:
    """Return the *pct*-th percentile of *values* (linear interpolation)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _run_case(client: TestClient, case: dict, iterations: int, budget_ms: float) -> BenchResult:
    latencies: List[float] = []
    errors = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            if case["method"] == "GET":
                r = client.get(case["endpoint"])
            else:
                r = client.post(case["endpoint"], json=case.get("body"))
            if r.status_code >= 400:
                errors += 1
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            errors += 1
    return BenchResult(
        case_id=case["id"],
        iterations=iterations,
        success_count=iterations - errors,
        error_count=errors,
        p50_ms=round(_pct(latencies, 50), 2),
        p90_ms=round(_pct(latencies, 90), 2),
        p95_ms=round(_pct(latencies, 95), 2),
        p99_ms=round(_pct(latencies, 99), 2),
        max_ms=round(max(latencies) if latencies else 0, 2),
        budget_ms=budget_ms,
        within_budget=(_pct(latencies, 95) <= budget_ms) if latencies else True,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Serving latency benchmark (P0a.8)")
    ap.add_argument("--iterations", type=int, default=50, help="measurement iterations")
    ap.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    ap.add_argument("--output", type=Path, default=None, help="JSON output path")
    ap.add_argument("--fail-on-budget", action="store_true",
                     help="exit non-zero if any case exceeds budget")
    args = ap.parse_args()

    client = TestClient(app)

    # Warmup.
    for _ in range(args.warmup):
        for case in BENCH_CASES:
            try:
                if case["method"] == "GET":
                    client.get(case["endpoint"])
                else:
                    client.post(case["endpoint"], json=case.get("body"))
            except Exception:
                pass

    # Measure.
    results: List[BenchResult] = []
    for case in BENCH_CASES:
        budget = DEFAULT_BUDGETS_MS.get(case["id"], 50)
        r = _run_case(client, case, args.iterations, budget)
        results.append(r)

    # Print summary.
    header = f"{'Case':<35s} {'P50':>7s} {'P95':>7s} {'Max':>7s} {'Budget':>7s} {'OK':>5s}"
    print(header)
    print("-" * len(header))
    any_over = False
    for r in results:
        flag = "OK" if r.within_budget else "OVER"
        if not r.within_budget:
            any_over = True
        print(f"{r.case_id:<35s} {r.p50_ms:>7.1f} {r.p95_ms:>7.1f} "
              f"{r.max_ms:>7.1f} {r.budget_ms:>7.0f} {flag:>5s}")
    print(f"\nErrors: {sum(r.error_count for r in results)}")
    print(f"Iterations per case: {args.iterations}")

    # Write JSON.
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "meta": {"iterations": args.iterations, "warmup": args.warmup,
                     "fail_on_budget": args.fail_on_budget, "label": "[measured]"},
            "results": [{
                "case_id": r.case_id, "iterations": r.iterations,
                "success_count": r.success_count, "error_count": r.error_count,
                "p50_ms": r.p50_ms, "p90_ms": r.p90_ms, "p95_ms": r.p95_ms,
                "p99_ms": r.p99_ms, "max_ms": r.max_ms,
                "budget_ms": r.budget_ms, "within_budget": r.within_budget,
            } for r in results],
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")

    if args.fail_on_budget and any_over:
        sys.exit(1)


if __name__ == "__main__":
    main()
