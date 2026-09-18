"""
Latency + throughput benchmark for the TensorRT-LLM engines built in this
session. Not TensorRT-LLM's own benchmark suite (that only ships hardcoded
support for falcon/llama-2/gpt-j and needs a C++ binary this pip install
doesn't have) — this drives the real hlapi.LLM API directly against our
actual built engines, with realistic short e-commerce-QA-style Chinese
prompts matching the PRD's use case.

Uses token-ID inputs, not raw strings — tensorrt_llm 0.10.0's hlapi has a
bug where LLM.generate() with string prompts asserts on a missing
tokenizer internally even when one was passed to LLM(); confirmed during
manual testing earlier in this session. Tokenizing ourselves sidesteps it.

Measures, separately and honestly:
  1. Single-stream latency (batch_size=1) percentiles — matches PRD's
     "单轮交互平均响应时间" metric.
  2. Batched throughput at batch_size 1/2/4/8 (engine's built max_batch_size)
     — matches PRD's "系统吞吐量" metric. This submits N prompts in ONE
     llm.generate() call, which is how TensorRT-LLM actually achieves
     throughput above 1/latency — NOT N sequential single-item calls.
"""
import json
import statistics
import sys
import time

sys.path.insert(0, "/workspace/trtllm_work")

PROMPTS = [
    "这件外套是什么面料？",
    "这条裙子适合搭配什么鞋子？",
    "这件衣服的领口是什么设计？",
    "这个水洗工艺有什么特点？",
    "这件外套的颜色是什么？",
    "这条裤子是什么版型？",
    "这件上衣的袖长是多少？",
    "这个图案叫什么风格？",
]


def bench_engine(engine_dir, tokenizer_dir, label, max_new_tokens=100):
    from transformers import AutoTokenizer
    from tensorrt_llm.hlapi import LLM, ModelConfig, SamplingConfig

    print(f"\n{'='*60}\nBenchmarking: {label}\nengine_dir={engine_dir}\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    cfg = ModelConfig(model_dir=engine_dir)
    llm = LLM(cfg)
    sc = SamplingConfig(max_new_tokens=max_new_tokens, end_id=tok.eos_token_id)

    prompt_ids = [tok.encode(p) for p in PROMPTS]

    results = {"label": label, "max_new_tokens": max_new_tokens}

    # --- 1. Single-stream latency (batch_size=1), 8 sequential requests ---
    latencies_ms = []
    for p, ids in zip(PROMPTS, prompt_ids):
        t0 = time.perf_counter()
        out = list(llm.generate([ids], sc))
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)
        text = tok.decode(out[0].token_ids)
        print(f"  [bs=1] {dt:7.1f} ms  | {p[:20]}... -> {text[:30]!r}")

    latencies_ms.sort()
    n = len(latencies_ms)
    results["single_stream"] = {
        "n": n,
        "mean_ms": round(statistics.mean(latencies_ms), 1),
        "p50_ms": round(latencies_ms[n // 2], 1),
        "p95_ms": round(latencies_ms[min(n - 1, int(n * 0.95))], 1),
        "p99_ms": round(latencies_ms[min(n - 1, int(n * 0.99))], 1),
        "min_ms": round(latencies_ms[0], 1),
        "max_ms": round(latencies_ms[-1], 1),
    }

    # --- 2. Batched throughput at increasing batch sizes ---
    results["batched_throughput"] = {}
    for bs in [1, 2, 4, 8]:
        batch = (prompt_ids * ((bs // len(prompt_ids)) + 1))[:bs]
        # warm-up
        list(llm.generate(batch, sc))
        t0 = time.perf_counter()
        out = list(llm.generate(batch, sc))
        dt = time.perf_counter() - t0
        qps = bs / dt
        results["batched_throughput"][f"bs={bs}"] = {
            "wall_time_s": round(dt, 3),
            "qps": round(qps, 2),
        }
        print(f"  [batch={bs}] wall={dt:.3f}s  QPS={qps:.2f}")

    del llm
    return results


def main():
    all_results = {}

    all_results["qwen_7b_chat_text"] = bench_engine(
        "/workspace/trtllm_work/engine_out_qwen7b/trt_engine",
        "/workspace/trtllm_work/models/Qwen-7B-Chat-tokenizer-only",
        "Qwen-7B-Chat (text-only, INT4-AWQ + INT8 KV-cache)",
    )

    all_results["qwen_vl_text_decoder"] = bench_engine(
        "/workspace/trtllm_work/qwenvl_engine",
        "/workspace/trtllm_work/models/Qwen-VL-7B-Chat",
        "Qwen-VL-7B-Chat text decoder (INT4 weight-only + INT8 KV-cache, text-only path)",
    )

    print("\n\n" + "=" * 60)
    print("FULL RESULTS (JSON)")
    print("=" * 60)
    print(json.dumps(all_results, indent=2, ensure_ascii=False))

    with open("/workspace/trtllm_work/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
