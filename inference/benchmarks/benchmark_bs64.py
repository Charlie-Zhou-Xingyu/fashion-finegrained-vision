import json
import sys
import time

sys.path.insert(0, "/workspace/trtllm_work")

PROMPTS = [
    "这件外套是什么面料？", "这条裙子适合搭配什么鞋子？", "这件衣服的领口是什么设计？",
    "这个水洗工艺有什么特点？", "这件外套的颜色是什么？", "这条裤子是什么版型？",
    "这件上衣的袖长是多少？", "这个图案叫什么风格？",
]


def main():
    from transformers import AutoTokenizer
    from tensorrt_llm.hlapi import LLM, ModelConfig, SamplingConfig

    engine_dir = "/workspace/trtllm_work/qwenvl_engine_bs64"
    tok = AutoTokenizer.from_pretrained("/workspace/trtllm_work/models/Qwen-VL-7B-Chat", trust_remote_code=True)
    cfg = ModelConfig(model_dir=engine_dir)
    llm = LLM(cfg)
    sc = SamplingConfig(max_new_tokens=100, end_id=tok.eos_token_id)
    prompt_ids = [tok.encode(p) for p in PROMPTS]

    results = {}
    for bs in [1, 8, 16, 32, 64]:
        batch = (prompt_ids * ((bs // len(prompt_ids)) + 1))[:bs]
        # 3 repeated trials per batch size to get a real sense of variance
        trials = []
        for trial in range(3):
            list(llm.generate(batch, sc))  # warm-up / discard
            t0 = time.perf_counter()
            out = list(llm.generate(batch, sc))
            dt = time.perf_counter() - t0
            trials.append(dt)
        mean_dt = sum(trials) / len(trials)
        qps = bs / mean_dt
        results[f"bs={bs}"] = {
            "trials_s": [round(t, 3) for t in trials],
            "mean_wall_time_s": round(mean_dt, 3),
            "qps": round(qps, 2),
        }
        print(f"[batch={bs:3d}] trials={[round(t,3) for t in trials]}  mean={mean_dt:.3f}s  QPS={qps:.2f}")

    print("\n" + json.dumps(results, indent=2))
    with open("/workspace/trtllm_work/benchmark_bs64_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
