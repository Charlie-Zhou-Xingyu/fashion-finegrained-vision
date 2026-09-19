"""D — inflight-batching throughput with the TensorRT-LLM C++ executor.

All earlier numbers in docs/tensorrt_llm_integration.md came from *static*
batching through the Python ``hlapi.LLM``: a batch starts together and the
whole batch waits for its longest member.  The C++ executor
(``tensorrt_llm.bindings.executor``) schedules per iteration — a finished
sequence's slot is refilled from the queue on the next step — which is what
"inflight / continuous batching" means and what production serving uses.

Workloads (greedy; ``min_length == max_new_tokens`` so every request generates
exactly its budget, the same convention as the static runs it is compared to):

  W1 burst-short   N identical-shape ~10-token prompts, 100 new tokens, all
                   enqueued at t=0.  Same shape as the static bs=N number, so
                   the delta is runtime overhead (C++ executor vs Python).
  W2 burst-hetero  N short prompts with per-request budgets U[20,200] (seeded).
                   W2b runs the same N with every budget = max(budgets): that is
                   what static batching would have cost.  W2a/W2b is the gain.
  W3 burst-long    N realistic prompts (in=500 / 1000 tokens, same construction
                   as long_prompt_bench.py), 100 new tokens.
  W4 closed-loop   C concurrent clients, each re-enqueues on completion, for T
                   seconds.  C > max_batch_size exercises the executor's queue —
                   the static path could not run those at all.
                   QPS / p50 / p95 are measured over [t0+3s, deadline] to
                   exclude ramp-up and drain.

Per burst the executor's iteration stats are read back (active requests, KV
blocks in use) so "the batch actually filled" is evidence, not assumption.

    python -u executor_inflight_bench.py <engine_dir> <live_file> \
        --policy GUARANTEED_NO_EVICT|MAX_UTILIZATION [--n 64] [--concurrency 16,32,64,96,128]
"""
import argparse
import datetime
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, "/workspace/trtllm_work")

MODEL_DIR = "/workspace/trtllm_work/models/Qwen-VL-7B-Chat"

PROMPTS = [
    "这件外套是什么面料？", "这条裙子适合搭配什么鞋子？", "这件衣服的领口是什么设计？",
    "这个水洗工艺有什么特点？", "这件外套的颜色是什么？", "这条裤子是什么版型？",
    "这件上衣的袖长是多少？", "这个图案叫什么风格？",
]
KNOWLEDGE = ("羊毛是一种天然蛋白质纤维，具有优良的保暖性、弹性和吸湿性，常用于大衣、西装和针织衫。棉纤维柔软透气、吸湿性强，但易皱缩水。"
             "真丝光泽柔和、手感滑爽，但耐磨性差且怕碱。涤纶强度高、抗皱免烫、快干，但吸湿性差易起静电。亚麻透气凉爽、易起皱。"
             "水洗工艺包括普洗、石磨洗、酵素洗、雪花洗和砂洗，分别赋予牛仔面料不同程度的做旧与柔软手感。刺绣工艺耐洗牢度高，"
             "印花工艺色彩丰富但多次洗涤后可能褪色。拼接工艺通过不同面料或色块组合形成层次感。褶皱工艺利用高温定型形成永久褶。")
ATTRS = '{"garment_category":"外套","fabric":"羊毛混纺","color":"驼色","collar_design":"翻领","sleeve_length":"长袖","style":"通勤","craft":"拼接"}'
QUESTION = "\n根据以上资料，这件外套的面料有什么特点，适合什么季节和场合？请用专业但易懂的语言回答。"
SYSTEM = "你是一名资深服饰专家，请结合面料知识与商品属性回答用户问题。\n参考资料：\n"


def build_long(tok, n_tokens):
    body = SYSTEM + (KNOWLEDGE * 20) + "\n商品属性：" + ATTRS + QUESTION
    ids = tok.encode(body)
    q = tok.encode(QUESTION)
    return ids[: n_tokens - len(q)] + q


# ── executor plumbing ─────────────────────────────────────────────────────────

def make_executor(ex, engine_dir, policy, kv_fraction):
    sched = ex.SchedulerConfig(getattr(ex.CapacitySchedulerPolicy, policy))
    kv = ex.KvCacheConfig(free_gpu_memory_fraction=kv_fraction) if kv_fraction else ex.KvCacheConfig()
    cfg = ex.ExecutorConfig(max_beam_width=1, scheduler_config=sched, kv_cache_config=kv,
                            batching_type=ex.BatchingType.INFLIGHT, iter_stats_max_iterations=200000)
    return ex.Executor(engine_dir, ex.ModelType.DECODER_ONLY, cfg)


def make_request(ex, ids, n_new, end_id):
    sc = ex.SamplingConfig(beam_width=1, top_k=1, min_length=n_new)          # greedy, exact budget
    oc = ex.OutputConfig(exclude_input_from_output=True)
    return ex.Request(input_token_ids=list(ids), max_new_tokens=n_new, streaming=False,
                      sampling_config=sc, output_config=oc, end_id=end_id, pad_id=end_id)


_EMPTY_ITER = {"iters": 0, "peak_active": 0, "mean_active": 0, "peak_scheduled": 0, "peak_queued": 0,
               "peak_kv_blocks": 0, "max_kv_blocks": 0, "tokens_per_block": 0, "mean_iter_ms": 0}


def iteration_summary(executor, wall=None):
    """Summarise the executor iterations since the last call.

    Reads the 0.10 binding attributes directly (IterationStats.num_active_requests,
    .inflight_batching_stats.num_scheduled_requests, .kv_cache_stats.*).  0.10 has
    no per-iteration latency or queue-length field: mean step time is wall/iters,
    and "queued" is active − scheduled (requests the scheduler held back).
    """
    stats = list(executor.get_latest_iteration_stats())
    if not stats:
        return dict(_EMPTY_ITER)
    active = [int(s.num_active_requests) for s in stats]
    sched = [int(getattr(s.inflight_batching_stats, "num_scheduled_requests", 0)) for s in stats]
    used = [int(s.kv_cache_stats.used_num_blocks) for s in stats]
    maxb = [int(s.kv_cache_stats.max_num_blocks) for s in stats]
    queued = [max(0, a - c) for a, c in zip(active, sched)]
    return {
        "iters": len(stats),
        "peak_active": max(active),
        "mean_active": round(statistics.fmean(active), 1),
        "peak_scheduled": max(sched),
        "peak_queued": max(queued),
        "peak_kv_blocks": max(used),
        "max_kv_blocks": max(maxb),
        "tokens_per_block": int(stats[0].kv_cache_stats.tokens_per_block),
        "mean_iter_ms": round(wall / len(stats) * 1000, 2) if wall else 0,
    }


def drain(executor, want, t_enq, on_final):
    got = 0
    while got < want:
        for r in executor.await_responses(datetime.timedelta(milliseconds=5000)):
            if r.has_error():
                raise RuntimeError(f"request {r.request_id}: {r.get_error_msg()}")
            if r.result.is_final:
                got += 1
                on_final(r)
    return got


def run_burst(ex, executor, reqs):
    lat, ntok = {}, {}
    executor.get_latest_iteration_stats()                          # reset the stats window
    t0 = time.perf_counter()
    ids = executor.enqueue_requests(reqs)
    t_enq = {rid: t0 for rid in ids}

    def on_final(r):
        lat[r.request_id] = time.perf_counter() - t_enq[r.request_id]
        ntok[r.request_id] = len(r.result.output_token_ids[0])

    drain(executor, len(ids), t_enq, on_final)
    wall = time.perf_counter() - t0
    return wall, sorted(lat.values()), sum(ntok.values()), iteration_summary(executor, wall)


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def run_closed_loop(ex, executor, ids, n_new, end_id, C, duration, warm=3.0):
    t_enq, done = {}, []                                            # done: (t_done, latency, ntok)
    executor.get_latest_iteration_stats()
    t0 = time.perf_counter()
    deadline = t0 + duration

    def submit():
        rid = executor.enqueue_request(make_request(ex, ids, n_new, end_id))
        t_enq[rid] = time.perf_counter()

    for _ in range(C):
        submit()
    while t_enq:
        for r in executor.await_responses(datetime.timedelta(milliseconds=1000)):
            if r.has_error():
                raise RuntimeError(f"request {r.request_id}: {r.get_error_msg()}")
            if r.result.is_final:
                now = time.perf_counter()
                done.append((now, now - t_enq.pop(r.request_id), len(r.result.output_token_ids[0])))
                if now < deadline:
                    submit()
    wall = time.perf_counter() - t0
    win = [d for d in done if t0 + warm <= d[0] <= deadline]
    win_len = max(deadline - (t0 + warm), 1e-9)
    lats = sorted(d[1] for d in win)
    return {
        "C": C, "duration_s": duration, "wall_s": round(wall, 2), "completed_total": len(done),
        "completed_in_window": len(win), "qps_window": round(len(win) / win_len, 2),
        "tok_s_window": round(sum(d[2] for d in win) / win_len, 0),
        "p50_s": round(pct(lats, 50), 3), "p95_s": round(pct(lats, 95), 3), "max_s": round(pct(lats, 100), 3),
        "iter": iteration_summary(executor, wall),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_dir")
    ap.add_argument("live")
    ap.add_argument("--policy", default="GUARANTEED_NO_EVICT")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--concurrency", default="16,32,64,96,128")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--kv-fraction", type=float, default=0.0, help="0 = executor default")
    ap.add_argument("--skip", default="", help="comma list of W1,W2,W3,W4 to skip")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    skip = set(x for x in args.skip.split(",") if x)

    def log(line):
        print(line, flush=True)
        with open(args.live, "a") as fh:
            fh.write(line + "\n")

    from transformers import AutoTokenizer
    import tensorrt_llm.bindings.executor as ex
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    end_id = getattr(tok, "eod_id", None) or tok.eos_token_id
    assert end_id is not None, "need an end id"

    with open(os.path.join(args.engine_dir, "config.json")) as fh:
        bc = json.load(fh).get("build_config", {})
    engine_meta = {k: bc.get(k) for k in ("max_batch_size", "max_input_len", "max_output_len",
                                          "max_num_tokens", "max_beam_width")}
    log(f"### {args.tag or os.path.basename(args.engine_dir)} policy={args.policy} engine={engine_meta} end_id={end_id}")

    t0 = time.perf_counter()
    executor = make_executor(ex, args.engine_dir, args.policy, args.kv_fraction)
    assert executor.can_enqueue_requests()
    log(f"[load] executor ready in {time.perf_counter()-t0:.1f}s")

    short_ids = [tok.encode(p) for p in PROMPTS]
    results = {"engine": engine_meta, "policy": args.policy}

    # warm-up
    run_burst(ex, executor, [make_request(ex, short_ids[i % 8], 20, end_id) for i in range(8)])

    if "W1" not in skip:
        for n in sorted({1, 8, 32, args.n}):
            reqs_fn = lambda: [make_request(ex, short_ids[i % 8], 100, end_id) for i in range(n)]
            trials, its = [], None
            for _ in range(3):
                wall, lat, ntok, its = run_burst(ex, executor, reqs_fn())
                trials.append(wall)
            m = statistics.fmean(trials)
            results[f"W1_n{n}"] = {"trials": trials, "qps": n / m, "tok_s": n * 100 / m, "iter": its}
            log(f"[W1 burst-short n={n:3d}] trials={[round(x,3) for x in trials]} mean={m:.3f}s "
                f"QPS={n/m:.2f} tok/s={n*100/m:.0f} p50={pct(lat,50):.3f}s max={pct(lat,100):.3f}s "
                f"peak_active={its['peak_active']} kv_blocks={its['peak_kv_blocks']}/{its['max_kv_blocks']} iters={its['iters']}")

    if "W2" not in skip:
        rng = random.Random(20260918)
        budgets = [rng.randint(20, 200) for _ in range(args.n)]
        mx, mean_b = max(budgets), statistics.fmean(budgets)
        for label, buds in (("W2a hetero", budgets), ("W2b static-equiv", [mx] * args.n)):
            trials, its = [], None
            for _ in range(2):
                wall, lat, ntok, its = run_burst(
                    ex, executor, [make_request(ex, short_ids[i % 8], b, end_id) for i, b in enumerate(buds)])
                trials.append(wall)
            m = statistics.fmean(trials)
            results[label.split()[0]] = {"trials": trials, "qps": args.n / m, "tok_s": ntok / m,
                                          "gen_tokens": ntok, "iter": its}
            log(f"[{label} n={args.n}] budgets max={mx} mean={mean_b:.0f} trials={[round(x,3) for x in trials]} "
                f"mean={m:.3f}s QPS={args.n/m:.2f} gen_tok={ntok} tok/s={ntok/m:.0f} "
                f"p50={pct(lat,50):.3f}s max={pct(lat,100):.3f}s peak_active={its['peak_active']} iters={its['iters']}")
        a, b = results["W2a"]["trials"], results["W2b"]["trials"]
        log(f"[W2 gain] static-equiv/hetero wall = {statistics.fmean(b)/statistics.fmean(a):.2f}x "
            f"(ideal if perfectly packed = {mx/mean_b:.2f}x)")

    if "W3" not in skip:
        for n_in in (500, 1000):
            ids = build_long(tok, n_in)
            for n in sorted({8, 32, args.n}):
                trials, its = [], None
                try:
                    for _ in range(2):
                        wall, lat, ntok, its = run_burst(ex, executor, [make_request(ex, ids, 100, end_id) for _ in range(n)])
                        trials.append(wall)
                except Exception as e:
                    log(f"[W3 burst-long in={len(ids)} n={n}] ERROR {type(e).__name__}: {str(e)[:160]}")
                    continue
                m = statistics.fmean(trials)
                results[f"W3_in{n_in}_n{n}"] = {"trials": trials, "qps": n / m, "iter": its}
                log(f"[W3 burst-long in={len(ids):4d} n={n:3d}] trials={[round(x,3) for x in trials]} mean={m:.3f}s "
                    f"QPS={n/m:.2f} prefill_tok/s={n*len(ids)/m:.0f} p50={pct(lat,50):.3f}s max={pct(lat,100):.3f}s "
                    f"peak_active={its['peak_active']} kv_blocks={its['peak_kv_blocks']}/{its['max_kv_blocks']} queued_peak={its['peak_queued']}")

    if "W4" not in skip:
        for C in [int(c) for c in args.concurrency.split(",") if c]:
            try:
                r = run_closed_loop(ex, executor, short_ids[0], 100, end_id, C, args.duration)
            except Exception as e:
                log(f"[W4 closed-loop C={C}] ERROR {type(e).__name__}: {str(e)[:160]}")
                continue
            results[f"W4_C{C}"] = r
            it = r["iter"]
            log(f"[W4 closed-loop C={C:3d} T={args.duration:.0f}s] QPS={r['qps_window']:.2f} tok/s={r['tok_s_window']:.0f} "
                f"p50={r['p50_s']:.3f}s p95={r['p95_s']:.3f}s max={r['max_s']:.3f}s done={r['completed_total']} "
                f"peak_active={it['peak_active']} peak_sched={it['peak_scheduled']} mean_active={it['mean_active']} "
                f"queued_peak={it['peak_queued']} kv_blocks={it['peak_kv_blocks']}/{it['max_kv_blocks']} "
                f"mean_iter_ms={it['mean_iter_ms']}")

    # decode one output as a sanity check that we are generating text, not garbage
    rid = executor.enqueue_request(make_request(ex, short_ids[0], 40, end_id))
    for r in executor.await_responses(rid):
        if r.result.is_final:
            log("[sample] " + repr(tok.decode(r.result.output_token_ids[0]))[:200])

    out = os.path.join(os.path.dirname(args.live), f"executor_{args.tag or os.path.basename(args.engine_dir)}_{args.policy}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False, default=str)
    log(f"[done] results -> {out}")
    log("D_RUN_OK")
    sys.stdout.flush()
    executor.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
