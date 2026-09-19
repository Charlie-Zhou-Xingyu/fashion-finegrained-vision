"""Closed-loop HTTP load test against trtllm_server.py (the PRD's "Locust" test, without Locust).

C concurrent clients each POST /v1/chat/completions and re-send on completion for T
seconds; reports QPS and latency percentiles over [warm, T] plus the server's own
/health counters (in-flight, rejections, p50/p95). Prompts cycle through 8 short
fashion questions (--long adds a ~900-token shared knowledge prefix).

    python http_load_test.py http://127.0.0.1:8082 --concurrency 16,64,96,192 --duration 30
"""
import argparse
import asyncio
import json
import statistics
import sys
import time

import httpx

PROMPTS = [
    "这件外套是什么面料？", "这条裙子适合搭配什么鞋子？", "这件衣服的领口是什么设计？",
    "这个水洗工艺有什么特点？", "这件外套的颜色是什么？", "这条裤子是什么版型？",
    "这件上衣的袖长是多少？", "这个图案叫什么风格？",
]
KNOWLEDGE = ("羊毛是一种天然蛋白质纤维，具有优良的保暖性、弹性和吸湿性，常用于大衣、西装和针织衫。棉纤维柔软透气、吸湿性强，但易皱缩水。"
             "真丝光泽柔和、手感滑爽，但耐磨性差且怕碱。涤纶强度高、抗皱免烫、快干，但吸湿性差易起静电。亚麻透气凉爽、易起皱。"
             "水洗工艺包括普洗、石磨洗、酵素洗、雪花洗和砂洗，分别赋予牛仔面料不同程度的做旧与柔软手感。刺绣工艺耐洗牢度高，"
             "印花工艺色彩丰富但多次洗涤后可能褪色。拼接工艺通过不同面料或色块组合形成层次感。褶皱工艺利用高温定型形成永久褶。")


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


async def client(i, base, deadline, warm, out, max_tokens, long, sem_stats, min_tokens=None):
    async with httpx.AsyncClient(base_url=base, timeout=120.0) as c:
        k = i
        while True:
            q = PROMPTS[k % 8]; k += 1
            # KNOWLEDGE*3 ≈ 780 tokens + ChatML ≈ 820 < max_input_len 1024 (KNOWLEDGE*6 was 413-rejected)
            content = (("你是服饰专家。参考资料：" + KNOWLEDGE * 3 + "\n") if long else "") + q
            body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0}
            if min_tokens:
                body["min_tokens"] = min_tokens
            t0 = time.perf_counter()
            try:
                r = await c.post("/v1/chat/completions", json=body)
                dt = time.perf_counter() - t0
                if r.status_code == 200:
                    m = r.json().get("meta", {})
                    out.append((t0, dt, m.get("latency_ms", 0) / 1000, 200))
                else:
                    out.append((t0, dt, 0, r.status_code))
            except Exception as e:  # noqa: BLE001
                out.append((t0, time.perf_counter() - t0, 0, -1))
                sem_stats["errors"].append(type(e).__name__)
            if time.perf_counter() >= deadline:
                return


async def run(base, C, duration, warm, max_tokens, long, min_tokens=None):
    out, stats = [], {"errors": []}
    t0 = time.perf_counter()
    await asyncio.gather(*(client(i, base, t0 + duration, warm, out, max_tokens, long, stats, min_tokens) for i in range(C)))
    win = [o for o in out if t0 + warm <= o[0] + o[1] <= t0 + duration]
    ok = [o for o in win if o[3] == 200]
    codes = {}
    for o in out:
        codes[o[3]] = codes.get(o[3], 0) + 1
    lat = [o[1] for o in ok]
    srv = [o[2] for o in ok]
    wl = max(duration - warm, 1e-9)
    async with httpx.AsyncClient(base_url=base, timeout=10) as c:
        h = (await c.get("/health")).json()
    return {
        "C": C, "qps": round(len(ok) / wl, 2), "n_ok_window": len(ok), "status_codes": codes,
        "p50_s": round(pct(lat, 50), 3), "p95_s": round(pct(lat, 95), 3), "max_s": round(pct(lat, 100), 3),
        "server_p50_s": round(pct(srv, 50), 3), "http_overhead_p50_ms": round((pct(lat, 50) - pct(srv, 50)) * 1000, 1),
        "errors": sorted(set(stats["errors"])), "health": {k: h.get(k) for k in ("executor", "latency", "limits")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--concurrency", default="16,64,96,192")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warm", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--min-tokens", type=int, default=None, help="force exactly this many new tokens (comparable to the executor bench)")
    ap.add_argument("--live", default=None)
    a = ap.parse_args()
    for C in [int(x) for x in a.concurrency.split(",") if x]:
        r = asyncio.run(run(a.base, C, a.duration, a.warm, a.max_tokens, a.long, a.min_tokens))
        line = (f"[HTTP closed-loop C={C:3d} T={a.duration:.0f}s long={a.long} min_tokens={a.min_tokens}] QPS={r['qps']} p50={r['p50_s']}s "
                f"p95={r['p95_s']}s max={r['max_s']}s server_p50={r['server_p50_s']}s http_overhead_p50={r['http_overhead_p50_ms']}ms "
                f"codes={r['status_codes']} errors={r['errors']} health={json.dumps(r['health'], ensure_ascii=False)}")
        print(line, flush=True)
        if a.live:
            open(a.live, "a").write(line + "\n")
    print("HTTP_LOAD_OK", flush=True)


if __name__ == "__main__":
    main()
