import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/trtllm_work/models/Qwen-VL-7B-Chat", trust_remote_code=True)
ref = json.load(open(sys.argv[1]))["outputs"]
def strip(seq):
    e = getattr(tok, "eos_token_id", None) or getattr(tok, "eod_id", None) or 151643
    return seq[:seq.index(e)] if e in seq else seq
print(f"{'candidate':>10} | exact | prefix32 | mean_first_diverge | n")
rows = {}
for path in sys.argv[2:]:
    d = json.load(open(path)); name = d["engine"].rsplit("/", 1)[-1]
    exact = pre = 0; div = []
    for p, r in ref.items():
        c = d["outputs"].get(p, []); r = strip(r); c = strip(c)
        exact += int(r == c)
        k = min(32, len(r), len(c)); pre += int(r[:k] == c[:k] and k > 0)
        m = next((i for i, (x, y) in enumerate(zip(r, c)) if x != y), min(len(r), len(c))); div.append(m)
    n = len(ref); rows[name] = (exact, pre, sum(div) / n)
    print(f"{name:>10} | {exact:>3}/{n} | {pre:>3}/{n}   | {sum(div)/n:6.1f} tok       | {n}")
print("\n=== 3 side-by-side samples (FP16 ref first) ===")
cands = [json.load(open(p)) for p in sys.argv[2:]]
for p in list(ref)[:3]:
    print("\nQ:", p); print("  FP16 :", tok.decode(strip(ref[p]))[:90].replace("\n", " "))
    for d in cands: print(f"  {d['engine'].rsplit('/',1)[-1][:5]:>5}:", tok.decode(strip(d["outputs"].get(p, [])))[:90].replace("\n", " "))
