"""Run 3.1.2 visual check: N random images × M queries. Single process, no batch."""
import random, subprocess, time, json
from pathlib import Path
from collections import Counter

IMG_DIR = Path(r"D:\Aliintern\fashion-ai-data\fashionai_attributes\round1_fashionAI_attributes_test_a\Images\lapel_design_labels")
OUT_DIR = Path(r"outputs\mvp_312_20random")
FP_MODEL = r"models\detectors\fashionpedia_yolov8s_19cls_balanced_v1_best.pt"
QUERIES = "口袋,拉链,袖子,领口,内搭"
N_IMAGES = 20

if OUT_DIR.exists():
    import shutil; shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

images = random.sample(sorted(IMG_DIR.glob("*.jpg")), N_IMAGES)
c = Counter()
total_t0 = time.time()

for i, img in enumerate(images, 1):
    t0 = time.time()
    print(f"\n[{i}/{N_IMAGES}] {img.name}", flush=True)
    r = subprocess.run(
        ["python", "tools/demo/query_region_online_demo.py",
         "--image", str(img), "--queries", QUERIES,
         "--output-dir", str(OUT_DIR), "--fp-model", FP_MODEL],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    stdout = r.stdout or ""
    for line in stdout.splitlines():
        if any(k in line for k in ("DONE", "DINO", "FALLBACK", "Error")):
            print(f"  {line}")
    if r.stderr:
        for line in r.stderr.splitlines()[-3:]:
            print(f"  [stderr] {line}")
    elapsed = time.time() - t0
    print(f"  -> {elapsed:.0f}s")

# Collect results.
for f in sorted(OUT_DIR.rglob("result.json")):
    d = json.load(open(f, encoding="utf-8"))
    c.update([(d.get("query", "?"), d.get("backend", "?"), d.get("status", "?"))])

print(f"\n{'='*60}")
print(f"TOTAL: {time.time()-total_t0:.0f}s | {N_IMAGES} images x {len(QUERIES.split(','))} queries")
print(f"{'='*60}")
print(f"  {'Query':<10} {'Backend':<28} {'Status':<14} Count")
print(f"  {'-'*10} {'-'*28} {'-'*14} -----")
for (q, bk, st), n in sorted(c.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), str(x[0][2]))):
    print(f"  {q or '-':<10} {bk or '-':<28} {st or '-':<14} x{n}")
