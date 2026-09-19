#!/bin/bash
# PRD 3.4 "GPU utilization >= 75%": sample nvidia-smi at 200 ms during closed-loop saturation runs.
W=/workspace/trtllm_work; set --; source $W/miniconda3/bin/activate && conda activate trtllm; cd $W
for CFG in "qwenvl_engine_bs64 64 64" "qwenvl_engine_bs128 128 128" "qwenvl_engine_bs64 64 8"; do
  set -- $CFG; ENGINE=$1; TAG=$2; C=$3
  SAMP=logs/gpuutil_${TAG}_C${C}.csv; : > $SAMP
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -lms 200 > $SAMP &
  SPID=$!
  python -u executor_inflight_bench.py $W/$ENGINE logs/gpuutil_${TAG}_C${C}_live.txt --skip W1,W2,W3 --concurrency $C --duration 30 --tag util${TAG} > logs/gpuutil_${TAG}_C${C}.log 2>&1
  kill $SPID
  echo "=== $ENGINE closed-loop C=$C ==="
  grep "W4" logs/gpuutil_${TAG}_C${C}_live.txt
  # window = samples 5..35 s of the run (executor load ~7 s + 3 s warm-up excluded by taking the middle)
  python - "$SAMP" <<'PY'
import sys, statistics
rows=[l.split(", ") for l in open(sys.argv[1]) if l.strip()]
n=len(rows); mid=rows[int(n*0.35):int(n*0.90)]
u=[float(r[1].split()[0]) for r in mid]; m=[float(r[2].split()[0]) for r in mid]; p=[float(r[4].split()[0]) for r in mid]
print(f"samples={len(mid)}/{n} gpu_util mean={statistics.fmean(u):.1f}% p10={sorted(u)[len(u)//10]:.0f}% min={min(u):.0f}% | mem_util mean={statistics.fmean(m):.1f}% | power mean={statistics.fmean(p):.0f}W")
PY
done
echo GPUUTIL_DONE
