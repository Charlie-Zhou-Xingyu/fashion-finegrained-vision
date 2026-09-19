#!/bin/bash
# run_d.sh <engine_dir> <tag> <policies comma> [extra args...]
W=/workspace/trtllm_work
ENGINE=$1; TAG=$2; POLICIES=$3; shift 3
EXTRA=("$@")
set --   # conda's activate script inherits $@ when sourced and rejects >1 positional arg
source "$W/miniconda3/bin/activate" && conda activate trtllm || { echo "D_${TAG}_ENV_FAILED"; exit 1; }
for P in ${POLICIES//,/ }; do
  LIVE="$W/logs/d_${TAG}_${P}_live.txt"; : > "$LIVE"
  echo "=== D_START tag=$TAG policy=$P $(date -u +%H:%M:%S) ==="
  python -u "$W/executor_inflight_bench.py" "$ENGINE" "$LIVE" --policy "$P" --tag "$TAG" "${EXTRA[@]}" \
    > "$W/logs/d_${TAG}_${P}.log" 2>&1
  grep -q D_RUN_OK "$LIVE" && echo "D_${TAG}_${P}_OK" || { echo "D_${TAG}_${P}_FAILED"; tr '\r' '\n' < "$W/logs/d_${TAG}_${P}.log" | grep -vE "Missing scale|json.exception|Optional value" | grep -iE "error|traceback|assert" | tail -n 6; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
