#!/bin/bash
# Self-sequencing D/E chain. Markers in logs/chain_de.log.
W=/workspace/trtllm_work
cd "$W"
echo "=== CHAIN_START $(date -u +%H:%M:%S) ==="
# 1) wait for the bs64 + ViT build job to finish (GPU must be free)
while ! grep -qE "ALL_DONE|VIT_FAILED|BUILD_BS64_FAILED|CONVERT_FAILED" logs/build_bs64.log 2>/dev/null; do sleep 10; done
grep -q BUILD_BS64_OK logs/build_bs64.log || { echo "CHAIN_ABORT_no_bs64_engine"; exit 1; }
grep -q VIT_OK logs/build_bs64.log && echo "VIT_ENGINE_OK" || echo "VIT_ENGINE_MISSING (D does not need it)"

# 2) D on the canonical bs64 engine, both scheduler policies
bash "$W/run_d.sh" "$W/qwenvl_engine_bs64" bs64 GUARANTEED_NO_EVICT,MAX_UTILIZATION --concurrency 16,32,64,96,128

# 3) E builds
bash "$W/build_e.sh" > logs/build_e.log 2>&1
grep -E "BUILD_.*_(OK|FAILED)|E_BUILDS_DONE" logs/build_e.log

# 4) D on whichever E engines built
if grep -q BUILD_qwenvl_engine_bs96_mnt16k_OK logs/build_e.log; then
  bash "$W/run_d.sh" "$W/qwenvl_engine_bs96_mnt16k" bs96 GUARANTEED_NO_EVICT,MAX_UTILIZATION --n 96 --skip W2 --concurrency 32,64,96,128,192
fi
if grep -q BUILD_qwenvl_engine_bs128_mnt16k_OK logs/build_e.log; then
  bash "$W/run_d.sh" "$W/qwenvl_engine_bs128_mnt16k" bs128 GUARANTEED_NO_EVICT --n 128 --skip W2 --concurrency 64,128,192,256
fi
if grep -q BUILD_qwenvl_engine_bs96_mnt16k_tpb32_OK logs/build_e.log; then
  bash "$W/run_d.sh" "$W/qwenvl_engine_bs96_mnt16k_tpb32" bs96tpb32 GUARANTEED_NO_EVICT --n 96 --skip W2,W3 --concurrency 64,96,128
fi
echo "=== CHAIN_DONE $(date -u +%H:%M:%S) ==="
