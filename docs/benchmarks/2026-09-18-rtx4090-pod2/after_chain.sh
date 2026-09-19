#!/bin/bash
# Runs after chain_de.sh finishes: re-run D on bs64 (the first attempt failed on env
# activation), then build the ViT engine in two passes (export ONNX, then --only_trt).
W=/workspace/trtllm_work
cd "$W"
echo "=== AFTER_CHAIN_WAIT $(date -u +%H:%M:%S) ==="
while ! grep -q "CHAIN_DONE" logs/chain_de.log 2>/dev/null; do sleep 15; done
echo "=== AFTER_CHAIN_START $(date -u +%H:%M:%S) ==="
bash "$W/run_d.sh" "$W/qwenvl_engine_bs64" bs64 GUARANTEED_NO_EVICT,MAX_UTILIZATION --concurrency 16,32,64,96,128

echo "=== VIT_PASS1 (onnx export; TRT build may OOM with the torch model resident) $(date -u +%H:%M:%S) ==="
set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
cd "$W/TensorRT-LLM/examples/qwenvl"
python3 vit_onnx_trt.py --pretrained_model_path "$W/models/Qwen-VL-7B-Chat" > "$W/logs/vit_pass1.log" 2>&1
ls -la visual_encoder/visual_encoder.onnx 2>&1 && echo VIT_ONNX_OK || echo VIT_ONNX_FAILED
if [ -f plan/visual_encoder/visual_encoder_fp16.plan ]; then
  echo "VIT_PLAN_OK_PASS1"
else
  echo "=== VIT_PASS2 (--only_trt) $(date -u +%H:%M:%S) ==="
  python3 vit_onnx_trt.py --pretrained_model_path "$W/models/Qwen-VL-7B-Chat" --only_trt > "$W/logs/vit_pass2.log" 2>&1
  [ -f plan/visual_encoder/visual_encoder_fp16.plan ] && echo VIT_PLAN_OK_PASS2 || { echo VIT_PLAN_FAILED; tr '\r' '\n' < "$W/logs/vit_pass2.log" | grep -iE "error|Traceback" | tail -n 5; }
fi
ls -la plan/visual_encoder/ 2>&1
echo "=== AFTER_CHAIN_DONE $(date -u +%H:%M:%S) ==="
