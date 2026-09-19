#!/bin/bash
# G — follow-up to F after the 0.10.0 finding "Paged Context FMHA doesn't work with int8 kv cache":
#   G1 W8A8 SmoothQuant engine WITHOUT paged-context FMHA (keeps INT8 KV) → executor W1/W3/W4
#   G2 W4A16 checkpoint with FP16 KV (no --int8_kv_cache) → bs96 engine WITH paged-context FMHA
#      → prefix caching (W5) and chunked context (W3/W4), each against a plain control on the same engine
W=/workspace/trtllm_work; cd "$W"; set -o pipefail; set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
while ! grep -q F_DONE logs/chain_f.log 2>/dev/null; do sleep 10; done
echo "=== G_START $(date -u +%H:%M:%S) ==="
BUILD_COMMON="--gemm_plugin=float16 --gpt_attention_plugin=float16 --lookup_plugin=float16 --max_input_len=1024 --max_output_len=256 --max_batch_size=96 --max_num_tokens=16384 --max_prompt_embedding_table_size=2048 --remove_input_padding=enable --paged_kv_cache=enable"

echo "=== G1 BUILD_SQ96 (int8 kv, no pcf) $(date -u +%H:%M:%S) ==="
if [ -f qwenvl_sq_ckpt/config.json ]; then
  trtllm-build --checkpoint_dir="$W/qwenvl_sq_ckpt" $BUILD_COMMON --output_dir="$W/qwenvl_engine_sq_bs96" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_sq96b.log \
    && echo BUILD_SQ96_OK || { echo BUILD_SQ96_FAILED; grep -iE "error" logs/build_sq96b.log | tail -5; }
  [ -f qwenvl_engine_sq_bs96/config.json ] && bash run_d.sh "$W/qwenvl_engine_sq_bs96" g1_sq96 GUARANTEED_NO_EVICT --n 96 --skip W2 --concurrency 64,96
else
  echo "SQ_CKPT_MISSING"
fi

echo "=== G2 CONVERT_FP16KV $(date -u +%H:%M:%S) ==="
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir "$W/models/Qwen-VL-7B-Chat" --output_dir "$W/qwenvl_ckpt_fp16kv" \
  --dtype float16 --use_weight_only --weight_only_precision int4 \
  2>&1 | tr '\r' '\n' | grep -vE "Loading checkpoint" > logs/convert_fp16kv.log && echo CONVERT_FP16KV_OK || { echo CONVERT_FP16KV_FAILED; tail -5 logs/convert_fp16kv.log; }
if [ -f qwenvl_ckpt_fp16kv/config.json ]; then
  echo "=== G2 BUILD_PCF_FP16KV $(date -u +%H:%M:%S) ==="
  trtllm-build --checkpoint_dir="$W/qwenvl_ckpt_fp16kv" $BUILD_COMMON --use_paged_context_fmha=enable --output_dir="$W/qwenvl_engine_bs96_fp16kv_pcf" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_pcf_fp16kv.log \
    && echo BUILD_PCF_FP16KV_OK || { echo BUILD_PCF_FP16KV_FAILED; grep -iE "error" logs/build_pcf_fp16kv.log | tail -5; }
  echo "=== G2 BUILD_FP16KV_PLAIN (control: same checkpoint, no pcf) ==="
  trtllm-build --checkpoint_dir="$W/qwenvl_ckpt_fp16kv" $BUILD_COMMON --output_dir="$W/qwenvl_engine_bs96_fp16kv" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_fp16kv.log \
    && echo BUILD_FP16KV_OK || echo BUILD_FP16KV_FAILED
  E=$W/qwenvl_engine_bs96_fp16kv_pcf
  if [ -f $E/config.json ]; then
    bash run_d.sh "$E" g2a_pcf_plain      GUARANTEED_NO_EVICT --n 96 --skip W2 --concurrency 64,96
    bash run_d.sh "$E" g2b_pcf_reuse      GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --block-reuse
    bash run_d.sh "$E" g2c_pcf_chunked    GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W5 --chunked --concurrency 64,96
    bash run_d.sh "$E" g2d_pcf_chunked_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2 --chunked --block-reuse --concurrency 64,96
  fi
  [ -f qwenvl_engine_bs96_fp16kv/config.json ] && bash run_d.sh "$W/qwenvl_engine_bs96_fp16kv" g2e_fp16kv_nopcf GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4
fi
echo "=== G_DONE $(date -u +%H:%M:%S) ==="
