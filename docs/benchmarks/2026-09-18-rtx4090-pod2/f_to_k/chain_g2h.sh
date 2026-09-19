#!/bin/bash
# G2 (retry after disk-full): FP16-KV W4A16 checkpoint → paged-context engine → reuse / chunked runs
# H: HTTP-level closed-loop load test through trtllm_server.py's executor backend
W=/workspace/trtllm_work; cd "$W"; set -o pipefail; set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
echo "=== G2H_START $(date -u +%H:%M:%S) ==="; df -h / | tail -1
BUILD_COMMON="--gemm_plugin=float16 --gpt_attention_plugin=float16 --lookup_plugin=float16 --max_input_len=1024 --max_output_len=256 --max_batch_size=96 --max_num_tokens=16384 --max_prompt_embedding_table_size=2048 --remove_input_padding=enable --paged_kv_cache=enable"

rm -rf qwenvl_ckpt_fp16kv
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir "$W/models/Qwen-VL-7B-Chat" --output_dir "$W/qwenvl_ckpt_fp16kv" \
  --dtype float16 --use_weight_only --weight_only_precision int4 \
  2>&1 | tr '\r' '\n' | grep -vE "Loading checkpoint" > logs/convert_fp16kv.log && echo CONVERT_FP16KV_OK || { echo CONVERT_FP16KV_FAILED; tail -3 logs/convert_fp16kv.log; }
if [ -f qwenvl_ckpt_fp16kv/config.json ]; then
  trtllm-build --checkpoint_dir="$W/qwenvl_ckpt_fp16kv" $BUILD_COMMON --use_paged_context_fmha=enable --output_dir="$W/qwenvl_engine_bs96_fp16kv_pcf" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_pcf_fp16kv.log \
    && echo BUILD_PCF_FP16KV_OK || { echo BUILD_PCF_FP16KV_FAILED; grep -iE "error" logs/build_pcf_fp16kv.log | tail -3; }
  rm -rf qwenvl_ckpt_fp16kv          # 5.7 GB; the engine is what we need
  E=$W/qwenvl_engine_bs96_fp16kv_pcf
  if [ -f $E/config.json ]; then
    bash run_d.sh "$E" g2a_pcf_plain         GUARANTEED_NO_EVICT --n 96 --skip W2 --concurrency 64,96
    bash run_d.sh "$E" g2b_pcf_reuse         GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --block-reuse
    bash run_d.sh "$E" g2c_pcf_chunked       GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W5 --chunked --concurrency 64,96
    bash run_d.sh "$E" g2d_pcf_chunked_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2 --chunked --block-reuse --concurrency 64,96
  fi
fi
echo "=== G2_DONE $(date -u +%H:%M:%S) ==="

# ---- H: HTTP through the executor-backed server ----
http_run() {  # $1=engine $2=tag $3=extra env $4=load args
  export MLLM_TEXT_ENGINE_DIR=$1 MLLM_TOKENIZER_DIR=$W/models/Qwen-VL-7B-Chat MLLM_BACKEND=executor
  eval "export $3"
  echo "=== H_START tag=$2 engine=$(basename $1) env=[$3] $(date -u +%H:%M:%S) ==="
  python -m uvicorn trtllm_server:app --host 127.0.0.1 --port 8082 --log-level warning > logs/h_server_$2.log 2>&1 &
  SPID=$!
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:8082/health | grep -q '"status":"ok"' && break; sleep 2; done
  curl -s http://127.0.0.1:8082/health | tee logs/h_health_$2.json; echo
  python -u http_load_test.py http://127.0.0.1:8082 $4 --live logs/h_$2_live.txt > logs/h_$2.log 2>&1 && echo "H_${2}_OK" || { echo "H_${2}_FAILED"; tail -5 logs/h_$2.log; }
  kill -INT $SPID; wait $SPID 2>/dev/null; sleep 3
  unset MLLM_KV_BLOCK_REUSE MLLM_CHUNKED_CONTEXT
}
http_run "$W/qwenvl_engine_bs96_mnt16k" w4a16_short "MLLM_MAX_QUEUE=1024" "--concurrency 16,64,96,192 --duration 30"
http_run "$W/qwenvl_engine_bs96_mnt16k" w4a16_long  "MLLM_MAX_QUEUE=1024" "--concurrency 32,96 --duration 30 --long"
http_run "$W/qwenvl_engine_sq_bs96"     sq_long      "MLLM_MAX_QUEUE=1024" "--concurrency 32,96 --duration 30 --long"
http_run "$W/qwenvl_engine_sq_bs96"     sq_short     "MLLM_MAX_QUEUE=1024" "--concurrency 96 --duration 30"
if [ -f $W/qwenvl_engine_bs96_fp16kv_pcf/config.json ]; then
  http_run "$W/qwenvl_engine_bs96_fp16kv_pcf" pcf_reuse_long "MLLM_MAX_QUEUE=1024 MLLM_KV_BLOCK_REUSE=1 MLLM_CHUNKED_CONTEXT=1" "--concurrency 32,96 --duration 30 --long"
fi
# limits: prove 413 / 429 actually fire
http_run "$W/qwenvl_engine_bs96_mnt16k" limits "MLLM_MAX_QUEUE=8 MLLM_MAX_INPUT_TOKENS=64" "--concurrency 64 --duration 10 --long"
echo "=== H_DONE $(date -u +%H:%M:%S) ==="
