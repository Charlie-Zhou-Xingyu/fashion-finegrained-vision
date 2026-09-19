#!/bin/bash
# I — reruns after two harness bugs: Response.error_msg accessor; HTTP long prompt exceeded max_input_len.
W=/workspace/trtllm_work; cd "$W"; set -o pipefail; set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
while ! grep -q H_DONE logs/chain_g2h.log 2>/dev/null; do sleep 10; done
echo "=== I_START $(date -u +%H:%M:%S) ==="
E=$W/qwenvl_engine_bs96_fp16kv_pcf
bash run_d.sh "$E" i2b_pcf_reuse         GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --block-reuse
bash run_d.sh "$E" i2c_pcf_chunked       GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W5 --chunked --concurrency 64,96
bash run_d.sh "$E" i2d_pcf_chunked_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2 --chunked --block-reuse --concurrency 64,96
echo "=== I2_DONE $(date -u +%H:%M:%S) ==="

http_run() {
  export MLLM_TEXT_ENGINE_DIR=$1 MLLM_TOKENIZER_DIR=$W/models/Qwen-VL-7B-Chat MLLM_BACKEND=executor
  eval "export $3"
  echo "=== H_START tag=$2 engine=$(basename $1) env=[$3] $(date -u +%H:%M:%S) ==="
  python -m uvicorn trtllm_server:app --host 127.0.0.1 --port 8082 --log-level warning > logs/h_server_$2.log 2>&1 &
  SPID=$!
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:8082/health | grep -q '"status":"ok"' && break; sleep 2; done
  python -u http_load_test.py http://127.0.0.1:8082 $4 --live logs/h_$2_live.txt > logs/h_$2.log 2>&1 && echo "H_${2}_OK" || { echo "H_${2}_FAILED"; tail -5 logs/h_$2.log; }
  curl -s http://127.0.0.1:8082/health > logs/h_health_$2.json
  kill -INT $SPID; wait $SPID 2>/dev/null; sleep 3
  unset MLLM_KV_BLOCK_REUSE MLLM_CHUNKED_CONTEXT
}
http_run "$W/qwenvl_engine_bs96_mnt16k" w4a16_short_min100 "MLLM_MAX_QUEUE=1024" "--concurrency 64,96,192 --duration 30 --min-tokens 100"
http_run "$W/qwenvl_engine_bs96_mnt16k" w4a16_long2        "MLLM_MAX_QUEUE=1024" "--concurrency 32,96 --duration 30 --long --min-tokens 100"
http_run "$W/qwenvl_engine_sq_bs96"     sq_long2           "MLLM_MAX_QUEUE=1024" "--concurrency 32,96 --duration 30 --long --min-tokens 100"
http_run "$E"                           pcf_reuse_long2    "MLLM_MAX_QUEUE=1024 MLLM_KV_BLOCK_REUSE=1 MLLM_CHUNKED_CONTEXT=1" "--concurrency 32,96 --duration 30 --long --min-tokens 100"
http_run "$E"                           pcf_plain_long2    "MLLM_MAX_QUEUE=1024" "--concurrency 32,96 --duration 30 --long --min-tokens 100"
echo "=== I_DONE $(date -u +%H:%M:%S) ==="
