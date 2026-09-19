#!/bin/bash
# J — prove the 429 (queue cap) and 504 (timeout + cancel) guards fire, with short prompts so 413 does not mask them.
W=/workspace/trtllm_work; cd "$W"; set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
while ! grep -q I_DONE logs/chain_i.log 2>/dev/null; do sleep 10; done
echo "=== J_START $(date -u +%H:%M:%S) ==="
http_run() {
  export MLLM_TEXT_ENGINE_DIR=$1 MLLM_TOKENIZER_DIR=$W/models/Qwen-VL-7B-Chat MLLM_BACKEND=executor
  eval "export $3"
  echo "=== H_START tag=$2 env=[$3] $(date -u +%H:%M:%S) ==="
  python -m uvicorn trtllm_server:app --host 127.0.0.1 --port 8082 --log-level warning > logs/h_server_$2.log 2>&1 &
  SPID=$!
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:8082/health | grep -q '"status":"ok"' && break; sleep 2; done
  python -u http_load_test.py http://127.0.0.1:8082 $4 --live logs/h_$2_live.txt > logs/h_$2.log 2>&1 && echo "H_${2}_OK" || echo "H_${2}_FAILED"
  curl -s http://127.0.0.1:8082/health > logs/h_health_$2.json; cat logs/h_health_$2.json; echo
  kill -INT $SPID; wait $SPID 2>/dev/null; sleep 3
  unset MLLM_MAX_QUEUE MLLM_REQUEST_TIMEOUT_S
}
http_run "$W/qwenvl_engine_bs96_mnt16k" guard429 "MLLM_MAX_QUEUE=8" "--concurrency 64 --duration 15 --min-tokens 100"
http_run "$W/qwenvl_engine_bs96_mnt16k" guard504 "MLLM_MAX_QUEUE=1024 MLLM_REQUEST_TIMEOUT_S=0.4" "--concurrency 16 --duration 15 --min-tokens 100"
http_run "$W/qwenvl_engine_bs96_mnt16k" guard_recover "MLLM_MAX_QUEUE=1024" "--concurrency 16 --duration 15 --min-tokens 100"
echo "=== J_DONE $(date -u +%H:%M:%S) ==="
