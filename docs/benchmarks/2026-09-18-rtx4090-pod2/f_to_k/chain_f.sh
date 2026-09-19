#!/bin/bash
# F — attack the measured bottlenecks (prefill compute, KV capacity) instead of batch size:
#   F1 prefix caching (KV block reuse) on PRD-shaped shared-prefix prompts
#   F2 chunked context for long prompts
#   F3 W8A8 SmoothQuant + executor + bs96
W=/workspace/trtllm_work; cd "$W"; set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
echo "=== F_START $(date -u +%H:%M:%S) ==="

echo "=== F0 baseline W5 on bs96 (no reuse) ==="
bash run_d.sh "$W/qwenvl_engine_bs96_mnt16k" f0_bs96 GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W3,W4
echo "=== F1a block reuse on bs96 (default engine, no paged-context FMHA) ==="
bash run_d.sh "$W/qwenvl_engine_bs96_mnt16k" f1a_bs96_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --block-reuse

echo "=== BUILD_PCF_START $(date -u +%H:%M:%S) ==="
trtllm-build --checkpoint_dir="$W/qwenvl_ckpt" \
  --gemm_plugin=float16 --gpt_attention_plugin=float16 --lookup_plugin=float16 \
  --max_input_len=1024 --max_output_len=256 --max_batch_size=96 --max_num_tokens=16384 \
  --max_prompt_embedding_table_size=2048 --remove_input_padding=enable --paged_kv_cache=enable \
  --use_paged_context_fmha=enable \
  --output_dir="$W/qwenvl_engine_bs96_pcf" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_pcf.log \
  && echo BUILD_PCF_OK || { echo BUILD_PCF_FAILED; grep -iE "error" logs/build_pcf.log | tail -5; }
if [ -f "$W/qwenvl_engine_bs96_pcf/config.json" ]; then
  echo "=== F1b block reuse on pcf engine ==="
  bash run_d.sh "$W/qwenvl_engine_bs96_pcf" f1b_pcf_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --block-reuse
  echo "=== F2a pcf engine, chunked context, no reuse ==="
  bash run_d.sh "$W/qwenvl_engine_bs96_pcf" f2a_pcf_chunked GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W5 --chunked --concurrency 64,96
  echo "=== F2b pcf engine, plain (control for the pcf build itself) ==="
  bash run_d.sh "$W/qwenvl_engine_bs96_pcf" f2b_pcf_plain GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W5 --concurrency 64,96
  echo "=== F2c pcf engine, chunked + reuse (production candidate) ==="
  bash run_d.sh "$W/qwenvl_engine_bs96_pcf" f2c_pcf_chunked_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2 --chunked --block-reuse --concurrency 64,96
fi

echo "=== F3 SmoothQuant convert $(date -u +%H:%M:%S) ==="
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir "$W/models/Qwen-VL-7B-Chat" --output_dir "$W/qwenvl_sq_ckpt" \
  --dtype float16 --smoothquant 0.5 --per_channel --per_token --int8_kv_cache \
  2>&1 | tr '\r' '\n' | grep -vE "Generating train split|Downloading|Loading checkpoint" > logs/sq_convert.log \
  && echo SQ_CONVERT_OK || { echo SQ_CONVERT_FAILED; tail -5 logs/sq_convert.log; }
if [ -f "$W/qwenvl_sq_ckpt/config.json" ]; then
  echo "=== BUILD_SQ96_START $(date -u +%H:%M:%S) ==="
  trtllm-build --checkpoint_dir="$W/qwenvl_sq_ckpt" \
    --gemm_plugin=float16 --gpt_attention_plugin=float16 --lookup_plugin=float16 \
    --max_input_len=1024 --max_output_len=256 --max_batch_size=96 --max_num_tokens=16384 \
    --max_prompt_embedding_table_size=2048 --remove_input_padding=enable --paged_kv_cache=enable \
    --use_paged_context_fmha=enable \
    --output_dir="$W/qwenvl_engine_sq_bs96_pcf" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_sq96.log \
    && echo BUILD_SQ96_OK || { echo BUILD_SQ96_FAILED; grep -iE "error" logs/build_sq96.log | tail -5; }
  if [ -f "$W/qwenvl_engine_sq_bs96_pcf/config.json" ]; then
    echo "=== F3 W8A8 executor ==="
    bash run_d.sh "$W/qwenvl_engine_sq_bs96_pcf" f3_sq96 GUARANTEED_NO_EVICT --n 96 --skip W2 --concurrency 64,96
    bash run_d.sh "$W/qwenvl_engine_sq_bs96_pcf" f3_sq96_chunked_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --chunked --block-reuse
  fi
fi
echo "=== F_DONE $(date -u +%H:%M:%S) ==="
