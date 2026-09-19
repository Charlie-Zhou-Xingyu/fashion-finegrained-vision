#!/bin/bash
# K — hypothesis: block reuse / chunked context fail only because of the prompt-table (lookup_plugin) path.
# Build the same FP16-KV W4A16 bs96 paged-context engine WITHOUT --lookup_plugin / --max_prompt_embedding_table_size
# (text-only: what the RAG path needs) and rerun plain / reuse / chunked.
W=/workspace/trtllm_work; cd "$W"; set -o pipefail; set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
while ! grep -q J_DONE logs/chain_j.log 2>/dev/null; do sleep 10; done
echo "=== K_START $(date -u +%H:%M:%S) ==="; df -h / | tail -1
rm -rf qwenvl_ckpt_fp16kv
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py --model_dir "$W/models/Qwen-VL-7B-Chat" --output_dir "$W/qwenvl_ckpt_fp16kv" \
  --dtype float16 --use_weight_only --weight_only_precision int4 2>&1 | grep -vE "Loading checkpoint" > logs/convert_fp16kv2.log && echo CONVERT_OK || { echo CONVERT_FAILED; exit 1; }
trtllm-build --checkpoint_dir="$W/qwenvl_ckpt_fp16kv" \
  --gemm_plugin=float16 --gpt_attention_plugin=float16 --max_input_len=1024 --max_output_len=256 \
  --max_batch_size=96 --max_num_tokens=16384 --remove_input_padding=enable --paged_kv_cache=enable \
  --use_paged_context_fmha=enable --output_dir="$W/qwenvl_engine_bs96_textonly_pcf" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_textonly_pcf.log \
  && echo BUILD_TEXTONLY_PCF_OK || { echo BUILD_TEXTONLY_PCF_FAILED; grep -iE error logs/build_textonly_pcf.log | tail -3; exit 1; }
rm -rf qwenvl_ckpt_fp16kv
E=$W/qwenvl_engine_bs96_textonly_pcf
bash run_d.sh "$E" k_text_plain   GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4
bash run_d.sh "$E" k_text_reuse   GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --block-reuse
bash run_d.sh "$E" k_text_chunked GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W5 --chunked --concurrency 64,96
bash run_d.sh "$E" k_text_chunked_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2,W4 --chunked --block-reuse
echo "=== K_DONE $(date -u +%H:%M:%S) ==="
