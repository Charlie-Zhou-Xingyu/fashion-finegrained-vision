#!/bin/bash
# L — production candidate for the text/RAG path: W8A8 SmoothQuant (prefill lever) + FP16 KV + paged-context FMHA
#     + text-only (no prompt table) so KV block reuse (prefix caching) works. Compare against K (W4A16 same config).
W=/workspace/trtllm_work; cd "$W"; set -o pipefail; set --
source "$W/miniconda3/bin/activate" && conda activate trtllm
echo "=== L_START $(date -u +%H:%M:%S) ==="; df -h / | tail -1
rm -rf qwenvl_sq_ckpt_fp16kv
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py --model_dir "$W/models/Qwen-VL-7B-Chat" --output_dir "$W/qwenvl_sq_ckpt_fp16kv" \
  --dtype float16 --smoothquant 0.5 --per_channel --per_token 2>&1 | tr '\r' '\n' | grep -vE "Generating train split|Downloading|Loading checkpoint|calibrating" > logs/sq_convert_fp16kv.log \
  && echo SQ_FP16KV_CONVERT_OK || { echo SQ_FP16KV_CONVERT_FAILED; tail -3 logs/sq_convert_fp16kv.log; exit 1; }
trtllm-build --checkpoint_dir="$W/qwenvl_sq_ckpt_fp16kv" \
  --gemm_plugin=float16 --gpt_attention_plugin=float16 --max_input_len=1024 --max_output_len=256 \
  --max_batch_size=96 --max_num_tokens=16384 --remove_input_padding=enable --paged_kv_cache=enable \
  --use_paged_context_fmha=enable --output_dir="$W/qwenvl_engine_sq_bs96_textonly_pcf" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" > logs/build_sq_textonly_pcf.log \
  && echo BUILD_SQ_TEXTONLY_PCF_OK || { echo BUILD_SQ_TEXTONLY_PCF_FAILED; grep -iE error logs/build_sq_textonly_pcf.log | tail -3; exit 1; }
rm -rf qwenvl_sq_ckpt_fp16kv
E=$W/qwenvl_engine_sq_bs96_textonly_pcf
bash run_d.sh "$E" l_sq_text_plain GUARANTEED_NO_EVICT --n 96 --skip W2 --concurrency 64,96
bash run_d.sh "$E" l_sq_text_reuse GUARANTEED_NO_EVICT --n 96 --skip W1,W2 --block-reuse --concurrency 64,96
echo "=== L_DONE $(date -u +%H:%M:%S) ==="
