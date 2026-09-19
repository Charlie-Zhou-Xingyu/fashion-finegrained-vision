#!/bin/bash
set -o pipefail
W=/workspace/trtllm_work
source "$W/miniconda3/bin/activate" && conda activate trtllm
cd "$W"
echo "=== CONVERT_START $(date -u +%H:%M:%S) ==="
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir "$W/models/Qwen-VL-7B-Chat" \
  --output_dir "$W/qwenvl_ckpt" \
  --dtype float16 --use_weight_only --weight_only_precision int4 --int8_kv_cache \
  || { echo CONVERT_FAILED; exit 1; }
echo CONVERT_OK
echo "=== BUILD_BS64_START $(date -u +%H:%M:%S) ==="
trtllm-build --checkpoint_dir="$W/qwenvl_ckpt" \
  --gemm_plugin=float16 --gpt_attention_plugin=float16 \
  --lookup_plugin=float16 --max_input_len=1024 --max_output_len=256 \
  --max_batch_size=64 --max_prompt_embedding_table_size=2048 \
  --remove_input_padding=enable --paged_kv_cache=enable \
  --output_dir="$W/qwenvl_engine_bs64" \
  || { echo BUILD_BS64_FAILED; exit 1; }
echo BUILD_BS64_OK
echo "=== VIT_START $(date -u +%H:%M:%S) ==="
cd "$W/TensorRT-LLM/examples/qwenvl"
python3 vit_onnx_trt.py --pretrained_model_path "$W/models/Qwen-VL-7B-Chat" --only_trt \
  || { echo VIT_FAILED; exit 1; }
echo VIT_OK
echo ALL_DONE
