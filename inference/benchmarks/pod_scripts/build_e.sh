#!/bin/bash
# E — larger max_batch_size without the build-time OOM.
# The bs=96 build on pod #1 died in tactic selection asking for a 6.4 GB buffer.
# Activation memory scales with max_num_tokens, which 0.10 defaults to
# max_batch_size * max_input_len (65536 for bs64; 98304 for bs96).  That default
# sizes the prefill buffers for "every slot does a full-length prefill in the
# same step", which inflight batching never needs.  Capping max_num_tokens is
# the lever; sequence lengths stay identical to the bs64 engine so the numbers
# remain comparable.
set -o pipefail
W=/workspace/trtllm_work
source "$W/miniconda3/bin/activate" && conda activate trtllm
cd "$W"
build() {  # $1=bs $2=max_num_tokens $3=tokens_per_block $4=outdir
  echo "=== BUILD_${4}_START $(date -u +%H:%M:%S) bs=$1 max_num_tokens=$2 tokens_per_block=$3 ==="
  trtllm-build --checkpoint_dir="$W/qwenvl_ckpt" \
    --gemm_plugin=float16 --gpt_attention_plugin=float16 \
    --lookup_plugin=float16 --max_input_len=1024 --max_output_len=256 \
    --max_batch_size=$1 --max_num_tokens=$2 --tokens_per_block=$3 \
    --max_prompt_embedding_table_size=2048 \
    --remove_input_padding=enable --paged_kv_cache=enable \
    --output_dir="$W/$4" 2>&1 | tr '\r' '\n' | grep -vE "Missing scale|json.exception|Optional value" \
    && echo "BUILD_${4}_OK" || echo "BUILD_${4}_FAILED"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}
build 96  16384 64 qwenvl_engine_bs96_mnt16k
build 128 16384 64 qwenvl_engine_bs128_mnt16k
build 96  16384 32 qwenvl_engine_bs96_mnt16k_tpb32
echo E_BUILDS_DONE
