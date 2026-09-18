# Shell steps actually executed on the pod, verbatim

Kept for provenance. The reusable forms are in `inference/benchmarks/`,
`inference/engines/qwen_vl_trtllm_builder.py` and the docs.

## Environment (step2_install.sh)
```bash
source /workspace/trtllm_work/miniconda3/bin/activate
conda create -n trtllm python=3.10 -y -c conda-forge --override-channels
conda activate trtllm
pip install --upgrade pip
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install --extra-index-url https://pypi.nvidia.com "tensorrt_llm==0.10.0"
```
Follow-up fixes applied afterwards (each after the import error it cures):
```bash
pip install --extra-index-url https://pypi.nvidia.com 'tensorrt-cu12==10.0.1' 'tensorrt==10.0.1'
apt-get install -y openmpi-bin libopenmpi-dev
pip install 'cuda-python==12.5.0'
pip install 'pynvml==11.5.0'
pip install 'rouge_score~=0.1.2' 'transformers_stream_generator==0.0.4' tiktoken 'mpmath==1.3.0'   # examples/quantization/requirements.txt minus nemo-toolkit
pip install einops matplotlib 'torchvision==0.17.2' --index-url https://download.pytorch.org/whl/cu121   # Qwen-VL modeling code imports
pip install 'datasets==2.19.0'
pip install 'sentencepiece~=0.1.99' fastapi 'uvicorn[standard]' httpx requests pydantic
git clone --depth 1 --branch v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git
```

## Qwen-VL text decoder — convert (step8) and build (step9, max_bs=8)
```bash
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir /workspace/trtllm_work/models/Qwen-VL-7B-Chat \
  --output_dir /workspace/trtllm_work/qwenvl_ckpt \
  --dtype float16 --use_weight_only --weight_only_precision int4 --int8_kv_cache

trtllm-build --checkpoint_dir=/workspace/trtllm_work/qwenvl_ckpt \
  --gemm_plugin=float16 --gpt_attention_plugin=float16 \
  --lookup_plugin=float16 --max_input_len=2048 --max_output_len=1024 \
  --max_batch_size=8 --max_prompt_embedding_table_size=2048 \
  --remove_input_padding=enable --paged_kv_cache=enable \
  --output_dir=/workspace/trtllm_work/qwenvl_engine
```

## ViT engine (step10, then --only_trt after the in-process OOM)
```bash
cd TensorRT-LLM/examples/qwenvl
python3 vit_onnx_trt.py --pretrained_model_path /workspace/trtllm_work/models/Qwen-VL-7B-Chat
python3 vit_onnx_trt.py --pretrained_model_path /workspace/trtllm_work/models/Qwen-VL-7B-Chat --only_trt
```

## Larger-batch rebuilds (step12 bs=64 / step13 bs=96 OOM / step14 bs=80)
```bash
trtllm-build --checkpoint_dir=/workspace/trtllm_work/qwenvl_ckpt \
  --gemm_plugin=float16 --gpt_attention_plugin=float16 \
  --lookup_plugin=float16 --max_input_len=1024 --max_output_len=256 \
  --max_batch_size=64 --max_prompt_embedding_table_size=2048 \
  --remove_input_padding=enable --paged_kv_cache=enable \
  --output_dir=/workspace/trtllm_work/qwenvl_engine_bs64
# identical with --max_batch_size=96 (OOM) and =80 (ok)
```

## SmoothQuant W8A8 (step16a convert, step16b build after the library patch)
```bash
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir /workspace/trtllm_work/models/Qwen-VL-7B-Chat \
  --output_dir /workspace/trtllm_work/qwenvl_sq_ckpt \
  --dtype float16 --smoothquant 0.5 --per_channel --per_token --int8_kv_cache

python3 inference/deployment/patches/patch_trtllm_0_10_0_qwen_smoothquant.py

trtllm-build --checkpoint_dir=/workspace/trtllm_work/qwenvl_sq_ckpt \
  --gemm_plugin=float16 --gpt_attention_plugin=float16 \
  --lookup_plugin=float16 --max_input_len=1024 --max_output_len=256 \
  --max_batch_size=64 --max_prompt_embedding_table_size=2048 \
  --remove_input_padding=enable --paged_kv_cache=enable \
  --output_dir=/workspace/trtllm_work/qwenvl_engine_sq_bs64
```

## Accuracy + long-prompt chain (step17 / step18)
```bash
python3 -u domain_eval.py --backend trt --engine_dir $W4 --out eval/domain_w4a16.json
python3 -u domain_eval.py --backend trt --engine_dir $W8 --out eval/domain_w8a8.json
python3 -u domain_eval.py --backend hf --out eval/domain_fp16.json
python3 compare_domain.py eval/domain_fp16.json eval/domain_w4a16.json eval/domain_w8a8.json
python3 -u TensorRT-LLM/examples/summarize.py --test_trt_llm --hf_model_dir $HF --data_type fp16 --engine_dir $W4 --max_ite 20
python3 -u TensorRT-LLM/examples/summarize.py --test_trt_llm --hf_model_dir $HF --data_type fp16 --engine_dir $W8 --max_ite 20
python3 -u TensorRT-LLM/examples/summarize.py --test_hf     --hf_model_dir $HF --data_type fp16 --engine_dir $W4 --max_ite 20   # --engine_dir is required even for --test_hf
python3 -u long_prompt_bench.py $W4 logs/b_w4a16_live.txt
python3 -u long_prompt_bench.py $W8 logs/b_w8a8_live.txt
```
