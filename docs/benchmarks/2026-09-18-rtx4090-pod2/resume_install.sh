#!/bin/bash
set -o pipefail
W=/workspace/trtllm_work
source "$W/miniconda3/bin/activate" && conda activate trtllm || { echo INSTALL_FAILED_activate; exit 1; }
echo "=== STEP_vision_reqs_fixed $(date -u +%H:%M:%S) ==="
pip install 'torchvision==0.17.2' --extra-index-url https://download.pytorch.org/whl/cu121 || { echo INSTALL_FAILED_torchvision; exit 1; }
pip install einops matplotlib || { echo INSTALL_FAILED_einops; exit 1; }
echo "=== STEP_datasets $(date -u +%H:%M:%S) ==="
pip install 'datasets==2.19.0' || { echo INSTALL_FAILED_datasets; exit 1; }
echo "=== STEP_serving_reqs $(date -u +%H:%M:%S) ==="
pip install 'sentencepiece~=0.1.99' fastapi 'uvicorn[standard]' httpx requests pydantic || { echo INSTALL_FAILED_serving_reqs; exit 1; }
echo "=== STEP_openmpi_wait $(date -u +%H:%M:%S) ==="
ldconfig -p | grep -q libmpi.so.40 || { echo INSTALL_FAILED_libmpi; exit 1; }
echo "=== STEP_clone $(date -u +%H:%M:%S) ==="
cd "$W"
[ -d "$W/TensorRT-LLM" ] || git clone --depth 1 --branch v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git || { echo INSTALL_FAILED_clone; exit 1; }
echo "=== STEP_patch $(date -u +%H:%M:%S) ==="
python "$W/patch_trtllm_0_10_0_qwen_smoothquant.py" || { echo INSTALL_FAILED_patch; exit 1; }
echo "=== STEP_verify $(date -u +%H:%M:%S) ==="
python - <<'PY' || { echo INSTALL_FAILED_verify; exit 1; }
import tensorrt_llm, torch, tensorrt, cuda, pynvml, transformers
print("tensorrt_llm", tensorrt_llm.__version__)
print("torch", torch.__version__, "cuda ok:", torch.cuda.is_available())
print("tensorrt", tensorrt.__version__)
print("transformers", transformers.__version__)
from tensorrt_llm.bindings import executor
print("executor bindings:", hasattr(executor, "Executor"), "inflight:", hasattr(executor.BatchingType, "INFLIGHT"))
PY
echo INSTALL_OK
