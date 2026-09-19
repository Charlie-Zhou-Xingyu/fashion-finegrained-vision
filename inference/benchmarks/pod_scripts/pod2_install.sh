#!/bin/bash
# Environment rebuild for pod #2 — mirrors docs/benchmarks/2026-09-18-rtx4090/steps_verbatim.md
# exactly (same order, same pins) so the result is comparable with the first pod.
# Markers: STEP_* lines + INSTALL_OK / INSTALL_FAILED_<step> for the poller.
set -o pipefail
W=/workspace/trtllm_work
cd "$W" || exit 1

step() { echo; echo "=== STEP_$1 $(date -u +%H:%M:%S) ==="; }
fail() { echo "INSTALL_FAILED_$1"; exit 1; }

step miniconda
if [ ! -d "$W/miniconda3" ]; then
  # wait for the background wget started by the prep step
  while [ ! -f "$W/Miniconda3-latest-Linux-x86_64.sh" ] || pgrep -f "wget.*Miniconda" > /dev/null; do sleep 5; done
  bash "$W/Miniconda3-latest-Linux-x86_64.sh" -b -p "$W/miniconda3" || fail miniconda
fi
source "$W/miniconda3/bin/activate" || fail activate

step conda_env
conda env list | grep -q '^trtllm ' || conda create -n trtllm python=3.10 -y -c conda-forge --override-channels || fail conda_env
conda activate trtllm || fail conda_activate
python --version
pip install --upgrade pip -q

step torch
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 || fail torch

step tensorrt_llm
pip install --extra-index-url https://pypi.nvidia.com "tensorrt_llm==0.10.0" || fail tensorrt_llm

step pins
pip install --extra-index-url https://pypi.nvidia.com 'tensorrt-cu12==10.0.1' 'tensorrt==10.0.1' || fail tensorrt_pin
pip install 'cuda-python==12.5.0' 'pynvml==11.5.0' || fail cuda_python
pip install 'rouge_score~=0.1.2' 'transformers_stream_generator==0.0.4' tiktoken 'mpmath==1.3.0' || fail quant_reqs
pip install "torchvision==0.17.2" --extra-index-url https://download.pytorch.org/whl/cu121 && pip install einops matplotlib || fail vision_reqs   # --index-url REPLACES PyPI: einops/matplotlib are not on the torch index
pip install 'datasets==2.19.0' || fail datasets
pip install 'sentencepiece~=0.1.99' fastapi 'uvicorn[standard]' httpx requests pydantic || fail serving_reqs

step openmpi_wait
# apt install runs in another tmux window; wait for its marker
while ! grep -q 'APT_DONE' "$W/logs/apt.log" 2>/dev/null; do sleep 5; done
ldconfig -p | grep -q libmpi.so.40 || fail libmpi

step clone
[ -d "$W/TensorRT-LLM" ] || git clone --depth 1 --branch v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git || fail clone

step patch
python "$W/patch_trtllm_0_10_0_qwen_smoothquant.py" || fail patch

step verify
python - <<'PY' || fail verify
import tensorrt_llm, torch, tensorrt, cuda, pynvml, transformers
print("tensorrt_llm", tensorrt_llm.__version__)
print("torch", torch.__version__, "cuda ok:", torch.cuda.is_available())
print("tensorrt", tensorrt.__version__)
print("transformers", transformers.__version__)
from tensorrt_llm.bindings import executor
print("executor bindings:", hasattr(executor, "Executor"), "inflight:", hasattr(executor.BatchingType, "INFLIGHT"))
PY
echo INSTALL_OK
