import time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download("Qwen/Qwen-VL-Chat", local_dir="/workspace/trtllm_work/models/Qwen-VL-7B-Chat",
                      max_workers=8)
print("MODEL_DOWNLOADED", p, f"{time.time()-t0:.0f}s", flush=True)
