"""
Is the vision encoder's output actually influencing generation, or is the
model pattern-matching on language priors regardless of the image?
demo.jpeg is TensorRT-LLM's bundled example — plausibly seen by Qwen-VL in
training/eval, and "woman + dog on a beach" is generic enough that a
language-only prior could produce a passable description without any pixels.

Three synthetic solid-color images with specific colors (near-zero chance
of being guessed from language priors). If answers track the actual color
across independent runs, the vision pathway is doing real work.
"""
import os
import sys

sys.path.insert(0, "/workspace/trtllm_work")
sys.path.insert(0, "/workspace/trtllm_work/TensorRT-LLM/examples/qwenvl")

from PIL import Image

COLORS = {
    "紫红色": (168, 32, 120),
    "青绿色": (24, 178, 140),
    "橙黄色": (232, 150, 18),
}


def main():
    import tensorrt as trt
    import torch
    from tensorrt_llm.runtime import Session, TensorInfo
    from vit_onnx_trt import Preprocss
    from run import QWenInfer

    vit_path = "/workspace/trtllm_work/TensorRT-LLM/examples/qwenvl/plan/visual_encoder/visual_encoder_fp16.plan"
    with open(vit_path, "rb") as f:
        session = Session.from_serialized_engine(f.read())
    stream = torch.cuda.current_stream().cuda_stream
    preprocess = Preprocss(448)

    def to_torch(dtype):
        return {trt.float16: torch.float16, trt.float32: torch.float32, trt.int32: torch.int32}[dtype]

    def run_vit(path):
        image = preprocess.encode([path]).to("cuda").contiguous()
        info = session.infer_shapes([TensorInfo("input", trt.DataType.FLOAT, image.shape)])
        outs = {t.name: torch.empty(tuple(t.shape), dtype=to_torch(t.dtype), device="cuda") for t in info}
        assert session.run({"input": image.float()}, outs, stream)
        torch.cuda.synchronize()
        return outs["output"]

    qinfer = QWenInfer("/workspace/trtllm_work/models/Qwen-VL-7B-Chat",
                       "/workspace/trtllm_work/qwenvl_engine_bs64", "info", None, None, num_beams=1)
    qinfer.qwen_model_init()

    print("=== VISION SANITY CHECK: synthetic solid-color images ===", flush=True)
    for name, rgb in COLORS.items():
        path = f"/tmp/synth_{name}.png"
        Image.new("RGB", (448, 448), rgb).save(path)
        embeds = run_vit(path)
        answer = qinfer.qwen_infer(embeds, [{"image": path}],
                                   "这张图片的背景是什么颜色？用一个词回答。", 30, num_beams=1, history=[])
        hit = "命中" if (name[0] in answer or name[:2] in answer) else "需人工判断"
        line = f"真实颜色: {name} RGB={rgb} | 模型回答: {answer!r} | {hit}"
        print(line, flush=True)
        with open("/workspace/trtllm_work/logs/vision_sanity_live.txt", "a") as lf:
            lf.write(line + "\n")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
