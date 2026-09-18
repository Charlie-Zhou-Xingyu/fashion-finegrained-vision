"""Greedy-decode a fixed set of Chinese fashion prompts with one backend and dump token ids.
Same raw prompt tokens for every backend (no chat template) so outputs are directly comparable."""
import argparse, json, os, sys
sys.path.insert(0, "/workspace/trtllm_work")

PROMPTS = [
    "这件外套是什么面料？", "这条裙子适合搭配什么鞋子？", "这件衣服的领口是什么设计？",
    "这个水洗工艺有什么特点？", "这件外套的颜色是什么？", "这条裤子是什么版型？",
    "这件上衣的袖长是多少？", "这个图案叫什么风格？",
    "羊毛大衣应该怎么保养？", "真丝面料和涤纶面料有什么区别？", "什么是阔腿裤，适合什么身材？",
    "牛仔布的水洗工艺有哪几种？", "亚麻衬衫容易起皱怎么办？", "刺绣工艺和印花工艺哪个更耐洗？",
    "这件针织衫适合春秋穿吗？", "西装外套的肩线应该在哪里？", "什么是A字裙？", "羽绒服的充绒量多少算保暖？",
    "皮革外套如何防止开裂？", "雪纺面料适合做什么款式？",
]
MAX_NEW = 64

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["trt", "hf"], required=True)
    ap.add_argument("--engine_dir"); ap.add_argument("--hf_dir", default="/workspace/trtllm_work/models/Qwen-VL-7B-Chat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir, trust_remote_code=True)
    ids = [tok.encode(p) for p in PROMPTS]
    # Qwen tokenizer sets eod_id (151643, <|endoftext|>) but not eos_token_id; ModelRunner asserts end_id is not None
    EOS = getattr(tok, "eos_token_id", None) or getattr(tok, "eod_id", None) or 151643
    outs = {}
    if a.backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.hf_dir, device_map="cuda", torch_dtype=torch.float16, fp16=True, trust_remote_code=True).eval()
        for p, i in zip(PROMPTS, ids):
            x = torch.tensor([i], device="cuda")
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, eos_token_id=EOS, pad_token_id=EOS)
            outs[p] = y[0, len(i):].tolist()
            print(p, "->", tok.decode(outs[p])[:60].replace("\n", " "), flush=True)
    else:
        import torch
        from tensorrt_llm.runtime import ModelRunner
        runner = ModelRunner.from_dir(engine_dir=a.engine_dir, rank=0)
        for p, i in zip(PROMPTS, ids):
            with torch.no_grad():
                r = runner.generate([torch.tensor(i, dtype=torch.int32)], max_new_tokens=MAX_NEW, end_id=EOS, pad_id=EOS,
                                    top_k=1, temperature=1.0, return_dict=True)
            seq = r["output_ids"][0, 0, len(i):].tolist()
            if EOS in seq: seq = seq[:seq.index(EOS)]
            outs[p] = seq
            print(p, "->", tok.decode(seq)[:60].replace("\n", " "), flush=True)
    json.dump({"backend": a.backend, "engine": a.engine_dir, "outputs": outs}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("WROTE", a.out, flush=True); sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
