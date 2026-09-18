"""Realistic-length prompt throughput: system + fabric knowledge + attribute JSON + question, padded to N tokens."""
import json, os, sys, time
sys.path.insert(0, "/workspace/trtllm_work")
ENGINE, LIVE = sys.argv[1], sys.argv[2]
KNOWLEDGE = ("羊毛是一种天然蛋白质纤维，具有优良的保暖性、弹性和吸湿性，常用于大衣、西装和针织衫。棉纤维柔软透气、吸湿性强，但易皱缩水。"
             "真丝光泽柔和、手感滑爽，但耐磨性差且怕碱。涤纶强度高、抗皱免烫、快干，但吸湿性差易起静电。亚麻透气凉爽、易起皱。"
             "水洗工艺包括普洗、石磨洗、酵素洗、雪花洗和砂洗，分别赋予牛仔面料不同程度的做旧与柔软手感。刺绣工艺耐洗牢度高，"
             "印花工艺色彩丰富但多次洗涤后可能褪色。拼接工艺通过不同面料或色块组合形成层次感。褶皱工艺利用高温定型形成永久褶。")
ATTRS = '{"garment_category":"外套","fabric":"羊毛混纺","color":"驼色","collar_design":"翻领","sleeve_length":"长袖","style":"通勤","craft":"拼接"}'
QUESTION = "\n根据以上资料，这件外套的面料有什么特点，适合什么季节和场合？请用专业但易懂的语言回答。"
SYSTEM = "你是一名资深服饰专家，请结合面料知识与商品属性回答用户问题。\n参考资料：\n"

def build(tok, n_tokens):
    body = SYSTEM + (KNOWLEDGE * 20) + "\n商品属性：" + ATTRS + QUESTION
    ids = tok.encode(body)
    q = tok.encode(QUESTION)
    return ids[: n_tokens - len(q)] + q  # keep the question at the end

def main():
    from transformers import AutoTokenizer
    from tensorrt_llm.hlapi import LLM, ModelConfig, SamplingConfig
    tok = AutoTokenizer.from_pretrained("/workspace/trtllm_work/models/Qwen-VL-7B-Chat", trust_remote_code=True)
    llm = LLM(ModelConfig(model_dir=ENGINE)); sc = SamplingConfig(max_new_tokens=100, end_id=tok.eos_token_id)
    for n in [500, 1000]:
        ids = build(tok, n)
        for bs in [1, 8, 32, 64]:
            batch = [ids] * bs
            try:
                list(llm.generate(batch, sc)); t = []
                for _ in range(2):
                    t0 = time.perf_counter(); list(llm.generate(batch, sc)); t.append(time.perf_counter() - t0)
                m = sum(t) / len(t)
                line = f"[in={len(ids):4d} bs={bs:3d}] trials={[round(x,3) for x in t]} mean={m:.3f}s QPS={bs/m:.2f} prefill_tok/s={bs*len(ids)/m:.0f}"
            except Exception as e:
                line = f"[in={len(ids):4d} bs={bs:3d}] ERROR {type(e).__name__}: {str(e)[:120]}"
            print(line, flush=True); open(LIVE, "a").write(line + "\n")
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
