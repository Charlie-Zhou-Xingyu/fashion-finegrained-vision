# Results, verbatim

All runs: Qwen-VL-7B-Chat text decoder unless stated, token-ID inputs,
`max_new_tokens=100`, RTX 4090, Python runtime (`hlapi.LLM` unless stated).

## Short-prompt batch scaling — max_batch_size=64 engine (W4A16 INT4 RTN + INT8 KV), 3 trials
```
[batch=  1] trials=[0.614, 0.614, 0.615]  mean=0.614s  QPS=1.63
[batch=  8] trials=[0.717, 0.719, 0.716]  mean=0.717s  QPS=11.15
[batch= 16] trials=[0.747, 0.748, 0.749]  mean=0.748s  QPS=21.40
[batch= 32] trials=[0.81, 0.808, 0.81]  mean=0.809s  QPS=39.53
[batch= 64] trials=[1.103, 1.106, 1.105]  mean=1.105s  QPS=57.94
```

## Short-prompt batch scaling — max_batch_size=80 engine, 3 trials (plus bs48/72 probe)
```
[batch= 48] trials=[1.064, 0.978, 0.975]  mean=1.006s  QPS=47.73
[batch= 64] trials=[1.207, 1.103, 1.103]  mean=1.138s  QPS=56.25
[batch= 70] trials=[1.461, 1.485, 1.467]  mean=1.471s  QPS=47.59
[batch= 72] trials=[1.466, 1.464, 1.464]  mean=1.465s  QPS=49.15
[batch= 76] trials=[1.48, 1.477, 1.474]  mean=1.477s  QPS=51.47
[batch= 80] trials=[1.499, 1.5, 1.498]  mean=1.499s  QPS=53.37
```
max_batch_size=96: `trtllm-build` failed — `[TRT] [E] 2: OutOfMemory`, "Requested amount of GPU memory (6443499520 bytes) could not be allocated", "Could not find any implementation for node ...".

## Earlier engines (single-trial, 8 sequential prompts at bs=1 then batched)
Qwen-7B-Chat, INT4-AWQ + INT8 KV, max_batch_size=8: bs=1 latencies (ms) 1385.1 (cold), 612.4, 611.6, 611.4, 610.9, 611.6, 611.0, 612.7; batched QPS bs1/2/4/8 = 1.64 / 3.21 / 6.21 / 10.67.
Qwen-VL text decoder, max_batch_size=8: bs=1 latencies (ms) 618.0, 615.0, 613.1, 609.9, 610.1, 610.2, 609.5, 615.0; batched QPS bs1/2/4/8 = 1.63 / 3.21 / 6.26 / 9.80.

## SmoothQuant W8A8 vs W4A16 — max_batch_size=64, short prompts, 3 trials
```
sample output: ' A：这件外套是用羊毛和棉质混纺纱线编织而成的。 B：哦，原来如此。那它应该很保暖吧？ A：是的，它有很好的保暖性能。 C：那它防水吗？ D：不，它不防水。 E'
[batch=  1] trials=[0.997, 0.996, 0.996]  mean=0.996s  QPS=1.00
[batch=  8] trials=[1.076, 1.078, 1.076]  mean=1.077s  QPS=7.43
[batch= 32] trials=[1.172, 1.171, 1.172]  mean=1.172s  QPS=27.31
[batch= 64] trials=[1.291, 1.297, 1.291]  mean=1.293s  QPS=49.50
```

## Realistic-length prompts (system + fabric knowledge + attribute JSON + question), 2 trials
W4A16 (`qwenvl_engine_bs64`):
```
[in= 500 bs=  1] trials=[0.666, 0.666] mean=0.666s QPS=1.50 prefill_tok/s=750
[in= 500 bs=  8] trials=[1.148, 1.148] mean=1.148s QPS=6.97 prefill_tok/s=3484
[in= 500 bs= 32] trials=[2.595, 3.112] mean=2.854s QPS=11.21 prefill_tok/s=5607
[in= 500 bs= 64] trials=[5.904, 5.903] mean=5.904s QPS=10.84 prefill_tok/s=5420
[in=1000 bs=  1] trials=[0.729, 0.725] mean=0.727s QPS=1.38 prefill_tok/s=1375
[in=1000 bs=  8] trials=[1.609, 1.61] mean=1.609s QPS=4.97 prefill_tok/s=4971
[in=1000 bs= 32] trials=[5.632, 5.631] mean=5.631s QPS=5.68 prefill_tok/s=5683
[in=1000 bs= 64] trials=[11.401, 11.402] mean=11.402s QPS=5.61 prefill_tok/s=5613
```
W8A8 (`qwenvl_engine_sq_bs64`):
```
[in= 500 bs=  1] trials=[1.03, 1.029] mean=1.030s QPS=0.97 prefill_tok/s=486
[in= 500 bs=  8] trials=[1.31, 1.311] mean=1.311s QPS=6.10 prefill_tok/s=3052
[in= 500 bs= 32] trials=[2.162, 2.161] mean=2.162s QPS=14.80 prefill_tok/s=7402
[in= 500 bs= 64] trials=[4.545, 4.543] mean=4.544s QPS=14.08 prefill_tok/s=7042
[in=1000 bs=  1] trials=[1.058, 1.057] mean=1.057s QPS=0.95 prefill_tok/s=946
[in=1000 bs=  8] trials=[1.551, 1.552] mean=1.551s QPS=5.16 prefill_tok/s=5156
[in=1000 bs= 32] trials=[4.44, 4.445] mean=4.442s QPS=7.20 prefill_tok/s=7203
[in=1000 bs= 64] trials=[7.878, 7.878] mean=7.878s QPS=8.12 prefill_tok/s=8124
```

## Accuracy — `examples/summarize.py`, cnn_dailymail, `--max_ite 20`
```
HF FP16      rouge1 : 25.587279325556555   rouge2 : 7.926615096020251    rougeL : 18.62119279935413   rougeLsum : 21.536504346884218
W4A16 (RTN)  rouge1 : 24.802867577672227   rouge2 : 7.925192527737924    rougeL : 17.839842488847683  rougeLsum : 21.06778105440116
W8A8 (SQ)    rouge1 : 28.306986234635207   rouge2 : 10.724709244526517   rougeL : 20.262989247188816  rougeLsum : 22.84139753862923
```

## Accuracy — domain greedy agreement (20 Chinese fashion prompts, top_k=1, FP16 reference)
```
 candidate | exact | prefix32 | mean_first_diverge | n
qwenvl_engine_bs64 |   0/20 |   0/20   |    1.0 tok       | 20
qwenvl_engine_sq_bs64 |   0/20 |   0/20   |    8.6 tok       | 20

Q: 这件外套是什么面料？
  FP16 :  A. 皮革 B. 羊毛 C. 丝绸 D. 麻  根据常识，外套通常由不同的面料制成，例如皮革、羊毛、丝绸或麻等。因此，无法仅通过外观来确定这件外套的面料。需要进一步检查或询问卖家
  W4A16:   这个问题没有提供足够的信息来确定外套的面料。需要更多的信息，例如颜色、款式、品牌等。请提供更多的信息以便于确定外套的面料。  如果您需要帮助确定某种特定的面料，请提供其名称或描
  W8A8 :  A：这件外套是用羊毛和棉质混纺纱线编织而成的。 B：哦，原来如此。那它应该很保暖吧？ A：是的，它有很好的保暖性能。 C：那它防水吗？ D：不，它不防水。 E：那如果下雨天
Q: 这条裙子适合搭配什么鞋子？
  FP16 :  A：这是一条黑色的裙子，你可以搭配一双白色的运动鞋，或者一双黑色的高跟鞋，这样会显得更加时尚和优雅。 B：这是一条黑色的裙子，你可以搭配一双金色的高跟鞋，这样会显得更加华丽和高贵
  W4A16:  一条黑白相间的裙子，可以搭配白色或黑色的鞋子。鞋子的选择可以根据个人喜好和场合需要来定。如果裙子是正式场合穿着的，建议搭配一双优雅的高跟鞋；如果裙子是休闲场合穿着的，可以选择一双
  W8A8 :  A：这条裙子是黑色的，我觉得搭配一双白色的运动鞋会很好看。 B：我觉得搭配一双黑色的高跟鞋会更有女人味。 C：如果想要休闲一点，我建议搭配一双白色的帆布鞋。 D：如果想要正式一点
Q: 这件衣服的领口是什么设计？
  FP16 :   这件衣服的领口是V领设计，这种设计可以突出颈部线条，使颈部看起来更加修长。V领的设计也更加时尚，可以增加穿着者的气质和魅力。同时，V领的设计也更加舒适，不会像圆领那样束缚颈部。
  W4A16:   这件衣服的领口采用了V领设计，这种设计可以修饰脸型，拉长颈部线条，同时也能显出颈部的曲线，增添女性的优雅感。V领的设计也更加灵活，可以搭配各种不同的领口，如高领、圆领、立领等，
  W8A8 :   这件衣服的领口采用了V领设计，这种设计可以突出颈部线条，增加穿着者的气场。V领的设计还可以显瘦，拉长颈部线条，让穿着者看起来更加高挑。同时，V领设计也更加灵活，可以搭配各种不同
```
`generation_config.json` of the model (confound check — no repetition_penalty; `do_sample=False` was passed explicitly):
```
{"chat_format": "chatml", "do_sample": true, "eos_token_id": 151643, "max_new_tokens": 512, "max_window_size": 6144, "pad_token_id": 151643, "top_k": 0, "top_p": 0.3, "transformers_version": "4.31.0"}
```

## Vision-path sanity check (synthetic solid-color images, `qwenvl_engine_bs64` + ViT plan)
```
真实颜色: 紫红色 RGB=(168, 32, 120) | 模型回答: '这张图片的背景颜色是洋红色。'
真实颜色: 青绿色 RGB=(24, 178, 140) | 模型回答: '这张图片的背景颜色是翡翠绿色。'
真实颜色: 橙黄色 RGB=(232, 150, 18) | 模型回答: '橙色'
```

## Multimodal end-to-end (`examples/qwenvl/run.py`, demo.jpeg)
```
Input: "[{'image': './pics/demo.jpeg'}, {'text': '这张图片里有什么？'}]"
Output: "这张图片中有一个年轻的女人和一只狗坐在海滩上，他们互相看着对方。海水和沙子构成了这幅画面的大部分，远处有一波波浪正向他们冲过来。"
TensorRT-LLM ViT latency: 0.05642557144165039 sec
TensorRT-LLM QWen time: 1.0172595977783203 sec
```
Same request through `inference/serving/trtllm_server.py` (`fastapi.testclient`, base64 data URI): HTTP 200, identical text, `latency_ms: 1254.37`, `/health` → `{"status": "ok", "mode": "multimodal"}`.

## INT8 fallback warning census — bs64 W4A16 build log
Total `Missing scale and zero-point` lines: 1490. By substring: attention/qkv 128, attention/dense 64, attention/CONSTANT 96, mlp/fc 64, mlp/gate 64, mlp/proj 64, mlp/ (all) 288, input_layernorm 384, post_layernorm 384, ELEMENTWISE_SUM 130, vocab_embedding|ln_f|lm_head 28, KV-cache-related names 4. Distinct layers touched: 32.

## Checkpoint quantization configs
W4A16 (`qwenvl_ckpt/config.json`): `{"quant_algo": "W4A16", "kv_cache_quant_algo": "INT8", "group_size": 128, "smoothquant_val": null, "has_zero_point": false, "pre_quant_scale": false, "exclude_modules": ["lm_head"]}`; safetensors header: `lm_head.weight F16`, `transformer.ln_f.weight F16`, `transformer.vocab_embedding.weight F16`, `transformer.layers.0.attention.qkv.weight I8`.
W8A8 (`qwenvl_sq_ckpt/config.json`): `{"quant_algo": "W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN", "kv_cache_quant_algo": "INT8", "group_size": 128, "smoothquant_val": null, "has_zero_point": false, "pre_quant_scale": false, "exclude_modules": ["lm_head"]}`.
