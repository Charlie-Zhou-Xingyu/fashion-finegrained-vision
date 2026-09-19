# Evidence — pod #2 (RunPod RTX 4090, 2026-09-18 23:08–00:00 UTC): D + E

**Provenance.** Unlike `../2026-09-18-rtx4090/` (reconstructed after the pod
was lost), this directory was `scp`'d straight off the box while it was still
running. Nothing here was retyped.

| file | what |
|---|---|
| `d_<engine>_<policy>_live.txt` | one line per benchmark step, written as each finished (the numbers quoted in the docs) |
| `executor_<engine>_<policy>.json` | the same results as structured JSON incl. per-burst executor iteration stats |
| `engine_configs/*.config.json` | `trtllm-build` output config of every engine benchmarked (max_batch_size, max_num_tokens, tokens_per_block, plugins, quant) |
| `build_e.log` | the three E builds (bs96 / bs128 / bs96-tpb32 with `--max_num_tokens 16384`) incl. TRT memory lines |
| `build_bs64.filtered.log` | checkpoint conversion + bs64 build + first ViT attempt (progress bars and the INT8-fallback warning noise stripped; the raw log is 60 MB) |
| `chain_de.log`, `after_chain.log` | orchestration markers with UTC timestamps |
| `install.log`, `install2.log` | the environment rebuild, incl. the `--index-url` failure and its fix |
| `pip_freeze.txt`, `nvidia_smi.txt` | the resolved environment and driver (580.159.04 — pod #1 had 570.195.03) |
| `*.sh`, `executor_inflight_bench.py`, `dl_model.py` | the scripts exactly as executed |

Engine recipe is identical to pod #1's `qwenvl_engine_bs64` (INT4 weight-only
RTN + INT8 KV cache, `lookup_plugin`, `max_input_len=1024`, `max_output_len=256`,
`remove_input_padding`, `paged_kv_cache`), so numbers are comparable; the
pod #1 bs64 static-batching figures reproduce on pod #2 within 1 % (57.94 →
57.38 QPS at bs64 W1).

Not in this directory: the ViT plan (built after the archive was taken; see
`after_chain.log` for pass 1's OOM and `vit_pass2` in the docs) and the
SmoothQuant W8A8 engine (not rebuilt on pod #2 — D/E did not need it).
