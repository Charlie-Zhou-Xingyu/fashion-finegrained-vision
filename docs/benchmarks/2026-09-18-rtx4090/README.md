# Benchmark evidence — RunPod RTX 4090, TensorRT-LLM 0.10.0, 2026-09-18

**Provenance — read this first.** The pod's container disk was stopped
before the `archive/` directory could be tar-fetched (the fetch failed on
connect). Everything in this directory was reconstructed **verbatim from
the tool outputs captured in the working session**, not re-run and not
re-typed from memory. Files that could not be reproduced verbatim were
deliberately **omitted rather than approximated**:

- `pip freeze` of the real resolved environment — not recovered. The
  pins that mattered are in `requirements-gpu-server.txt` (each with the
  reason it exists).
- The per-prompt token-id JSONs from the domain greedy-agreement eval
  (`eval/domain_{fp16,w4a16,w8a8}.json`) — not recovered; only the summary
  table and three side-by-side samples were captured (see
  `results_verbatim.md`).
- The SmoothQuant engine's build-stats lines — not captured (only its
  `SQ_BUILD_OK` marker was).

Contents:

| file | what |
|---|---|
| `results_verbatim.md` | every benchmark / eval output line, grouped by experiment, exactly as printed |
| `registry.json` | the engine registry as it stood on the box (paths are the pod's) |
| `engine_build_stats.txt` | weight / activation / KV-pool figures from the `trtllm-build` logs |
| `versions.txt` | resolved versions of the components that determine the numbers |
| `steps_verbatim.md` | the build / quantize / eval shell scripts that were actually executed |

The generalized, reusable forms of the scripts live in `inference/benchmarks/`;
the narrative and interpretation live in `docs/tensorrt_llm_integration.md`.
