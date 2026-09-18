"""Real-model smoke test for hybrid retrieval: lexical vs dense+RRF vs +rerank.

Loads BGE-M3 and BGE-reranker-v2-m3 from the local HF cache (no download
unless RAG_ALLOW_MODEL_DOWNLOAD=1) and prints, for a handful of queries, the
top-3 ids from each configuration plus wall-clock cost.  CPU is fine for a
15-document KB.  Not a pytest test — it needs ~4.5 GB of weights.

    python inference/benchmarks/rag_hybrid_smoke.py [--device mps]
"""

from __future__ import annotations

import argparse
import time

from inference.rag.settings import DenseSettings, GenerationSettings, RagSettings, RerankerSettings
from inference.serving.rag_service import RagService

QUERIES = [
    "什么是纤维",                    # exact lexical hit — hybrid should keep it on top
    "衣服洗了会缩水是什么原因",        # paraphrase: no KB term appears verbatim
    "哪种材质比较环保",               # semantic: sustainability category, no lexical anchor
    "涤纶和聚酯纤维是一回事吗",        # alias resolution
    "夏天穿什么面料凉快",             # out-of-KB → ideally low scores / few hits
]


def _fmt(hits):
    return ", ".join(f"{h.id}({h.match_type},{h.score:.2f})" for h in hits[:3]) or "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--embedding-model", default="BAAI/bge-m3",
                    help="hub id or a local snapshot directory (offline deployments should pass the directory: "
                         "FlagEmbedding's offline hub check demands a byte-complete snapshot incl. onnx/)")
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--index-backend", default="numpy", choices=["numpy", "faiss"],
                    help="numpy is exact and dependency-free; faiss on macOS pip needs KMP_DUPLICATE_LIB_OK=TRUE")
    args = ap.parse_args()

    lexical = RagService()
    t0 = time.perf_counter()
    hybrid = RagService(rag_settings=RagSettings(
        dense=DenseSettings(enabled=True, device=args.device, allow_model_download=args.allow_download,
                            embedding_model=args.embedding_model, index_backend=args.index_backend),
        reranker=RerankerSettings(enabled=True, device=args.device, allow_model_download=args.allow_download,
                                  model=args.reranker_model),
        generation=GenerationSettings(),
    ))
    print(f"[load] index={args.index_backend} dense={hybrid._dense is not None} reranker={hybrid._reranker is not None} "
          f"startup_warnings={[w.code for w in hybrid._startup_warnings]} in {time.perf_counter()-t0:.1f}s",
          flush=True)
    if hybrid._dense is None:
        raise SystemExit("dense stack not available — nothing to smoke-test")

    dense_only = RagService(rag_settings=hybrid._rag_settings, dense_retriever=hybrid._dense)
    dense_only._reranker = None

    for q in QUERIES:
        print(f"\nQ: {q}")
        t = time.perf_counter(); r = lexical.retrieve(q); print(f"  lexical   {1000*(time.perf_counter()-t):6.1f}ms  {_fmt(r.hits)}")
        t = time.perf_counter(); r = dense_only.retrieve(q); print(f"  +dense    {1000*(time.perf_counter()-t):6.1f}ms  {_fmt(r.hits)}")
        t = time.perf_counter(); r = hybrid.retrieve(q); print(f"  +rerank   {1000*(time.perf_counter()-t):6.1f}ms  {_fmt(r.hits)}")
        print(f"            tools={r.used_tools} mode={r.meta['retrieval']['retrieval_mode']}", flush=True)


if __name__ == "__main__":
    main()
