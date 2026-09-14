"""不调用生成模型，直接查看模型将收到的证据与预算诊断。"""

import argparse
import json
from pathlib import Path

from .context import build_context
from .evidence_context import build_evidence_context
from .hybrid_retrieve import RetrievalEngine
from .query_guard import static_corpus_rejection_reason
from .retrieve import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--chunks-file", default="data/generated/chunks.json")
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rerank-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--context-policy", choices=("legacy", "blocks"), default="blocks")
    parser.add_argument("--max-chars", type=int, default=2200)
    args = parser.parse_args()
    reason = static_corpus_rejection_reason(args.question)
    if reason:
        print(f"能力守卫拒绝，无上下文：{reason}")
        return
    path = Path(__file__).resolve().parent.parent / args.chunks_file
    engine = RetrievalEngine(load_chunks(path), device=args.device, rerank=args.rerank,
                             rerank_device=args.rerank_device, cache_path=path.parent / "vector_index.npz")
    hits = engine.search(args.question, args.method)
    results = [hit.as_result() for hit in hits]
    package = (build_evidence_context(results, args.max_chars) if args.context_policy == "blocks"
               else build_context(results, max_chars=args.max_chars, extract_evidence=args.method == "bm25"))
    print(f"候选数：{len(hits)}；实际来源数：{len(package.sources)}；字符：{len(package.text)}/{args.max_chars}")
    print("\n实际上下文：\n" + package.text)
    print("\n打包诊断：\n" + json.dumps(package.diagnostics, ensure_ascii=False, indent=2))
    print("\n重排输入诊断：\n" + json.dumps(engine.last_rerank_stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
