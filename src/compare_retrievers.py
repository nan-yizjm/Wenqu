"""在同一语料和原始题集上比较三种检索；不调用付费模型。"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from .hybrid_retrieve import RetrievalEngine
from .query_guard import static_corpus_rejection_reason
from .retrieve import load_chunks

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_revision() -> dict[str, str]:
    # HEAD 不包含未提交的改动，因此额外保存实际源码指纹。
    return {path.name: sha256_file(path) for path in sorted((ROOT / "src").glob("*.py"))}


def fraction(numerator: int | float, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize(records: list[dict], top_k: int) -> dict:
    inside = [r for r in records if not r["ood"]]
    outside = [r for r in records if r["ood"]]
    ranks = [r["first_source_rank"] for r in inside]
    return {
        "in_domain_count": len(inside), "ood_count": len(outside),
        "source_hit_at_1": fraction(sum(rank == 1 for rank in ranks), len(inside)),
        f"source_hit_at_{top_k}": fraction(sum(rank is not None for rank in ranks), len(inside)),
        f"mrr_at_{top_k}": fraction(sum(1 / rank for rank in ranks if rank is not None), len(inside)),
        "in_domain_rejected": sum(r["rejected"] for r in inside),
        "ood_rejection_rate": fraction(sum(r["rejected"] for r in outside), len(outside)),
        "average_in_domain_query_ms": fraction(sum(r["elapsed_ms"] for r in inside), len(inside)),
        "failed_ids": [r["id"] for r in records if (
            (r["ood"] and not r["rejected"]) or (not r["ood"] and r["first_source_rank"] is None)
        )],
    }


def evaluate_case(engine: RetrievalEngine, case: dict, method: str, top_k: int) -> dict:
    expected = {name.replace("\\", "/") for name in case.get("expected_source_files", [])}
    started = time.perf_counter()
    reason = static_corpus_rejection_reason(case["question"])
    hits = [] if reason else engine.search(case["question"], method, top_k)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    first_rank = next((rank for rank, hit in enumerate(hits, 1)
                       if hit.chunk["source_file"].replace("\\", "/") in expected), None)
    return {
        "id": case["id"], "question": case["question"], "ood": not expected,
        "expected_source_files": sorted(expected),
        "rejected": bool(reason), "rejection_reason": reason,
        "first_source_rank": first_rank, "elapsed_ms": elapsed_ms,
        "rerank_stats": engine.last_rerank_stats if not reason else {},
        "hits": [{
            "chunk_id": hit.chunk["id"], "source_file": hit.chunk["source_file"],
            "heading_path": hit.chunk["heading_path"], "score": hit.score,
            "raw_score": hit.raw_score, "ranks": hit.ranks,
            "channel_scores": hit.channel_scores, "contributions": hit.contributions,
            "quality_multiplier": hit.quality_multiplier, "quality_reason": hit.quality_reason,
            "retrieval_score": hit.retrieval_score, "pre_rerank_rank": hit.pre_rerank_rank,
            "rerank_score": hit.rerank_score, "rerank_input_tokens": hit.rerank_input_tokens,
            "rerank_truncated": hit.rerank_truncated,
        } for hit in hits],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-files", nargs="+", default=["data/eval_set.json", "data/eval_holdout.json"])
    parser.add_argument("--methods", nargs="+", choices=("bm25", "vector", "hybrid"), default=["bm25", "vector", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-quality-rules", action="store_true")
    parser.add_argument("--chunks-file", default="data/generated/chunks.json")
    parser.add_argument("--include-rerank", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--rerank-device", choices=("cuda", "cpu"), default=None)
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.candidate_k:
        parser.error("要求 1 <= top_k <= candidate_k")
    started = time.perf_counter()
    chunks_path = ROOT / args.chunks_file
    engine = RetrievalEngine(load_chunks(chunks_path), device=args.device,
                             candidate_k=args.candidate_k, rrf_k=args.rrf_k,
                             quality_rules=not args.no_quality_rules,
                             cache_path=chunks_path.parent / "vector_index.npz",
                             rerank=args.include_rerank, rerank_top_n=args.rerank_top_n,
                             rerank_device=args.rerank_device)
    if any(method != "bm25" for method in args.methods):
        engine.ensure_vector()
        engine.vector.search("预热查询", 1)
    if args.include_rerank:
        engine.ensure_reranker().score("预热查询", engine.chunks[:1])
    setup_seconds = round(time.perf_counter() - started, 3)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config": vars(args), "setup_seconds": setup_seconds,
        "chunks_sha256": sha256_file(chunks_path), "source_sha256": source_revision(),
        "vector_metadata": engine.vector.metadata if engine.vector else None,
        "vector_cache_status": engine.vector.cache_status if engine.vector else None,
        "truncated_chunks": engine.vector.truncated_chunk_count if engine.vector else None,
        "reranker_metadata": engine.reranker.metadata if engine.reranker else None,
        "metric_definition": "Source Hit@K: top K 命中任一标注来源文件；不验证具体片段是否含答案。MRR 截断于 K。",
        "datasets": {},
    }
    for name in args.eval_files:
        path = ROOT / name
        cases = json.loads(path.read_text(encoding="utf-8"))
        dataset = {"eval_sha256": sha256_file(path), "methods": {}, "rank_differences": []}
        variants = [(method, False) for method in dict.fromkeys(args.methods)]
        if args.include_rerank:
            variants += [(method, True) for method in dict.fromkeys(args.methods)]
        for method, rerank_enabled in variants:
            engine.rerank_enabled = rerank_enabled
            records = [evaluate_case(engine, case, method, args.top_k) for case in cases]
            summary = summarize(records, args.top_k)
            name_of_method = method + ("+rerank" if rerank_enabled else "")
            dataset["methods"][name_of_method] = {"summary": summary, "records": records}
            print(f"{path.name} | {name_of_method}: {json.dumps(summary, ensure_ascii=False)}", flush=True)
        for position, case in enumerate(cases):
            if not case.get("expected_source_files"):
                continue
            ranks = {method: data["records"][position]["first_source_rank"]
                     for method, data in dataset["methods"].items()}
            if len(set(ranks.values())) > 1:
                dataset["rank_differences"].append({"id": case["id"], "question": case["question"], "ranks": ranks})
        report["datasets"][name] = dataset
    output = ROOT / "data/generated" / f"retrieval_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\n报告：{output}\n模型加载/索引初始化/预热：{setup_seconds} 秒（不计入单题延迟）")


if __name__ == "__main__":
    main()
