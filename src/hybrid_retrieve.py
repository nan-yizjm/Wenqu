"""统一检索入口；RRF 只融合名次，质量规则仅在最终排序应用一次。"""

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

from .bm25 import BM25Index, get_quality_adjustment, search_bm25
from .query_guard import static_corpus_rejection_reason
from .retrieve import load_chunks, tokenize_query


@dataclass
class SearchHit:
    chunk: dict[str, str]
    score: float = 0.0
    raw_score: float = 0.0
    quality_multiplier: float = 1.0
    quality_reason: str | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    matched_tokens: set[str] = field(default_factory=set)
    retrieval_score: float | None = None
    pre_rerank_rank: int | None = None
    rerank_score: float | None = None
    rerank_input_tokens: int | None = None
    rerank_truncated: bool = False

    def as_result(self):
        return self.chunk, self.score, self.matched_tokens


def fuse_rankings(rankings: dict[str, list], rrf_k: int = 60) -> list[SearchHit]:
    """每一路中同一 chunk 只投票一次；名次从 1 开始。"""
    if rrf_k < 1:
        raise ValueError("rrf_k 必须大于 0")
    fused: dict[str, SearchHit] = {}
    for channel, results in rankings.items():
        seen: set[str] = set()
        for rank, (chunk, score, tokens) in enumerate(results, 1):
            chunk_id = chunk["id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            hit = fused.setdefault(chunk_id, SearchHit(chunk=chunk))
            hit.ranks[channel] = rank
            hit.channel_scores[channel] = score
            contribution = 1.0 / (rrf_k + rank)
            hit.contributions[channel] = contribution
            hit.raw_score += contribution
            hit.matched_tokens.update(tokens)
    return sorted(fused.values(), key=lambda hit: (-hit.raw_score, hit.chunk["id"]))


class RetrievalEngine:
    def __init__(
        self, chunks: list[dict[str, str]], device: str = "cuda",
        candidate_k: int = 20, rrf_k: int = 60, quality_rules: bool = True,
        cache_path: Path | None = None, vector_index=None,
        rerank: bool = False, rerank_top_n: int = 20, reranker=None,
        rerank_device: str | None = None,
    ) -> None:
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("candidate_k 与 rrf_k 必须为正数")
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.chunk_tokens = {chunk["id"]: set(frequencies) for chunk, frequencies in
                             zip(chunks, self.bm25.term_frequencies, strict=True)}
        self.vector = vector_index
        self.device = device
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.quality_rules = quality_rules
        self.cache_path = cache_path
        if rerank_top_n < 1 or (rerank and rerank_top_n > candidate_k):
            raise ValueError("要求 1 <= rerank_top_n <= candidate_k")
        self.rerank_enabled = rerank
        self.rerank_top_n = rerank_top_n
        self.reranker = reranker
        self.rerank_device = rerank_device or device
        self.last_rerank_stats = {}

    def ensure_reranker(self):
        if self.reranker is None:
            from .reranker import LocalReranker
            self.reranker = LocalReranker(device=self.rerank_device)
        return self.reranker

    def ensure_vector(self):
        # 词法检索、空查询与被守卫拦截的问题不加载 GPU 模型。
        if self.vector is None:
            from .vector_retrieve import DEFAULT_CACHE, VectorIndex
            self.vector = VectorIndex(
                self.chunks, device=self.device,
                cache_path=self.cache_path if self.cache_path is not None else DEFAULT_CACHE,
            )
        return self.vector

    def search(self, query: str, method: str = "bm25", top_k: int = 5) -> list[SearchHit]:
        self.last_rerank_stats = {}
        if method not in ("bm25", "vector", "hybrid"):
            raise ValueError(f"未知检索方式：{method}")
        if top_k < 1 or self.candidate_k < top_k:
            raise ValueError("要求 1 <= top_k <= candidate_k")
        if self.rerank_enabled and self.rerank_top_n < top_k:
            raise ValueError("重排序候选数不能小于最终 top_k")
        if not query.strip() or not self.chunks:
            return []
        count = self.candidate_k
        # 单路可在全库打分后降权；混合模式是有限候选窗口的融合。
        if method != "hybrid":
            count = len(self.chunks)
        rankings = {}
        if method in ("bm25", "hybrid"):
            rankings["bm25"] = search_bm25(
                query, self.chunks, self.bm25, top_k=count, raw=True,
            )
        if method in ("vector", "hybrid"):
            query_tokens = tokenize_query(query)
            rankings["vector"] = [
                (chunk, score, query_tokens & self.chunk_tokens[chunk["id"]])
                for chunk, score in self.ensure_vector().search(query, top_k=count)
            ]
        if method == "hybrid":
            hits = fuse_rankings(rankings, self.rrf_k)
        else:
            hits = [SearchHit(
                chunk=chunk, raw_score=score, ranks={method: rank},
                channel_scores={method: score}, matched_tokens=tokens,
            ) for rank, (chunk, score, tokens) in enumerate(rankings[method], 1)]
        for hit in hits:
            if self.quality_rules:
                hit.quality_multiplier, hit.quality_reason = get_quality_adjustment(hit.chunk)
            # 避免负余弦乘 0.35 反而升高；负分保持原分，不奖励低信息片段。
            hit.score = hit.raw_score * hit.quality_multiplier if hit.raw_score > 0 else hit.raw_score
        hits.sort(key=lambda hit: (-hit.score, hit.chunk["id"]))
        if self.rerank_enabled and hits:
            candidates = hits[:self.rerank_top_n]
            reranker = self.ensure_reranker()
            values = reranker.score(query, [hit.chunk for hit in candidates])
            self.last_rerank_stats = {
                "candidate_count": len(values),
                "truncated_pairs": sum(count > reranker.max_length for _, count in values),
                "max_pair_tokens": max((count for _, count in values), default=0),
            }
            hits = [replace(hit, retrieval_score=hit.score, pre_rerank_rank=rank,
                            score=score, rerank_score=score, rerank_input_tokens=count,
                            rerank_truncated=count > reranker.max_length)
                    for rank, (hit, (score, count)) in enumerate(zip(candidates, values, strict=True), 1)]
            # 重排 logit 不再乘质量系数；否则负值会被意外抬高，且重复惩罚。
            hits.sort(key=lambda hit: (-hit.score, hit.pre_rerank_rank, hit.chunk["id"]))
        return hits[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 / E5 / RRF 排序及各通道贡献。")
    parser.add_argument("query")
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-quality-rules", action="store_true")
    parser.add_argument("--chunks-file", default="data/generated/chunks.json")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--rerank-device", choices=("cuda", "cpu"), default=None)
    args = parser.parse_args()
    reason = static_corpus_rejection_reason(args.query)
    if reason:
        print(f"能力守卫拒绝：{reason}")
        return
    root = Path(__file__).resolve().parent.parent
    chunks_path = root / args.chunks_file
    engine = RetrievalEngine(
        load_chunks(chunks_path), device=args.device,
        candidate_k=args.candidate_k, rrf_k=args.rrf_k,
        quality_rules=not args.no_quality_rules,
        cache_path=chunks_path.parent / "vector_index.npz", rerank=args.rerank,
        rerank_top_n=args.rerank_top_n, rerank_device=args.rerank_device,
    )
    hits = engine.search(args.query, args.method, args.top_k)
    if engine.vector is not None:
        print(f"向量缓存：{engine.vector.cache_status}")
    for rank, hit in enumerate(hits, 1):
        print(f"\n[{rank}] final={hit.score:.6f}, raw={hit.raw_score:.6f}")
        print(f"{hit.chunk['source_file']} → {hit.chunk['heading_path']}")
        print(f"通道名次：{hit.ranks}；通道原分：{hit.channel_scores}")
        print(f"RRF 贡献：{hit.contributions}；质量系数：{hit.quality_multiplier}")
        if hit.rerank_score is not None:
            print(f"重排前名次：{hit.pre_rerank_rank}；前分数：{hit.retrieval_score:.6f}；重排 logit：{hit.rerank_score:.4f}")
            print(f"重排联合输入 token：{hit.rerank_input_tokens}；是否截断：{hit.rerank_truncated}")
        if hit.quality_reason:
            print(f"降权原因：{hit.quality_reason}")
        print(hit.chunk["text"][:180].replace("\n", " "))


if __name__ == "__main__":
    main()
