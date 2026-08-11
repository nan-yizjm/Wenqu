from dataclasses import dataclass, field

try:
    from .bm25 import BM25Index, search_bm25
except ImportError:
    from bm25 import BM25Index, search_bm25


RRF_K = 60


@dataclass
class FusedResult:
    """保存多子查询合并后的检索结果。"""

    chunk: dict[str, str]
    score: float = 0.0
    matched_tokens: set[str] = field(default_factory=set)


def split_subject_list(text: str) -> list[str]:
    """把“RAG、Agent 和多模态能力”拆成多个概念。"""
    normalized_text = text

    for separator in ("、", "和", "与", "以及"):
        normalized_text = normalized_text.replace(separator, "|")

    return [
        part.strip()
        for part in normalized_text.split("|")
        if part.strip()
    ]


def decompose_query(query: str) -> list[str]:
    """只拆分包含“分别”的列举型问题；其他问题保持原样。"""
    if "分别" not in query:
        return [query]

    before, after = query.split("分别", maxsplit=1)

    if "在" in before:
        subjects_text, context = before.split("在", maxsplit=1)
        shared_suffix = f"在{context} {after}".strip()
    else:
        subjects_text = before
        shared_suffix = after.strip()

    subjects = split_subject_list(subjects_text)

    if len(subjects) < 2:
        return [query]

    return [
        f"{subject} {shared_suffix}".strip()
        for subject in subjects
    ]


def search_multi_query(
    query: str,
    chunks: list[dict[str, str]],
    bm25_index: BM25Index,
    top_k: int = 5,
    per_query_k: int = 3,
) -> list[tuple[dict[str, str], float, set[str]]]:
    """拆分复杂问题、分别检索，并用 RRF 合并结果。"""
    subqueries = decompose_query(query)

    if len(subqueries) == 1:
        return search_bm25(
            query,
            chunks,
            bm25_index,
            top_k=top_k,
        )

    fused_results: dict[str, FusedResult] = {}

    for subquery in subqueries:
        partial_results = search_bm25(
            subquery,
            chunks,
            bm25_index,
            top_k=per_query_k,
        )

        for rank, (chunk, _, matched_tokens) in enumerate(
            partial_results,
            start=1,
        ):
            chunk_id = chunk["id"]

            if chunk_id not in fused_results:
                fused_results[chunk_id] = FusedResult(chunk=chunk)

            fused_result = fused_results[chunk_id]

            # Reciprocal Rank Fusion:
            # 排名越靠前，贡献越大；不同子查询的原始分数无需直接比较。
            fused_result.score += 1 / (RRF_K + rank)
            fused_result.matched_tokens.update(matched_tokens)

    results = [
        (
            fused_result.chunk,
            fused_result.score,
            fused_result.matched_tokens,
        )
        for fused_result in fused_results.values()
    ]

    results.sort(key=lambda item: (-item[1], item[0]["id"]))

    return results[:top_k]