from pathlib import Path

from bm25 import BM25Index
from evaluate import (
    evaluate_case,
    get_expected_rank,
    load_eval_set,
)
from retrieve import build_idf, load_chunks


TOP_K = 5


def format_rank(rank: int | None) -> str:
    """把排名格式化为容易阅读的文本。"""
    return str(rank) if rank is not None else "未召回"


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    chunks_path = project_dir / "data" / "generated" / "chunks.json"
    eval_path = project_dir / "data" / "eval_set.json"

    chunks = load_chunks(chunks_path)
    eval_set = load_eval_set(eval_path)

    idf = build_idf(chunks)
    bm25_index = BM25Index(chunks)

    bm25_reports = [
        evaluate_case(
            case,
            chunks,
            idf,
            bm25_index,
            retrieval_method="bm25",
            top_k=TOP_K,
        )
        for case in eval_set
    ]

    multi_query_reports = [
        evaluate_case(
            case,
            chunks,
            idf,
            bm25_index,
            retrieval_method="multi_query",
            top_k=TOP_K,
        )
        for case in eval_set
    ]

    improved = 0
    worsened = 0
    unchanged = 0

    print("BM25 与 Multi-Query 排名差异")
    print("=" * 88)

    for bm25_report, multi_query_report in zip(
        bm25_reports,
        multi_query_reports,
    ):
        if bm25_report["evaluation_type"] != "in_domain":
            continue

        bm25_rank = get_expected_rank(bm25_report)
        multi_query_rank = get_expected_rank(multi_query_report)

        if bm25_rank == multi_query_rank:
            unchanged += 1
            continue

        if bm25_rank is None:
            improved += 1
            change = "改善：未召回 → 已召回"
        elif multi_query_rank is None:
            worsened += 1
            change = "退化：已召回 → 未召回"
        elif multi_query_rank < bm25_rank:
            improved += 1
            change = "改善：排名上升"
        else:
            worsened += 1
            change = "退化：排名下降"

        print(f"\n{bm25_report['id']} | {change}")
        print(f"问题：{bm25_report['question']}")
        print(f"BM25 排名：{format_rank(bm25_rank)}")
        print(f"Multi-Query 排名：{format_rank(multi_query_rank)}")

    print("\n" + "=" * 88)
    print("差异汇总")
    print(f"改善题数：{improved}")
    print(f"退化题数：{worsened}")
    print(f"排名不变题数：{unchanged}")


if __name__ == "__main__":
    main()