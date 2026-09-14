import json
from pathlib import Path
import argparse

# from retrieve import load_chunks, search_chunks
try:
    from .bm25 import BM25Index, search_bm25
    from .retrieve import build_idf, load_chunks, search_chunks
    from .query_guard import static_corpus_rejection_reason
    from .multi_query import search_multi_query
except ImportError:
    from bm25 import BM25Index, search_bm25
    from retrieve import build_idf, load_chunks, search_chunks
    from query_guard import static_corpus_rejection_reason
    from multi_query import search_multi_query

# TOP_K = 5
# # RETRIEVAL_METHOD = "idf"
# RETRIEVAL_METHOD = "bm25"
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL_METHOD = "bm25"

def parse_args() -> argparse.Namespace:
    """读取命令行中的评测配置。"""
    parser = argparse.ArgumentParser(
        description="评测 Markdown 知识库检索效果。"
    )

    parser.add_argument(
        "--eval-file",
        default="eval_set.json",
        help="data 目录下的评测文件名。",
    )

    parser.add_argument(
        "--method",
        # choices=("idf", "bm25"),
        choices=("idf", "bm25", "multi_query"),
        default=DEFAULT_RETRIEVAL_METHOD,
        help="使用的检索方法。",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="每个问题返回的检索结果数。",
    )

    return parser.parse_args()

def load_eval_set(eval_path: Path) -> list[dict]:
    """读取人工编写的检索评测题集。"""
    if not eval_path.is_file():
        raise FileNotFoundError(f"找不到评测集：{eval_path}")

    return json.loads(eval_path.read_text(encoding="utf-8"))


def normalize_path(path: str) -> str:
    """统一 Windows 与跨平台路径分隔符。"""
    return path.replace("\\", "/")


# def evaluate_case(
#     case: dict,
#     chunks: list[dict[str, str]],
#     top_k: int,
# ) -> dict:
# def evaluate_case(
#     case: dict,
#     chunks: list[dict[str, str]],
#     idf: dict[str, float],
#     top_k: int,
# ) -> dict:
# def evaluate_case(
#     case: dict,
#     chunks: list[dict[str, str]],
#     idf: dict[str, float],
#     bm25_index: BM25Index,
#     top_k: int,
# ) -> dict:
def evaluate_case(
    case: dict,
    chunks: list[dict[str, str]],
    idf: dict[str, float],
    bm25_index: BM25Index,
    retrieval_method: str,
    top_k: int,
) -> dict:
    """评测一条问题：正确来源是否被召回。"""
    question = case["question"]
    expected_sources = {
        normalize_path(source)
        for source in case["expected_source_files"]
    }

    # results = search_chunks(question, chunks, top_k=top_k)
    # results = search_chunks(question, chunks, idf, top_k=top_k)
    # if RETRIEVAL_METHOD == "idf":
    rejection_reason = static_corpus_rejection_reason(question)

    if rejection_reason is not None:
        results = []
    elif retrieval_method == "idf":
        results = search_chunks(
            question,
            chunks,
            idf,
            top_k=top_k,
        )
    elif retrieval_method == "bm25":
        results = search_bm25(
            question,
            chunks,
            bm25_index,
            top_k=top_k,
        )
    elif retrieval_method == "multi_query":
        results = search_multi_query(
            question,
            chunks,
            bm25_index,
            top_k=top_k,
        )
    else:
        raise ValueError(
            f"不支持的检索方法：{retrieval_method}"
        )

    retrieved_sources = {
        normalize_path(chunk["source_file"])
        for chunk, _, _ in results
    }

    matched_sources = sorted(expected_sources & retrieved_sources)

    if expected_sources:
        passed = bool(matched_sources)
        evaluation_type = "in_domain"
    else:
        passed = not retrieved_sources
        evaluation_type = "out_of_domain"

    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "evaluation_type": evaluation_type,
        "passed": passed,
        "expected_sources": sorted(expected_sources),
        "retrieved_sources": sorted(retrieved_sources),
        "matched_sources": matched_sources,
        "results": results,
        "rejection_reason": rejection_reason,
    }

def get_expected_rank(report: dict) -> int | None:
    """返回第一个期望来源在检索结果中的排名。"""
    expected_sources = set(report["expected_sources"])

    for rank, (chunk, _, _) in enumerate(
        report["results"],
        start=1,
    ):
        source_file = normalize_path(chunk["source_file"])

        if source_file in expected_sources:
            return rank

    return None


def print_case_report(report: dict) -> None:
    """打印单条问题的评测结果。"""
    status = "PASS" if report["passed"] else "FAIL"

    print("-" * 70)
    print(f"[{status}] {report['id']} | {report['category']}")
    print(f"问题：{report['question']}")

    if report["evaluation_type"] == "in_domain":
        print(f"期望来源：{report['expected_sources']}")
        print(f"命中来源：{report['matched_sources']}")
    else:
        print("期望行为：知识库外问题，不应召回任何资料。")

    if not report["results"]:
        print("实际结果：没有召回任何片段。")
        if report["rejection_reason"] is not None:
            print(f"拒答原因：{report['rejection_reason']}")
        return

    print("实际 Top-K 来源：")

    for index, (chunk, score, _) in enumerate(report["results"], start=1):
        print(
            f"  [{index}] 分数={score} | "
            f"{normalize_path(chunk['source_file'])} | "
            f"{chunk['heading_path']}"
        )


def print_summary(reports: list[dict], top_k: int) -> None:
    """输出整体评测指标与失败问题。"""
    in_domain_reports = [
        report
        for report in reports
        if report["evaluation_type"] == "in_domain"
    ]
    out_of_domain_reports = [
        report
        for report in reports
        if report["evaluation_type"] == "out_of_domain"
    ]

    in_domain_passed = sum(report["passed"] for report in in_domain_reports)
    out_of_domain_passed = sum(
        report["passed"] for report in out_of_domain_reports
    )

    source_recall = (
        in_domain_passed / len(in_domain_reports)
        if in_domain_reports
        else 0.0
    )
    ranks = [
        get_expected_rank(report)
        for report in in_domain_reports
    ]

    recall_at_1 = sum(
        rank == 1
        for rank in ranks
    ) / len(in_domain_reports) if in_domain_reports else 0.0

    mrr = sum(
        1 / rank if rank is not None else 0.0
        for rank in ranks
    ) / len(in_domain_reports) if in_domain_reports else 0.0
    rejection_rate = (
        out_of_domain_passed / len(out_of_domain_reports)
        if out_of_domain_reports
        else 0.0
    )

    print("=" * 70)
    print("评测汇总")
    print(f"知识库内问题数：{len(in_domain_reports)}")
    print(f"Source Recall@{top_k}：{source_recall:.1%}")
    print(f"Source Recall@1：{recall_at_1:.1%}")
    print(f"MRR：{mrr:.3f}")
    print(f"知识库外问题数：{len(out_of_domain_reports)}")
    print(f"OOD Rejection Rate：{rejection_rate:.1%}")

    failed_reports = [report for report in reports if not report["passed"]]

    print(f"失败问题数：{len(failed_reports)}")

    for report in failed_reports:
        print(f"  - {report['id']}：{report['question']}")


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    chunks_path = project_dir / "data" / "generated" / "chunks.json"
    # eval_path = project_dir / "data" / "eval_set.json"
    eval_path = project_dir / "data" / args.eval_file

    # chunks = load_chunks(chunks_path)
    # eval_set = load_eval_set(eval_path)
    chunks = load_chunks(chunks_path)
    eval_set = load_eval_set(eval_path)

    idf = build_idf(chunks)
    bm25_index = BM25Index(chunks)

    # print(f"检索方法：{RETRIEVAL_METHOD}")
    print(f"检索方法：{args.method}")
    print(f"评测文件：{args.eval_file}")
    print(f"Top-K：{args.top_k}")

    idf = build_idf(chunks)

    # reports = [
    #     # evaluate_case(case, chunks, top_k=TOP_K)
    #     evaluate_case(case, chunks, idf, top_k=TOP_K)
    #     for case in eval_set
    # ]
    # reports = [
    #     evaluate_case(
    #         case,
    #         chunks,
    #         idf,
    #         bm25_index,
    #         top_k=TOP_K,
    #     )
    #     for case in eval_set
    # ]
    reports = [
        evaluate_case(
            case,
            chunks,
            idf,
            bm25_index,
            retrieval_method=args.method,
            top_k=args.top_k,
        )
        for case in eval_set
    ]

    for report in reports:
        print_case_report(report)

    # print_summary(reports, top_k=TOP_K)
    print_summary(reports, top_k=args.top_k)


if __name__ == "__main__":
    main()
