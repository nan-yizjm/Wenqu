"""运行真实模型生成评测，并保存可复查报告。"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .answer import RAGAnswerer, create_client
from .generation_metrics import evaluate_answer
from .compare_retrievers import sha256_file, source_revision


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_text: str) -> Path:
    """将相对项目路径转换为绝对路径。"""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cases(eval_file: Path) -> list[dict[str, Any]]:
    """加载并检查生成评测题。"""
    data = json.loads(eval_file.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("生成评测文件必须是 JSON 数组")

    required_fields = {
        "id",
        "question",
        "category",
        "expected_behavior",
        "must_mention",
        "must_cite",
    }

    for case in data:
        missing = required_fields - set(case)
        if missing:
            raise ValueError(
                f"评测题 {case.get('id')} 缺少字段：{sorted(missing)}"
            )

    return data


def evaluate_case(
    answerer: RAGAnswerer,
    case: dict[str, Any],
) -> dict[str, Any]:
    """运行一道题，并保存回答、来源、指标和耗时。"""
    started_at = time.perf_counter()

    try:
        result = answerer.answer(case["question"])
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        checks = evaluate_answer(case, result)

        citation_validation = result.citation_validation

        return {
            "id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "expected_behavior": case["expected_behavior"],
            "answer": result.answer,
            "rejected": result.rejected,
            "retrieval_method": result.retrieval_method,
            "generation_calls": result.generation_calls,
            "separate_coverage_ok": result.separate_coverage_ok,
            "context_text": result.context_text,
            "retrieval_trace": result.retrieval_trace,
            "context_diagnostics": result.context_diagnostics,
            "rerank_stats": answerer.engine.last_rerank_stats if not result.rejected else {},
            "sources": [
                {
                    "label": source.label,
                    "source_file": source.source_file,
                    "heading_path": source.heading_path,
                    "chunk_id": source.chunk_id,
                    "line_start": source.line_start, "line_end": source.line_end,
                }
                for source in result.sources
            ],
            "cited_labels": (
                list(citation_validation.cited_labels)
                if citation_validation
                else []
            ),
            "invalid_labels": (
                list(citation_validation.invalid_labels)
                if citation_validation
                else []
            ),
            "checks": checks,
            "passed": checks["passed"],
            "elapsed_ms": round(elapsed_ms, 2),
        }

    except Exception as error:
        elapsed_ms = (time.perf_counter() - started_at) * 1000

        return {
            "id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "expected_behavior": case["expected_behavior"],
            "answer": "",
            "rejected": False,
            "sources": [],
            "cited_labels": [],
            "invalid_labels": [],
            "checks": {},
            "passed": False,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": f"{type(error).__name__}: {error}",
        }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """从单题记录中计算总体指标。"""
    total = len(records)
    passed = sum(record["passed"] for record in records)

    citation_required = [
        record
        for record in records
        if record["expected_behavior"] != "reject"
        and record.get("checks", {}).get("citation_valid") is not None
    ]

    citation_valid = sum(
        record.get("checks", {}).get("citation_valid", False)
        for record in citation_required
    )

    ood_records = [
        record
        for record in records
        if record["expected_behavior"] == "reject"
    ]

    ood_rejected = sum(record["rejected"] for record in ood_records)

    category_summary: dict[str, dict[str, int]] = {}

    for record in records:
        category = record["category"]
        summary = category_summary.setdefault(
            category,
            {"total": 0, "passed": 0},
        )
        summary["total"] += 1
        summary["passed"] += int(record["passed"])

    return {
        "total": total,
        "error_count": sum("error" in record for record in records),
        "generation_calls": sum(record.get("generation_calls", 0) for record in records),
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "citation_required": len(citation_required),
        "citation_valid": citation_valid,
        "citation_valid_rate": (
            round(citation_valid / len(citation_required), 4)
            if citation_required
            else None
        ),
        "ood_total": len(ood_records),
        "ood_rejected": ood_rejected,
        "ood_rejection_rate": (
            round(ood_rejected / len(ood_records), 4)
            if ood_records
            else None
        ),
        "average_elapsed_ms": (
            round(
                sum(record["elapsed_ms"] for record in records) / total,
                2,
            )
            if total
            else 0.0
        ),
        "by_category": category_summary,
        "failed_ids": [
            record["id"]
            for record in records
            if not record["passed"]
        ],
    }


def print_summary(summary: dict[str, Any]) -> None:
    """打印适合终端阅读的评测汇总。"""
    print("=" * 70)
    print("生成评测汇总")
    print("=" * 70)
    print(f"总题数：{summary['total']}")
    print(f"通过题数：{summary['passed']}")
    print(f"总体通过率：{summary['pass_rate']:.1%}")
    citation_rate = summary['citation_valid_rate']
    ood_rate = summary['ood_rejection_rate']
    print(
        "引用标签有效率："
        f"{format(citation_rate, '.1%') if citation_rate is not None else 'N/A'} "
        f"({summary['citation_valid']}/{summary['citation_required']})"
    )
    print(
        "OOD 拒绝率："
        f"{format(ood_rate, '.1%') if ood_rate is not None else 'N/A'} "
        f"({summary['ood_rejected']}/{summary['ood_total']})"
    )
    print(f"平均耗时：{summary['average_elapsed_ms']:.2f} ms")

    print("\n分类结果：")
    for category, values in summary["by_category"].items():
        print(
            f"- {category}: "
            f"{values['passed']}/{values['total']} 通过"
        )

    if summary["failed_ids"]:
        print("\n失败题目：")
        for case_id in summary["failed_ids"]:
            print(f"- {case_id}")
    else:
        print("\n失败题目：无")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 RAG 生成评测。",
    )
    parser.add_argument(
        "--eval-file",
        default="data/generation_eval.json",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "deepseek"),
        default="ollama",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前 N 道题，用于小规模调试。",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="评测报告输出路径。",
    )
    parser.add_argument("--retriever", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=2200)
    parser.add_argument("--chunks-file", default="data/generated/chunks.json")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--rerank-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--context-policy", choices=("legacy", "blocks"), default="legacy")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正数")
    output_file = (
        resolve_project_path(args.output_file) if args.output_file
        else PROJECT_ROOT / "data/generated" /
        f"generation_eval_{args.provider}_{args.retriever}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    if output_file.exists():
        parser.error("报告文件已存在；请使用新文件名，以免覆盖之前的实验")

    eval_file = resolve_project_path(args.eval_file)
    cases = load_cases(eval_file)

    if args.limit is not None:
        cases = cases[: args.limit]

    chunks_path = resolve_project_path(args.chunks_file)

    source_at_start = source_revision()
    chunks_at_start = sha256_file(chunks_path)
    eval_at_start = sha256_file(eval_file)
    started = time.perf_counter()
    answerer = RAGAnswerer(
        chunks_path=chunks_path,
        client=create_client(args.provider),
        retrieval_method=args.retriever, device=args.device,
        retrieval_top_k=args.top_k, max_context_chars=args.max_context_chars,
        rerank=args.rerank, rerank_top_n=args.rerank_top_n,
        rerank_device=args.rerank_device, context_policy=args.context_policy,
    )
    # 单独记录初始化；避免把 E5 模型加载混入第一道生成题的耗时。
    if args.retriever != "bm25":
        answerer.engine.ensure_vector().search("预热查询", 1)
    if args.rerank:
        answerer.engine.ensure_reranker().score("预热查询", answerer.chunks[:1])
    setup_seconds = round(time.perf_counter() - started, 3)

    records = []
    for index, case in enumerate(cases, 1):
        records.append(evaluate_case(answerer, case))
        print(f"[{index}/{len(cases)}] {case['id']}: {'通过' if records[-1]['passed'] else '未通过'}", flush=True)
    summary = build_summary(records)

    report = {
        "provider": args.provider,
        "model": getattr(answerer.client, "model", None),
        "created_at": datetime.now().astimezone().isoformat(),
        "retriever": args.retriever,
        "setup_seconds": setup_seconds,
        "config": {"top_k": args.top_k, "max_context_chars": args.max_context_chars,
                   "candidate_k": answerer.engine.candidate_k, "rrf_k": answerer.engine.rrf_k,
                   "quality_rules": answerer.engine.quality_rules, "device": args.device,
                   "chunks_file": str(chunks_path), "rerank": args.rerank,
                   "rerank_top_n": args.rerank_top_n, "context_policy": args.context_policy},
        "reranker_metadata": answerer.engine.reranker.metadata if answerer.engine.reranker else None,
        "source_sha256": source_at_start,
        "chunks_sha256": chunks_at_start,
        "eval_sha256": eval_at_start,
        "inputs_changed_during_run": (
            source_revision() != source_at_start or sha256_file(chunks_path) != chunks_at_start
            or sha256_file(eval_file) != eval_at_start
        ),
        "vector_metadata": answerer.engine.vector.metadata if answerer.engine.vector else None,
        "eval_file": str(eval_file),
        "summary": summary,
        "records": records,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\n详细报告：{output_file}")


if __name__ == "__main__":
    main()
