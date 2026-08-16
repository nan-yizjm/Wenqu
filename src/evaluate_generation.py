"""运行真实模型生成评测，并保存可复查报告。"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .answer import RAGAnswerer, create_client
from .generation_metrics import evaluate_answer


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
            "sources": [
                {
                    "label": source.label,
                    "source_file": source.source_file,
                    "heading_path": source.heading_path,
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
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "citation_required": len(citation_required),
        "citation_valid": citation_valid,
        "citation_valid_rate": (
            round(citation_valid / len(citation_required), 4)
            if citation_required
            else 0.0
        ),
        "ood_total": len(ood_records),
        "ood_rejected": ood_rejected,
        "ood_rejection_rate": (
            round(ood_rejected / len(ood_records), 4)
            if ood_records
            else 0.0
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
    print(
        "引用有效率："
        f"{summary['citation_valid_rate']:.1%} "
        f"({summary['citation_valid']}/{summary['citation_required']})"
    )
    print(
        "OOD 拒绝率："
        f"{summary['ood_rejection_rate']:.1%} "
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
    args = parser.parse_args()

    eval_file = resolve_project_path(args.eval_file)
    cases = load_cases(eval_file)

    if args.limit is not None:
        cases = cases[: args.limit]

    chunks_path = PROJECT_ROOT / "data" / "generated" / "chunks.json"

    answerer = RAGAnswerer(
        chunks_path=chunks_path,
        client=create_client(args.provider),
    )

    records = [
        evaluate_case(answerer, case)
        for case in cases
    ]
    summary = build_summary(records)

    report = {
        "provider": args.provider,
        "eval_file": str(eval_file),
        "summary": summary,
        "records": records,
    }

    output_file = (
        resolve_project_path(args.output_file)
        if args.output_file
        else PROJECT_ROOT
        / "data"
        / "generated"
        / f"generation_eval_{args.provider}.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_summary(summary)
    print(f"\n详细报告：{output_file}")


if __name__ == "__main__":
    main()