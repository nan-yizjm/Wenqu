"""用本地 Ollama 评估 SFT 前后的证据回答、引用与资料不足行为。"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

from .citations import extract_citation_labels
from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256
from .llm import OllamaClient
from .sft_data import DEFAULT_OUTPUT, load_sft_data

FORBIDDEN_SPECULATION = ("可能", "推测", "根据常识", "一般来说", "我认为")


def score_response(case, answer):
    normalized = answer.lower()
    cited = list(extract_citation_labels(answer))
    expected = case["required_citations"]
    available = set(case["available_citations"])
    behavior = ("资料不足" not in answer if case["expected_behavior"] == "answer"
                else answer.startswith("资料不足：") and "结论：" not in answer)
    format_correct = (answer.startswith("结论：") and "\n依据：" in answer
                      if case["expected_behavior"] == "answer"
                      else answer.startswith("资料不足：") and "\n" not in answer.strip())
    mentions = all(term.lower() in normalized for term in case["must_mention"])
    citations_available = all(label in available for label in cited)
    citations_exact = cited == expected
    no_speculation = not any(phrase in answer for phrase in FORBIDDEN_SPECULATION)
    checks = {
        "behavior_correct": behavior,
        "format_correct": format_correct,
        "mentions_present": mentions,
        "citations_available": citations_available,
        "citations_exact": citations_exact,
        "no_forbidden_speculation": no_speculation,
    }
    checks["passed"] = all(checks.values())
    return checks, cited


def summarize(records):
    categories = {}
    for category in sorted({record["category"] for record in records}):
        subset = [record for record in records if record["category"] == category]
        categories[category] = {"total": len(subset), "passed": sum(item["passed"] for item in subset)}
    check_names = next((record["checks"].keys() for record in records if record.get("checks")), [])
    checks = {name: sum(record.get("checks", {}).get(name, False) for record in records)
              for name in check_names if name != "passed"}
    return {
        "total": len(records),
        "passed": sum(record["passed"] for record in records),
        "pass_rate": sum(record["passed"] for record in records) / len(records) if records else None,
        "errors": sum("error" in record for record in records),
        "checks_passed": checks,
        "by_category": categories,
        "average_wall_ms": (sum(record["elapsed_ms"] for record in records) / len(records)) if records else None,
        "failed_ids": [record["id"] for record in records if not record["passed"]],
    }


def run(data_dir=DEFAULT_OUTPUT, split="test", model="qwen2.5:7b", limit=None, output=None):
    data_dir = Path(data_dir).resolve()
    dataset, manifest, fingerprint = load_sft_data(data_dir)
    cases = dataset[split][:limit]
    if not cases:
        raise ValueError("评估切片为空")
    options = {"temperature": 0, "seed": 42, "num_predict": 128}
    client = OllamaClient(model=model, generation_options=options)
    records = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        try:
            answer = client.chat(case["messages"])
            checks, cited = score_response(case, answer)
            record = {
                "id": case["id"], "group_id": case["group_id"], "category": case["category"],
                "expected_behavior": case["expected_behavior"], "answer": answer,
                "target_response": case["response"], "cited_labels": cited,
                "checks": checks, "passed": checks["passed"],
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "ollama_metrics": client.last_metrics,
            }
        except Exception as error:
            record = {
                "id": case["id"], "group_id": case["group_id"], "category": case["category"],
                "expected_behavior": case["expected_behavior"], "answer": "", "target_response": case["response"],
                "cited_labels": [], "checks": {}, "passed": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        print(f"[{index}/{len(cases)}] {case['id']}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    summary = summarize(records)
    result = {
        "schema": "rag-sft-behavior-baseline-v1",
        "role": "pre_finetuning_baseline",
        "provider": "ollama",
        "model": model,
        "generation_options": options,
        "data_dir": str(data_dir),
        "data_fingerprint": fingerprint,
        "data_artifacts": manifest["artifacts"],
        "split": split,
        "limit": limit,
        "evaluator_sha256": sha256(Path(__file__)),
        "summary": summary,
        "records": records,
        "scoring_contract": {
            "answer": "两行结论/依据，包含预期事实，引用标签顺序与目标完全一致",
            "abstain": "单行以资料不足开头，不输出结论或引用",
            "note": "这是确定性格式/内容检查，不使用 LLM-as-Judge",
        },
    }
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated" / (
        f"sft_baseline_{model.replace(':', '_')}_{split}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, result)
    print("=" * 64)
    print(f"通过：{summary['passed']}/{summary['total']} ({summary['pass_rate']:.1%})")
    print(f"分类：{json.dumps(summary['by_category'], ensure_ascii=False)}")
    print(f"失败：{summary['failed_ids'] or '无'}")
    print(f"详细报告：{output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.split, args.model, args.limit, args.output)


if __name__ == "__main__":
    main()
