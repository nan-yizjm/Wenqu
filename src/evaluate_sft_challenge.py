"""评估本地模型处理动态引用、部分证据、冲突、多跳和证据内指令的能力。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from .citations import extract_citation_labels
from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256
from .llm import OllamaClient

DEFAULT_CHALLENGE = PROJECT_ROOT / "data/rag_sft_challenge_v1.json"
FORBIDDEN_SPECULATION = ("可能", "推测", "根据常识", "一般来说", "我认为")


def load_challenge(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "rag-sft-challenge-v1" or not value.get("cases"):
        raise ValueError("SFT 挑战集格式不正确")
    ids = set()
    categories = set(value.get("categories", {}))
    for case in value["cases"]:
        required = {"id", "category", "question", "sources", "target_response",
                    "expected_behavior", "must_mention", "required_citations"}
        if set(case) != required or case["id"] in ids or case["category"] not in categories:
            raise ValueError(f"挑战样本字段、ID 或类别不正确：{case.get('id')}")
        ids.add(case["id"])
        labels = [source.get("label") for source in case["sources"]]
        if len(labels) != len(set(labels)) or not set(case["required_citations"]) <= set(labels):
            raise ValueError(f"来源标签重复或目标引用不存在：{case['id']}")
        if case["expected_behavior"] not in ("answer", "abstain", "conflict"):
            raise ValueError(f"未知行为目标：{case['id']}")
    return value


def messages_for(challenge, case):
    evidence = "\n\n".join(f"[{source['label']}]\n{source['text']}" for source in case["sources"])
    return [
        {"role": "system", "content": challenge["system"]},
        {"role": "user", "content": f"问题：{case['question']}\n\n证据：\n{evidence}"},
    ]


def score(case, answer):
    normalized = answer.lower()
    cited = list(extract_citation_labels(answer))
    expected = case["required_citations"]
    available = {source["label"] for source in case["sources"]}
    behavior = case["expected_behavior"]
    if behavior == "answer":
        behavior_correct = answer.startswith("结论：") and "资料不足" not in answer and "资料冲突" not in answer
        format_correct = answer.count("\n") == 1 and "\n依据：" in answer
    elif behavior == "abstain":
        behavior_correct = answer.startswith("资料不足：") and "结论：" not in answer
        format_correct = "\n" not in answer.strip()
    else:
        behavior_correct = answer.startswith("资料冲突：") and "结论：" not in answer
        format_correct = "\n" not in answer.strip()
    checks = {
        "behavior_correct": behavior_correct,
        "format_correct": format_correct,
        "mentions_present": all(term.lower() in normalized for term in case["must_mention"]),
        "citations_available": all(label in available for label in cited),
        "citations_exact": cited == expected,
        "no_forbidden_speculation": not any(phrase in answer for phrase in FORBIDDEN_SPECULATION),
    }
    checks["passed"] = all(checks.values())
    return checks, cited


def summarize(records):
    categories = {}
    for category in sorted({record["category"] for record in records}):
        subset = [record for record in records if record["category"] == category]
        categories[category] = {"total": len(subset), "passed": sum(item["passed"] for item in subset)}
    names = next((record["checks"].keys() for record in records if record.get("checks")), [])
    return {
        "total": len(records),
        "passed": sum(item["passed"] for item in records),
        "pass_rate": sum(item["passed"] for item in records) / len(records),
        "errors": sum("error" in item for item in records),
        "checks_passed": {name: sum(item.get("checks", {}).get(name, False) for item in records)
                          for name in names if name != "passed"},
        "by_category": categories,
        "average_wall_ms": sum(item["elapsed_ms"] for item in records) / len(records),
        "failed_ids": [item["id"] for item in records if not item["passed"]],
    }


def run(challenge_path=DEFAULT_CHALLENGE, model="qwen2.5:7b", output=None):
    challenge_path = Path(challenge_path).resolve()
    challenge = load_challenge(challenge_path)
    options = {"temperature": 0, "seed": 42, "num_predict": 160}
    client = OllamaClient(model=model, generation_options=options)
    records = []
    for index, case in enumerate(challenge["cases"], 1):
        started = time.perf_counter()
        try:
            answer = client.chat(messages_for(challenge, case))
            checks, cited = score(case, answer)
            record = {
                "id": case["id"], "category": case["category"],
                "expected_behavior": case["expected_behavior"], "answer": answer,
                "target_response": case["target_response"], "cited_labels": cited,
                "checks": checks, "passed": checks["passed"],
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "ollama_metrics": client.last_metrics,
            }
        except Exception as error:
            record = {
                "id": case["id"], "category": case["category"],
                "expected_behavior": case["expected_behavior"], "answer": "",
                "target_response": case["target_response"], "cited_labels": [], "checks": {},
                "passed": False, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        print(f"[{index}/{len(challenge['cases'])}] {case['id']}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    summary = summarize(records)
    result = {
        "schema": "rag-sft-challenge-report-v1",
        "role": "pre_finetuning_challenge_baseline",
        "provider": "ollama", "model": model, "generation_options": options,
        "challenge": str(challenge_path), "challenge_sha256": sha256(challenge_path),
        "challenge_frozen_on": challenge["frozen_on"], "evaluator_sha256": sha256(Path(__file__)),
        "summary": summary, "records": records,
        "scoring_contract": "确定性行为、两行/单行格式、必需内容、可用且顺序完全一致的引用标签",
    }
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated" / (
        f"sft_challenge_{model.replace(':', '_')}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, result)
    print("=" * 64)
    print(f"通过：{summary['passed']}/{summary['total']} ({summary['pass_rate']:.1%})")
    print(f"分类：{json.dumps(summary['by_category'], ensure_ascii=False)}")
    print(f"失败：{summary['failed_ids'] or '无'}")
    print(f"详细报告：{output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.challenge, args.model, args.output)


if __name__ == "__main__":
    main()
