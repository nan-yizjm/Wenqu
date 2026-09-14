"""在同一模型与索引上比较真实 prompt token 预算、证据保留和生成结果。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

from .adapter_runtime import AdapterRAGRuntime
from .config import PROJECT_ROOT
from .experiment_utils import sha256
from .settings import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--question", default="PagedAttention 是什么？它解决什么问题？")
    parser.add_argument("--budgets", nargs="+", type=int, default=[2048, 1024, 768, 512])
    parser.add_argument("--generate-budgets", nargs="+", type=int, default=[2048, 512])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    budgets = list(dict.fromkeys(args.budgets))
    if any(budget < 32 for budget in budgets):
        raise ValueError("所有 budget 必须至少为 32")
    if not set(args.generate_budgets) <= set(budgets):
        raise ValueError("generate-budgets 必须是 budgets 的子集")

    runtime = AdapterRAGRuntime(
        load_settings(args.config), args.adapter,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=max(budgets),
    )
    answerer = runtime.answerer
    records = []
    for budget in budgets:
        answerer.max_prompt_tokens = budget
        started = time.perf_counter()
        prepared = answerer.prepare(args.question)
        prepare_ms = (time.perf_counter() - started) * 1000
        record = {
            "max_prompt_tokens": budget,
            "prepare_ms": round(prepare_ms, 2),
            "prompt_tokens": prepared.prompt_tokens,
            "context_char_budget": prepared.context_char_budget,
            "context_chars": len(prepared.context_text),
            "immediate": prepared.immediate_result is not None,
            "sources": [asdict(source) for source in prepared.sources],
            "context_diagnostics": list(prepared.context_diagnostics),
            "context": prepared.context_text,
        }
        if budget in args.generate_budgets and prepared.immediate_result is None:
            answer = answerer.client.chat(list(prepared.messages))
            result = answerer.finalize(prepared, answer)
            record.update({
                "generated": True,
                "answer": answer,
                "citation_validation": asdict(result.citation_validation),
                "model_metrics": dict(answerer.client.last_metrics),
            })
        else:
            record["generated"] = False
            if prepared.immediate_result is not None:
                record["immediate_answer"] = prepared.immediate_result.answer
        records.append(record)

    available = [record for record in records if not record["immediate"]]
    generated = [record for record in records if record["generated"]]
    checks = {
        "all_available_prompts_within_budget": all(
            record["prompt_tokens"] <= record["max_prompt_tokens"] for record in available
        ),
        "full_budget_keeps_configured_context": (
            records[0]["context_char_budget"] == runtime.settings.max_context_chars
        ),
        "at_least_one_budget_repacked": any(
            not record["immediate"]
            and record["context_char_budget"] < runtime.settings.max_context_chars
            for record in records
        ),
        "generated_answers_have_valid_labels": all(
            not record["citation_validation"]["invalid_labels"]
            and bool(record["citation_validation"]["cited_labels"])
            for record in generated
        ),
        "model_prompt_count_matches_prepare": all(
            record["model_metrics"]["prompt_tokens"] == record["prompt_tokens"]
            for record in generated
        ),
    }
    report = {
        "schema": "real-prompt-token-budget-lab-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "question": args.question,
        "service": runtime.info(),
        "parameters": {
            "budgets": budgets,
            "generate_budgets": args.generate_budgets,
            "max_new_tokens": args.max_new_tokens,
        },
        "records": records,
        "checks": checks,
        "passed": all(checks.values()),
        "source_signature": {
            "src/token_budget_lab.py": sha256(Path(__file__)),
            "src/answer.py": sha256(PROJECT_ROOT / "src/answer.py"),
            "src/adapter_runtime.py": sha256(PROJECT_ROOT / "src/adapter_runtime.py"),
            "src/hf_qlora_client.py": sha256(PROJECT_ROOT / "src/hf_qlora_client.py"),
        },
        "interpretation": (
            "字符预算只决定证据打包上限；最终是否可生成由模型 chat template 的真实"
            "prompt token 数判定。低预算会减少完整证据块，而不是对 token 序列静默截断。"
        ),
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "token_budget_lab_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = [{
        "budget": record["max_prompt_tokens"],
        "prompt_tokens": record["prompt_tokens"],
        "context_char_budget": record["context_char_budget"],
        "context_chars": record["context_chars"],
        "sources": len(record["sources"]),
        "immediate": record["immediate"],
        "generated": record["generated"],
    } for record in records]
    print(json.dumps({"summary": summary, "checks": checks, "report": str(output)},
                     ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
