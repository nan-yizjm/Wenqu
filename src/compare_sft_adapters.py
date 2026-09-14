"""严格核对同基座 QLoRA 前后报告，并列出逐题改善与退化。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256

SCHEMA = "hf-rag-sft-challenge-report-v1"
INVARIANTS = (
    "model_name",
    "model_revision",
    "quantization",
    "challenge_sha256",
    "evaluator_sha256",
    "generation",
)


def load_report(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or not isinstance(value.get("records"), list):
        raise ValueError(f"不是同基座 SFT 挑战报告：{path}")
    ids = [record.get("id") for record in value["records"]]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError(f"报告题目 ID 缺失或重复：{path}")
    return path, value


def compare(before_path, after_path, output=None):
    before_path, before = load_report(before_path)
    after_path, after = load_report(after_path)
    mismatches = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in INVARIANTS if before.get(name) != after.get(name)
    }
    before_by_id = {record["id"]: record for record in before["records"]}
    after_by_id = {record["id"]: record for record in after["records"]}
    if set(before_by_id) != set(after_by_id):
        mismatches["case_ids"] = {
            "only_before": sorted(set(before_by_id) - set(after_by_id)),
            "only_after": sorted(set(after_by_id) - set(before_by_id)),
        }
    if mismatches:
        raise ValueError(
            "前后报告不是受控对比：" + json.dumps(mismatches, ensure_ascii=False, default=str)
        )
    if before.get("adapter") is not None or after.get("adapter") is None:
        raise ValueError("before 必须是无 adapter 基线，after 必须加载 adapter")

    transitions = []
    for case_id in before_by_id:
        old = before_by_id[case_id]
        new = after_by_id[case_id]
        if not old["passed"] and new["passed"]:
            status = "fixed"
        elif old["passed"] and not new["passed"]:
            status = "regressed"
        elif old["passed"]:
            status = "kept_pass"
        else:
            status = "kept_fail"
        transitions.append({
            "id": case_id,
            "category": old["category"],
            "status": status,
            "before_answer": old.get("answer", ""),
            "after_answer": new.get("answer", ""),
            "before_checks": old.get("checks", {}),
            "after_checks": new.get("checks", {}),
        })
    counts = {
        status: sum(item["status"] == status for item in transitions)
        for status in ("fixed", "regressed", "kept_pass", "kept_fail")
    }
    categories = {}
    for category in sorted({item["category"] for item in transitions}):
        items = [item for item in transitions if item["category"] == category]
        categories[category] = {
            "before_passed": sum(before_by_id[item["id"]]["passed"] for item in items),
            "after_passed": sum(after_by_id[item["id"]]["passed"] for item in items),
            "total": len(items),
        }
    result = {
        "schema": "controlled-sft-adapter-comparison-v1",
        "before": {"path": str(before_path), "sha256": sha256(before_path),
                   "pass_rate": before["summary"]["pass_rate"]},
        "after": {"path": str(after_path), "sha256": sha256(after_path),
                  "pass_rate": after["summary"]["pass_rate"], "adapter": after["adapter"],
                  "adapter_artifacts": after.get("adapter_artifacts")},
        "controlled_invariants": {name: before.get(name) for name in INVARIANTS},
        "pass_rate_delta": after["summary"]["pass_rate"] - before["summary"]["pass_rate"],
        "transition_counts": counts,
        "by_category": categories,
        "transitions": transitions,
        "source_sha256": sha256(Path(__file__)),
    }
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated" / (
        "sft_adapter_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, result)
    print(f"通过率：{before['summary']['pass_rate']:.1%} -> {after['summary']['pass_rate']:.1%} "
          f"({result['pass_rate_delta']:+.1%})")
    print("变化：" + json.dumps(counts, ensure_ascii=False))
    for category, values in categories.items():
        print(f"- {category}: {values['before_passed']}/{values['total']} -> "
              f"{values['after_passed']}/{values['total']}")
    changed = [item for item in transitions if item["status"] in ("fixed", "regressed")]
    print("逐题变化：" + (", ".join(f"{item['id']}={item['status']}" for item in changed) or "无"))
    print(f"报告：{output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compare(args.before, args.after, args.output)


if __name__ == "__main__":
    main()
