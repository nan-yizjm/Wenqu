"""生成有明确行为目标的 RAG SFT 数据：有证据则回答并引用，证据不足则拒答。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256

DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/rag_sft_v1"
SYSTEM_PROMPT = """你是一个只依据给定证据回答问题的知识库助手。
有充分证据时，严格输出两行：
结论：<只陈述证据支持的答案>
依据：<使用 [S1] 形式列出支持该结论的来源>
证据不足时，只输出：
资料不足：给定证据无法回答这个问题。
不得使用外部知识，不得虚构来源标签。"""


def _stable_order(values, seed, namespace):
    return sorted(values, key=lambda value: hashlib.sha256(
        f"{namespace}:{seed}:{value}".encode("utf-8")).hexdigest())


def _user_message(question, sources):
    evidence = "\n\n".join(f"[{label}]\n{text}" for label, text in sources)
    return f"问题：{question}\n\n证据：\n{evidence}"


def _answer_record(case_id, group_id, category, question, sources, conclusion,
                   must_mention, required_citations):
    citations = "".join(f"[{label}]" for label in required_citations)
    return {
        "id": case_id,
        "group_id": group_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(question, sources)},
        ],
        "response": f"结论：{conclusion}\n依据：{citations}",
        "expected_behavior": "answer",
        "must_mention": must_mention,
        "required_citations": required_citations,
        "available_citations": [label for label, _ in sources],
    }


def _abstain_record(case_id, group_id, question, sources):
    return {
        "id": case_id,
        "group_id": group_id,
        "category": "insufficient",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(question, sources)},
        ],
        "response": "资料不足：给定证据无法回答这个问题。",
        "expected_behavior": "abstain",
        "must_mention": ["资料不足"],
        "required_citations": [],
        "available_citations": [label for label, _ in sources],
    }


def synthetic_sft_records(group_count=60):
    """本项目编写的设备卡片；每个事实主体形成一个不可拆分的 split group。"""
    if group_count < 15:
        raise ValueError("至少需要 15 个事实组，才能形成有意义的三路划分")
    people = ("小林", "小周", "小陈", "小许", "小吴", "小赵")
    places = ("一号实验室", "二号实验室", "东侧工作室", "西侧工作室", "资料室", "测试间")
    modes = ("标准模式", "安静模式", "节能模式", "检查模式", "记录模式")
    cycles = (3, 5, 7, 10, 14, 21)
    records = []
    for index in range(group_count):
        device = f"设备A{index + 1:03d}"
        owner = people[index % len(people)]
        place = places[(index * 5 + 1) % len(places)]
        mode = modes[(index * 3 + 2) % len(modes)]
        cycle = cycles[(index * 7 + 3) % len(cycles)]
        group = f"device-{index + 1:03d}"
        sources = [
            ("S1", f"{device}存放在{place}，日常负责人是{owner}。"),
            ("S2", f"{device}使用{mode}运行；维护卡要求每{cycle}天检查一次。"),
            ("S3", "通用记录规范要求操作人员在工作结束后保存结果，但没有记录设备采购信息。"),
        ]
        # 三种问法轮换，避免所有问题只有一个固定表面模板。
        owner_questions = (
            f"谁负责{device}的日常工作？",
            f"{device}的负责人是谁？",
            f"请根据资料说明{device}由谁负责。",
        )
        combined_questions = (
            f"{device}存放在哪里，并且多久检查一次？",
            f"请同时给出{device}的位置和检查周期。",
            f"根据记录，{device}位于何处？维护间隔是多少天？",
        )
        insufficient_questions = (
            f"{device}的采购价格是多少？",
            f"请给出{device}的购买日期。",
            f"{device}的生产厂家是哪一家？",
        )
        variant = index % 3
        records.append(_answer_record(
            f"{group}-single", group, "single_source", owner_questions[variant], sources,
            f"{device}的日常负责人是{owner}。", [device, owner], ["S1"]))
        records.append(_answer_record(
            f"{group}-multi", group, "multi_source", combined_questions[variant], sources,
            f"{device}存放在{place}，每{cycle}天检查一次。",
            [device, place, str(cycle)], ["S1", "S2"]))
        records.append(_abstain_record(
            f"{group}-insufficient", group, insufficient_questions[variant], sources))
    return records


def _validate_record(record):
    required = {"id", "group_id", "category", "messages", "response", "expected_behavior",
                "must_mention", "required_citations", "available_citations"}
    if set(record) != required:
        raise ValueError(f"SFT 样本字段不正确：{record.get('id')}")
    if [message.get("role") for message in record["messages"]] != ["system", "user"]:
        raise ValueError("messages 必须按 system、user 排列；assistant 单独保存在 response")
    if record["messages"][0]["content"] != SYSTEM_PROMPT:
        raise ValueError("样本 system prompt 不一致")
    if not set(record["required_citations"]) <= set(record["available_citations"]):
        raise ValueError("目标引用不在实际证据标签中")
    if not all(term.lower() in record["response"].lower() for term in record["must_mention"]):
        raise ValueError("参考回答缺少 must_mention")
    if record["expected_behavior"] == "answer":
        if not record["response"].startswith("结论：") or "\n依据：" not in record["response"]:
            raise ValueError("回答样本没有遵循两行输出契约")
        if not all(f"[{label}]" in record["response"] for label in record["required_citations"]):
            raise ValueError("参考回答缺少必需引用")
    elif record["expected_behavior"] == "abstain":
        if record["response"] != "资料不足：给定证据无法回答这个问题。" or record["required_citations"]:
            raise ValueError("拒答样本的输出契约不正确")
    else:
        raise ValueError("expected_behavior 只能是 answer/abstain")


def prepare_sft_data(output=DEFAULT_OUTPUT, seed=20260914, group_count=60):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("SFT 数据版本已经存在；请使用新的 --output，保留原版本")
    records = synthetic_sft_records(group_count)
    for record in records:
        _validate_record(record)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("SFT 样本 ID 重复")
    groups = _stable_order({record["group_id"] for record in records}, seed, "split")
    validation_count = max(1, round(len(groups) * 0.1))
    test_count = max(1, round(len(groups) * 0.1))
    if validation_count + test_count >= len(groups):
        raise ValueError("分组数量不足以产生 train/validation/test")
    validation_groups = set(groups[:validation_count])
    test_groups = set(groups[validation_count:validation_count + test_count])
    dataset = {"train": [], "validation": [], "test": []}
    for record in records:
        split = ("validation" if record["group_id"] in validation_groups else
                 "test" if record["group_id"] in test_groups else "train")
        dataset[split].append(record)
    for split in dataset:
        dataset[split] = sorted(
            dataset[split], key=lambda item: hashlib.sha256(
                f"order:{seed}:{item['id']}".encode("utf-8")).hexdigest())
    group_sets = {split: {record["group_id"] for record in values} for split, values in dataset.items()}
    if any(group_sets[left] & group_sets[right]
           for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("SFT 数据发生事实主体泄漏")
    diagnostics = {
        f"{split}_{unit}": (len(values) if unit == "examples" else len(group_sets[split]))
        for split, values in dataset.items() for unit in ("examples", "groups")
    }
    diagnostics["category_counts"] = {
        split: {category: sum(item["category"] == category for item in values)
                for category in ("single_source", "multi_source", "insufficient")}
        for split, values in dataset.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="preparing-sft-") as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        save_report(staging / "dataset.json", dataset)
        save_report(staging / "system_prompt.json", {"system": SYSTEM_PROMPT})
        artifacts = {name: sha256(staging / name) for name in ("dataset.json", "system_prompt.json")}
        save_report(staging / "manifest.json", {
            "schema": "rag-sft-data-v1",
            "source": "project_authored_synthetic_device_cards_v1",
            "seed": seed,
            "split_policy": "fact_entity_group_hash_before_any_training_rendering",
            "chat_storage": "system_and_user_messages_plus_separate_assistant_response",
            "diagnostics": diagnostics,
            "artifacts": artifacts,
            "limitations": [
                "合成事实卡片只验证格式、引用和资料不足行为，不代表开放领域问答",
                "答案值和句式会跨 group 重复；group 隔离只防止同一设备事实跨 split",
                "尚未套用具体 tokenizer 的 chat template，也未生成 response-only labels",
            ],
        })
        os.rename(staging, output)
    return diagnostics


def load_sft_data(path=DEFAULT_OUTPUT):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "rag-sft-data-v1":
        raise ValueError("SFT manifest 格式不正确")
    for name, expected in manifest.get("artifacts", {}).items():
        if sha256(path / name) != expected:
            raise ValueError(f"SFT 数据文件发生变化：{name}")
    dataset = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
    if set(dataset) != {"train", "validation", "test"}:
        raise ValueError("SFT 数据必须包含 train/validation/test")
    groups = {split: {item["group_id"] for item in records} for split, records in dataset.items()}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("加载时发现事实主体跨 split")
    for records in dataset.values():
        for record in records:
            _validate_record(record)
    return dataset, manifest, sha256(path / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--groups", type=int, default=60)
    args = parser.parse_args()
    diagnostics = prepare_sft_data(args.output, args.seed, args.groups)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"SFT 数据版本：{args.output.resolve()}")


if __name__ == "__main__":
    main()
