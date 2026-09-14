"""用固定 Qwen tokenizer/chat template 构造 response-only SFT 标签并检查长度。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

import torch
from transformers import AutoTokenizer
import transformers

from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256
from .sft_data_v2 import DEFAULT_OUTPUT as DEFAULT_DATA, load

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_MAX_LENGTH = 512


def load_tokenizer(allow_download=False):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, local_files_only=not allow_download,
    )
    if not tokenizer.chat_template:
        raise ValueError("目标 tokenizer 没有 chat template")
    if tokenizer.pad_token_id is None:
        raise ValueError("目标 tokenizer 没有 pad_token_id，不能静默猜测")
    return tokenizer


def _input_ids(rendered):
    """Transformers 4.x 常返回 list，5.x 默认可能返回 BatchEncoding。"""
    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("单样本 chat template 意外返回多个序列")
        rendered = rendered[0]
    return list(rendered)


def response_only_tokens(record, tokenizer, max_length=DEFAULT_MAX_LENGTH):
    """先渲染 assistant 起始标记，再只监督 assistant 正文和结束标记。"""
    prompt_ids = _input_ids(tokenizer.apply_chat_template(
        record["messages"], tokenize=True, add_generation_prompt=True,
    ))
    full_messages = [*record["messages"], {"role": "assistant", "content": record["response"]}]
    full_ids = _input_ids(tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
    ))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"chat template 的训练序列不以 generation prompt 开头：{record['id']}")
    if len(full_ids) > max_length:
        raise ValueError(
            f"{record['id']} 共 {len(full_ids)} tokens，超过 {max_length}；本版拒绝静默截断证据或答案")
    response_tokens = len(full_ids) - len(prompt_ids)
    if response_tokens < 2:
        raise ValueError(f"assistant 监督 token 过少：{record['id']}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
        "prompt_tokens": len(prompt_ids),
        "response_tokens": response_tokens,
        "total_tokens": len(full_ids),
    }


def collate_response_only(examples, pad_token_id):
    if not examples:
        raise ValueError("batch 不能为空")
    length = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        size = len(example["input_ids"])
        input_ids[row, :size] = torch.tensor(example["input_ids"])
        attention_mask[row, :size] = 1
        labels[row, :size] = torch.tensor(example["labels"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _statistics(values):
    return {"min": min(values), "max": max(values), "mean": statistics.fmean(values),
            "median": statistics.median(values)}


def analyze(data_dir=DEFAULT_DATA, max_length=DEFAULT_MAX_LENGTH, allow_download=False, output=None):
    data_dir = Path(data_dir).resolve()
    dataset, data_manifest, data_fingerprint = load(data_dir)
    tokenizer = load_tokenizer(allow_download)
    encoded = {split: [response_only_tokens(record, tokenizer, max_length) for record in records]
               for split, records in dataset.items()}
    split_stats = {}
    for split, examples in encoded.items():
        totals = [example["total_tokens"] for example in examples]
        prompts = [example["prompt_tokens"] for example in examples]
        responses = [example["response_tokens"] for example in examples]
        split_stats[split] = {
            "examples": len(examples), "total_tokens": _statistics(totals),
            "prompt_tokens": _statistics(prompts), "supervised_response_tokens": _statistics(responses),
            "supervised_fraction": sum(responses) / sum(totals),
            "over_limit": sum(total > max_length for total in totals),
        }
    first_record = dataset["train"][0]
    first = encoded["train"][0]
    supervised_ids = [token for token, label in zip(first["input_ids"], first["labels"], strict=True)
                      if label != -100]
    preview = {
        "id": first_record["id"],
        "logical_messages": first_record["messages"],
        "target_response": first_record["response"],
        "prompt_tokens_masked": first["prompt_tokens"],
        "response_tokens_supervised": first["response_tokens"],
        "rendered_full_text": tokenizer.decode(first["input_ids"], skip_special_tokens=False),
        "decoded_supervised_suffix": tokenizer.decode(supervised_ids, skip_special_tokens=False),
    }
    result = {
        "schema": "rag-sft-tokenization-report-v1",
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "license_from_model_card": "apache-2.0",
        "tokenizer_class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "max_length": max_length, "data_dir": str(data_dir),
        "data_fingerprint": data_fingerprint, "data_artifacts": data_manifest["artifacts"],
        "transformers_version": transformers.__version__, "torch_version": str(torch.__version__),
        "split_stats": split_stats, "preview": preview,
        "label_contract": [
            "system/user/assistant-header labels are -100",
            "assistant response plus its chat-template end marker are supervised",
            "right padding input uses tokenizer pad id; padding labels remain -100",
            "over-length examples raise instead of silently truncating evidence or response",
        ],
        "source_sha256": sha256(Path(__file__)),
    }
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated" / (
        "sft_tokenization_qwen2_5_1_5b_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, result)
    for split, stats in split_stats.items():
        print(f"{split}: {stats['examples']} 条；总长度 {stats['total_tokens']['min']}.."
              f"{stats['total_tokens']['max']}；回答监督占比 {stats['supervised_fraction']:.1%}")
    print(f"tokenizer：{MODEL_NAME}@{MODEL_REVISION[:12]}")
    print(f"详细报告：{output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--allow-download", action="store_true",
                        help="仅在本机没有固定 tokenizer 快照时允许下载；不会下载模型权重")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.data_dir, args.max_length, args.allow_download, args.output)


if __name__ == "__main__":
    main()
