"""评估教学字符模型在选择验证集、模板变化、格式变化和领域变化上的差距。"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256
from .tiny_decoder import DecoderConfig, TinyDecoder
from .training_data import BOS, IGNORE, UNK, CharTokenizer, load_data, make_windows

DEFAULT_CHALLENGE = PROJECT_ROOT / "data/tiny_lm_challenge.json"


def load_challenge(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "tiny-lm-challenge-v1" or not isinstance(value.get("cases"), list):
        raise ValueError("挑战集格式不正确")
    allowed = set(value.get("categories", {}))
    ids, texts = set(), set()
    for case in value["cases"]:
        if set(case) != {"id", "category", "text"}:
            raise ValueError("每个挑战样本只能包含 id/category/text")
        if case["category"] not in allowed or not all(isinstance(case[key], str) and case[key] for key in case):
            raise ValueError("挑战样本字段为空或类别未声明")
        normalized = case["text"].replace("\r\n", "\n").strip()
        if case["id"] in ids or normalized in texts:
            raise ValueError("挑战集存在重复 ID 或文本")
        ids.add(case["id"])
        texts.add(normalized)
        case["text"] = normalized
    if not value["cases"]:
        raise ValueError("挑战集为空")
    return value


def encode_documents(records, tokenizer):
    return [{"id": item["id"], "ids": tokenizer.encode(item["text"], document=True)} for item in records]


@torch.inference_mode()
def evaluate_records(model, records, tokenizer, block_size, device, batch_size=32):
    """同时报告全部目标和排除 UNK 后的结果，避免把词表问题混入句式问题。"""
    model.eval()
    windows = make_windows(encode_documents(records, tokenizer), block_size)
    xs, ys, _ = windows
    loss_sum = 0.0
    known_loss_sum = 0.0
    correct = 0
    known_correct = 0
    tokens = 0
    known_tokens = 0
    unk_tokens = 0
    for start in range(0, len(xs), batch_size):
        x = xs[start:start + batch_size].to(device)
        y = ys[start:start + batch_size].to(device)
        logits = model(x)[0]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = y.reshape(-1)
        valid = flat_targets != IGNORE
        known = valid & (flat_targets != UNK)
        losses = F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE, reduction="none")
        predicted = flat_logits.argmax(dim=-1)
        loss_sum += float(losses[valid].sum())
        known_loss_sum += float(losses[known].sum())
        correct += int((predicted[valid] == flat_targets[valid]).sum())
        known_correct += int((predicted[known] == flat_targets[known]).sum())
        tokens += int(valid.sum())
        known_tokens += int(known.sum())
        unk_tokens += int((valid & (flat_targets == UNK)).sum())
    if tokens == 0 or known_tokens == 0:
        raise ValueError("评估切片没有足够的有效/已知字符目标")
    nll = loss_sum / tokens
    known_nll = known_loss_sum / known_tokens
    return {
        "documents": len(records),
        "windows": len(xs),
        "tokens": tokens,
        "unknown_targets": unk_tokens,
        "unknown_rate": unk_tokens / tokens,
        "nll_all_targets": nll,
        "perplexity_all_targets": math.exp(nll) if nll < 700 else None,
        "top1_accuracy_all_targets": correct / tokens,
        "known_tokens": known_tokens,
        "nll_known_targets": known_nll,
        "perplexity_known_targets": math.exp(known_nll) if known_nll < 700 else None,
        "top1_accuracy_known_targets": known_correct / known_tokens,
    }


def model_from_checkpoint(checkpoint, device):
    model = TinyDecoder(DecoderConfig(**checkpoint["identity"]["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def run(checkpoint_path: Path, challenge_path=DEFAULT_CHALLENGE, device="cuda", output=None):
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint_path).resolve()
    challenge_path = Path(challenge_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") not in ("tiny-lm-weights-v1", "tiny-lm-training-v1"):
        raise ValueError("只接受本项目 tiny LM 的 best.pt/last.pt")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；改用 --device cpu")
    challenge = load_challenge(challenge_path)
    identity = checkpoint["identity"]
    dataset_dir = PROJECT_ROOT / "data/generated/tiny_lm_data_v1"
    dataset, tokenizer, manifest, dataset_fingerprint = load_data(dataset_dir)
    if tokenizer.tokens != checkpoint["vocab"] or dataset_fingerprint != identity["data_fingerprint"]:
        raise ValueError("挑战评估的数据/词表与检查点训练身份不一致")
    training_texts = {item["text"].replace("\r\n", "\n").strip()
                      for split in ("train", "validation") for item in dataset[split]}
    overlap = [case["id"] for case in challenge["cases"] if case["text"] in training_texts]
    if overlap:
        raise ValueError(f"挑战文本与训练/选择验证文本完全重复：{overlap}")
    block_size = identity["train_config"]["block_size"]
    if block_size > identity["model_config"]["max_seq_len"]:
        raise ValueError("检查点的训练窗口超过模型位置窗口")
    trained = model_from_checkpoint(checkpoint, device)
    # 复建这次实验的 step-0 权重，作为“有没有学到训练分布”的参照。
    torch.manual_seed(identity["train_config"]["seed"])
    random_model = TinyDecoder(DecoderConfig(**identity["model_config"])).to(device)

    slices = {
        "selection_validation": [{"id": item["id"], "text": item["text"]} for item in dataset["validation"]]
    }
    grouped = defaultdict(list)
    for case in challenge["cases"]:
        grouped[case["category"]].append(case)
    slices.update(grouped)
    results = {}
    for name, records in slices.items():
        trained_metrics = evaluate_records(trained, records, tokenizer, block_size, device)
        random_metrics = evaluate_records(random_model, records, tokenizer, block_size, device)
        results[name] = {
            "trained": trained_metrics,
            "step0_random": random_metrics,
            "known_nll_change_from_step0": trained_metrics["nll_known_targets"] - random_metrics["nll_known_targets"],
        }

    prompts = ("小周来到", "人物：小林。地点：", "什么是注意力？")
    generations = []
    for prompt in prompts:
        generated = trained.generate([BOS, *tokenizer.encode(prompt)], max_new_tokens=48)
        generations.append({
            "prompt": prompt,
            "text": tokenizer.decode(generated["ids"]),
            "prompt_unknown_characters": sum(1 for item in tokenizer.encode(prompt) if item == UNK),
            "stop_reason": generated["stop_reason"],
        })
    result = {
        "schema": "tiny-lm-generalization-report-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "challenge": str(challenge_path),
        "challenge_sha256": sha256(challenge_path),
        "challenge_frozen_on": challenge["frozen_on"],
        "evaluator_sha256": sha256(Path(__file__)),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_source": manifest["source"],
        "device": device,
        "results": results,
        "generations": generations,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_contract": [
            "selection_validation 参与过 best checkpoint 选择，不是独立测试",
            "challenge 在本次评估前冻结；后续不根据结果改 v1 模型或样本",
            "template/format 的 known-target NLL 用于分离词表未知字符，但输入中的 UNK 影响仍存在",
            "低 teacher-forced NLL 不等于自由生成、问答或事实能力",
        ],
    }
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated" / (
        "tiny_lm_generalization_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, result)
    print(f"模型 step：{checkpoint['step']}；设备：{device}")
    for name, values in results.items():
        trained_metrics = values["trained"]
        print(f"{name}: known NLL={trained_metrics['nll_known_targets']:.4f}, "
              f"all NLL={trained_metrics['nll_all_targets']:.4f}, "
              f"UNK={trained_metrics['unknown_rate']:.1%}, "
              f"top1={trained_metrics['top1_accuracy_all_targets']:.1%}")
    print(f"详细报告：{output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.challenge, args.device, args.output)


if __name__ == "__main__":
    main()
