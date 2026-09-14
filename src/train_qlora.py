"""在 8 GB GPU 上对固定 Qwen2.5-1.5B-Instruct 做 response-only QLoRA。"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import tomllib

# 在第一次 CUDA 计算前设置；bitsandbytes 内核仍不承诺跨版本逐位一致。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import PROJECT_ROOT
from .experiment_utils import GPUSampler, save_report, sha256, synchronize, torch_memory
from .qlora_model import load_base
from .sft_data_v2 import DEFAULT_OUTPUT as DEFAULT_DATA, load as load_data
from .sft_tokenization import (
    DEFAULT_MAX_LENGTH,
    MODEL_NAME,
    MODEL_REVISION,
    collate_response_only,
    response_only_tokens,
)

DEFAULT_CONFIG = PROJECT_ROOT / "qlora.toml"


@dataclass(frozen=True)
class TrainPlan:
    seed: int
    epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_clip: float

    def __post_init__(self):
        positive_ints = ("epochs", "micro_batch_size", "gradient_accumulation_steps")
        for name in positive_ints:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} 必须是正整数")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed 必须是非负整数")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps 必须是非负整数")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数")
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning_rate 和 gradient_clip 必须大于 0")


def read_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    if set(values) != {"model_name", "model_revision", "data_dir", "max_length", "lora", "training"}:
        raise ValueError("QLoRA 配置字段不完整或出现未知顶层字段")
    if values["model_name"] != MODEL_NAME or values["model_revision"] != MODEL_REVISION:
        raise ValueError("配置中的模型名/commit 与代码固定版本不一致")
    if type(values["max_length"]) is not int or values["max_length"] < 1:
        raise ValueError("max_length 必须是正整数")
    lora = values["lora"]
    if set(lora) != {"r", "alpha", "dropout", "target_modules"}:
        raise ValueError("lora 配置字段必须为 r/alpha/dropout/target_modules")
    if type(lora["r"]) is not int or lora["r"] < 1 or type(lora["alpha"]) is not int or lora["alpha"] < 1:
        raise ValueError("LoRA r/alpha 必须是正整数")
    if not 0 <= lora["dropout"] < 1 or lora["target_modules"] != "all-linear":
        raise ValueError("本实验要求 dropout 在 [0,1)，target_modules 固定为 all-linear")
    return path, values, TrainPlan(**values["training"])


def source_signature(config_path):
    files = ("qlora_model.py", "sft_data_v2.py", "sft_tokenization.py", "train_qlora.py")
    return {
        **{f"src/{name}": sha256(PROJECT_ROOT / "src" / name) for name in files},
        str(Path(config_path).relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(config_path),
    }


def package_versions():
    names = ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
    return {name: importlib.metadata.version(name) for name in names}


def learning_rate(update, total_updates, plan):
    if plan.warmup_steps and update <= plan.warmup_steps:
        return plan.learning_rate * update / plan.warmup_steps
    decay_steps = max(1, total_updates - plan.warmup_steps)
    progress = min(1.0, max(0.0, (update - plan.warmup_steps) / decay_steps))
    return plan.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def encode_split(records, tokenizer, max_length):
    return [response_only_tokens(record, tokenizer, max_length) for record in records]


def data_loader(examples, tokenizer, batch_size, *, shuffle=False, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: collate_response_only(batch, tokenizer.pad_token_id),
    )


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def evaluate_loss(model, examples, tokenizer, batch_size):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        loader = data_loader(examples, tokenizer, batch_size)
        for batch in loader:
            batch = move_batch(batch, model.device)
            target_tokens = int((batch["labels"] != -100).sum())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("验证 loss 不是有限值")
            total_loss += float(loss) * target_tokens
            total_tokens += target_tokens
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("验证集没有受监督 token")
    nll = total_loss / total_tokens
    return {
        "nll": nll,
        "perplexity": math.exp(nll) if nll < 700 else None,
        "supervised_tokens": total_tokens,
    }


def train(config_path=DEFAULT_CONFIG, *, output=None, max_updates=None, allow_download=False):
    started = time.perf_counter()
    config_path, config, plan = read_config(config_path)
    config_dir = config_path.parent
    data_dir = (config_dir / config["data_dir"]).resolve()
    if not data_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("训练数据必须位于当前项目中")
    dataset, data_manifest, data_fingerprint = load_data(data_dir)
    signature = source_signature(config_path)

    torch.manual_seed(plan.seed)
    torch.cuda.manual_seed_all(plan.seed)
    torch.set_float32_matmul_precision("high")
    model, tokenizer = load_base(allow_download=allow_download, training=True)
    train_examples = encode_split(dataset["train"], tokenizer, config["max_length"])
    val_examples = encode_split(dataset["validation"], tokenizer, config["max_length"])

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if not trainable_parameters or trainable_parameters >= total_parameters:
        raise RuntimeError("LoRA 可训练参数计数异常")

    micro_batches = math.ceil(len(train_examples) / plan.micro_batch_size)
    updates_per_epoch = math.ceil(micro_batches / plan.gradient_accumulation_steps)
    planned_updates = updates_per_epoch * plan.epochs
    run_limit = planned_updates if max_updates is None else max_updates
    if type(run_limit) is not int or not 1 <= run_limit <= planned_updates:
        raise ValueError(f"max_updates 必须位于 1..{planned_updates}")
    is_smoke = run_limit < planned_updates

    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated/qlora_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ("_smoke" if is_smoke else ""))
    output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(
        trainable, lr=plan.learning_rate, weight_decay=plan.weight_decay,
        foreach=False, fused=False,
    )

    post_load_memory = torch_memory()
    initial_validation = evaluate_loss(model, val_examples, tokenizer, plan.micro_batch_size)
    print(
        f"可训练参数：{trainable_parameters:,}/{total_parameters:,} "
        f"({trainable_parameters / total_parameters:.3%})；初始 val NLL={initial_validation['nll']:.4f}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    history = []
    update = 0
    best_nll = float("inf")
    best_epoch = None
    optimizer.zero_grad(set_to_none=True)
    sampler = GPUSampler(interval=1.0)

    with sampler:
        for epoch in range(1, plan.epochs + 1):
            epoch_started = time.perf_counter()
            loader = data_loader(
                train_examples, tokenizer, plan.micro_batch_size,
                shuffle=True, seed=plan.seed + epoch,
            )
            batch_count = len(loader)
            epoch_loss_sum = 0.0
            epoch_tokens = 0
            grad_norms = []
            current_lr = None
            for batch_index, batch in enumerate(loader):
                group_position = batch_index % plan.gradient_accumulation_steps
                if group_position == 0:
                    group_size = min(plan.gradient_accumulation_steps, batch_count - batch_index)
                    current_lr = learning_rate(update + 1, planned_updates, plan)
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                batch = move_batch(batch, model.device)
                supervised_tokens = int((batch["labels"] != -100).sum())
                model.train()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_loss = model(**batch).loss
                if not bool(torch.isfinite(raw_loss)):
                    raise RuntimeError("训练 loss 不是有限值；当前 adapter 不保存为有效结果")
                (raw_loss / group_size).backward()
                epoch_loss_sum += float(raw_loss.detach()) * supervised_tokens
                epoch_tokens += supervised_tokens
                group_finished = group_position + 1 == group_size or batch_index + 1 == batch_count
                if group_finished:
                    norm = torch.nn.utils.clip_grad_norm_(
                        trainable, plan.gradient_clip, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    grad_norms.append(float(norm))
                    if update == 1 or update % 5 == 0 or update == run_limit:
                        print(
                            f"update {update}/{planned_updates} | epoch {epoch}/{plan.epochs} "
                            f"| batch loss {float(raw_loss.detach()):.4f} | lr {current_lr:.2e}",
                            flush=True,
                        )
                    if update >= run_limit:
                        break

            synchronize()
            validation = evaluate_loss(model, val_examples, tokenizer, plan.micro_batch_size)
            epoch_record = {
                "epoch": epoch,
                "updates_completed": update,
                "train_response_nll": epoch_loss_sum / epoch_tokens,
                "train_supervised_tokens_seen": epoch_tokens,
                "validation": validation,
                "learning_rate_after_epoch": current_lr,
                "gradient_norm_before_clip": {
                    "min": min(grad_norms), "max": max(grad_norms),
                    "mean": sum(grad_norms) / len(grad_norms),
                },
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            epoch_adapter = output / f"epoch_{epoch:02d}_adapter"
            model.save_pretrained(epoch_adapter, safe_serialization=True)
            if validation["nll"] < best_nll:
                best_nll = validation["nll"]
                best_epoch = epoch
                model.save_pretrained(output / "best_adapter", safe_serialization=True)
            print(
                f"epoch {epoch} 完成 | train NLL {epoch_record['train_response_nll']:.4f} "
                f"| val NLL {validation['nll']:.4f}",
                flush=True,
            )
            if update >= run_limit:
                break

    model.save_pretrained(output / "final_adapter", safe_serialization=True)
    synchronize()
    if source_signature(config_path) != signature or load_data(data_dir)[2] != data_fingerprint:
        raise RuntimeError("训练期间配置、源码或数据发生变化，本次结果不能作为固定实验")
    report = {
        "schema": "rag-qlora-run-v1",
        "role": "smoke" if is_smoke else "complete_training",
        "run_dir": str(output),
        "complete": update == planned_updates,
        "updates_completed": update,
        "planned_updates": planned_updates,
        "updates_per_epoch": updates_per_epoch,
        "config": config,
        "train_plan": asdict(plan),
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "load": "bitsandbytes NF4 double quantization, BF16 compute",
            "trainable_parameters": trainable_parameters,
            "total_parameter_elements_reported": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
        },
        "data": {
            "directory": str(data_dir),
            "fingerprint": data_fingerprint,
            "manifest_diagnostics": data_manifest["diagnostics"],
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
        },
        "initial_validation": initial_validation,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "source_signature": signature,
        "package_versions": package_versions(),
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "post_load_memory": post_load_memory,
            "training_peak_memory": torch_memory(),
            "nvidia_smi_sampling": sampler.report(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "best_adapter": str(output / "best_adapter"),
            "final_adapter": str(output / "final_adapter"),
        },
        "limitations": [
            "合成数据只验证引用、拒答、冲突、多跳和证据内指令等目标行为",
            "验证集用于选择 best adapter；冻结 challenge 才是独立测试",
            "未保存优化器状态，因此当前脚本不提供中途恢复；adapter 权重不能替代完整训练检查点",
            "固定随机种子不等于 bitsandbytes/CUDA 跨版本逐位可复现",
            "4-bit 量化参数元素计数不是实际显存字节数",
        ],
    }
    save_report(output / "summary.json", report)
    print(f"QLoRA 结束：{output}", flush=True)
    print(f"最佳 epoch={best_epoch}；val NLL={best_nll:.4f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-updates", type=int,
                        help="仅做显式 smoke run；小于计划步数时报告标记为 incomplete")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    try:
        train(args.config, output=args.output, max_updates=args.max_updates,
              allow_download=args.allow_download)
    except torch.OutOfMemoryError:
        print(
            "CUDA 显存不足。当前进程不会静默修改 batch；请记录失败后显式调整 "
            "micro_batch_size/gradient_accumulation_steps，并保持有效 batch 不变。",
            flush=True,
        )
        raise
    except KeyboardInterrupt:
        print("训练被中断；已完成 epoch 的 adapter 仍保留，但不能从它恢复优化器状态。", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
