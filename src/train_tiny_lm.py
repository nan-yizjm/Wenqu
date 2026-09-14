"""教学训练循环：移位标签、PAD 忽略、AdamW、梯度裁剪、验证和可恢复检查点。"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import time
import tomllib

# 必须在本进程第一次 CUDA 计算前设置；不改变系统环境变量。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

from .config import PROJECT_ROOT
from .experiment_utils import sha256, torch_memory
from .tiny_decoder import DecoderConfig, TinyDecoder
from .training_data import BOS, IGNORE, CharTokenizer, load_data, make_windows

DEFAULT_CONFIG = PROJECT_ROOT / "tiny_lm.toml"
PROMPTS = ("在图书馆里，", "小周来到", "工作结束时，")


@dataclass(frozen=True)
class TrainConfig:
    device: str = "cuda"
    seed: int = 42
    steps: int = 300
    batch_size: int = 16
    block_size: int = 96
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    warmup_steps: int = 20
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batch_size: int = 32

    def __post_init__(self):
        for key in ("seed", "steps", "batch_size", "block_size", "warmup_steps", "eval_every", "eval_batch_size"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key in ("seed", "warmup_steps") else 1):
                raise ValueError(f"{key} 的整数范围不正确")
        if self.device not in ("cpu", "cuda") or self.warmup_steps >= self.steps:
            raise ValueError("device 为 cpu/cuda，warmup_steps 必须小于总 steps")
        for key in ("learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
                raise ValueError(f"{key} 必须是允许范围内的有限数值")


def source_signature():
    return {name: sha256(PROJECT_ROOT / "src" / name) for name in (
        "attention_lab.py", "tiny_decoder.py", "training_data.py", "train_tiny_lm.py")}


def atomic_write(path, write):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as handle:
            temporary = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_write(path, lambda handle: handle.write(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")))


def save_checkpoint(path, value):
    atomic_write(path, lambda handle: torch.save(value, handle))


def copy_weights(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def next_token_loss(logits, targets, reduction="mean"):
    # logits 是未 softmax 的词表分数；PAD 目标用 -100 忽略。
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE, reduction=reduction)


@torch.inference_mode()
def evaluate(model, windows, device, batch_size):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    xs, ys, _ = windows
    try:
        for start in range(0, len(xs), batch_size):
            x, y = xs[start:start + batch_size].to(device), ys[start:start + batch_size].to(device)
            total_loss += float(next_token_loss(model(x)[0], y, "sum"))
            total_tokens += int((y != IGNORE).sum())
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("验证集中没有有效目标")
    nll = total_loss / total_tokens
    if not math.isfinite(nll):
        raise ValueError("验证损失不是有限值")
    return {"nll": nll, "perplexity": math.exp(nll) if nll < 700 else None, "tokens": total_tokens}


def learning_rate(step, config):
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def samples(model, tokenizer):
    output = []
    for prompt in PROMPTS:
        ids = [BOS, *tokenizer.encode(prompt)]
        if len(ids) >= model.config.max_seq_len:
            continue
        generated = model.generate(ids, max_new_tokens=min(48, model.config.max_seq_len - len(ids)))
        output.append({"prompt": prompt, "text": tokenizer.decode(generated["ids"]),
                       "stop_reason": generated["stop_reason"]})
    return output


def train(dataset_dir, config: TrainConfig, model_options=None, *, output=None, resume=None, stop_after=None):
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；请在教学配置中选择 cpu")
    torch.manual_seed(config.seed)
    dataset, tokenizer, manifest, data_fingerprint = load_data(Path(dataset_dir))
    model_config = DecoderConfig(vocab_size=len(tokenizer.tokens), **(model_options or {}))
    if config.block_size > model_config.max_seq_len:
        raise ValueError("训练 block_size 不能超过模型位置窗口")
    train_windows = make_windows(dataset["train"], config.block_size)
    validation_windows = make_windows(dataset["validation"], config.block_size)
    signature = source_signature()
    identity = {"model_config": asdict(model_config), "train_config": asdict(config),
                "data_fingerprint": data_fingerprint, "source_signature": signature,
                "torch_version": str(torch.__version__)}
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint and (checkpoint.get("schema") != "tiny-lm-training-v1" or checkpoint.get("identity") != identity):
        raise ValueError("检查点的数据/模型/训练计划/源码/设备/torch 版本不匹配，拒绝伪装成连续训练")
    start_step = checkpoint["step"] if checkpoint else 0
    end_step = config.steps if stop_after is None else stop_after
    if not start_step < end_step <= config.steps:
        raise ValueError("要求 检查点 step < stop_after <= 计划总步数")
    model = TinyDecoder(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay, foreach=False, fused=False)
    sample_rng = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        sample_rng.set_state(checkpoint["sample_rng"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        history = list(checkpoint["history"])
        initial_samples = checkpoint["initial_samples"]
        best_state = checkpoint["best_model_state"]
        best_step, best_nll = checkpoint["best_step"], checkpoint["best_nll"]
    else:
        history = []
        initial_samples = samples(model, tokenizer)
        best_state, best_step, best_nll = copy_weights(model), 0, float("inf")
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated/tiny_lm_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    parent_hash = sha256(Path(resume)) if resume else None
    common = {"identity": identity, "vocab": tokenizer.tokens}

    def save_best():
        save_checkpoint(output / "best.pt", {"schema": "tiny-lm-weights-v1", **common,
                                              "model_state": best_state, "step": best_step, "validation_nll": best_nll})

    def save_last(step):
        save_checkpoint(output / "last.pt", {
            "schema": "tiny-lm-training-v1", **common, "step": step,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "sample_rng": sample_rng.get_state(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
            "history": history, "initial_samples": initial_samples,
            "best_model_state": best_state, "best_step": best_step, "best_nll": best_nll,
        })
        write_json(output / "history.json", history)

    def validate(step, batch_loss=None, grad_norm=None, lr=None):
        nonlocal best_state, best_step, best_nll
        train_metrics = evaluate(model, train_windows, config.device, config.eval_batch_size)
        val_metrics = evaluate(model, validation_windows, config.device, config.eval_batch_size)
        history.append({"step": step, "train": train_metrics, "validation": val_metrics,
                        "last_batch_loss_before_update": batch_loss, "grad_norm_before_clip": grad_norm, "learning_rate": lr})
        if val_metrics["nll"] < best_nll:
            best_nll, best_step, best_state = val_metrics["nll"], step, copy_weights(model)
            save_best()
        save_last(step)
        print(f"step {step}/{config.steps} | train NLL {train_metrics['nll']:.4f} | val NLL {val_metrics['nll']:.4f}", flush=True)

    if checkpoint:
        save_best()  # 新目录保存历史最佳权重，不依赖父目录文件一直留在原路径。
        save_last(start_step)
    else:
        validate(0)
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(start_step + 1, end_step + 1):
        model.train()
        positions = torch.randint(len(train_windows[0]), (config.batch_size,), generator=sample_rng)
        x, y = train_windows[0][positions].to(config.device), train_windows[1][positions].to(config.device)
        lr = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(x)[0], y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("训练 loss 非有限值；保留上一次完整检查点，不保存半步状态")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if step % config.eval_every == 0 or step == end_step:
            validate(step, float(loss.detach()), float(norm), lr)
    if source_signature() != signature or load_data(Path(dataset_dir))[3] != data_fingerprint:
        raise RuntimeError("运行过程中源码或数据变化；不能把此次结果作为可复现实验")
    final_samples = samples(model, tokenizer)
    summary = {
        "schema": "tiny-lm-run-v1", "run_dir": str(output), "identity": identity,
        "resume_from": str(Path(resume).resolve()) if resume else None, "parent_checkpoint_sha256": parent_hash,
        "start_step": start_step, "end_step": end_step, "complete": end_step == config.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": len(train_windows[0]), "validation_windows": len(validation_windows[0]),
        "dataset": manifest["diagnostics"], "history": history, "best_step": best_step, "best_validation_nll": best_nll,
        "initial_samples": initial_samples, "current_samples": final_samples,
        "elapsed_seconds_this_invocation": time.perf_counter() - started, "torch_cuda_memory": torch_memory(),
        "limitations": ["合成字符级语料，不是知识问答/指令微调", "验证集参与选 best，没有独立测试集",
                        "同模板不同组合的验证结果不能证明通用泛化", "只支持同设备/同代码/同计划恢复，不承诺跨硬件逐位一致"]}
    write_json(output / "summary.json", summary)
    print(f"本次训练结束：{output}；最佳 step={best_step}；不替换 RAG 的 Qwen 模型", flush=True)
    return summary


def generate_from_checkpoint(path, prompt, device="cpu", max_new_tokens=48):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") not in ("tiny-lm-training-v1", "tiny-lm-weights-v1"):
        raise ValueError("不是本项目的教学检查点")
    tokenizer = CharTokenizer(checkpoint["vocab"])
    model = TinyDecoder(DecoderConfig(**checkpoint["identity"]["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    generated = model.generate([BOS, *tokenizer.encode(prompt)], max_new_tokens)
    return {"text": tokenizer.decode(generated["ids"]), "stop_reason": generated["stop_reason"],
            "step": checkpoint["step"], "warning": "教学模型，仅学过有限合成短文，不是知识库助手"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, help="只恢复本项目自己生成的 last.pt")
    parser.add_argument("--stop-after", type=int, help="本次只跑到该步；不改变计划总 steps 或学习率曲线")
    parser.add_argument("--generate", type=Path, help="从本项目 best.pt / last.pt 生成，不继续训练")
    parser.add_argument("--prompt", default="在图书馆里，")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.generate:
        print(json.dumps(generate_from_checkpoint(args.generate, args.prompt, args.device or "cpu"), ensure_ascii=False, indent=2))
        return
    with args.config.open("rb") as handle:
        values = tomllib.load(handle)
    if set(values) != {"dataset_dir", "model", "training"}:
        parser.error("配置必须包含 dataset_dir、model、training")
    training = dict(values["training"])
    if args.device:
        training["device"] = args.device
    try:
        train((args.config.resolve().parent / values["dataset_dir"]).resolve(), TrainConfig(**training), values["model"],
              resume=args.resume, stop_after=args.stop_after)
    except KeyboardInterrupt:
        print("训练已中断；请从最近一次完整的 last.pt 恢复，不保存被打断的半步更新。")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
