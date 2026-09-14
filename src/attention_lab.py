"""手写单层因果多头注意力与动态 KV Cache。随机张量实验，不是训练好的语言模型。"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import math
import statistics
import time

import torch
from torch import nn

from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256, synchronize


def causal_mask(query_length: int, key_length: int, past_length: int, device=None):
    """True = 允许关注。query 的绝对位置从 past_length 开始，而非每次从 0 开始。"""
    if query_length < 1 or past_length < 0 or key_length != past_length + query_length:
        raise ValueError("要求 key_length = past_length + query_length，且 query_length > 0")
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    return key_positions[None, :] <= query_positions[:, None]


@dataclass(frozen=True)
class KVCache:
    key: torch.Tensor               # [B, H, cached_length, head_dim]
    value: torch.Tensor
    owner: object = field(repr=False)  # 防止把另一个 attention 实例的 cache 拿来混用。

    @property
    def length(self):
        return self.key.shape[-2]

    @property
    def logical_bytes(self):
        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, heads=4):
        super().__init__()
        if d_model < 1 or heads < 1 or d_model % heads:
            raise ValueError("d_model 必须能被正整数 heads 整除")
        self.d_model, self.heads, self.head_dim = d_model, heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self._cache_owner = object()
        self.reset_work()

    def reset_work(self):
        self.projected_positions = 0
        self.dense_score_elements = 0

    def project(self, x):
        batch, length, _ = x.shape
        fused = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        # 整理 head 布局；contiguous 不保证新存储，首次缓存会显式 clone。
        return tuple(part.transpose(1, 2).contiguous() for part in fused.unbind(dim=2))

    @staticmethod
    def attend(query, key, value, allowed):
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ value

    def forward(self, x, cache: KVCache | None = None, *, use_cache=False):
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] == 0:
            raise ValueError("输入必须为非空 [batch, tokens, d_model]")
        if cache is not None and not use_cache:
            raise ValueError("传入 cache 时必须 use_cache=True")
        if use_cache and torch.is_grad_enabled():
            raise ValueError("本实验的缓存只用于推理；请使用 torch.inference_mode()")
        batch, length, _ = x.shape
        query, key, value = self.project(x)
        past_length = 0
        if cache is not None:
            if cache.owner is not self._cache_owner:
                raise ValueError("不能混用另一个 attention 实例的 cache")
            if (cache.key.ndim != 4 or cache.key.shape != cache.value.shape
                    or cache.key.shape[:2] != (batch, self.heads)
                    or cache.key.shape[-1] != self.head_dim
                    or any(t.dtype != key.dtype or t.device != key.device for t in (cache.key, cache.value))):
                raise ValueError("cache 的 batch/head/维度/类型/设备不兼容")
            past_length = cache.length
            key = torch.cat((cache.key, key), dim=-2)
            value = torch.cat((cache.value, value), dim=-2)
        elif use_cache:
            # 单 token 时切片可能已经 contiguous，仍引用整个 fused QKV。
            # clone 才保证 cache 不因为一个小 view 保留 Q 的底层存储。
            key, value = key.clone(), value.clone()
        allowed = causal_mask(length, key.shape[-2], past_length, x.device)
        attended = self.attend(query, key, value, allowed)
        merged = attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        result = self.output(merged)
        self.projected_positions += batch * length
        self.dense_score_elements += batch * self.heads * length * key.shape[-2]
        updated = KVCache(key, value, self._cache_owner) if use_cache else None
        return result, updated


def decode_fixed_inputs(model, inputs, prefix, *, cached, collect=True):
    """相同输入序列、逐步增加前缀；不用模型采样，以免输入变化干扰数值对比。"""
    if not 1 <= prefix < inputs.shape[1]:
        raise ValueError("prefix 必须在 1～总长度减 1 之间")
    first, cache = model(inputs[:, :prefix], use_cache=cached)
    outputs = [first] if collect else None
    for end in range(prefix + 1, inputs.shape[1] + 1):
        block = inputs[:, end - 1:end] if cached else inputs[:, :end]
        result, cache = model(block, cache, use_cache=cached)
        if collect:
            outputs.append(result[:, -1:].clone())
    return torch.cat(outputs, dim=1) if collect else None, cache


def measure(function, repeats):
    function()  # 热身不计入正式时间。
    synchronize()
    gpu = torch.cuda.is_initialized()
    baseline = torch.cuda.memory_allocated() if gpu else None
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        function()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "cuda_extra_peak_allocated_bytes": (torch.cuda.max_memory_allocated() - baseline if gpu else None)}


def run_experiment(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = CausalSelfAttention(args.d_model, args.heads).to(args.device).eval()
    total = args.prefix + args.decode
    inputs = torch.randn(args.batch, total, args.d_model, device=args.device)
    with torch.inference_mode():
        full, _ = model(inputs)
        model.reset_work()
        recomputed, _ = decode_fixed_inputs(model, inputs, args.prefix, cached=False)
        recompute_work = {"projected_positions": model.projected_positions,
                          "dense_score_elements": model.dense_score_elements}
        model.reset_work()
        incremental, cache = decode_fixed_inputs(model, inputs, args.prefix, cached=True)
        cached_work = {"projected_positions": model.projected_positions,
                       "dense_score_elements": model.dense_score_elements}
        errors = {"full_vs_recomputed_max_abs": float((full - recomputed).abs().max()),
                  "full_vs_cached_max_abs": float((full - incremental).abs().max())}
        torch.testing.assert_close(full, incremental, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(full, recomputed, rtol=1e-4, atol=1e-5)
        query, key, value = model.project(inputs)
        reference = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0)
        reference = model.output(reference.transpose(1, 2).contiguous().view_as(inputs))
        torch.testing.assert_close(full, reference, rtol=1e-4, atol=1e-5)
        errors["handwritten_vs_torch_sdpa_max_abs"] = float((full - reference).abs().max())
        # 故意复现一个错误：decode 的矩形 mask 每次从第 0 行重新 tril。
        next_query = query[:, :, args.prefix:args.prefix + 1]
        past_keys, past_values = key[:, :, :args.prefix + 1], value[:, :, :args.prefix + 1]
        right_mask = causal_mask(1, args.prefix + 1, args.prefix, inputs.device)
        wrong_mask = torch.ones_like(right_mask).tril()
        wrong_error = float((model.attend(next_query, past_keys, past_values, right_mask)
                             - model.attend(next_query, past_keys, past_values, wrong_mask)).abs().max())
        recomputed_timing = measure(lambda: decode_fixed_inputs(model, inputs, args.prefix, cached=False, collect=False), args.repeats)
        cached_timing = measure(lambda: decode_fixed_inputs(model, inputs, args.prefix, cached=True, collect=False), args.repeats)
    result = {
        "created_at": datetime.now().astimezone().isoformat(), "device": args.device,
        "torch": torch.__version__, "dtype": str(inputs.dtype), "seed": 42, "cpu_threads": 4,
        "input_shape": list(inputs.shape), "heads": args.heads, "head_dim": model.head_dim,
        "prefix": args.prefix, "decode_steps": args.decode, "repeats": args.repeats,
        "equivalence_passed": True, "errors": errors,
        "cache_shape": list(cache.key.shape), "cache_logical_bytes": cache.logical_bytes,
        "cache_formula_bytes": 2 * args.batch * args.heads * total * model.head_dim * inputs.element_size(),
        "wrong_mask_demo": {"correct_visible_keys": int(right_mask.sum()),
                            "wrong_visible_keys": int(wrong_mask.sum()), "max_abs_error": wrong_error},
        "recomputed": {"work": recompute_work, "timing": recomputed_timing},
        "cached": {"work": cached_work, "timing": cached_timing},
        "measured_speed_ratio": recomputed_timing["median_ms"] / cached_timing["median_ms"],
        "source_sha256": sha256(PROJECT_ROOT / "src/attention_lab.py"),
        "limitations": ["单层随机权重注意力，不是完整 Transformer 或已训练 LLM",
                        "固定输入、相同前缀，未执行自回归 token 采样", "未实现位置编码、padding mask、GQA、分页或滑动窗口",
                        "动态 torch.cat 会复制旧缓存；工程实现通常预分配或分块管理",
                        "小张量 Python/分配/核启动开销可能占主导，不保证缓存总是更快"]}
    path = PROJECT_ROOT / "data/generated" / (f"attention_{args.device}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(path, result)
    print(f"输入 {list(inputs.shape)}；KV {result['cache_shape']}；{cache.logical_bytes} bytes")
    print("数值一致性：通过；最大误差：", errors)
    print("QKV 处理的位置数：全前缀重算", recompute_work["projected_positions"], "→ 缓存", cached_work["projected_positions"])
    print(f"故意错误的 mask：可见 key {int(right_mask.sum())} → {int(wrong_mask.sum())}；误差 {wrong_error:.6f}")
    print(f"热计时：重算 {recomputed_timing['median_ms']:.3f} ms；缓存 {cached_timing['median_ms']:.3f} ms；比值 {result['measured_speed_ratio']:.2f}")
    print(f"报告：{path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prefix", type=int, default=64)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not (1 <= args.prefix <= 512 and 1 <= args.decode <= 64 and 1 <= args.batch <= 4
            and 1 <= args.repeats <= 20 and 1 <= args.heads <= 16 and 1 <= args.d_model <= 512):
        parser.error("实验规模超限；请保留为可观察的小张量实验")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用；先使用 --device cpu")
    run_experiment(args)


if __name__ == "__main__":
    main()
