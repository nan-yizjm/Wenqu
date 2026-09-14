"""教学用 decoder-only 语言模型：字符 ID → logits；不是训练好的通用 LLM。"""

from dataclasses import dataclass

import torch
from torch import nn

from .attention_lab import CausalSelfAttention, KVCache


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        for key in ("vocab_size", "max_seq_len", "d_model", "heads", "layers"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} 必须为正整数")
        if self.vocab_size < 4 or self.d_model % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("词表至少 4 项、d_model 可整除 heads，dropout 在 [0,1)")


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config.d_model, config.heads)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, 4 * config.d_model), nn.GELU(),
                                 nn.Linear(4 * config.d_model, config.d_model))
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cache=None, use_cache=False):
        attended, cache = self.attention(self.attn_norm(x), cache, use_cache=use_cache)
        x = x + self.residual_dropout(attended)
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x, cache


class TinyDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, caches: tuple[KVCache, ...] | None = None, *, use_cache=False):
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0 or ids.dtype != torch.long:
            raise ValueError("输入必须为非空的 LongTensor [batch, tokens]")
        if caches is not None and not use_cache:
            raise ValueError("传入缓存时必须 use_cache=True")
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError("缓存生成需要 model.eval() 和 torch.inference_mode()")
        past = 0
        if caches is not None:
            if len(caches) != len(self.blocks) or len({item.length for item in caches}) != 1:
                raise ValueError("各层缓存数量/长度不一致")
            past = caches[0].length
        if past + ids.shape[1] > self.config.max_seq_len:
            raise ValueError("超过位置编码窗口；本版不静默重置位置或裁剪缓存")
        # 追加 token 的位置从 past 开始，不能每个 decode 步骤从 0 开始。
        positions = torch.arange(past, past + ids.shape[1], device=ids.device)
        x = self.input_dropout(self.token_embedding(ids) + self.position_embedding(positions)[None])
        new_caches = []
        for index, block in enumerate(self.blocks):
            x, cache = block(x, caches[index] if caches else None, use_cache)
            if use_cache:
                new_caches.append(cache)
        return self.lm_head(self.final_norm(x)), tuple(new_caches) if use_cache else None

    @torch.inference_mode()
    def generate(self, prefix_ids: list[int], max_new_tokens=48):
        """单样本贪心生成；不使用温度采样，便于观察训练前后的变化。"""
        if max_new_tokens < 1 or not prefix_ids or len(prefix_ids) >= self.config.max_seq_len:
            raise ValueError("前缀为空、已占满窗口或 max_new_tokens 不合法")
        was_training = self.training
        self.eval()
        try:
            ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.token_embedding.weight.device)
            logits, cache = self(ids, use_cache=True)
            output = list(prefix_ids)
            available = self.config.max_seq_len - len(prefix_ids)
            stop = "context_limit" if max_new_tokens >= available else "max_new_tokens"
            for index in range(min(max_new_tokens, available)):
                scores = logits[0, -1].clone()
                scores[0:2] = float("-inf")  # 不生成 PAD/BOS；EOS=2，UNK=3。
                next_id = int(scores.argmax())
                output.append(next_id)
                if next_id == 2:
                    stop = "eos"
                    break
                if index + 1 < min(max_new_tokens, available):
                    logits, cache = self(torch.tensor([[next_id]], device=ids.device), cache, use_cache=True)
            return {"ids": output, "stop_reason": stop, "generated_tokens": len(output) - len(prefix_ids)}
        finally:
            self.train(was_training)
