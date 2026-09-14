"""固定 Qwen2.5 1.5B 基座的 4-bit NF4 加载方式，供训练和评估共用。"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .sft_tokenization import MODEL_NAME, MODEL_REVISION, load_tokenizer


def quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base(*, allow_download=False, training=False):
    if not torch.cuda.is_available():
        raise ValueError("QLoRA 入口需要 CUDA")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("当前 GPU/PyTorch 未报告 BF16 支持")
    tokenizer = load_tokenizer(allow_download)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=not allow_download,
        quantization_config=quantization_config(),
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = not training
    return model, tokenizer


def load_for_inference(*, adapter=None, allow_download=False):
    model, tokenizer = load_base(allow_download=allow_download, training=False)
    if adapter:
        from peft import PeftModel
        adapter = Path(adapter).resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise ValueError("adapter 目录缺少 adapter_config.json")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer
