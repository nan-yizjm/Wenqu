"""本地 Hugging Face QLoRA 推理客户端；显式实验，不替换 Ollama 默认。"""

from pathlib import Path
from queue import Empty
import threading
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from .experiment_utils import sha256, synchronize, torch_memory
from .qlora_model import load_for_inference
from .sft_tokenization import MODEL_NAME, MODEL_REVISION


class EventStoppingCriteria(StoppingCriteria):
    """在每个 decode 步读取线程事件，并记录模型确实观察到了取消。"""

    def __init__(self, event: threading.Event):
        self.event = event
        self.checks = 0
        self.observed_event = False

    def __call__(self, input_ids, scores, **kwargs):
        self.checks += 1
        should_stop = self.event.is_set()
        self.observed_event |= should_stop
        return torch.full(
            (input_ids.shape[0],),
            should_stop,
            dtype=torch.bool,
            device=input_ids.device,
        )


class HFQLoRAClient:
    """实现 RAG 所需的最小 chat 接口，并保留本次调用的 GPU 指标。"""

    def __init__(self, adapter: Path, max_new_tokens: int = 160):
        adapter = Path(adapter).resolve()
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens 必须位于 1..512")
        required = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
        if not all(path.is_file() for path in required):
            raise ValueError("adapter 目录缺少 config 或 safetensors 权重")
        started = time.perf_counter()
        self._model, self._tokenizer = load_for_inference(adapter=adapter, allow_download=False)
        synchronize()
        self.adapter = adapter
        self.max_new_tokens = max_new_tokens
        self.model = f"{MODEL_NAME}@{MODEL_REVISION[:12]} + {adapter.name}"
        self.load_ms = (time.perf_counter() - started) * 1000
        self.adapter_artifacts = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in required
        }
        self.last_metrics = {}
        self._generation_lock = threading.Lock()
        self.max_context_tokens = int(self._model.config.max_position_embeddings)

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        """按实际 chat template 计数，不创建 CUDA tensor，也不做截断。"""
        if not messages or any(message.get("role") not in ("system", "user", "assistant")
                               or not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("messages 必须是非空的 system/user/assistant 文本消息")
        token_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        if hasattr(token_ids, "keys"):
            token_ids = token_ids["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            return int(token_ids.shape[-1])
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            token_ids = token_ids[0]
        return len(token_ids)

    def _inputs(self, messages):
        if not messages or any(message.get("role") not in ("system", "user", "assistant")
                               or not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("messages 必须是非空的 system/user/assistant 文本消息")
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

    def chat(self, messages: list[dict[str, str]]) -> str:
        with self._generation_lock:
            inputs = self._inputs(messages)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            synchronize()
            generated = output[0, prompt_tokens:]
            answer = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not answer:
                raise RuntimeError("本地 QLoRA 模型返回空回答")
            self.last_metrics = {
                "load_ms": self.load_ms,
                "generation_ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(generated.numel()),
                "stop_reason": "eos_or_length",
                "streamed": False,
                "torch_cuda_memory": torch_memory(),
            }
            return answer

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        cancel_event: threading.Event | None = None,
    ):
        """逐段产生文本；关闭迭代器或设置事件都会请求模型在下一 decode 步停止。"""
        cancel_event = cancel_event or threading.Event()
        inputs = self._inputs(messages)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            timeout=30.0,
        )
        criterion = EventStoppingCriteria(cancel_event)
        state = {"output": None, "error": None}

        def generate():
            try:
                with torch.inference_mode():
                    state["output"] = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([criterion]),
                        streamer=streamer,
                    )
            except BaseException as error:
                state["error"] = error
                # generate 异常时未必会发送 stop_signal，主动结束迭代器。
                streamer.end()

        with self._generation_lock:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            worker = threading.Thread(target=generate, name="hf-qlora-generate", daemon=False)
            worker.start()
            chunks = 0
            first_chunk_ms = None
            stream_exhausted = False
            try:
                while True:
                    try:
                        text = next(streamer)
                    except StopIteration:
                        stream_exhausted = True
                        break
                    except Empty as error:
                        cancel_event.set()
                        raise RuntimeError("等待本地模型流式输出超时") from error
                    if text:
                        chunks += 1
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - started) * 1000
                        yield text
            finally:
                # 消费者提前 break/断开时，generator.close() 会走到这里。
                # 自然读到 stop_signal 后，generate 线程可能只差函数返回，不应误记为取消。
                if not stream_exhausted and worker.is_alive():
                    cancel_event.set()
                worker.join(timeout=30)
                if worker.is_alive():
                    raise RuntimeError("已请求取消，但本地生成线程未在 30 秒内退出")
                synchronize()
                output = state["output"]
                generated_tokens = (int(output.shape[-1]) - prompt_tokens) if output is not None else 0
                self.last_metrics = {
                    "load_ms": self.load_ms,
                    "generation_ms": (time.perf_counter() - started) * 1000,
                    "first_chunk_ms": first_chunk_ms,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "stream_chunks": chunks,
                    "streamed": True,
                    "cancel_requested": cancel_event.is_set(),
                    "cancel_observed_by_model": criterion.observed_event,
                    "stopping_checks": criterion.checks,
                    "stop_reason": "cancelled" if criterion.observed_event else "eos_or_length",
                    "worker_alive_after_join": worker.is_alive(),
                    "torch_cuda_memory": torch_memory(),
                }
            if state["error"] is not None:
                raise RuntimeError("本地流式生成失败") from state["error"]
