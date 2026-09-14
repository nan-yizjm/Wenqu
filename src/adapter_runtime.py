"""常驻 QLoRA adapter 的实验运行时；一次加载，多次问答。"""

from dataclasses import asdict
from pathlib import Path
import threading
import time

import torch

from .adapter_rag import sft_messages
from .answer import RAGAnswerer, combine_phase_timings
from .experiment_utils import sha256
from .hf_qlora_client import HFQLoRAClient
from .index_store import IndexStore
from .settings import Settings
from .vector_retrieve import VectorIndex


class AdapterRAGRuntime:
    """把固定 adapter、固定索引和向量模型保留在同一个本机进程内。"""

    provider = "hf-qlora-experimental"

    def __init__(self, settings: Settings, adapter: Path, max_new_tokens: int = 160,
                 max_prompt_tokens: int = 2048):
        started = time.perf_counter()
        self.settings = settings
        self.store = IndexStore(settings.store_dir)
        selected = self.store.current()
        if selected is None:
            raise RuntimeError("没有已发布索引，请先运行 python -m src.index_store build")
        version_path, manifest = selected
        if manifest["vault_dir"] != str(settings.vault_dir.resolve()):
            raise ValueError("当前索引不属于配置中的笔记目录")

        self.version = manifest["version"]
        self.version_path = version_path
        if type(max_prompt_tokens) is not int or max_prompt_tokens < 32:
            raise ValueError("max_prompt_tokens 必须是至少 32 的整数")
        client = HFQLoRAClient(adapter, max_new_tokens=max_new_tokens)
        if max_prompt_tokens + max_new_tokens > client.max_context_tokens:
            raise ValueError(
                "max_prompt_tokens + max_new_tokens 超过模型上下文窗口："
                f"{max_prompt_tokens} + {max_new_tokens} > {client.max_context_tokens}"
            )
        self.answerer = RAGAnswerer(
            version_path / "chunks.json",
            client,
            retrieval_top_k=settings.top_k,
            max_context_chars=settings.max_context_chars,
            retrieval_method=settings.retriever,
            device=settings.device,
            candidate_k=settings.candidate_k,
            rerank=settings.rerank,
            rerank_top_n=settings.rerank_top_n,
            rerank_device=settings.rerank_device,
            context_policy=settings.context_policy,
            message_builder=sft_messages,
            enable_concept_repair=False,
            message_token_counter=client.count_chat_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
        if settings.retriever in ("hybrid", "vector"):
            self.answerer.engine.vector = VectorIndex(
                self.answerer.engine.chunks,
                device=settings.device,
                cache_path=version_path / "vector_index.npz",
                read_only=True,
            )
        if settings.rerank:
            self.answerer.engine.ensure_reranker()

        self.startup_ms = round((time.perf_counter() - started) * 1000, 2)
        self.request_count = 0
        self.last_request_summary = None

    def info(self):
        client = self.answerer.client
        return {
            "index_version": self.version,
            "provider": self.provider,
            "model": client.model,
            "adapter": str(client.adapter),
            "adapter_artifacts": client.adapter_artifacts,
            "retriever": self.settings.retriever,
            "rerank": self.settings.rerank,
            "context_policy": self.settings.context_policy,
            "chunks": len(self.answerer.chunks),
            "max_new_tokens": client.max_new_tokens,
            "max_prompt_tokens": self.answerer.max_prompt_tokens,
            "model_context_tokens": client.max_context_tokens,
            "startup_ms": self.startup_ms,
            "startup_breakdown": {
                "model_adapter_load_ms": round(client.load_ms, 2),
                "index_retriever_and_other_ms": round(max(0.0, self.startup_ms - client.load_ms), 2),
            },
            "index_manifest_sha256": sha256(self.version_path / "manifest.json"),
            "requests_completed": self.request_count,
            "last_request_summary": self.last_request_summary,
            "version_policy": "pinned_until_restart",
            "warning": "实验 adapter；引用标签合法不等于回答完整或陈述均被证据支持。",
        }

    def backend_ready(self) -> bool:
        client = self.answerer.client
        return (
            torch.cuda.is_available()
            and getattr(client, "_model", None) is not None
            and (client.adapter / "adapter_config.json").is_file()
            and (client.adapter / "adapter_model.safetensors").is_file()
        )

    def answer(self, question: str):
        question = question.strip()
        if not question or len(question) > self.settings.max_question_chars:
            raise ValueError("问题为空或超过配置的长度限制")

        # OOD guard 不会调用模型；先清空可防止把上一题生成指标误挂到拒答上。
        self.answerer.client.last_metrics = {}
        started = time.perf_counter()
        result = self.answerer.answer(question)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.request_count += 1
        validation = result.citation_validation
        model_metrics = dict(self.answerer.client.last_metrics)
        generation_ms = float(model_metrics.get("generation_ms", 0.0))
        payload = {
            **self.info(),
            "request_sequence": self.request_count,
            "answer": result.answer,
            "rejected": result.rejected,
            "sources": [asdict(source) for source in result.sources],
            "generation_calls": result.generation_calls,
            "citation_validation": ({**asdict(validation), "is_valid": validation.is_valid}
                                    if validation else None),
            "retrieval_method": result.retrieval_method,
            "retrieval_trace": list(result.retrieval_trace),
            "context_diagnostics": list(result.context_diagnostics),
            "prompt_tokens": result.prompt_tokens,
            "prompt_token_budget": result.prompt_token_budget,
            "context_char_budget": result.context_char_budget,
            "phase_timings_ms": result.phase_timings_ms,
            "local_model_metrics": model_metrics,
            "non_generation_ms": round(max(0.0, elapsed_ms - generation_ms), 2),
            "elapsed_ms": round(elapsed_ms, 2),
        }
        self.last_request_summary = {
            "request_sequence": self.request_count,
            "mode": "complete",
            "rejected": result.rejected,
            "generation_calls": result.generation_calls,
            "stop_reason": model_metrics.get("stop_reason"),
            "elapsed_ms": payload["elapsed_ms"],
            "phase_timings_ms": result.phase_timings_ms,
        }
        payload["last_request_summary"] = self.last_request_summary
        return payload

    def stream_answer(self, question: str, cancel_event: threading.Event):
        """产生结构化流事件；取消的半截文本不会伪装成最终答案。"""
        question = question.strip()
        if not question or len(question) > self.settings.max_question_chars:
            raise ValueError("问题为空或超过配置的长度限制")
        if not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event 必须是 threading.Event")

        self.answerer.client.last_metrics = {}
        started = time.perf_counter()
        sequence = self.request_count + 1
        prepared = self.answerer.prepare(question)

        if prepared.immediate_result is not None:
            self.request_count += 1
            result = prepared.immediate_result
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            phase_timings = combine_phase_timings(
                prepared.prepare_timings_ms, 0.0, 0.0, elapsed_ms
            )
            self.last_request_summary = {
                "request_sequence": sequence,
                "mode": "complete",
                "rejected": result.rejected,
                "generation_calls": 0,
                "stop_reason": "guard_or_no_context",
                "elapsed_ms": elapsed_ms,
                "phase_timings_ms": phase_timings,
            }
            yield {
                "type": "final",
                "complete": True,
                "request_sequence": sequence,
                **self._result_payload(result, started, {}, phase_timings),
            }
            return

        yield {
            "type": "metadata",
            "complete": False,
            "request_sequence": sequence,
            "index_version": self.version,
            "model": self.answerer.client.model,
            "sources": [asdict(source) for source in prepared.sources],
            "retrieval_method": self.settings.retriever,
            "prompt_tokens": prepared.prompt_tokens,
            "prompt_token_budget": prepared.prompt_token_budget,
            "context_char_budget": prepared.context_char_budget,
        }
        pieces = []
        generation_started = time.perf_counter()
        for text in self.answerer.client.stream_chat(
            list(prepared.messages), cancel_event=cancel_event
        ):
            pieces.append(text)
            yield {"type": "token", "text": text}
        generation_wall_ms = (time.perf_counter() - generation_started) * 1000

        metrics = dict(self.answerer.client.last_metrics)
        partial_or_full = "".join(pieces).strip()
        self.request_count += 1
        if metrics.get("cancel_observed_by_model"):
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            phase_timings = combine_phase_timings(
                prepared.prepare_timings_ms, generation_wall_ms, 0.0, elapsed_ms
            )
            self.last_request_summary = {
                "request_sequence": sequence,
                "mode": "cancelled",
                "rejected": False,
                "generation_calls": 1,
                "stop_reason": "cancelled",
                "elapsed_ms": elapsed_ms,
                "generated_tokens": metrics.get("generated_tokens"),
                "phase_timings_ms": phase_timings,
            }
            yield {
                "type": "cancelled",
                "complete": False,
                "request_sequence": sequence,
                "index_version": self.version,
                "partial_answer": partial_or_full,
                "generation_calls": 1,
                "citation_validation": None,
                "local_model_metrics": metrics,
                "phase_timings_ms": phase_timings,
                "elapsed_ms": elapsed_ms,
            }
            return

        if not partial_or_full:
            raise RuntimeError("本地 QLoRA 流自然结束但没有产生可见文本")
        finalize_started = time.perf_counter()
        result = self.answerer.finalize(prepared, partial_or_full, generation_calls=1)
        finalize_ms = (time.perf_counter() - finalize_started) * 1000
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        phase_timings = combine_phase_timings(
            prepared.prepare_timings_ms, generation_wall_ms, finalize_ms, elapsed_ms
        )
        self.last_request_summary = {
            "request_sequence": sequence,
            "mode": "complete",
            "rejected": False,
            "generation_calls": 1,
            "stop_reason": metrics.get("stop_reason"),
            "elapsed_ms": elapsed_ms,
            "generated_tokens": metrics.get("generated_tokens"),
            "phase_timings_ms": phase_timings,
        }
        yield {
            "type": "final",
            "complete": True,
            "request_sequence": sequence,
            **self._result_payload(result, started, metrics, phase_timings),
        }

    def _result_payload(self, result, started, model_metrics, phase_timings=None):
        elapsed_ms = (time.perf_counter() - started) * 1000
        validation = result.citation_validation
        generation_ms = float(model_metrics.get("generation_ms", 0.0))
        return {
            "index_version": self.version,
            "answer": result.answer,
            "rejected": result.rejected,
            "sources": [asdict(source) for source in result.sources],
            "generation_calls": result.generation_calls,
            "citation_validation": ({**asdict(validation), "is_valid": validation.is_valid}
                                    if validation else None),
            "retrieval_method": result.retrieval_method,
            "prompt_tokens": result.prompt_tokens,
            "prompt_token_budget": result.prompt_token_budget,
            "context_char_budget": result.context_char_budget,
            "phase_timings_ms": phase_timings or result.phase_timings_ms,
            "local_model_metrics": model_metrics,
            "non_generation_ms": round(max(0.0, elapsed_ms - generation_ms), 2),
            "elapsed_ms": round(elapsed_ms, 2),
        }
