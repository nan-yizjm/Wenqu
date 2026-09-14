"""用实验 QLoRA adapter 回答真实版本化知识库；不会修改 rag.toml 或默认后端。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from .answer import RAGAnswerer, print_answer_result
from .config import PROJECT_ROOT
from .evaluate_sft_challenge import DEFAULT_CHALLENGE, load_challenge
from .experiment_utils import save_report, sha256
from .hf_qlora_client import HFQLoRAClient
from .index_store import IndexStore
from .settings import DEFAULT_CONFIG, load_settings
from .vector_retrieve import VectorIndex

SFT_SYSTEM_PROMPT = load_challenge(DEFAULT_CHALLENGE)["system"]


def sft_messages(question: str, context: str):
    return [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}\n\n证据：\n{context}"},
    ]


def build_answerer(config_path, adapter, max_new_tokens=160, max_prompt_tokens=2048):
    config_path = Path(config_path).resolve()
    if type(max_prompt_tokens) is not int or max_prompt_tokens < 32:
        raise ValueError("max_prompt_tokens 必须是至少 32 的整数")
    settings = load_settings(config_path)
    store = IndexStore(settings.store_dir)
    selected = store.current()
    if selected is None:
        raise RuntimeError("没有已发布索引，请先运行 python -m src.index_store build")
    version_path, manifest = selected
    if manifest["vault_dir"] != str(settings.vault_dir.resolve()):
        raise ValueError("当前索引不属于配置中的笔记目录")
    client = HFQLoRAClient(adapter, max_new_tokens=max_new_tokens)
    if max_prompt_tokens + max_new_tokens > client.max_context_tokens:
        raise ValueError("prompt 与生成预算之和超过模型上下文窗口")
    answerer = RAGAnswerer(
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
        answerer.engine.vector = VectorIndex(
            answerer.engine.chunks,
            device=settings.device,
            cache_path=version_path / "vector_index.npz",
            read_only=True,
        )
    if settings.rerank:
        answerer.engine.ensure_reranker()
    return answerer, manifest["version"], settings, config_path, version_path


def run(question, adapter, config_path=DEFAULT_CONFIG, max_new_tokens=160,
        max_prompt_tokens=2048, output=None):
    answerer, version, settings, config_path, version_path = build_answerer(
        config_path, adapter, max_new_tokens=max_new_tokens,
        max_prompt_tokens=max_prompt_tokens)
    result = answerer.answer(question)
    validation = result.citation_validation
    report = {
        "schema": "experimental-adapter-rag-run-v1",
        "mode": "experimental_hf_qlora",
        "model": answerer.client.model,
        "adapter": str(answerer.client.adapter),
        "adapter_artifacts": answerer.client.adapter_artifacts,
        "index_version": version,
        "index_manifest_sha256": sha256(version_path / "manifest.json"),
        "config": settings.public_dict(),
        "config_sha256": sha256(config_path),
        "prompt_contract": {
            "system_source": str(DEFAULT_CHALLENGE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "system_source_sha256": sha256(DEFAULT_CHALLENGE),
            "evidence_prefix": "问题/证据",
            "concept_repair": False,
            "max_prompt_tokens": max_prompt_tokens,
        },
        "question": question,
        "answer": result.answer,
        "rejected_by_static_guard": result.rejected,
        "sources": [asdict(source) for source in result.sources],
        "citation_validation": ({**asdict(validation), "is_valid": validation.is_valid}
                                if validation else None),
        "retrieval_method": result.retrieval_method,
        "retrieval_trace": list(result.retrieval_trace),
        "context": result.context_text,
        "context_diagnostics": list(result.context_diagnostics),
        "generation_calls": result.generation_calls,
        "local_model_metrics": answerer.client.last_metrics,
        "source_signature": {
            "src/adapter_rag.py": sha256(Path(__file__)),
            "src/hf_qlora_client.py": sha256(PROJECT_ROOT / "src/hf_qlora_client.py"),
            "src/answer.py": sha256(PROJECT_ROOT / "src/answer.py"),
            "src/evidence_context.py": sha256(PROJECT_ROOT / "src/evidence_context.py"),
            "src/hybrid_retrieve.py": sha256(PROJECT_ROOT / "src/hybrid_retrieve.py"),
        },
        "warning": "实验 adapter 存在部分证据过度作答回归；引用标签合法不等于每个陈述都有证据。",
    }
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated" / (
        "adapter_rag_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, report)
    return result, report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result, report, output = run(
        args.question, args.adapter, args.config, args.max_new_tokens,
        args.max_prompt_tokens, args.output)
    print(json.dumps({
        "mode": report["mode"],
        "model": report["model"],
        "adapter": report["adapter"],
        "adapter_artifacts": report["adapter_artifacts"],
        "index_version": report["index_version"],
        "retriever": report["retrieval_method"],
        "context_policy": report["config"]["context_policy"],
        "concept_repair": report["prompt_contract"]["concept_repair"],
    }, ensure_ascii=False, indent=2))
    print_answer_result(result)
    if report["local_model_metrics"]:
        print("\n本地模型指标：")
        print(json.dumps(report["local_model_metrics"], ensure_ascii=False, indent=2))
    print(f"\n实验报告：{output}")


if __name__ == "__main__":
    main()
