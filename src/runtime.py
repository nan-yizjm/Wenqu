"""CLI / HTTP 共用的版本固定运行时；启动后不会偷偷换语料。"""

import argparse
from dataclasses import asdict
import json
import time
from urllib.request import urlopen

from .answer import RAGAnswerer, create_client
from .index_store import IndexStore
from .settings import DEFAULT_CONFIG, Settings, load_settings
from .vector_retrieve import VectorIndex


class RAGRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = IndexStore(settings.store_dir)
        selected = self.store.current()
        if selected is None:
            raise RuntimeError("没有已发布索引，请先运行 python -m src.index_store build")
        path, manifest = selected
        if manifest["vault_dir"] != str(settings.vault_dir.resolve()):
            raise ValueError("当前索引不属于配置中的笔记目录，请先 build 新版本")
        self.version = manifest["version"]
        client = create_client(settings.provider)
        self.answerer = RAGAnswerer(
            path / "chunks.json", client, retrieval_method=settings.retriever,
            retrieval_top_k=settings.top_k, max_context_chars=settings.max_context_chars,
            candidate_k=settings.candidate_k, rrf_k=settings.rrf_k,
            device=settings.device, rerank=settings.rerank,
            rerank_top_n=settings.rerank_top_n, rerank_device=settings.rerank_device,
            context_policy=settings.context_policy)
        engine = self.answerer.engine
        if settings.retriever in ("hybrid", "vector"):
            # 严格只读：升级库导致缓存不兼容时，不能在历史版本目录内重写文件。
            engine.vector = VectorIndex(engine.chunks, device=settings.device,
                                        cache_path=path / "vector_index.npz", read_only=True)
        if settings.rerank:
            engine.ensure_reranker()

    def info(self):
        return {"index_version": self.version, "provider": self.settings.provider,
                "model": self.answerer.client.model, "retriever": self.settings.retriever,
                "rerank": self.settings.rerank, "context_policy": self.settings.context_policy,
                "chunks": len(self.answerer.chunks), "version_policy": "pinned_until_restart"}

    def backend_ready(self) -> bool:
        if self.settings.provider != "ollama":
            # 不为健康检查消耗 API 费用；只能证明客户端已配置，不能证明远端可用。
            return True
        try:
            client = self.answerer.client
            with urlopen(client.base_url + "/api/tags", timeout=2) as response:
                names = {item["name"] for item in json.load(response).get("models", [])}
            model = client.model if ":" in client.model else client.model + ":latest"
            return model in names
        except (OSError, ValueError, KeyError):
            return False

    def answer(self, question: str):
        question = question.strip()
        if not question or len(question) > self.settings.max_question_chars:
            raise ValueError("问题为空或超过配置的长度限制")
        started = time.perf_counter()
        result = self.answerer.answer(question)
        validation = result.citation_validation
        return {
            **self.info(), "answer": result.answer, "rejected": result.rejected,
            "sources": [asdict(source) for source in result.sources],
            "generation_calls": result.generation_calls,
            "retrieval_method": result.retrieval_method,
            "citation_validation": ({**asdict(validation), "is_valid": validation.is_valid}
                                    if validation else None),
            "phase_timings_ms": result.phase_timings_ms,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    runtime = RAGRuntime(load_settings(args.config))
    print(json.dumps(runtime.info(), ensure_ascii=False, indent=2))
    if args.question:
        print(json.dumps(runtime.answer(args.question), ensure_ascii=False, indent=2))
        return
    print("输入完整问题，:quit 退出；本进程固定以上索引版本，每题独立。")
    while True:
        try:
            question = input("\n> ").strip()
            if question == ":quit":
                break
            if question:
                result = runtime.answer(question)
                print(result["answer"])
                for source in result["sources"]:
                    print(f"[{source['label']}] {source['source_file']} → {source['heading_path']}")
                print(f"版本：{result['index_version']}；耗时：{result['elapsed_ms']} ms")
        except (EOFError, KeyboardInterrupt):
            break
        except (ValueError, RuntimeError, OSError) as error:
            print(f"本次请求失败：{type(error).__name__}；请检查索引和模型服务。")


if __name__ == "__main__":
    main()
