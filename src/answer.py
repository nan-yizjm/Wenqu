"""基于本地 Markdown 知识库的 RAG 问答流程。"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Index, search_bm25
from .context import Source, build_context
from .llm import ChatClient, DeepSeekClient, OllamaClient
from .query_guard import static_corpus_rejection_reason
from .retrieve import load_chunks
from .citations import CitationValidation, validate_citations


SYSTEM_PROMPT = """你是一个个人大模型知识库问答助手。

你只能依据用户问题后提供的“知识库资料”回答，不得使用资料之外的
模型常识补充事实。

必须遵守以下规则：
1. 只陈述资料中明确支持的事实。
2. 每个重要结论后都要就近标注来源编号，例如 [S1]。
3. 如果资料不足以支持问题的某一部分，只说明资料不足，不要继续解释该部分。
4. 禁止使用“推测”“可能”“也许”“大概”“一般而言”“根据常识”等表达。
5. 禁止在同一个回答中先说“资料不足”，再继续推断资料外结论。
6. 不要编造来源编号，也不要引用没有提供的资料。
7. 回答尽量简洁，优先输出 1 到 3 条资料明确支持的结论。
"""


@dataclass(frozen=True)
class AnswerResult:
    """一次 RAG 问答的最终结果。"""

    answer: str
    sources: tuple[Source, ...]
    rejected: bool
    citation_validation: CitationValidation | None = None


class RAGAnswerer:
    """连接查询边界、BM25 检索、上下文构建与模型生成。"""

    def __init__(
        self,
        chunks_path: Path,
        client: ChatClient,
        retrieval_top_k: int = 5,
        max_context_chars: int = 2200,
    ) -> None:
        self.client = client
        self.retrieval_top_k = retrieval_top_k
        self.max_context_chars = max_context_chars

        self.chunks = load_chunks(chunks_path)
        self.index = BM25Index(self.chunks)

    def answer(self, question: str) -> AnswerResult:
        """回答一个问题，必要时拒绝超出静态知识库能力边界的请求。"""
        question = question.strip()

        if not question:
            raise ValueError("问题不能为空")

        rejection_reason = static_corpus_rejection_reason(question)
        if rejection_reason:
            return AnswerResult(
                answer=f"当前知识库助手无法处理这个请求：{rejection_reason}",
                sources=(),
                rejected=True,
            )

        retrieval_results = search_bm25(
            query=question,
            chunks=self.chunks,
            index=self.index,
            top_k=self.retrieval_top_k,
        )

        context_package = build_context(
            retrieval_results,
            max_chars=self.max_context_chars,
        )

        if not context_package.sources:
            return AnswerResult(
                answer=(
                    "当前知识库中没有检索到足以回答这个问题的相关资料。"
                ),
                sources=(),
                rejected=False,
            )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    "知识库资料：\n"
                    f"{context_package.text}\n\n"
                    "请严格依据上述资料回答，并在重要结论后标注 [S1]、[S2] "
                    "等对应来源。"
                ),
            },
        ]

        answer = self.client.chat(messages)

        citation_validation = validate_citations(
            answer,
            context_package.sources,
        )

        return AnswerResult(
            answer=answer,
            sources=context_package.sources,
            rejected=False,
            citation_validation=citation_validation,
        )


def create_client(provider: str) -> ChatClient:
    """根据供应商名称创建模型客户端。"""
    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ),
        )

    if provider == "deepseek":
        return DeepSeekClient(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )

    raise ValueError(
        f"不支持的模型供应商：{provider}；可选值为 ollama 或 deepseek"
    )


def print_answer_result(result: AnswerResult) -> None:
    """以适合命令行阅读的格式输出回答与来源。"""
    print("\n回答：")
    print(result.answer)

    if result.sources:
        print("\n引用来源：")
        for source in result.sources:
            print(
                f"[{source.label}] "
                f"{source.source_file} "
                f"→ {source.heading_path}"
            )

    if result.citation_validation is not None:
        validation = result.citation_validation

        print("\n引用校验：")
        if validation.is_valid:
            labels = ", ".join(validation.cited_labels)
            print(f"通过：回答引用的标签均来自本次上下文（{labels}）。")
        elif not validation.has_citations:
            print("警告：模型回答中没有找到任何来源引用。")
        else:
            labels = ", ".join(validation.invalid_labels)
            print(f"警告：回答引用了不存在的标签：{labels}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="向个人 Obsidian 知识库提问。",
    )
    parser.add_argument(
        "question",
        help="要询问的问题。",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "deepseek"),
        default=os.getenv("RAG_LLM_PROVIDER", "ollama"),
        help="使用本地 Ollama 或 DeepSeek API。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="先从 BM25 取回多少条候选片段。",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=2200,
        help="允许交给模型的知识库资料最大字符数。",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "data" / "generated" / "chunks.json"

    answerer = RAGAnswerer(
        chunks_path=chunks_path,
        client=create_client(args.provider),
        retrieval_top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )

    result = answerer.answer(args.question)
    print_answer_result(result)


if __name__ == "__main__":
    main()