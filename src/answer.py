"""基于本地 Markdown 知识库的 RAG 问答流程。"""

import argparse
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .hybrid_retrieve import RetrievalEngine
from .context import Source, build_context
from .evidence_context import build_evidence_context
from .llm import ChatClient, DeepSeekClient, OllamaClient
from .query_guard import static_corpus_rejection_reason
from .retrieve import load_chunks
from .citations import CitationValidation, validate_citations
from .config import load_env_file


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
8. 当问题列出多个概念并要求“分别”说明时，必须逐项回答每个概念；
   若资料只支持它们的共同结论，应明确说明无法依据资料区分细节。
"""


TECHNICAL_TERM_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{1,}\b")
BULLET_LINE_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def requested_separate_concepts(question: str) -> tuple[str, ...]:
    """提取“分别”之前列出的英文技术概念。"""
    if "分别" not in question:
        return ()

    prefix = question.split("分别", maxsplit=1)[0]
    concepts: list[str] = []

    for match in TECHNICAL_TERM_PATTERN.finditer(prefix):
        concept = match.group()

        if concept not in concepts:
            concepts.append(concept)

    return tuple(concepts)


def has_separate_concept_coverage(
    answer: str,
    concepts: tuple[str, ...],
) -> bool:
    """判断每个概念是否位于不同的列表条目中。"""
    if not concepts:
        return True

    covered: set[str] = set()
    normalized_concepts = {concept.lower() for concept in concepts}

    for line in answer.splitlines():
        if not BULLET_LINE_PATTERN.match(line):
            continue

        # 只认条目开头的概念标签，正文引用别的概念不算替它作答。
        content = BULLET_LINE_PATTERN.sub("", line).replace("**", "").replace("`", "").lower()
        for concept in normalized_concepts:
            if re.match(re.escape(concept) + r"(?![a-z0-9-])", content):
                # '- GQA 和 MQA ...' 仍是合并条目，需要修复。
                tail = content[len(concept):].lstrip()
                if any(re.match(r"(?:、|和|与|/|,|，|and\b)\s*" + re.escape(other) + r"(?![a-z0-9-])", tail)
                       for other in normalized_concepts - {concept}):
                    continue
                covered.add(concept)
    return covered == normalized_concepts


@dataclass(frozen=True)
class AnswerResult:
    """一次 RAG 问答的最终结果。"""

    answer: str
    sources: tuple[Source, ...]
    rejected: bool
    citation_validation: CitationValidation | None = None
    retrieval_method: str = "bm25"
    generation_calls: int = 0
    separate_coverage_ok: bool | None = None
    context_text: str = ""
    retrieval_trace: tuple[dict, ...] = ()
    context_diagnostics: tuple[dict, ...] = ()
    prompt_tokens: int | None = None
    prompt_token_budget: int | None = None
    context_char_budget: int | None = None
    phase_timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedAnswer:
    """完成能力判断、检索和上下文后，等待模型生成的不可变输入。"""

    question: str
    immediate_result: AnswerResult | None
    messages: tuple[dict[str, str], ...] = ()
    sources: tuple[Source, ...] = ()
    concepts: tuple[str, ...] = ()
    context_text: str = ""
    retrieval_trace: tuple[dict, ...] = ()
    context_diagnostics: tuple[dict, ...] = ()
    prompt_tokens: int | None = None
    prompt_token_budget: int | None = None
    context_char_budget: int | None = None
    prepare_timings_ms: dict[str, float] = field(default_factory=dict)


def _timed_call(timings: dict[str, float], phase: str, function, *args):
    """累计一个非嵌套准备阶段；用 perf_counter 避免系统时钟调整。"""
    started = time.perf_counter()
    try:
        return function(*args)
    finally:
        timings[phase] = timings.get(phase, 0.0) + (time.perf_counter() - started) * 1000


def _finish_prepare_timings(timings: dict[str, float], started: float) -> dict[str, float]:
    total = (time.perf_counter() - started) * 1000
    measured = sum(timings.values())
    result = {name: round(value, 2) for name, value in timings.items()}
    result["other_ms"] = round(max(0.0, total - measured), 2)
    result["total_ms"] = round(total, 2)
    return result


def combine_phase_timings(prepare_timings: dict[str, float], generation_ms: float,
                          finalize_ms: float, total_ms: float) -> dict[str, float]:
    """展开准备子阶段，同时保留可相加的三段顶层时间。"""
    result = {f"prepare_{name}": value for name, value in prepare_timings.items()
              if name != "total_ms"}
    prepare_total = float(prepare_timings.get("total_ms", 0.0))
    result.update({
        "prepare_total_ms": round(prepare_total, 2),
        "generation_wall_ms": round(generation_ms, 2),
        "finalize_ms": round(finalize_ms, 2),
        "runtime_overhead_ms": round(max(
            0.0, total_ms - prepare_total - generation_ms - finalize_ms
        ), 2),
        "runtime_total_ms": round(total_ms, 2),
    })
    return result


class RAGAnswerer:
    """连接能力边界、可切换检索、上下文构建与模型生成。"""

    def __init__(
        self,
        chunks_path: Path,
        client: ChatClient,
        retrieval_top_k: int = 5,
        max_context_chars: int = 2200,
        retrieval_method: str = "bm25",
        device: str = "cuda",
        candidate_k: int = 20,
        rerank: bool = False,
        rerank_top_n: int = 20,
        rerank_device: str | None = None,
        context_policy: str = "legacy",
        message_builder: Callable[[str, str], list[dict[str, str]]] | None = None,
        enable_concept_repair: bool = True,
        message_token_counter: Callable[[list[dict[str, str]]], int] | None = None,
        max_prompt_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.retrieval_top_k = retrieval_top_k
        self.max_context_chars = max_context_chars
        if retrieval_top_k < 1 or max_context_chars < 1:
            raise ValueError("检索数量与上下文预算必须为正数")
        if retrieval_method not in ("bm25", "vector", "hybrid"):
            raise ValueError(f"未知检索方式：{retrieval_method}")
        self.retrieval_method = retrieval_method
        if context_policy not in ("legacy", "blocks"):
            raise ValueError("context_policy 必须为 legacy 或 blocks")
        self.context_policy = context_policy
        self.message_builder = message_builder
        self.enable_concept_repair = enable_concept_repair
        if (message_token_counter is None) != (max_prompt_tokens is None):
            raise ValueError("message_token_counter 与 max_prompt_tokens 必须同时提供")
        if max_prompt_tokens is not None and (type(max_prompt_tokens) is not int
                                              or max_prompt_tokens < 32):
            raise ValueError("max_prompt_tokens 必须是至少 32 的整数")
        self.message_token_counter = message_token_counter
        self.max_prompt_tokens = max_prompt_tokens

        self.chunks = load_chunks(chunks_path)
        self.engine = RetrievalEngine(
            self.chunks, device=device, candidate_k=max(candidate_k, retrieval_top_k),
            cache_path=chunks_path.parent / "vector_index.npz",
            rerank=rerank, rerank_top_n=rerank_top_n, rerank_device=rerank_device,
        )
        self.index = self.engine.bm25

    def _pack_context(self, retrieval_results, max_chars):
        if self.context_policy == "blocks":
            return build_evidence_context(retrieval_results, max_chars)
        return build_context(
            retrieval_results,
            max_chars=max_chars,
            extract_evidence=self.retrieval_method == "bm25",
        )

    def _build_messages(self, question, context_text):
        messages = self.message_builder(question, context_text) if self.message_builder else [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    "知识库资料：\n"
                    f"{context_text}\n\n"
                    "请严格依据上述资料回答，并在重要结论后标注 [S1]、[S2] "
                    "等对应来源。"
                ),
            },
        ]
        if not messages or any(message.get("role") not in ("system", "user", "assistant")
                               or not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("message_builder 必须返回非空的合法文本消息")
        return messages

    def prepare(self, question: str) -> PreparedAnswer:
        """执行模型调用之前的守卫、检索、上下文和消息构造。"""
        prepare_started = time.perf_counter()
        prepare_timings: dict[str, float] = {}
        question = question.strip()

        if not question:
            raise ValueError("问题不能为空")

        rejection_reason = _timed_call(
            prepare_timings, "guard_ms", static_corpus_rejection_reason, question
        )
        if rejection_reason:
            timings = _finish_prepare_timings(prepare_timings, prepare_started)
            result = AnswerResult(
                answer=f"当前知识库助手无法处理这个请求：{rejection_reason}",
                sources=(),
                rejected=True,
                retrieval_method=self.retrieval_method,
                phase_timings_ms={
                    **{f"prepare_{key}": value for key, value in timings.items()
                       if key != "total_ms"},
                    "prepare_total_ms": timings["total_ms"],
                },
            )
            return PreparedAnswer(question=question, immediate_result=result,
                                  prepare_timings_ms=timings)

        hits = _timed_call(
            prepare_timings, "retrieval_ms", self.engine.search,
            question, self.retrieval_method, self.retrieval_top_k,
        )
        retrieval_results = [hit.as_result() for hit in hits]

        context_char_budget = self.max_context_chars
        context_package = _timed_call(
            prepare_timings, "context_pack_ms", self._pack_context,
            retrieval_results, context_char_budget,
        )

        if not context_package.sources:
            timings = _finish_prepare_timings(prepare_timings, prepare_started)
            result = AnswerResult(
                answer=(
                    "当前知识库中没有检索到足以回答这个问题的相关资料。"
                ),
                sources=(),
                rejected=False,
                retrieval_method=self.retrieval_method,
                context_diagnostics=context_package.diagnostics,
                phase_timings_ms={
                    **{f"prepare_{key}": value for key, value in timings.items()
                       if key != "total_ms"},
                    "prepare_total_ms": timings["total_ms"],
                },
            )
            return PreparedAnswer(question=question, immediate_result=result,
                                  prepare_timings_ms=timings)

        messages = _timed_call(
            prepare_timings, "message_build_ms", self._build_messages,
            question, context_package.text,
        )
        prompt_tokens = None
        budget_diagnostics = []
        if self.message_token_counter is not None:
            prompt_tokens = _timed_call(
                prepare_timings, "token_count_ms", self.message_token_counter, messages
            )
            budget_diagnostics.append({
                "context_char_budget": context_char_budget,
                "context_chars": len(context_package.text),
                "prompt_tokens": prompt_tokens,
                "has_sources": bool(context_package.sources),
            })
            if prompt_tokens > self.max_prompt_tokens:
                empty_messages = _timed_call(
                    prepare_timings, "message_build_ms", self._build_messages, question, ""
                )
                empty_tokens = _timed_call(
                    prepare_timings, "token_count_ms", self.message_token_counter, empty_messages
                )
                if empty_tokens > self.max_prompt_tokens:
                    diagnostic = {
                        "reason": "prompt_token_budget_exhausted_by_question_and_instructions",
                        "prompt_tokens_without_context": empty_tokens,
                        "max_prompt_tokens": self.max_prompt_tokens,
                        "attempts": budget_diagnostics,
                    }
                    timings = _finish_prepare_timings(prepare_timings, prepare_started)
                    result = AnswerResult(
                        answer="问题与系统提示已超过本地模型输入 token 预算，请缩短问题。",
                        sources=(), rejected=False, retrieval_method=self.retrieval_method,
                        context_diagnostics=(diagnostic,), prompt_tokens=empty_tokens,
                        prompt_token_budget=self.max_prompt_tokens, context_char_budget=0,
                        phase_timings_ms={
                            **{f"prepare_{key}": value for key, value in timings.items()
                               if key != "total_ms"},
                            "prepare_total_ms": timings["total_ms"],
                        },
                    )
                    return PreparedAnswer(question=question, immediate_result=result,
                                          prepare_timings_ms=timings)

                best = None
                low, high = 1, context_char_budget - 1
                while low <= high:
                    candidate_budget = (low + high) // 2
                    candidate_package = _timed_call(
                        prepare_timings, "context_pack_ms", self._pack_context,
                        retrieval_results, candidate_budget,
                    )
                    candidate_messages = _timed_call(
                        prepare_timings, "message_build_ms", self._build_messages,
                        question, candidate_package.text,
                    )
                    candidate_tokens = _timed_call(
                        prepare_timings, "token_count_ms",
                        self.message_token_counter, candidate_messages,
                    )
                    budget_diagnostics.append({
                        "context_char_budget": candidate_budget,
                        "context_chars": len(candidate_package.text),
                        "prompt_tokens": candidate_tokens,
                        "has_sources": bool(candidate_package.sources),
                    })
                    if not candidate_package.sources:
                        low = candidate_budget + 1
                    elif candidate_tokens <= self.max_prompt_tokens:
                        best = (candidate_budget, candidate_package,
                                candidate_messages, candidate_tokens)
                        low = candidate_budget + 1
                    else:
                        high = candidate_budget - 1

                if best is None:
                    diagnostic = {
                        "reason": "prompt_token_budget_cannot_fit_citable_evidence",
                        "prompt_tokens_without_context": empty_tokens,
                        "max_prompt_tokens": self.max_prompt_tokens,
                        "attempts": budget_diagnostics,
                    }
                    timings = _finish_prepare_timings(prepare_timings, prepare_started)
                    result = AnswerResult(
                        answer="模型输入 token 预算不足，无法放入可引用的知识库证据。",
                        sources=(), rejected=False, retrieval_method=self.retrieval_method,
                        context_diagnostics=(diagnostic,), prompt_tokens=empty_tokens,
                        prompt_token_budget=self.max_prompt_tokens, context_char_budget=0,
                        phase_timings_ms={
                            **{f"prepare_{key}": value for key, value in timings.items()
                               if key != "total_ms"},
                            "prepare_total_ms": timings["total_ms"],
                        },
                    )
                    return PreparedAnswer(question=question, immediate_result=result,
                                          prepare_timings_ms=timings)
                context_char_budget, context_package, messages, prompt_tokens = best

            budget_diagnostics.append({
                "reason": "prompt_token_budget",
                "max_prompt_tokens": self.max_prompt_tokens,
                "selected_prompt_tokens": prompt_tokens,
                "configured_context_char_budget": self.max_context_chars,
                "selected_context_char_budget": context_char_budget,
                "selected_context_chars": len(context_package.text),
            })

        timings = _finish_prepare_timings(prepare_timings, prepare_started)
        return PreparedAnswer(
            question=question,
            immediate_result=None,
            messages=tuple(messages),
            sources=context_package.sources,
            concepts=requested_separate_concepts(question),
            context_text=context_package.text,
            context_diagnostics=tuple(context_package.diagnostics) + tuple(budget_diagnostics),
            retrieval_trace=tuple({
                "chunk_id": hit.chunk["id"], "source_file": hit.chunk["source_file"],
                "heading_path": hit.chunk["heading_path"], "score": hit.score,
                "raw_score": hit.raw_score, "ranks": hit.ranks,
                "channel_scores": hit.channel_scores, "contributions": hit.contributions,
                "quality_multiplier": hit.quality_multiplier,
                "retrieval_score": hit.retrieval_score, "pre_rerank_rank": hit.pre_rerank_rank,
                "rerank_score": hit.rerank_score, "rerank_input_tokens": hit.rerank_input_tokens,
                "rerank_truncated": hit.rerank_truncated,
            } for hit in hits),
            prompt_tokens=prompt_tokens,
            prompt_token_budget=self.max_prompt_tokens,
            context_char_budget=context_char_budget,
            prepare_timings_ms=timings,
        )

    def finalize(self, prepared: PreparedAnswer, answer: str, generation_calls: int = 1) -> AnswerResult:
        """只把完整生成结果转成 AnswerResult；半截流不得调用此方法。"""
        if prepared.immediate_result is not None:
            raise ValueError("无需生成的 PreparedAnswer 不能 finalize")
        citation_validation = validate_citations(answer, prepared.sources)
        return AnswerResult(
            answer=answer,
            sources=prepared.sources,
            rejected=False,
            citation_validation=citation_validation,
            retrieval_method=self.retrieval_method,
            generation_calls=generation_calls,
            separate_coverage_ok=(
                has_separate_concept_coverage(answer, prepared.concepts)
                if len(prepared.concepts) >= 2 else None
            ),
            context_text=prepared.context_text,
            context_diagnostics=prepared.context_diagnostics,
            retrieval_trace=prepared.retrieval_trace,
            prompt_tokens=prepared.prompt_tokens,
            prompt_token_budget=prepared.prompt_token_budget,
            context_char_budget=prepared.context_char_budget,
            phase_timings_ms={
                **{f"prepare_{key}": value
                   for key, value in prepared.prepare_timings_ms.items()
                   if key != "total_ms"},
                "prepare_total_ms": prepared.prepare_timings_ms.get("total_ms", 0.0),
            },
        )

    def answer(self, question: str) -> AnswerResult:
        """回答一个问题，必要时拒绝超出静态知识库能力边界的请求。"""
        runtime_started = time.perf_counter()
        prepared = self.prepare(question)
        if prepared.immediate_result is not None:
            total_ms = (time.perf_counter() - runtime_started) * 1000
            return replace(
                prepared.immediate_result,
                phase_timings_ms=combine_phase_timings(
                    prepared.prepare_timings_ms, 0.0, 0.0, total_ms
                ),
            )

        messages = list(prepared.messages)
        generation_started = time.perf_counter()
        answer = self.client.chat(messages)
        generation_calls = 1

        if (self.enable_concept_repair and len(prepared.concepts) >= 2
                and not has_separate_concept_coverage(answer, prepared.concepts)):
            concept_list = "、".join(prepared.concepts)
            repair_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": answer,
                },
                {
                    "role": "user",
                    "content": (
                        f"用户要求分别说明：{concept_list}。"
                        "请严格依据同一份知识库资料，重写上一回答："
                        "每个概念必须单独成为一个列表条目，并各自标注"
                        "对应来源；不要把多个概念合并在同一条中。"
                    ),
                },
            ]
            answer = self.client.chat(repair_messages)
            generation_calls += 1
        generation_ms = (time.perf_counter() - generation_started) * 1000
        finalize_started = time.perf_counter()
        result = self.finalize(prepared, answer, generation_calls=generation_calls)
        finalize_ms = (time.perf_counter() - finalize_started) * 1000
        total_ms = (time.perf_counter() - runtime_started) * 1000
        return replace(
            result,
            phase_timings_ms=combine_phase_timings(
                prepared.prepare_timings_ms, generation_ms, finalize_ms, total_ms
            ),
        )


def create_client(provider: str) -> ChatClient:
    """根据供应商名称创建模型客户端。"""
    # 旧 CLI 明确选择读取项目 .env；产品入口不会调用这里。
    load_env_file()
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
    print(f"\n检索方式：{result.retrieval_method}；生成调用次数：{result.generation_calls}")
    if result.separate_coverage_ok is False:
        print("覆盖提醒：重写一次后仍未检测到逐项回答；不再继续调用模型。")

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
        help="传给上下文构建的候选片段数。",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=2200,
        help="允许交给模型的知识库资料最大字符数。",
    )
    parser.add_argument("--retriever", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--chunks-file", default="data/generated/chunks.json")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--rerank-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--context-policy", choices=("legacy", "blocks"), default="legacy")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / args.chunks_file

    answerer = RAGAnswerer(
        chunks_path=chunks_path,
        client=create_client(args.provider),
        retrieval_top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        retrieval_method=args.retriever,
        device=args.device,
        candidate_k=args.candidate_k,
        rerank=args.rerank, rerank_top_n=args.rerank_top_n,
        rerank_device=args.rerank_device, context_policy=args.context_policy,
    )

    result = answerer.answer(args.question)
    print_answer_result(result)


if __name__ == "__main__":
    main()
