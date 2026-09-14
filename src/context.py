"""将检索结果组织成可直接交给大模型的带引用上下文。"""

from dataclasses import dataclass
from typing import Sequence

from .retrieve import tokenize


RetrievalResult = tuple[dict[str, str], float, set[str]]

DEFAULT_MIN_CONTENT_CHARS_PER_SOURCE = 240


@dataclass(frozen=True)
class Source:
    """一个可展示给用户的知识库来源。"""

    label: str
    source_file: str
    heading_path: str
    chunk_id: str = ""
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class ContextPackage:
    """提供给模型的上下文及其可追溯来源。"""

    text: str
    sources: tuple[Source, ...]
    diagnostics: tuple[dict, ...] = ()


def extract_relevant_excerpt(
    text: str,
    matched_tokens: set[str],
) -> str:
    """从较长片段中优先保留包含查询证据的 Markdown 行。

    BM25 已给出命中的 token；这里利用它定位原文中的相关行。这样
    ``KV Cache 优化`` 这类包含多条路线的大片段不会总从开头截断，
    而能把命中 GQA、MQA、MLA 的具体 bullet 交给模型。
    """
    normalized_text = text.strip()

    if not normalized_text or not matched_tokens:
        return normalized_text

    lines = normalized_text.splitlines()
    matched_indexes = [
        index
        for index, line in enumerate(lines)
        if matched_tokens & set(tokenize(line))
    ]

    if not matched_indexes:
        return normalized_text

    selected_indexes: set[int] = set(matched_indexes)

    # 相关 bullet 前的一行通常是“压缩路线”“架构路线”等小标题，
    # 保留它能让模型知道该证据处于什么分类下。
    for index in matched_indexes:
        previous_index = index - 1
        if previous_index >= 0 and lines[previous_index].strip():
            selected_indexes.add(previous_index)

    return "\n".join(
        line
        for index, line in enumerate(lines)
        if index in selected_indexes and line.strip()
    )


def build_context(
    results: Sequence[RetrievalResult],
    max_chars: int = 2200,
    min_content_chars_per_source: int = (
        DEFAULT_MIN_CONTENT_CHARS_PER_SOURCE
    ),
    extract_evidence: bool = True,
) -> ContextPackage:
    """将已排序的检索结果格式化为有引用编号的上下文。

    每个能放入预算的候选先获得一小段内容配额，避免高排名长片段
    独占上下文、导致低排名但关键的证据从未传给模型。剩余空间再按
    原始排名分配。所有片段始终保留可追溯的来源信息。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if min_content_chars_per_source <= 0:
        raise ValueError("min_content_chars_per_source 必须大于 0")

    selected: list[tuple[str, dict[str, str], str, Source]] = []
    used_header_chars = 0

    for chunk, _score, matched_tokens in results:
        if not chunk["text"].strip():
            continue
        label = f"S{len(selected) + 1}"

        header = (
            f"[{label}]\n"
            f"来源：{chunk['source_file']}\n"
            f"标题路径：{chunk['heading_path']}\n"
            "内容："
        )

        separator_length = 2 if selected else 0

        # 至少要为内容留一个字符；否则该来源对模型没有证据价值。
        if (
            used_header_chars
            + separator_length
            + len(header)
            + len(selected) + 1
            > max_chars
        ):
            break

        source = Source(
            label=label,
            source_file=chunk["source_file"],
            heading_path=chunk["heading_path"],
            chunk_id=chunk.get("id", ""),
            line_start=chunk.get("line_start"), line_end=chunk.get("line_end"),
        )
        content = (
            extract_relevant_excerpt(chunk["text"], matched_tokens)
            if extract_evidence else chunk["text"].strip()
        )
        selected.append((header, chunk, content, source))
        used_header_chars += separator_length + len(header)

    if not selected:
        return ContextPackage(text="", sources=())

    available_content_chars = max_chars - used_header_chars
    base_budget = min(
        min_content_chars_per_source,
        available_content_chars // len(selected),
    )
    content_budgets = [
        min(len(content), base_budget)
        for _header, _chunk, content, _source in selected
    ]
    remaining_content_chars = (
        available_content_chars - sum(content_budgets)
    )

    # 在每个来源已经有基础证据的前提下，优先给高排名结果补足内容。
    for index, (_header, _chunk, content, _source) in enumerate(selected):
        additional_chars = min(
            len(content) - content_budgets[index],
            remaining_content_chars,
        )
        content_budgets[index] += additional_chars
        remaining_content_chars -= additional_chars

        if remaining_content_chars == 0:
            break

    blocks: list[str] = []
    sources: list[Source] = []

    for (header, _chunk, content, source), content_budget in zip(
        selected,
        content_budgets,
        strict=True,
    ):
        if len(content) > content_budget:
            content = content[: content_budget - 1].rstrip() + "…"

        blocks.append(header + content)
        sources.append(source)

    return ContextPackage(
        text="\n\n".join(blocks),
        sources=tuple(sources),
    )
