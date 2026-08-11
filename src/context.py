"""将检索结果组织成可直接交给大模型的带引用上下文。"""

from dataclasses import dataclass
from typing import Sequence


RetrievalResult = tuple[dict[str, str], float, set[str]]


@dataclass(frozen=True)
class Source:
    """一个可展示给用户的知识库来源。"""

    label: str
    source_file: str
    heading_path: str


@dataclass(frozen=True)
class ContextPackage:
    """提供给模型的上下文及其可追溯来源。"""

    text: str
    sources: tuple[Source, ...]


def build_context(
    results: Sequence[RetrievalResult],
    max_chars: int = 2200,
) -> ContextPackage:
    """将已排序的检索结果格式化为有引用编号的上下文。

    按 BM25 原有排名依次选择片段。最后一个能放入预算的片段允许截断，
    以便充分利用上下文窗口，同时始终保留可追溯的来源信息。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    blocks: list[str] = []
    sources: list[Source] = []

    for chunk, _score, _matched_tokens in results:
        label = f"S{len(sources) + 1}"

        header = (
            f"[{label}]\n"
            f"来源：{chunk['source_file']}\n"
            f"标题路径：{chunk['heading_path']}\n"
            "内容："
        )

        separator_length = 2 if blocks else 0
        used_chars = len("\n\n".join(blocks))
        remaining_chars = max_chars - used_chars - separator_length

        # 连“来源头信息”都无法容纳时，不再加入更低排名的结果。
        if remaining_chars <= len(header):
            break

        content_budget = remaining_chars - len(header)
        content = chunk["text"].strip()

        if len(content) > content_budget:
            content = content[: content_budget - 1].rstrip() + "…"

        blocks.append(header + content)
        sources.append(
            Source(
                label=label,
                source_file=chunk["source_file"],
                heading_path=chunk["heading_path"],
            )
        )

    return ContextPackage(
        text="\n\n".join(blocks),
        sources=tuple(sources),
    )