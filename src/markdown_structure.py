"""轻量 Markdown 边界识别；保留原文字符坐标，不是完整 CommonMark 解析器。"""

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str = "paragraph"
    continuation: bool = False


def trim_span(text: str, start: int, end: int, kind="paragraph", continuation=False) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end, kind, continuation)


def closes_fence(line: str, marker: str) -> bool:
    value = line.strip()
    return len(value) >= len(marker) and set(value) == {marker[0]}


def sections_with_offsets(text: str, title: str) -> list[tuple[str, Span]]:
    lines = text.splitlines(keepends=True)
    body_line = 0
    if lines and lines[0].strip() == "---":
        closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if closing is not None:
            body_line = closing + 1
    position = sum(len(line) for line in lines[:body_line])
    start = position
    stack = [(1, title or "未命名文档")]
    fence = None
    sections = []
    for line in lines[body_line:]:
        marker = FENCE.match(line)
        if fence is not None:
            if closes_fence(line, fence):
                fence = None
        elif marker:
            fence = marker[1]
        elif match := HEADING.match(line.rstrip("\r\n")):
            span = trim_span(text, start, position)
            if span.start < span.end:
                sections.append((" > ".join(value for _, value in stack), span))
            level = len(match[1])
            stack = [(depth, value) for depth, value in stack if depth < level]
            stack.append((level, match[2]))
            start = position + len(line)
        position += len(line)
    span = trim_span(text, start, len(text))
    if span.start < span.end:
        sections.append((" > ".join(value for _, value in stack), span))
    return sections


def markdown_blocks(text: str) -> list[Span]:
    """空行分段，围栏代码整体保留，列表按条目切，表格按连续行切。"""
    lines = text.splitlines(keepends=True)
    spans = []
    start = None
    kind = "paragraph"
    fence = None
    position = 0

    def flush(end):
        nonlocal start
        if start is not None:
            span = trim_span(text, start, end, kind)
            if span.start < span.end:
                spans.append(span)
        start = None

    for line in lines:
        if fence is not None:
            if closes_fence(line, fence):
                flush(position + len(line))
                fence = None
            position += len(line)
            continue
        if marker := FENCE.match(line):
            flush(position)
            start, kind, fence = position, "code", marker[1]
        elif not line.strip():
            flush(position)
        else:
            next_kind = "list" if LIST_ITEM.match(line) else "table" if line.lstrip().startswith("|") else "paragraph"
            if next_kind == "list" or (start is not None and (next_kind == "table") != (kind == "table")):
                flush(position)
            if start is None:
                start, kind = position, next_kind
        position += len(line)
    flush(len(text))
    return spans
