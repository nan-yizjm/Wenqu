"""按完整原文块打包证据：去重、轮流分配预算、显示省略诊断。"""

import re

from .context import ContextPackage, Source
from .markdown_structure import FENCE, Span, closes_fence, markdown_blocks
from .retrieve import tokenize

MAX_LINKED_LIST_CHARS = 800


def canonical(text: str) -> str:
    # 不折叠内部空白：代码缩进和字符串里的空格可能改变语义。
    return text.replace("\r\n", "\n").strip()


def semantic_spans(text: str) -> list[Span]:
    """把短连续列表及其冒号引导段视为一个证据单元。"""
    raw = markdown_blocks(text)
    grouped: list[Span] = []
    index = 0
    while index < len(raw):
        span = raw[index]
        if span.kind != "list":
            grouped.append(span)
            index += 1
            continue
        members = [span]
        index += 1
        while index < len(raw) and raw[index].kind == "list":
            gap = text[members[-1].end:raw[index].start]
            if gap.strip() or gap.replace("\r", "").count("\n") > 1:
                break
            members.append(raw[index])
            index += 1
        cluster = Span(members[0].start, members[-1].end, "list")
        cluster_size = cluster.end - cluster.start
        if cluster_size > MAX_LINKED_LIST_CHARS:
            grouped.extend(members)
            continue
        if grouped and grouped[-1].kind == "paragraph":
            lead = grouped[-1]
            gap = text[lead.end:cluster.start]
            lead_text = text[lead.start:lead.end].rstrip()
            combined_size = cluster.end - lead.start
            if (not gap.strip() and gap.replace("\r", "").count("\n") <= 2
                    and lead_text.endswith(("：", ":"))
                    and combined_size <= MAX_LINKED_LIST_CHARS):
                grouped[-1] = Span(lead.start, cluster.end, "list")
                continue
        grouped.append(cluster)
    return grouped


def build_evidence_context(results, max_chars=2200) -> ContextPackage:
    if max_chars < 1:
        raise ValueError("max_chars 必须为正数")
    entries, diagnostics = [], []
    seen_chunks = set()
    for chunk, _score, matched_tokens in results:
        text = chunk["text"].strip()
        key = canonical(text)
        if not key or key in seen_chunks:
            diagnostics.append({"chunk_id": chunk.get("id"), "reason": "empty_or_duplicate_chunk"})
            continue
        seen_chunks.add(key)
        label = f"S{len(entries) + 1}"
        header = f"[{label}]\n来源：{chunk['source_file']}\n标题路径：{chunk['heading_path']}\n"
        if chunk.get("line_start") is not None:
            header += f"文档快照行号：{chunk['line_start']}-{chunk['line_end']}\n"
        if chunk.get("continuation"):
            header += "说明：包含过长原文块的续片，不保证是完整代码或完整段落。\n"
        header += "内容："
        units = []
        spans = semantic_spans(text)
        if chunk.get("continuation"):
            # 续片可能从代码块内部开始，单独解析会把原来的闭围栏当作开围栏。
            # 将整片原文作为明确标注的文本块保留，使用更长的外层围栏。
            from .markdown_structure import Span
            spans = [Span(0, len(text), "fragment")]
        for span in spans:
            content = text[span.start:span.end]
            if span.kind == "fragment":
                width = max([len(match.group()) for match in re.finditer(r"`+", content)] + [2]) + 1
                fence = "`" * width
                content = fence + "text\n" + content + "\n" + fence
            # 被切开的围栏不应吞掉后续来源；只补语法围栏，不补事实。
            if span.kind == "code" and (marker := FENCE.match(content.splitlines()[0])):
                lines = content.splitlines()
                if len(lines) < 2 or not closes_fence(lines[-1], marker[1]):
                    content += "\n" + marker[1]
            units.append({"position": span.start, "content": content,
                          "matches": len(matched_tokens & set(tokenize(content)))})
        pending = sorted(units, key=lambda unit: (-unit["matches"], unit["position"]))
        entries.append({"chunk": chunk, "label": label, "header": header, "pending": pending,
                        "selected": [], "total": len(units), "duplicate_blocks": 0, "budget_skips": 0})
    used_chars, used_sources = 0, 0
    seen_blocks = set()
    # 一轮每个来源选一个完整块；没有按词面过滤全部正文。
    while any(entry["pending"] for entry in entries):
        for entry in entries:
            if not entry["pending"]:
                continue
            unit = entry["pending"].pop(0)
            key = canonical(unit["content"])
            if key in seen_blocks:
                entry["duplicate_blocks"] += 1
                continue
            added_chars = len(unit["content"])
            if entry["selected"]:
                added_chars += 2
            else:
                added_chars += len(entry["header"]) + (2 if used_sources else 0)
            if used_chars + added_chars > max_chars:
                entry["budget_skips"] += 1
                continue
            if not entry["selected"]:
                used_sources += 1
            used_chars += added_chars
            seen_blocks.add(key)
            entry["selected"].append(unit)
    blocks, sources = [], []
    for entry in entries:
        chunk = entry["chunk"]
        diagnostics.append({"chunk_id": chunk.get("id"), "total_blocks": entry["total"],
                            "selected_blocks": len(entry["selected"]),
                            "duplicate_blocks": entry["duplicate_blocks"],
                            "budget_skipped_blocks": entry["budget_skips"]})
        if not entry["selected"]:
            continue
        content = "\n\n".join(unit["content"] for unit in sorted(entry["selected"], key=lambda unit: unit["position"]))
        blocks.append(entry["header"] + content)
        sources.append(Source(entry["label"], chunk["source_file"], chunk["heading_path"],
                              chunk.get("id", ""), chunk.get("line_start"), chunk.get("line_end")))
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        raise AssertionError("完整块打包超过字符预算")
    return ContextPackage(text, tuple(sources), tuple(diagnostics))
