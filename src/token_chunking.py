"""从文档快照生成独立的 token 限长语料，不覆盖原 chunks.json。"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from .markdown_structure import Span, markdown_blocks, sections_with_offsets, trim_span
from .vector_retrieve import MODEL_NAME, MODEL_REVISION, PASSAGE_TEMPLATE

ROOT = Path(__file__).resolve().parent.parent


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TokenChunker:
    def __init__(self, tokenizer, max_tokens=480, overlap_tokens=32):
        if not 32 <= max_tokens <= 512 or not 0 <= overlap_tokens < max_tokens:
            raise ValueError("要求 32 <= max_tokens <= 512，且 0 <= overlap_tokens < max_tokens")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=False, verbose=False)["input_ids"])

    def passage_count(self, heading: str, text: str) -> int:
        return self.count(PASSAGE_TEMPLATE.format(heading_path=heading, text=text))

    def split_oversize(self, text: str, span: Span, heading: str) -> list[Span]:
        pieces = []
        start = span.start
        while start < span.end:
            if self.passage_count(heading, text[start:span.end]) <= self.max_tokens:
                end = span.end
            else:
                low, high = start, span.end
                while low < high:
                    middle = (low + high + 1) // 2
                    if self.passage_count(heading, text[start:middle]) <= self.max_tokens:
                        low = middle
                    else:
                        high = middle - 1
                end = low
                # 优先行/句边界；只有单行、单句过长时才切字符。
                boundaries = [m.end() + start for m in re.finditer(r"\n|[。！？；.!?;]\s*", text[start:end])]
                useful = [value for value in boundaries if value > start + (end - start) // 2]
                if useful:
                    end = useful[-1]
                while end > start and self.passage_count(heading, text[start:end]) > self.max_tokens:
                    end -= 1
                if end <= start:
                    raise ValueError(f"标题与前缀耗尽 token 预算：{heading}")
            part = trim_span(text, start, end, span.kind, continuation=True)
            if part.start < part.end:
                pieces.append(part)
            start = end
        return pieces

    def split_section(self, text: str, heading: str) -> list[tuple[Span, int]]:
        if self.passage_count(heading, "测") > self.max_tokens:
            raise ValueError(f"标题太长，无法保留正文：{heading}")
        units = []
        for block in markdown_blocks(text):
            if self.passage_count(heading, text[block.start:block.end]) <= self.max_tokens:
                units.append(block)
            else:
                units.extend(self.split_oversize(text, block, heading))
        groups = []
        current = []
        for unit in units:
            if current and self.passage_count(heading, text[current[0].start:unit.end]) > self.max_tokens:
                groups.append(current)
                # 只重叠完整的小块，不把上一片大段代码的尾巴切下来。
                tail = []
                for previous in reversed(current):
                    if previous.continuation or self.count(text[previous.start:current[-1].end]) > self.overlap_tokens:
                        break
                    tail.insert(0, previous)
                current = tail
                if current and self.passage_count(heading, text[current[0].start:unit.end]) > self.max_tokens:
                    current = []
            current.append(unit)
        if current:
            groups.append(current)
        return [(Span(group[0].start, group[-1].end, "mixed", any(x.continuation for x in group)),
                 len(group)) for group in groups]

    def create_chunks(self, documents: list[dict]) -> tuple[list[dict], dict]:
        chunks = []
        occurrences = Counter()
        uncovered_sections = 0
        for document in documents:
            raw = document["text"]
            source = document["source_file"].replace("\\", "/")
            for heading, section in sections_with_offsets(raw, document.get("title", "")):
                text = raw[section.start:section.end]
                spans = self.split_section(text, heading)
                covered_end = 0
                for span, block_count in spans:
                    if text[covered_end:span.start].strip():
                        uncovered_sections += 1
                    covered_end = max(covered_end, span.end)
                    content = text[span.start:span.end]
                    start, end = section.start + span.start, section.start + span.end
                    identity = digest(source + "\0" + heading + "\0" + content)[:24]
                    occurrences[identity] += 1
                    chunk_id = f"{source}#tok-{identity}-{occurrences[identity]}"
                    token_count = self.passage_count(heading, content)
                    if token_count > self.max_tokens:
                        raise AssertionError("分片后的完整 passage 仍超限")
                    chunks.append({
                        "id": chunk_id, "source_file": source, "document_title": document.get("title", ""),
                        "heading_path": heading, "text": content, "embedding_tokens": token_count,
                        "source_start_char": start, "source_end_char": end,
                        "line_start": raw.count("\n", 0, start) + 1,
                        "line_end": raw.count("\n", 0, end - 1) + 1,
                        "document_sha256": digest(raw), "continuation": span.continuation,
                        "block_count": block_count,
                    })
                if text[covered_end:].strip():
                    uncovered_sections += 1
        if uncovered_sections:
            raise AssertionError(f"发现未覆盖正文：{uncovered_sections}")
        return chunks, {
            "documents": len(documents), "chunks": len(chunks),
            "max_passage_tokens": max((c["embedding_tokens"] for c in chunks), default=0),
            "over_limit_chunks": sum(c["embedding_tokens"] > self.max_tokens for c in chunks),
            "continuation_chunks": sum(c["continuation"] for c in chunks),
            "uncovered_sections": uncovered_sections,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="data/generated/documents.json")
    parser.add_argument("--output-dir", default="data/generated/token_v1")
    parser.add_argument("--max-tokens", type=int, default=480)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    args = parser.parse_args()
    source_path, target = ROOT / args.documents, ROOT / args.output_dir
    if target.exists():
        parser.error("输出目录已存在；请换一个新目录，保留旧索引用于比较")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, local_files_only=True)
    documents = json.loads(source_path.read_text(encoding="utf-8"))
    chunks, diagnostics = TokenChunker(tokenizer, args.max_tokens, args.overlap_tokens).create_chunks(documents)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(), "schema": "token-chunks-v1",
        "documents_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "model": MODEL_NAME, "revision": MODEL_REVISION, "passage_template": PASSAGE_TEMPLATE,
        "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
        "diagnostics": diagnostics,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix="token-build-") as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        for filename, data in (("chunks.json", chunks), ("manifest.json", manifest)):
            (staging / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.rename(staging, target)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"新语料：{target / 'chunks.json'}；原语料未修改")


if __name__ == "__main__":
    main()
