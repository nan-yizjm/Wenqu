import json
import re
from pathlib import Path

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")  # 匹配 Markdown 标题的正则表达式
MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150

def read_markdown(path:Path) ->str:
    """读取 UTF-8 编码的 Markdown 文件。"""
    return path.read_text(encoding="utf-8")

def extract_title(markdown_text:str) ->str | None:
    """返回第一个一级 Markdown 标题；如果找不到则返回 None。"""
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()     # strip() 去除多余空格

    return None

def  find_markdown_files(root_dir:Path) ->list[Path]:
    """递归返回目录中全部 Markdown 文件，并按路径排序。"""
    if not root_dir.is_dir():
        raise NotADirectoryError(f"不是有效目录：{root_dir}")

    return sorted(root_dir.rglob("*.md"))

def parse_markdown_file(path:Path,root_dir:Path) ->dict[str,str]:
    """把一篇 Markdown 文件转换为结构化文档。"""
    markdown_text = read_markdown(path)

    return {
        "source_file":str(path.relative_to(root_dir)),
        "title":extract_title(markdown_text) or "",
        "text":markdown_text,
    }

def save_json(data:list[dict[str,str]],output_path:Path) ->None:
    """把结构化文档保存为格式化 JSON。"""
    output_path.parent.mkdir(parents=True,exist_ok=True)

    output_path.write_text(
        json.dumps(data,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )

def strip_frontmatter(markdown_text:str) ->str:
    """移除 Markdown 文件开头的 YAML Frontmatter。"""
    lines = markdown_text.splitlines()

    if not lines or lines[0].strip() !="---":
        return markdown_text.strip()

    for index in range(1,len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index +1 :]).strip()

    return markdown_text.strip()

def append_section(
        sections:list[dict[str,str]],
        heading_path:list[str],
        lines:list[str],
) ->None:
    """若当前章节有正文，就把它追加为一个章节片段。"""
    text = "\n".join(lines).strip()

    if text:
        sections.append(
            {
                "heading_path":" > ".join(heading_path),
                "text":text,
            }
        )

def split_by_headings(
        markdown_text:str,
        fallback_title:str,
) ->list[dict[str,str]]:
    """按 Markdown 标题层级将文本切分为章节片段。"""
    sections:list[dict[str,str]] = []
    current_heading_path = [fallback_title or "未命名文档"]
    current_lines:list[str] = []
    in_code_block = False

    for line in markdown_text.splitlines():
        stripped_line = line.strip()

        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        heading_match = HEADING_PATTERN.match(line)

        if not in_code_block and heading_match:
            append_section(sections,current_heading_path,current_lines)

            level =len(heading_match.group(1))
            heading = heading_match.group(2).strip()

            if level == 1:
                # 一级标题就是新的根，不再保留 fallback_title。
                current_heading_path = [heading]

            else:
                # 二级标题保留一级标题，三级标题保留一级和二级标题，以此类推。
                parent_path = current_heading_path[: level - 1]

                if not parent_path:
                    parent_path = [fallback_title or "未命名文档"]

                current_heading_path = parent_path + [heading]

            current_lines = []
            continue

        current_lines.append(line)

    append_section(sections,current_heading_path,current_lines)

    return sections

def split_long_text(
        text:str,
        max_chars:int,
        overlap_chars:int,
) ->list[str]:
    """把过长文本切成限制长度的片段，并保留相邻片段重叠。"""
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须大于等于 0 且小于 max_chars")

    if len(text) <= max_chars:
        return [text]

    pieces:list[str] = []
    start = 0

    while start <len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            last_newline = text.rfind("\n", start, end)

            if last_newline > start + max_chars//2:
                end = last_newline

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end == len(text):
            break

        start = end - overlap_chars

    return pieces

def create_chunks(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """把所有原始文档切分成带来源信息、限制长度的章节片段。"""
    chunks: list[dict[str, str]] = []

    for document in documents:
        clean_text = strip_frontmatter(document["text"])
        sections = split_by_headings(clean_text, document["title"])

        for section_index, section in enumerate(sections):
            text_pieces = split_long_text(
                section["text"],
                max_chars=MAX_CHUNK_CHARS,
                overlap_chars=OVERLAP_CHARS,
            )

            for piece_index, text_piece in enumerate(text_pieces):
                chunks.append(
                    {
                        "id": (
                            f"{document['source_file']}"
                            f"#section-{section_index}-part-{piece_index}"
                        ),
                        "source_file": document["source_file"],
                        "document_title": document["title"],
                        "heading_path": section["heading_path"],
                        "text": text_piece,
                    }
                )

    return chunks

def main():
    import argparse
    from .settings import DEFAULT_CONFIG, load_settings

    parser = argparse.ArgumentParser(description="从配置中的 Markdown 目录生成旧版实验语料。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    vault_dir = load_settings(args.config).vault_dir

    project_dir = Path(__file__).resolve().parent.parent
    generated_dir = project_dir / "data" / "generated"
    documents_path = generated_dir / "documents.json"
    chunks_path = generated_dir / "chunks.json"

    markdown_files = find_markdown_files(vault_dir)
    documents = [
        parse_markdown_file(markdown_file, vault_dir)
        for markdown_file in markdown_files
    ]
    chunks = create_chunks(documents)

    save_json(documents, documents_path)
    save_json(chunks, chunks_path)

    print(f"扫描目录：{vault_dir}")
    print(f"原始文档数：{len(documents)}")
    print(f"章节片段数：{len(chunks)}")
    print(f"文档输出：{documents_path}")
    print(f"片段输出：{chunks_path}")

    if chunks:
        first_chunk = chunks[0]
        preview = first_chunk["text"][:120].replace("\n", " ")

        print("-" * 60)
        print(f"示例来源：{first_chunk['source_file']}")
        print(f"示例标题路径：{first_chunk['heading_path']}")
        print(f"示例片段：{preview}...")


if __name__ == "__main__":
    main()
