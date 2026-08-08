import json
import re
from pathlib import Path

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")  # 匹配 Markdown 标题的正则表达式

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

def save_documents(documents:list[dict[str,str]],output_path:Path) ->None:
    """把结构化文档保存为格式化 JSON。"""
    output_path.parent.mkdir(parents=True,exist_ok=True)

    output_path.write_text(
        json.dumps(documents,ensure_ascii=False,indent=2),
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
        section.append(
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
            heading = heading_match.grouo(2).strip()

def main():
    vault_dir = Path(
        r"C:\Users\zjm\Documents\Obsidian Vault\10_技术学习\概念笔记\大模型"
    )

    project_dir = Path(__file__).resolve().parent.parent
    output_path = project_dir / "data" / "generated" / "documents.json"

    markdown_files = find_markdown_files(vault_dir)
    documents = [
        parse_markdown_file(markdown_file, vault_dir)
        for markdown_file in markdown_files
    ]

    save_documents(documents, output_path)

    print(f"扫描目录：{vault_dir}")
    print(f"结构化文档数：{len(documents)}")
    print(f"JSON 输出：{output_path}")

    if documents:
        first_document = documents[0]
        print("-" * 60)
        print(f"示例来源：{first_document['source_file']}")
        print(f"示例标题：{first_document['title']}")
        print(f"示例字符数：{len(first_document['text'])}")


if __name__ == "__main__":
    main()