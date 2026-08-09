import json
import re
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[0-9a-z\u4e00-\u9fff]+")   # 匹配中英文字符和数字的正则表达式


def load_chunks(chunks_path: Path) -> list[dict[str, str]]:
    """从 JSON 文件加载已经切分好的 Markdown 片段。"""
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"找不到片段文件：{chunks_path}\n"
            "请先运行 src/ingest.py 生成 chunks.json。"
        )

    return json.loads(chunks_path.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    """将中英文文本转换为用于朴素检索的二元字符 token。"""
    normalized_text = "".join(TOKEN_PATTERN.findall(text.lower()))

    if len(normalized_text) <= 1:
        return [normalized_text] if normalized_text else []

    return [
        normalized_text[index : index + 2]
        for index in range(len(normalized_text) - 1)
    ]


def score_chunk(
    query_tokens: set[str],
    chunk: dict[str, str],
) -> tuple[int, set[str]]:
    """按 Query 与正文、标题路径的重叠程度给一个片段打分。"""
    text_tokens = set(tokenize(chunk["text"]))
    heading_tokens = set(tokenize(chunk["heading_path"]))

    text_matches = query_tokens & text_tokens
    heading_matches = query_tokens & heading_tokens

    score = len(text_matches) + len(heading_matches)
    matched_tokens = text_matches | heading_matches

    return score, matched_tokens


def search_chunks(
    query: str,
    chunks: list[dict[str, str]],
    top_k: int = 5,
) -> list[tuple[dict[str, str], int, set[str]]]:
    """返回与问题最匹配的前 top_k 个文本片段。"""
    query_tokens = set(tokenize(query))

    if not query_tokens:
        return []

    results: list[tuple[dict[str, str], int, set[str]]] = []

    for chunk in chunks:
        score, matched_tokens = score_chunk(query_tokens, chunk)

        if score > 0:
            results.append((chunk, score, matched_tokens))

    results.sort(key=lambda item: (-item[1], item[0]["id"]))

    return results[:top_k]


def print_results(
    query: str,
    results: list[tuple[dict[str, str], int, set[str]]],
) -> None:
    """以易读格式打印检索结果。"""
    print("-" * 70)
    print(f"问题：{query}")

    if not results:
        print("没有找到包含相关关键词的片段。")
        return

    for index, (chunk, score, matched_tokens) in enumerate(results, start=1):
        preview = chunk["text"][:180].replace("\n", " ")

        print(f"\n[{index}] 分数：{score}")
        print(f"来源：{chunk['source_file']}")
        print(f"标题路径：{chunk['heading_path']}")
        print(f"匹配 token：{', '.join(sorted(matched_tokens))}")
        print(f"预览：{preview}...")


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    chunks_path = project_dir / "data" / "generated" / "chunks.json"

    chunks = load_chunks(chunks_path)

    print(f"已加载 {len(chunks)} 个文本片段。")
    print("输入问题开始检索；输入“退出”结束程序。")

    while True:
        query = input("\n问题：").strip()

        if query in {"退出", "exit", "quit"}:
            print("检索结束。")
            break

        if not query:
            print("问题不能为空。")
            continue

        results = search_chunks(query, chunks)
        print_results(query, results)


if __name__ == "__main__":
    main()