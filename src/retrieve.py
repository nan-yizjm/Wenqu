import json
import math
import re
from pathlib import Path


# TOKEN_PATTERN = re.compile(r"[0-9a-z\u4e00-\u9fff]+")   # 匹配中英文字符和数字的正则表达式
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
ASCII_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9]+$")

def load_chunks(chunks_path: Path) -> list[dict[str, str]]:
    """从 JSON 文件加载已经切分好的 Markdown 片段。"""
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"找不到片段文件：{chunks_path}\n"
            "请先运行 src/ingest.py 生成 chunks.json。"
        )

    return json.loads(chunks_path.read_text(encoding="utf-8"))


# def tokenize(text: str) -> list[str]:
#     """将中英文文本转换为用于朴素检索的二元字符 token。"""
#     normalized_text = "".join(TOKEN_PATTERN.findall(text.lower()))

#     if len(normalized_text) <= 1:
#         return [normalized_text] if normalized_text else []

#     return [
#         normalized_text[index : index + 2]
#         for index in range(len(normalized_text) - 1)
#     ]
def tokenize(text: str) -> list[str]:
    """分离英文技术词，并将中文连续文本切成二元字组。"""
    tokens: list[str] = []

    for part in TOKEN_PATTERN.findall(text.lower()):
        if ASCII_TOKEN_PATTERN.fullmatch(part):
            # 英文和数字技术词作为一个整体保留：
            # PagedAttention -> pagedattention
            # KV Cache -> kv, cache
            tokens.append(part)
            continue

        # 中文连续文本使用二元字组：
        # 显存优化 -> 显存、存优、优化
        if len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(
                part[index : index + 2]
                for index in range(len(part) - 1)
            )

    return tokens

QUERY_STOP_PHRASES = (
    # 问句模板：它们描述提问方式，而不是知识主题。
    "解决什么问题",
    "分别起什么作用",
    "有什么关系",
    "什么是",
    "是什么",
    # 这里**故意没有**"有哪些风险"：整串替换会把承载意图的"风险"一起吃掉，
    # 查询退化成纯主题词组，BM25 就分不清"提到过这个词的页面"和"讲这个词
    # 有什么风险的页面"。实测（dev 26 题，两套语料）去掉它只影响 retrieval-005
    # 一题，严格名次 4→2 / 3→2、宽松 2→1，其余 25 题分毫不动；保留它则
    # 由下面的"有哪些"接手匹配，"风险"得以留进查询。别再把它加回来。
    "有哪些",
    "有什么",
    "为什么",
    "如何",
    "怎么样",
    "分别",
    "作用",
    "问题",
    "它",
    # 范围词和流程问句中的通用词。应在二元组切分前清除，
    # 否则“系统从”会产生“统从”这类罕见但无语义的 token。
    "一个",
    "一种",
    "这个",
    "该",
    "基础",
    "用户问题",
    "用户",
    "系统",
    "最终回答",
    "回答",
    "最终",
    "通常",
    "经历",
    "哪些",
    "步骤",
    "能力",
    "方面",
    "从",
    "到",
)

# 这些字常在中文问句中连接多个成分。将它们替换为空白，可避免
# 生成“宽和”“和高”之类跨成分二元组；不包含“能”，以免破坏“性能”。
QUERY_CONNECTOR_PATTERN = re.compile(r"[的了和与在对将把由及等]")


def tokenize_query(text: str) -> set[str]:
    """去除常见疑问表达，再提取真正的检索词。"""
    cleaned_text = text.lower()

    for phrase in sorted(
        QUERY_STOP_PHRASES,
        key=len,
        reverse=True,
    ):
        cleaned_text = cleaned_text.replace(phrase, " ")

    cleaned_text = QUERY_CONNECTOR_PATTERN.sub(" ", cleaned_text)

    return set(tokenize(cleaned_text))

def build_idf(chunks: list[dict[str, str]]) -> dict[str, float]:
    """计算每个 token 在全部文本片段中的逆文档频率。"""
    document_frequency: dict[str, int] = {}

    for chunk in chunks:
        searchable_text = (
            f"{chunk['heading_path']}\n"
            f"{chunk['text']}"
        )
        unique_tokens = set(tokenize(searchable_text))

        for token in unique_tokens:
            document_frequency[token] = (
                document_frequency.get(token, 0) + 1
            )

    chunk_count = len(chunks)

    return {
        token: math.log((chunk_count + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }

# def score_chunk(
#     query_tokens: set[str],
#     chunk: dict[str, str],
# ) -> tuple[int, set[str]]:
#     """按 Query 与正文、标题路径的重叠程度给一个片段打分。"""
#     text_tokens = set(tokenize(chunk["text"]))
#     heading_tokens = set(tokenize(chunk["heading_path"]))

#     text_matches = query_tokens & text_tokens
#     heading_matches = query_tokens & heading_tokens

#     score = len(text_matches) + len(heading_matches)
#     matched_tokens = text_matches | heading_matches

#     return score, matched_tokens
def score_chunk(
    query_tokens: set[str],
    chunk: dict[str, str],
    idf: dict[str, float],
) -> tuple[float, set[str]]:
    """按 token 的稀有程度为片段打分。"""
    text_tokens = set(tokenize(chunk["text"]))
    heading_tokens = set(tokenize(chunk["heading_path"]))

    text_matches = query_tokens & text_tokens
    heading_matches = query_tokens & heading_tokens

    text_score = sum(idf.get(token, 0.0) for token in text_matches)

    # 标题是章节主题的浓缩，因此给标题中的匹配额外一次权重。
    heading_score = sum(
        idf.get(token, 0.0)
        for token in heading_matches
    )

    score = text_score + heading_score
    matched_tokens = text_matches | heading_matches

    return score, matched_tokens

# def search_chunks(
#     query: str,
#     chunks: list[dict[str, str]],
#     top_k: int = 5,
# ) -> list[tuple[dict[str, str], int, set[str]]]:
def search_chunks(
    query: str,
    chunks: list[dict[str, str]],
    idf: dict[str, float],
    top_k: int = 5,
) -> list[tuple[dict[str, str], float, set[str]]]:
    """返回与问题最匹配的前 top_k 个文本片段。"""
    # query_tokens = set(tokenize(query))
    query_tokens = tokenize_query(query)

    if not query_tokens:
        return []

    # results: list[tuple[dict[str, str], int, set[str]]] = []
    results: list[tuple[dict[str, str], float, set[str]]] = []

    for chunk in chunks:
        # score, matched_tokens = score_chunk(query_tokens, chunk)
        score, matched_tokens = score_chunk(query_tokens, chunk, idf)

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

        # print(f"\n[{index}] 分数：{score}")
        print(f"\n[{index}] 分数：{score:.2f}")
        print(f"来源：{chunk['source_file']}")
        print(f"标题路径：{chunk['heading_path']}")
        print(f"匹配 token：{', '.join(sorted(matched_tokens))}")
        print(f"预览：{preview}...")


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    chunks_path = project_dir / "data" / "generated" / "chunks.json"

    chunks = load_chunks(chunks_path)
    idf = build_idf(chunks)

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

        # results = search_chunks(query, chunks)
        results = search_chunks(query, chunks, idf)
        print_results(query, results)


if __name__ == "__main__":
    main()
