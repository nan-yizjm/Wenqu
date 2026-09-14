import argparse
import sys
from pathlib import Path

from .bm25 import BM25Index, search_bm25
from .retrieve import load_chunks, tokenize_query


def print_separator() -> None:
    print("=" * 72)


def print_explanation(
    rank: int,
    chunk: dict[str, str],
    explanation: object,
) -> None:
    print(
        f"\n[{rank}] 最终分：{explanation.score:.4f} "
        f"（BM25 原始分：{explanation.raw_score:.4f}）"
    )

    if explanation.quality_reason:
        print(
            f"质量规则：×{explanation.quality_multiplier:.2f}；"
            f"{explanation.quality_reason}"
        )
    print(f"来源：{chunk['source_file']}")
    print(f"标题路径：{chunk['heading_path']}")
    print(
        "长度信息："
        f"文档 token 数={explanation.document_length}，"
        f"平均长度={explanation.average_document_length:.2f}，"
        f"长度归一化={explanation.length_normalization:.4f}"
    )

    print("Token 贡献：")
    for item in explanation.token_contributions:
        print(
            f"  - {item.token}: "
            f"TF={item.term_frequency}, "
            f"标题出现={item.heading_frequency}, "
            f"正文出现={item.text_frequency}, "
            f"IDF={item.idf:.4f}, "
            f"贡献={item.score:.4f}"
        )

    preview = chunk["text"][:220].replace("\n", " ")
    print(f"预览：{preview}...")


def main() -> None:
    # Windows 的部分终端仍使用 GBK；遇到知识库中的特殊 Unicode
    # 符号时以替代字符显示，避免诊断命令整体中断。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="解释一个问题的 BM25 检索排序原因。"
    )
    parser.add_argument("query", help="要分析的用户问题")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="显示前几个候选结果，默认 5",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    chunks_path = project_dir / "data" / "generated" / "chunks.json"

    chunks = load_chunks(chunks_path)
    index = BM25Index(chunks)
    query_tokens = tokenize_query(args.query)

    print_separator()
    print(f"问题：{args.query}")
    print(f"查询 token：{', '.join(sorted(query_tokens)) or '无'}")
    print(f"索引片段数：{len(chunks)}")

    results = search_bm25(
        args.query,
        chunks,
        index,
        top_k=args.top_k,
    )

    if not results:
        print("\n没有召回任何结果。")
        return

    chunk_positions = {
        chunk["id"]: position
        for position, chunk in enumerate(chunks)
    }

    print("\n检索解释：")
    for rank, (chunk, _, _) in enumerate(results, start=1):
        document_index = chunk_positions[chunk["id"]]
        explanation = index.explain_score(
            query_tokens,
            document_index,
        )
        print_explanation(rank, chunk, explanation)

    print_separator()


if __name__ == "__main__":
    main()
