"""连续使用同一检索实例；各问题独立，不偷偷拼接多轮历史。"""

import argparse
import time
from pathlib import Path

from .answer import RAGAnswerer, create_client, print_answer_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("ollama", "deepseek"), default="ollama")
    parser.add_argument("--retriever", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--chunks-file", default="data/generated/chunks.json")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--context-policy", choices=("legacy", "blocks"), default="legacy")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    answerer = RAGAnswerer(root / args.chunks_file, create_client(args.provider),
                          retrieval_method=args.retriever, device=args.device,
                          rerank=args.rerank, rerank_device=args.rerank_device,
                          context_policy=args.context_policy)
    print(f"语料：{args.chunks_file}；重排序：{args.rerank}；上下文策略：{args.context_policy}")
    print("知识库问答：输入完整问题；:method bm25|vector|hybrid 切换；:quit 退出。")
    print("每道题独立；暂不支持依赖上一题的‘它呢’等省略问法。笔记更新后需重新导入并重启。")
    while True:
        try:
            question = input(f"\n[{answerer.retrieval_method}] > ").strip()
            if question == ":quit":
                break
            if not question:
                continue
            if question.startswith(":method "):
                method = question.split(maxsplit=1)[1].strip()
                if method in ("bm25", "vector", "hybrid"):
                    answerer.retrieval_method = method
                else:
                    print("可选方式：bm25 / vector / hybrid")
                continue
            started = time.perf_counter()
            print_answer_result(answerer.answer(question))
            print(f"本次总耗时：{time.perf_counter() - started:.2f} 秒（首次向量查询含模型加载）")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        except (ValueError, RuntimeError, OSError) as error:
            print(f"本次请求未完成：{error}")
            print("请检查模型缓存、Ollama 服务或网络配置；可换 bm25 / CPU 后重试。")


if __name__ == "__main__":
    main()
