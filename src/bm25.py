import math
from collections import Counter
try:
    from .retrieve import tokenize_query, tokenize
except ImportError:
    from retrieve import tokenize_query, tokenize


class BM25Index:
    """一个针对当前 Markdown 片段集合的 BM25 索引。"""

    def __init__(
        self,
        chunks: list[dict[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []

        for chunk in chunks:
            # 标题路径重复一次，作为轻量标题加权。
            searchable_text = (
                f"{chunk['heading_path']}\n"
                f"{chunk['heading_path']}\n"
                f"{chunk['text']}"
            )

            tokens = tokenize(searchable_text)
            term_frequency = Counter(tokens)

            self.term_frequencies.append(term_frequency)
            self.document_lengths.append(len(tokens))

            for token in set(tokens):
                self.document_frequencies[token] += 1

        self.document_count = len(chunks)

        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )

        self.idf = {
            token: math.log(
                1
                + (
                    (self.document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )
            for token, document_frequency in self.document_frequencies.items()
        }

    def score(
        self,
        query_tokens: set[str],
        document_index: int,
    ) -> tuple[float, set[str]]:
        """计算某个查询对某个片段的 BM25 分数。"""
        term_frequency = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        if self.average_document_length == 0:
            return 0.0, set()

        score = 0.0
        matched_tokens: set[str] = set()

        length_normalization = (
            1
            - self.b
            + self.b
            * document_length
            / self.average_document_length
        )

        for token in query_tokens:
            frequency = term_frequency.get(token, 0)

            if frequency == 0:
                continue

            token_idf = self.idf.get(token, 0.0)

            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * length_normalization

            score += token_idf * numerator / denominator
            matched_tokens.add(token)

        return score, matched_tokens


def search_bm25(
    query: str,
    chunks: list[dict[str, str]],
    index: BM25Index,
    top_k: int = 5,
) -> list[tuple[dict[str, str], float, set[str]]]:
    """使用 BM25 返回前 top_k 个片段。"""
    query_tokens = tokenize_query(query)

    if not query_tokens:
        return []

    results: list[tuple[dict[str, str], float, set[str]]] = []

    for document_index, chunk in enumerate(chunks):
        score, matched_tokens = index.score(
            query_tokens,
            document_index,
        )

        if score > 0:
            results.append((chunk, score, matched_tokens))

    results.sort(key=lambda item: (-item[1], item[0]["id"]))

    return results[:top_k]