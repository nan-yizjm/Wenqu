import math
from dataclasses import dataclass
from collections import Counter
try:
    from .retrieve import tokenize_query, tokenize
except ImportError:
    from retrieve import tokenize_query, tokenize

@dataclass(frozen=True)
class TokenContribution:
    """一个查询 token 对某个文档 BM25 分数的贡献。"""

    token: str
    term_frequency: int
    heading_frequency: int
    text_frequency: int
    idf: float
    score: float


@dataclass(frozen=True)
class ScoreExplanation:
    """一个文档相对某个查询的 BM25 评分解释。"""

    score: float
    raw_score: float
    quality_multiplier: float
    quality_reason: str | None
    matched_tokens: frozenset[str]
    document_length: int
    average_document_length: float
    length_normalization: float
    token_contributions: tuple[TokenContribution, ...]


LOW_INFORMATION_HEADING_MARKERS = (
    "待继续拆分",
)

LOW_INFORMATION_MULTIPLIER = 0.35


def get_quality_adjustment(
    chunk: dict[str, str],
) -> tuple[float, str | None]:
    """识别导航型、待完善型片段，避免其仅因短小而主导排序。"""
    heading_path = chunk["heading_path"]

    for marker in LOW_INFORMATION_HEADING_MARKERS:
        if marker in heading_path:
            return (
                LOW_INFORMATION_MULTIPLIER,
                f"标题路径包含“{marker}”，属于待完善的导航型片段。",
            )

    return 1.0, None


class BM25Index:
    """一个针对当前 Markdown 片段集合的 BM25 索引。"""

    def __init__(
        self,
        chunks: list[dict[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
        heading_repeat: int = 1,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 必须为正数")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b 必须落在 0 到 1 之间")
        if heading_repeat < 0:
            raise ValueError("heading_repeat 不能为负数")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.heading_repeat = heading_repeat

        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []
        self.heading_frequencies: list[Counter[str]] = []
        self.text_frequencies: list[Counter[str]] = []

        for chunk in chunks:
            self.heading_frequencies.append(Counter(tokenize(chunk["heading_path"])))
            self.text_frequencies.append(Counter(tokenize(chunk["text"])))
            # 标题路径额外重复 `heading_repeat` 次，作为轻量标题加权。
            # 默认 1 表示标题出现两次，与历史行为逐字节一致：这个参数是为单变量
            # 实验才放开的，默认值不动。
            searchable_text = "\n".join(
                [chunk["heading_path"]] * (heading_repeat + 1) + [chunk["text"]])

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
        raw: bool = False,
    ) -> tuple[float, set[str]]:
        """计算某个查询对某个片段的 BM25 分数。"""
        explanation = self.explain_score(
            query_tokens,
            document_index,
        )

        value = explanation.raw_score if raw else explanation.score
        return value, set(explanation.matched_tokens)

    def explain_score(
        self,
        query_tokens: set[str],
        document_index: int,
    ) -> ScoreExplanation:
        """返回 BM25 分数及其组成，供检索诊断使用。"""
        term_frequency = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]
        chunk = self.chunks[document_index]

        if self.average_document_length == 0:
            return ScoreExplanation(
                score=0.0,
                raw_score=0.0,
                quality_multiplier=1.0,
                quality_reason=None,
                matched_tokens=frozenset(),
                document_length=document_length,
                average_document_length=0.0,
                length_normalization=0.0,
                token_contributions=(),
            )

        length_normalization = (
            1
            - self.b
            + self.b
            * document_length
            / self.average_document_length
        )

        heading_frequency = self.heading_frequencies[document_index]
        text_frequency = self.text_frequencies[document_index]

        raw_score = 0.0
        contributions: list[TokenContribution] = []

        for token in sorted(query_tokens):
            frequency = term_frequency.get(token, 0)

            if frequency == 0:
                continue

            token_idf = self.idf.get(token, 0.0)
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * length_normalization
            token_score = token_idf * numerator / denominator

            raw_score += token_score
            contributions.append(
                TokenContribution(
                    token=token,
                    term_frequency=frequency,
                    heading_frequency=heading_frequency.get(token, 0),
                    text_frequency=text_frequency.get(token, 0),
                    idf=token_idf,
                    score=token_score,
                )
            )

        quality_multiplier, quality_reason = get_quality_adjustment(
            chunk
        )
        score = raw_score * quality_multiplier

        return ScoreExplanation(
            score=score,
            raw_score=raw_score,
            quality_multiplier=quality_multiplier,
            quality_reason=quality_reason,
            matched_tokens=frozenset(
                item.token for item in contributions
            ),
            document_length=document_length,
            average_document_length=self.average_document_length,
            length_normalization=length_normalization,
            token_contributions=tuple(contributions),
        )


def search_bm25(
    query: str,
    chunks: list[dict[str, str]],
    index: BM25Index,
    top_k: int = 5,
    raw: bool = False,
) -> list[tuple[dict[str, str], float, set[str]]]:
    """使用 BM25 返回前 top_k 个片段。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    query_tokens = tokenize_query(query)

    if not query_tokens:
        return []

    results: list[tuple[dict[str, str], float, set[str]]] = []

    for document_index, chunk in enumerate(chunks):
        score, matched_tokens = index.score(
            query_tokens,
            document_index,
            raw=raw,
        )

        if score > 0:
            results.append((chunk, score, matched_tokens))

    results.sort(key=lambda item: (-item[1], item[0]["id"]))

    return results[:top_k]
