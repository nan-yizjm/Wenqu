"""回答引用的提取与结构校验。"""

import re
from dataclasses import dataclass
from typing import Sequence

from .context import Source


BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")
LABEL_PATTERN = re.compile(r"S\d+", re.IGNORECASE)


@dataclass(frozen=True)
class CitationValidation:
    """模型回答中引用标签的结构校验结果。"""

    cited_labels: tuple[str, ...]
    invalid_labels: tuple[str, ...]

    @property
    def has_citations(self) -> bool:
        """回答中是否至少出现了一个引用标签。"""
        return bool(self.cited_labels)

    @property
    def is_valid(self) -> bool:
        """是否所有引用都来自本次实际提供的上下文。"""
        return self.has_citations and not self.invalid_labels


def extract_citation_labels(answer: str) -> tuple[str, ...]:
    """从 [S1]、[S1, S2]、[S1][S2] 等格式中提取不重复标签。"""
    labels: list[str] = []

    for bracket_content in BRACKET_PATTERN.findall(answer):
        for raw_label in LABEL_PATTERN.findall(bracket_content):
            label = raw_label.upper()

            if label not in labels:
                labels.append(label)

    return tuple(labels)


def validate_citations(
    answer: str,
    sources: Sequence[Source],
) -> CitationValidation:
    """校验回答中出现的引用是否都存在于本次上下文来源内。"""
    cited_labels = extract_citation_labels(answer)
    available_labels = {source.label for source in sources}

    invalid_labels = tuple(
        label
        for label in cited_labels
        if label not in available_labels
    )

    return CitationValidation(
        cited_labels=cited_labels,
        invalid_labels=invalid_labels,
    )