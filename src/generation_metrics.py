"""生成式 RAG 回答的确定性评测指标。"""

from typing import Any

from .answer import AnswerResult


FORBIDDEN_SPECULATION_PHRASES = (
    "推测",
    "可能",
    "一般而言",
    "根据常识",
    "我认为",
    "大概",
)

ABSTENTION_MARKERS = (
    "知识库资料不足",
    "资料不足",
    "无法提供",
    "无法回答",
    "未提供详细",
    "无法直接找到",
    "未给出明确支持",
)


def evaluate_answer(
    case: dict[str, Any],
    result: AnswerResult,
) -> dict[str, bool]:
    """按评测题定义，对一次回答执行不依赖模型的检查。"""
    answer = result.answer
    normalized_answer = answer.lower()

    expected_behavior = case["expected_behavior"]
    required_mentions = case["must_mention"]
    must_cite = case["must_cite"]

    if expected_behavior == "answer":
        behavior_correct = not result.rejected
    elif expected_behavior == "reject":
        behavior_correct = result.rejected
    elif expected_behavior == "abstain":
        behavior_correct = (
            not result.rejected
            and any(marker in answer for marker in ABSTENTION_MARKERS)
        )
    else:
        raise ValueError(f"未知 expected_behavior：{expected_behavior}")

    mentions_present = all(
        term.lower() in normalized_answer
        for term in required_mentions
    )

    citation_valid = bool(
        result.citation_validation
        and result.citation_validation.is_valid
    )

    no_forbidden_speculation = not any(
        phrase in answer
        for phrase in FORBIDDEN_SPECULATION_PHRASES
    )

    passed = (
        behavior_correct
        and mentions_present
        and no_forbidden_speculation
        and (not must_cite or citation_valid)
    )

    return {
        "behavior_correct": behavior_correct,
        "mentions_present": mentions_present,
        "citation_valid": citation_valid,
        "no_forbidden_speculation": no_forbidden_speculation,
        "passed": passed,
    }