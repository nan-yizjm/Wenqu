import unittest

from src.answer import AnswerResult
from src.citations import CitationValidation
from src.context import Source
from src.generation_metrics import evaluate_answer


SOURCE = Source(
    label="S1",
    source_file="note.md",
    heading_path="测试标题",
)

VALID_CITATION = CitationValidation(
    cited_labels=("S1",),
    invalid_labels=(),
)


class GenerationMetricsTests(unittest.TestCase):
    def test_grounded_answer_passes(self) -> None:
        case = {
            "expected_behavior": "answer",
            "must_mention": ["PagedAttention", "KV Cache"],
            "must_cite": True,
        }
        result = AnswerResult(
            answer="PagedAttention 用于管理 KV Cache。[S1]",
            sources=(SOURCE,),
            rejected=False,
            citation_validation=VALID_CITATION,
        )

        checks = evaluate_answer(case, result)

        self.assertTrue(checks["passed"])

    def test_ood_question_must_be_rejected(self) -> None:
        case = {
            "expected_behavior": "reject",
            "must_mention": [],
            "must_cite": False,
        }
        result = AnswerResult(
            answer="当前知识库助手无法处理实时天气请求。",
            sources=(),
            rejected=True,
        )

        checks = evaluate_answer(case, result)

        self.assertTrue(checks["behavior_correct"])
        self.assertTrue(checks["passed"])

    def test_insufficient_material_must_not_guess(self) -> None:
        case = {
            "expected_behavior": "abstain",
            "must_mention": [],
            "must_cite": True,
        }
        result = AnswerResult(
            answer="当前知识库资料不足，无法提供完整 CUDA Kernel。[S1]",
            sources=(SOURCE,),
            rejected=False,
            citation_validation=VALID_CITATION,
        )

        checks = evaluate_answer(case, result)

        self.assertTrue(checks["behavior_correct"])
        self.assertTrue(checks["passed"])

    def test_speculation_causes_failure(self) -> None:
        case = {
            "expected_behavior": "answer",
            "must_mention": ["KV Cache"],
            "must_cite": True,
        }
        result = AnswerResult(
            answer="这可能优化 KV Cache 管理。[S1]",
            sources=(SOURCE,),
            rejected=False,
            citation_validation=VALID_CITATION,
        )

        checks = evaluate_answer(case, result)

        self.assertFalse(checks["no_forbidden_speculation"])
        self.assertFalse(checks["passed"])

    def test_evidence_based_abstention_passes(self) -> None:
        case = {
            "expected_behavior": "abstain",
            "must_mention": [],
            "must_cite": True,
        }
        result = AnswerResult(
            answer=(
                "资料中未给出明确支持完整 CUDA Kernel 的内容。[S1]"
            ),
            sources=(SOURCE,),
            rejected=False,
            citation_validation=VALID_CITATION,
        )

        checks = evaluate_answer(case, result)

        self.assertTrue(checks["behavior_correct"])
        self.assertTrue(checks["passed"])

if __name__ == "__main__":
    unittest.main()