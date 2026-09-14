from pathlib import Path
import tempfile
import unittest

from src.evaluate_sft_behavior import score_response
from src.sft_data import load_sft_data, prepare_sft_data


class SFTDataTests(unittest.TestCase):
    def test_fact_entities_are_isolated_and_targets_are_self_consistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sft"
            diagnostics = prepare_sft_data(path, group_count=30)
            dataset, _, _ = load_sft_data(path)
            groups = {split: {item["group_id"] for item in values} for split, values in dataset.items()}
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertEqual(diagnostics["train_examples"] + diagnostics["validation_examples"] +
                             diagnostics["test_examples"], 90)

    def test_scoring_distinguishes_grounded_answer_and_safe_abstention(self):
        answer_case = {
            "expected_behavior": "answer", "must_mention": ["小林"],
            "required_citations": ["S1"], "available_citations": ["S1", "S2"],
        }
        checks, _ = score_response(answer_case, "结论：负责人是小林。\n依据：[S1]")
        self.assertTrue(checks["passed"])
        wrong, _ = score_response(answer_case, "结论：负责人是小林。\n依据：[S2]")
        self.assertFalse(wrong["citations_exact"])
        abstain_case = {
            "expected_behavior": "abstain", "must_mention": ["资料不足"],
            "required_citations": [], "available_citations": ["S1"],
        }
        refused, _ = score_response(abstain_case, "资料不足：给定证据无法回答这个问题。")
        self.assertTrue(refused["passed"])


if __name__ == "__main__":
    unittest.main()
