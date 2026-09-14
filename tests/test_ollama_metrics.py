import unittest

from src.llm import ollama_metrics


class OllamaMetricsTests(unittest.TestCase):
    def test_units_and_missing_values(self):
        result = ollama_metrics({"load_duration": 2_000_000_000, "eval_count": 40,
                                 "eval_duration": 500_000_000}, 2600)
        self.assertEqual(result["load_ms"], 2000)
        self.assertEqual(result["decode_tokens_per_second"], 80)
        self.assertIsNone(result["prompt_eval_cached_count"])
        self.assertIsNone(result["prompt_eval_ms"])
        self.assertIsNone(ollama_metrics({"eval_count": 1, "eval_duration": 0}, 1)["decode_tokens_per_second"])
