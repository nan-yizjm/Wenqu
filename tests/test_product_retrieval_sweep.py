import unittest

from src.evaluate_product_retrieval import build_conditions, parse_sweep


class ParseSweepTests(unittest.TestCase):
    """实验项必须自带检索方式：同一个参数在不同路径上含义不同。"""

    def test_reads_mode_key_and_value(self):
        self.assertEqual(parse_sweep(["hybrid:rrf_k=10"]), [("hybrid:rrf_k=10", "hybrid", {"rrf_k": 10})])
        self.assertEqual(parse_sweep(["bm25:bm25_b=0.3"]), [("bm25:bm25_b=0.3", "bm25", {"bm25_b": 0.3})])

    def test_accepts_the_short_names_used_by_the_plan(self):
        """计划书里写作 b 和 k1；短名要落到真正的设置键上，标签仍保留原写法。"""
        self.assertEqual(parse_sweep(["bm25:b=0.3"]), [("bm25:b=0.3", "bm25", {"bm25_b": 0.3})])
        self.assertEqual(parse_sweep(["bm25:k1=1.2"]), [("bm25:k1=1.2", "bm25", {"bm25_k1": 1.2})])

    def test_rejects_a_missing_mode_or_assignment(self):
        for spec in ("rrf_k=10", "hybrid:rrf_k", "vector:rrf_k=10", "hybrid:rrf_k=x"):
            with self.subTest(spec=spec):
                with self.assertRaises(SystemExit):
                    parse_sweep([spec])

    def test_rejects_a_parameter_that_does_not_exist(self):
        with self.assertRaises(SystemExit):
            parse_sweep(["hybrid:nope=1"])

    def test_rejects_a_duplicate_specification(self):
        with self.assertRaises(SystemExit):
            parse_sweep(["bm25:b=0.3", "bm25:b=0.3"])


class BuildConditionsTests(unittest.TestCase):
    def test_baseline_is_the_two_modes_and_allows_swapping_the_whole_parameter_set(self):
        conditions = build_conditions([], rerank=False)
        self.assertEqual([label for label, _, _, _ in conditions], ["bm25", "hybrid"])
        for label, mode, patch, wants_rerank in conditions:
            self.assertEqual(label, mode)
            self.assertEqual(patch, {})
            self.assertFalse(wants_rerank)

    def test_each_sweep_becomes_its_own_condition(self):
        conditions = build_conditions(parse_sweep(["hybrid:rrf_k=10", "bm25:b=0.3"]), rerank=False)
        self.assertEqual([label for label, _, _, _ in conditions],
                         ["bm25", "hybrid", "hybrid:rrf_k=10", "bm25:b=0.3"])

    def test_rerank_duplicates_every_condition_including_the_sweeps(self):
        conditions = build_conditions(parse_sweep(["bm25:heading_repeat=0"]), rerank=True)
        self.assertEqual([label for label, _, _, _ in conditions],
                         ["bm25", "hybrid", "bm25:heading_repeat=0",
                          "bm25+rerank", "hybrid+rerank", "bm25:heading_repeat=0+rerank"])
        self.assertEqual([wants for _, _, _, wants in conditions],
                         [False, False, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
