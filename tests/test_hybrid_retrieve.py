import unittest
from unittest.mock import patch

from src.hybrid_retrieve import RetrievalEngine, fuse_rankings
from src.context import build_context
from src.answer import has_separate_concept_coverage
from src.compare_retrievers import summarize


def chunk(identifier, heading="原理"):
    return {"id": identifier, "heading_path": heading, "source_file": "a.md",
            "document_title": "笔记", "text": "PagedAttention 管理 KV Cache"}


class HybridTests(unittest.TestCase):
    def test_rrf_uses_ranks_not_incomparable_raw_scores(self):
        a, b = chunk("a"), chunk("b")
        hits = fuse_rankings({"bm25": [(a, 9000, set()), (b, 10, set())],
                              "vector": [(b, .95, set()), (a, .90, set())]})
        self.assertAlmostEqual(hits[0].raw_score, 1/61 + 1/62)
        self.assertAlmostEqual(hits[1].raw_score, hits[0].raw_score)

    def test_duplicate_in_one_channel_only_votes_once(self):
        a = chunk("a")
        hits = fuse_rankings({"bm25": [(a, 1, set()), (a, 1, set())]})
        self.assertAlmostEqual(hits[0].raw_score, 1/61)

    def test_quality_applied_once_after_fusion(self):
        a = chunk("a", "待继续拆分的概念")
        engine = RetrievalEngine([a])
        with patch.object(engine, "ensure_vector") as vector:
            vector.return_value.search.return_value = [(a, .9)]
            hit = engine.search("PagedAttention", "hybrid", 1)[0]
        self.assertAlmostEqual(hit.raw_score, 2/61)
        self.assertAlmostEqual(hit.score, (2/61) * .35)

    def test_bm25_and_blank_queries_do_not_load_vector(self):
        engine = RetrievalEngine([chunk("a")])
        with patch.object(engine, "ensure_vector", side_effect=AssertionError("不应加载")):
            self.assertEqual(engine.search(" ", "hybrid"), [])
            self.assertTrue(engine.search("PagedAttention", "bm25"))
        with self.assertRaises(ValueError):
            engine.search("x", "hybrid", 21)

    def test_tiny_context_never_exceeds_character_budget(self):
        results = [(chunk(str(i)), 1, set()) for i in range(5)]
        for budget in range(1, 400):
            self.assertLessEqual(len(build_context(results, max_chars=budget).text), budget)

    def test_duplicate_labels_do_not_count_as_three_concepts(self):
        self.assertFalse(has_separate_concept_coverage(
            "- MLA：一\n- MLA：二\n- MLA：三", ("MLA", "GQA", "MQA")))

    def test_empty_denominator_is_not_zero_percent(self):
        self.assertIsNone(summarize([], 5)["ood_rejection_rate"])
