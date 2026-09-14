import unittest
from unittest.mock import patch

from src.hybrid_retrieve import RetrievalEngine


class FakeReranker:
    max_length = 512

    def __init__(self):
        self.seen = []

    def score(self, query, chunks):
        self.seen = [chunk["id"] for chunk in chunks]
        return [(-5.0 + index, 520 if index == 0 else 80) for index in range(len(chunks))]


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"id": str(i), "heading_path": "原理", "text": "PagedAttention 缓存",
                        "source_file": "a.md"} for i in range(3)]

    def test_rerank_is_bounded_and_negative_score_not_reweighted(self):
        reranker = FakeReranker()
        engine = RetrievalEngine(self.chunks, rerank=True, rerank_top_n=2, reranker=reranker)
        hits = engine.search("PagedAttention", "bm25", 2)
        self.assertEqual(len(reranker.seen), 2)
        self.assertEqual(hits[0].rerank_score, -4.0)
        self.assertEqual(hits[0].pre_rerank_rank, 2)
        self.assertEqual(engine.last_rerank_stats["truncated_pairs"], 1)
        self.assertIsNotNone(hits[0].retrieval_score)

    def test_disabled_empty_and_invalid_options(self):
        engine = RetrievalEngine(self.chunks)
        with patch.object(engine, "ensure_reranker", side_effect=AssertionError("不应加载")):
            self.assertTrue(engine.search("PagedAttention"))
        engine.rerank_enabled = True
        with patch.object(engine, "ensure_reranker", side_effect=AssertionError("不应加载")):
            self.assertEqual(engine.search(" "), [])
        with self.assertRaises(ValueError):
            RetrievalEngine(self.chunks, candidate_k=2, rerank=True, rerank_top_n=3)

    def test_rerank_failure_is_not_silently_changed_to_retrieval(self):
        engine = RetrievalEngine(self.chunks, rerank=True, reranker=FakeReranker())
        with patch.object(engine.reranker, "score", side_effect=RuntimeError("模型出错")):
            with self.assertRaises(RuntimeError):
                engine.search("PagedAttention")
