import unittest
from unittest.mock import patch

from src.bm25 import BM25Index, search_bm25
from src.retrieve import tokenize
from src.hybrid_retrieve import RetrievalEngine, fuse_rankings
from src.context import build_context
from src.answer import has_separate_concept_coverage
from src.compare_retrievers import summarize


def chunk(identifier, heading="原理", text="PagedAttention 管理 KV Cache"):
    return {"id": identifier, "heading_path": heading, "source_file": "a.md",
            "document_title": "笔记", "text": text}


class HybridTests(unittest.TestCase):
    def test_bm25_only_path_still_equals_search_bm25(self):
        """产品只剩 engine 这一条检索路径，它必须与旧函数逐位等价。

        接管前 `retrieve()` 直接调 `search_bm25(raw=False)`：分数含质量系数、
        按 (分数降序, id 升序) 排。换成 `engine.search('bm25')` 后若哪一步走样
        （少了质量系数、换了排序键），界面上只是排序变得"有点怪"，没有测试
        就会被当成调优问题查很久。
        """
        chunks = [
            chunk("a"), chunk("b", "待继续拆分的概念"), chunk("c"),
            {"id": "d", "heading_path": "日志", "source_file": "b.md",
             "document_title": "笔记", "text": "今天下雨，缓存没命中。"},
        ]
        engine = RetrievalEngine(chunks)
        for query in ("PagedAttention KV Cache", "缓存", "不存在的词"):
            for top_k in (1, 2, 4):
                with self.subTest(query=query, top_k=top_k):
                    expected = search_bm25(query, chunks, engine.bm25, top_k=top_k, raw=False)
                    actual = engine.search(query, "bm25", top_k)
                    self.assertEqual([hit.chunk["id"] for hit in actual],
                                     [item[0]["id"] for item in expected])
                    for hit, (_, score, tokens) in zip(actual, expected, strict=True):
                        self.assertAlmostEqual(hit.score, score)
                        self.assertEqual(hit.matched_tokens, tokens)
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


class RetrievalParameterTests(unittest.TestCase):
    """参数化的唯一目的是做单变量实验，默认值必须与历史行为完全一致。

    这里刻意不写"默认等于默认"这种空断言：标题重复的历史行为是标题出现两次，
    所以直接按分词结果核对文档长度，公式错了才测得出。
    """

    def test_default_heading_repeat_still_puts_the_heading_in_twice(self):
        chunks = [chunk("a"), chunk("b", heading="KV Cache 与显存")]
        index = BM25Index(chunks)

        self.assertEqual(index.heading_repeat, 1)
        for position, item in enumerate(chunks):
            heading = len(tokenize(item["heading_path"]))
            text = len(tokenize(item["text"]))
            self.assertEqual(index.document_lengths[position], heading * 2 + text)

    def test_heading_repeat_zero_means_no_extra_weight_not_no_heading(self):
        """0 表示"标题不额外加权"，标题本身仍然出现一次。

        这一点容易写错预期：0 不是"把标题从检索文本里删掉"。所以这里核对的是
        词频（1 次 vs 2 次），而不是"命中消失"。
        """
        chunks = [chunk("a", heading="PagedAttention 原理", text="与标题无关的正文。")]
        plain = BM25Index(chunks, heading_repeat=0)
        weighted = BM25Index(chunks)

        heading = len(tokenize("PagedAttention 原理"))
        text = len(tokenize("与标题无关的正文。"))
        self.assertEqual(plain.document_lengths[0], heading + text)
        self.assertEqual(plain.term_frequencies[0]["pagedattention"], 1)
        self.assertEqual(weighted.term_frequencies[0]["pagedattention"], 2)
        self.assertTrue(search_bm25("PagedAttention", chunks, plain))

    def test_two_extra_repeats_are_counted_as_three_headings(self):
        chunks = [chunk("a", heading="原理", text="正文。")]
        index = BM25Index(chunks, heading_repeat=2)
        heading = len(tokenize("原理"))
        self.assertEqual(index.document_lengths[0], heading * 3 + len(tokenize("正文。")))

    def test_engine_passes_the_bm25_parameters_through(self):
        engine = RetrievalEngine([chunk("a")], k1=1.2, b=0.3, heading_repeat=0)

        self.assertEqual(engine.bm25.k1, 1.2)
        self.assertEqual(engine.bm25.b, 0.3)
        self.assertEqual(engine.bm25.heading_repeat, 0)

    def test_impossible_parameters_are_rejected(self):
        for kwargs in ({"k1": 0}, {"k1": -1.0}, {"b": -0.1}, {"b": 1.1},
                       {"heading_repeat": -1}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    RetrievalEngine([chunk("a")], **kwargs)
