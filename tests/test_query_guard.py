import unittest

from src.query_guard import static_corpus_rejection_reason


class StaticCorpusQueryGuardTests(unittest.TestCase):
    def test_rejects_realtime_question(self) -> None:
        reason = static_corpus_rejection_reason(
            "北京明天的天气怎么样？"
        )

        self.assertIsNotNone(reason)
        self.assertIn("实时或最新外部数据", reason)

    def test_rejects_external_action(self) -> None:
        reason = static_corpus_rejection_reason(
            "请帮我预订明天从北京飞往上海的机票。"
        )

        self.assertIsNotNone(reason)
        self.assertIn("外部动作", reason)

    def test_rejects_missing_personal_context(self) -> None:
        reason = static_corpus_rejection_reason(
            "请根据我的简历推荐一份适合我的工作。"
        )

        self.assertIsNotNone(reason)
        self.assertIn("个人资料", reason)

    def test_allows_static_technical_question(self) -> None:
        reason = static_corpus_rejection_reason(
            "PagedAttention 如何管理 KV Cache？"
        )

        self.assertIsNone(reason)

    def test_allows_static_conceptual_question(self) -> None:
        reason = static_corpus_rejection_reason(
            "RLHF、DPO 和 RLVR 的区别是什么？"
        )

        self.assertIsNone(reason)

    def test_allows_soft_realtime_word_with_static_context(self) -> None:
        for question in (
            "最新论文对这个问题的结论是什么？",
            "价格模型是怎么设计的？",
            "这份文档里的最新进展是什么？",
            "实时推理的延迟怎么优化？",
            "我的笔记里当前阶段的目标是什么？",
        ):
            with self.subTest(question=question):
                self.assertIsNone(static_corpus_rejection_reason(question))

    def test_allows_current_word_in_plain_technical_question(self) -> None:
        """“当前”在技术提问里太常见，无静态语境也不该拦。"""
        for question in (
            "当前 分页管理 是怎么做的？",
            "当前有哪些瓶颈？",
            "现在的检索实现和以前有什么区别？",
            "当前版本的向量维度是多少？",
        ):
            with self.subTest(question=question):
                self.assertIsNone(static_corpus_rejection_reason(question))

    def test_still_rejects_soft_realtime_word_without_context(self) -> None:
        reason = static_corpus_rejection_reason("今天有什么新消息？")

        self.assertIsNotNone(reason)
        self.assertIn("实时或最新外部数据", reason)

    def test_hard_realtime_word_ignores_static_context(self) -> None:
        """静态语境只救得回软词；硬词是无歧义的实时数据请求。"""
        reason = static_corpus_rejection_reason("我的笔记里有没有今天的股价？")

        self.assertIsNotNone(reason)
        self.assertIn("实时或最新外部数据", reason)


if __name__ == "__main__":
    unittest.main()