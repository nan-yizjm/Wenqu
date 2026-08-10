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


if __name__ == "__main__":
    unittest.main()