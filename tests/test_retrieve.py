import unittest

from src.retrieve import tokenize_query


class TokenizeQueryTests(unittest.TestCase):
    def test_removes_inverted_definition_question_template(self) -> None:
        tokens = tokenize_query(
            "PagedAttention 是什么？它解决什么问题？"
        )

        self.assertEqual(tokens, {"pagedattention"})

    def test_removes_rag_process_question_scaffolding(self) -> None:
        tokens = tokenize_query(
            "一个基础 RAG 系统从用户问题到最终回答通常经历哪些步骤？"
        )

        self.assertEqual(tokens, {"rag"})

    def test_keeps_the_topic_word_inside_a_risk_question(self) -> None:
        """问句模板可以去掉，但承载意图的"风险"必须留下。

        "有哪些风险"整串替换会把"风险"一起吃掉，查询退化成纯主题词组，BM25 就
        分不清"提到过这个词的页面"与"讲这个词有什么风险的页面"。实测这会让
        retrieval-005 的期望文档从第 2 名掉到第 4 名。
        """
        tokens = tokenize_query("Prompt Injection 有哪些风险？")

        self.assertEqual(tokens, {"prompt", "injection", "风险"})
        self.assertNotIn("有哪些", tokens)


if __name__ == "__main__":
    unittest.main()
