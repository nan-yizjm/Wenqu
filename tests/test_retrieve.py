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


if __name__ == "__main__":
    unittest.main()
