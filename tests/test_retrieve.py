import unittest

from src.retrieve import tokenize_query


class TokenizeQueryTests(unittest.TestCase):
    def test_removes_inverted_definition_question_template(self) -> None:
        tokens = tokenize_query(
            "PagedAttention 是什么？它解决什么问题？"
        )

        self.assertEqual(tokens, {"pagedattention"})


if __name__ == "__main__":
    unittest.main()