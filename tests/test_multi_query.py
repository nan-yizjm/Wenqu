import unittest

from src.multi_query import decompose_query


class MultiQueryTests(unittest.TestCase):
    def test_decomposes_agent_component_question(self) -> None:
        query = (
            "Skills、Memory 和 Context Engineering "
            "在 Agent 中分别起什么作用？"
        )

        result = decompose_query(query)

        self.assertEqual(
            result,
            [
                "Skills 在 Agent 中 起什么作用？",
                "Memory 在 Agent 中 起什么作用？",
                "Context Engineering 在 Agent 中 起什么作用？",
            ],
        )

    def test_decomposes_capability_question(self) -> None:
        query = "RAG、Agent 和多模态能力分别为大模型补足了什么？"

        result = decompose_query(query)

        self.assertEqual(
            result,
            [
                "RAG 为大模型补足了什么？",
                "Agent 为大模型补足了什么？",
                "多模态能力 为大模型补足了什么？",
            ],
        )

    def test_keeps_relationship_question_intact(self) -> None:
        query = "GRPO 和 RLVR 有什么关系？"

        result = decompose_query(query)

        self.assertEqual(result, [query])


if __name__ == "__main__":
    unittest.main()