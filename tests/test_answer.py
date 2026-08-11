import json
import tempfile
import unittest
from pathlib import Path

from src.answer import RAGAnswerer


class FakeClient:
    """用于测试的假模型：记录收到的消息，不访问任何真实模型。"""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return "PagedAttention 通过分页方式管理 KV Cache，从而减少碎片。[S1]"


class RAGAnswererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.temp_dir.name) / "chunks.json"

        chunks = [
            {
                "id": "inference#paged-attention",
                "source_file": "06_推理与模型服务/推理与模型服务.md",
                "document_title": "推理与模型服务",
                "heading_path": "推理与模型服务 > 系统级优化",
                "text": (
                    "PagedAttention 像操作系统分页一样管理 KV Cache，"
                    "用于减少显存碎片。"
                ),
            }
        ]

        self.chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False),
            encoding="utf-8",
        )

        self.client = FakeClient()
        self.answerer = RAGAnswerer(
            chunks_path=self.chunks_path,
            client=self.client,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_answer_uses_retrieved_context_and_returns_sources(self) -> None:
        result = self.answerer.answer(
            "PagedAttention 解决什么问题？"
        )

        self.assertFalse(result.rejected)
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn("[S1]", self.client.calls[0][1]["content"])
        self.assertEqual(result.sources[0].label, "S1")
        self.assertIn("PagedAttention", result.answer)

    def test_realtime_question_is_rejected_without_calling_llm(self) -> None:
        result = self.answerer.answer("北京明天天气怎么样？")

        self.assertTrue(result.rejected)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(result.sources, ())


if __name__ == "__main__":
    unittest.main()