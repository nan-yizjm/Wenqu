import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.answer import (
    RAGAnswerer,
    has_separate_concept_coverage,
    requested_separate_concepts,
)


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
        self.assertIn("prepare_retrieval_ms", result.phase_timings_ms)
        self.assertIn("generation_wall_ms", result.phase_timings_ms)
        self.assertIn("finalize_ms", result.phase_timings_ms)
        self.assertGreaterEqual(
            result.phase_timings_ms["runtime_total_ms"],
            result.phase_timings_ms["prepare_total_ms"],
        )

    def test_realtime_question_is_rejected_without_calling_llm(self) -> None:
        result = self.answerer.answer("北京明天天气怎么样？")

        self.assertTrue(result.rejected)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(result.sources, ())
        self.assertEqual(result.phase_timings_ms["generation_wall_ms"], 0.0)
        self.assertIn("prepare_guard_ms", result.phase_timings_ms)

    def test_hybrid_rejection_does_not_load_embedding_model(self) -> None:
        self.answerer.retrieval_method = "hybrid"
        with patch.object(self.answerer.engine, "ensure_vector", side_effect=AssertionError("不应加载")):
            result = self.answerer.answer("北京明天天气怎么样？")
        self.assertTrue(result.rejected)
        self.assertEqual(result.generation_calls, 0)

    def test_repairs_at_most_once_and_reports_remaining_gap(self) -> None:
        # FakeClient 永远只回答 PagedAttention，修复后仍不会覆盖两个概念。
        result = self.answerer.answer("PagedAttention 和 GQA 分别是什么？")
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(result.generation_calls, 2)
        self.assertFalse(result.separate_coverage_ok)
        self.assertTrue(result.context_text)
        self.assertTrue(result.retrieval_trace)

    def test_extracts_concepts_before_separate_question_word(self) -> None:
        concepts = requested_separate_concepts(
            "MLA、GQA 和 MQA 分别通过什么方式降低 KV Cache 的压力？"
        )

        self.assertEqual(concepts, ("MLA", "GQA", "MQA"))

    def test_requires_different_list_items_for_each_concept(self) -> None:
        concepts = ("MLA", "GQA", "MQA")

        self.assertFalse(
            has_separate_concept_coverage(
                "- MLA、GQA 和 MQA 都能降低 KV Cache 压力。",
                concepts,
            )
        )
        self.assertTrue(
            has_separate_concept_coverage(
                "- MLA：压缩 KV 表示。\n"
                "- GQA：共享一组 KV 头。\n"
                "- MQA：共享一组 KV 头。",
                concepts,
            )
        )

    def test_real_token_budget_repacks_context_without_retrieving_twice(self) -> None:
        chunks = [{
            "id": "long#paged-attention",
            "source_file": "long.md",
            "document_title": "Long",
            "heading_path": "Long > PagedAttention",
            "text": "PagedAttention " + "证据内容" * 200,
        }]
        self.chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        client = FakeClient()
        answerer = RAGAnswerer(
            self.chunks_path,
            client,
            max_context_chars=1000,
            message_builder=lambda _question, context: [{"role": "user", "content": context}],
            message_token_counter=lambda messages: len(messages[0]["content"]),
            max_prompt_tokens=220,
        )
        with patch.object(answerer.engine, "search", wraps=answerer.engine.search) as search:
            prepared = answerer.prepare("PagedAttention 是什么？")

        self.assertIsNone(prepared.immediate_result)
        self.assertEqual(search.call_count, 1)
        self.assertLessEqual(prepared.prompt_tokens, 220)
        self.assertLess(prepared.context_char_budget, 1000)
        self.assertTrue(any(item.get("reason") == "prompt_token_budget"
                            for item in prepared.context_diagnostics))

    def test_budget_exhausted_by_prompt_does_not_call_model(self) -> None:
        client = FakeClient()
        answerer = RAGAnswerer(
            self.chunks_path,
            client,
            message_builder=lambda question, context: [
                {"role": "user", "content": "固定提示" * 20 + question + context}
            ],
            message_token_counter=lambda messages: len(messages[0]["content"]),
            max_prompt_tokens=32,
        )
        result = answerer.answer("PagedAttention？")
        self.assertEqual(result.generation_calls, 0)
        self.assertEqual(client.calls, [])
        self.assertIn("token 预算", result.answer)
        self.assertEqual(result.context_diagnostics[0]["reason"],
                         "prompt_token_budget_exhausted_by_question_and_instructions")


if __name__ == "__main__":
    unittest.main()
