import unittest

from src.context import build_context


class BuildContextTests(unittest.TestCase):
    def test_builds_numbered_context_and_sources(self) -> None:
        results = [
            (
                {
                    "source_file": "06_推理与模型服务/推理与模型服务.md",
                    "heading_path": "推理与模型服务 > 系统级优化",
                    "text": "PagedAttention 像操作系统分页一样管理 KV Cache。",
                },
                8.5,
                {"pagedattention"},
            ),
            (
                {
                    "source_file": "02_基础设施与框架/基础设施与框架.md",
                    "heading_path": "基础设施与框架 > 推理框架",
                    "text": "vLLM 使用 PagedAttention 提升推理吞吐。",
                },
                7.2,
                {"pagedattention"},
            ),
        ]

        package = build_context(results)

        self.assertIn("[S1]", package.text)
        self.assertIn("[S2]", package.text)
        self.assertEqual(package.sources[0].label, "S1")
        self.assertEqual(
            package.sources[0].source_file,
            "06_推理与模型服务/推理与模型服务.md",
        )
        self.assertEqual(package.sources[1].label, "S2")

    def test_respects_character_budget(self) -> None:
        results = [
            (
                {
                    "source_file": "note.md",
                    "heading_path": "测试标题",
                    "text": "知识" * 200,
                },
                1.0,
                {"知识"},
            )
        ]

        package = build_context(results, max_chars=100)

        self.assertLessEqual(len(package.text), 100)
        self.assertEqual(len(package.sources), 1)
        self.assertTrue(package.text.endswith("…"))

    def test_reserves_context_for_lower_ranked_sources(self) -> None:
        results = [
            (
                {
                    "source_file": f"note-{index}.md",
                    "heading_path": f"标题 {index}",
                    "text": f"证据 {index} " + "内容" * 100,
                },
                float(4 - index),
                {"证据"},
            )
            for index in range(1, 4)
        ]

        package = build_context(
            results,
            max_chars=360,
            min_content_chars_per_source=40,
        )

        self.assertEqual(len(package.sources), 3)
        self.assertIn("证据 3", package.text)
        self.assertLessEqual(len(package.text), 360)

    def test_prefers_matching_evidence_lines_over_chunk_beginning(self) -> None:
        results = [
            (
                {
                    "source_file": "kv-cache.md",
                    "heading_path": "推理服务 > KV Cache 优化",
                    "text": (
                        "这一行与问题无关。\n"
                        "这一行也只是背景。\n"
                        "**架构路线：**\n"
                        "- **GQA / MQA**：减少 KV head 数量。\n"
                        "- **MLA**：压缩 KV 表示。"
                    ),
                },
                8.0,
                {"gqa", "mqa", "mla"},
            )
        ]

        package = build_context(results, max_chars=300)

        self.assertIn("架构路线", package.text)
        self.assertIn("GQA / MQA", package.text)
        self.assertIn("MLA", package.text)
        self.assertNotIn("这一行与问题无关", package.text)


if __name__ == "__main__":
    unittest.main()
