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


if __name__ == "__main__":
    unittest.main()