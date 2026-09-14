import unittest

from src.evidence_context import build_evidence_context


def result(text, identifier="a", **extras):
    return ({"id": identifier, "source_file": identifier + ".md", "heading_path": "原理",
             "text": text, **extras}, 1.0, {"pagedattention"})


class EvidenceContextTests(unittest.TestCase):
    def test_duplicate_text_appears_only_once(self):
        package = build_evidence_context([result("同一段资料"), result("同一段资料", "b")])
        self.assertEqual(len(package.sources), 1)
        self.assertTrue(any(d.get("reason") == "empty_or_duplicate_chunk" for d in package.diagnostics))

    def test_dedup_preserves_meaningful_code_whitespace(self):
        package = build_evidence_context([
            result("```py\nif ready:\n    run()\n```"),
            result("```py\nif ready:\nrun()\n```", "b"),
        ])
        self.assertEqual(len(package.sources), 2)

    def test_whole_bullets_and_budget(self):
        bullet = "- PagedAttention：通过分页管理 KV Cache，减少碎片。"
        inputs = [result(bullet + "\n- 其他技术：另外的解释。"), result("另一个完整段落。", "b")]
        for budget in range(1, 350):
            package = build_evidence_context(inputs, budget)
            self.assertLessEqual(len(package.text), budget)
            if "PagedAttention" in package.text:
                self.assertIn(bullet, package.text)

    def test_semantically_relevant_block_without_same_words_is_not_filtered(self):
        package = build_evidence_context([result("减少键值头数量可以降低缓存占用。")])
        self.assertIn("减少键值头数量", package.text)

    def test_colon_lead_and_short_list_are_one_semantic_unit(self):
        text = ("RAG 的基本流程分为两阶段：\n\n"
                "- 离线：提取、切分、向量化。\n"
                "- 在线：提问、检索、注入、生成。\n\n"
                "其他说明。")
        package = build_evidence_context([result(text)], max_chars=300)
        self.assertIn("RAG 的基本流程分为两阶段", package.text)
        self.assertIn("- 离线：提取、切分、向量化。", package.text)
        self.assertIn("- 在线：提问、检索、注入、生成。", package.text)

    def test_code_fence_closed_and_continuation_is_explicit(self):
        package = build_evidence_context([result("```py\nx=1")])
        self.assertTrue(package.text.endswith("```"))
        fragment = build_evidence_context([result("print('tail')\n```\n下一个段落。", continuation=True)])
        self.assertIn("续片", fragment.text)
        self.assertIn("````text", fragment.text)
        self.assertTrue(fragment.text.endswith("````"))

    def test_long_indivisible_block_is_skipped_not_sliced(self):
        package = build_evidence_context([result("一" * 2000)], max_chars=100)
        self.assertEqual(package.text, "")
        self.assertEqual(package.diagnostics[0]["budget_skipped_blocks"], 1)
