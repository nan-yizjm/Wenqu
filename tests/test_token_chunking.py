import unittest

from src.markdown_structure import markdown_blocks, sections_with_offsets
from src.token_chunking import TokenChunker


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text) + 2))}


class TokenChunkingTests(unittest.TestCase):
    def setUp(self):
        self.chunker = TokenChunker(CharacterTokenizer(), max_tokens=96, overlap_tokens=12)

    def test_exact_source_offsets_limit_and_no_loss(self):
        raw = "---\ntags: [test]\n---\n# 标题\n\n" + "中文🙂数据；" * 100 + "\n\n- 完整条目。\n"
        chunks, report = self.chunker.create_chunks([{"source_file": "a.md", "title": "标题", "text": raw}])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(report["uncovered_sections"], 0)
        self.assertEqual(report["over_limit_chunks"], 0)
        for chunk in chunks:
            self.assertLessEqual(chunk["embedding_tokens"], 96)
            self.assertEqual(chunk["text"], raw[chunk["source_start_char"]:chunk["source_end_char"]])
            self.assertGreaterEqual(chunk["line_start"], 6)

    def test_headings_in_both_fence_types_are_not_sections(self):
        raw = "# Real\n\n~~~~python\n# Not a heading\n~~~\n# Still code\n~~~~\n\n## Next\nBody"
        sections = sections_with_offsets(raw, "Fallback")
        self.assertEqual([heading for heading, _ in sections], ["Real", "Real > Next"])
        self.assertEqual(markdown_blocks("```py\nx=1\n\nx=2\n```\n")[0].kind, "code")

    def test_unrelated_section_insertion_preserves_content_ids(self):
        raw = "# Doc\n## Existing\n原来的一小段内容。"
        before, _ = self.chunker.create_chunks([{"source_file": "a.md", "text": raw}])
        after, _ = self.chunker.create_chunks([{"source_file": "a.md", "text": raw.replace("## Existing", "## New\n新增内容。\n## Existing")}])
        original = next(chunk for chunk in after if chunk["text"] == before[0]["text"])
        self.assertEqual(before[0]["id"], original["id"])
        self.assertNotEqual(before[0]["line_start"], original["line_start"])

    def test_long_code_is_marked_and_short_code_stays_whole(self):
        short = "# Doc\n```py\nx=1\n```"
        chunks, _ = self.chunker.create_chunks([{"source_file": "s.md", "text": short}])
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0]["continuation"])
        long = "# Doc\n```py\n" + "print('很长的代码')\n" * 30 + "```"
        chunks, _ = self.chunker.create_chunks([{"source_file": "l.md", "text": long}])
        self.assertTrue(any(chunk["continuation"] for chunk in chunks))

    def test_empty_document_and_invalid_budget(self):
        self.assertEqual(self.chunker.create_chunks([{"source_file": "empty.md", "text": "# Heading\n"}])[0], [])
        with self.assertRaises(ValueError):
            TokenChunker(CharacterTokenizer(), max_tokens=20)
