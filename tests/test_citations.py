import unittest

from src.citations import extract_citation_labels, validate_citations
from src.context import Source


class CitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = (
            Source(
                label="S1",
                source_file="note-a.md",
                heading_path="标题 A",
            ),
            Source(
                label="S2",
                source_file="note-b.md",
                heading_path="标题 B",
            ),
        )

    def test_extracts_multiple_citation_formats(self) -> None:
        labels = extract_citation_labels(
            "结论一 [S1]；结论二 [S1, S2]；结论三 [S2]。"
        )

        self.assertEqual(labels, ("S1", "S2"))

    def test_accepts_existing_citations(self) -> None:
        result = validate_citations(
            "PagedAttention 管理 KV Cache [S1][S2]。",
            self.sources,
        )

        self.assertTrue(result.has_citations)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.invalid_labels, ())

    def test_rejects_nonexistent_citation_label(self) -> None:
        result = validate_citations(
            "这个结论来自不存在的资料 [S99]。",
            self.sources,
        )

        self.assertTrue(result.has_citations)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_labels, ("S99",))


if __name__ == "__main__":
    unittest.main()