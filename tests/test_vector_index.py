import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.vector_retrieve import VectorIndex


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def get_embedding_dimension(self):
        return 2

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [list(range(len(text))) for text in texts]}

    def encode(self, texts, **kwargs):
        self.calls.append(texts)
        if isinstance(texts, str):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class VectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "index.npz"
        self.chunks = [{"id": "a", "heading_path": "标题", "text": "第一段"},
                       {"id": "b", "heading_path": "标题", "text": "第二段"}]

    def make_index(self, chunks=None, **kwargs):
        encoder = FakeEncoder()
        index = VectorIndex(self.chunks if chunks is None else chunks,
                            cache_path=self.path, encoder=encoder, **kwargs)
        return index, encoder

    def test_cache_hit_does_not_encode_corpus_but_encodes_query(self):
        original, _ = self.make_index()
        cached, encoder = self.make_index()
        self.assertEqual(cached.cache_status, "hit")
        self.assertEqual(encoder.calls, [])
        np.testing.assert_array_equal(original.embeddings, cached.embeddings)
        cached.search("一个完整的问题？")
        self.assertEqual(encoder.calls, ["query: 一个完整的问题？"])

    def test_content_order_and_model_config_invalidate_cache(self):
        self.make_index()
        reversed_index, _ = self.make_index(self.chunks[::-1])
        self.assertEqual(reversed_index.cache_status, "stale")
        changed = [dict(chunk) for chunk in self.chunks[::-1]]
        changed[0]["text"] = "新增的知识"
        self.assertEqual(self.make_index(changed)[0].cache_status, "stale")
        self.assertEqual(self.make_index(changed, max_length=256)[0].cache_status, "stale")

    def test_corrupt_cache_is_rebuilt(self):
        self.path.write_bytes(b"not an npz file")
        index, encoder = self.make_index()
        self.assertEqual(index.cache_status, "invalid")
        self.assertEqual(len(encoder.calls), 1)

    def test_invalid_matrix_is_rebuilt(self):
        index, _ = self.make_index()
        np.savez(self.path, embeddings=np.array([[float("nan"), 0]] * 2, dtype=np.float32),
                 metadata=json.dumps(index.metadata), truncated_chunk_count=0)
        self.assertEqual(self.make_index()[0].cache_status, "invalid")

    def test_empty_index_and_blank_query(self):
        index, encoder = self.make_index([])
        self.assertEqual(index.search("问题"), [])
        self.assertEqual(index.embeddings.shape, (0, 2))
        self.assertEqual(encoder.calls, [])
        self.assertEqual(index.search("  "), [])
        with self.assertRaises(ValueError):
            index.search("问题", 0)

    def test_duplicate_id_rejected_and_truncation_counted(self):
        with self.assertRaises(ValueError):
            self.make_index([self.chunks[0]] * 2)
        long_chunk = [{"id": "long", "heading_path": "标题", "text": "文" * 600}]
        self.assertEqual(self.make_index(long_chunk)[0].truncated_chunk_count, 1)

    def test_incremental_rows_follow_passage_not_old_position_or_id(self):
        class DistinctEncoder(FakeEncoder):
            def encode(self, texts, **kwargs):
                self.calls.append(texts)
                return np.asarray([[1., 0.] if "第一" in text else [0., 1.] for text in texts], dtype=np.float32)
        original = VectorIndex(self.chunks, cache_path=self.path, encoder=DistinctEncoder())
        changed_metadata = [{**chunk, "id": "new-" + chunk["id"], "line_start": 30}
                            for chunk in self.chunks[::-1]]
        encoder = DistinctEncoder()
        reused = VectorIndex(changed_metadata, cache_path=None, encoder=encoder,
                             reuse_from=(self.chunks, self.path))
        self.assertEqual((reused.reused_count, reused.encoded_count), (2, 0))
        self.assertEqual(encoder.calls, [])
        np.testing.assert_array_equal(reused.embeddings, original.embeddings[::-1])
        incompatible = VectorIndex(changed_metadata, cache_path=None, encoder=DistinctEncoder(),
                                   revision="different-model", reuse_from=(self.chunks, self.path))
        self.assertEqual(incompatible.reuse_status, "incompatible")
        self.assertEqual(incompatible.encoded_count, 2)
