from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.index_store import IndexStore, file_hash, writer_lock
from src.settings import Settings, load_settings
from src.vector_retrieve import VectorIndex
from tests.test_token_chunking import CharacterTokenizer
from tests.test_vector_index import FakeEncoder


class IndexStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.vault = self.root / "notes"
        self.vault.mkdir()
        self.settings = Settings(self.vault, self.root / "store", device="cpu")
        self.store = IndexStore(self.settings.store_dir)
        self.note = self.vault / "a.md"
        self.note.write_text("# A\n## First\n第一段内容。\n## Second\n第二段内容。", encoding="utf-8")

    def build(self, **kwargs):
        return self.store.build(self.settings, tokenizer=CharacterTokenizer(), encoder=FakeEncoder(), **kwargs)

    def test_noop_avoids_loading_encoder_and_tokenizer(self):
        first = self.build()
        with patch("src.index_store.VectorIndex", side_effect=AssertionError("不应加载模型")):
            result = self.store.build(self.settings)
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["version"], first["version"])
        self.assertEqual(len(self.store.list_versions()), 1)

    def test_edit_only_encodes_changed_passage_and_preserves_old(self):
        first = self.build()
        old_path, _ = self.store.current()
        original_hash = file_hash(old_path / "chunks.json")
        self.note.write_text("# A\n## First\n修改第一段。\n\n## Second\n第二段内容。", encoding="utf-8")
        second = self.build()
        self.assertEqual(second["modified_documents"], 1)
        self.assertEqual(second["encoded_chunks"], 1)
        self.assertEqual(second["reused_chunks"], 1)
        self.assertEqual(file_hash(old_path / "chunks.json"), original_hash)
        self.assertNotEqual(first["version"], second["version"])
        self.store.activate(first["version"])
        self.assertEqual(self.store.current()[1]["version"], first["version"])
        self.store.activate(second["version"])

    def test_add_delete_and_empty_safety(self):
        first = self.build()
        self.note.unlink()  # 仅测试创建的临时笔记，不接触真实 Vault。
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(self.store.current()[1]["version"], first["version"])
        (self.vault / "new.md").write_text("# New\n新笔记内容。", encoding="utf-8")
        second = self.build()
        self.assertEqual((second["added_documents"], second["deleted_documents"]), (1, 1))
        path, _ = self.store.current()
        self.assertEqual({c["source_file"] for c in json.loads((path / "chunks.json").read_text(encoding="utf-8"))}, {"new.md"})
        (self.vault / "new.md").unlink()
        self.assertEqual(self.build(allow_empty=True)["chunks"], 0)

    def test_encoder_failure_keeps_current(self):
        first = self.build()
        self.note.write_text("# A\n变化后的内容。", encoding="utf-8")
        with patch("src.index_store.VectorIndex", side_effect=RuntimeError("模拟 GPU 错误")):
            with self.assertRaises(RuntimeError):
                self.build()
        self.assertEqual(self.store.current()[1]["version"], first["version"])
        self.assertFalse(list(self.store.root.glob("building-*")))

    def test_corruption_path_traversal_and_writer_lock(self):
        first = self.build()
        with self.assertRaises(ValueError):
            self.store.activate("../../notes")
        with writer_lock(self.store.root):
            with self.assertRaises(RuntimeError):
                self.build()
        path, _ = self.store.current()
        (path / "chunks.json").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            self.store.activate(first["version"])

    def test_readonly_cache_does_not_repair_version(self):
        self.build()
        path, _ = self.store.current()
        chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        before = file_hash(path / "vector_index.npz")
        with self.assertRaises(ValueError):
            VectorIndex(chunks, cache_path=path / "vector_index.npz", encoder=FakeEncoder(),
                        max_length=256, read_only=True)
        self.assertEqual(file_hash(path / "vector_index.npz"), before)

    def test_source_changes_during_build_are_not_published(self):
        first = self.build()
        self.note.write_text("# A\n准备编码的内容。", encoding="utf-8")
        real_vector = VectorIndex
        def changing_encoder(*args, **kwargs):
            self.note.write_text("# A\n编码期间再次修改。", encoding="utf-8")
            return real_vector(*args, **kwargs)
        with patch("src.index_store.VectorIndex", side_effect=changing_encoder):
            with self.assertRaises(RuntimeError):
                self.build()
        self.assertEqual(self.store.current()[1]["version"], first["version"])

    def test_config_paths_validation_and_unknown_keys(self):
        config = self.root / "rag.toml"
        config.write_text("vault_dir='notes'\nstore_dir='store'\n", encoding="utf-8")
        self.assertEqual(load_settings(config).vault_dir, self.vault.resolve())
        config.write_text("vault_dir='notes'\nstore_dir='store'\napi_key='not-a-real-key'", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_settings(config)
        for bad in ({"top_k": 30}, {"rerank": "false"}, {"host": "0.0.0.0"},
                    {"store_dir": self.vault / "store"}, {"request_timeout": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(self.settings, **bad)
