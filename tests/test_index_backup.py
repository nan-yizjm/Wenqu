import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from src.index_backup import create_backup, inspect_backup, restore_backup
from src.index_store import ARTIFACTS, IndexStore, atomic_json, file_hash


class IndexBackupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = self.root / "source"
        self.version = "20260914T120000000000Z-1234abcd"
        version_path = self.store / "versions" / self.version
        version_path.mkdir(parents=True)
        values = {"documents.json": b"[]", "chunks.json": b"[]", "vector_index.npz": b"fake-npz"}
        for name, raw in values.items():
            (version_path / name).write_bytes(raw)
        manifest = {
            "schema": "index-version-v1", "version": self.version,
            "vault_dir": "private-vault", "stats": {"documents": 0, "chunks": 0},
            "artifacts": {name: file_hash(version_path / name) for name in ARTIFACTS},
        }
        (version_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        atomic_json(self.store / "current.json", {
            "version": self.version,
            "manifest_sha256": file_hash(version_path / "manifest.json"),
        })
        self.backup = self.root / "backup.zip"

    def test_create_verify_and_restore_to_new_target(self):
        created = create_backup(self.store, self.backup)
        self.assertEqual(created["index_version"], self.version)
        self.assertEqual(inspect_backup(self.backup)["backup_sha256"], created["backup_sha256"])
        target = self.root / "restored"
        restored = restore_backup(self.backup, target)
        self.assertTrue(restored["verified_after_publish"])
        self.assertEqual(IndexStore(target).current()[1]["version"], self.version)
        self.assertTrue((target / "restore_receipt.json").is_file())

    def test_changed_member_is_rejected_without_publishing_target(self):
        create_backup(self.store, self.backup)
        corrupt = self.root / "corrupt.zip"
        with ZipFile(self.backup, "r") as original, ZipFile(corrupt, "w", ZIP_DEFLATED) as changed:
            for item in original.infolist():
                raw = original.read(item.filename)
                if item.filename == "index/chunks.json":
                    raw = b"[corrupted]"
                changed.writestr(item.filename, raw)
        target = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "备份成员校验失败"):
            restore_backup(corrupt, target)
        self.assertFalse(target.exists())

    def test_existing_target_and_extra_member_are_rejected(self):
        create_backup(self.store, self.backup)
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            restore_backup(self.backup, existing)
        malicious = self.root / "unexpected.zip"
        with ZipFile(self.backup, "r") as original, ZipFile(malicious, "w", ZIP_DEFLATED) as changed:
            for item in original.infolist():
                changed.writestr(item.filename, original.read(item.filename))
            changed.writestr("../escape.txt", b"bad")
        with self.assertRaisesRegex(ValueError, "非法或重复路径"):
            inspect_backup(malicious)


if __name__ == "__main__":
    unittest.main()
