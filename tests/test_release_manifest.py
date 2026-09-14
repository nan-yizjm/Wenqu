import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.index_store import ARTIFACTS, atomic_json, file_hash
from src.release_manifest import _run, assert_public, hash_inventory, public_index_summary


class ReleaseManifestTests(unittest.TestCase):
    def test_hash_inventory_is_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            first = hash_inventory(["b.txt", "a.txt"], root)
            second = hash_inventory(["a.txt", "b.txt"], root)
            self.assertEqual(first, second)

    def test_public_index_summary_omits_private_paths_and_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            version = "20260914T120000000000Z-1234abcd"
            path = store / "versions" / version
            path.mkdir(parents=True)
            for name in ARTIFACTS:
                (path / name).write_bytes(b"private content " + name.encode())
            manifest = {
                "schema": "index-version-v1", "version": version,
                "created_at": "2026-09-14T12:00:00+00:00",
                "vault_dir": "C:/Users/person/private-vault",
                "files": {"secret-note.md": "abc"},
                "stats": {"documents": 1, "chunks": 2}, "pipeline": {},
                "artifacts": {name: file_hash(path / name) for name in ARTIFACTS},
            }
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            atomic_json(store / "current.json", {
                "version": version, "manifest_sha256": file_hash(path / "manifest.json")})
            summary = public_index_summary(store)
            raw = json.dumps(summary, ensure_ascii=False)
            self.assertNotIn("private-vault", raw)
            self.assertNotIn("secret-note.md", raw)
            self.assertNotIn("vault_dir", raw)
            self.assertEqual(summary["stats"]["chunks"], 2)

    def test_public_guard_rejects_private_markers(self):
        with self.assertRaises(ValueError):
            assert_public({"source_file": "secret.md"})

    def test_external_command_timeout_becomes_data_not_hang(self):
        import subprocess
        with patch("src.release_manifest.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(["docker"], 2)):
            code, stdout, stderr = _run(["docker", "version"], timeout=2)
        self.assertEqual((code, stdout), (124, ""))
        self.assertIn("timed out", stderr)


if __name__ == "__main__":
    unittest.main()
