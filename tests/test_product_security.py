"""真实 API 边界回归：恶意输入必须被拒绝且不能改变产品数据。"""
from hashlib import sha256
import io
import json
import unittest
import zipfile

from tests.product_harness import build_harness


class ProductSecurityTests(unittest.TestCase):
    def setUp(self):
        self.client, _, _, self.app = build_harness(self)

    def test_ollama_url_uses_hostname_not_prefix(self):
        for url in ("http://localhost.evil.example", "http://127.0.0.1@evil.example",
                    "http://localhost:11434@evil.example", "http://127.0.0.10:11434",
                    "https://localhost", "http://localhost:0", "http://localhost:65536",
                    "http://localhost/a", "http://localhost?x=1", "http://localhost#x",
                    "http://localhost\\evil", "http://localhost\n"):
            with self.subTest(url=url):
                self.assertEqual(self.client.patch("/api/v1/settings", json={
                    "ollama_base_url": url}).status_code, 422)
        for url in ("http://127.0.0.1:11434", "http://localhost:11435/"):
            self.assertEqual(self.client.patch("/api/v1/settings", json={
                "ollama_base_url": url}).status_code, 200)

    def test_requests_with_origin_must_be_same_origin(self):
        for origin in ("http://127.0.0.1:9999", "http://localhost:8765",
                       "http://127.0.0.1:8765@evil.example", "https://127.0.0.1:8765",
                       "null", "http://127.0.0.1:8765/path", "http://127.0.0.1:bad"):
            with self.subTest(origin=origin):
                self.assertEqual(self.client.post("/api/v1/conversations", json={"title": "x"},
                    headers={"Origin": origin}).status_code, 403)
        self.assertEqual(self.client.get("/api/v1/health", headers={
            "Origin": "http://127.0.0.1:8765"}).status_code, 200)
        self.assertEqual(self.client.get("/api/v1/conversations").json()["conversations"], [])

    def _archive(self, files, manifest=None):
        if manifest is None:
            manifest = {"format": 1, "files": {
                name: {"size": len(raw), "sha256": sha256(raw).hexdigest()}
                for name, raw in files.items()}}
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, raw in files.items():
                archive.writestr(name, raw)
        return output.getvalue()

    def test_windows_unsafe_members_and_collisions_leave_data_intact(self):
        original = self.client.post("/api/v1/system/backup").content
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            database = archive.read("workspace.sqlite3")
        baseline = self.client.get("/api/v1/documents").json()
        before = {p.relative_to(self.app.state.paths.corpus): p.read_bytes()
                  for p in self.app.state.paths.corpus.rglob("*") if p.is_file()}
        bad_names = ["corpus/..\\..\\escape.txt", "corpus/C:\\escape.txt", "corpus/../escape",
                     "corpus/CON.txt", "corpus/NUL", "corpus/a:stream", "corpus/a.",
                     "corpus/a ", "corpus//a", "corpus/./a", "workspace.sqlite3/child"]
        samples = [{"workspace.sqlite3": database, name: b"bad"} for name in bad_names]
        samples.append({"workspace.sqlite3": database, "corpus/A.txt": b"a", "corpus/a.txt": b"b"})
        samples.append({"workspace.sqlite3": database, "corpus/a": b"a", "corpus/a/b": b"b"})
        for files in samples:
            with self.subTest(names=list(files)):
                result = self.client.post("/api/v1/system/restore", files={"file": (
                    "malicious.zip", self._archive(files), "application/zip")})
                self.assertEqual(result.status_code, 422)
                self.assertEqual(self.client.get("/api/v1/documents").json(), baseline)
        self.assertEqual(before, {p.relative_to(self.app.state.paths.corpus): p.read_bytes()
                                  for p in self.app.state.paths.corpus.rglob("*") if p.is_file()})
        self.assertFalse(self.app.state.restore_pending)
        self.assertEqual(list(self.app.state.paths.runtime.glob("restore-*")), [])

    def test_malformed_manifest_and_database_are_422(self):
        samples = [self._archive({}, value) for value in ([], None, "text", {"format": True, "files": {}})]
        samples.append(self._archive({"workspace.sqlite3": b"not sqlite"}))
        for metadata in ([], None, {"size": "5", "sha256": "bad"}, {"size": -1, "sha256": "a" * 64}):
            samples.append(self._archive({"workspace.sqlite3": b"x"},
                {"format": 1, "files": {"workspace.sqlite3": metadata}}))
        # None is intentionally JSON null rather than the helper's default manifest.
        samples.append(self._archive({}, "null"))
        for data in samples:
            self.assertEqual(self.client.post("/api/v1/system/restore", files={"file": (
                "bad.zip", data, "application/zip")}).status_code, 422)
        self.assertFalse(self.app.state.restore_pending)

    def test_upload_rejects_unsupported_file_without_importing_it(self):
        baseline = self.client.get("/api/v1/documents").json()
        response = self.client.post("/api/v1/documents/upload", files={"file": (
            "payload.exe", b"MZ", "application/octet-stream")})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get("/api/v1/documents").json(), baseline)
