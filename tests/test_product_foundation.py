from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing

from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.database import Database, MIGRATIONS
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


class ProductFoundationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.paths = ProductPaths(Path(temporary.name) / "用户数据")
        self.credentials = MemoryCredentialStore()
        self.model_manager = MemoryRetrievalModelManager()
        self.client = TestClient(create_product_app(
            self.paths, self.credentials, retrieval_model_manager=self.model_manager),
                                 base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_starts_without_materials_model_or_index(self):
        health = self.client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["runtime"]["status"], "not_configured")
        self.assertEqual(health.json()["database_schema"], 4)
        self.assertTrue(self.paths.database.is_file())
        self.assertFalse(self.client.get("/api/v1/setup").json()["steps"]["retrieval_model"])

    def test_retrieval_model_is_prepared_explicitly(self):
        started = self.client.post("/api/v1/setup/retrieval-model")
        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.json()["status"], "ready")
        setup = self.client.get("/api/v1/setup").json()
        self.assertTrue(setup["steps"]["retrieval_model"])
        self.assertEqual(setup["retrieval_model"]["device"], "cpu")

    def test_settings_and_secret_are_separate(self):
        updated = self.client.patch("/api/v1/settings", json={
            "display_name": "技术资料", "provider": "deepseek",
        })
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["settings"]["display_name"], "技术资料")
        saved = self.client.put("/api/v1/credentials/deepseek",
                                json={"api_key": "fixture-secret-key"})
        self.assertEqual(saved.status_code, 200)
        raw = self.paths.database.read_bytes()
        self.assertNotIn(b"fixture-secret-key", raw)
        self.assertTrue(self.client.get("/api/v1/settings").json()["deepseek_key_configured"])

    def test_private_host_and_unknown_settings_rejected(self):
        self.assertEqual(self.client.get("/api/v1/health", headers={"Host": "evil.test"}).status_code, 403)
        self.assertEqual(self.client.patch("/api/v1/settings", json={"unknown": True}).status_code, 422)
        self.assertEqual(self.client.patch("/api/v1/settings", json={
            "ollama_base_url": "http://remote.test:11434"}).status_code, 422)

    def test_settings_can_be_echoed_back_to_patch(self):
        """设置页把整份 settings 原样回写。

        settings 表里除用户设置外还存着 active_index_version 这类内部记账，
        一旦它们混进对外暴露的 settings，每次保存都会 422——界面上表现为
        「保存失败」，而且改哪个字段都没用。
        """
        self.client.app.state.database.set_settings({"active_index_version": "idx_fixture"})
        self.client.patch("/api/v1/settings", json={"display_name": "技术资料"})

        reported = self.client.get("/api/v1/setup").json()["settings"]
        self.assertNotIn("active_index_version", reported)

        echoed = self.client.patch("/api/v1/settings", json=reported)
        self.assertEqual(echoed.status_code, 200, echoed.text)
        self.assertEqual(echoed.json()["settings"], reported)

    def test_existing_schema_two_is_backed_up_and_migrated(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "旧版数据").ensure()
        with closing(sqlite3.connect(paths.database)) as connection:
            connection.executescript(MIGRATIONS[1]); connection.executescript(MIGRATIONS[2])
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
        database = Database(paths)
        self.assertEqual(database.schema_version(), 4)
        self.assertTrue(database.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"))
        self.assertEqual(len(list(paths.backups.glob("workspace-before-v4-*.sqlite3"))), 1)

    def test_migration_failure_starts_recovery_mode_and_restores_backup(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "恢复数据")
        healthy = Database(paths)
        self.assertIsNone(healthy.migration_error)
        MIGRATIONS[5] = "THIS IS NOT VALID SQL;"
        try:
            app = create_product_app(
                paths, MemoryCredentialStore(),
                retrieval_model_manager=MemoryRetrievalModelManager())
            with TestClient(app, base_url="http://127.0.0.1:8765") as client:
                self.assertEqual(client.get("/api/v1/health").json()["status"],
                                 "recovery_required")
                setup = client.get("/api/v1/setup").json()
                self.assertTrue(setup["recovery_required"])
                self.assertTrue(setup["recovery_backups"])
                self.assertEqual(client.get("/api/v1/search", params={"q": "RAG"}).status_code,
                                 503)
                restored = client.post("/api/v1/system/recovery/restore", json={
                    "backup_name": setup["recovery_backups"][0]["name"]})
                self.assertEqual(restored.status_code, 200)
                self.assertTrue(restored.json()["restart_required"])
        finally:
            MIGRATIONS.pop(5, None)
        reopened = Database(paths)
        self.assertIsNone(reopened.migration_error)
        self.assertEqual(reopened.schema_version(), 4)


if __name__ == "__main__":
    unittest.main()
