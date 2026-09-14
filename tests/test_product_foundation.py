from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.database import Database
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
        self.assertEqual(health.json()["database_schema"], 1)
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


if __name__ == "__main__":
    unittest.main()
