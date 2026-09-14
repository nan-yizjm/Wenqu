import json
from pathlib import Path
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


class AnswerClient:
    def stream_chat(self, messages, cancel_event=None):
        yield "RAG 先检索证据，再注入提示词并生成回答 [S1]。"


class ProductOrganizeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.paths = ProductPaths(Path(temporary.name) / "产品数据")
        app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
            chat_client_factory=lambda _settings: AnswerClient())
        self.client = TestClient(app, base_url="http://127.0.0.1:8765")
        self.client.__enter__(); self.addCleanup(self.client.__exit__, None, None, None)
        uploaded = self.client.post("/api/v1/documents/upload", files={"file": (
            "RAG基础.md", "# RAG\n\n## 在线流程\n\n用户提问后，检索证据、注入提示词、生成回答。".encode(),
            "text/markdown")}).json()
        self.document_id = uploaded["document_id"]
        conversation = self.client.post("/api/v1/conversations", json={}).json()
        response = self.client.post(
            f"/api/v1/conversations/{conversation['id']}/messages/stream",
            json={"question": "RAG 在线流程是什么？"})
        events = [json.loads(line) for line in response.text.splitlines() if line]
        self.message_id = events[-1]["message_id"]

    def test_favorite_copies_answer_sources_and_exports_utf8_markdown(self):
        created = self.client.post("/api/v1/favorites", json={"message_id": self.message_id})
        self.assertEqual(created.status_code, 200)
        favorite = created.json()
        self.assertEqual(favorite["sources"][0]["locator"]["start_line"], 5)
        # 重复收藏幂等，不制造两条成果。
        repeated = self.client.post("/api/v1/favorites", json={"message_id": self.message_id}).json()
        self.assertEqual(repeated["id"], favorite["id"])
        updated = self.client.patch(f"/api/v1/favorites/{favorite['id']}", json={
            "title": "RAG：问答/流程", "note": "用于复习和核对。"}).json()
        self.assertEqual(updated["note"], "用于复习和核对。")
        exported = self.client.get(f"/api/v1/favorites/{favorite['id']}/export")
        self.assertEqual(exported.status_code, 200)
        text = exported.content.decode("utf-8")
        self.assertIn("# RAG：问答/流程", text)
        self.assertIn("## 回答", text)
        self.assertIn("第 5–5 行", text)
        files = list(self.paths.exports.glob("*.md"))
        self.assertEqual(len(files), 1)
        self.assertNotIn("/", files[0].name)

    def test_feedback_is_local_upsert_and_favorite_can_be_deleted(self):
        first = self.client.post("/api/v1/feedback", json={
            "message_id": self.message_id, "kind": "missing", "note": "少了离线阶段"})
        self.assertTrue(first.json()["stored_locally"])
        self.client.post("/api/v1/feedback", json={
            "message_id": self.message_id, "kind": "helpful"})
        row = self.client.app.state.database.fetchone(
            "SELECT kind, note FROM answer_feedback WHERE message_id=?", (self.message_id,))
        self.assertEqual(row["kind"], "helpful")
        self.assertEqual(row["note"], "")
        self.assertEqual(self.client.post("/api/v1/feedback", json={
            "message_id": self.message_id, "kind": "unknown"}).status_code, 422)
        favorite = self.client.post("/api/v1/favorites", json={"message_id": self.message_id}).json()
        self.assertEqual(self.client.delete(f"/api/v1/favorites/{favorite['id']}").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/favorites").json()["favorites"], [])

    def test_stopped_answer_cannot_be_favorited(self):
        conversation = self.client.post("/api/v1/conversations", json={}).json()
        cancel = threading.Event(); cancel.set()
        events = list(self.client.app.state.chat.stream(
            conversation["id"], "RAG 在线流程是什么？", cancel_event=cancel))
        response = self.client.post("/api/v1/favorites", json={
            "message_id": events[-1]["message_id"]})
        self.assertEqual(response.status_code, 409)
        self.assertIn("只有完整回答", response.json()["message"])


if __name__ == "__main__":
    unittest.main()
