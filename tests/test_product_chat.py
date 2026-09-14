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


class FakeStreamingClient:
    def __init__(self, captured):
        self.captured = captured

    def stream_chat(self, messages, cancel_event=None):
        self.captured.append(messages)
        for part in ("PagedAttention 使用分页管理 KV Cache", "，减少显存碎片 [S1]。"):
            yield part


class ProductChatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.captured = []
        self.paths = ProductPaths(Path(temporary.name) / "产品数据")
        app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
            chat_client_factory=lambda settings: FakeStreamingClient(self.captured),
        )
        self.client = TestClient(app, base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/documents/upload", files={"file": (
            "推理.md", "# 推理\n\n## PagedAttention\n\n分页管理 KV Cache，减少显存碎片。".encode(),
            "text/markdown")})
        self.conversation = self.client.post(
            "/api/v1/conversations", json={"title": "推理讨论"}).json()

    def events(self, body):
        response = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/stream", json=body)
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.text.splitlines() if line]

    def test_stream_persists_answer_and_snapshot_sources(self):
        events = self.events({"question": "PagedAttention 是什么？"})
        self.assertEqual([item["type"] for item in events],
                         ["retrieval", "generation", "token", "token", "final"])
        final = events[-1]
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["sources"][0]["label"], "S1")
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        self.assertEqual(len(loaded["messages"]), 2)
        answer = loaded["messages"][-1]
        self.assertIn("[S1]", answer["content"])
        self.assertTrue(answer["index_version"].startswith("idx_"))
        self.assertEqual(answer["sources"][0]["locator"]["kind"], "markdown")

    def test_followup_rewrites_retrieval_but_history_is_not_evidence(self):
        self.events({"question": "PagedAttention 是什么？"})
        self.events({"question": "它解决什么问题？"})
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        last = loaded["messages"][-1]
        self.assertIn("PagedAttention 是什么", last["retrieval_query"])
        self.assertIn("当前追问", last["retrieval_query"])
        prompt = self.captured[-1]
        self.assertIn("不可引用为事实", prompt[1]["content"])
        self.assertIn("当前检索证据", prompt[-1]["content"])

    def test_static_ood_is_rejected_without_model_call(self):
        events = self.events({"question": "北京明天天气怎么样？"})
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["rejected"])
        self.assertEqual(self.captured, [])

    def test_unknown_conversation_and_retry_are_rejected_before_streaming(self):
        missing = self.client.post(
            "/api/v1/conversations/not-found/messages/stream", json={"question": "RAG 是什么"})
        self.assertEqual(missing.status_code, 404)
        retry = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/stream",
            json={"retry_message_id": "not-found"})
        self.assertEqual(retry.status_code, 404)

    def test_cancelled_generation_is_saved_as_stopped_and_retryable(self):
        cancel = threading.Event()
        stream = self.client.app.state.chat.stream(
            self.conversation["id"], "PagedAttention 是什么？", cancel_event=cancel)
        events = [next(stream), next(stream)]
        message_id = events[-1]["message_id"]
        stopped = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/{message_id}/stop")
        self.assertTrue(stopped.json()["stopping"])
        events.extend(stream)
        self.assertEqual(events[-1]["type"], "stopped")
        message_id = events[-1]["message_id"]
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        stopped = next(item for item in loaded["messages"] if item["id"] == message_id)
        self.assertEqual(stopped["status"], "stopped")
        retried = self.events({"retry_message_id": message_id})
        self.assertEqual(retried[-1]["status"], "complete")

    def test_deepseek_without_credential_fails_without_switching_provider(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        app = create_product_app(
            ProductPaths(Path(temporary.name) / "隔离数据"), MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True)
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            client.patch("/api/v1/settings", json={"provider": "deepseek"})
            client.post("/api/v1/documents/upload", files={"file": (
                "RAG.md", "# RAG\n\n检索证据后生成回答。".encode(), "text/markdown")})
            conversation = client.post("/api/v1/conversations", json={}).json()
            response = client.post(
                f"/api/v1/conversations/{conversation['id']}/messages/stream",
                json={"question": "RAG 如何生成回答？"})
            events = [json.loads(line) for line in response.text.splitlines() if line]
            self.assertEqual(events[-1]["type"], "error")
            self.assertEqual(events[-1]["error"], "deepseek_not_configured")
            loaded = client.get(f"/api/v1/conversations/{conversation['id']}").json()
            self.assertEqual(loaded["messages"][-1]["provider"], "deepseek")
            self.assertEqual(loaded["messages"][-1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
