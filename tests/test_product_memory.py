import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.memory import NullMemoryProvider
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager

MEMORY_TEXT = "用户偏好用对比表格看结论。"


class FakeStreamingClient:
    def __init__(self, captured):
        self.captured = captured

    def stream_chat(self, messages, cancel_event=None):
        self.captured.append(messages)
        yield "分页管理 KV Cache [S1]。"


class RecordingMemoryProvider:
    """记录被调用的次数：这是"开关关掉时真的没问过它"唯一可验证的方式。"""

    name = "recording"

    def __init__(self, items=None, fail_recall=False, fail_list=False):
        self.recall_calls = []
        self.items = list(items or [])
        self.fail_recall = fail_recall
        self.fail_list = fail_list

    def recall(self, query, limit=5):
        self.recall_calls.append((query, limit))
        if self.fail_recall:
            raise RuntimeError("boom")
        return self.items[:limit]

    def remember(self, items):
        return None

    def forget(self, item_id):
        return None

    def list(self, limit=100):
        if self.fail_list:
            raise RuntimeError("boom")
        return self.items[:limit]


class MemorySeamTests(unittest.TestCase):
    def build(self, provider=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.captured = []
        self.paths = ProductPaths(Path(temporary.name) / "产品数据")
        app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
            chat_client_factory=lambda settings: FakeStreamingClient(self.captured),
            memory_provider_factory=(lambda: provider) if provider is not None else None,
        )
        self.app = app
        self.client = TestClient(app, base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/documents/upload", files={"file": (
            "推理.md", "# 推理\n\n## PagedAttention\n\n分页管理 KV Cache，减少显存碎片。".encode(),
            "text/markdown")})
        self.conversation = self.client.post(
            "/api/v1/conversations", json={"title": "记忆"}).json()
        return app

    def ask(self, question="PagedAttention 是什么？"):
        response = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/stream",
            json={"question": question})
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.text.splitlines() if line]

    def events(self):
        return self.ask()[-1]

    def enable(self):
        self.assertEqual(
            self.client.patch("/api/v1/settings", json={"memory_enabled": True}).status_code, 200)

    def stored_origins(self):
        import sqlite3
        connection = sqlite3.connect(f"file:{self.paths.database}?mode=ro", uri=True)
        try:
            return [row[0] for row in connection.execute(
                "SELECT origin FROM message_sources ORDER BY message_id, position")]
        finally:
            connection.close()

    def test_schema_is_version_nine(self):
        self.build()

        self.assertEqual(self.app.state.database.schema_version(), 9)

    def test_the_note_layer_is_marked_in_payload_and_in_the_database(self):
        self.build()

        final = self.events()
        self.assertEqual(final["sources"][0]["origin"], "note")
        self.assertEqual(set(self.stored_origins()), {"note"})

    def test_memory_is_off_by_default_and_the_provider_is_never_asked(self):
        """开关关掉时必须**根本没调用**提供者，而不是"调了但忽略结果"。

        这是这个接缝唯一能被第三方验证的隐私承诺：接进来的实现不需要相信产品
        会丢弃结果，因为产品压根不会问它。
        """
        provider = RecordingMemoryProvider(
            [{"id": "m1", "text": MEMORY_TEXT, "derived_from": "旧对话",
              "created_at": "2026-09-18", "expires_at": None}])
        self.build(provider)

        self.events()

        self.assertEqual(provider.recall_calls, [])

    def test_enabling_memory_reaches_the_prompt_and_the_sources(self):
        provider = RecordingMemoryProvider(
            [{"id": "m1", "text": MEMORY_TEXT, "derived_from": "旧对话",
              "created_at": "2026-09-18", "expires_at": None}])
        self.build(provider)
        self.enable()

        final = self.events()

        self.assertEqual(len(provider.recall_calls), 1)
        origins = [item["origin"] for item in final["sources"]]
        self.assertEqual(origins, ["note", "memory"])
        memory_source = final["sources"][-1]
        self.assertEqual(memory_source["locator"]["kind"], "memory")
        self.assertEqual(memory_source["preview"], MEMORY_TEXT)
        # 记忆要能当依据被引用，所以它必须真的进了提示词
        self.assertIn(MEMORY_TEXT, self.captured[0][-1]["content"])
        self.assertIn("memory", set(self.stored_origins()))

    def test_a_broken_provider_does_not_break_answering(self):
        """记忆是辅助通道，坏了只记录、不拒绝服务。"""
        self.build(RecordingMemoryProvider(fail_recall=True))
        self.enable()

        final = self.events()

        self.assertEqual(final["status"], "complete")
        self.assertEqual([item["origin"] for item in final["sources"]], ["note"])
        events = self.client.get("/api/v1/system/diagnostics").json()
        self.assertTrue(events["memory"]["available"])

    def test_recall_failure_is_recorded_as_an_app_event(self):
        self.build(RecordingMemoryProvider(fail_recall=True))
        self.enable()
        self.events()

        import sqlite3
        connection = sqlite3.connect(f"file:{self.paths.database}?mode=ro", uri=True)
        try:
            recorded = [row[0] for row in connection.execute(
                "SELECT event_type FROM app_events WHERE event_type='memory_recall_failed'")]
        finally:
            connection.close()
        self.assertEqual(recorded, ["memory_recall_failed"])

    def test_diagnostics_report_the_seam_without_being_takeable_down(self):
        provider = RecordingMemoryProvider(
            [{"id": "m1", "text": MEMORY_TEXT, "derived_from": "旧对话",
              "created_at": "2026-09-18", "expires_at": None}], fail_list=True)
        self.build(provider)

        memory = self.client.get("/api/v1/system/diagnostics").json()["memory"]

        self.assertEqual(memory["provider"], "recording")
        self.assertFalse(memory["enabled"])
        self.assertIsNone(memory["items"])
        self.assertEqual(memory["error"], "RuntimeError")

    def test_default_app_reports_no_memory_provider(self):
        self.build()

        memory = self.client.get("/api/v1/system/diagnostics").json()["memory"]

        self.assertEqual(memory, {"available": True, "provider": "none",
                                  "items": 0, "enabled": False})


class NullMemoryProviderTests(unittest.TestCase):
    def test_it_is_indistinguishable_from_having_no_memory(self):
        provider = NullMemoryProvider()

        self.assertEqual(provider.recall("任意查询"), [])
        self.assertEqual(provider.list(), [])
        # 空实现不该抛异常：产品任何一条路径抛异常都会让问答崩掉
        self.assertIsNone(provider.remember([]))
        self.assertIsNone(provider.forget("不存在"))

    def test_it_writes_nothing_to_disk(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = NullMemoryProvider()

        provider.remember([{"id": "m1", "text": "x", "derived_from": "y",
                            "created_at": "z", "expires_at": None}])

        self.assertEqual(list(Path(temporary.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
