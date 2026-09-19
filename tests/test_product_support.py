import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile

from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


class AnswerClient:
    def stream_chat(self, messages, cancel_event=None):
        yield "资料说明 RAG 会检索后生成 [S1]。"


def app_for(paths):
    return create_product_app(
        paths, MemoryCredentialStore(),
        retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
        chat_client_factory=lambda _settings: AnswerClient())


def seed(client):
    client.post("/api/v1/documents/upload", files={"file": (
        "私人标题.md", "# RAG\n\n检索后生成。".encode(), "text/markdown")})
    conversation = client.post("/api/v1/conversations", json={"title": "私人会话"}).json()
    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages/stream",
        json={"question": "我的私人问题：RAG 是什么？"})
    events = [json.loads(line) for line in response.text.splitlines() if line]
    favorite = client.post("/api/v1/favorites", json={
        "message_id": events[-1]["message_id"]}).json()
    return favorite


class ProductSupportTests(unittest.TestCase):
    def test_full_backup_restore_preserves_data_and_creates_safety_backup(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "产品数据")
        with TestClient(app_for(paths), base_url="http://127.0.0.1:8765") as client:
            favorite = seed(client)
            backup = client.post("/api/v1/system/backup")
            self.assertEqual(backup.status_code, 200)
            self.assertTrue(backup.content.startswith(b"PK"))
            client.delete(f"/api/v1/favorites/{favorite['id']}")
            client.post("/api/v1/documents/upload", files={"file": (
                "later.md", b"# Later\n\nnot in backup", "text/markdown")})
            restored = client.post("/api/v1/system/restore", files={"file": (
                "backup.zip", backup.content, "application/zip")})
            self.assertEqual(restored.status_code, 200)
            self.assertTrue(restored.json()["restart_required"])
            self.assertEqual(client.get("/api/v1/favorites").status_code, 409)
        with TestClient(app_for(paths), base_url="http://127.0.0.1:8765") as reopened:
            favorites = reopened.get("/api/v1/favorites").json()["favorites"]
            self.assertEqual([item["id"] for item in favorites], [favorite["id"]])
            documents = reopened.get("/api/v1/documents").json()["documents"]
            self.assertEqual([item["display_name"] for item in documents], ["私人标题.md"])
        self.assertTrue(list(paths.backups.glob("before-restore-*.zip")))

    def test_tampered_and_path_traversal_backups_are_rejected(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "产品数据")
        with TestClient(app_for(paths), base_url="http://127.0.0.1:8765") as client:
            seed(client)
            original = client.post("/api/v1/system/backup").content
            source = zipfile.ZipFile(io.BytesIO(original))
            manifest = json.loads(source.read("manifest.json"))
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as archive:
                for info in source.infolist():
                    raw = source.read(info.filename)
                    if info.filename == "workspace.sqlite3": raw += b"tamper"
                    archive.writestr(info.filename, raw)
            source.close()
            response = client.post("/api/v1/system/restore", files={"file": (
                "tampered.zip", data.getvalue(), "application/zip")})
            self.assertEqual(response.status_code, 422)

            traversal = io.BytesIO()
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("manifest.json", json.dumps({"format": 1, "files": {}}))
                archive.writestr("../evil.txt", b"bad")
            response = client.post("/api/v1/system/restore", files={"file": (
                "bad.zip", traversal.getvalue(), "application/zip")})
            self.assertEqual(response.status_code, 422)

            duplicate = io.BytesIO()
            with zipfile.ZipFile(duplicate, "w") as archive:
                archive.writestr("manifest.json", json.dumps({"format": 1, "files": {}}))
                archive.writestr("manifest.json", json.dumps({"format": 1, "files": {}}))
            response = client.post("/api/v1/system/restore", files={"file": (
                "duplicate.zip", duplicate.getvalue(), "application/zip")})
            self.assertEqual(response.status_code, 422)

    def test_diagnostic_export_excludes_content_paths_and_secrets(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "很私人的路径")
        credentials = MemoryCredentialStore(); credentials.set_deepseek("fixture-secret-key")
        app = create_product_app(
            paths, credentials, retrieval_model_manager=MemoryRetrievalModelManager(),
            material_run_inline=True, chat_client_factory=lambda _settings: AnswerClient())
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            seed(client)
            response = client.get("/api/v1/system/diagnostics/export")
            report = response.content.decode("utf-8")
            value = json.loads(report)
            self.assertTrue(value["deepseek_key_configured"])
            self.assertNotIn("fixture-secret-key", report)
            self.assertNotIn("私人标题", report)
            self.assertNotIn("我的私人问题", report)
            self.assertNotIn(str(paths.root), report)


class ShutdownTests(unittest.TestCase):
    """退出这件事只有两种诚实回答：**我让启动器停了**，或者**我停不了**。

    产品是个常驻的本地服务（`console=False`、没有托盘），所以"退出"只能由启动器做到。
    一个没有接启动器的实例如果回答"正在退出"，界面就会显示"已退出"而进程还活着——
    这正是这个项目在联网那一块守过的同一条线：**做不到就说做不到，别报成功**。
    """

    def build(self, **kwargs):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "产品数据")
        return create_product_app(
            paths, MemoryCredentialStore(), retrieval_model_manager=MemoryRetrievalModelManager(),
            material_run_inline=True, chat_client_factory=lambda _settings: AnswerClient(), **kwargs)

    def test_an_instance_without_a_launcher_says_it_cannot_quit(self):
        with TestClient(self.build(), base_url="http://127.0.0.1:8765") as client:
            response = client.post("/api/v1/system/shutdown")

            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"], "shutdown_unavailable")
            # 说清楚"为什么"和"那怎么办"：光说失败用户不知道下一步该做什么。
            self.assertIn("启动器", response.json()["message"])

    def test_a_launcher_backed_instance_asks_to_exit_and_states_what_survives(self):
        calls = []
        with TestClient(self.build(shutdown_callback=lambda: calls.append(1)),
                        base_url="http://127.0.0.1:8765") as client:
            response = client.post("/api/v1/system/shutdown")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "shutting_down")
            self.assertEqual(len(calls), 1)
            # 退出不是删数据。这句话必须由接口自己说出来，不能只靠前端文案。
            self.assertIn("资料", response.json()["message"])
            # 连点两次不该出问题：界面上的按钮可能被点两下。
            self.assertEqual(client.post("/api/v1/system/shutdown").status_code, 200)
            self.assertEqual(len(calls), 2)

    def test_shutdown_counts_what_it_must_wait_for_before_saying_done(self):
        """uvicorn 收到退出标志后会等在跑的流结束——这期间界面显示"已退出"就是
        假话。接口必须数清楚自己在等什么，让界面能如实说"正在等待"。"""
        with TestClient(self.build(shutdown_callback=lambda: None),
                        base_url="http://127.0.0.1:8765") as client:
            self.assertEqual(
                client.post("/api/v1/system/shutdown").json()["waiting"],
                {"imports": 0, "answers": 0, "artifacts": 0})

            # 三类在跑的任务各塞一个：一条回答、一份产出、一个导入作业。
            client.app.state.chat._active["msg_x"] = threading.Event()
            client.app.state.studio._active["art_x"] = threading.Event()
            with client.app.state.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO import_jobs(id, job_type, status, payload_json, created_at)"
                    " VALUES ('job_x', 'file', 'running', '{}', '2026-09-19T00:00:00')")

            self.assertEqual(
                client.post("/api/v1/system/shutdown").json()["waiting"],
                {"imports": 1, "answers": 1, "artifacts": 1})


if __name__ == "__main__":
    unittest.main()
