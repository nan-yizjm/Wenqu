import json
from pathlib import Path
import shutil
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
        self.conversation_id = conversation["id"]
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

    def test_favorite_keeps_the_retrieval_evidence_of_its_sources(self):
        """收藏是"以后回看的证据"，当时的名次和命中词必须一起留下来。

        它们只存在 `score_json` 里；丢了的话，收藏页的来源列表会比问答页少一半
        信息，而用户恰恰是为了核对才收藏的。
        """
        answer = self.client.get(
            f"/api/v1/conversations/{self.conversation_id}").json()["messages"][-1]
        created = self.client.post("/api/v1/favorites", json={"message_id": self.message_id}).json()
        for source, original in zip(created["sources"], answer["sources"], strict=True):
            self.assertEqual(source["score"], original["score"])
            self.assertEqual(source["matched_tokens"], original["matched_tokens"])
            self.assertEqual(source["channels"], original["channels"])
        # 收藏和实时消息同样只出界面认识的键，不该漏出表结构。
        self.assertEqual(set(created["sources"][0]), set(answer["sources"][0]))

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

    def _ask(self, question):
        response = self.client.post(
            f"/api/v1/conversations/{self.conversation_id}/messages/stream",
            json={"question": question})
        events = [json.loads(line) for line in response.text.splitlines() if line]
        return events[-1]["message_id"]

    def _favorite(self, message_id=None):
        return self.client.post(
            "/api/v1/favorites", json={"message_id": message_id or self.message_id}).json()

    def test_favorite_tags_are_normalized_and_can_be_cleared(self):
        """标签是用户自己造的检索维度，空白和重复必须在这里收干净。

        否则界面上的标签墙会出现两个看起来一样的标签，点哪个都只筛出一半。
        """
        favorite = self._favorite()
        self.assertEqual(favorite["tags"], [])
        updated = self.client.patch(f"/api/v1/favorites/{favorite['id']}", json={
            "tags": ["  检索  ", "检索", "在线\n流程", "   "]}).json()
        self.assertEqual(updated["tags"], ["检索", "在线 流程"])
        self.assertEqual(
            self.client.get(f"/api/v1/favorites/{favorite['id']}").json()["tags"],
            ["检索", "在线 流程"])
        self.assertEqual(self.client.patch(f"/api/v1/favorites/{favorite['id']}", json={
            "tags": []}).json()["tags"], [])
        self.assertEqual(self.client.patch(f"/api/v1/favorites/{favorite['id']}", json={
            "tags": [f"标签{i}" for i in range(9)]}).status_code, 422)

    def test_favorite_list_filters_by_tag_kind_age_and_library(self):
        favorite = self._favorite()
        self.client.patch(f"/api/v1/favorites/{favorite['id']}", json={"tags": ["检索"]})
        self.client.post("/api/v1/feedback", json={
            "message_id": self.message_id, "kind": "missing"})

        listing = self.client.get("/api/v1/favorites").json()
        self.assertEqual(listing["total"], 1)
        self.assertEqual([item["id"] for item in listing["favorites"]], [favorite["id"]])
        self.assertEqual(listing["tags"], ["检索"])
        self.assertEqual(listing["favorites"][0]["feedback_kind"], "missing")
        self.assertEqual([item["name"] for item in listing["libraries"]], ["上传文件"])

        def filtered(**params):
            return self.client.get("/api/v1/favorites", params=params).json()

        self.assertEqual(len(filtered(tag="检索")["favorites"]), 1)
        self.assertEqual(filtered(tag="没这个标签")["favorites"], [])
        self.assertEqual(len(filtered(feedback="missing")["favorites"]), 1)
        self.assertEqual(filtered(feedback="helpful")["favorites"], [])
        self.assertEqual(len(filtered(days="7d")["favorites"]), 1)
        with self.client.app.state.database.transaction() as connection:
            connection.execute("UPDATE favorites SET updated_at=? WHERE id=?",
                               ("2020-01-01T00:00:00+00:00", favorite["id"]))
        self.assertEqual(filtered(days="7d")["favorites"], [])
        self.assertEqual(filtered(days="30d")["favorites"], [])
        # 筛掉之后选项仍要来自全部收藏，否则用户没法再切回来。
        self.assertEqual(filtered(days="7d")["tags"], ["检索"])
        self.assertEqual(len(filtered()["favorites"]), 1)
        self.assertEqual(self.client.get(
            "/api/v1/favorites", params={"days": "昨天"}).status_code, 422)
        self.assertEqual(self.client.get(
            "/api/v1/favorites", params={"feedback": "unknown"}).status_code, 422)

        root = Path(tempfile.mkdtemp()) / "另一库"
        root.mkdir(); self.addCleanup(shutil.rmtree, root.parent, True)
        (root / "另一库.md").write_text("# 另一库\n\n另一个资料库里的内容。", encoding="utf-8")
        self.client.post("/api/v1/libraries/folders", json={"path": str(root)})
        second = self._favorite(self._ask("另一库讲的是什么？"))
        options = self.client.get("/api/v1/favorites").json()["libraries"]
        self.assertEqual(sorted(item["name"] for item in options), ["上传文件", "另一库"])
        other = next(item["id"] for item in options if item["name"] == "另一库")
        matched = filtered(library=other)["favorites"]
        self.assertIn(second["id"], [item["id"] for item in matched])
        self.assertEqual(filtered(library="lib_不存在")["favorites"], [])

    def test_collection_export_joins_favorites_into_one_dated_file(self):
        """专题导出要能一次拿走整批结论，且文件名在 Windows 上真的能落盘。"""
        first = self._favorite()
        self.client.patch(f"/api/v1/favorites/{first['id']}", json={
            "title": "RAG：在线/流程", "note": "留着复习。"})
        second = self._favorite(self._ask("RAG 在线流程是什么？"))
        self.client.get(f"/api/v1/favorites/{first['id']}/export")
        collected = self.client.get("/api/v1/favorites/collection/export", params={
            "ids": f"{first['id']},{second['id']}", "title": "RAG 专题：总览"})
        self.assertEqual(collected.status_code, 200)
        text = collected.content.decode("utf-8")
        self.assertIn("# RAG 专题：总览", text)
        self.assertIn("> 收录 2 条收藏", text)
        self.assertIn("## 目录", text)
        self.assertIn("## 1. RAG：在线/流程", text)
        self.assertIn("## 2. RAG 在线流程是什么？", text)
        self.assertEqual(text.count("## 回答"), 2)
        self.assertIn("第 5–5 行", text)
        files = list(self.paths.exports.glob("*.md"))
        self.assertEqual(len(files), 2)
        collection = next(path for path in files if "专题" in path.name)
        self.assertNotIn("/", collection.name)
        self.assertRegex(collection.name, r"-\d{4}-\d{2}-\d{2}\.md$")
        self.assertEqual(self.client.get(
            "/api/v1/favorites/collection/export", params={"ids": ""}).status_code, 409)
        self.assertEqual(self.client.get("/api/v1/favorites/collection/export", params={
            "ids": "fav_不存在"}).status_code, 404)

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
