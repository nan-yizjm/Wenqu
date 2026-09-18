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
        self.assertEqual(health.json()["database_schema"], 10)
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
        self.client.patch("/api/v1/settings", json={"display_name": "技术资料", "theme": "dark"})

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
        self.assertEqual(database.schema_version(), 10)
        self.assertTrue(database.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"))
        # 备份名用的是**最后一个待执行版本**（database.py 的 pending[-1]），
        # 不是用户升级前的版本号。加新迁移时这行要跟着改。
        self.assertEqual(len(list(paths.backups.glob("workspace-before-v10-*.sqlite3"))), 1)

    def test_upgrading_a_v7_database_backfills_the_note_layer(self):
        """v7 → v8 是用户升级时真正会走的那条路，存量行必须被补成 'note'。

        这是**唯一**会碰到真实数据的迁移场景：v8 之前装过的人，升级后打开旧
        会话与旧收藏。补不成的话，旧来源缺"这条依据来自哪一层"的唯一线索，
        而新写入的来源都带着 origin——新旧混在一起比全都缺更难解释。

        已有测试只覆盖"全新库建到 v8"和"从 v2/v4 升级"，都碰不到存量来源行。
        """
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "升级数据").ensure()
        with closing(sqlite3.connect(paths.database)) as connection:
            for version in range(1, 8):
                connection.executescript(MIGRATIONS[version])
            connection.execute("PRAGMA user_version = 7")
            connection.execute("""INSERT INTO conversations(id, title, created_at, updated_at)
                VALUES ('conv_a', '旧会话', 'c', 'u')""")
            connection.execute("""INSERT INTO messages(id, conversation_id, role, content,
                    status, created_at)
                VALUES ('msg_a', 'conv_a', 'assistant', '旧回答', 'complete', 'c')""")
            connection.execute("""INSERT INTO message_sources(message_id, label, position,
                    chunk_id, document_id, version_id, title, media_type, heading_path,
                    locator_json, preview, score_json)
                VALUES ('msg_a', 'S1', 1, 'chk_a', 'doc_a', 'ver_a', '旧笔记', 'markdown',
                        '标题', '{"kind":"markdown","start_line":1,"end_line":2}', '旧片段', NULL)""")
            connection.execute("""INSERT INTO favorites(id, message_id, title, question, answer,
                    created_at, updated_at)
                VALUES ('fav_a', 'msg_a', '旧收藏', '旧问题', '旧回答', 'c', 'u')""")
            connection.execute("""INSERT INTO favorite_sources(favorite_id, label, position,
                    chunk_id, document_id, version_id, title, media_type, heading_path,
                    locator_json, preview, score_json)
                VALUES ('fav_a', 'S1', 1, 'chk_a', 'doc_a', 'ver_a', '旧笔记', 'markdown',
                        '标题', '{"kind":"markdown","start_line":1,"end_line":2}', '旧片段', NULL)""")
            connection.commit()

        database = Database(paths)
        self.assertIsNone(database.migration_error)
        self.assertEqual(database.schema_version(), 10)
        self.assertEqual(database.fetchone(
            "SELECT origin FROM message_sources WHERE label='S1'")["origin"], "note")
        self.assertEqual(database.fetchone(
            "SELECT origin FROM favorite_sources WHERE label='S1'")["origin"], "note")
        # v10 加的是"这轮有没有出过网"和"证据有没有被窗口裁掉"。**旧行必须是
        # NULL（没有记录）**：那时候这些字段还不存在，补任何一句结论都是替历史
        # 作证——'off' 会说"当时没联网"，非空 note 会说"当时裁过证据"，而我们
        # 对当时的实情一无所知。
        self.assertIsNone(database.fetchone(
            "SELECT web_state_json FROM messages WHERE id='msg_a'")["web_state_json"])
        self.assertIsNone(database.fetchone(
            "SELECT evidence_note FROM messages WHERE id='msg_a'")["evidence_note"])
        self.assertEqual(len(list(paths.backups.glob("workspace-before-v10-*.sqlite3"))), 1)

    def test_a_v9_database_upgrades_without_losing_its_turns(self):
        """v9 → v10 是**装了 0.2.1 的人**升级时真正会走的那条路。

        v10 只是想给"这轮有没有出过网"补一个记录位，它不该动到任何已有东西：
        会话、消息正文、来源都原样，只有那个新列是 NULL——**没有记录**，不是 "off"。
        """
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "补记录数据").ensure()
        with closing(sqlite3.connect(paths.database)) as connection:
            for version in range(1, 10):
                connection.executescript(MIGRATIONS[version])
            connection.execute("PRAGMA user_version = 9")
            connection.execute("""INSERT INTO conversations(id, title, created_at, updated_at)
                VALUES ('conv_b', '旧会话', 'c', 'u')""")
            connection.execute("""INSERT INTO messages(id, conversation_id, role, content,
                    status, created_at)
                VALUES ('msg_b', 'conv_b', 'assistant', '旧回答正文', 'complete', 'c')""")
            connection.execute("""INSERT INTO artifacts(id, kind, title, topic, status, created_at)
                VALUES ('art_b', 'guide', '旧产出', '旧主题', 'failed', 'c')""")
            connection.commit()

        database = Database(paths)
        self.assertIsNone(database.migration_error)
        self.assertEqual(database.schema_version(), 10)
        self.assertEqual(database.fetchone(
            "SELECT content FROM messages WHERE id='msg_b'")["content"], "旧回答正文")
        self.assertIsNone(database.fetchone(
            "SELECT web_state_json FROM messages WHERE id='msg_b'")["web_state_json"])
        self.assertIsNone(database.fetchone(
            "SELECT evidence_note FROM messages WHERE id='msg_b'")["evidence_note"])
        self.assertIsNone(database.fetchone(
            "SELECT evidence_note FROM artifacts WHERE id='art_b'")["evidence_note"])

    def test_migration_failure_starts_recovery_mode_and_restores_backup(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "恢复数据").ensure()
        # 手工停在 v4，这样 pending 会是 [5, 6, 7, 8, 9]：v5 的表重建真的跑起来，
        # 再让 v9 失败，才能验到重建与建表都被回滚。
        with closing(sqlite3.connect(paths.database)) as connection:
            for version in (1, 2, 3, 4):
                connection.executescript(MIGRATIONS[version])
            connection.execute("PRAGMA user_version = 4")
            connection.commit()
        # 造假迁移之前先把真的存下来。**不能改成 `MIGRATIONS.pop(9, None)`**：
        # MIGRATIONS 是模块级字典，pop 掉之后这个进程里所有后续测试建出来的库
        # 都停在 v8，而产品代码已按 v9 建产出物表——表现为一批八竿子打不着
        # 的模块报 "no such table: artifacts"。
        # （2026-09-18 实测：v8 那次同样的写法让 219 项里 18 项变红。）
        real_nine = MIGRATIONS[9]
        MIGRATIONS[9] = "THIS IS NOT VALID SQL;"
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
                # v5 跑完的那半截必须被回滚。留在中间状态的话，磁盘上要么
                # 多一张 documents_new，要么整张 documents 不见了。
                with closing(sqlite3.connect(paths.database)) as probe:
                    self.assertEqual(probe.execute("PRAGMA user_version").fetchone()[0], 4)
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM sqlite_master WHERE name='documents_new'").fetchone())
                    self.assertIsNotNone(probe.execute(
                        "SELECT name FROM sqlite_master WHERE name='documents'").fetchone())
                    # v6 的两条 ADD COLUMN 落在 v5 之后、v8 之前，同样必须被回滚。
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM pragma_table_info('message_sources')"
                        " WHERE name='score_json'").fetchone())
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM pragma_table_info('favorites')"
                        " WHERE name='tags_json'").fetchone())
                    # v8 的两条 ADD COLUMN 落在最后一批待执行版本里，同样必须被回滚。
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM pragma_table_info('message_sources')"
                        " WHERE name='origin'").fetchone())
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM pragma_table_info('favorite_sources')"
                        " WHERE name='origin'").fetchone())
                    # v9 是建表（不是加列），回滚后连表都不该存在。
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM sqlite_master WHERE name='artifacts'").fetchone())
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM sqlite_master WHERE name='artifact_sources'").fetchone())
                    # v10 的 ADD COLUMN 排在被弄坏的 v9 之后，本该根本没跑到；
                    # 断言它在，是为了钉住"整批迁移在同一个事务里"这件事。
                    self.assertIsNone(probe.execute(
                        "SELECT name FROM pragma_table_info('messages')"
                        " WHERE name='web_state_json'").fetchone())
                self.assertEqual(client.get("/api/v1/search", params={"q": "RAG"}).status_code,
                                 503)
                restored = client.post("/api/v1/system/recovery/restore", json={
                    "backup_name": setup["recovery_backups"][0]["name"]})
                self.assertEqual(restored.status_code, 200)
                self.assertTrue(restored.json()["restart_required"])
        finally:
            MIGRATIONS[9] = real_nine
        reopened = Database(paths)
        self.assertIsNone(reopened.migration_error)
        # 失败的那次已被回滚，重开时会拿**真的** v9 再跑一遍，再接着跑到 v10。
        self.assertEqual(reopened.schema_version(), 10)

    def test_documents_table_rebuild_keeps_rows_and_widens_media_type(self):
        """v5 重建 documents，搬数据必须一字不差。

        重建走的是 `INSERT INTO documents_new SELECT * FROM documents`，
        列顺序如果和 v2 的定义对不上，数据会静默错位——所以这里逐列核对。
        """
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        paths = ProductPaths(Path(temporary.name) / "重建数据").ensure()
        with closing(sqlite3.connect(paths.database)) as connection:
            for version in (1, 2, 3, 4):
                connection.executescript(MIGRATIONS[version])
            connection.execute("PRAGMA user_version = 4")
            connection.execute(
                """INSERT INTO libraries(id, name, kind, root_path, active, created_at, updated_at)
                   VALUES ('lib_a', '笔记', 'folder', 'D:/notes', 1, 'now', 'now')""")
            connection.execute(
                """INSERT INTO documents(id, library_id, source_kind, relative_path, display_name,
                        media_type, status, error, source_mtime_ns, source_size, checksum,
                        current_version_id, removed_at, created_at, updated_at)
                   VALUES ('doc_a', 'lib_a', 'folder', 'a.md', 'a.md', 'markdown', 'ready',
                        NULL, 123, 456, 'sum_a', 'ver_a', NULL, 'c', 'u')""")
            connection.execute(
                """INSERT INTO document_versions(id, document_id, version_number, checksum,
                        snapshot_path, extracted_path, created_at)
                   VALUES ('ver_a', 'doc_a', 1, 'sum_a', 's.md', 'e.json', 'c')""")
            connection.execute(
                """INSERT INTO document_chunks(id, document_id, version_id, position,
                        heading_path, locator_json, text)
                   VALUES ('chk_a', 'doc_a', 'ver_a', 0, '标题', '{}', '正文')""")
            connection.commit()

        database = Database(paths)
        self.assertIsNone(database.migration_error)
        self.assertEqual(database.schema_version(), 10)

        row = database.fetchone("SELECT * FROM documents WHERE id = 'doc_a'")
        self.assertEqual(row["relative_path"], "a.md")
        self.assertEqual(row["display_name"], "a.md")
        self.assertEqual(row["media_type"], "markdown")
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["source_mtime_ns"], 123)
        self.assertEqual(row["source_size"], 456)
        self.assertEqual(row["checksum"], "sum_a")
        self.assertEqual(row["current_version_id"], "ver_a")
        self.assertEqual(row["created_at"], "c")
        self.assertEqual(row["updated_at"], "u")
        self.assertEqual(row["removed_at"], None)
        # 子表没被 DROP TABLE 带走，外键也仍然指得回来。
        self.assertEqual(database.fetchone(
            "SELECT text FROM document_chunks WHERE id = 'chk_a'")["text"], "正文")

        with closing(database.connect()) as connection:
            connection.execute(
                """INSERT INTO documents(id, library_id, source_kind, relative_path, display_name,
                        media_type, status, created_at, updated_at)
                   VALUES ('doc_nb', 'lib_a', 'upload', 'n.ipynb', 'n.ipynb', 'notebook',
                        'ready', 'c', 'u')""")
            connection.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO documents(id, library_id, source_kind, relative_path,
                            display_name, media_type, status, created_at, updated_at)
                       VALUES ('doc_bad', 'lib_a', 'upload', 'x.docx', 'x.docx', 'docx',
                            'ready', 'c', 'u')""")


if __name__ == "__main__":
    unittest.main()
