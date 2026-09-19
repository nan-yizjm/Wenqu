from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


def notebook_bytes(cells: list[dict]) -> bytes:
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
                      ensure_ascii=False).encode("utf-8")


def text_pdf(text: str) -> bytes:
    """构造一页标准 PDF，避免测试依赖额外的 PDF 生成库。"""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(output)


class ProductMaterialTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = ProductPaths(self.root / "产品数据")
        app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(),
            material_run_inline=True,
        )
        self.client = TestClient(app, base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_folder_import_update_search_and_historical_source(self):
        folder = self.root / "中文资料"
        folder.mkdir()
        note = folder / "推理.md"
        note.write_text("# 推理服务\n\n## PagedAttention\n\n分页管理 KV Cache，减少显存碎片。", encoding="utf-8")
        connected = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)})
        self.assertEqual(connected.status_code, 200)
        documents = self.client.get("/api/v1/documents").json()["documents"]
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["status"], "ready")
        first_version = documents[0]["current_version_id"]

        result = self.client.get("/api/v1/search", params={"q": "PagedAttention 显存碎片"}).json()
        self.assertEqual(len(result["results"]), 1)
        hit = result["results"][0]
        self.assertEqual(hit["locator"]["kind"], "markdown")
        self.assertEqual(hit["locator"]["start_line"], 5)
        source = self.client.get(
            f"/api/v1/documents/{hit['document_id']}/versions/{first_version}/source").json()
        self.assertIn("减少显存碎片", source["text"])

        note.write_text("# 推理服务\n\n## PagedAttention\n\n分页管理 KV Cache，并提升批处理吞吐。", encoding="utf-8")
        refreshed = self.client.post(f"/api/v1/libraries/{connected.json()['library_id']}/refresh")
        self.assertEqual(refreshed.status_code, 200)
        updated = self.client.get("/api/v1/documents").json()["documents"][0]
        self.assertNotEqual(updated["current_version_id"], first_version)
        historical = self.client.get(
            f"/api/v1/documents/{updated['id']}/versions/{first_version}/source")
        self.assertIn("减少显存碎片", historical.json()["text"])

    def test_markdown_upload_remove_and_registered_id_boundary(self):
        uploaded = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("RAG.md", "# RAG\n\n提问、检索、注入上下文、生成答案。".encode("utf-8"), "text/markdown")},
        )
        self.assertEqual(uploaded.status_code, 200)
        document_id = uploaded.json()["document_id"]
        self.assertTrue(self.client.get("/api/v1/search", params={"q": "检索 上下文"}).json()["results"])
        self.assertEqual(self.client.delete(f"/api/v1/documents/{document_id}").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/search", params={"q": "检索 上下文"}).json()["results"], [])
        self.assertEqual(self.client.get(
            "/api/v1/documents/not-a-document/versions/not-a-version/file").status_code, 404)

    def test_batch_removal_publishes_one_snapshot_and_reports_the_missing(self):
        """批量移除 = 一次事务 + **一次**快照；混进的坏 id 如实回来，不静默消失。

        逐条调单选接口每篇都会重建一次检索快照（多插一行 index_versions），
        选中 30 篇就是 30 次重建。批量接口的全部意义就在这个数字只涨 1。
        """
        ids = []
        for name in ("第一篇", "第二篇"):
            uploaded = self.client.post("/api/v1/documents/upload", files={"file": (
                f"{name}.md", f"# {name}\n\n{name}的内容足够被检索到。".encode("utf-8"),
                "text/markdown")})
            self.assertEqual(uploaded.status_code, 200)
            ids.append(uploaded.json()["document_id"])
        database = self.client.app.state.database
        versions_before = database.fetchone("SELECT COUNT(*) AS n FROM index_versions")["n"]

        result = self.client.post("/api/v1/documents/delete", json={"ids": ids + ["ghost-id"]}).json()

        self.assertEqual(result["deleted"], 2)
        self.assertEqual(result["skipped"], [
            {"id": "ghost-id", "label": None, "code": "not_found", "reason": "资料不存在。"}])
        self.assertEqual(database.fetchone("SELECT COUNT(*) AS n FROM index_versions")["n"],
                         versions_before + 1)
        self.assertEqual(self.client.get("/api/v1/search", params={"q": "足够被检索"}).json()["results"], [])

    def test_duplicate_ids_are_collapsed_not_counted_twice(self):
        """同一个 id 出现两次只算一次：第二条只会撞上 not_found，那份 skipped 是噪音。"""
        uploaded = self.client.post("/api/v1/documents/upload", files={"file": (
            "重复.md", "# 重复\n\n只删一次。".encode("utf-8"), "text/markdown")})
        document_id = uploaded.json()["document_id"]

        result = self.client.post("/api/v1/documents/delete",
                                  json={"ids": [document_id, document_id]}).json()

        self.assertEqual(result["deleted"], 1)
        self.assertEqual(result["skipped"], [])

    def test_batch_delete_rejects_empty_and_oversized_selections(self):
        """空请求回 200 会把一次没执行的删除显示成"已完成"；超量多半是调用方算错了。

        响应用的是产品统一的 422 形状（error + fields），不是 FastAPI 默认的 detail。
        """
        empty = self.client.post("/api/v1/documents/delete", json={"ids": []})
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(empty.json()["error"], "invalid_request")
        oversized = self.client.post("/api/v1/documents/delete",
                                     json={"ids": [f"id-{n}" for n in range(501)]})
        self.assertEqual(oversized.status_code, 422)
        self.assertEqual(oversized.json()["fields"][0]["type"], "too_long")

    def test_removing_folder_documents_is_reversed_by_the_next_refresh(self):
        """文件夹来源的"移除"是暂时的：文件还在磁盘上，刷新资料库它就回来了。

        这不是缺陷而是事实，界面的确认文案必须照实说。这条测试把它钉住，
        免得哪天文案先写成了"永久删除"而行为不是。
        """
        folder = self.root / "批量资料"
        folder.mkdir()
        (folder / "笔记.md").write_text("# 笔记\n\n被移除后随刷新回来的内容。", encoding="utf-8")
        library = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)}).json()
        documents = self.client.get("/api/v1/documents").json()["documents"]
        self.assertEqual([d["status"] for d in documents], ["ready"])

        result = self.client.post("/api/v1/documents/delete",
                                  json={"ids": [documents[0]["id"]]}).json()
        self.assertEqual(result["deleted"], 1)
        self.assertEqual(self.client.get("/api/v1/documents").json()["documents"], [])

        self.client.post(f"/api/v1/libraries/{library['library_id']}/refresh")
        refreshed = self.client.get("/api/v1/documents").json()["documents"]
        self.assertEqual([d["status"] for d in refreshed], ["ready"])
        self.assertEqual(refreshed[0]["display_name"], "笔记.md")

    def test_disconnecting_a_library_hides_its_documents_and_search(self):
        """断开归类 = 来源停用 + 其下资料移除 + 检索立即排除；**磁盘文件不动**。

        一次事务里做完停用与软删，快照只发布一次；刷新旧 id 回 404，
        因为 refresh 只认 `active=1` 的归类。
        """
        folder = self.root / "断开资料"
        folder.mkdir()
        (folder / "第一篇.md").write_text("# 第一篇\n\n断开后应当搜不到的内容。", encoding="utf-8")
        (folder / "第二篇.md").write_text("# 第二篇\n\n另一篇断开后搜不到。", encoding="utf-8")
        library = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)}).json()
        library_id = library["library_id"]
        self.assertTrue(self.client.get(
            "/api/v1/search", params={"q": "断开后应当搜不到"}).json()["results"])
        database = self.client.app.state.database
        versions_before = database.fetchone("SELECT COUNT(*) AS n FROM index_versions")["n"]

        result = self.client.delete(f"/api/v1/libraries/{library_id}")
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body["deleted"], 2)
        self.assertEqual(body["name"], "断开资料")

        self.assertEqual(
            [item["id"] for item in self.client.get("/api/v1/libraries").json()["libraries"]
             if item["id"] == library_id], [])
        self.assertEqual(self.client.get("/api/v1/documents").json()["documents"], [])
        self.assertEqual(self.client.get(
            "/api/v1/search", params={"q": "断开后应当搜不到"}).json()["results"], [])
        self.assertEqual(database.fetchone("SELECT COUNT(*) AS n FROM index_versions")["n"],
                         versions_before + 1)
        self.assertEqual(self.client.post(
            f"/api/v1/libraries/{library_id}/refresh").status_code, 404)
        self.assertTrue((folder / "第一篇.md").is_file())
        self.assertTrue((folder / "第二篇.md").is_file())

    def test_reconnecting_the_same_folder_after_disconnect_imports_fresh(self):
        """断开后同一路径重新连接：旧软删行不复用，新归类从头导入。"""
        folder = self.root / "重连资料"
        folder.mkdir()
        (folder / "笔记.md").write_text("# 笔记\n\n重连后重新导入的内容。", encoding="utf-8")
        first = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)}).json()
        self.assertEqual(self.client.delete(
            f"/api/v1/libraries/{first['library_id']}").status_code, 200)

        second = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)}).json()
        self.assertNotIn("already_connected", second)
        self.assertNotEqual(second["library_id"], first["library_id"])
        documents = self.client.get("/api/v1/documents").json()["documents"]
        self.assertEqual([d["status"] for d in documents], ["ready"])
        self.assertTrue(self.client.get(
            "/api/v1/search", params={"q": "重连后重新导入"}).json()["results"])

    def test_delete_library_rejects_uploads_and_unknown_ids(self):
        """上传库不是"归类"，断开没有意义（逐篇移除即可）；未知 id 保持 404。"""
        uploads = [item for item in self.client.get("/api/v1/libraries").json()["libraries"]
                   if item["kind"] == "uploads"]
        self.assertEqual(uploads, [])
        library_id = "lib_ghost"
        missing = self.client.delete(f"/api/v1/libraries/{library_id}")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"], "library_not_found")

    def test_delete_library_rejects_the_uploads_library_directly(self):
        """uploads 库真实存在（上传后），但它不是文件夹归类，断开要回 422。"""
        self.client.post("/api/v1/documents/upload", files={"file": (
            "上传.md", "# 上传\n\n上传库的内容。".encode("utf-8"), "text/markdown")})
        uploads = [item for item in self.client.get("/api/v1/libraries").json()["libraries"]
                   if item["kind"] == "uploads"]
        self.assertEqual(len(uploads), 1)
        rejected = self.client.delete(f"/api/v1/libraries/{uploads[0]['id']}")
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(rejected.json()["error"], "library_not_removable")
        self.assertEqual(len(self.client.get("/api/v1/documents").json()["documents"]), 1)

    def test_revived_document_of_a_disconnected_library_stays_out_of_search(self):
        """断开与刷新扫描并发时，扫描可能把个别文档的 removed_at 清回去；
        快照查询里的 `libraries.active=1` 是最后一道闸：来源已停用，
        资料就算"复活"也进不了检索。"""
        folder = self.root / "竞态资料"
        folder.mkdir()
        (folder / "笔记.md").write_text("# 笔记\n\n断开后再复活也搜不到的内容。", encoding="utf-8")
        library = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)}).json()
        self.assertEqual(self.client.delete(
            f"/api/v1/libraries/{library['library_id']}").status_code, 200)

        database = self.client.app.state.database
        with database.transaction() as connection:
            connection.execute(
                "UPDATE documents SET status='ready', removed_at=NULL WHERE library_id=?",
                (library["library_id"],))
        # 直接改库不会进内存快照；真实竞态里是扫描收尾的 _publish_snapshot()
        # 把复活文档带进检索，这里显式走同一条路，active 过滤才有被测的机会。
        self.client.app.state.materials._publish_snapshot()

        self.assertEqual(self.client.get(
            "/api/v1/search", params={"q": "复活也搜不到"}).json()["results"], [])
        self.assertEqual(self.client.get("/api/v1/documents").json()["documents"], [])

    def test_text_pdf_keeps_page_and_scanned_pdf_fails_clearly(self):
        valid = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("paper.pdf", text_pdf("PagedAttention manages KV Cache pages."), "application/pdf")},
        )
        self.assertEqual(valid.status_code, 200)
        hit = self.client.get("/api/v1/search", params={"q": "PagedAttention Cache"}).json()["results"][0]
        self.assertEqual(hit["locator"], {"kind": "pdf", "page": 1})
        pdf_response = self.client.get(
            f"/api/v1/documents/{hit['document_id']}/versions/{hit['version_id']}/file")
        self.assertEqual(pdf_response.status_code, 200)
        self.assertTrue(pdf_response.content.startswith(b"%PDF"))

        buffer = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(buffer)
        scanned = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("scan.pdf", buffer.getvalue(), "application/pdf")},
        )
        self.assertEqual(scanned.status_code, 200)
        documents = self.client.get("/api/v1/documents").json()["documents"]
        failed = next(item for item in documents if item["display_name"] == "scan.pdf")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("OCR", failed["error"])
        retried = self.client.post(f"/api/v1/documents/{failed['id']}/retry")
        self.assertEqual(retried.status_code, 200)
        failed_again = next(item for item in self.client.get(
            "/api/v1/documents").json()["documents"] if item["id"] == failed["id"])
        self.assertEqual(failed_again["status"], "failed")

    def test_bundled_resources_are_listed_and_served(self):
        index = self.client.get("/api/v1/resources").json()
        self.assertIn("欢迎使用.md", [item["name"] for item in index["examples"]])
        self.assertIn("用户指南.md", [item["name"] for item in index["docs"]])

        guide = self.client.get("/api/v1/resources/docs/用户指南.md")
        self.assertEqual(guide.status_code, 200)
        self.assertIn("text/markdown", guide.headers["content-type"])

        for name in ("README.md", "试用反馈台账.md"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.client.get(f"/api/v1/resources/docs/{name}").status_code, 404)

    def test_bundled_example_import_is_idempotent_and_searchable(self):
        first = self.client.post("/api/v1/resources/import", json={"name": "欢迎使用.md"})
        self.assertEqual(first.status_code, 200)
        self.assertFalse(first.json()["already_imported"])

        documents = self.client.get("/api/v1/documents").json()["documents"]
        imported = [item for item in documents if item["display_name"] == "欢迎使用.md"]
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["status"], "ready")

        hits = self.client.get("/api/v1/search", params={"q": "PagedAttention"}).json()
        self.assertTrue(hits["results"])
        self.assertEqual(hits["results"][0]["title"], "欢迎使用.md")

        again = self.client.post("/api/v1/resources/import", json={"name": "欢迎使用.md"})
        self.assertTrue(again.json()["already_imported"])
        self.assertEqual(again.json()["document_id"], imported[0]["id"])
        self.assertEqual(len(self.client.get(
            "/api/v1/documents").json()["documents"]), 1)

    def test_unknown_bundled_resource_is_rejected(self):
        missing = self.client.post("/api/v1/resources/import", json={"name": "不存在.md"})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(self.client.get(
            "/api/v1/documents").json()["documents"], [])

    def test_notebook_cell_locator_lines_into_the_extracted_text(self):
        """notebook 的 locator 行号必须落在拼接文本的真实位置上。

        来源面板拿 locator 的行号去 `source().text` 里取行高亮，两边的行号
        一旦错位，界面就会高亮到别的单元格——而且搜出来的片段看起来还是对的。
        """
        cells = [
            {"cell_type": "markdown", "metadata": {},
             "source": ["# 推理优化\n", "\n", "先看分页注意力的动机。"]},
            {"cell_type": "code", "metadata": {}, "execution_count": 1,
             "source": ["import math\n", "print(math.pi)"],
             "outputs": [{"output_type": "stream", "name": "stdout", "text": ["3.14"]}]},
            {"cell_type": "markdown", "metadata": {},
             "source": ["## PagedAttention\n", "\n", "分页管理 KV Cache。"]},
        ]
        uploaded = self.client.post("/api/v1/documents/upload", files={
            "file": ("优化.ipynb", notebook_bytes(cells), "application/json")})
        self.assertEqual(uploaded.status_code, 200)

        results = self.client.get(
            "/api/v1/search", params={"q": "分页管理 KV Cache"}).json()["results"]
        hit = next(item for item in results if item["locator"]["cell"] == 3)
        self.assertEqual(hit["media_type"], "notebook")
        self.assertEqual(hit["heading_path"], "优化 > 单元格 3 > PagedAttention")
        self.assertEqual(hit["locator"]["kind"], "notebook")

        source = self.client.get(
            f"/api/v1/documents/{hit['document_id']}/versions/{hit['version_id']}/source").json()
        lines = source["text"].splitlines()
        quoted = "\n".join(lines[hit["locator"]["start_line"] - 1:hit["locator"]["end_line"]])
        self.assertIn("分页管理 KV Cache", quoted)

        code = next(item for item in self.client.get(
            "/api/v1/search", params={"q": "print math.pi"}).json()["results"]
            if item["locator"]["cell"] == 2)
        self.assertEqual(code["heading_path"], "优化 > 单元格 2 · 代码")
        self.assertIn("print(math.pi)", "\n".join(
            lines[code["locator"]["start_line"] - 1:code["locator"]["end_line"]]))

        served = self.client.get(
            f"/api/v1/documents/{hit['document_id']}/versions/{hit['version_id']}/file")
        self.assertEqual(served.status_code, 200)
        self.assertIn("application/x-ipynb+json", served.headers["content-type"])

    def test_notebook_rich_output_is_dropped_and_reported(self):
        cells = [{"cell_type": "code", "metadata": {}, "source": ["plot()"], "outputs": [
            {"output_type": "display_data",
             "data": {"image/png": "aGVsbG8=", "text/plain": ["<Figure size 640x480>"]}},
            {"output_type": "execute_result", "data": {"text/plain": ["42"]}},
            {"output_type": "stream", "name": "stdout", "text": ["done\n"]},
        ]}]
        self.client.post("/api/v1/documents/upload", files={
            "file": ("画图.ipynb", notebook_bytes(cells), "application/json")})

        document = self.client.get("/api/v1/documents").json()["documents"][0]
        self.assertEqual(document["status"], "ready")
        self.assertIn("未收录", document["error"])
        source = self.client.get(
            f"/api/v1/documents/{document['id']}/versions/{document['current_version_id']}/source")
        self.assertIn("<Figure size 640x480>", source.json()["text"])
        self.assertNotIn("aGVsbG8=", source.json()["text"])

    def test_broken_notebook_fails_with_a_clear_message(self):
        for name, payload, expected in (
            ("乱码.ipynb", b"not json at all", "有效的 JSON"),
            ("缺单元格.ipynb", json.dumps({"nbformat": 4}).encode("utf-8"), "cells"),
        ):
            with self.subTest(name=name):
                self.client.post("/api/v1/documents/upload",
                                 files={"file": (name, payload, "application/json")})
                document = next(item for item in self.client.get(
                    "/api/v1/documents").json()["documents"] if item["display_name"] == name)
                self.assertEqual(document["status"], "failed")
                self.assertIn(expected, document["error"])

    def test_folder_scan_picks_up_notebooks_next_to_markdown(self):
        folder = self.root / "混合资料"
        folder.mkdir()
        (folder / "笔记.md").write_text("# 笔记\n\n普通 Markdown。", encoding="utf-8")
        (folder / "实验.ipynb").write_text(
            notebook_bytes([{"cell_type": "markdown", "metadata": {},
                             "source": ["# 实验\n", "\n", "记录一次消融实验。"]}])
            .decode("utf-8"), encoding="utf-8")

        connected = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)})
        self.assertEqual(connected.status_code, 200)
        documents = self.client.get("/api/v1/documents").json()["documents"]
        self.assertEqual(sorted(item["media_type"] for item in documents),
                         ["markdown", "notebook"])
        self.assertTrue(all(item["status"] == "ready" for item in documents))

    def test_folder_scan_takes_pdfs_and_isolates_a_scanned_one(self):
        """目录里混进扫描件不应该拖垮整个 job。

        扫描件提取不出文字，是必然失败的；如果它让整轮扫描中断，用户会看到
        文件夹里其他好文件一个都没进来。
        """
        folder = self.root / "论文"
        folder.mkdir()
        (folder / "笔记.md").write_text("# 笔记\n\n先写的笔记。", encoding="utf-8")
        (folder / "有文字.pdf").write_bytes(text_pdf("PagedAttention manages KV Cache pages."))
        buffer = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(buffer)
        (folder / "扫描件.pdf").write_bytes(buffer.getvalue())

        connected = self.client.post("/api/v1/libraries/folders", json={"path": str(folder)})
        job_id = connected.json()["job_id"]
        job = next(item for item in self.client.get(
            "/api/v1/import-jobs").json()["jobs"] if item["id"] == job_id)
        self.assertEqual(job["status"], "completed_with_errors")
        self.assertEqual((job["total"], job["failed"]), (3, 1))

        documents = {item["display_name"]: item for item in self.client.get(
            "/api/v1/documents").json()["documents"]}
        self.assertEqual(documents["笔记.md"]["status"], "ready")
        self.assertEqual(documents["有文字.pdf"]["status"], "ready")
        self.assertEqual(documents["有文字.pdf"]["media_type"], "pdf")
        self.assertEqual(documents["扫描件.pdf"]["status"], "failed")
        self.assertIn("OCR", documents["扫描件.pdf"]["error"])

        hit = self.client.get(
            "/api/v1/search", params={"q": "PagedAttention Cache"}).json()["results"][0]
        self.assertEqual(hit["locator"], {"kind": "pdf", "page": 1})


if __name__ == "__main__":
    unittest.main()
