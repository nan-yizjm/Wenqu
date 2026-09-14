from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


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


if __name__ == "__main__":
    unittest.main()
