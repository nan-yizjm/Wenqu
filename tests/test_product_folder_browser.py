import ctypes
import os
import tempfile
import unittest
from pathlib import Path

from src.product import folder_browser
from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.folder_browser import list_directory
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


class FolderBrowserTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("笔记", "附件", "zeta", "Alpha"):
            (self.root / name).mkdir()
        # 这个选择器只选文件夹，文件不该出现在可选项里。
        (self.root / "readme.md").write_text("x", encoding="utf-8")
        (self.root / ".obsidian").mkdir()

    def names(self, path=None):
        return [item["name"] for item in list_directory(str(path or self.root))["entries"]]

    def test_lists_only_directories(self):
        listed = self.names()
        self.assertIn("笔记", listed)
        self.assertNotIn("readme.md", listed)

    def test_excludes_dot_directories(self):
        self.assertNotIn(".obsidian", self.names())

    def test_sorts_case_insensitively_like_the_file_manager(self):
        listed = self.names()
        self.assertEqual(listed, sorted(listed, key=str.casefold))

    def test_entry_paths_are_absolute(self):
        # 前端把 path 原样回传给后端，相对路径会让面包屑逐级跑偏。
        for item in list_directory(str(self.root))["entries"]:
            self.assertTrue(os.path.isabs(item["path"]), item["path"])

    def test_reports_parent_and_roots_so_the_ui_can_go_up(self):
        listing = list_directory(str(self.root))
        self.assertEqual(Path(listing["parent"]), self.root.parent)
        self.assertTrue(listing["roots"])
        self.assertFalse(listing["truncated"])

    def test_defaults_to_the_home_directory(self):
        self.assertTrue(Path(list_directory(None)["path"]).is_dir())

    def test_rejects_a_file(self):
        with self.assertRaises(ValueError):
            list_directory(str(self.root / "readme.md"))

    def test_rejects_a_missing_path(self):
        with self.assertRaises(ValueError):
            list_directory(str(self.root / "没有这个目录"))

    def test_truncates_and_says_so_instead_of_pretending_it_listed_all(self):
        self.addCleanup(setattr, folder_browser, "MAX_ENTRIES", folder_browser.MAX_ENTRIES)
        folder_browser.MAX_ENTRIES = 2
        listing = list_directory(str(self.root))
        self.assertEqual(len(listing["entries"]), 2)
        self.assertTrue(listing["truncated"])

    @unittest.skipUnless(os.name == "nt", "隐藏属性只存在于 Windows")
    def test_excludes_windows_hidden_directories(self):
        hidden = self.root / "系统目录"
        hidden.mkdir()
        ctypes.windll.kernel32.SetFileAttributesW(str(hidden), 2)  # FILE_ATTRIBUTE_HIDDEN
        self.assertNotIn("系统目录", self.names())

    @unittest.skipUnless(os.name == "nt", "盘符只存在于 Windows")
    def test_drive_root_has_a_usable_name_and_no_parent(self):
        listing = list_directory("C:\\")
        self.assertEqual(listing["path"], "C:\\")
        # 盘符根的 Path().name 是空串；不兜底的话面包屑上会出现一个空按钮。
        self.assertEqual(listing["name"], "C:\\")
        self.assertIsNone(listing["parent"])


class FolderEndpointTests(unittest.TestCase):
    """HTTP 层：确认参数能走到列目录，读不到时回 422 而不是 500。"""

    def setUp(self):
        from fastapi.testclient import TestClient

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # 被列的目录和产品自己的数据目录必须是两个地方，否则数据目录会作为
        # 一个子目录混进结果里，测的就不是"只列这个目录"了。
        self.root = Path(temporary.name) / "资料"
        self.root.mkdir()
        (self.root / "笔记").mkdir()
        app = create_product_app(
            ProductPaths(Path(temporary.name) / "产品数据"), MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True)
        self.client = TestClient(app, base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_lists_the_requested_directory(self):
        body = self.client.get("/api/v1/system/folders",
                               params={"path": str(self.root)}).json()
        self.assertEqual([item["name"] for item in body["entries"]], ["笔记"])

    def test_defaults_to_the_home_directory_without_a_path(self):
        body = self.client.get("/api/v1/system/folders").json()
        self.assertTrue(Path(body["path"]).is_dir())
        self.assertTrue(body["roots"])

    def test_an_unreadable_path_is_a_422_not_a_500(self):
        response = self.client.get("/api/v1/system/folders",
                                   params={"path": str(self.root / "没有这个目录")})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "invalid_folder")

    def test_the_old_system_dialog_endpoint_is_gone(self):
        """旧接口靠 powershell 弹系统对话框，对话框会被建在浏览器窗口后面。

        回 405 而不是 404：这个路径仍被前端兜底路由接住（它只认 GET），
        FastAPI 因此报"方法不允许"。总之不能再是 200。
        """
        self.assertEqual(self.client.post("/api/v1/system/pick-folder").status_code, 405)
