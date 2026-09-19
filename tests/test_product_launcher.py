import sys
import unittest

from src.product.launcher import relaunch_command


class RelaunchCommandTests(unittest.TestCase):
    """重启命令是纯函数，所以它必须被钉死——写错的后果是"点了重启、页面再也等不到服务"。

    三个约束：端口固定（页面等的就是它）、不再开浏览器（用户已经在页面上）、
    源码运行必须走 `-m`（直接执行 product_entry.py 会让 sys.path[0] 变成 src/，
    绝对导入 `from src.product...` 就找不到包）。
    """

    def test_a_frozen_build_reruns_the_executable(self):
        command = relaunch_command(8765, argv=["C:/app/ObsidianRAG.exe", "--data-root", "D:/data"],
                                   frozen=True, module=None)

        self.assertEqual(command, [sys.executable, "--data-root", "D:/data",
                                   "--port", "8765", "--no-browser"])

    def test_source_runs_keep_the_module_form(self):
        command = relaunch_command(
            8765, argv=["C:/p/src/product_entry.py", "--data-root", "D:/data"],
            frozen=False, module="src.product_entry")

        self.assertEqual(command, [sys.executable, "-m", "src.product_entry",
                                   "--data-root", "D:/data", "--port", "8765", "--no-browser"])

    def test_an_existing_port_is_replaced_not_duplicated(self):
        """原来带 --port 时必须换成实际端口：`available_port()` 可能选中另一个号，
        页面等的是启动时那一个。"""
        for argv in (["x.py", "--port", "9000", "--data-root", "D:/data"],
                     ["x.py", "--port=9000", "--data-root", "D:/data"]):
            with self.subTest(argv=argv):
                command = relaunch_command(8765, argv=argv, frozen=True, module=None)

                self.assertEqual(command.count("--port"), 1)
                self.assertEqual(command[-2:], ["8765", "--no-browser"])
                self.assertNotIn("9000", command)
                self.assertIn("--data-root", command)

    def test_browser_flag_is_not_carried_over(self):
        command = relaunch_command(8765, argv=["x.py", "--no-browser", "--data-root", "D:/data"],
                                   frozen=True, module=None)

        self.assertEqual(command.count("--no-browser"), 1)

    def test_without_a_module_name_it_falls_back_to_the_script_path(self):
        command = relaunch_command(8765, argv=["C:/p/src/product_entry.py"], frozen=False, module="")

        self.assertEqual(command, [sys.executable, "C:/p/src/product_entry.py",
                                   "--port", "8765", "--no-browser"])


if __name__ == "__main__":
    unittest.main()
