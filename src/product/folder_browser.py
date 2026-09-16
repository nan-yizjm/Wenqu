"""应用内的文件夹浏览：只读列目录，供网页端的选择器使用。

为什么不再用系统文件夹对话框：安装版是 PyInstaller 的 windowed 进程（无控制台、
无主窗口），它拉起的 PowerShell 子进程拿不到前台权限，Windows 会把对话框创建在
浏览器窗口**后面**。z 序实测（`#32770` 排在正在前台的 Chrome 之前还是之后）显示
对话框确实被压在 Chrome 底下：用户点了等于没反应，界面又没有反馈，于是反复点击，
堆出一串僵住的对话框。给对话框加一个 TopMost 的 owner 窗体也无效，同样实测过。

改成由后端列目录、前端画选择器之后，行为完全由 HTTP 决定，可以被离线测试覆盖，
也不再依赖 powershell.exe（企业 EDR 经常拦"脚本宿主拉起 WinForms 弹窗"这种模式）。
"""

from __future__ import annotations

import ctypes
import os
import stat
import string
from pathlib import Path

# 一个目录最多回这么多子目录。系统盘根目录、node_modules 这类动辄上千项，
# 全量回给前端既慢又没法看；超出时如实上报 truncated，不假装列全了。
MAX_ENTRIES = 500


def _drive_roots() -> list[dict[str, str]]:
    """可用盘符。非 Windows 只给根目录。"""
    if os.name != "nt":
        return [{"name": "/", "path": "/"}]
    # GetLogicalDrives 只读一个位图；用 os.path.exists 去试探每个字母会碰到
    # 没插盘的软驱/读卡器，那会卡住好几秒。
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    return [{"name": f"{letter}:", "path": f"{letter}:\\"}
            for index, letter in enumerate(string.ascii_uppercase)
            if bitmask >> index & 1]


def _is_hidden(entry: os.DirEntry) -> bool:
    """隐藏项不作为可选项。系统目录（AppData、$RECYCLE.BIN 之类）只会是噪声。"""
    if entry.name.startswith("."):
        return True
    if os.name != "nt":
        return False
    try:
        attributes = entry.stat(follow_symlinks=False).st_file_attributes
    except OSError:
        # 读不到属性说明基本没有访问权限，直接当作不可选。
        return True
    return bool(attributes & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))


def _subdirectories(target: Path) -> tuple[list[dict[str, str]], bool]:
    entries: list[dict[str, str]] = []
    truncated = False
    with os.scandir(target) as iterator:
        for entry in iterator:
            if len(entries) >= MAX_ENTRIES:
                truncated = True
                break
            if _is_hidden(entry):
                continue
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            entries.append({"name": entry.name, "path": str(Path(entry.path))})
    # 不区分大小写排序，和资源管理器一致。
    entries.sort(key=lambda item: item["name"].casefold())
    return entries, truncated


def list_directory(path: str | None = None) -> dict:
    """列出 `path` 下的子目录。缺省从用户主目录开始，不猜别的。"""
    raw = (path or "").strip() or str(Path.home())
    target = Path(raw).expanduser()
    try:
        if not target.is_dir():
            raise ValueError("这个位置不存在或读不到，请换一个文件夹。")
        entries, truncated = _subdirectories(target)
    except PermissionError as error:
        raise ValueError("没有权限读取这个文件夹，请换一个。") from error
    except OSError as error:
        raise ValueError("这个位置不存在或读不到，请换一个文件夹。") from error

    resolved = Path(os.path.abspath(target))
    parent = resolved.parent
    return {
        "path": str(resolved),
        # 盘符根的 name 是空串，退回完整路径，否则面包屑上会出现一个空格。
        "name": resolved.name or str(resolved),
        "parent": None if parent == resolved else str(parent),
        "entries": entries,
        "truncated": truncated,
        "roots": _drive_roots(),
    }
