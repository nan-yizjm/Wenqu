"""调用 Windows 自带文件夹选择对话框；不接受用户提供的命令文本。"""

import subprocess


SCRIPT = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '选择包含 Markdown 文件的资料目录'
$dialog.ShowNewFolderButton = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.SelectedPath
}
"""


def pick_folder() -> str | None:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", SCRIPT],
        capture_output=True, timeout=300, creationflags=flags,
    )
    if completed.returncode != 0:
        raise RuntimeError("Windows 文件夹选择器启动失败。")
    value = completed.stdout.decode("utf-8", errors="replace").strip()
    return value or None
