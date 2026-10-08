"""无头浏览器截图：把一张**自包含**的 HTML 渲染成 PNG。

为什么不用 puppeteer / playwright：冻结版不打包 Node（`product.spec` 里没有 node，
`dist/ObsidianRAG/` 只有 exe + `_internal`），用户机器也不保证有 Node——"复用
`scripts/ui_acceptance.cjs` 的 CDP 通道"这条路在交付版根本不成立。本机浏览器
（Edge / Chrome）自带 `--headless=new --screenshot=`，纯 stdlib 就能驱动，
不需要任何新依赖。

三条踩出来的纪律，都不是风格问题：

1. **每次渲染用一个一次性 `--user-data-dir`。** Windows 上浏览器会把命令行**转交
   给已经在运行的实例**（`msedge.exe --version` 打印的是"正在现有浏览器会话中打开"），
   共用 profile 会撞车，还可能碰到使用者本人的浏览器会话与标签页。
2. **Windows 上要 `CREATE_NO_WINDOW`。** 产品是 `console=False` 的 GUI 程序，
   不加它每导一张图都会闪一个黑色控制台窗口。
3. **`returncode == 0` 不算成功，出图才算。** 浏览器对不少参数问题只打日志不改
   退出码，所以拿到文件后还要校验 PNG 头与尺寸——这里宁可多一步，也不要给用户一个
   0 字节的"成功"。

截图尺寸是**确定**的：`--window-size=W,H` × `--force-device-scale-factor=S`
恰好得到 `W×S` 与 `H×S` 像素（2026-09-18 实测：1080×700 @2 → 2160×1400）。
所以调用方只要算准内容高度，就能拿到一张不裁切、不拖白边的图。
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
import signal


# 显式指定浏览器（便携版遇到"找不到浏览器"时，这是唯一的人工出口）。
RENDERER_ENV = "OBSIDIAN_RAG_BROWSER"
DEFAULT_TIMEOUT = 90
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# 路径按"最可能是装机自带"排序：Windows 一定有 Edge，Chrome 是可选安装。
_WINDOWS_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)
_UNIX_COMMANDS = ("google-chrome", "chromium", "chromium-browser",
                  "microsoft-edge", "msedge", "chrome")

_LABELS = {"msedge": "Microsoft Edge", "chrome": "Google Chrome",
           "chromium": "Chromium", "microsoft-edge": "Microsoft Edge"}


class RendererUnavailable(RuntimeError):
    """本机找不到可用的浏览器。**这是降级路径**，不是错误——界面要如实说明并给出
    已导出的 HTML，让用户用自己的浏览器打开。"""


class RenderFailed(RuntimeError):
    """浏览器跑了，但没出图（超时、参数不被接受、进程崩了）。"""


def _env_browser() -> Path | None:
    value = (os.environ.get(RENDERER_ENV) or "").strip().strip('"')
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def browser_candidates() -> list[Path]:
    """按优先级列出本机可能的浏览器，**不判断是否存在**（便于诊断与测试）。"""
    if os.name == "nt":
        return [*(Path(item) for item in _WINDOWS_CANDIDATES),
                *(Path(found) for found in
                  (shutil.which(name) for name in _UNIX_COMMANDS) if found)]
    return [Path(found) for found in
            (shutil.which(name) for name in _UNIX_COMMANDS) if found]


def find_browser() -> Path | None:
    """返回一个真实存在的浏览器可执行文件；`OBSIDIAN_RAG_BROWSER` 优先级最高。

    **只有 env 指的路径不存在时才回落到自动探测**——反过来（探测优先）会让
    "我明明指定了"变成一句空话，而这条 env 正是降级时的唯一出口。
    """
    override = _env_browser()
    if override is not None:
        return override
    return next((path for path in browser_candidates() if path.is_file()), None)


def browser_label(path: Path | str) -> str:
    stem = Path(path).stem.lower()
    return _LABELS.get(stem, Path(path).name)


def png_dimensions(path: Path) -> tuple[int, int] | None:
    """从 IHDR 读宽高。**不用第三方库**，也用不着解码整张图。"""
    try:
        header = Path(path).read_bytes()[:24]
    except OSError:
        return None
    if len(header) < 24 or header[:8] != PNG_SIGNATURE:
        return None
    return struct.unpack(">II", header[16:24])


def _creation_flags() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def render_args(browser: Path, html_path: Path, png_path: Path, profile: Path,
                width: int, height: int, scale: int) -> list[str]:
    """命令行拼装单独抽出来：这样"参数到底传了什么"是可以断言的，不必真起浏览器。

    两个文件路径在这里 **resolve**：`file://` URL 不接受相对路径（`as_uri()` 会直接
    抛错），而 `--screenshot=` 是相对浏览器自己的工作目录解释的——两者都必须绝对，
    与其指望每个调用方记得，不如在这里一次做掉。
    """
    return [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        f"--force-device-scale-factor={scale}",
        f"--user-data-dir={profile}",
        f"--window-size={width},{height}",
        f"--screenshot={Path(png_path).resolve()}",
        Path(html_path).resolve().as_uri(),
    ]


def _run_browser(args, timeout):
    """超时只终止本次启动的进程树，不触碰用户的浏览器会话。"""
    options = _creation_flags() if os.name == "nt" else {"start_new_session": True}
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=10, **_creation_flags())
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except (OSError, subprocess.TimeoutExpired):
                pass
            finally:
                process.kill()
                process.communicate()
            raise
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def screenshot(html_path, png_path, *, width: int, height: int, scale: int = 2,
               timeout: int = DEFAULT_TIMEOUT, browser=None) -> dict:
    """自动探测逐个尝试；显式指定的浏览器失败时直接报告。"""
    override = browser or (os.environ.get(RENDERER_ENV) or "").strip().strip('"')
    candidates = ([Path(override)] if override else
                  list(dict.fromkeys(p for p in browser_candidates() if p.is_file())))
    if not candidates or not candidates[0].is_file():
        raise RendererUnavailable("没有找到可用的浏览器（Edge 或 Chrome）。")
    failures = []
    started = time.perf_counter()
    for index, candidate in enumerate(candidates):
        remaining = timeout - (time.perf_counter() - started)
        if remaining <= 0:
            break
        try:
            result = _screenshot_once(html_path, png_path, width=width, height=height,
                                      scale=scale, timeout=remaining / (len(candidates) - index),
                                      browser=candidate)
            result["attempt_failures"] = failures
            return result
        except RenderFailed as error:
            failures.append({"browser": browser_label(candidate), "message": str(error)})
    raise RenderFailed("；".join(f'{item["browser"]}: {item["message"]}' for item in failures)
                       or "渲染超时。")


def _screenshot_once(html_path, png_path, *, width: int, height: int, scale: int = 2,
               timeout: int = DEFAULT_TIMEOUT, browser=None) -> dict:
    """把 `html_path` 渲染成 `png_path`，返回渲染记录。

    先写临时文件、成功后再 `replace` 到位（与导出 Markdown 同一套原子落盘）：渲染
    失败时不会在 `exports/` 里留下半张图，也不会覆盖上一次成功的那张。
    """
    html_path, png_path = Path(html_path).resolve(), Path(png_path).resolve()
    target = Path(browser) if browser else find_browser()
    if target is None or not Path(target).is_file():
        raise RendererUnavailable(
            "没有找到可用的浏览器（Edge 或 Chrome）。可以先设置环境变量 "
            f"{RENDERER_ENV} 指向浏览器可执行文件。")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件必须**继续以 .png 结尾**：无头 Chrome 按扩展名判断图片格式，给个
    # `.tmp` 会当场报 `Unsupported screenshot image file type: .tmp` 且不写文件。
    staging = png_path.with_name(f".{png_path.stem}.{uuid.uuid4().hex}.staged.png")
    staging.unlink(missing_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="obsidian-rag-render-"))
    args = render_args(Path(target), html_path, staging, profile, width, height, scale)
    started = time.perf_counter()
    try:
        try:
            done = _run_browser(args, timeout)
        except subprocess.TimeoutExpired:
            raise RenderFailed(f"渲染超时：超过 {timeout} 秒仍未出图。") from None
        except OSError as error:
            raise RenderFailed(f"无法启动浏览器：{error}") from error
        elapsed = int((time.perf_counter() - started) * 1000)
        if done.returncode != 0:
            raise RenderFailed(f"浏览器退出码 {done.returncode}：{_last_line(done.stderr)}")
        if not staging.is_file() or staging.stat().st_size == 0:
            raise RenderFailed(f"浏览器没有写出图片：{_last_line(done.stderr)}")
        size = png_dimensions(staging)
        if size is None:
            raise RenderFailed("渲染出的文件不是 PNG。")
        staging.replace(png_path)
    finally:
        staging.unlink(missing_ok=True)
        shutil.rmtree(profile, ignore_errors=True)
    return {
        "browser": browser_label(target),
        "browser_path": str(target),
        "milliseconds": elapsed,
        "bytes": png_path.stat().st_size,
        "width": size[0],
        "height": size[1],
    }


def _last_line(raw) -> str:
    text = (raw or b"").decode("utf-8", "replace").strip()
    if not text:
        return "浏览器没有输出任何信息。"
    return text.splitlines()[-1][:300]
