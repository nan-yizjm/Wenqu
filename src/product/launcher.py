"""Windows 产品启动器：单实例、自动选择端口、打开浏览器。"""

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading
import time
import webbrowser

from .app import create_product_app
from .paths import ProductPaths


def available_port(preferred=8765):
    for port in range(preferred, preferred + 20):
        with socket.socket() as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("8765-8784 端口均被占用")


def relaunch_command(port: int, argv: list[str] | None = None,
                     frozen: bool | None = None, module: str | None = None) -> list[str]:
    """重启自己时该执行的命令。三个决定都有具体原因：

    - **端口固定**：前端正等着这个端口回来。让它重新挑端口（`available_port`）
      可能拿到另一个号，页面就永远等不到了——那看起来像"重启失败"。
    - **不再开浏览器**：用户已经在页面上，重启只是把服务换一茬；再弹一个标签页是噪音。
    - **模块名走 `-m`**：`sys.argv[0]` 是 `product_entry.py` 的路径，直接执行它会让
      `sys.path[0]` 变成 `src/`，`from src.product...` 这个绝对导入就找不到包了。
      所以源码运行必须用 `-m <模块名>` 并把 cwd 语义保留下来；冻结版没有这个问题，
      直接再执行 exe。
    """
    argv = list(sys.argv if argv is None else argv)
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if module is None:
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        module = getattr(spec, "name", None)
    if is_frozen:
        base = [sys.executable]
    elif module:
        base = [sys.executable, "-m", module]
    else:
        base = [sys.executable, argv[0]]
    kept, index = [], 1
    while index < len(argv):
        item = argv[index]
        if item == "--port":                      # 连值一起丢掉，由这里统一追加
            index += 2
            continue
        if item == "--no-browser" or item.startswith("--port="):
            index += 1
            continue
        kept.append(item)
        index += 1
    return [*base, *kept, "--port", str(port), "--no-browser"]


class InstanceLock:
    def __init__(self, path: Path):
        self.path, self.handle = path, None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.handle.write(b"0"); self.handle.flush(); self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self.handle.close(); self.handle = None
            return False

    def close(self):
        if not self.handle: return
        self.handle.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt; msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl; fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close(); self.handle = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths = ProductPaths(args.data_root.resolve()) if args.data_root else ProductPaths.default()
    paths.ensure()
    logging.basicConfig(
        filename=paths.logs / "launcher.log", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", force=True,
    )
    logging.info("launcher_start frozen=%s", bool(getattr(__import__('sys'), 'frozen', False)))
    lock = InstanceLock(paths.runtime / "product.lock")
    instance_file = paths.runtime / "instance.json"
    if not lock.acquire():
        port = 8765
        try:
            port = int(json.loads(instance_file.read_text(encoding="utf-8"))["port"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
        webbrowser.open(f"http://127.0.0.1:{port}")
        return
    try:
        import uvicorn
        port = args.port or available_port()
        temporary = instance_file.with_suffix(".tmp")
        temporary.write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
        os.replace(temporary, instance_file)
        app_holder = {}
        config = uvicorn.Config(
            lambda: app_holder["app"], host="127.0.0.1", port=port,
            access_log=False, log_level="info", factory=True, log_config=None,
        )
        server = uvicorn.Server(config)
        # 退出与重启共用一个出口：都让 uvicorn 停，区别只在停完做什么。
        # 意图记在局部字典里，因为回调是闭包——它要能改这个值。
        intent = {"restart": False}

        def stop(restart: bool = False):
            logging.info("stop_requested restart=%s", restart)
            intent["restart"] = intent["restart"] or restart
            server.should_exit = True

        app_holder["app"] = create_product_app(
            paths, shutdown_callback=lambda: stop(False), restart_callback=lambda: stop(True))
        if not args.no_browser:
            threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(
                f"http://127.0.0.1:{port}")), daemon=True).start()
        logging.info("server_start port=%s", port)
        server.run()
        logging.info("server_stopped restart=%s", intent["restart"])
    except BaseException:
        logging.exception("launcher_failed")
        if getattr(sys, "frozen", False):
            return 1
        raise
    finally:
        try:
            instance_file.unlink(missing_ok=True)
        except OSError:
            pass
        lock.close()
    if intent["restart"]:
        # 锁与 instance.json 上面都已经放掉了，新实例能重新拿到锁。
        # 注意 Windows 的 `os.execv` 与 POSIX 不是一回事：它是"起一个新进程、结束本进程"
        # （Windows 没有真正的 exec），**PID 会变**——所以"重启成功"的判据是端口与
        # 启动时刻（health 的 started_at），不是 PID。
        command = relaunch_command(port)
        logging.info("relaunch command=%s", command)
        os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    main()
