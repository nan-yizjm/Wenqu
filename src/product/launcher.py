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
        app_holder["app"] = create_product_app(
            paths, shutdown_callback=lambda: setattr(server, "should_exit", True))
        if not args.no_browser:
            threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(
                f"http://127.0.0.1:{port}")), daemon=True).start()
        logging.info("server_start port=%s", port)
        server.run()
        logging.info("server_stopped")
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
    return 0


if __name__ == "__main__":
    main()
