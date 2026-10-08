"""Integration acceptance against a running frozen app in an isolated test directory."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import time
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import httpx


@contextmanager
def ollama_fixture():
    """Local protocol fixture, not a real model; it exercises the frozen HTTP client."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(b'{"models":[{"name":"acceptance-fixed-response"}]}')
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/api/chat' and payload['stream'] is True
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.end_headers()
            text = '测试替身：Wenqu 将历史来源保存为文档快照 [S1]。' * 8
            try:
                for i in range(0, len(text), 8):
                    self.wfile.write((json.dumps({'message': {'content': text[i:i+8]}, 'done': False},
                                                ensure_ascii=False)+'\n').encode())
                    self.wfile.flush(); time.sleep(0.02)
                self.wfile.write(b'{"done":true,"message":{"content":""}}\n')
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown(); server.server_close(); thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/release/packaged-acceptance.json"))
    args = parser.parse_args()
    results = []
    with ollama_fixture() as fixture_port, httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=30, trust_env=False) as client:
        root = Path(client.get("/api/v1/setup").json()["data_root"]).resolve()
        allowed = (Path(__file__).resolve().parents[1] / "output/release").resolve()
        if not root.is_relative_to(allowed) or root == allowed:
            raise SystemExit("Refusing to modify non-isolated product data")
        def check(name, condition):
            assert condition, name
            results.append(name)
            print("PASS", name, flush=True)
        check("frozen CPU delivery", client.get("/api/v1/system/diagnostics").json()["frozen"])
        check("version 0.2.3", client.get("/api/v1/health").json()["version"] == "0.2.3")
        check("spoofed local URL rejected", client.patch("/api/v1/settings", json={
            "ollama_base_url": "http://localhost.evil.example"}).status_code == 422)
        check("cross-origin write rejected", client.post("/api/v1/conversations", json={},
            headers={"Origin": "http://127.0.0.1:9999"}).status_code == 403)
        uploaded = client.post("/api/v1/documents/upload", files={"file": (
            "封装验收.md", "# 封装验收\n\n## 版本快照\n\nWenqu 将历史来源保存为文档快照。".encode(), "text/markdown")}).json()
        document_id = uploaded["document_id"]
        hit = None
        for _ in range(60):
            hits = client.get("/api/v1/search", params={"q": "文档快照"}).json()["results"]
            hit = next((h for h in hits if h["document_id"] == document_id), None)
            if hit:
                break
            time.sleep(0.25)
        check("Chinese Markdown import and BM25 search", hit is not None)
        source = client.get(f"/api/v1/documents/{document_id}/versions/{hit['version_id']}/source")
        check("historical source locator", source.status_code == 200 and "文档快照" in source.json()["text"])
        conversation = client.post("/api/v1/conversations", json={"title": "备份恢复验收"}).json()
        stream = client.post(f"/api/v1/conversations/{conversation['id']}/messages/stream",
            json={"question": "北京明天天气怎么样？"})
        events = [json.loads(line) for line in stream.text.splitlines() if line]
        check("offline grounded rejection stream", events[-1].get("rejected") is True)
        client.patch("/api/v1/settings", json={"provider": "ollama",
            "ollama_model": "acceptance-fixed-response", "ollama_base_url": f"http://127.0.0.1:{fixture_port}"}).raise_for_status()
        stream = client.post(f"/api/v1/conversations/{conversation['id']}/messages/stream",
            json={"question": "Wenqu 如何保存历史文档快照？"})
        events = [json.loads(line) for line in stream.text.splitlines() if line]
        check("frozen Ollama protocol stream (fixed test response)", events[-1].get("status") == "complete"
              and any(e.get("type") == "token" for e in events) and events[-1].get("sources"))
        favorite = client.post("/api/v1/favorites", json={"message_id": events[-1]["message_id"]})
        check("favorite save", favorite.status_code == 200)
        favorite_id = favorite.json()["id"]
        exported = client.get(f"/api/v1/favorites/{favorite_id}/export")
        check("UTF-8 Markdown export", exported.status_code == 200 and "回答" in exported.content.decode("utf-8"))
        with client.stream('POST', f"/api/v1/conversations/{conversation['id']}/messages/stream",
                           json={"question": "文档快照如何保存？"}) as stream:
            stopped = False
            for line in stream.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event['type'] == 'token' and not stopped:
                    client.post(f"/api/v1/conversations/{conversation['id']}/messages/{event['message_id']}/stop").raise_for_status()
                    stopped = True
                if event['type'] == 'stopped':
                    break
            check('stream stop persisted', stopped and event['type'] == 'stopped')
        backup = client.post("/api/v1/system/backup").content
        with zipfile.ZipFile(io.BytesIO(backup)) as archive:
            db = archive.read("workspace.sqlite3")
        output = io.BytesIO()
        evil = "corpus/..\\..\\outside.txt"
        files = {"workspace.sqlite3": db, evil: b"bad"}
        manifest = {"format": 1, "files": {name: {"size": len(data), "sha256": sha256(data).hexdigest()}
                                              for name, data in files.items()}}
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, data in files.items():
                archive.writestr(name, data)
        check("Windows traversal backup rejected", client.post("/api/v1/system/restore", files={"file": (
            "bad.zip", output.getvalue(), "application/zip")}).status_code == 422)
        client.delete(f"/api/v1/documents/{document_id}").raise_for_status()
        restored = client.post("/api/v1/system/restore", files={"file": ("good.zip", backup, "application/zip")})
        check("valid backup restore", restored.status_code == 200 and restored.json()["restart_required"])
        check("writes blocked pending restart", client.post("/api/v1/conversations", json={}).status_code == 409)
        started = client.get("/api/v1/health").json()["started_at"]
        check("restart endpoint reachable", client.post("/api/v1/system/restart").status_code == 200)
        health = None
        for _ in range(120):
            try:
                health = client.get("/api/v1/health", timeout=2).json()
                if health["started_at"] != started and not health["restart_required"]:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        check("frozen app restarted after restore", health and health["started_at"] != started and not health["restart_required"])
        check("restored document searchable", any(h["document_id"] == document_id for h in
            client.get("/api/v1/search", params={"q": "文档快照"}).json()["results"]))
        check("restored favorite retained", client.get(f"/api/v1/favorites/{favorite_id}").status_code == 200)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": len(results), "checks": results,
        "generation": "local Ollama protocol fixture with fixed labelled responses; not model quality"}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
