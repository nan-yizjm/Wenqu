"""Reproducible UI demo using synthetic notes and explicitly labelled fixed responses.

Run from the repository root: python -m scripts.demo_product
This never uses the default personal data directory, downloads models or reads API keys.
"""
import argparse
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import uvicorn

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


class DemoClient:
    json_mode = False

    def stream_chat(self, messages, cancel_event=None):
        if self.json_mode:
            output = json.dumps({"sections": [
                {"h": "测试演示：快速检索"},
                {"s": "BM25 基于词项匹配，无需先下载向量模型。", "src": [1]},
                {"s": "导入资料后可以直接使用关键词搜索。", "src": [1]},
                {"h": "测试演示：核对与保存"},
                {"s": "以下固定响应用于界面验收，不是模型质量评测。", "src": []}
            ]}, ensure_ascii=False)
        else:
            output = ("**测试演示 · 固定响应，非真实模型生成**\n\n"
                      "### 为什么默认使用 BM25？\n\n"
                      "BM25 基于词项匹配，无需先下载向量模型，适合导入后立即开始关键词搜索 [S1]。\n\n"
                      "关键词与标题、章节及正文吻合时，可以找到相关片段。点击引用核对原文，再把有用结论加入收藏 [S1]。\n\n"
                      "这份内容用于展示流式输出、来源阅读与收藏流程，不代表模型回答质量。")
        for i in range(0, len(output), 12):
            if cancel_event and cancel_event.is_set():
                return
            yield output[i:i + 12]
            time.sleep(0.015)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--data-root", type=Path, default=Path("output/release/showcase-data"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    if root == ProductPaths.default().root:
        raise SystemExit("Refusing to use personal product data")
    marker = root / ".wenqu-synthetic-demo"
    if root.exists() and any(root.iterdir()) and not marker.exists():
        raise SystemExit("Refusing to reuse a non-demo directory")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text("Synthetic notes and fixed responses only.\n", encoding="utf-8")
    app = create_product_app(ProductPaths(root), MemoryCredentialStore(),
        retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
        chat_client_factory=lambda settings: DemoClient())
    with TestClient(app, base_url=f"http://127.0.0.1:{args.port}") as client:
        client.patch("/api/v1/settings", json={"theme": "light", "theme_template": "paper",
            "ollama_model": "测试演示（固定响应）", "onboarding_complete": True})
        if not client.get("/api/v1/documents").json()["documents"]:
            for note in sorted((Path(__file__).resolve().parents[1] / "examples/demo").glob("*.md")):
                client.post("/api/v1/documents/upload", files={"file": (
                    "示例-" + note.name, note.read_bytes(), "text/markdown")}).raise_for_status()
        if not client.get("/api/v1/conversations").json()["conversations"]:
            conversation = client.post("/api/v1/conversations", json={"title": "测试演示：BM25 检索"}).json()
            response = client.post(f"/api/v1/conversations/{conversation['id']}/messages/stream",
                json={"question": "BM25 为什么适合快速启动？"})
            events = [json.loads(line) for line in response.text.splitlines() if line]
            favorite = client.post("/api/v1/favorites", json={"message_id": events[-1]["message_id"]}).json()
            client.patch(f"/api/v1/favorites/{favorite['id']}", json={
                "title": "测试演示：BM25 的启动优势", "note": "合成资料与固定响应，用于展示收藏和 Markdown 导出。",
                "tags": ["示例", "检索"]})
            client.get(f"/api/v1/favorites/{favorite['id']}/export").raise_for_status()
            for kind in ("guide", "mindmap"):
                client.post("/api/v1/artifacts/stream", json={"topic": "BM25 检索与证据", "kind": kind}).raise_for_status()
    print(f"Synthetic demo: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
