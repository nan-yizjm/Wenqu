"""产出相关测试的共用夹具（**不含任何 `test_` 方法**，所以放这里而不是测试模块）。

抽出来的理由：`test_product_studio.py` 与 `test_product_infographic.py` 都要"一个
带假模型的产品应用"。分成两份的话，"API 测试里的应用"会慢慢长成和"服务测试里的
应用"不一样的东西，而它们本该是同一个。

**不要在这里放 TestCase**：继承带 `test_` 方法的父类会把父类用例再跑一遍
（仓库里有过 179→199 的事故）。反过来，测试模块也别 `from tests.test_xxx import`——
那会把对方的用例也导进来再跑一遍。
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi.testclient import TestClient

from src.product.app import create_product_app, current_settings
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager
from src.product.studio import StudioService


def source(number, title, heading_path, text="正文"):
    """一条与 `materials.retrieve()` 同形的来源，用来单测纯函数。"""
    return {"chunk_id": f"c{number}", "document_id": f"d{number}", "version_id": f"v{number}",
            "title": title, "media_type": "markdown", "heading_path": heading_path,
            "locator": {"kind": "chunk"}, "preview": text, "text": text,
            "score": 1.0, "matched_tokens": [], "channels": {}}


def stored_source(label, title, heading_path, origin="note"):
    """一条与 `artifact_sources` 读回来同形的来源（`source_record()` 的形状）。

    与上面的 `source()` 的差别不是风格：`source()` 是"检索结果"，这个是"落库后读回
    的记录"，多了 `label` 与 `origin`——信息图正是拿它画的。
    """
    return {"label": label, "chunk_id": f"c{label}", "document_id": f"d{title}",
            "version_id": "v1", "title": title, "media_type": "markdown",
            "heading_path": heading_path,
            "locator": {"kind": "markdown", "start_line": 1, "end_line": 8},
            "preview": "片段预览", "score": 1.0, "matched_tokens": [], "channels": {},
            "origin": origin}


DEFAULT_GUIDE_CHUNKS = ("分页管理 KV Cache [S1]。", "\n显存碎片减少 [S1]。",
                        "\n这句没有来源。")


class FakeGuideClient:
    """按 token 吐出一篇指南，用来在没有模型的情况下测量回链。

    默认三句里有一句不带来源，所以"命中率 2/3"这个断言是真的在被算出来的；
    传 `chunks=()` 就得到一篇空产出，用来测失败路径。
    """

    def __init__(self, captured, chunks=None):
        self.captured = captured
        self.chunks = list(DEFAULT_GUIDE_CHUNKS if chunks is None else chunks)

    def stream_chat(self, messages, cancel_event=None):
        self.captured.append(messages)
        yield from self.chunks


def new_service(app, client_factory=None) -> StudioService:
    """在同一个应用上再建一个产出服务。

    用于"服务被重新构造"的场景（构造时会把遗留的 `running` 收敛掉）——这样那条
    测试直接对着真实构造过程，不必自己拼五个参数、也就不会跟这里长歪。
    """
    return StudioService(
        app.state.database, app.state.paths, app.state.materials,
        lambda: current_settings(app.state.database), MemoryCredentialStore(),
        client_factory=client_factory)


def build_harness(test_case, upload=True, client_chunks=None):
    """建一个带假模型的应用，返回 `(client, service, captured, app)`。

    数据目录刻意用**含中文的路径**（"产品数据"）：信息图导出要把它变成 `file://`
    URL 交给浏览器，非 ASCII 路径是这条链上最容易出事的一环，夹具顺手覆盖掉。
    """
    temporary = tempfile.TemporaryDirectory()
    test_case.addCleanup(temporary.cleanup)
    captured = []
    paths = ProductPaths(Path(temporary.name) / "产品数据")

    def factory(settings):
        return FakeGuideClient(captured, client_chunks)

    app = create_product_app(
        paths, MemoryCredentialStore(),
        retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
        chat_client_factory=factory,
    )
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    client.__enter__()
    test_case.addCleanup(client.__exit__, None, None, None)
    if upload:
        client.post("/api/v1/documents/upload", files={"file": (
            "推理.md",
            "# 推理\n\n## PagedAttention\n\n分页管理 KV Cache，减少显存碎片。".encode(),
            "text/markdown")})
    service = new_service(app, factory)
    return client, service, captured, app
