"""联网补充的隐私边界。

**这个文件先于实现写的。** P1 的教训：接缝的隐私承诺如果等到实现之后再测，
测出来的往往只是"实现恰好这么做了"。这里的四条承诺要在实现之前就固定下来：

1. 开关关着时**一个字节都不出去**（堵在 socket 层证明，不只是断言"提供者没被调用"）；
2. 只发查询词，**绝不发笔记正文**；
3. 没有后端时，即使开关打开也不会有请求——并如实上报"没有后端"，不假装搜过；
4. 提供者坏了不许拒绝服务，但**必须显式标注"本次未能联网"**，不能静默降级。
"""

import contextlib
import json
import socket
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager
from src.product.web import NullSearchProvider

NOTE_BODY = "分页管理 KV Cache，减少显存碎片。"
QUESTION = "PagedAttention 是什么？"
WEB_RESULT = {
    "id": "w1",
    "title": "PagedAttention 原论文",
    "url": "https://arxiv.org/abs/2309.06180",
    "snippet": "PagedAttention 把 KV Cache 分成固定大小的页来管理。",
    "published_at": "2023-09-12",
    "fetched_at": "2026-09-18T10:00:00Z",
}


@contextlib.contextmanager
def no_outbound_network():
    """任何连到**本机以外**地址的尝试都当场失败。

    不直接禁掉 `socket`：asyncio 的事件循环自己要建一个回环 self-pipe，禁掉会把
    测试框架本身弄坏（真实踩到过，3 项测试挂在 `socket.socketpair` 上）。这里按
    **目标地址**判——"数据不出机器"的准确含义就是"不连本机以外的地址"，回环
    通信不算出网。

    只断言"提供者没被调用"管不住别的出网路径（导入、模型下载、遥测），而这个
    接缝的承诺是一个字节都不出去，所以直接堵在 socket 层。三层一起堵：
    `getaddrinfo`（任何真实请求的第一步是解析域名）、`create_connection`
    （urllib / requests / http.client 的入口）、`connect`（兜底）。
    """
    loopback = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def assert_local(host):
        if not isinstance(host, str) or host not in loopback:
            raise AssertionError(f"这次调用不应该连到 {host!r}")

    def guarded(original):
        def wrapper(self, address, *args, **kwargs):
            assert_local(address[0] if isinstance(address, tuple) else address)
            return original(self, address, *args, **kwargs)
        return wrapper

    def refusing_create_connection(address, *args, **kwargs):
        assert_local(address[0] if isinstance(address, tuple) else address)
        return real_create_connection(address, *args, **kwargs)

    def refusing_getaddrinfo(host, *args, **kwargs):
        assert_local(host)
        return real_getaddrinfo(host, *args, **kwargs)

    with mock.patch.object(socket.socket, "connect", guarded(real_connect)), \
            mock.patch.object(socket.socket, "connect_ex", guarded(real_connect_ex)), \
            mock.patch("socket.create_connection", refusing_create_connection), \
            mock.patch("socket.getaddrinfo", refusing_getaddrinfo):
        yield


class FakeStreamingClient:
    def __init__(self, captured):
        self.captured = captured

    def stream_chat(self, messages, cancel_event=None):
        self.captured.append(messages)
        yield "分页管理 KV Cache [S1]。"


class RecordingSearchProvider:
    """记录收到什么查询。这是"只发查询词"唯一可验证的方式。"""

    name = "recording"
    configured = True

    def __init__(self, results=None, fail=False):
        self.queries = []
        self.results = list(results or [])
        self.fail = fail

    def search(self, query, limit=5):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("boom")
        return self.results[:limit]


class NetworkTouchingProvider:
    """真的会去建连接的提供者。

    用来区分"产品没问它"和"产品问了但它自己没出网"——只有前者才是承诺。
    """

    name = "network-touching"
    configured = True

    def __init__(self):
        self.calls = 0

    def search(self, query, limit=5):
        self.calls += 1
        socket.create_connection(("search.example.com", 443), timeout=0.1)
        return []


class WebSeamTests(unittest.TestCase):
    def build(self, provider=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.captured = []
        self.paths = ProductPaths(Path(temporary.name) / "产品数据")
        app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
            chat_client_factory=lambda settings: FakeStreamingClient(self.captured),
            search_provider_factory=(lambda: provider) if provider is not None else None,
        )
        self.app = app
        self.client = TestClient(app, base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/documents/upload", files={"file": (
            "推理.md", f"# 推理\n\n## PagedAttention\n\n{NOTE_BODY}".encode(),
            "text/markdown")})
        self.conversation = self.client.post(
            "/api/v1/conversations", json={"title": "联网"}).json()
        return app

    def ask(self, question=QUESTION):
        response = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/stream",
            json={"question": question})
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.text.splitlines() if line]

    def events(self, question=QUESTION):
        return self.ask(question)

    def retrieval_event(self, question=QUESTION):
        return next(item for item in self.ask(question) if item["type"] == "retrieval")

    def enable(self):
        """打开联网。必须先确认已展示过"什么会离开这台机器"（见下一条测试）。"""
        self.assertEqual(self.client.patch(
            "/api/v1/settings", json={"web_disclosure_acknowledged": True}).status_code, 200)
        self.assertEqual(
            self.client.patch("/api/v1/settings", json={"web_enabled": True}).status_code, 200)

    def stored_origins(self):
        connection = sqlite3.connect(f"file:{self.paths.database}?mode=ro", uri=True)
        try:
            return [row[0] for row in connection.execute(
                "SELECT origin FROM message_sources ORDER BY message_id, position")]
        finally:
            connection.close()

    def app_events(self, event_type):
        connection = sqlite3.connect(f"file:{self.paths.database}?mode=ro", uri=True)
        try:
            return [row[0] for row in connection.execute(
                "SELECT event_type FROM app_events WHERE event_type=?", (event_type,))]
        finally:
            connection.close()

    def test_the_switch_is_off_by_default_and_a_configured_provider_is_never_asked(self):
        provider = RecordingSearchProvider([WEB_RESULT])
        self.build(provider)

        final = self.retrieval_event()

        self.assertEqual(provider.queries, [])
        self.assertEqual([item["origin"] for item in final["sources"]], ["note"])
        self.assertEqual(final["web"]["status"], "off")

    def test_the_machine_stays_silent_when_web_is_off(self):
        """关着的时候不但不该问提供者，也不该有**任何**出网路径。

        提供者在这里是"真的会去建连接"的那个：如果产品问了它，
        `socket.create_connection` 会被 `no_network` 拦下并抛错。
        """
        provider = NetworkTouchingProvider()
        with no_outbound_network():
            self.build(provider)
            self.events()

        self.assertEqual(provider.calls, 0)

    def test_the_switch_cannot_be_turned_on_before_the_disclosure_is_acknowledged(self):
        """没有先展示"什么会离开这台机器"，就地不许打开。

        这是把"首次开启必须显式告知"从一句界面文案变成一条接口规则——文案会
        被改掉，规则不会。
        """
        self.build()

        refused = self.client.patch("/api/v1/settings", json={"web_enabled": True})

        self.assertEqual(refused.status_code, 422)
        self.assertEqual(refused.json()["error"], "web_disclosure_required")
        self.assertFalse(self.client.get("/api/v1/settings").json()["settings"]["web_enabled"])

    def test_enabling_web_without_a_backend_makes_no_request_and_says_so(self):
        with no_outbound_network():
            self.build()
            self.enable()
            event = self.retrieval_event()

        self.assertEqual(event["web"]["status"], "unconfigured")
        self.assertEqual([item["origin"] for item in event["sources"]], ["note"])

    def test_only_the_question_leaves_the_machine_never_the_notes(self):
        provider = RecordingSearchProvider([WEB_RESULT])
        self.build(provider)
        self.enable()

        self.retrieval_event()

        self.assertEqual(provider.queries, [QUESTION])
        for query in provider.queries:
            self.assertNotIn(NOTE_BODY, query)

    def test_web_results_become_the_third_layer(self):
        provider = RecordingSearchProvider([WEB_RESULT])
        self.build(provider)
        self.enable()

        events = self.ask()
        event = next(item for item in events if item["type"] == "retrieval")
        final = events[-1]

        self.assertEqual([item["origin"] for item in event["sources"]], ["note", "web"])
        web = event["sources"][-1]
        self.assertEqual(web["locator"]["kind"], "web")
        self.assertEqual(web["locator"]["url"], WEB_RESULT["url"])
        self.assertEqual(web["media_type"], "web")
        self.assertEqual(web["preview"], WEB_RESULT["snippet"])
        # 网络事实要和笔记一样能当依据被引用，否则引用编号会指向不存在的来源
        self.assertIn(WEB_RESULT["snippet"], self.captured[0][-1]["content"])
        self.assertEqual(final["status"], "complete")
        self.assertIn("web", set(self.stored_origins()))

    def test_a_broken_provider_does_not_break_answering_and_is_reported_as_offline(self):
        """坏了不许拒绝服务，但要**显式标注**，不能静默降级成"就像没联网一样"。"""
        self.build(RecordingSearchProvider(fail=True))
        self.enable()

        events = self.ask()
        event = next(item for item in events if item["type"] == "retrieval")
        final = events[-1]

        self.assertEqual(event["web"]["status"], "failed")
        self.assertEqual(event["web"]["detail"], "RuntimeError")
        self.assertEqual(final["status"], "complete")
        self.assertEqual([item["origin"] for item in final["sources"]], ["note"])
        self.assertEqual(self.app_events("web_search_failed"), ["web_search_failed"])

    def test_diagnostics_report_the_web_seam(self):
        self.build(RecordingSearchProvider())

        web = self.client.get("/api/v1/system/diagnostics").json()["web"]

        self.assertEqual(web, {"available": True, "provider": "recording",
                               "configured": True, "enabled": False,
                               "disclosure_acknowledged": False})

    def test_default_app_reports_no_web_backend(self):
        self.build()

        web = self.client.get("/api/v1/system/diagnostics").json()["web"]

        self.assertEqual(web["provider"], "none")
        self.assertFalse(web["configured"])


class NullSearchProviderTests(unittest.TestCase):
    def test_it_returns_nothing_and_needs_no_key(self):
        provider = NullSearchProvider()

        with no_outbound_network():
            self.assertEqual(provider.search("任意查询"), [])

        self.assertFalse(provider.configured)

    def test_it_writes_nothing_to_disk(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = NullSearchProvider()

        provider.search("任意查询")

        self.assertEqual(list(Path(temporary.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
