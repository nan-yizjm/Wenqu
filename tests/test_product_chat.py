import json
from pathlib import Path
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient

from src.llm import OllamaBusy
from src.product.app import create_product_app
from src.product.chat import (BYTES_PER_TOKEN, EVIDENCE_CHUNKS, OLLAMA_CONTEXT_TOKENS,
                              OUTPUT_RESERVE_TOKENS, PROMPT_OVERHEAD_TOKENS,
                              default_chat_client, evidence_budget_bytes,
                              evidence_reduction_note, fit_evidence, safe_error)
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager


class SafeErrorTests(unittest.TestCase):
    """问答页与产出页显示同一句话的出处。忙碌与断连是两回事，必须分开说。"""

    def test_an_ollama_busy_timeout_passes_the_reason_through(self):
        code, message = safe_error(OllamaBusy(
            "本机 Ollama 在 120 秒内没有返回数据：它可能正被其他任务占用。"))
        self.assertEqual(code, "ollama_busy")
        self.assertIn("没有返回", message)
        self.assertNotIn("请确认服务已启动", message)

    def test_a_genuine_disconnect_keeps_the_old_code_and_wording(self):
        code, message = safe_error(RuntimeError(
            "无法连接到 Ollama。请确认 Ollama 已安装并正在运行。"))
        self.assertEqual(code, "ollama_unavailable")
        self.assertIn("请确认", message)


class FakeStreamingClient:
    def __init__(self, captured):
        self.captured = captured

    def stream_chat(self, messages, cancel_event=None):
        self.captured.append(messages)
        for part in ("PagedAttention 使用分页管理 KV Cache", "，减少显存碎片 [S1]。"):
            yield part


class ProductChatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.captured = []
        self.paths = ProductPaths(Path(temporary.name) / "产品数据")
        self.app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True,
            chat_client_factory=lambda settings: FakeStreamingClient(self.captured),
        )
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/documents/upload", files={"file": (
            "推理.md", "# 推理\n\n## PagedAttention\n\n分页管理 KV Cache，减少显存碎片。".encode(),
            "text/markdown")})
        self.conversation = self.client.post(
            "/api/v1/conversations", json={"title": "推理讨论"}).json()

    def events(self, body):
        response = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/stream", json=body)
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.text.splitlines() if line]

    def test_stream_persists_answer_and_snapshot_sources(self):
        events = self.events({"question": "PagedAttention 是什么？"})
        self.assertEqual([item["type"] for item in events],
                         ["retrieval", "generation", "token", "token", "final"])
        final = events[-1]
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["sources"][0]["label"], "S1")
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        self.assertEqual(len(loaded["messages"]), 2)
        answer = loaded["messages"][-1]
        self.assertIn("[S1]", answer["content"])
        self.assertTrue(answer["index_version"].startswith("idx_"))
        self.assertEqual(answer["sources"][0]["locator"]["kind"], "markdown")

    def test_evidence_window_is_eight_chunks_not_five(self):
        """一轮问答放几个片段是实测调过的参数，不能被悄悄改回去。

        这里放足够多的片段，让 5 和 8 真的能区分开。实测依据：用 5 时 dev 集
        26 题里有 3 题的期望文档**根本不在证据里**，无从给出正确引用；加到 8
        时这 3 题全部改善、零退化（`docs/产品检索评测-2026-09-16.md` §13.4）。
        """
        # 每节 520 字符，都在 800 的切片上限之内，所以一节至少一个片段；
        # 16 节保证可用片段数远多于 8，考的是"取几个"而不是"有没有得取"。
        sections = "\n\n".join(
            f"## 分页 {index}\n\nPagedAttention 的第 {index} 个要点：{'填' * 480}"
            for index in range(1, 17))
        self.client.post("/api/v1/documents/upload", files={
            "file": ("长笔记.md", f"# 长笔记\n\n{sections}".encode(), "text/markdown")})

        final = self.events({"question": "PagedAttention 是什么？"})[-1]
        self.assertEqual(len(final["sources"]), 8)
        # 引用编号必须连续：缺一个，界面就会出现点不开的引用。
        self.assertEqual([item["label"] for item in final["sources"]],
                         [f"S{i}" for i in range(1, 9)])

    def test_replayed_sources_carry_the_same_evidence_as_the_live_stream(self):
        """刚答完和翻旧的必须是同一个形状。

        实时事件里带上分数、重放时查不到，界面就得为"刚答完"和"翻旧的"写两套
        判断；反过来（重放多出键）同样麻烦。所以这里逐键比对。
        """
        events = self.events({"question": "PagedAttention 是什么？"})
        live = events[0]["sources"][0]
        self.assertEqual(live["label"], "S1")
        self.assertGreater(live["score"], 0)
        self.assertIn("pagedattention", live["matched_tokens"])
        # 只开关键词检索时命中通道是 bm25；向量的名次要等混合检索才有。
        self.assertEqual(live["channels"], {"bm25": 1})

        replayed = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()["messages"][-1]["sources"][0]
        self.assertEqual(replayed, live)

    def test_followup_rewrites_retrieval_but_history_is_not_evidence(self):
        self.events({"question": "PagedAttention 是什么？"})
        self.events({"question": "它解决什么问题？"})
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        last = loaded["messages"][-1]
        self.assertIn("PagedAttention 是什么", last["retrieval_query"])
        self.assertIn("当前追问", last["retrieval_query"])
        prompt = self.captured[-1]
        self.assertIn("不可引用为事实", prompt[1]["content"])
        self.assertIn("当前检索证据", prompt[-1]["content"])

    def test_static_ood_is_rejected_without_model_call(self):
        events = self.events({"question": "北京明天天气怎么样？"})
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["rejected"])
        self.assertEqual(self.captured, [])

    def test_guard_rejection_is_marked_and_can_be_overridden(self):
        question = "今天的推理优化有什么进展？"
        rejected = self.events({"question": question})
        self.assertTrue(rejected[-1]["rejected"])
        message_id = rejected[-1]["message_id"]
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        marked = next(item for item in loaded["messages"] if item["id"] == message_id)
        self.assertEqual(marked["error_code"], "guard_rejected")
        self.assertEqual(self.captured, [])

        overridden = self.events({"question": question, "skip_guard": True})
        self.assertNotIn("rejected", overridden[-1])
        self.assertEqual(overridden[-1]["status"], "complete")
        self.assertTrue(self.captured)

    def test_unknown_conversation_and_retry_are_rejected_before_streaming(self):
        missing = self.client.post(
            "/api/v1/conversations/not-found/messages/stream", json={"question": "RAG 是什么"})
        self.assertEqual(missing.status_code, 404)
        retry = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/stream",
            json={"retry_message_id": "not-found"})
        self.assertEqual(retry.status_code, 404)

    def test_cancelled_generation_is_saved_as_stopped_and_retryable(self):
        cancel = threading.Event()
        stream = self.client.app.state.chat.stream(
            self.conversation["id"], "PagedAttention 是什么？", cancel_event=cancel)
        events = [next(stream), next(stream)]
        message_id = events[-1]["message_id"]
        stopped = self.client.post(
            f"/api/v1/conversations/{self.conversation['id']}/messages/{message_id}/stop")
        self.assertTrue(stopped.json()["stopping"])
        events.extend(stream)
        self.assertEqual(events[-1]["type"], "stopped")
        message_id = events[-1]["message_id"]
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        stopped = next(item for item in loaded["messages"] if item["id"] == message_id)
        self.assertEqual(stopped["status"], "stopped")
        retried = self.events({"retry_message_id": message_id})
        self.assertEqual(retried[-1]["status"], "complete")

    def test_deepseek_without_credential_fails_without_switching_provider(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        app = create_product_app(
            ProductPaths(Path(temporary.name) / "隔离数据"), MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(), material_run_inline=True)
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            client.patch("/api/v1/settings", json={"provider": "deepseek"})
            client.post("/api/v1/documents/upload", files={"file": (
                "RAG.md", "# RAG\n\n检索证据后生成回答。".encode(), "text/markdown")})
            conversation = client.post("/api/v1/conversations", json={}).json()
            response = client.post(
                f"/api/v1/conversations/{conversation['id']}/messages/stream",
                json={"question": "RAG 如何生成回答？"})
            events = [json.loads(line) for line in response.text.splitlines() if line]
            self.assertEqual(events[-1]["type"], "error")
            self.assertEqual(events[-1]["error"], "deepseek_not_configured")
            loaded = client.get(f"/api/v1/conversations/{conversation['id']}").json()
            self.assertEqual(loaded["messages"][-1]["provider"], "deepseek")
            self.assertEqual(loaded["messages"][-1]["status"], "failed")

    def test_delete_conversation_removes_its_messages(self):
        self.events({"question": "PagedAttention 是什么？"})
        response = self.client.delete(f"/api/v1/conversations/{self.conversation['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["messages"], 2)
        self.assertEqual(self.client.get("/api/v1/conversations").json()["conversations"], [])
        self.assertEqual(self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").status_code, 404)

    def test_delete_conversation_keeps_favorites_and_reports_how_many(self):
        """删会话是"丢掉这段对话记录"，不是"丢掉我挑出来的结论"。

        favorites 建的时候自己存了一份 question / answer 与来源快照，没有指向
        messages 的外键，级联删不到它。这里把这个行为固定成测试，免得以后有人
        顺手给它补个外键、把用户的收藏一起带走。
        """
        final = self.events({"question": "PagedAttention 是什么？"})[-1]
        favorite = self.client.post(
            "/api/v1/favorites", json={"message_id": final["message_id"]}).json()
        body = self.client.delete(f"/api/v1/conversations/{self.conversation['id']}").json()
        self.assertEqual(body["kept_favorites"], 1)
        kept = self.client.get(f"/api/v1/favorites/{favorite['id']}").json()
        self.assertIn("PagedAttention", kept["question"])

    def test_delete_missing_conversation_is_not_found(self):
        self.assertEqual(
            self.client.delete("/api/v1/conversations/conv_不存在").status_code, 404)

    def _pin_active_stream(self, message_id):
        """把某个回答伪造成"正在生成"。

        真跑一个不会结束的流不好做；这里直接往 _active 里挂一条属于该会话的消息，
        考的正是"删除前会不会先查有没有在跑"。
        """
        chat = self.client.app.state.chat
        with chat._active_lock:
            chat._active[message_id] = threading.Event()
        self.addCleanup(lambda: chat._active.pop(message_id, None))

    def test_delete_is_refused_while_an_answer_is_streaming(self):
        # 生成中途删会话，后面写 message_sources 会撞外键约束、在流里抛异常。
        message_id = self.events({"question": "PagedAttention 是什么？"})[-1]["message_id"]
        self._pin_active_stream(message_id)
        response = self.client.delete(f"/api/v1/conversations/{self.conversation['id']}")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "conversation_busy")
        # 拦住就得是完整的拦住，不能删一半。
        self.assertEqual(self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").status_code, 200)

    def test_a_stream_in_another_conversation_does_not_block_this_delete(self):
        """守卫要按会话算。退化成"有任何一个流在跑就都不许删"会让界面莫名其妙。"""
        other = self.client.post("/api/v1/conversations", json={"title": "另一个会话"}).json()
        message_id = self.events({"question": "PagedAttention 是什么？"})[-1]["message_id"]
        self._pin_active_stream(message_id)
        self.assertEqual(
            self.client.delete(f"/api/v1/conversations/{other['id']}").status_code, 200)

    def _upload_long_note(self, sections=20):
        body = "\n\n".join(
            f"## 分页 {index}\n\nPagedAttention 的第 {index} 个要点：{'填' * 700}"
            for index in range(1, sections + 1))
        self.client.post("/api/v1/documents/upload", files={
            "file": ("超长笔记.md", f"# 超长笔记\n\n{body}".encode(), "text/markdown")})

    def test_oversized_evidence_is_trimmed_and_reported(self):
        """ollama 签约窗口 8192 token，证据按字节预算**整条**裁；丢了几条必须说出来。

        现状是双层静默：这层把 9000 字符硬切，Ollama 再按模型默认 4096 截断——
        两层都不说话，指南/回答质量变差却查不出原因。"""
        self._upload_long_note()
        events = self.events({"question": "PagedAttention 是什么？"})
        retrieval = events[0]
        # 20 条长片段远超 8192 token 窗口的证据预算：必须整条裁，不能硬切。
        self.assertLess(len(retrieval["sources"]), EVIDENCE_CHUNKS)
        self.assertIn("没有参与生成", retrieval["evidence_note"])
        self.assertIn(str(OLLAMA_CONTEXT_TOKENS), retrieval["evidence_note"])
        # 引用编号必须连续——裁掉的是尾部，不是中间。
        labels = [item["label"] for item in retrieval["sources"]]
        self.assertEqual(labels, [f"S{i}" for i in range(1, len(labels) + 1)])
        # 真正发出去的证据不许超预算：这是"签约窗口"的实体含义。
        prompt = self.captured[0][-1]["content"]
        evidence_part = prompt.split("当前检索证据：\n", 1)[1]
        self.assertLessEqual(
            len(evidence_part.encode("utf-8")),
            evidence_budget_bytes(OLLAMA_CONTEXT_TOKENS))

    def test_final_and_replay_carry_the_same_trimmed_sources(self):
        """裁剪要贯穿事件流、落库与重放：翻旧的答案看到的来源和刚答完的一致，
        不能实时看到 5 条、翻旧的又变回 8 条。"""
        self._upload_long_note()
        events = self.events({"question": "PagedAttention 是什么？"})
        final = events[-1]
        self.assertEqual(len(final["sources"]), len(events[0]["sources"]))
        loaded = self.client.get(
            f"/api/v1/conversations/{self.conversation['id']}").json()
        answer = loaded["messages"][-1]
        self.assertEqual(len(answer["sources"]), len(final["sources"]))
        # 说明也必须从库里读回来：实时帧里有不算数，翻旧的还得说同样的话。
        self.assertEqual(answer["evidence_note"], events[0]["evidence_note"])

    def test_deepseek_path_keeps_the_full_window(self):
        """DeepSeek 窗口大得多：没有证据被裁、也没有裁剪说明——预算只属于
        ollama 分支，不能把 deepseek 的证据也砍了。"""
        store = MemoryCredentialStore()
        store.set_deepseek("sk-test-00000000")
        self.app.state.credentials = store
        self.client.patch("/api/v1/settings", json={"provider": "deepseek"})
        self._upload_long_note()
        events = self.events({"question": "PagedAttention 是什么？"})
        self.assertEqual(len(events[0]["sources"]), EVIDENCE_CHUNKS)
        self.assertIsNone(events[0].get("evidence_note"))


class FitEvidenceTests(unittest.TestCase):
    """证据按模型窗口预算整条裁。裁是减法不是切：标签要么在要么不在，
    半条证据会让界面出现一个模型从没见过的引用编号。"""

    def _results(self, count, char_count=800):
        return [{"title": f"笔记{index}", "heading_path": "第一章 > 第二节",
                 "text": "证" * char_count} for index in range(1, count + 1)]

    def test_oversized_evidence_drops_whole_tail_chunks(self):
        # 一条 800 个中文字 ≈ 2400+ 字节（加头行），6000 字节只装得下 2 条。
        kept, dropped = fit_evidence(self._results(5), 6000)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 3)

    def test_within_budget_drops_nothing(self):
        kept, dropped = fit_evidence(self._results(5), 10 ** 9)
        self.assertEqual((len(kept), dropped), (5, 0))

    def test_budget_counts_utf8_bytes_not_characters(self):
        """同样 1000 个"字符"，英文 ≈ 1000 字节装得下，中文 ≈ 3000 字节装不下
        ——字符口径对中文会高估可装条数，正好把窗口撑爆。"""
        ascii_one = [{"title": "a", "heading_path": "h", "text": "a" * 1000}]
        cjk_one = [{"title": "一", "heading_path": "一", "text": "一" * 1000}]
        self.assertEqual(len(fit_evidence(ascii_one, 1200)[0]), 1)
        # 装不下也要保底 1 条：半份证据仍好过空手，空手该走 no_evidence 分支。
        self.assertEqual(len(fit_evidence(cjk_one, 1200)[0]), 1)

    def test_reduction_note_says_both_counts_and_the_window(self):
        note = evidence_reduction_note(12, 5, OLLAMA_CONTEXT_TOKENS)
        self.assertIn("12", note)
        self.assertIn("5", note)
        self.assertIn(str(OLLAMA_CONTEXT_TOKENS), note)
        self.assertIn("没有参与生成", note)


class OllamaContextWindowTests(unittest.TestCase):
    def test_ollama_client_signs_a_context_window(self):
        """不显式传 num_ctx，Ollama 就用模型默认（实测 4096），证据会被**静默**
        截断——不报错、不说明，指南质量变差却说不出为什么。窗口必须是产品的
        签约值，不是碰运气。"""
        settings = {"provider": "ollama", "ollama_model": "qwen2.5:7b",
                    "ollama_base_url": "http://127.0.0.1:11434"}
        client = default_chat_client(settings, MemoryCredentialStore())
        self.assertEqual(client.generation_options.get("num_ctx"), OLLAMA_CONTEXT_TOKENS)

    def test_evidence_budget_leaves_room_for_the_answer(self):
        # 预算 = 窗口 - 系统与主题裕量 - 回答预留，再折算字节。回答没有预留的话，
        # 证据把窗口塞满，模型一个字都吐不出来或被截成半句。
        self.assertEqual(evidence_budget_bytes(OLLAMA_CONTEXT_TOKENS),
                         (OLLAMA_CONTEXT_TOKENS - PROMPT_OVERHEAD_TOKENS
                          - OUTPUT_RESERVE_TOKENS) * BYTES_PER_TOKEN)


if __name__ == "__main__":
    unittest.main()
