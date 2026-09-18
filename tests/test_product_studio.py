import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from src.product.app import create_product_app, current_settings
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager
from src.product.studio import StudioService, backlink_report, build_mindmap


def source(number, title, heading_path, text="正文"):
    """一条与 `materials.retrieve()` 同形的来源，用来单测纯函数。"""
    return {"chunk_id": f"c{number}", "document_id": f"d{number}", "version_id": f"v{number}",
            "title": title, "media_type": "markdown", "heading_path": heading_path,
            "locator": {"kind": "chunk"}, "preview": text, "text": text,
            "score": 1.0, "matched_tokens": [], "channels": {}}


class BacklinkReportTests(unittest.TestCase):
    """`backlink_report` 是这个模块的核心产物——产出功能唯一的诚实分数。"""

    def test_a_line_with_a_valid_label_counts_as_sourced(self):
        report = backlink_report("分页管理 KV Cache [S1]。\n显存碎片减少。", {"S1"})

        self.assertEqual(report["assertions"], 2)
        self.assertEqual(report["with_source"], 1)
        self.assertEqual(report["hit_rate"], 0.5)
        self.assertEqual(report["missing_count"], 1)
        self.assertEqual(report["cited_labels"], ["S1"])

    def test_headings_and_code_blocks_are_not_assertions(self):
        """标题是结构、代码是引用，都不该被算成"没带来源的句子"。

        这条是防虚低：如果把标题算进去，任何一篇正常带标题的指南命中率都会被
        无理由拉低，指标就废了。
        """
        content = "# 标题\n\n## 小节\n\n真结论 [S1]。\n\n```python\nx = 1\n```\n"

        report = backlink_report(content, {"S1"})

        self.assertEqual(report["assertions"], 1)
        self.assertEqual(report["hit_rate"], 1.0)
        self.assertEqual(report["missing_count"], 0)

    def test_an_empty_document_reports_no_rate_rather_than_a_perfect_one(self):
        """没有断言行时必须是 None。

        报 1.0 会让"模型什么都没写出来"看起来像满分——这正是指标最容易说谎的
        地方，所以单独固定住。
        """
        for content in ("", "\n\n", "# 只有一个标题\n"):
            with self.subTest(content=content):
                report = backlink_report(content, {"S1"})
                self.assertEqual(report["assertions"], 0)
                self.assertIsNone(report["hit_rate"])

    def test_a_label_that_does_not_exist_counts_as_missing(self):
        """引用了不存在的编号比缺来源更糟：它看起来像有来源。"""
        report = backlink_report("结论 [S9]。", {"S1", "S2"})

        self.assertEqual(report["with_source"], 0)
        self.assertEqual(report["hit_rate"], 0.0)
        self.assertEqual(report["invalid_labels"], ["S9"])
        self.assertEqual(report["missing_count"], 1)

    def test_a_line_citing_several_labels_counts_once(self):
        """按**行**计数，不是按编号计数：一句引三个来源仍只是一句带来源的断言。"""
        report = backlink_report("结论 [S2][S1][S3]。", {"S1", "S2", "S3"})

        self.assertEqual(report["assertions"], 1)
        self.assertEqual(report["with_source"], 1)
        # 编号按数字序，不是按出现序——否则界面上 S10 会排在 S2 前面
        self.assertEqual(report["cited_labels"], ["S1", "S2", "S3"])

    def test_missing_lines_are_reported_with_their_line_numbers(self):
        report = backlink_report("# 标题\n\n有来源 [S1]。\n没来源的一句。", {"S1"})

        self.assertEqual(report["missing"], [{"line": 4, "text": "没来源的一句。"}])


class MindmapTests(unittest.TestCase):
    """思维导图零模型调用，所以它可以被完全确定性地测试。"""

    def test_the_tree_follows_the_heading_path(self):
        mindmap = build_mindmap("推理优化", [source(1, "推理.md", "推理.md > 推理 > PagedAttention")])

        document = mindmap["tree"]["children"][0]
        section = document["children"][0]
        leaf = section["children"][0]
        self.assertEqual(mindmap["tree"]["label"], "推理优化")
        self.assertEqual([document["label"], section["label"], leaf["label"]],
                         ["推理.md", "推理", "PagedAttention"])
        self.assertEqual(leaf["sources"], ["S1"])

    def test_chunks_under_the_same_heading_share_one_node(self):
        """一个知识点被几段讲到时该合并成一个节点、编号挂在一起。"""
        mindmap = build_mindmap("推理优化", [
            source(1, "推理.md", "推理.md > 推理 > PagedAttention"),
            source(2, "推理.md", "推理.md > 推理 > PagedAttention")])

        leaf = mindmap["tree"]["children"][0]["children"][0]["children"][0]
        self.assertEqual(leaf["sources"], ["S1", "S2"])
        self.assertEqual(mindmap["linked_chunks"], 2)
        # 两个片段走同一条路径 ⇒ 只有「文档 + 推理 + PagedAttention」三个节点
        self.assertEqual(mindmap["node_count"], 4)

    def test_the_document_title_is_not_duplicated(self):
        """pdf 的 heading_path 是 `标题 > 第 N 页`，不能出现"标题 > 标题"。"""
        mindmap = build_mindmap("主题", [source(1, "论文.pdf", "论文.pdf > 第 3 页")])

        document = mindmap["tree"]["children"][0]
        self.assertEqual(document["label"], "论文.pdf")
        self.assertEqual([child["label"] for child in document["children"]], ["第 3 页"])

    def test_every_node_is_supported_by_at_least_one_chunk(self):
        """节点必须有片段支撑：没有编号的中间节点只能是"有子节点的分组"。"""
        mindmap = build_mindmap("主题", [
            source(1, "a.md", "a.md > 甲"), source(2, "b.md", "b.md > 乙")])

        def walk(node):
            if node["level"]:
                self.assertTrue(node["sources"] or node["children"], node["label"])
            for child in node["children"]:
                walk(child)

        walk(mindmap["tree"])
        self.assertEqual(mindmap["node_count"], 5)

    def test_it_reports_coverage_as_constructional_not_as_a_score(self):
        """导图必然"每个节点都带编号"，报成命中率会误导用户去比较两个不同的东西。"""
        mindmap = build_mindmap("主题", [source(1, "a.md", "a.md > 甲")])

        self.assertNotIn("hit_rate", mindmap)
        self.assertIn("构造结果", mindmap["coverage_note"])

    def test_mermaid_export_survives_labels_that_break_the_syntax(self):
        mindmap = build_mindmap("主题 (一)", [source(1, "a.md", "a.md > 甲[乙](丙)")])

        mermaid = mindmap["mermaid"]
        self.assertTrue(mermaid.startswith("mindmap\n"))
        self.assertNotIn("(", mermaid.split("\n")[1].replace("root((", "").replace("))", ""))
        self.assertIn("·", mermaid)


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


def build_harness(test_case, upload=True, client_chunks=None):
    """建一个带假模型的应用，返回 `(client, service, captured)`。

    服务层测试与 API 层测试共用这一个搭建过程：分成两份的话，"API 测试里的应用"
    会慢慢长成和"服务测试里的应用"不一样的东西，而它们本该是同一个。
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
    service = StudioService(
        app.state.database, app.state.materials,
        lambda: current_settings(app.state.database), MemoryCredentialStore(),
        client_factory=factory)
    return client, service, captured, app


class StudioServiceTests(unittest.TestCase):
    def build(self, upload=True, client_chunks=None):
        client, service, captured, app = build_harness(self, upload, client_chunks)
        self.client, self.service, self.captured, self.app = client, service, captured, app
        return service

    def events(self, kind="guide", topic="分页管理"):
        return list(self.service.stream(topic, kind))

    def test_a_guide_reports_how_many_sentences_carried_a_source(self):
        self.build()

        events = self.events()
        final = events[-1]

        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["backlink"]["assertions"], 3)
        self.assertEqual(final["backlink"]["with_source"], 2)
        self.assertAlmostEqual(final["backlink"]["hit_rate"], 2 / 3)
        self.assertEqual(final["backlink"]["missing_count"], 1)

    def test_the_guide_is_written_from_the_retrieved_fragments(self):
        """指南必须真的建立在片段上，而不是像一篇通稿。"""
        self.build()

        self.events()

        prompt = self.captured[0][-1]["content"]
        self.assertIn("分页管理 KV Cache", prompt)
        self.assertIn("[S1]", prompt)

    def test_the_token_stream_is_forwarded_for_the_guide(self):
        self.build(client_chunks=("甲", "乙"))

        events = self.events()

        self.assertEqual([event["text"] for event in events if event["type"] == "token"], ["甲", "乙"])
        self.assertEqual(events[-1]["content"], "甲乙")

    def test_a_stored_guide_recomputes_its_rate_instead_of_storing_it(self):
        """命中率按正文现算：落库的分数会和正文各自演化，而正文才是唯一事实来源。"""
        service = self.build()
        artifact_id = self.events()[-1]["artifact_id"]

        stored = service.get_artifact(artifact_id)

        self.assertEqual(stored["kind"], "guide")
        self.assertAlmostEqual(stored["backlink"]["hit_rate"], 2 / 3)
        self.assertEqual([item["label"] for item in stored["sources"]], ["S1"])
        # 来源形状必须与消息来源逐键一致，否则前端要为产出再写一套判断
        self.assertEqual(set(stored["sources"][0]),
                         {"label", "chunk_id", "document_id", "version_id", "title", "media_type",
                          "heading_path", "locator", "preview", "score", "matched_tokens",
                          "channels", "origin"})
        self.assertEqual(stored["sources"][0]["locator"]["kind"], "markdown")
        self.assertEqual(stored["sources"][0]["origin"], "note")

    def test_a_mindmap_is_built_without_asking_a_model(self):
        service = self.build()

        events = self.events(kind="mindmap")
        final = events[-1]

        self.assertEqual([event["type"] for event in events], ["retrieval", "final"])
        self.assertEqual(self.captured, [])
        self.assertEqual(final["mindmap"]["tree"]["label"], "分页管理")
        stored = service.get_artifact(final["artifact_id"])
        self.assertIsNotNone(stored["mindmap"])
        self.assertNotIn("backlink", stored)

    def test_no_evidence_fails_the_artifact_instead_of_writing_an_empty_one(self):
        """没有片段时应当明确失败并留一条可查看的记录，而不是产出一篇空指南。"""
        service = self.build(upload=False)

        final = self.events()[-1]

        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error"], "no_evidence")
        stored = service.get_artifact(final["artifact_id"])
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["error_code"], "no_evidence")
        self.assertIsNone(stored["backlink"]["hit_rate"])

    def test_a_generation_failure_is_recorded_with_a_usable_message(self):
        self.build(client_chunks=())

        class Broken:
            def stream_chat(self, messages, cancel_event=None):
                raise RuntimeError("deepseek_not_configured")
                yield  # pragma: no cover

        self.service.client_factory = lambda settings: Broken()
        events = self.events()
        final = events[-1]

        self.assertEqual(final["type"], "error")
        self.assertEqual(final["error"], "deepseek_not_configured")
        self.assertEqual(final["message"], "尚未配置 DeepSeek API Key，请前往设置。")
        self.assertEqual(self.service.get_artifact(final["artifact_id"])["status"], "failed")

    def test_artifacts_left_running_by_a_restart_are_marked_stopped(self):
        """进程重启后不该留下永远"生成中"的记录，界面会一直转圈。"""
        service = self.build()
        self.service._create("art_stuck", "guide", "旧主题", "旧主题")

        StudioService(self.app.state.database, self.app.state.materials,
                      lambda: current_settings(self.app.state.database), MemoryCredentialStore())

        self.assertEqual(service.get_artifact("art_stuck")["status"], "stopped")
        self.assertEqual(service.get_artifact("art_stuck")["error_code"], "application_restarted")

    def test_deleting_an_artifact_takes_its_sources_with_it(self):
        service = self.build()
        artifact_id = self.events()[-1]["artifact_id"]

        self.assertEqual(service.delete_artifact(artifact_id), {"deleted": True})

        self.assertEqual(service.list_artifacts(), [])
        with self.assertRaises(KeyError):
            service.get_artifact(artifact_id)

    def test_an_empty_topic_is_refused_before_anything_is_written(self):
        service = self.build()

        with self.assertRaises(ValueError):
            list(service.stream("   ", "guide"))

        self.assertEqual(service.list_artifacts(), [])


class StudioApiTests(unittest.TestCase):
    """走真实 HTTP：服务能跑通不等于界面上点得到，所以这里的断言都真的发请求。"""

    def build(self, upload=True, client_chunks=None):
        self.client, self.service, self.captured, self.app = build_harness(
            self, upload, client_chunks)

    def stream(self, topic="分页管理", kind="guide"):
        response = self.client.post("/api/v1/artifacts/stream",
                                    json={"topic": topic, "kind": kind})
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.text.splitlines() if line]

    def test_the_stream_ends_with_a_backlink_report(self):
        self.build()

        events = self.stream()
        final = events[-1]

        self.assertEqual([event["type"] for event in events],
                         ["retrieval", "token", "token", "token", "final"])
        self.assertEqual(events[0]["sources"][0]["label"], "S1")
        self.assertEqual(final["backlink"]["missing_count"], 1)

    def test_an_artifact_can_be_listed_read_and_deleted(self):
        self.build()
        artifact_id = self.stream()[-1]["artifact_id"]

        listed = self.client.get("/api/v1/artifacts").json()["artifacts"]
        self.assertEqual([item["id"] for item in listed], [artifact_id])
        self.assertEqual(listed[0]["kind"], "guide")

        detail = self.client.get(f"/api/v1/artifacts/{artifact_id}").json()
        self.assertAlmostEqual(detail["backlink"]["hit_rate"], 2 / 3)
        self.assertEqual(detail["sources"][0]["label"], "S1")

        self.assertEqual(self.client.delete(f"/api/v1/artifacts/{artifact_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/artifacts/{artifact_id}").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/artifacts").json()["artifacts"], [])

    def test_a_blank_topic_is_refused_before_the_stream_is_opened(self):
        """非法请求必须是 4xx，不能变成一个"莫名其妙断掉"的流。"""
        self.build()

        response = self.client.post("/api/v1/artifacts/stream",
                                    json={"topic": "   ", "kind": "guide"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "invalid_artifact")
        self.assertEqual(self.client.get("/api/v1/artifacts").json()["artifacts"], [])

    def test_an_unknown_kind_is_refused(self):
        self.build()

        response = self.client.post("/api/v1/artifacts/stream",
                                    json={"topic": "分页管理", "kind": "pptx"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "invalid_request")

    def test_a_missing_artifact_reports_a_404_not_a_500(self):
        self.build()

        self.assertEqual(self.client.get("/api/v1/artifacts/art_missing").status_code, 404)
        self.assertEqual(self.client.delete("/api/v1/artifacts/art_missing").status_code, 404)
        self.assertEqual(
            self.client.post("/api/v1/artifacts/art_missing/stop").status_code, 404)

    def test_diagnostics_count_the_artifacts(self):
        self.build()
        self.stream()

        studio = self.client.get("/api/v1/system/diagnostics").json()["studio"]

        self.assertTrue(studio["available"])
        self.assertEqual(studio["by_status"], {"complete": 1})

    def test_an_empty_library_fails_the_artifact_visibly(self):
        """没有资料时用户会先看到来源区是空的，再看到明确的失败原因。"""
        self.build(upload=False)

        final = self.stream()[-1]

        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error"], "no_evidence")
        self.assertIn("没有找到", final["message"])


if __name__ == "__main__":
    unittest.main()
