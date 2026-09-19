import json
import threading
import unittest

from src.product.chat import OLLAMA_CONTEXT_TOKENS
from src.product.studio import (STUDIO_EVIDENCE_CHUNKS, backlink_report, build_mindmap,
                                render_json_guide)
from tests.product_harness import (build_harness, new_service, source)


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

    def test_a_list_item_is_still_an_assertion_but_listed_without_its_marker(self):
        """指南正文是列表（一行一句），判定与展示要分开：

        - 判定：列表项**算断言**，且仍是一行一句 ⇒ 命中率口径和以前一样是句子级；
        - 展示：清单里挂一个 `- ` 是噪声，用户要读的是句子本身。
        """
        report = backlink_report("## 小节\n- 有来源 [S1]。\n- 没来源的一句。", {"S1"})

        self.assertEqual(report["assertions"], 2)
        self.assertEqual(report["with_source"], 1)
        self.assertEqual(report["missing"], [{"line": 3, "text": "没来源的一句。"}])


class MindmapTests(unittest.TestCase):
    """思维导图零模型调用，所以它可以被完全确定性地测试。"""

    def test_the_tree_follows_the_heading_path(self):
        """夹具照真实数据来：md 的 `heading_path` 首段是**文档里的一级标题**（不带
        扩展名），而 `title` 是文件名（带扩展名）。夹具写成 `"推理.md > 推理 > ..."`
        时首段与文件名逐字相同，那条"要不要折叠这层"的判据永远碰不到真实情形——
        线上于是多出一层同义节点（学习记录 49）。
        """
        mindmap = build_mindmap("推理优化", [source(1, "推理.md", "推理 > PagedAttention")])

        document = mindmap["tree"]["children"][0]
        section = document["children"][0]
        self.assertEqual(mindmap["tree"]["label"], "推理优化")
        self.assertEqual([document["label"], section["label"]], ["推理.md", "PagedAttention"])
        self.assertEqual(section["sources"], ["S1"])

    def test_chunks_under_the_same_heading_share_one_node(self):
        """一个知识点被几段讲到时该合并成一个节点、编号挂在一起。"""
        mindmap = build_mindmap("推理优化", [
            source(1, "推理.md", "推理 > PagedAttention"),
            source(2, "推理.md", "推理 > PagedAttention")])

        leaf = mindmap["tree"]["children"][0]["children"][0]
        self.assertEqual(leaf["sources"], ["S1", "S2"])
        self.assertEqual(mindmap["linked_chunks"], 2)
        # 两个片段走同一条路径 ⇒ 只有「文档 + PagedAttention」两个节点
        self.assertEqual(mindmap["node_count"], 3)

    def test_the_document_title_is_not_duplicated(self):
        """两种真实形状都不能出现"标题 > 标题"。

        pdf 的 heading_path 首段直接就是文件名；无 H1 的 md 也是（`markdown_chunks`
        在没有一级标题时拿文件名兜底）。两种情况首段与 `title` 都可能逐字相同或只差
        一个扩展名，判据必须都能命中。
        """
        for title, path, expected in (
                ("论文.pdf", "论文.pdf > 第 3 页", "第 3 页"),
                ("无标题.md", "无标题.md > 甲", "甲"),
                ("推理.md", "推理 > PagedAttention", "PagedAttention"),
                ("笔记.md", "笔记.markdown > 乙", "乙")):
            with self.subTest(title=title, path=path):
                mindmap = build_mindmap("主题", [source(1, title, path)])
                document = mindmap["tree"]["children"][0]
                self.assertEqual(document["label"], title)
                self.assertEqual([child["label"] for child in document["children"]], [expected])

    def test_a_document_heading_that_differs_from_the_file_name_is_kept(self):
        """文档里的一级标题与文件名不同名时**不能折叠**：那是两个不同的信息，
        折叠等于把文档自己的标题丢掉。"""
        mindmap = build_mindmap("主题", [source(1, "note1.md", "检索与融合 > 融合")])

        document = mindmap["tree"]["children"][0]
        self.assertEqual([child["label"] for child in document["children"]], ["检索与融合"])
        self.assertEqual(document["children"][0]["children"][0]["label"], "融合")

    def test_a_group_node_reports_the_labels_below_it(self):
        """分组节点自己没有片段，但要能说出"这一组包含哪些编号"。

        界面拿它把"这篇笔记贡献了 S1–S4"标出来。父节点只显示 S1 是骗人的（它下面
        还有别的），而把每个编号在每一层祖先上重复一遍又太吵——所以只在节点自己
        没有片段时用它。
        """
        mindmap = build_mindmap("主题", [
            source(1, "a.md", "a.md > 甲"), source(2, "a.md", "a.md > 乙"),
            source(3, "b.md", "b.md > 丙")])

        root = mindmap["tree"]
        first = root["children"][0]
        self.assertEqual(root["aggregate"], ["S1", "S2", "S3"])
        self.assertEqual(first["aggregate"], ["S1", "S2"])
        self.assertEqual(first["sources"], [])
        self.assertEqual(first["children"][0]["aggregate"], ["S1"])
        # 有自己片段的节点不受影响：aggregate 只做加法，不改 sources 的口径。
        self.assertEqual(mindmap["linked_chunks"], 3)

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


class RenderJsonGuideTests(unittest.TestCase):
    """渲染器是纯函数：只换形状，绝不补链——这是 36 号否决"程序补链"后守住的线。"""

    def test_renders_headings_and_sentences_with_model_given_sources(self):
        """每句一个列表项——这是渲染形态的一部分，不是排版偏好。

        句子若渲染成普通行，Markdown 会把连续的行合并成一个段落（软换行渲染成
        空格），界面上一整篇指南挤成一坨。所以这里钉住 `- `：它同时保证"一行一句"
        （命中率按行数算，口径不受影响）。实测形态见学习记录 49。
        """
        content, error = render_json_guide(
            '{"sections": [{"h": "分页"}, {"s": "KV Cache 减少碎片。", "src": [1, 3]},'
            ' {"s": "没有来源的一句。"}]}')

        self.assertIsNone(error)
        self.assertEqual(content, "## 分页\n- KV Cache 减少碎片。 [S1][S3]\n- 没有来源的一句。")

    def test_a_markdown_fence_from_prompt_constrained_models_is_stripped(self):
        """DeepSeek 走提示词约束，可能带 ```json 围栏；Ollama 文法约束不会有。"""
        raw = '```json\n{"sections": [{"s": "一句。", "src": [2]}]}\n```'

        self.assertEqual(render_json_guide(raw), ("- 一句。 [S2]", None))

    def test_a_non_integer_source_is_dropped_not_guessed(self):
        """src 里的非整数直接丢掉：渲染器不猜"它大概是 S1"。"""
        content, error = render_json_guide(
            '{"sections": [{"s": "一句。", "src": ["1", 2, "x"]}]}')

        self.assertIsNone(error)
        self.assertEqual(content, "- 一句。 [S2]")

    def test_repeated_section_keys_are_merged_instead_of_losing_all_but_the_last(self):
        """实测退化形状：同一个对象里出现多个 `sections` 键（学习记录 49）。

        qwen2.5:7b 在长指南上会这样吐：`{"sections": [第一节], "sections": [第二节], ...}`。
        标准解析只留最后一个，于是十二句陈述剩两句——**而命中率仍是 100%**（剩下的
        两句都带来源）。数字对、内容少、界面看不出异常，所以必须按顺序拼回来。
        """
        content, error = render_json_guide(
            '{"sections": [{"h": "融合"}, {"s": "第一句。", "src": [1]}],'
            ' "sections": [{"h": "重排"}, {"s": "第二句。", "src": [2]}],'
            ' "sections": [{"h": "评测"}, {"s": "第三句。", "src": [3]}]}')

        self.assertIsNone(error)
        self.assertEqual(content,
                         "## 融合\n- 第一句。 [S1]\n## 重排\n- 第二句。 [S2]\n## 评测\n- 第三句。 [S3]")

    def test_several_top_level_objects_are_concatenated(self):
        """另一种退化：模型收不住，把 JSON 对象一个接一个写下去。"""
        content, error = render_json_guide(
            '{"sections": [{"s": "第一句。", "src": [1]}]}\n'
            '{"sections": [{"s": "第二句。", "src": [2]}]}')

        self.assertIsNone(error)
        self.assertEqual(content, "- 第一句。 [S1]\n- 第二句。 [S2]")

    def test_a_truncated_tail_keeps_what_was_already_complete(self):
        """尾部半截（模型写崩或被截断）不该把前面已经拿到的部分一起丢掉。"""
        content, error = render_json_guide(
            '{"sections": [{"s": "完整的一句。", "src": [1]}]}{"sections": [{"h": "半')

        self.assertIsNone(error)
        self.assertEqual(content, "- 完整的一句。 [S1]")

    def test_a_bare_top_level_array_is_still_accepted(self):
        """提示词约束的模型可能直接给数组，不带 `sections` 外壳。"""
        content, error = render_json_guide('[{"s": "一句。", "src": [1]}]')

        self.assertIsNone(error)
        self.assertEqual(content, "- 一句。 [S1]")

    def test_invalid_output_reports_a_reason_instead_of_faking_content(self):
        for raw, reason in (("不是 JSON", "json_parse_failed"),
                            ('{"no": "sections"}', "json_shape_failed"),
                            ('{"sections": []}', "json_empty"),
                            ('{"sections": [42]}', "json_empty")):
            with self.subTest(raw=raw):
                content, error = render_json_guide(raw)
                self.assertIsNone(content)
                self.assertTrue(error.startswith(reason), error)


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
        self.build(client_chunks=('{"sections": [{"s": "内容到位。", "src": [1]}',
                                  ', {"s": "第二句。", "src": [1]}]}'))

        events = self.events()

        self.assertEqual([event["text"] for event in events if event["type"] == "token"],
                         ['{"sections": [{"s": "内容到位。", "src": [1]}',
                          ', {"s": "第二句。", "src": [1]}]}'])
        # final 里是**渲染后**的正文：token 是原始 JSON 碎片，用户看到的不是它。
        self.assertEqual(events[-1]["content"], "- 内容到位。 [S1]\n- 第二句。 [S1]")

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

    def test_unparseable_output_fails_the_artifact_instead_of_faking_a_guide(self):
        """模型没吐 JSON（DeepSeek 走提示词约束，可能跑题）：如实失败，原始输出留在
        正文里供排查，而不是把一段散文当指南存进去假装成功。"""
        self.build(client_chunks=("我按你的主题写一篇关于", "分页管理的文章如下："))

        final = self.events()[-1]

        self.assertEqual(final["type"], "error")
        self.assertEqual(final["error"], "invalid_model_output")
        self.assertIn("无法解析", final["message"])
        stored = self.service.get_artifact(final["artifact_id"])
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["error_code"], "invalid_model_output")
        self.assertIn("分页管理的文章", stored["content"])

    def test_a_cancelled_guide_renders_what_completed_and_stores_the_rest_as_is(self):
        """停止的两副面孔：JSON 已完整写出的部分照常渲染；渲染不出的半截如实存原文。
        详情页要看到模型真正写到哪，而不是一份像样的假正文。"""
        chunks = ('{"sections": [{"s": "第一句。", "src": [1]}',
                  ', {"s": "第二句。", "src": [1]}]}')

        class Cancelling:
            def __init__(self, cancel_after):
                self.cancel_after = cancel_after

            def stream_chat(self, messages, cancel_event=None):
                for index, chunk in enumerate(chunks):
                    yield chunk
                    if index == self.cancel_after:
                        cancel_event.set()

        service = self.build()
        # 吐完整个 JSON 才取消：能渲染，正文就是渲染结果。
        cancel = threading.Event()
        service.client_factory = lambda settings: Cancelling(1)
        stopped = list(service.stream("分页管理", "guide", cancel_event=cancel))[-1]
        self.assertEqual(stopped["type"], "stopped")
        self.assertEqual(stopped["content"], "- 第一句。 [S1]\n- 第二句。 [S1]")
        self.assertEqual(stopped["backlink"]["with_source"], 2)
        self.assertEqual(service.get_artifact(stopped["artifact_id"])["status"], "stopped")

        # 只吐了第一片就取消：渲染不出正文，存的就是原始 JSON 碎片。
        cancel = threading.Event()
        service.client_factory = lambda settings: Cancelling(0)
        stopped = list(service.stream("分页管理", "guide", cancel_event=cancel))[-1]
        self.assertEqual(stopped["type"], "stopped")
        self.assertEqual(stopped["content"], chunks[0])
        self.assertEqual(service.get_artifact(stopped["artifact_id"])["status"], "stopped")

    def test_json_mode_is_requested_only_on_the_ollama_path(self):
        """format=json 由 OllamaClient 带进 payload，且只在产出这条线开：问答共用同一个
        client_factory，全局开了会把问答输出也变成 JSON。"""
        self.build()
        created = []
        inner = self.service.client_factory

        def factory(settings):
            client = inner(settings)
            created.append(client)
            return client

        self.service.client_factory = factory
        self.events()
        self.assertTrue(created[0].json_mode)

        self.client.patch("/api/v1/settings", json={"provider": "deepseek"})
        self.events()
        self.assertFalse(created[1].json_mode)

    def test_artifacts_left_running_by_a_restart_are_marked_stopped(self):
        """进程重启后不该留下永远"生成中"的记录，界面会一直转圈。"""
        service = self.build()
        self.service._create("art_stuck", "guide", "旧主题", "旧主题")

        new_service(self.app)

        self.assertEqual(service.get_artifact("art_stuck")["status"], "stopped")
        self.assertEqual(service.get_artifact("art_stuck")["error_code"], "application_restarted")

    def test_deleting_an_artifact_takes_its_sources_with_it(self):
        service = self.build()
        artifact_id = self.events()[-1]["artifact_id"]

        self.assertEqual(service.delete_artifact(artifact_id), {"deleted": True})

        self.assertEqual(service.list_artifacts(), [])
        with self.assertRaises(KeyError):
            service.get_artifact(artifact_id)

    def test_batch_deletion_skips_the_running_one_and_counts_files(self):
        """批量删产出：正在生成的跳过（不能让生成线程撞外键），文件删几个报几个。

        `files_removed` 数的是**真删掉的文件**：`unlink(missing_ok=True)` 会把
        "文件本来就不在"也算一次，那样报出来的数字就没法核实了。
        """
        service = self.build()
        artifact_id = self.events()[-1]["artifact_id"]
        title = service.get_artifact(artifact_id)["title"]
        png = service._infographic_paths(title, artifact_id)["png"]
        png.write_bytes(b"png")
        # "还在生成" = 记录存在 + 在 _active 里（值是 cancel 事件，占位 None 即可）。
        # 只塞 _active 不建记录的话，它会先撞上 not_found，根本轮不到 busy 分支。
        self.service._create("art-busy", "guide", "主题", "主题")
        service._active["art-busy"] = None

        result = service.delete_artifacts([artifact_id, "art-busy", "art-ghost"])

        self.assertEqual(result["deleted"], 1)
        self.assertEqual(result["files_removed"], 1)
        self.assertFalse(png.exists())
        skipped = {item["id"]: item for item in result["skipped"]}
        self.assertEqual(set(skipped), {"art-busy", "art-ghost"})
        self.assertEqual(skipped["art-busy"]["code"], "busy")
        self.assertEqual(skipped["art-busy"]["label"], "主题")
        self.assertIn("还在生成", skipped["art-busy"]["reason"])
        self.assertEqual(skipped["art-ghost"]["code"], "not_found")
        # 数据库里只剩那份"还在生成"的产出
        self.assertEqual([a["id"] for a in service.list_artifacts()], ["art-busy"])

    def test_single_deletion_still_refuses_a_running_artifact(self):
        """单选接口的"还在生成"语义不变：报 RuntimeError 而不是悄悄跳过。"""
        service = self.build()
        self.service._create("art-busy", "guide", "主题", "主题")
        service._active["art-busy"] = None

        with self.assertRaises(RuntimeError):
            service.delete_artifact("art-busy")
        self.assertEqual([a["id"] for a in service.list_artifacts()], ["art-busy"])

    def test_batch_deletion_of_a_running_artifact_via_the_api_reports_busy(self):
        """走 API 时 busy 不该变成 500：正常 200 + skipped。

        harness 里 service 与 app.state.studio 是两个实例（各有自己的 _active），
        所以"正在生成"必须塞到**路由真正用的那个**实例上。
        """
        self.build()
        self.service._create("art-busy", "guide", "主题", "主题")
        self.client.app.state.studio._active["art-busy"] = None

        response = self.client.post("/api/v1/artifacts/delete",
                                    json={"ids": ["art-busy", "art-ghost"]})

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["deleted"], 0)
        self.assertEqual({item["code"] for item in result["skipped"]}, {"busy", "not_found"})

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

    def test_oversized_evidence_is_trimmed_and_reported_for_ollama(self):
        """ollama 签约窗口装不下 12 条长片段：整条裁、来源只留**真正参与生成**的、
        final 里说明裁剪。现状是 12000 字符硬切 + Ollama 按 4096 静默截断，两层
        都不说话。"""
        self.build()
        body = "\n\n".join(
            f"## 分页 {index}\n\nPagedAttention 的第 {index} 个要点：{'填' * 700}"
            for index in range(1, 21))
        self.client.post("/api/v1/documents/upload", files={
            "file": ("超长笔记.md", f"# 超长笔记\n\n{body}".encode(), "text/markdown")})

        events = self.stream(topic="分页 PagedAttention")
        retrieval, final = events[0], events[-1]

        self.assertLess(len(retrieval["sources"]), STUDIO_EVIDENCE_CHUNKS)
        self.assertIn("没有参与生成", final["evidence_note"])
        self.assertIn(str(OLLAMA_CONTEXT_TOKENS), final["evidence_note"])
        self.assertEqual(len(final["sources"]), len(retrieval["sources"]))
        # 落库的来源也是裁剪后的：详情页不能出现一个模型没看过的 [S12]。
        detail = self.client.get(f"/api/v1/artifacts/{final['artifact_id']}").json()
        self.assertEqual(len(detail["sources"]), len(retrieval["sources"]))

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
