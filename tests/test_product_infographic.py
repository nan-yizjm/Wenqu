"""P4 图片产出：信息图模型、无头渲染封装、导出服务与三条路由。

分四组，各有各的理由：

- `InfographicModelTests` 是纯函数（不碰浏览器、不碰数据库），跑得快，覆盖版面算术
  与"图里不许出现外链"这类**静态可证**的纪律。
- `HeadlessTests` 盯的是 `headless.py` 自己：找不到浏览器要降级、退码非 0、超时、
  参数必须与请求的像素尺寸一致。**只有一条真起浏览器**，其余用替身——因为"没有
  浏览器的那台机器"才是最需要这条路径正确的地方，而它恰恰是最难在本地复现的。
- `InfographicServiceTests` 用**假渲染器**验证服务层：落盘、记录、降级、删产出连
  带删导出物。真渲染的那条单独放 `RealRenderTests`，没有浏览器的机器上跳过。
- `InfographicApiTests` 走真实 HTTP，因为"服务能跑通"不等于"界面上点得到"。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.product import headless, infographic, studio
from src.product.headless import RendererUnavailable, RenderFailed
from tests.product_harness import build_harness, stored_source


def temp_dir(test_case) -> Path:
    """建一个临时目录并登记清理。

    用 `addCleanup` 而不是 `with` 块：有好几条测试要在"用完之后"再断言（比如临时
    文件确实被清干净了），`with` 块里断言不了那件事。
    """
    temporary = tempfile.TemporaryDirectory()
    test_case.addCleanup(temporary.cleanup)
    return Path(temporary.name)


def record(kind="guide", sources=None, backlink=None, mindmap=None, title="分页管理笔记",
           status="complete"):
    return {"id": "art_0123456789abcdef", "kind": kind, "title": title, "topic": "分页管理",
            "status": status, "content": "", "index_version": "idx_20260918",
            "error_code": None, "created_at": "2026-09-18T08:00:00+00:00",
            "completed_at": "2026-09-18T08:01:00+00:00",
            "sources": sources if sources is not None else [
                stored_source("S1", "推理.md", "推理 > PagedAttention"),
                stored_source("S2", "推理.md", "推理 > PagedAttention"),
                stored_source("S3", "缓存.md", "缓存 > 分页"),
            ],
            **({"backlink": backlink} if backlink is not None else {}),
            **({"mindmap": mindmap} if mindmap is not None else {})}


def rate(hit_rate=2 / 3):
    return {"assertions": 3, "with_source": 2, "hit_rate": hit_rate, "cited_labels": ["S1"],
            "invalid_labels": [], "missing_count": 1, "missing": []}


def fake_screenshot(pixels=None, milliseconds=812):
    """假渲染器：写出一个**真 PNG 头**（让 `png_dimensions` 读得出尺寸），返回记录。

    `pixels=(w, h)` 用来假装浏览器无视了 `--window-size`——那是"图能出来但尺寸不对"
    的情形，必须有办法测到。
    """
    def render(html_path, png_path, *, width, height, scale=1, **kwargs):
        size = pixels or (width * scale, height * scale)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(png_bytes(*size))
        return {"browser": "Fake Browser", "browser_path": "fake", "milliseconds": milliseconds,
                "bytes": png_path.stat().st_size, "width": size[0], "height": size[1]}

    return render


def png_bytes(width, height):
    return (headless.PNG_SIGNATURE + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height)
            + b"\x08\x06\x00\x00\x00" + b"\x00" * 16)


class InfographicModelTests(unittest.TestCase):
    def test_documents_are_grouped_in_retrieval_order_with_relative_bars(self):
        """条形按"相对最多的那篇"画：几篇都只有 1 个片段时，按绝对比例会全长一个样。"""
        model = infographic.build_model(record())

        self.assertEqual([item["title"] for item in model["documents"]],
                         ["推理.md", "缓存.md"])
        self.assertEqual([item["count"] for item in model["documents"]], [2, 1])
        self.assertEqual([item["labels"] for item in model["documents"]],
                         [["S1", "S2"], ["S3"]])
        self.assertEqual([item["share"] for item in model["documents"]], [1.0, 0.5])
        self.assertEqual(model["stats"]["chunks"], 3)
        self.assertEqual(model["stats"]["documents"], 2)
        self.assertEqual(model["stats"]["sections"], 2)

    def test_the_origin_layer_is_counted_and_stated(self):
        """P2 的联网层要在这张图上看得见：用户有权知道这份产出的来源里有几条出过网。"""
        model = infographic.build_model(record(sources=[
            stored_source("S1", "推理.md", "推理 > 甲"),
            stored_source("S2", "外部.md", "外部", origin="web")]))

        self.assertEqual(model["stats"]["origin_text"], "笔记 1 · 联网 1")

        html = infographic.render_html(model)
        self.assertIn("来源层级 笔记 1 · 联网 1", html)

    def test_the_subtitle_keeps_the_long_values_short(self):
        """副标题是**单行不换行**的，长的值会把排在最后的「来源层级」整段挤掉。

        真实渲染里踩到过：完整的 36 位索引哈希加上带小数位与时区的 ISO 时间戳占满整行，
        图末尾变成 `来源层级 …`——有信息的那一半反而看不见。所以这两个值上版面时用短
        写法（与产出详情页一致），原始值仍然留在模型里，要完整信息的人取得到。
        """
        long_record = record()
        long_record["index_version"] = "idx_a5eb0601897f4994a177219ad7c34ebb"
        long_record["created_at"] = "2026-09-18T08:55:08.198763+00:00"

        model = infographic.build_model(long_record)
        html = infographic.render_html(model)

        self.assertEqual(model["index_label"], "idx_a5eb")
        self.assertEqual(model["created_label"], "2026-09-18 08:55")
        self.assertNotIn("idx_a5eb0601897f4994a177219ad7c34ebb", html)
        self.assertNotIn("08:55:08.198763", html)
        self.assertIn("索引 idx_a5eb ·", html)
        self.assertIn("生成于 2026-09-18 08:55 ·", html)
        # 来源层级排在最后，正是被挤掉的那一个——必须还在。
        self.assertIn("来源层级 笔记 3", html)
        # 短的是"上版面的写法"，不是丢掉的值。
        self.assertEqual(model["index_version"], "idx_a5eb0601897f4994a177219ad7c34ebb")
        self.assertEqual(model["created_at"], "2026-09-18T08:55:08.198763+00:00")

    def test_only_a_guide_carries_a_backlink_rate(self):
        """导图放的是节点数，不是命中率——它的覆盖率是构造结果，必然接近 1。"""
        guide = infographic.build_model(record(backlink=rate()))
        mindmap = infographic.build_model(record(kind="mindmap", mindmap={"node_count": 7}))

        self.assertEqual(guide["rate"]["value"], "66.7%")
        self.assertEqual(guide["rate"]["label"], "回链命中率")
        self.assertEqual(mindmap["rate"]["value"], "7")
        self.assertEqual(mindmap["rate"]["label"], "结构节点")
        self.assertNotIn("回链命中率", infographic.render_html(mindmap))
        self.assertIn("构造结果", infographic.render_html(mindmap))

    def test_an_empty_guide_shows_a_dash_instead_of_a_perfect_score(self):
        """没有断言行时命中率是 `None`：报 100% 就是数字说谎。"""
        model = infographic.build_model(record(backlink=rate(hit_rate=None)))

        self.assertEqual(model["rate"]["value"], "—")
        self.assertIn("—", infographic.render_html(model))

    def test_the_hit_rate_is_printed_on_the_image(self):
        """图里必须带上这个真分数，而不是只把它留在接口响应里。"""
        html = infographic.render_html(infographic.build_model(record(backlink=rate())))

        self.assertIn("66.7%", html)
        self.assertIn("条陈述带来源 2 / 3", html)
        self.assertIn("缺来源 1 条", html)
        self.assertIn("不是“这句话对不对”", html)

    def test_long_lists_are_cut_but_the_omission_is_stated(self):
        """截断不是问题，悄悄截断才是：必须写出还有多少没画进去。"""
        sources = [stored_source(f"S{number}", f"文档{number}.md", f"标题{number}")
                   for number in range(1, 12)]
        model = infographic.build_model(record(sources=sources))

        self.assertEqual(len(model["documents"]), infographic.MAX_DOCUMENT_ROWS)
        self.assertEqual(len(model["headings"]), infographic.MAX_HEADING_ROWS)
        self.assertEqual(model["omitted"], {"documents": 4, "headings": 3})

        html = infographic.render_html(model)
        self.assertIn("另有 4 篇文档未画出", html)
        self.assertIn("另有 3 个章节未画出", html)

    def test_the_layout_height_grows_with_the_rows_it_printed(self):
        """高度是算出来的：多一行就必须多一行的高度，否则截图会裁掉最后一行。"""
        small = infographic.build_model(record(sources=[stored_source("S1", "a.md", "甲")]))
        big = infographic.build_model(record(sources=[
            stored_source("S1", "a.md", "甲"), stored_source("S2", "b.md", "乙")]))

        self.assertEqual(big["height"] - small["height"],
                         infographic.DOCUMENT_ROW_HEIGHT + infographic.HEADING_ROW_HEIGHT)

    def test_both_kinds_reserve_the_same_band_for_their_score_explanation(self):
        """导图那条不是空槽，而是"不适用 + 为什么"。留空会被读成"这里没做完"，
        而"导图本来就没有这个分数"恰好是这张图最该讲清的一件事。"""
        guide = infographic.build_model(record(backlink=rate()))
        mindmap = infographic.build_model(record(kind="mindmap", mindmap={"node_count": 3}))

        self.assertEqual(guide["height"], mindmap["height"])
        html = infographic.render_html(mindmap)
        self.assertIn("不适用", html)
        self.assertIn("不计命中率", html)
        self.assertIn("构造结果", html)

    def test_style_heights_agree_with_the_layout_constants(self):
        """版面常量与 CSS 是同一份契约。改了一边不改另一边，图就会被裁或者拖白边，
        而**两种毛病都不会让任何断言变红**——所以这条测试专门盯着它。"""
        expected = {".head": infographic.HEAD_HEIGHT, ".meta": infographic.META_HEIGHT,
                    ".backlink": infographic.BACKLINK_HEIGHT,
                    "h2": infographic.SECTION_TITLE_HEIGHT,
                    ".row": infographic.DOCUMENT_ROW_HEIGHT,
                    ".headings .row": infographic.HEADING_ROW_HEIGHT,
                    ".more": infographic.MORE_HEIGHT, ".foot": infographic.FOOTER_HEIGHT}

        for selector, height in expected.items():
            with self.subTest(selector=selector):
                self.assertEqual(_style_height(selector), height)

    def test_the_page_size_comes_from_the_model_not_from_css(self):
        model = infographic.build_model(record())

        html = infographic.render_html(model)

        self.assertIn(f"--page-width:{model['width']}px", html)
        self.assertIn(f"--page-height:{model['height']}px", html)
        self.assertIn("var(--page-height)", infographic.STYLE)

    def test_the_page_references_nothing_outside_itself(self):
        """自包含：没有外链、没有字体 CDN、没有脚本。

        两个理由，缺一不可：① 渲染结果不该取决于网络；② 这是"数据不出本机"在图片
        形态上的延续——一张图不该因为排版去请求任何外部地址。
        """
        html = infographic.render_html(infographic.build_model(record(backlink=rate())))

        for token in ("http://", "https://", "<script", "<img", "@import", "url("):
            with self.subTest(token=token):
                self.assertNotIn(token, html)

    def test_dangerous_text_is_escaped(self):
        """标题与路径来自用户文件，一律转义后再进模板。"""
        model = infographic.build_model(record(
            title="<script>alert(1)</script>",
            sources=[stored_source("S1", "a<b>.md", "甲 & 乙")]))

        html = infographic.render_html(model)

        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("a&lt;b&gt;.md", html)
        self.assertIn("甲 &amp; 乙", html)

    def test_a_record_without_sources_still_renders(self):
        """一份失败/空的产出也要能导出：图上写"0 个片段"比报错有用。"""
        model = infographic.build_model(record(sources=[], status="failed"))

        html = infographic.render_html(model)

        self.assertEqual(model["stats"]["chunks"], 0)
        self.assertIn("未完成", html)


def _style_height(selector: str) -> int | None:
    block = re.search(rf"{re.escape(selector)}\s*\{{([^}}]*)\}}", infographic.STYLE)
    if not block:
        return None
    found = re.search(r"height:(\d+)px", block.group(1))
    return int(found.group(1)) if found else None


class HeadlessTests(unittest.TestCase):
    def test_autodetection_falls_back_and_records_the_failed_browser(self):
        folder = temp_dir(self)
        browser = folder / "second-browser.exe"
        browser.touch()
        calls = []
        def run(args, timeout):
            calls.append(args[0])
            if len(calls) == 1:
                return subprocess.CompletedProcess(args, 0, b"", b"first failed")
            Path(args[-2].split("=", 1)[1]).write_bytes(png_bytes(20, 20))
            return subprocess.CompletedProcess(args, 0, b"", b"")
        with mock.patch.dict(os.environ, {headless.RENDERER_ENV: ""}), \
                mock.patch.object(headless, "browser_candidates", return_value=[Path(sys.executable), browser]), \
                mock.patch.object(headless, "_run_browser", side_effect=run):
            result = headless.screenshot(folder / "a.html", folder / "a.png", width=10, height=10)
        self.assertEqual(result["browser_path"], str(browser))
        self.assertEqual(len(result["attempt_failures"]), 1)
        self.assertEqual(headless.png_dimensions(folder / "a.png"), (20, 20))
        self.assertEqual(len(calls), 2)

    def test_explicit_browser_failure_does_not_fall_back_or_overwrite_png(self):
        folder = temp_dir(self)
        output = folder / "existing.png"
        output.write_bytes(png_bytes(20, 20))
        with mock.patch.object(headless, "_run_browser", return_value=
                               subprocess.CompletedProcess([], 1, b"", b"failure")) as run:
            with self.assertRaises(RenderFailed):
                headless.screenshot(folder / "a.html", output, width=10, height=10,
                                    browser=sys.executable)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(output.read_bytes(), png_bytes(20, 20))
        self.assertFalse(list(folder.glob("*.staged.png")))

    def test_a_timeout_reserves_time_for_the_next_browser(self):
        folder = temp_dir(self)
        browsers = [folder / "edge.exe", folder / "chrome.exe"]
        for browser in browsers:
            browser.touch()
        calls = []
        def run(args, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(args, timeout)
            Path(args[-2].split("=", 1)[1]).write_bytes(png_bytes(20, 20))
            return subprocess.CompletedProcess(args, 0, b"", b"")
        with mock.patch.dict(os.environ, {headless.RENDERER_ENV: ""}), \
                mock.patch.object(headless, "browser_candidates", return_value=browsers), \
                mock.patch.object(headless, "_run_browser", side_effect=run), \
                mock.patch.object(headless.time, "perf_counter", side_effect=[0, 0, 0, 5, 5, 5, 6]):
            result = headless.screenshot(folder / "a.html", folder / "a.png",
                                         width=10, height=10, timeout=10)
        self.assertEqual(calls, [5, 5])
        self.assertEqual(result["browser"], "Google Chrome")
        self.assertIn("超时", result["attempt_failures"][0]["message"])

    def test_the_environment_variable_wins_over_autodetection(self):
        """`OBSIDIAN_RAG_BROWSER` 是"找不到浏览器"时唯一的人工出口，必须真的生效。"""
        with mock.patch.dict(os.environ, {headless.RENDERER_ENV: sys.executable}):
            self.assertEqual(headless.find_browser(), Path(sys.executable))

    def test_a_dead_environment_variable_path_falls_back_to_autodetection(self):
        """指错了文件不该让渲染直接死掉——回落到自动探测，比报"没有浏览器"有用。"""
        with mock.patch.dict(os.environ, {headless.RENDERER_ENV: r"Z:\并不存在\浏览器.exe"}), \
                mock.patch.object(headless, "browser_candidates",
                                  return_value=[Path(sys.executable)]):
            self.assertEqual(headless.find_browser(), Path(sys.executable))

    def test_no_browser_at_all_is_a_degradation_not_a_crash(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(headless, "browser_candidates", return_value=[]):
            os.environ.pop(headless.RENDERER_ENV, None)

            self.assertIsNone(headless.find_browser())
            with self.assertRaises(RendererUnavailable):
                headless.screenshot("a.html", "a.png", width=10, height=10)

    def test_arguments_ask_for_exactly_the_requested_pixel_size(self):
        """尺寸靠 `--window-size × --force-device-scale-factor` 决定（实测过），
        所以这两个参数必须成对出现，否则图会大一圈或小一圈。"""
        args = headless.render_args(Path("browser.exe"), Path("a.html"), Path("a.png"),
                                    Path("profile"), 1080, 948, 2)

        self.assertIn("--window-size=1080,948", args)
        self.assertIn("--force-device-scale-factor=2", args)
        self.assertIn("--headless=new", args)
        self.assertEqual(args[-1], Path("a.html").resolve().as_uri())
        # 一次性的 user-data-dir：Windows 上浏览器会把命令行转交给已在运行的实例，
        # 共用 profile 会撞车，还可能碰到使用者本人的浏览器会话。
        self.assertIn("--user-data-dir=profile", args)

    def test_the_screenshot_path_must_still_look_like_a_png(self):
        """无头 Chrome 按扩展名判断图片格式，临时文件用 `.tmp` 会被当场拒绝。"""
        args = headless.render_args(Path("browser.exe"), Path("a.html"),
                                    Path(".a.staged.png"), Path("p"), 10, 10, 1)

        self.assertTrue(args[-2].startswith("--screenshot="))
        self.assertTrue(args[-2].endswith(".png"))

    def test_a_browser_that_exits_non_zero_is_a_failure(self):
        """拿 Python 当"浏览器"：它认不出 `--headless=new`，退码非 0 且不出图。"""
        with self.assertRaises(RenderFailed) as caught:
            headless.screenshot("a.html", "a.png", width=10, height=10,
                                browser=sys.executable)

        self.assertIn("退出码", str(caught.exception))

    def test_a_timeout_is_reported_as_a_failure(self):
        """真机上的超时在这里用替身模拟：要测的是"超时被如实报出来"，不是浏览器的
        启动速度。真边界在实机验收跑过（见 docs/学习记录/37）。"""
        with mock.patch.object(headless, "_run_browser",
                               side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            with self.assertRaises(RenderFailed) as caught:
                headless.screenshot("a.html", "a.png", width=10, height=10, timeout=1,
                                    browser=sys.executable)

        self.assertIn("超时", str(caught.exception))

    def test_a_zero_exit_with_a_non_png_output_is_still_a_failure(self):
        """退码 0 不等于出图成功：浏览器对不少参数问题只打日志不改退码，
        所以拿到文件还要验 PNG 头与尺寸。"""
        folder = temp_dir(self)

        def run(args, timeout):
            Path(args[-2].split("=", 1)[1]).write_bytes("这不是 PNG".encode())
            return subprocess.CompletedProcess(args, 0, b"", b"")

        with mock.patch.object(headless, "_run_browser", side_effect=run):
            with self.assertRaises(RenderFailed) as caught:
                headless.screenshot(folder / "a.html", folder / "a.png", width=10, height=10,
                                    browser=sys.executable)

        self.assertIn("不是 PNG", str(caught.exception))
        self.assertEqual(sorted(path.name for path in folder.iterdir()), [])

    def test_the_png_size_is_read_from_the_header(self):
        folder = temp_dir(self)
        good, bad = folder / "a.png", folder / "b.png"
        good.write_bytes(png_bytes(2160, 1896))
        bad.write_bytes("这不是 PNG".encode())

        self.assertEqual(headless.png_dimensions(good), (2160, 1896))
        self.assertIsNone(headless.png_dimensions(bad))
        self.assertIsNone(headless.png_dimensions(folder / "不存在.png"))

    def test_the_browser_label_says_which_one_was_used(self):
        """渲染记录里要能看出用的哪个浏览器——出问题时第一个要问的就是它。"""
        self.assertEqual(headless.browser_label(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                         "Google Chrome")
        self.assertEqual(headless.browser_label(r"C:\x\msedge.exe"), "Microsoft Edge")
        self.assertEqual(headless.browser_label(r"C:\x\browser.exe"), "browser.exe")


class InfographicServiceTests(unittest.TestCase):
    def build(self, upload=True):
        self.client, self.service, self.captured, self.app = build_harness(self, upload)
        return self.service

    def guide(self):
        final = list(self.service.stream("分页管理", "guide"))[-1]
        return final["artifact_id"]

    def test_export_writes_a_png_an_html_and_a_render_record(self):
        service = self.build()
        artifact_id = self.guide()

        with mock.patch.object(studio, "screenshot", fake_screenshot(milliseconds=743)):
            payload = service.export_infographic(artifact_id)

        files = payload["files"]
        exports = service.paths.exports
        self.assertTrue((exports / files["png"]).is_file())
        self.assertTrue((exports / files["html"]).is_file())
        self.assertTrue((exports / files["record"]).is_file())
        self.assertEqual(payload["render"]["status"], "complete")
        self.assertEqual(payload["render"]["milliseconds"], 743)
        self.assertEqual(payload["render"]["browser"], "Fake Browser")
        self.assertTrue(payload["render"]["pixels_match"])
        self.assertFalse(payload["degraded"])
        self.assertEqual(payload["layout"]["pixels"],
                         {"width": payload["layout"]["width"] * 2,
                          "height": payload["layout"]["height"] * 2})
        # 导出的是**当前**正文算出来的分数，不是生成时缓存的那份。
        self.assertAlmostEqual(payload["backlink"]["hit_rate"], 2 / 3)

    def test_the_export_record_can_be_read_back_after_a_reload(self):
        service = self.build()
        artifact_id = self.guide()

        with mock.patch.object(studio, "screenshot", fake_screenshot(milliseconds=612)):
            service.export_infographic(artifact_id)

        again = service.infographic_export(artifact_id)

        self.assertEqual(again["artifact_id"], artifact_id)
        self.assertEqual(again["export"]["render"]["milliseconds"], 612)
        self.assertTrue((service.paths.exports / again["export"]["files"]["png"]).is_file())

    def test_a_corrupt_record_reads_as_not_exported_instead_of_crashing(self):
        """记录文件坏了不该把界面卡住——当作没导出过，比抛 500 有用。"""
        service = self.build()
        artifact_id = self.guide()
        record_path = service._infographic_paths("分页管理", artifact_id)["record"]
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text("{ 这不是 JSON", encoding="utf-8")

        self.assertIsNone(service.infographic_export(artifact_id)["export"])

    def test_without_a_browser_the_export_degrades_and_keeps_the_html(self):
        """降级要满足三件事：如实说明、留下 HTML、**不留下一个假 PNG**。"""
        service = self.build()
        artifact_id = self.guide()

        with mock.patch.object(studio, "screenshot",
                               side_effect=RendererUnavailable("没有找到可用的浏览器")):
            payload = service.export_infographic(artifact_id)

        self.assertTrue(payload["degraded"])
        self.assertEqual(payload["render"]["status"], "unavailable")
        self.assertEqual(payload["render"]["reason"], "browser_unavailable")
        self.assertIn("没有找到可用的浏览器", payload["render"]["message"])
        self.assertIsNone(payload["render"]["milliseconds"])
        html = service.paths.exports / payload["files"]["html"]
        self.assertTrue(html.is_file())
        self.assertIn("来源分布", html.read_text(encoding="utf-8"))
        self.assertFalse((service.paths.exports / payload["files"]["png"]).exists())
        # 降级也要记一笔：否则"我明明导出过"会变成一句无法追查的话。
        self.assertTrue((service.paths.exports / payload["files"]["record"]).is_file())
        self.assertEqual(service.infographic_export(artifact_id)["export"]["render"]["reason"],
                         "browser_unavailable")

    def test_a_failed_render_reports_its_own_reason(self):
        service = self.build()
        artifact_id = self.guide()

        with mock.patch.object(studio, "screenshot", side_effect=RenderFailed("渲染超过 90 秒")):
            payload = service.export_infographic(artifact_id)

        self.assertEqual(payload["render"]["reason"], "render_failed")
        self.assertIn("渲染超过 90 秒", payload["render"]["message"])

    def test_a_browser_that_ignores_the_window_size_is_visible_in_the_record(self):
        """图出来了但尺寸不对：记录里必须看得出这个不一致，不能被"反正有图了"盖过去。"""
        service = self.build()
        artifact_id = self.guide()

        with mock.patch.object(studio, "screenshot", fake_screenshot(pixels=(800, 600))):
            payload = service.export_infographic(artifact_id)

        self.assertEqual(payload["render"]["pixels"], {"width": 800, "height": 600})
        self.assertFalse(payload["render"]["pixels_match"])
        self.assertFalse(payload["degraded"])

    def test_the_image_file_is_missing_before_the_first_export(self):
        service = self.build()
        artifact_id = self.guide()

        self.assertIsNone(service.infographic_export(artifact_id)["export"])
        with self.assertRaises(KeyError):
            service.infographic_file(artifact_id)

    def test_an_unknown_artifact_is_refused_everywhere(self):
        service = self.build()

        for call in (service.export_infographic, service.infographic_export,
                     service.infographic_model, service.infographic_file):
            with self.subTest(call=call.__name__):
                with self.assertRaises(KeyError):
                    call("art_不存在")

    def test_deleting_the_artifact_takes_its_export_with_it(self):
        """产出没了却留下一张图，用户点开只会觉得是幽灵文件。"""
        service = self.build()
        artifact_id = self.guide()
        with mock.patch.object(studio, "screenshot", fake_screenshot()):
            service.export_infographic(artifact_id)
        exports = service.paths.exports
        before = sorted(path.name for path in exports.iterdir())
        self.assertEqual(len(before), 3)

        service.delete_artifact(artifact_id)

        self.assertEqual(sorted(path.name for path in exports.iterdir()), [])

    def test_a_mindmap_exports_too_and_shows_nodes_instead_of_a_rate(self):
        service = self.build()
        artifact_id = list(service.stream("分页管理", "mindmap"))[-1]["artifact_id"]

        with mock.patch.object(studio, "screenshot", fake_screenshot()):
            payload = service.export_infographic(artifact_id)

        html = (service.paths.exports / payload["files"]["html"]).read_text(encoding="utf-8")
        self.assertIsNone(payload["backlink"])
        self.assertIn("结构节点", html)
        self.assertNotIn("回链命中率", html)


class InfographicApiTests(unittest.TestCase):
    """走真实 HTTP：服务层能跑通不等于界面上点得到。"""

    def build(self):
        self.client, self.service, self.captured, self.app = build_harness(self)

    def guide(self):
        response = self.client.post("/api/v1/artifacts/stream",
                                    json={"topic": "分页管理", "kind": "guide"})
        return json.loads(response.text.splitlines()[-1])["artifact_id"]

    def export(self, artifact_id):
        with mock.patch.object(studio, "screenshot", fake_screenshot(milliseconds=705)):
            return self.client.post(f"/api/v1/artifacts/{artifact_id}/infographic")

    def test_the_export_round_trip_over_http(self):
        self.build()
        artifact_id = self.guide()

        created = self.export(artifact_id)
        self.assertEqual(created.status_code, 200)
        payload = created.json()
        self.assertFalse(payload["degraded"])
        self.assertEqual(payload["render"]["milliseconds"], 705)

        stored = self.client.get(f"/api/v1/artifacts/{artifact_id}/infographic").json()
        self.assertEqual(stored["export"]["render"]["milliseconds"], 705)

        inline = self.client.get(f"/api/v1/artifacts/{artifact_id}/infographic.png")
        self.assertEqual(inline.status_code, 200)
        self.assertEqual(inline.headers["content-type"], "image/png")
        self.assertTrue(inline.content.startswith(headless.PNG_SIGNATURE))
        # 预览与下载是同一个文件的两个用法，不该是两套东西。
        self.assertNotIn("content-disposition", inline.headers)
        download = self.client.get(
            f"/api/v1/artifacts/{artifact_id}/infographic.png?download=1")
        # 文件名是中文，Starlette 按 RFC 5987 写成 `filename*=utf-8''` 百分号编码，
        # 所以这里断言的是"是个附件、名字指到那个 PNG"，不是断言原样中文。
        self.assertIn("attachment", download.headers["content-disposition"])
        self.assertIn("filename*=", download.headers["content-disposition"])
        self.assertIn(".png", download.headers["content-disposition"])

    def test_a_degraded_export_still_answers_200_and_says_so(self):
        """降级不是"失败"：用户图没拿到，但事情说清楚了，HTML 也在。"""
        self.build()
        artifact_id = self.guide()

        with mock.patch.object(studio, "screenshot",
                               side_effect=RendererUnavailable("没有找到可用的浏览器")):
            response = self.client.post(f"/api/v1/artifacts/{artifact_id}/infographic")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["degraded"])
        self.assertIn("html_path", body["files"])
        self.assertEqual(body["render"]["reason"], "browser_unavailable")

    def test_the_png_route_says_what_is_missing_before_the_first_export(self):
        self.build()
        artifact_id = self.guide()

        response = self.client.get(f"/api/v1/artifacts/{artifact_id}/infographic.png")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "infographic_missing")

    def test_an_unknown_artifact_is_a_404_on_every_route(self):
        self.build()

        for method, path in (("post", "/api/v1/artifacts/art_x/infographic"),
                             ("get", "/api/v1/artifacts/art_x/infographic"),
                             ("get", "/api/v1/artifacts/art_x/infographic.png")):
            with self.subTest(path=path, method=method):
                response = getattr(self.client, method)(path)
                self.assertEqual(response.status_code, 404)

    def test_deleting_through_http_takes_the_export_with_it(self):
        self.build()
        artifact_id = self.guide()
        self.export(artifact_id)

        self.client.delete(f"/api/v1/artifacts/{artifact_id}")

        self.assertEqual(
            self.client.get(f"/api/v1/artifacts/{artifact_id}/infographic.png").status_code, 404)


class RealRenderTests(unittest.TestCase):
    """真的起一次本机浏览器。没有浏览器的机器上跳过——降级路径另有确定性测试，
    所以跳过不会让这套测试变成"什么都没验"。"""

    @unittest.skipUnless(headless.find_browser(), "本机没有可用的浏览器")
    def test_a_real_browser_writes_a_png_of_exactly_the_requested_pixels(self):
        """这条把整条链走通：非 ASCII 数据目录 → `file://` URL → 真实浏览器 → PNG。"""
        client, service, captured, app = build_harness(self)
        artifact_id = list(service.stream("分页管理", "guide"))[-1]["artifact_id"]

        payload = service.export_infographic(artifact_id)

        self.assertEqual(payload["render"]["status"], "complete",
                         payload["render"].get("message"))
        self.assertTrue(payload["render"]["pixels_match"])
        self.assertEqual(payload["render"]["pixels"], payload["layout"]["pixels"])
        self.assertGreater(payload["render"]["milliseconds"], 0)
        self.assertGreater(payload["render"]["bytes"], 1000)
        # 记录里必须说得出"用的哪个浏览器"，出问题时第一个要问的就是它。
        self.assertIn(Path(payload["render"]["browser_path"]),
                      [headless.find_browser(), *headless.browser_candidates()])
        path = service.paths.exports / payload["files"]["png"]
        self.assertEqual(headless.png_dimensions(path), (payload["layout"]["pixels"]["width"],
                                                        payload["layout"]["pixels"]["height"]))
        # 渲染用的 HTML 留在导出目录里：图没出来时它就是用户的退路，也是排查现场。
        self.assertIn("来源分布",
                      (service.paths.exports / payload["files"]["html"]).read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
