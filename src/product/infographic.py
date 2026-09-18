"""信息图：把一份产出的**来源**画成一张 PNG 的 HTML 源。

这是 P4「图片产出」的落点，形态上刻意选成**一个导出器**，而不是第三种产出类型：

- 计划里原写"信息图是产出形态之一"，动手后确认**做成导出器更对**。第三种类型要动
  `artifacts.kind` 的 `CHECK(kind IN ('guide','mindmap'))`，那是一次重建表的迁移，
  而重建 `artifacts` 会连带 `artifact_sources` 的 `ON DELETE CASCADE`——为了一个
  **不带新能力**的形态去冒这个风险不划算：信息图要展示的东西（来源分布、章节结构、
  命中率）**已经全部存在库里**，不需要新检索、不需要新表。
- 更要紧的是它**不调模型**。模板是固定的，所以它承载不了自由文本；而它该承载的
  恰好是"结构 + 来源"，这些本来就是语料里的既有事实，可复现。

于是这张图的每个元素都能追到原文：每个条形后面是具体的片段编号，每个编号指向
`artifact_sources` 里的一行。**图里没有一个字是模型写的**——这是产品那句"可溯源"
在图片形态上的落法，也是这张图和"AI 生成配图"的根本区别。

高度是**算出来**的，不是试出来的：所有文本行都 `nowrap + ellipsis`（永不换行），
每个区块在 CSS 里写死高度，于是 `build_model()` 的加总就等于实际内容高度，截图时
`--window-size` 给多少就正好是多少，既不裁切也不拖白边。**改版面常量就要同步改
`STYLE` 里的高度**——两边是同一份契约，`layout_matches_style` 那条测试盯着它。
"""

from __future__ import annotations

from html import escape


# ---- 版面常量（右列数值必须与 STYLE 中同名区块的高度一致） ----------------------

WIDTH = 1080
SCALE = 2
PAD = 44
HEAD_HEIGHT = 112
META_HEIGHT = 88
BACKLINK_HEIGHT = 96
SECTION_TITLE_HEIGHT = 38
DOCUMENT_ROW_HEIGHT = 52
HEADING_ROW_HEIGHT = 44
MORE_HEIGHT = 34
FOOTER_HEIGHT = 76
GAP = 16
# 一屏画不完就不画，但**要如实说还有多少没画**——截断本身不是问题，悄悄截断才是。
# 上限存在的真正理由是版面高度可算（见模块 docstring）。
MAX_DOCUMENT_ROWS = 7
MAX_HEADING_ROWS = 8

KIND_LABELS = {"guide": "学习指南", "mindmap": "思维导图"}
ORIGIN_LABELS = {"note": "笔记", "web": "联网"}
STATUS_LABELS = {"complete": "已完成", "stopped": "已停止", "failed": "未完成", "running": "生成中"}

FOOTER_NOTE = "本图不含模型生成内容：每个编号都指向一份资料片段，可回原文核对。"
RATE_NOTE = "命中率数的是“这句话有没有写来源”，不是“这句话对不对”。"
MINDMAP_NOTE = "思维导图由片段标题层级构造，不经过模型，因此不计命中率。"


def _documents(sources: list[dict]) -> list[dict]:
    """按文档把片段归拢。顺序按首次出现，**不重排**——检索顺序本身就是信息
    （排在前面的更相关），重排会让这张图看起来和检索结果不是一回事。"""
    grouped: dict[str, dict] = {}
    for item in sources:
        title = item.get("title") or "未命名文档"
        entry = grouped.setdefault(title, {"title": title, "labels": []})
        entry["labels"].append(item.get("label", ""))
    documents = list(grouped.values())
    for entry in documents:
        entry["count"] = len(entry["labels"])
    biggest = max((entry["count"] for entry in documents), default=0)
    for entry in documents:
        entry["share"] = (entry["count"] / biggest) if biggest else 0.0
    return documents


def _headings(sources: list[dict]) -> list[dict]:
    """章节：按 `heading_path` 去重，编号挂在一起。"""
    grouped: dict[str, list[str]] = {}
    for item in sources:
        path = (item.get("heading_path") or "").strip()
        if path:
            grouped.setdefault(path, []).append(item.get("label", ""))
    return [{"path": path, "labels": labels} for path, labels in grouped.items()]


def _rate_cell(record: dict) -> dict:
    """右上角那一格：指南报命中率（真分数），思维导图报节点数（另一个真数字）。

    导图**绝不放命中率**：它的覆盖率是构造结果（节点就是拿片段建的），必然接近 1，
    和指南的命中率并排显示会让人以为两者可比。这条纪律在 `studio.py` 里写过一次，
    这里再写一次，是因为版面本身很容易让人"顺手把那个数也放上去"。
    """
    if record.get("kind") == "mindmap":
        node_count = (record.get("mindmap") or {}).get("node_count")
        return {"value": str(node_count) if node_count else "—", "label": "结构节点"}
    backlink = record.get("backlink") or {}
    if backlink.get("hit_rate") is None:
        return {"value": "—", "label": "回链命中率"}
    return {"value": f"{backlink['hit_rate'] * 100:.1f}%", "label": "回链命中率"}


def _short_time(value: str) -> str:
    """ISO 时间戳 → 图上用的 "YYYY-MM-DD HH:MM"。

    与产出详情页一致（那里是 `created_at.slice(0, 16)`）。秒以下的小数位和 `+00:00`
    时区在图上没有信息量，却会占掉半行——副标题是**单行不换行**的，被它挤掉的正是
    排在最后的「来源层级」。真实渲染过一版才看出来：整行以 `来源层级 …` 结尾，
    有信息的那一半反而看不见了。
    """
    return value[:16].replace("T", " ") if value else "未记录"


def _short_index(value: str) -> str:
    """索引版本 → 前 8 位，与产出详情页一致。

    真实值是 `idx_` 加 32 位十六进制，整整 36 个字符。完整贴上只会把副标题挤爆；
    要完整值可以从产出详情页或 `render.json` 里取，图上不需要。
    """
    return value[:8] if value else "未记录"


def build_model(record: dict) -> dict:
    """一条产出记录 → 信息图模型（**含版面宽度与高度**）。

    `record` 就是 `StudioService.get_artifact()` 的返回值：来源、命中率、导图树都
    已经在里面，所以这里不碰数据库、不碰模型、不碰网络，纯函数。
    """
    sources = record.get("sources") or []
    documents, headings = _documents(sources), _headings(sources)
    origins: dict[str, int] = {}
    for item in sources:
        origin = item.get("origin") or "note"
        origins[origin] = origins.get(origin, 0) + 1

    document_rows = documents[:MAX_DOCUMENT_ROWS]
    heading_rows = headings[:MAX_HEADING_ROWS]
    omitted = {"documents": len(documents) - len(document_rows),
               "headings": len(headings) - len(heading_rows)}

    # 只有指南有真命中率；导图那一格换成节点数（见 `_rate_cell`）。
    backlink = record.get("backlink") if record.get("kind") == "guide" else None
    origin_text = f"笔记 {origins.get('note', 0)}"
    if origins.get("web"):
        origin_text += f" · 联网 {origins['web']}"

    # 记分说明条**两种产出都占位**：导图那一条写的是"不适用"和为什么，不是空白。
    # 一条空槽比一段解释更容易被当成"没做完"，而且两种产出的总高度因此一致，
    # 版面算术不用分叉（见下面的高度加总）。
    height = PAD + HEAD_HEIGHT + GAP + META_HEIGHT + GAP + BACKLINK_HEIGHT
    height += GAP + SECTION_TITLE_HEIGHT + len(document_rows) * DOCUMENT_ROW_HEIGHT
    height += MORE_HEIGHT if omitted["documents"] else 0
    height += GAP + SECTION_TITLE_HEIGHT + len(heading_rows) * HEADING_ROW_HEIGHT
    height += MORE_HEIGHT if omitted["headings"] else 0
    height += GAP + FOOTER_HEIGHT + PAD

    return {
        "artifact_id": record.get("id", ""),
        "kind": record.get("kind", "guide"),
        "kind_label": KIND_LABELS.get(record.get("kind"), "产出"),
        "status_label": STATUS_LABELS.get(record.get("status"), record.get("status") or ""),
        "title": record.get("title") or "未命名产出",
        "topic": record.get("topic") or "",
        "created_at": record.get("created_at") or "",
        "index_version": record.get("index_version") or "未记录",
        # 副标题专用的短写法：原始的 created_at / index_version 照旧留在模型里，
        # 但**上版面的**是这两个——见两个 helper 的注释（单行不换行，长的会挤掉后面）。
        "created_label": _short_time(record.get("created_at") or ""),
        "index_label": _short_index(record.get("index_version") or "未记录"),
        "stats": {"chunks": len(sources), "documents": len(documents),
                  "sections": len(headings), "origin_text": origin_text},
        "rate": _rate_cell(record),
        "backlink": backlink,
        "rate_note": RATE_NOTE if backlink else MINDMAP_NOTE,
        "documents": document_rows,
        "headings": heading_rows,
        "omitted": omitted,
        "footer": FOOTER_NOTE,
        "width": WIDTH,
        "height": height,
        "scale": SCALE,
    }


# 静态样式。动态尺寸走 CSS 自定义属性从行内 style 传进来，这样这份样式表里
# 一个花括号都不需要转义（f-string 里写 CSS 是上一个版本的翻车点）。
STYLE = """
 * { margin:0; padding:0; box-sizing:border-box; }
 html, body { background:#f6f3ee; color:#241f1b;
   font-family:"Microsoft YaHei","PingFang SC","Segoe UI",sans-serif; }
 .page { width:var(--page-width); height:var(--page-height); overflow:hidden;
   padding:44px; display:flex; flex-direction:column; gap:16px; }
 .band { flex:none; overflow:hidden; }
 .head { height:112px; border-bottom:3px solid #8a6a4f; }
 .chip { display:inline-block; height:26px; line-height:26px; padding:0 12px;
   background:#8a6a4f; color:#fdfbf8; border-radius:13px; font-size:14px; }
 h1 { font-size:30px; line-height:42px; margin-top:12px; white-space:nowrap;
   overflow:hidden; text-overflow:ellipsis; }
 .sub { font-size:15px; line-height:22px; color:#6d635a; margin-top:6px;
   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
 .meta { height:88px; display:flex; gap:14px; }
 .meta .cell { flex:1; background:#fffdfa; border:1px solid #e3dbd1; border-radius:10px;
   padding:14px 16px; overflow:hidden; }
 .meta .cell b { display:block; font-size:24px; line-height:30px; white-space:nowrap;
   overflow:hidden; text-overflow:ellipsis; }
 .meta .cell span { display:block; font-size:14px; color:#6d635a; margin-top:6px;
   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
 .backlink { height:96px; background:#fffdfa; border:1px solid #e3dbd1; border-radius:10px;
   padding:12px 18px; }
 .backlink .rate { display:flex; align-items:baseline; gap:12px; height:34px;
   overflow:hidden; }
 .backlink .rate b { font-size:30px; line-height:34px; color:#8a6a4f; }
 .backlink .rate span { font-size:14px; color:#6d635a; }
 .track { height:10px; background:#ece5dc; border-radius:5px; margin-top:8px;
   overflow:hidden; }
 .track i { display:block; height:10px; background:#8a6a4f; border-radius:5px; }
 .note { margin-top:8px; font-size:13px; line-height:18px; color:#6d635a;
   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
 .section { flex:none; overflow:hidden; }
 h2 { height:38px; line-height:38px; font-size:19px; color:#4a3f36;
   border-left:4px solid #8a6a4f; padding-left:10px; }
 .row { height:52px; display:flex; align-items:center; gap:14px;
   border-bottom:1px solid #ece5dc; }
 .row .name { width:296px; font-size:15px; white-space:nowrap; overflow:hidden;
   text-overflow:ellipsis; }
 .row .bar { flex:1; height:16px; background:#ece5dc; border-radius:8px; overflow:hidden; }
 .row .bar i { display:block; height:16px; background:#b08b64; border-radius:8px; }
 .row .tags { width:196px; text-align:right; font-size:13px; color:#6d635a;
   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
 .headings .row { height:44px; }
 .headings .row .name { width:auto; flex:1; }
 .more { height:34px; line-height:34px; font-size:13px; color:#6d635a;
   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
 .foot { height:76px; border-top:1px solid #e3dbd1; padding-top:14px; font-size:13px;
   line-height:22px; color:#6d635a; }
 .foot div { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
"""


def _tags(labels) -> str:
    return escape(" ".join(label for label in labels if label) or "—")


def _bar(share: float) -> str:
    """条形宽度按"相对最多的那篇"算，不是占全体的比例：一堆 1 个片段的文档如果按
    绝对比例画，全长一个样，分布就看不出来了。"""
    width = max(2.0, min(100.0, share * 100))
    return f'<span class="bar"><i style="width:{width:.1f}%"></i></span>'


def _row(name: str, bar: str, labels) -> str:
    return (f'<div class="row"><span class="name">{escape(str(name))}</span>{bar}'
            f'<span class="tags">{_tags(labels)}</span></div>')


def _section(title: str, rows: list[str], extra: str, classes: str = "") -> str:
    return (f'<section class="band section {classes}"><h2>{escape(title)}</h2>'
            f'{"".join(rows)}{extra}</section>')


def _more(omitted: int, unit: str, total: int) -> str:
    if not omitted:
        return ""
    return f'<div class="more">另有 {omitted} {unit}未画出（去重后共 {total} {unit}）</div>'


def _meta(model: dict) -> str:
    """四个统计格。来源层级**不在这排**：它的值（"笔记 4 · 联网 2"）比另外几个长，
    挤进等宽格子就会被省略号吃掉，所以它去副标题（见 `render_html`）。"""
    stats = model["stats"]
    cells = [
        {"value": stats["chunks"], "label": "引用片段"},
        {"value": stats["documents"], "label": "来源文档"},
        {"value": stats["sections"], "label": "去重章节"},
        model["rate"],
    ]
    return '<section class="band meta">' + "".join(
        f'<div class="cell"><b>{escape(str(item["value"]))}</b>'
        f'<span>{escape(str(item["label"]))}</span></div>' for item in cells) + "</section>"


def _score_band(model: dict) -> str:
    """记分说明条。指南给真分数，导图给"不适用"与理由——**两种都要说清**。

    导图那条如果留空，读者只会看到一个空槽；而"这里本来就没分数"恰好是这张图最
    需要讲清楚的一件事（它的覆盖率是构造结果，必然接近 1）。
    """
    backlink = model["backlink"]
    if not backlink:
        return ('<section class="band backlink">'
                '<div class="rate"><b>不适用</b>'
                '<span>思维导图不计命中率：节点是拿片段标题构造的，覆盖率是构造结果，'
                '必然接近 100%</span></div>'
                '<div class="track"></div>'
                f'<div class="note">{escape(model["rate_note"])}</div>'
                "</section>")
    rate = backlink["hit_rate"]
    percent = f"{rate * 100:.1f}%" if rate is not None else "—"
    return ('<section class="band backlink">'
            f'<div class="rate"><b>{percent}</b>'
            f'<span>条陈述带来源 {backlink["with_source"]} / {backlink["assertions"]} ·'
            f' 缺来源 {backlink["missing_count"]} 条</span></div>'
            f'<div class="track"><i style="width:{percent}"></i></div>'
            f'<div class="note">{escape(model["rate_note"])}</div>'
            "</section>")


def render_html(model: dict) -> str:
    """固定模板 → 自包含 HTML。

    **一个外链都没有**：字体只用系统字体、样式内联、没有图片、没有脚本。这既是
    "数据不出本机"的延续，也让渲染结果与网络状态无关（有测试盯着这一条）。
    """
    chip = model["kind_label"]
    if model["status_label"] and model["status_label"] != "已完成":
        chip = f'{chip} · {model["status_label"]}'
    subtitle = (f'主题：{model["topic"] or "未记录"} · 生成于 {model["created_label"]}'
                f' · 索引 {model["index_label"]} · 来源层级 {model["stats"]["origin_text"]}')
    documents = _section(
        "来源分布（条形长度＝该文档贡献的片段数）",
        [_row(item["title"], _bar(item["share"]), item["labels"]) for item in model["documents"]],
        _more(model["omitted"]["documents"], "篇文档", model["stats"]["documents"]))
    headings = _section(
        "章节结构（每一行都是一个真实标题路径）",
        [_row(item["path"], "", item["labels"]) for item in model["headings"]],
        _more(model["omitted"]["headings"], "个章节", model["stats"]["sections"]),
        classes="headings")
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{escape(model["title"])} · 来源画像</title>
<style>{STYLE}</style></head>
<body><div class="page" style="--page-width:{model["width"]}px;--page-height:{model["height"]}px">
<header class="band head">
 <div class="chip">{escape(chip)}</div>
 <h1>{escape(model["title"])}</h1>
 <div class="sub">{escape(subtitle)}</div>
</header>
{_meta(model)}
{_score_band(model)}
{documents}
{headings}
<footer class="band foot">
 <div>{escape(model["footer"])}</div>
 <div>数据源 {escape(model["artifact_id"])} · 由 Obsidian RAG 渲染 · 版面
   {model["width"]}×{model["height"]}，导出 {model["width"] * model["scale"]}×
   {model["height"] * model["scale"]} 像素</div>
</footer>
</div></body></html>
"""
