"""收藏、回答反馈与可直接放入 Obsidian 的 Markdown 导出。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
import uuid

from .chat import source_record
from .database import Database, utc_now


FEEDBACK_KINDS = {"helpful", "missing", "citation_wrong", "answer_wrong"}
MAX_TAGS = 8
MAX_TAG_CHARS = 24
# 「最近 N 天」的取值。筛选放在服务层做，所以这里是天数而不是 SQL 片段。
RANGE_DAYS = {"7d": 7, "30d": 30}
# Windows 保留的设备名。带扩展名也一样打不开（`CON.md` 存不进去），所以
# 撞上就换前缀，不能指望后缀能救。
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL",
                  *(f"COM{i}" for i in range(1, 10)),
                  *(f"LPT{i}" for i in range(1, 10))}


def _id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _location(source):
    locator = source["locator"]
    if locator.get("kind") == "pdf":
        return f"第 {locator.get('page')} 页"
    if locator.get("kind") == "notebook":
        return f"单元格 {locator.get('cell')}（第 {locator.get('start_line')}–{locator.get('end_line')} 行）"
    return f"第 {locator.get('start_line')}–{locator.get('end_line')} 行"


def safe_filename(title, fallback):
    """标题 → 能落盘的文件名主干（不含扩展名）。

    `fallback` 是标题被清空后的兜底，返回值一定非空：单条导出和专题导出都
    要拼文件名，各写一遍这段过滤规则迟早会分叉。
    """
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")[:80]
    if name.upper() in RESERVED_NAMES:
        name = f"收藏-{name}"
    return name or fallback


def _write_atomic(path, text):
    """先写临时文件再改名，导到一半中断不会留下半份文件。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return path


def _tags(raw):
    """标签列是 JSON 数组；引入标签之前存下的行是 NULL，解成空表。"""
    try:
        values = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [value for value in values if isinstance(value, str)]


def _normalize_tags(values):
    if not isinstance(values, list):
        raise ValueError("标签必须是一个列表。")
    cleaned = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("标签必须是文本。")
        tag = re.sub(r"\s+", " ", value).strip()[:MAX_TAG_CHARS]
        if tag and tag not in cleaned:
            cleaned.append(tag)
    # 先去重再数：贴进十个一样的标签不该报错。
    if len(cleaned) > MAX_TAGS:
        raise ValueError(f"标签最多 {MAX_TAGS} 个。")
    return cleaned


def _question_heading(favorite):
    """小节标题用收藏的标题而不是原始问题：用户改过标题时，改过的更有意义。"""
    return favorite["title"] or "收藏的回答"


def _body_block(favorite):
    """一条收藏的正文：备注、回答、来源。单条导出与专题导出共用。"""
    lines = [f"> 问题：{favorite['question']}", ""]
    if favorite["note"]:
        lines += ["## 备注", "", favorite["note"], ""]
    lines += ["## 回答", "", favorite["answer"], "", "## 来源", ""]
    for source in favorite["sources"]:
        lines.append(
            f"- [{source['label']}] {source['title']} — {source['heading_path']}（{_location(source)}）")
    return lines + [""]


def _provenance_block(favorite):
    return [f"生成时间：{favorite['generated_at'] or favorite['created_at']}",
            f"模型：{favorite['provider'] or '未知'} / {favorite['model'] or '未知'}",
            f"索引版本：{favorite['index_version'] or '未知'}"]


class OrganizeService:
    def __init__(self, database: Database, paths):
        self.database, self.paths = database, paths

    def _favorite(self, favorite_id):
        row = self.database.fetchone("SELECT * FROM favorites WHERE id=?", (favorite_id,))
        if not row:
            raise KeyError("收藏不存在。")
        sources = [source_record(item) for item in self.database.fetchall(
            "SELECT * FROM favorite_sources WHERE favorite_id=? ORDER BY position", (favorite_id,))]
        record = dict(row)
        record["tags"] = _tags(record.pop("tags_json", None))
        return {**record, "sources": sources}

    def _summaries(self):
        """列表用的收藏概览：来源数、涉及资料库、反馈类型、标签。

        资料库按 favorite 分组单独查一次，而不是在 SQL 里 GROUP_CONCAT：资料库
        名字里可以带逗号，拼起来再切回去迟早出错。
        """
        rows = self.database.fetchall("""SELECT f.*,
            (SELECT COUNT(*) FROM favorite_sources s WHERE s.favorite_id=f.id) source_count,
            (SELECT af.kind FROM answer_feedback af WHERE af.message_id=f.message_id) feedback_kind
            FROM favorites f ORDER BY updated_at DESC""")
        libraries = {}
        for row in self.database.fetchall("""SELECT DISTINCT s.favorite_id, l.id, l.name
            FROM favorite_sources s
            JOIN documents d ON d.id=s.document_id
            JOIN libraries l ON l.id=d.library_id"""):
            libraries.setdefault(row["favorite_id"], []).append(
                {"id": row["id"], "name": row["name"]})
        records = []
        for row in rows:
            record = dict(row)
            record["tags"] = _tags(record.pop("tags_json", None))
            record["libraries"] = libraries.get(record["id"], [])
            records.append(record)
        return records

    def list_favorites(self, library=None, tag=None, feedback=None, days=None):
        """筛选后的收藏列表，附带界面填筛选器用的选项和未筛选的总数。

        选项取自全部收藏而不是筛完的这一批：否则选了一个资料库，别的资料库就
        从下拉里消失了，用户切不回去。

        `library` 的语义是"这条收藏的检索证据里有这个资料库的资料"。一次检索只
        取前 8 段，所以真实资料库上它会收窄成有意义的一批；测试里那两三段的
        小语料上则几乎条条命中，那是语料太小，不是筛选写错了。
        """
        if feedback is not None and feedback not in FEEDBACK_KINDS:
            raise ValueError("反馈类型无效。")
        if days is not None and days not in RANGE_DAYS:
            raise ValueError("时间范围无效。")
        summaries = self._summaries()
        cutoff = None if days is None else (
            datetime.now(timezone.utc) - timedelta(days=RANGE_DAYS[days])).isoformat()

        def keep(item):
            if library and library not in {entry["id"] for entry in item["libraries"]}:
                return False
            if tag and tag not in item["tags"]:
                return False
            if feedback and item["feedback_kind"] != feedback:
                return False
            # 时间戳都出自 utc_now()，格式一致，可以直接比字符串。
            return not (cutoff and item["updated_at"] < cutoff)

        options = {}
        for item in summaries:
            for entry in item["libraries"]:
                options[entry["id"]] = entry["name"]
        return {
            "favorites": [item for item in summaries if keep(item)],
            "libraries": [{"id": key, "name": name}
                          for key, name in sorted(options.items(), key=lambda pair: pair[1])],
            "tags": sorted({value for item in summaries for value in item["tags"]}),
            "total": len(summaries),
        }

    def get_favorite(self, favorite_id):
        return self._favorite(favorite_id)

    def create_favorite(self, message_id):
        existing = self.database.fetchone("SELECT id FROM favorites WHERE message_id=?", (message_id,))
        if existing:
            return self._favorite(existing["id"])
        message = self.database.fetchone("""SELECT a.*, u.content question
            FROM messages a JOIN messages u ON u.id=a.reply_to_message_id
            WHERE a.id=? AND a.role='assistant'""", (message_id,))
        if not message:
            raise KeyError("回答不存在。")
        if message["status"] != "complete":
            raise ValueError("只有完整回答可以收藏。")
        sources = self.database.fetchall(
            "SELECT * FROM message_sources WHERE message_id=? ORDER BY position", (message_id,))
        if not sources:
            raise ValueError("该回答没有可核对来源，暂不支持收藏。")
        favorite_id, now = _id("fav"), utc_now()
        title = re.sub(r"\s+", " ", message["question"]).strip()[:80] or "收藏的回答"
        with self.database.transaction() as connection:
            connection.execute("""INSERT INTO favorites(id, message_id, title, note, question,
                answer, provider, model, index_version, generated_at, created_at, updated_at)
                VALUES (?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (favorite_id, message_id, title, message["question"], message["content"],
                 message["provider"], message["model"], message["index_version"],
                 message["completed_at"], now, now))
            for source in sources:
                connection.execute("""INSERT INTO favorite_sources(favorite_id, label, position,
                    chunk_id, document_id, version_id, title, media_type, heading_path, locator_json,
                    preview, score_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (favorite_id, source["label"], source["position"], source["chunk_id"], source["document_id"],
                     source["version_id"], source["title"], source["media_type"],
                     source["heading_path"], source["locator_json"], source["preview"],
                     source["score_json"]))
        return self._favorite(favorite_id)

    def update_favorite(self, favorite_id, title=None, note=None, tags=None):
        if not self.database.fetchone("SELECT id FROM favorites WHERE id=?", (favorite_id,)):
            raise KeyError("收藏不存在。")
        fields, values = [], []
        if title is not None:
            title = re.sub(r"\s+", " ", title).strip()
            if not title:
                raise ValueError("收藏标题不能为空。")
            fields.append("title=?"); values.append(title[:100])
        if note is not None:
            fields.append("note=?"); values.append(note.strip()[:4000])
        if tags is not None:
            fields.append("tags_json=?")
            values.append(json.dumps(_normalize_tags(tags), ensure_ascii=False))
        if fields:
            fields.append("updated_at=?"); values.append(utc_now()); values.append(favorite_id)
            with self.database.transaction() as connection:
                connection.execute(f"UPDATE favorites SET {', '.join(fields)} WHERE id=?", values)
        return self._favorite(favorite_id)

    def delete_favorite(self, favorite_id):
        with self.database.transaction() as connection:
            cursor = connection.execute("DELETE FROM favorites WHERE id=?", (favorite_id,))
        if not cursor.rowcount:
            raise KeyError("收藏不存在。")

    def save_feedback(self, message_id, kind, note=""):
        if kind not in FEEDBACK_KINDS:
            raise ValueError("反馈类型无效。")
        message = self.database.fetchone(
            "SELECT id FROM messages WHERE id=? AND role='assistant'", (message_id,))
        if not message:
            raise KeyError("回答不存在。")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute("""INSERT INTO answer_feedback(
                id, message_id, kind, note, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET kind=excluded.kind, note=excluded.note,
                    updated_at=excluded.updated_at""",
                (_id("feedback"), message_id, kind, note.strip()[:2000], now, now))
        return {"message_id": message_id, "kind": kind, "stored_locally": True}

    def export_markdown(self, favorite_id):
        favorite = self._favorite(favorite_id)
        lines = [f"# {favorite['title']}", "", *_body_block(favorite),
                 "---", "", *_provenance_block(favorite), ""]
        path = self.paths.exports / (
            f"{safe_filename(favorite['title'], '收藏回答')}-{favorite_id[-8:]}.md")
        return _write_atomic(path, "\n".join(lines))

    def export_collection(self, title, favorite_ids):
        """把一批收藏合成一份专题 Markdown。

        `favorite_ids` 由界面按当前筛选结果和排列顺序传下来，所以导出的就是眼
        前这一批、这个次序。按日期命名：同一天用同一个标题再导一次就覆盖它，
        和单条导出一样，导出是"当前选择的快照"，不是追加的账本。
        """
        if not favorite_ids:
            raise ValueError("没有可导出的收藏。")
        favorites = [self._favorite(favorite_id) for favorite_id in favorite_ids]
        heading = re.sub(r"\s+", " ", title).strip()[:80] or "收藏专题"
        stamp = utc_now()
        lines = [f"# {heading}", "",
                 f"> 收录 {len(favorites)} 条收藏，导出于 {stamp}", "",
                 "## 目录", ""]
        lines += [f"{number}. {_question_heading(favorite)}"
                  for number, favorite in enumerate(favorites, 1)]
        for number, favorite in enumerate(favorites, 1):
            lines += ["", f"## {number}. {_question_heading(favorite)}", "",
                      *_body_block(favorite), "---", "", *_provenance_block(favorite), ""]
        path = self.paths.exports / f"{safe_filename(heading, '收藏专题')}-{stamp[:10]}.md"
        return _write_atomic(path, "\n".join(lines))