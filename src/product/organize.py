"""收藏、回答反馈与可直接放入 Obsidian 的 Markdown 导出。"""

from __future__ import annotations

import json
import re
import uuid

from .chat import source_record
from .database import Database, utc_now


FEEDBACK_KINDS = {"helpful", "missing", "citation_wrong", "answer_wrong"}


def _id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _location(source):
    locator = source["locator"]
    if locator.get("kind") == "pdf":
        return f"第 {locator.get('page')} 页"
    if locator.get("kind") == "notebook":
        return f"单元格 {locator.get('cell')}（第 {locator.get('start_line')}–{locator.get('end_line')} 行）"
    return f"第 {locator.get('start_line')}–{locator.get('end_line')} 行"


class OrganizeService:
    def __init__(self, database: Database, paths):
        self.database, self.paths = database, paths

    def _favorite(self, favorite_id):
        row = self.database.fetchone("SELECT * FROM favorites WHERE id=?", (favorite_id,))
        if not row:
            raise KeyError("收藏不存在。")
        sources = [source_record(item) for item in self.database.fetchall(
            "SELECT * FROM favorite_sources WHERE favorite_id=? ORDER BY position", (favorite_id,))]
        return {**dict(row), "sources": sources}

    def list_favorites(self):
        rows = self.database.fetchall("""SELECT f.*,
            (SELECT COUNT(*) FROM favorite_sources s WHERE s.favorite_id=f.id) source_count
            FROM favorites f ORDER BY updated_at DESC""")
        return [dict(row) for row in rows]

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

    def update_favorite(self, favorite_id, title=None, note=None):
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
        lines = [f"# {favorite['title']}", "", f"> 问题：{favorite['question']}", ""]
        if favorite["note"]:
            lines += ["## 备注", "", favorite["note"], ""]
        lines += ["## 回答", "", favorite["answer"], "", "## 来源", ""]
        for source in favorite["sources"]:
            lines.append(
                f"- [{source['label']}] {source['title']} — {source['heading_path']}（{_location(source)}）")
        lines += ["", "---", "",
                  f"生成时间：{favorite['generated_at'] or favorite['created_at']}",
                  f"模型：{favorite['provider'] or '未知'} / {favorite['model'] or '未知'}",
                  f"索引版本：{favorite['index_version'] or '未知'}", ""]
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", favorite["title"]).strip(" .")[:80]
        if safe_name.upper() in {"CON", "PRN", "AUX", "NUL",
                                  *(f"COM{i}" for i in range(1, 10)),
                                  *(f"LPT{i}" for i in range(1, 10))}:
            safe_name = f"收藏-{safe_name}"
        path = self.paths.exports / f"{safe_name or '收藏回答'}-{favorite_id[-8:]}.md"
        temporary = path.with_suffix(".tmp")
        temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        temporary.replace(path)
        return path
