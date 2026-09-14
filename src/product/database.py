"""SQLite 数据库、版本迁移和迁移前备份。"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import threading

from .paths import ProductPaths


MIGRATIONS = {
    1: """
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE app_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """,
    2: """
        CREATE TABLE libraries (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('folder', 'uploads')),
            root_path TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            library_id TEXT NOT NULL REFERENCES libraries(id),
            source_kind TEXT NOT NULL CHECK(source_kind IN ('folder', 'upload')),
            relative_path TEXT NOT NULL,
            display_name TEXT NOT NULL,
            media_type TEXT NOT NULL CHECK(media_type IN ('markdown', 'pdf')),
            status TEXT NOT NULL,
            error TEXT,
            source_mtime_ns INTEGER,
            source_size INTEGER,
            checksum TEXT,
            current_version_id TEXT,
            removed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(library_id, relative_path)
        );
        CREATE TABLE document_versions (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id),
            version_number INTEGER NOT NULL,
            checksum TEXT NOT NULL,
            snapshot_path TEXT NOT NULL,
            extracted_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(document_id, version_number)
        );
        CREATE TABLE document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id),
            version_id TEXT NOT NULL REFERENCES document_versions(id),
            position INTEGER NOT NULL,
            heading_path TEXT NOT NULL,
            locator_json TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE INDEX idx_chunks_version ON document_chunks(version_id, position);
        CREATE TABLE index_versions (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE import_jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            total INTEGER NOT NULL DEFAULT 0,
            completed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            message TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        );
    """,
    3: """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK(status IN ('complete', 'streaming', 'stopped', 'failed')),
            reply_to_message_id TEXT,
            retry_of_message_id TEXT,
            provider TEXT,
            model TEXT,
            index_version TEXT,
            retrieval_query TEXT,
            error_code TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);
        CREATE TABLE message_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            position INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            title TEXT NOT NULL,
            media_type TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            locator_json TEXT NOT NULL,
            preview TEXT NOT NULL,
            UNIQUE(message_id, label)
        );
        CREATE INDEX idx_message_sources_message ON message_sources(message_id, position);
    """,
    4: """
        CREATE TABLE favorites (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            index_version TEXT,
            generated_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE favorite_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            favorite_id TEXT NOT NULL REFERENCES favorites(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            position INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            title TEXT NOT NULL,
            media_type TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            locator_json TEXT NOT NULL,
            preview TEXT NOT NULL,
            UNIQUE(favorite_id, label)
        );
        CREATE TABLE answer_feedback (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK(kind IN ('helpful', 'missing', 'citation_wrong', 'answer_wrong')),
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_favorites_updated ON favorites(updated_at);
    """,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, paths: ProductPaths):
        self.paths = paths.ensure()
        self.path = self.paths.database
        self._write_lock = threading.RLock()
        self.migrate()

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @contextmanager
    def transaction(self):
        with self._write_lock, closing(self.connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def schema_version(self):
        if not self.path.exists():
            return 0
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def migrate(self):
        current = self.schema_version()
        pending = [version for version in sorted(MIGRATIONS) if version > current]
        if not pending:
            return
        if self.path.exists() and self.path.stat().st_size:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = self.paths.backups / f"workspace-before-v{pending[-1]}-{stamp}.sqlite3"
            shutil.copy2(self.path, backup)
        with self.transaction() as connection:
            for version in pending:
                connection.executescript(MIGRATIONS[version])
                connection.execute(f"PRAGMA user_version = {version}")

    def get_settings(self):
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT key, value_json FROM settings").fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def set_settings(self, values: dict):
        now = utc_now()
        with self.transaction() as connection:
            for key, value in values.items():
                connection.execute(
                    """INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
                       updated_at=excluded.updated_at""",
                    (key, json.dumps(value, ensure_ascii=False), now),
                )

    def event(self, event_type: str, detail: dict | None = None):
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO app_events(event_type, detail_json, created_at) VALUES (?, ?, ?)",
                (event_type, json.dumps(detail or {}, ensure_ascii=False), utc_now()),
            )

    def fetchall(self, sql: str, parameters=()):
        with closing(self.connect()) as connection:
            return connection.execute(sql, parameters).fetchall()

    def fetchone(self, sql: str, parameters=()):
        with closing(self.connect()) as connection:
            return connection.execute(sql, parameters).fetchone()
