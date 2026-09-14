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
