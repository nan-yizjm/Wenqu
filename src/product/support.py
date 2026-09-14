"""可校验的数据备份/恢复和不含正文、密钥、路径的诊断报告。"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import uuid
import zipfile

from . import PRODUCT_VERSION
from .database import MIGRATIONS, utc_now


BACKUP_FORMAT = 1
MAX_BACKUP_BYTES = 1024 * 1024 * 1024
MAX_FILES = 20_000
INCLUDED_DIRECTORIES = ("corpus", "snapshots", "exports")


def _hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SupportService:
    def __init__(self, database, paths, settings_getter, credentials,
                 runtime_getter=lambda: {}):
        self.database, self.paths = database, paths
        self.settings_getter, self.credentials = settings_getter, credentials
        self.runtime_getter = runtime_getter

    def _database_snapshot(self, destination):
        with closing(self.database.connect()) as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)

    def create_backup(self, prefix="ObsidianRAG-backup"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = self.paths.backups / f"{prefix}-{stamp}.zip"
        staging_db = self.paths.runtime / f"backup-{uuid.uuid4().hex}.sqlite3"
        self._database_snapshot(staging_db)
        files = {"workspace.sqlite3": staging_db}
        for directory in INCLUDED_DIRECTORIES:
            root = getattr(self.paths, directory)
            for path in root.rglob("*"):
                if path.is_file():
                    files[f"{directory}/{path.relative_to(root).as_posix()}"] = path
        manifest = {
            "format": BACKUP_FORMAT, "product_version": PRODUCT_VERSION,
            "created_at": utc_now(),
            "files": {name: {"sha256": _hash(path), "size": path.stat().st_size}
                      for name, path in sorted(files.items())},
        }
        temporary = output.with_suffix(".tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False,
                                                              indent=2).encode("utf-8"))
                for name, path in sorted(files.items()):
                    archive.write(path, name)
            temporary.replace(output)
        finally:
            staging_db.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
        return output

    @staticmethod
    def _safe_member(name):
        value = PurePosixPath(name)
        return (not value.is_absolute() and ".." not in value.parts and
                value.parts and value.parts[0] in {*INCLUDED_DIRECTORIES,
                                                    "workspace.sqlite3", "manifest.json"})

    def restore_backup(self, data: bytes):
        if not data or len(data) > MAX_BACKUP_BYTES:
            raise ValueError("备份文件为空或超过 1 GB 限制。")
        stage = self.paths.runtime / f"restore-{uuid.uuid4().hex}"
        stage.mkdir()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_FILES or sum(item.file_size for item in infos) > MAX_BACKUP_BYTES:
                    raise ValueError("备份展开后超过安全限制。")
                file_names = [item.filename for item in infos if not item.is_dir()]
                names = set(file_names)
                if len(names) != len(file_names):
                    raise ValueError("备份包含重复文件名。")
                if "manifest.json" not in names or not all(self._safe_member(name) for name in names):
                    raise ValueError("备份包含非法路径或缺少清单。")
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                if manifest.get("format") != BACKUP_FORMAT or not isinstance(manifest.get("files"), dict):
                    raise ValueError("备份格式版本不受支持。")
                expected = set(manifest["files"])
                if names != expected | {"manifest.json"} or "workspace.sqlite3" not in expected:
                    raise ValueError("备份成员与清单不一致。")
                for name, metadata in manifest["files"].items():
                    raw = archive.read(name)
                    if len(raw) != metadata.get("size") or sha256(raw).hexdigest() != metadata.get("sha256"):
                        raise ValueError(f"备份文件校验失败：{name}")
                    target = stage.joinpath(*PurePosixPath(name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
            candidate = stage / "workspace.sqlite3"
            with closing(sqlite3.connect(candidate)) as connection:
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("备份数据库完整性检查失败。")
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version < 1 or version > max(MIGRATIONS):
                raise ValueError("备份数据库版本与当前产品不兼容。")

            safety = self.create_backup(prefix="before-restore")
            rollback = self.paths.runtime / f"rollback-{uuid.uuid4().hex}"
            rollback.mkdir()
            moved, installed = [], []
            targets = [(candidate, self.paths.database)] + [
                (stage / name, getattr(self.paths, name)) for name in INCLUDED_DIRECTORIES]
            try:
                for source, destination in targets:
                    old = rollback / destination.name
                    if destination.exists():
                        os.replace(destination, old); moved.append((old, destination))
                    if source.exists():
                        os.replace(source, destination); installed.append(destination)
                    elif destination.suffix == "":
                        destination.mkdir()
                return {"restored": True, "restart_required": True,
                        "safety_backup": safety.name}
            except Exception:
                for destination in reversed(installed):
                    if destination.is_dir(): shutil.rmtree(destination, ignore_errors=True)
                    else: destination.unlink(missing_ok=True)
                for old, destination in reversed(moved):
                    if old.exists(): os.replace(old, destination)
                raise
            finally:
                shutil.rmtree(rollback, ignore_errors=True)
        except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("不是有效的 Obsidian RAG 备份。") from error
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def diagnostic_report(self, model_state, material_summary):
        settings = self.settings_getter()
        safe_settings = {
            "provider": settings.get("provider"),
            "ollama_model": settings.get("ollama_model"),
            "retrieval_mode": settings.get("retrieval_mode"),
            "onboarding_complete": settings.get("onboarding_complete"),
        }
        return {
            "generated_at": utc_now(), "product_version": PRODUCT_VERSION,
            "database_schema": self.database.schema_version(),
            "settings": safe_settings,
            "deepseek_key_configured": self.credentials.has_deepseek(),
            "retrieval_model": {key: model_state.get(key) for key in
                                ("status", "model", "dimension", "device")},
            "materials": material_summary,
            "counts": {
                "conversations": self.database.fetchone("SELECT COUNT(*) n FROM conversations")["n"],
                "favorites": self.database.fetchone("SELECT COUNT(*) n FROM favorites")["n"],
                "feedback": self.database.fetchone("SELECT COUNT(*) n FROM answer_feedback")["n"],
            },
            "runtime": {"status": self.runtime_getter().get("status")},
            "privacy": "不包含密钥、问题、回答、文件名、正文或本机绝对路径",
        }

    def write_diagnostic_report(self, model_state, material_summary):
        path = self.paths.logs / f"diagnostics-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(self.diagnostic_report(model_state, material_summary),
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        return path
