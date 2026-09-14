"""资料登记、版本快照、文本提取、后台导入与可定位搜索。"""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import threading
import uuid

from ..bm25 import BM25Index, search_bm25
from .database import Database, utc_now
from .paths import ProductPaths


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 2000
MAX_CHUNK_CHARS = 1600
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _split_lines(lines: list[tuple[int, str]]):
    pieces, current, size = [], [], 0
    for line_number, line in lines:
        extra = len(line) + (1 if current else 0)
        if current and size + extra > MAX_CHUNK_CHARS:
            pieces.append(current)
            current, size = [], 0
        current.append((line_number, line))
        size += extra
    if current:
        pieces.append(current)
    return pieces


def markdown_chunks(text: str, fallback_title: str) -> list[dict]:
    """按标题切分，同时保留原文件的一基行号。"""
    sections, headings, body = [], [fallback_title], []
    in_fence = False

    def flush():
        nonlocal body
        meaningful = [(number, line) for number, line in body if line.strip()]
        for piece in _split_lines(meaningful):
            sections.append({
                "heading_path": " > ".join(headings),
                "locator": {"kind": "markdown", "start_line": piece[0][0],
                            "end_line": piece[-1][0]},
                "text": "\n".join(line for _, line in piece),
            })
        body = []

    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            body.append((line_number, line))
            continue
        match = None if in_fence else HEADING.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            if level == 1:
                headings = [title]
            else:
                parent = headings[:level - 1] or [fallback_title]
                headings = parent + [title]
        else:
            body.append((line_number, line))
    flush()
    return sections


def pdf_chunks(data: bytes, title: str) -> tuple[list[dict], str | None]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF 已加密，当前版本无法读取。")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF 页数超过限制（{MAX_PDF_PAGES} 页）。")
    chunks, empty_pages = [], []
    for page_number, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as error:
            empty_pages.append(page_number)
            continue
        if not text:
            empty_pages.append(page_number)
            continue
        numbered = list(enumerate(text.splitlines(), 1))
        for piece in _split_lines(numbered):
            chunks.append({
                "heading_path": f"{title} > 第 {page_number} 页",
                "locator": {"kind": "pdf", "page": page_number},
                "text": "\n".join(line for _, line in piece),
            })
    if not chunks:
        raise ValueError("PDF 未提取到文字，可能是扫描件；首版暂不提供 OCR。")
    warning = f"{len(empty_pages)} 页未提取到文字" if empty_pages else None
    return chunks, warning


@dataclass(frozen=True)
class SearchSnapshot:
    version_id: str | None
    chunks: tuple[dict, ...]
    index: BM25Index


class MaterialService:
    def __init__(self, database: Database, paths: ProductPaths, run_inline=False):
        self.database, self.paths = database, paths
        self.run_inline = run_inline
        self._snapshot_lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._recover_interrupted_jobs()
        self._snapshot = self._load_snapshot()

    def _recover_interrupted_jobs(self):
        """进程中断后留下可理解、可重试的状态，同时保留旧可用版本。"""
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute("""
                UPDATE import_jobs SET status='failed', message='程序在导入完成前退出，请重试。',
                    finished_at=? WHERE status IN ('pending', 'running')
            """, (now,))
            connection.execute("""
                UPDATE documents SET status='failed', error='导入被中断，请重试。', updated_at=?
                WHERE status='processing' AND current_version_id IS NULL
            """, (now,))
            connection.execute("""
                UPDATE documents SET status='ready', error='上次更新被中断，当前仍使用旧版本。',
                    updated_at=? WHERE status='updating' AND current_version_id IS NOT NULL
            """, (now,))

    def _load_snapshot(self):
        rows = self.database.fetchall("""
            SELECT c.*, d.display_name, d.media_type
            FROM document_chunks c
            JOIN documents d ON d.id=c.document_id AND d.current_version_id=c.version_id
            WHERE d.removed_at IS NULL AND d.status IN ('ready', 'updating')
            ORDER BY d.id, c.position
        """)
        chunks = tuple({
            "id": row["id"], "document_id": row["document_id"],
            "version_id": row["version_id"], "document_title": row["display_name"],
            "media_type": row["media_type"], "heading_path": row["heading_path"],
            "locator": json.loads(row["locator_json"]), "text": row["text"],
        } for row in rows)
        active = self.database.get_settings().get("active_index_version")
        return SearchSnapshot(active, chunks, BM25Index(list(chunks)))

    def _publish_snapshot(self):
        rows = self.database.fetchall("""
            SELECT c.*, d.display_name, d.media_type
            FROM document_chunks c
            JOIN documents d ON d.id=c.document_id AND d.current_version_id=c.version_id
            WHERE d.removed_at IS NULL AND d.status IN ('ready', 'updating')
            ORDER BY d.id, c.position
        """)
        chunks = [{
            "id": row["id"], "document_id": row["document_id"],
            "version_id": row["version_id"], "document_title": row["display_name"],
            "media_type": row["media_type"], "heading_path": row["heading_path"],
            "locator": json.loads(row["locator_json"]), "text": row["text"],
        } for row in rows]
        version_id = _id("idx")
        documents = len({chunk["document_id"] for chunk in chunks})
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO index_versions VALUES (?, 'ready', ?, ?, ?)",
                (version_id, documents, len(chunks), utc_now()),
            )
            connection.execute("""
                INSERT INTO settings(key, value_json, updated_at) VALUES('active_index_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
                    updated_at=excluded.updated_at
            """, (json.dumps(version_id), utc_now()))
        prepared = SearchSnapshot(version_id, tuple(chunks), BM25Index(chunks))
        with self._snapshot_lock:
            self._snapshot = prepared
        return prepared

    def setup_summary(self):
        row = self.database.fetchone("""
            SELECT COUNT(*) total,
                   SUM(CASE WHEN status IN ('ready', 'updating') AND removed_at IS NULL THEN 1 ELSE 0 END) ready
            FROM documents
        """)
        with self._snapshot_lock:
            snapshot = self._snapshot
        return {"total_documents": int(row["total"] or 0), "ready_documents": int(row["ready"] or 0),
                "chunk_count": len(snapshot.chunks), "index_version": snapshot.version_id}

    def list_libraries(self):
        rows = self.database.fetchall("""
            SELECT l.*, COUNT(d.id) document_count,
                   SUM(CASE WHEN d.status IN ('ready', 'updating') AND d.removed_at IS NULL THEN 1 ELSE 0 END) ready_count
            FROM libraries l LEFT JOIN documents d ON d.library_id=l.id
            WHERE l.active=1 GROUP BY l.id ORDER BY l.created_at
        """)
        return [dict(row) for row in rows]

    def list_documents(self):
        rows = self.database.fetchall("""
            SELECT d.id, d.library_id, d.relative_path, d.display_name, d.media_type,
                   d.status, d.error, d.current_version_id, d.updated_at, l.name library_name
            FROM documents d JOIN libraries l ON l.id=d.library_id
            WHERE d.removed_at IS NULL AND l.active=1
            ORDER BY d.updated_at DESC, d.display_name
        """)
        return [dict(row) for row in rows]

    def list_jobs(self):
        return [dict(row) for row in self.database.fetchall(
            "SELECT * FROM import_jobs ORDER BY created_at DESC LIMIT 50")]

    def _job(self, job_type: str, payload: dict):
        job_id = _id("job")
        with self.database.transaction() as connection:
            connection.execute("""
                INSERT INTO import_jobs(id, job_type, status, payload_json, created_at)
                VALUES (?, ?, 'pending', ?, ?)
            """, (job_id, job_type, json.dumps(payload, ensure_ascii=False), utc_now()))
        return job_id

    def _dispatch(self, job_id: str, target, *args):
        def run():
            try:
                with self._write_lock:
                    target(job_id, *args)
            except Exception as error:
                with self.database.transaction() as connection:
                    connection.execute("""
                        UPDATE import_jobs SET status='failed', message=?, finished_at=? WHERE id=?
                    """, (str(error)[:500], utc_now(), job_id))
        if self.run_inline:
            run()
        else:
            threading.Thread(target=run, name=f"import-{job_id}", daemon=True).start()

    def connect_folder(self, root_path: str, name: str | None = None):
        root = Path(root_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("所选 Markdown 文件夹不存在或不可读取。")
        existing = self.database.fetchone(
            "SELECT id FROM libraries WHERE kind='folder' AND active=1 AND root_path=?",
            (str(root),))
        if existing:
            result = self.refresh_library(existing["id"])
            return {"library_id": existing["id"], **result, "already_connected": True}
        library_id, now = _id("lib"), utc_now()
        with self.database.transaction() as connection:
            connection.execute("""
                INSERT INTO libraries VALUES (?, ?, 'folder', ?, 1, ?, ?)
            """, (library_id, (name or root.name)[:100], str(root), now, now))
        job_id = self._job("scan_folder", {"library_id": library_id})
        self._dispatch(job_id, self._scan_folder, library_id)
        return {"library_id": library_id, "job_id": job_id}

    def refresh_library(self, library_id: str):
        row = self.database.fetchone(
            "SELECT * FROM libraries WHERE id=? AND active=1 AND kind='folder'", (library_id,))
        if not row:
            raise KeyError("资料文件夹不存在。")
        job_id = self._job("scan_folder", {"library_id": library_id})
        self._dispatch(job_id, self._scan_folder, library_id)
        return {"job_id": job_id}

    def upload(self, filename: str, data: bytes):
        if not data:
            raise ValueError("上传文件为空。")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("文件超过 50 MB 上传限制。")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".md", ".markdown", ".pdf"}:
            raise ValueError("只支持 Markdown 和 PDF 文件。")
        library = self.database.fetchone("SELECT * FROM libraries WHERE kind='uploads' AND active=1")
        if library:
            library_id = library["id"]
        else:
            library_id, now = _id("lib"), utc_now()
            with self.database.transaction() as connection:
                connection.execute("INSERT INTO libraries VALUES (?, '上传文件', 'uploads', NULL, 1, ?, ?)",
                                   (library_id, now, now))
        document_id = _id("doc")
        relative = f"{document_id}-{Path(filename).name}"
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute("""
                INSERT INTO documents(id, library_id, source_kind, relative_path, display_name,
                    media_type, status, source_size, created_at, updated_at)
                VALUES (?, ?, 'upload', ?, ?, ?, 'processing', ?, ?, ?)
            """, (document_id, library_id, relative, Path(filename).name,
                  "pdf" if suffix == ".pdf" else "markdown", len(data), now, now))
        pending = self.paths.runtime / "uploads" / relative
        _atomic_write(pending, data)
        job_id = self._job("upload", {"document_id": document_id})
        self._dispatch(job_id, self._import_upload, document_id, pending)
        return {"document_id": document_id, "job_id": job_id}

    def _start_job(self, job_id: str, total: int):
        with self.database.transaction() as connection:
            connection.execute("""
                UPDATE import_jobs SET status='running', total=?, started_at=? WHERE id=?
            """, (total, utc_now(), job_id))

    def _advance_job(self, job_id: str, failed=False, message=None):
        with self.database.transaction() as connection:
            connection.execute("""
                UPDATE import_jobs SET completed=completed+1, failed=failed+?, message=? WHERE id=?
            """, (1 if failed else 0, message, job_id))

    def _finish_job(self, job_id: str):
        with self.database.transaction() as connection:
            row = connection.execute("SELECT failed FROM import_jobs WHERE id=?", (job_id,)).fetchone()
            status = "completed_with_errors" if row["failed"] else "completed"
            connection.execute(
                "UPDATE import_jobs SET status=?, finished_at=? WHERE id=?",
                (status, utc_now(), job_id),
            )

    def _scan_folder(self, job_id: str, library_id: str):
        library = self.database.fetchone("SELECT * FROM libraries WHERE id=? AND active=1", (library_id,))
        if not library:
            raise ValueError("资料文件夹已被移除。")
        root = Path(library["root_path"])
        files = sorted(path for path in root.rglob("*") if path.is_file()
                       and path.suffix.lower() in {".md", ".markdown"}
                       and not any(part.startswith(".") for part in path.relative_to(root).parts))
        self._start_job(job_id, len(files))
        seen, changed = set(), False
        for path in files:
            relative = path.relative_to(root).as_posix()
            seen.add(relative)
            try:
                changed = self._import_folder_file(library_id, path, relative) or changed
                self._advance_job(job_id)
            except Exception as error:
                row = self.database.fetchone(
                    "SELECT id FROM documents WHERE library_id=? AND relative_path=?",
                    (library_id, relative))
                if row:
                    with self.database.transaction() as connection:
                        current = connection.execute(
                            "SELECT current_version_id FROM documents WHERE id=?", (row["id"],)).fetchone()
                        connection.execute(
                            "UPDATE documents SET status='failed', error=?, updated_at=? WHERE id=?",
                            (str(error)[:500], utc_now(), row["id"]),
                        )
                        if current["current_version_id"]:
                            connection.execute(
                                "UPDATE documents SET status='ready', error=? WHERE id=?",
                                (f"更新失败，仍使用旧版本：{str(error)[:380]}", row["id"]),
                            )
                self._advance_job(job_id, True, f"{relative}: {str(error)[:300]}")
        with self.database.transaction() as connection:
            current = connection.execute("""
                SELECT id, relative_path FROM documents
                WHERE library_id=? AND source_kind='folder' AND removed_at IS NULL
            """, (library_id,)).fetchall()
            for row in current:
                if row["relative_path"] not in seen:
                    connection.execute("""
                        UPDATE documents SET status='removed', removed_at=?, updated_at=? WHERE id=?
                    """, (utc_now(), utc_now(), row["id"]))
                    changed = True
        if changed:
            self._publish_snapshot()
        self._finish_job(job_id)

    def _import_folder_file(self, library_id: str, path: Path, relative: str):
        before = path.stat()
        data = path.read_bytes()
        stat = path.stat()
        if (before.st_mtime_ns, before.st_size) != (stat.st_mtime_ns, stat.st_size):
            raise ValueError("文件在导入过程中发生变化，请稍后刷新重试。")
        row = self.database.fetchone(
            "SELECT * FROM documents WHERE library_id=? AND relative_path=?", (library_id, relative))
        if row and row["checksum"] == sha256(data).hexdigest() and row["removed_at"] is None:
            return False
        if row:
            document_id = row["id"]
            with self.database.transaction() as connection:
                connection.execute("""
                    UPDATE documents SET status=?, removed_at=NULL, error=NULL,
                        source_mtime_ns=?, source_size=?, updated_at=? WHERE id=?
                """, ("updating" if row["current_version_id"] else "processing",
                      stat.st_mtime_ns, stat.st_size, utc_now(), document_id))
        else:
            document_id, now = _id("doc"), utc_now()
            with self.database.transaction() as connection:
                connection.execute("""
                    INSERT INTO documents(id, library_id, source_kind, relative_path, display_name,
                        media_type, status, source_mtime_ns, source_size, created_at, updated_at)
                    VALUES (?, ?, 'folder', ?, ?, 'markdown', 'processing', ?, ?, ?, ?)
                """, (document_id, library_id, relative, path.name, stat.st_mtime_ns,
                      stat.st_size, now, now))
        self._store_version(document_id, path.name, "markdown", data)
        return True

    def _import_upload(self, job_id: str, document_id: str, pending: Path):
        self._start_job(job_id, 1)
        row = self.database.fetchone("SELECT * FROM documents WHERE id=?", (document_id,))
        try:
            self._store_version(document_id, row["display_name"], row["media_type"], pending.read_bytes())
            self._advance_job(job_id)
            self._publish_snapshot()
            pending.unlink(missing_ok=True)
        except Exception as error:
            with self.database.transaction() as connection:
                connection.execute("UPDATE documents SET status='failed', error=?, updated_at=? WHERE id=?",
                                   (str(error)[:500], utc_now(), document_id))
            self._advance_job(job_id, True, str(error)[:500])
        self._finish_job(job_id)

    def _store_version(self, document_id: str, title: str, media_type: str, data: bytes):
        digest = sha256(data).hexdigest()
        if media_type == "markdown":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ValueError("Markdown 不是有效的 UTF-8 编码，请转换编码后重试。") from error
            chunks, warning = markdown_chunks(text, Path(title).stem), None
            suffix = ".md"
        else:
            chunks, warning = pdf_chunks(data, title)
            text, suffix = "\n\n".join(chunk["text"] for chunk in chunks), ".pdf"
        if not chunks:
            raise ValueError("文档没有可检索的文字内容。")
        row = self.database.fetchone(
            "SELECT COALESCE(MAX(version_number), 0) n FROM document_versions WHERE document_id=?",
            (document_id,))
        number, version_id = int(row["n"]) + 1, _id("ver")
        snapshot_rel = f"{document_id}/{version_id}{suffix}"
        extracted_rel = f"{document_id}/{version_id}.txt"
        _atomic_write(self.paths.snapshots / snapshot_rel, data)
        _atomic_write(self.paths.corpus / extracted_rel, text.encode("utf-8"))
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute("""
                INSERT INTO document_versions VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (version_id, document_id, number, digest, snapshot_rel, extracted_rel, now))
            for position, chunk in enumerate(chunks):
                connection.execute("""
                    INSERT INTO document_chunks VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (_id("chk"), document_id, version_id, position, chunk["heading_path"],
                      json.dumps(chunk["locator"], ensure_ascii=False), chunk["text"]))
            connection.execute("""
                UPDATE documents SET status='ready', error=?, checksum=?, current_version_id=?,
                    removed_at=NULL, updated_at=? WHERE id=?
            """, (warning, digest, version_id, now, document_id))

    def remove_document(self, document_id: str):
        row = self.database.fetchone("SELECT id FROM documents WHERE id=? AND removed_at IS NULL",
                                     (document_id,))
        if not row:
            raise KeyError("资料不存在。")
        with self.database.transaction() as connection:
            connection.execute("UPDATE documents SET status='removed', removed_at=?, updated_at=? WHERE id=?",
                               (utc_now(), utc_now(), document_id))
        self._publish_snapshot()

    def retry_document(self, document_id: str):
        row = self.database.fetchone(
            "SELECT * FROM documents WHERE id=? AND removed_at IS NULL", (document_id,))
        if not row:
            raise KeyError("资料不存在。")
        if row["source_kind"] == "folder":
            return self.refresh_library(row["library_id"])
        pending = self.paths.runtime / "uploads" / row["relative_path"]
        if not pending.is_file():
            raise ValueError("原上传文件已不存在，请重新上传。")
        with self.database.transaction() as connection:
            connection.execute("UPDATE documents SET status='processing', error=NULL, updated_at=? WHERE id=?",
                               (utc_now(), document_id))
        job_id = self._job("upload", {"document_id": document_id})
        self._dispatch(job_id, self._import_upload, document_id, pending)
        return {"job_id": job_id}

    def retrieve(self, query: str, top_k=8):
        """返回本次不可变快照上的完整检索片段，供搜索页和问答共同使用。"""
        with self._snapshot_lock:
            snapshot = self._snapshot
        results = search_bm25(query, list(snapshot.chunks), snapshot.index, top_k=top_k)
        return {"query": query, "index_version": snapshot.version_id, "results": [{
            "chunk_id": chunk["id"], "document_id": chunk["document_id"],
            "version_id": chunk["version_id"], "title": chunk["document_title"],
            "media_type": chunk["media_type"], "heading_path": chunk["heading_path"],
            "locator": chunk["locator"], "preview": chunk["text"][:360],
            "text": chunk["text"],
            "score": round(score, 4), "matched_tokens": sorted(tokens),
        } for chunk, score, tokens in results]}

    def search(self, query: str, top_k=8):
        result = self.retrieve(query, top_k)
        for item in result["results"]:
            item.pop("text", None)
        return result

    def source(self, document_id: str, version_id: str):
        row = self.database.fetchone("""
            SELECT v.*, d.display_name, d.media_type
            FROM document_versions v JOIN documents d ON d.id=v.document_id
            WHERE v.id=? AND v.document_id=?
        """, (version_id, document_id))
        if not row:
            raise KeyError("来源版本不存在。")
        extracted = (self.paths.corpus / row["extracted_path"]).resolve()
        if not extracted.is_relative_to(self.paths.corpus.resolve()) or not extracted.is_file():
            raise ValueError("来源快照损坏或已丢失。")
        return {"document_id": document_id, "version_id": version_id,
                "title": row["display_name"], "media_type": row["media_type"],
                "text": extracted.read_text(encoding="utf-8")}

    def source_file(self, document_id: str, version_id: str):
        row = self.database.fetchone("""
            SELECT v.snapshot_path, d.media_type, d.display_name
            FROM document_versions v JOIN documents d ON d.id=v.document_id
            WHERE v.id=? AND v.document_id=?
        """, (version_id, document_id))
        if not row:
            raise KeyError("来源版本不存在。")
        path = (self.paths.snapshots / row["snapshot_path"]).resolve()
        if not path.is_relative_to(self.paths.snapshots.resolve()) or not path.is_file():
            raise ValueError("来源快照损坏或已丢失。")
        return path, row["media_type"], row["display_name"]
