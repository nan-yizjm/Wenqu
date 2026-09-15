"""资料登记、版本快照、文本提取、后台导入与可定位搜索。"""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import threading
import uuid

from ..bm25 import BM25Index
from ..hybrid_retrieve import RetrievalEngine
from . import resources
from .database import Database, utc_now
from .paths import ProductPaths


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 2000
# E5 的窗口是 512 token，超出的部分会被静默截断。本机笔记语料上实测（739 个
# 产品片段，字符/token 约 2.23）：1600 字时 18.4% 的片段超窗，1200 字 12.5%，
# 1000 字 7.3%，800 字 0.9%。取 800：截断基本消失，片段数只多 28%。
MAX_CHUNK_CHARS = 800
MAX_NOTEBOOK_CELLS = 5000
MAX_CELL_OUTPUT_CHARS = 4000
# 混合检索的候选窗口；接口把 top_k 限死在 20，正好等于这个值。
CANDIDATE_K = 20
RRF_K = 60
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MEDIA_TYPES = {".md": "markdown", ".markdown": "markdown", ".pdf": "pdf", ".ipynb": "notebook"}


def media_type_for(name: str) -> str | None:
    return MEDIA_TYPES.get(Path(name).suffix.lower())


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


def _joined_text(value) -> str:
    """nbformat 的字符串字段既可能是 str 也可能是逐行 list。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part for part in value if isinstance(part, str))
    return ""


def _output_text(output: dict) -> tuple[str, bool]:
    """取一段输出的纯文本，返回值第二项表示是否丢弃了富媒体。"""
    kind = output.get("output_type")
    if kind == "stream":
        return _joined_text(output.get("text")), False
    if kind in ("execute_result", "display_data"):
        data = output.get("data")
        if not isinstance(data, dict):
            return "", False
        return _joined_text(data.get("text/plain")), any(key != "text/plain" for key in data)
    return "", False


def notebook_chunks(data: bytes, title: str) -> tuple[list[dict], str, str | None]:
    """把 .ipynb 拆成可检索片段，并给出配对的纯文本。

    返回的 text 是各单元格按顺序拼起来的，locator 里的行号就指向它——来源
    面板靠行号高亮，所以两边必须来自同一次拼接。
    """
    try:
        notebook = json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as error:
        raise ValueError("Notebook 不是有效的 UTF-8 编码。") from error
    except json.JSONDecodeError as error:
        raise ValueError("Notebook 不是有效的 JSON，可能已损坏。") from error
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise ValueError("Notebook 缺少 cells，暂不支持这种格式。")
    if len(cells) > MAX_NOTEBOOK_CELLS:
        raise ValueError(f"Notebook 单元格数量超过限制（{MAX_NOTEBOOK_CELLS} 个）。")

    chunks: list[dict] = []
    body: list[str] = []
    dropped_media = truncated = 0

    def emit(block: str) -> int:
        """写入一个文本块，返回它在一基行号下的起始行。"""
        start = len(body) + 1
        body.extend(block.splitlines() or [""])
        return start

    def section(label: str):
        emit(f"--- 单元格 {label} ---")

    def collect(index: int, cell_type: str, label: str, start: int, text: str):
        """把一段单元格文本切片；行号与拼接文本严格对齐。"""
        numbered = [(start + offset, line) for offset, line in enumerate(text.splitlines())]
        for piece in _split_lines([item for item in numbered if item[1].strip()]):
            chunks.append({
                "heading_path": f"{title} > 单元格 {index} · {label}",
                "locator": {"kind": "notebook", "cell": index, "cell_type": cell_type,
                            "start_line": piece[0][0], "end_line": piece[-1][0]},
                "text": "\n".join(line for _, line in piece),
            })

    for index, cell in enumerate(cells, 1):
        if not isinstance(cell, dict):
            continue
        source = _joined_text(cell.get("source")).rstrip("\n")
        cell_type = cell.get("cell_type")
        if cell_type == "code":
            if source.strip():
                section(f"{index} · 代码")
                collect(index, "code", "代码", emit(source), source)
            outputs = cell.get("outputs")
            collected = []
            for output in outputs if isinstance(outputs, list) else []:
                if not isinstance(output, dict):
                    continue
                text, rich = _output_text(output)
                dropped_media += int(rich)
                if text.strip():
                    collected.append(text.rstrip("\n"))
            if collected:
                joined = "\n".join(collected)
                cut = len(joined) > MAX_CELL_OUTPUT_CHARS
                if cut:
                    joined, truncated = joined[:MAX_CELL_OUTPUT_CHARS], truncated + 1
                section(f"{index} · 输出")
                if cut:
                    emit(f"（输出超过 {MAX_CELL_OUTPUT_CHARS} 字，已截断）")
                collect(index, "output", "输出", emit(joined), joined)
        elif source.strip():
            # markdown 与 raw 都是正文，交给 markdown_chunks 复用标题切分；
            # 行号按单元格在拼接文本里的起点整体平移。
            section(f"{index} · {cell_type or 'markdown'}")
            cell_start = emit(source)
            for piece in markdown_chunks(source, f"单元格 {index}"):
                chunks.append({
                    "heading_path": f"{title} > {piece['heading_path']}",
                    "locator": {"kind": "notebook", "cell": index,
                                "cell_type": cell_type or "markdown",
                                "start_line": cell_start + piece["locator"]["start_line"] - 1,
                                "end_line": cell_start + piece["locator"]["end_line"] - 1},
                    "text": piece["text"],
                })

    notes = []
    if dropped_media:
        notes.append(f"{dropped_media} 处图片或富媒体输出未收录")
    if truncated:
        notes.append(f"{truncated} 处输出过长已截断")
    return chunks, "\n".join(body), "；".join(notes) or None


@dataclass(frozen=True)
class SearchSnapshot:
    version_id: str | None
    chunks: tuple[dict, ...]
    index: BM25Index
    engine: RetrievalEngine


class MaterialService:
    def __init__(self, database: Database, paths: ProductPaths, run_inline=False,
                 settings_getter=None, encoder_factory=None, model_ready=None):
        self.database, self.paths = database, paths
        self.run_inline = run_inline
        self._settings_getter = settings_getter or (lambda: {})
        # 返回已加载的编码器；只允许在后台任务线程里调用，模型没准备好会返回 None。
        self._encoder_factory = encoder_factory
        # 便宜的就绪判断，不加载模型，可以在请求线程上调用。
        self._model_ready = model_ready or (lambda: True)
        self._encoder = None
        self._snapshot_lock = threading.RLock()
        self._vector_lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._vector_ready: tuple[str, object] | None = None
        self._vector_building: str | None = None
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
        engine = self._build_engine(active, list(chunks))
        return SearchSnapshot(active, chunks, engine.bm25, engine)

    def _vector_cache_path(self, version_id: str) -> Path:
        return self.paths.indexes / version_id / "vector_index.npz"

    def _build_engine(self, version_id: str | None, chunks: list[dict]):
        """每个索引版本一个引擎；向量索引随后台任务挂上去，挂之前只走 BM25。"""
        return RetrievalEngine(
            chunks, device="cpu", candidate_k=CANDIDATE_K, rrf_k=RRF_K,
            cache_path=self._vector_cache_path(version_id) if version_id else None,
        )

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
        engine = self._build_engine(version_id, chunks)
        prepared = SearchSnapshot(version_id, tuple(chunks), engine.bm25, engine)
        with self._snapshot_lock:
            self._snapshot = prepared
        # 上一版的向量索引故意留着：它已经不是"当前版本"，`retrieve()` 靠版本号
        # 判断会直接忽略它，但它是增量复用的唯一来源，清掉就等于整库重编码。
        self.ensure_vector_index()
        return prepared

    def _encoder_instance(self):
        """只在后台任务线程里调用：这里可能会加载模型。"""
        if self._encoder is None and self._encoder_factory is not None:
            self._encoder = self._encoder_factory()
        return self._encoder

    def _vector_state(self, snapshot: SearchSnapshot | None = None):
        with self._vector_lock:
            ready, building = self._vector_ready, self._vector_building
        if snapshot is None:
            with self._snapshot_lock:
                snapshot = self._snapshot
        current = snapshot.version_id
        if ready is not None and ready[0] == current:
            return {"status": "ready", "version_id": current}
        if building is not None and building == current:
            return {"status": "building", "version_id": current}
        return {"status": "off", "version_id": current}

    def ensure_vector_index(self):
        """按需为当前版本排一个向量索引任务；已经有或正在建就什么都不做。"""
        with self._snapshot_lock:
            snapshot = self._snapshot
        if not snapshot.chunks or snapshot.version_id is None:
            return None
        if self._settings_getter().get("retrieval_mode") != "hybrid":
            return None
        if self._encoder_factory is None or not self._model_ready():
            return None
        with self._vector_lock:
            ready = self._vector_ready
            if (ready is not None and ready[0] == snapshot.version_id) \
                    or self._vector_building == snapshot.version_id:
                return None
        with self._vector_lock:
            self._vector_building = snapshot.version_id
        job_id = self._job("build_vector_index", {
            "version_id": snapshot.version_id, "chunks": len(snapshot.chunks)})
        self._dispatch(job_id, self._build_vector_index, snapshot.version_id)
        return job_id

    def _build_vector_index(self, job_id: str, version_id: str):
        from ..vector_retrieve import VectorIndex

        try:
            self._start_job(job_id, 1)
            with self._snapshot_lock:
                snapshot = self._snapshot
            if snapshot.version_id != version_id:
                raise ValueError("资料在建立向量索引期间发生了变化，这一版已作废。")
            encoder = self._encoder_instance()
            if encoder is None:
                raise ValueError("检索模型尚未准备完成，请先在设置页准备检索模型。")
            chunks = list(snapshot.chunks)
            index = VectorIndex(
                chunks, device="cpu", encoder=encoder,
                cache_path=self._vector_cache_path(version_id),
                reuse_from=self._previous_vector_index(),
            )
            with self._snapshot_lock:
                current = self._snapshot
            if current.version_id == version_id:
                current.engine.vector = index
                with self._vector_lock:
                    self._vector_ready = (version_id, index)
            self._advance_job(job_id)
            self._finish_job(job_id)
        finally:
            # 失败也要放开，否则这个版本再也不会重试。
            with self._vector_lock:
                if self._vector_building == version_id:
                    self._vector_building = None

    def _previous_vector_index(self):
        """给增量复用找一个旧缓存：内存里那一份就是上一版。

        索引版本号是 `index_versions` 的主键，`document_chunks.version_id` 指的
        却是文档版本——两个号不在同一个空间里，所以没有"这一版索引当初用了哪些
        片段"的持久记录，重启后就反查不出来。重启情况下要么整库指纹没变、缓存
        直接命中，要么就老老实实重新编码一遍。
        """
        with self._vector_lock:
            previous = self._vector_ready
        if previous is None:
            return None
        _, index = previous
        if index.cache_path is None or not Path(index.cache_path).is_file():
            return None
        return index.chunks, index.cache_path

    def setup_summary(self):
        row = self.database.fetchone("""
            SELECT COUNT(*) total,
                   SUM(CASE WHEN status IN ('ready', 'updating') AND removed_at IS NULL THEN 1 ELSE 0 END) ready
            FROM documents
        """)
        with self._snapshot_lock:
            snapshot = self._snapshot
        return {"total_documents": int(row["total"] or 0), "ready_documents": int(row["ready"] or 0),
                "chunk_count": len(snapshot.chunks), "index_version": snapshot.version_id,
                "vector_index": self._vector_state(snapshot)}

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
            raise ValueError("所选资料文件夹不存在或不可读取。")
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
        media_type = media_type_for(filename)
        if media_type is None:
            raise ValueError("只支持 Markdown、PDF 和 Jupyter Notebook 文件。")
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
                  media_type, len(data), now, now))
        pending = self.paths.runtime / "uploads" / relative
        _atomic_write(pending, data)
        job_id = self._job("upload", {"document_id": document_id})
        self._dispatch(job_id, self._import_upload, document_id, pending)
        return {"document_id": document_id, "job_id": job_id}

    def import_bundled(self, name: str):
        """把随包示例导入上传资料库；同一份示例重复导入是幂等的。"""
        path = resources.resolve_bundled("examples", name)
        if path is None:
            raise KeyError("随包示例不存在。")
        data = path.read_bytes()
        digest = sha256(data).hexdigest()
        existing = self.database.fetchone("""
            SELECT d.id, d.checksum FROM documents d JOIN libraries l ON l.id=d.library_id
            WHERE l.kind='uploads' AND d.display_name=? AND d.removed_at IS NULL
        """, (name,))
        if existing and existing["checksum"] == digest:
            return {"document_id": existing["id"], "job_id": None, "already_imported": True}
        if existing:
            self.remove_document(existing["id"])
        return {**self.upload(name, data), "already_imported": False}

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
                       and media_type_for(path.name) is not None
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
        media_type = media_type_for(path.name)
        if media_type is None:
            raise ValueError("这个文件类型暂不支持。")
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
                    VALUES (?, ?, 'folder', ?, ?, ?, 'processing', ?, ?, ?, ?)
                """, (document_id, library_id, relative, path.name, media_type, stat.st_mtime_ns,
                      stat.st_size, now, now))
        self._store_version(document_id, path.name, media_type, data)
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
        elif media_type == "notebook":
            chunks, text, warning = notebook_chunks(data, Path(title).stem)
            suffix = ".ipynb"
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
        """返回本次不可变快照上的完整检索片段，供搜索页和问答共同使用。

        混合检索要求当前版本的向量索引已经建好；没建好就退回 BM25，并把
        实际用的方式报给界面，而不是让用户以为开了却什么都没变。
        """
        with self._snapshot_lock:
            snapshot = self._snapshot
        requested = self._settings_getter().get("retrieval_mode", "bm25")
        with self._vector_lock:
            ready = self._vector_ready
        hybrid = requested == "hybrid" and ready is not None and ready[0] == snapshot.version_id
        hits = snapshot.engine.search(
            query, method="hybrid" if hybrid else "bm25", top_k=top_k)
        return {
            "query": query, "index_version": snapshot.version_id,
            "retrieval_mode": requested, "effective_mode": "hybrid" if hybrid else "bm25",
            "results": [{
                "chunk_id": hit.chunk["id"], "document_id": hit.chunk["document_id"],
                "version_id": hit.chunk["version_id"], "title": hit.chunk["document_title"],
                "media_type": hit.chunk["media_type"], "heading_path": hit.chunk["heading_path"],
                "locator": hit.chunk["locator"], "preview": hit.chunk["text"][:360],
                "text": hit.chunk["text"],
                "score": round(hit.score, 6), "matched_tokens": sorted(hit.matched_tokens),
                "channels": dict(hit.ranks),
                "channel_scores": {name: round(value, 6)
                                   for name, value in hit.channel_scores.items()},
                "quality_reason": hit.quality_reason,
            } for hit in hits],
        }

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
