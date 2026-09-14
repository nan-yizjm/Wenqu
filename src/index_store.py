"""读取笔记 → 增量分片/编码 → 完整版本 → 原子切换 current；不修改原笔记。"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from .ingest import extract_title
from .settings import DEFAULT_CONFIG, Settings, load_settings
from .token_chunking import TokenChunker
from .vector_retrieve import MODEL_NAME, MODEL_REVISION, PASSAGE_TEMPLATE, VectorIndex

ARTIFACTS = ("documents.json", "chunks.json", "vector_index.npz")
VERSION_PATTERN = re.compile(r"\d{8}T\d{12}Z-[a-f0-9]{8}")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def atomic_json(path: Path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def writer_lock(store: Path):
    """OS 锁随进程退出释放；保留一字节锁文件不表示仍被占用。"""
    store.mkdir(parents=True, exist_ok=True)
    with (store / ".writer.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("另一个进程正在更新索引，请稍后重试") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def scan_documents(root: Path) -> tuple[list[dict], dict[str, str]]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError("笔记目录不存在；不会把它当作全部笔记被删除")
    paths = []
    def scan_error(error):
        raise error  # 无权限目录不可静默跳过，否则会被误判为删除。
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=scan_error):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if not name.startswith(".")
                         and not (parent / name).is_symlink() and not (parent / name).is_junction())
        paths.extend(parent / name for name in files if name.lower().endswith(".md"))
    documents, hashes = [], {}
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("笔记中存在符号链接或越界路径，请先移出扫描范围")
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        name = path.relative_to(root).as_posix()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        documents.append({"source_file": name, "title": extract_title(text) or path.stem, "text": text})
    return documents, hashes


def pipeline_signature(settings: Settings) -> dict:
    source = Path(__file__).parent
    return {
        "schema": "incremental-token-v1", "max_tokens": settings.max_tokens,
        "overlap_tokens": settings.overlap_tokens, "model": MODEL_NAME,
        "revision": MODEL_REVISION, "passage_template": PASSAGE_TEMPLATE,
        "source_hashes": {name: file_hash(source / name) for name in (
            "index_store.py", "token_chunking.py", "markdown_structure.py", "vector_retrieve.py", "ingest.py")},
        "packages": {name: package_version(name) for name in (
            "torch", "sentence-transformers", "transformers", "numpy")},
    }


class IndexStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def version_path(self, version: str) -> Path:
        if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
            raise ValueError("非法索引版本号")
        path = (self.root / "versions" / version).resolve()
        if not path.is_relative_to(self.root / "versions"):
            raise ValueError("索引版本路径越界")
        return path

    def verify(self, version: str) -> tuple[Path, dict]:
        path = self.version_path(version)
        manifest = read_json(path / "manifest.json")
        if (manifest.get("schema") != "index-version-v1" or manifest.get("version") != version
                or set(manifest.get("artifacts", {})) != set(ARTIFACTS)):
            raise ValueError("索引清单格式不正确")
        for name in ARTIFACTS:
            artifact = path / name
            if artifact.is_symlink() or not artifact.resolve().is_relative_to(path):
                raise ValueError("索引文件路径越界")
            if file_hash(artifact) != manifest["artifacts"][name]:
                raise ValueError(f"索引文件校验失败：{name}；拒绝激活损坏版本")
        return path, manifest

    def current(self) -> tuple[Path, dict] | None:
        pointer_path = self.root / "current.json"
        if not pointer_path.exists():
            return None
        pointer = read_json(pointer_path)
        path, manifest = self.verify(pointer["version"])
        if file_hash(path / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("current 指针与版本清单不一致")
        return path, manifest

    def _activate(self, version: str):
        path, manifest = self.verify(version)
        atomic_json(self.root / "current.json", {
            "version": version, "manifest_sha256": file_hash(path / "manifest.json"),
        })
        return manifest

    def activate(self, version: str):
        with writer_lock(self.root):
            return self._activate(version)

    def list_versions(self):
        versions = []
        for path in sorted((self.root / "versions").glob("*"), reverse=True):
            if VERSION_PATTERN.fullmatch(path.name):
                try:
                    _, manifest = self.verify(path.name)
                    versions.append({"version": path.name, "valid": True, "stats": manifest["stats"]})
                except (OSError, ValueError, KeyError):
                    versions.append({"version": path.name, "valid": False})
        return versions

    def build(self, settings: Settings, *, allow_empty=False, tokenizer=None, encoder=None):
        if self.root != settings.store_dir.resolve():
            raise ValueError("配置与索引目录不一致")
        started = time.perf_counter()
        with writer_lock(self.root):
            current = self.current()
            old_path, old = current if current else (None, {})
            documents, hashes = scan_documents(settings.vault_dir)
            if not documents and not allow_empty:
                raise ValueError("没有 Markdown 文档；如确实要发布空索引，显式使用 --allow-empty")
            signature = pipeline_signature(settings)
            old_hashes = old.get("files", {})
            added = sorted(hashes.keys() - old_hashes.keys())
            deleted = sorted(old_hashes.keys() - hashes.keys())
            changed = sorted(k for k in hashes.keys() & old_hashes.keys() if hashes[k] != old_hashes[k])
            compatible = (old.get("pipeline") == signature
                          and old.get("vault_dir") == str(settings.vault_dir.resolve()))
            if compatible and not added and not deleted and not changed:
                return {"status": "unchanged", "version": old["version"],
                        "documents": len(documents), "encoded_chunks": 0,
                        "elapsed_seconds": round(time.perf_counter() - started, 3)}
            old_chunks = read_json(old_path / "chunks.json") if old_path else []
            by_source = defaultdict(list)
            for chunk in old_chunks:
                by_source[chunk["source_file"]].append(chunk)
            chunker = None
            chunks, parsed_documents = [], 0
            for document in documents:
                name = document["source_file"]
                if compatible and hashes[name] == old_hashes.get(name):
                    chunks.extend(by_source[name])
                    continue
                if chunker is None:
                    if tokenizer is None:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(
                            MODEL_NAME, revision=MODEL_REVISION, local_files_only=True)
                    chunker = TokenChunker(tokenizer, settings.max_tokens, settings.overlap_tokens)
                pieces, _ = chunker.create_chunks([document])
                chunks.extend(pieces)
                parsed_documents += 1
            versions_dir = self.root / "versions"
            versions_dir.mkdir(exist_ok=True)
            version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex[:8]
            with tempfile.TemporaryDirectory(dir=self.root, prefix="building-") as temporary:
                staging = Path(temporary) / "ready"
                staging.mkdir()
                write_json(staging / "documents.json", documents)
                write_json(staging / "chunks.json", chunks)
                vector = VectorIndex(
                    chunks, device=settings.device, cache_path=staging / "vector_index.npz", encoder=encoder,
                    reuse_from=(old_chunks, old_path / "vector_index.npz") if old_path else None)
                if vector.truncated_chunk_count:
                    raise ValueError("构建后的片段超过模型 token 上限，拒绝发布")
                # 编码期间用户仍可编辑；源文件变了就保留旧 current，下次再构建。
                if scan_documents(settings.vault_dir)[1] != hashes or pipeline_signature(settings) != signature:
                    raise RuntimeError("构建过程中笔记或构建代码变化；本次未发布，请重试")
                stats = {
                    "documents": len(documents), "chunks": len(chunks),
                    "added_documents": len(added), "modified_documents": len(changed),
                    "deleted_documents": len(deleted), "parsed_documents": parsed_documents,
                    "reused_chunks": vector.reused_count, "encoded_chunks": vector.encoded_count,
                    "reuse_status": vector.reuse_status,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                manifest = {
                    "schema": "index-version-v1", "version": version,
                    "parent_version": old.get("version"), "created_at": datetime.now(timezone.utc).isoformat(),
                    "vault_dir": str(settings.vault_dir.resolve()), "pipeline": signature, "files": hashes,
                    "changes": {"added": added, "modified": changed, "deleted": deleted},
                    "stats": stats, "artifacts": {name: file_hash(staging / name) for name in ARTIFACTS},
                }
                write_json(staging / "manifest.json", manifest)
                os.rename(staging, self.version_path(version))
            self._activate(version)
            return {"status": "built", "version": version, **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "status", "list", "activate"))
    parser.add_argument("version", nargs="?")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    store = IndexStore(settings.store_dir)
    if args.action == "build":
        result = store.build(settings, allow_empty=args.allow_empty)
    elif args.action == "activate":
        if not args.version:
            parser.error("activate 需要版本号；可用 list 查看")
        result = {"activated": store.activate(args.version)["version"], "note": "运行中的服务仍固定旧版本，重启后生效"}
    elif args.action == "list":
        result = store.list_versions()
    else:
        current = store.current()
        result = {"current": current[1]["version"], "stats": current[1]["stats"]} if current else {"current": None}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
