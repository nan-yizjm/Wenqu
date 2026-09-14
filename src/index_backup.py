"""为版本化索引创建可校验备份，并只恢复到全新的目录。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from .index_store import ARTIFACTS, IndexStore, atomic_json, file_hash


BACKUP_SCHEMA = "index-backup-v1"
MANIFEST_MEMBER = "recovery_manifest.json"
INDEX_PREFIX = "index"
INDEX_MEMBERS = ("manifest.json", *ARTIFACTS)
EXPECTED_MEMBERS = {MANIFEST_MEMBER, *(f"{INDEX_PREFIX}/{name}" for name in INDEX_MEMBERS)}


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def _safe_member_names(archive: ZipFile) -> set[str]:
    names = set()
    for item in archive.infolist():
        name = item.filename
        path = PurePosixPath(name)
        if (item.is_dir() or path.is_absolute() or ".." in path.parts
                or "\\" in name or name in names):
            raise ValueError(f"备份包含非法或重复路径：{name!r}")
        names.add(name)
    if names != EXPECTED_MEMBERS:
        missing = sorted(EXPECTED_MEMBERS - names)
        unexpected = sorted(names - EXPECTED_MEMBERS)
        raise ValueError(f"备份成员不完整或超出白名单；缺少={missing}，额外={unexpected}")
    return names


def inspect_backup(backup: Path, *, include_payload: bool = False) -> dict:
    """先校验外层成员，再校验内层索引清单和所有字节哈希。"""
    backup = Path(backup).resolve()
    if not backup.is_file() or backup.is_symlink():
        raise FileNotFoundError("索引备份不存在或不是普通文件")
    try:
        with ZipFile(backup, "r") as archive:
            _safe_member_names(archive)
            payload = {name: archive.read(name) for name in EXPECTED_MEMBERS}
    except (BadZipFile, OSError, RuntimeError) as error:
        raise ValueError("索引备份不是可读取的完整 ZIP") from error

    try:
        recovery = json.loads(payload[MANIFEST_MEMBER].decode("utf-8"))
        index_manifest = json.loads(payload[f"{INDEX_PREFIX}/manifest.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("备份清单不是有效 UTF-8 JSON") from error

    version = recovery.get("index_version")
    if (recovery.get("schema") != BACKUP_SCHEMA
            or not isinstance(recovery.get("members"), dict)
            or set(recovery["members"]) != {f"{INDEX_PREFIX}/{name}" for name in INDEX_MEMBERS}):
        raise ValueError("恢复清单格式不正确")
    if index_manifest.get("schema") != "index-version-v1" or index_manifest.get("version") != version:
        raise ValueError("恢复清单与索引版本清单不一致")

    for name, expected in recovery["members"].items():
        raw = payload[name]
        if (not isinstance(expected, dict) or expected.get("size_bytes") != len(raw)
                or expected.get("sha256") != _sha256_bytes(raw)):
            raise ValueError(f"备份成员校验失败：{name}")
    manifest_raw = payload[f"{INDEX_PREFIX}/manifest.json"]
    if recovery.get("index_manifest_sha256") != _sha256_bytes(manifest_raw):
        raise ValueError("恢复清单中的索引 manifest 哈希不一致")
    if set(index_manifest.get("artifacts", {})) != set(ARTIFACTS):
        raise ValueError("索引 manifest 的制品集合不正确")
    for name in ARTIFACTS:
        if index_manifest["artifacts"][name] != _sha256_bytes(payload[f"{INDEX_PREFIX}/{name}"]):
            raise ValueError(f"索引内部制品校验失败：{name}")

    result = {
        "schema": recovery["schema"],
        "index_version": version,
        "created_at": recovery.get("created_at"),
        "backup_sha256": file_hash(backup),
        "backup_size_bytes": backup.stat().st_size,
        "member_count": len(payload),
        "index_stats": index_manifest.get("stats", {}),
    }
    if include_payload:
        result["payload"] = payload
        result["index_manifest"] = index_manifest
    return result


def create_backup(store_root: Path, output: Path, version: str | None = None) -> dict:
    store = IndexStore(Path(store_root))
    if version is None:
        selected = store.current()
        if selected is None:
            raise ValueError("索引没有 current 版本，无法创建备份")
        version_path, manifest = selected
    else:
        version_path, manifest = store.verify(version)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("备份文件已存在；为避免覆盖，请换一个新文件名")

    payload = {f"{INDEX_PREFIX}/manifest.json": (version_path / "manifest.json").read_bytes()}
    payload.update({f"{INDEX_PREFIX}/{name}": (version_path / name).read_bytes() for name in ARTIFACTS})
    recovery = {
        "schema": BACKUP_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_version": manifest["version"],
        "index_manifest_sha256": _sha256_bytes(payload[f"{INDEX_PREFIX}/manifest.json"]),
        "members": {
            name: {"sha256": _sha256_bytes(raw), "size_bytes": len(raw)}
            for name, raw in sorted(payload.items())
        },
        "privacy": (
            "该归档包含 documents.json/chunks.json，可能含私人笔记正文；"
            "不要提交 Git 或上传公共制品库。"
        ),
    }

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(MANIFEST_MEMBER, _json_bytes(recovery))
            for name, raw in sorted(payload.items()):
                archive.writestr(name, raw)
        # 写后立即从 ZIP 重新读取并验证，不能只相信写入路径。
        inspect_backup(temporary)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return inspect_backup(output)


def restore_backup(backup: Path, target_root: Path) -> dict:
    """恢复到不存在的新目录；不提供覆盖模式。"""
    target = Path(target_root).resolve()
    if target.exists():
        raise FileExistsError("恢复目标已存在；本工具拒绝覆盖或合并现有索引")
    target.parent.mkdir(parents=True, exist_ok=True)
    inspected = inspect_backup(backup, include_payload=True)
    payload = inspected.pop("payload")
    manifest = inspected.pop("index_manifest")
    version = inspected["index_version"]

    with tempfile.TemporaryDirectory(dir=target.parent, prefix="restoring-") as temporary:
        staging = Path(temporary) / "ready"
        version_path = staging / "versions" / version
        version_path.mkdir(parents=True)
        for name in INDEX_MEMBERS:
            (version_path / name).write_bytes(payload[f"{INDEX_PREFIX}/{name}"])
        atomic_json(staging / "current.json", {
            "version": version,
            "manifest_sha256": _sha256_bytes(payload[f"{INDEX_PREFIX}/manifest.json"]),
        })
        restored_path, restored_manifest = IndexStore(staging).current()
        if restored_manifest["version"] != version or restored_path != version_path.resolve():
            raise ValueError("恢复后的 current 指针没有指向预期版本")
        receipt = {
            "schema": "index-restore-receipt-v1",
            "restored_at": datetime.now(timezone.utc).isoformat(),
            "index_version": version,
            "backup_sha256": inspected["backup_sha256"],
            "verified_before_publish": True,
        }
        atomic_json(staging / "restore_receipt.json", receipt)
        os.replace(staging, target)

    # 原子发布完成后再次走公开读取路径验证。
    _, final_manifest = IndexStore(target).current()
    return {
        **inspected,
        "target": str(target),
        "restored_version": final_manifest["version"],
        "verified_after_publish": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create", help="备份 current 或指定版本")
    create.add_argument("--store", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--version")
    verify = subparsers.add_parser("verify", help="只校验，不解包")
    verify.add_argument("backup", type=Path)
    restore = subparsers.add_parser("restore", help="恢复到不存在的新目录")
    restore.add_argument("backup", type=Path)
    restore.add_argument("target", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        result = create_backup(args.store, args.output, args.version)
    elif args.action == "verify":
        result = inspect_backup(args.backup)
    else:
        result = restore_backup(args.backup, args.target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
