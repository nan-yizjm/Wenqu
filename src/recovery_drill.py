"""用真实索引演练：有效备份、篡改拒绝、恢复到新目录、恢复后校验。"""

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .config import PROJECT_ROOT
from .index_backup import create_backup, restore_backup
from .index_store import IndexStore, file_hash


def make_tampered_copy(source: Path, output: Path):
    """重写一个 CRC 正常但内容哈希不匹配的 ZIP，验证的是业务哈希而非 ZIP CRC。"""
    with ZipFile(source, "r") as original, ZipFile(output, "x", ZIP_DEFLATED) as changed:
        for item in original.infolist():
            raw = original.read(item.filename)
            if item.filename == "index/chunks.json":
                raw += b"\n"
            changed.writestr(item.filename, raw)


def run_drill(store_root: Path, output_root: Path, *, runtime_config: Path | None = None,
              question: str = "PagedAttention 是什么？它解决什么问题？") -> dict:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    source_path, source_manifest = IndexStore(store_root).current() or (None, None)
    if source_path is None:
        raise ValueError("源索引没有 current 版本")

    backup = output_root / "index-backup.zip"
    corrupt = output_root / "index-backup-tampered.zip"
    invalid_target = output_root / "invalid-restore-must-not-exist"
    restored = output_root / "restored-index"
    backup_result = create_backup(store_root, backup)
    make_tampered_copy(backup, corrupt)
    corrupt_error = None
    try:
        restore_backup(corrupt, invalid_target)
    except (ValueError, OSError) as error:
        corrupt_error = f"{type(error).__name__}: {error}"
    restore_result = restore_backup(backup, restored)
    restored_path, restored_manifest = IndexStore(restored).current()

    source_hashes = {name: file_hash(source_path / name)
                     for name in ("manifest.json", *source_manifest["artifacts"])}
    restored_hashes = {name: file_hash(restored_path / name) for name in source_hashes}
    checks = {
        "backup_created_and_verified": backup_result["index_version"] == source_manifest["version"],
        "tampered_valid_zip_rejected": bool(corrupt_error),
        "failed_restore_published_nothing": not invalid_target.exists(),
        "restore_used_new_directory": restored.is_dir(),
        "current_pointer_is_readable": restored_manifest["version"] == source_manifest["version"],
        "all_index_bytes_match_source": restored_hashes == source_hashes,
        "restore_receipt_exists": (restored / "restore_receipt.json").is_file(),
    }
    runtime_result = None
    if runtime_config is not None:
        # 延迟导入，纯备份/恢复操作不应强制加载向量与服务依赖。
        from .runtime import RAGRuntime
        from .settings import load_settings
        settings = replace(load_settings(runtime_config), store_dir=restored)
        runtime = RAGRuntime(settings)
        runtime_result = runtime.answer(question)
        validation = runtime_result.get("citation_validation") or {}
        checks.update({
            "restored_runtime_uses_expected_version": runtime_result["index_version"] == source_manifest["version"],
            "restored_runtime_generated_once": runtime_result["generation_calls"] == 1,
            "restored_runtime_citations_valid": bool(validation.get("is_valid")),
        })
    report = {
        "schema": "index-recovery-drill-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source": {
            "version": source_manifest["version"],
            "stats": source_manifest.get("stats", {}),
            "artifact_hashes": source_hashes,
        },
        "backup": backup_result,
        "tamper_scenario": {
            "method": "append one newline to chunks.json and rebuild a CRC-valid ZIP",
            "error": corrupt_error,
            "invalid_target_exists": invalid_target.exists(),
        },
        "restore": restore_result,
        "restored_artifact_hashes": restored_hashes,
        "runtime_validation": runtime_result,
        "checks": checks,
        "passed": all(checks.values()),
        "privacy": "实验目录中的 ZIP 和恢复目录包含笔记正文，必须保持 Git 忽略。",
    }
    report_path = output_root / "recovery_drill_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"passed": report["passed"], "checks": checks, "tamper_error": corrupt_error,
            "version": source_manifest["version"], "report": str(report_path),
            "restored_store": str(restored), "backup": str(backup),
            "runtime": ({
                "provider": runtime_result["provider"],
                "retriever": runtime_result["retriever"],
                "answer": runtime_result["answer"],
                "elapsed_ms": runtime_result["elapsed_ms"],
                "sources": len(runtime_result["sources"]),
            } if runtime_result else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runtime-config", type=Path,
                        help="可选：用恢复目录覆盖配置中的 store_dir，执行一次真实问答")
    parser.add_argument("--question", default="PagedAttention 是什么？它解决什么问题？")
    args = parser.parse_args()
    output = args.output_root or PROJECT_ROOT / "data/generated" / (
        "recovery_drill_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    result = run_drill(args.store, output, runtime_config=args.runtime_config,
                       question=args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
