"""生成不含笔记路径、正文或密钥值的可分享供应链清单。"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import platform
import re
import subprocess

from .config import PROJECT_ROOT
from .index_store import IndexStore, file_hash


RECIPE_FILES = (
    ".dockerignore", "Dockerfile", "compose.yaml",
    "requirements-service.txt", "requirements-vector.txt",
    "container/rag.container.toml",
)
HOST_PACKAGES = (
    "torch", "numpy", "sentence-transformers", "transformers",
    "fastapi", "uvicorn", "pydantic", "starlette", "httpx",
)


def _run(command: list[str], *, timeout: int = 20) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=timeout,
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out after {timeout}s"
    except OSError as error:
        return 127, "", f"{type(error).__name__}: {error}"


def _canonical_hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def runtime_source_files(root: Path = PROJECT_ROOT) -> list[str]:
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    sources = sorted(set(re.findall(r"src/[A-Za-z0-9_./-]+\.py", dockerfile)))
    if not sources:
        raise ValueError("Dockerfile 中没有找到显式的 src/*.py COPY 清单")
    return sources


def hash_inventory(paths: list[str], root: Path = PROJECT_ROOT) -> dict:
    entries = []
    for relative in sorted(set(paths)):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"供应链输入不存在或不是普通文件：{relative}")
        entries.append({"path": relative.replace("\\", "/"), "sha256": file_hash(path),
                        "size_bytes": path.stat().st_size})
    return {"files": entries, "inventory_sha256": _canonical_hash(entries)}


def package_inventory(names: tuple[str, ...] = HOST_PACKAGES) -> dict:
    packages = []
    for name in sorted(names, key=str.lower):
        try:
            packages.append({"name": name, "version": package_version(name)})
        except PackageNotFoundError:
            packages.append({"name": name, "version": None})
    return {"packages": packages, "inventory_sha256": _canonical_hash(packages)}


def git_state() -> dict:
    head_code, head, _ = _run(["git", "rev-parse", "HEAD"])
    branch_code, branch, _ = _run(["git", "branch", "--show-current"])
    status_code, status, _ = _run(["git", "status", "--porcelain=v1"])
    return {
        "head": head if head_code == 0 else None,
        "branch": branch if branch_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "changed_entry_count": len(status.splitlines()) if status_code == 0 else None,
    }


def public_index_summary(store_root: Path) -> dict:
    selected = IndexStore(store_root).current()
    if selected is None:
        return {"available": False}
    path, manifest = selected
    artifacts = {
        name: {"sha256": manifest["artifacts"][name], "size_bytes": (path / name).stat().st_size}
        for name in sorted(manifest["artifacts"])
    }
    pipeline = manifest.get("pipeline", {})
    return {
        "available": True,
        "schema": manifest["schema"],
        "version": manifest["version"],
        "created_at": manifest.get("created_at"),
        "manifest_sha256": file_hash(path / "manifest.json"),
        "stats": manifest.get("stats", {}),
        "artifacts": artifacts,
        "artifact_inventory_sha256": _canonical_hash(artifacts),
        "pipeline": {
            key: pipeline.get(key) for key in (
                "schema", "max_tokens", "overlap_tokens", "model", "revision",
                "source_hashes", "packages",
            )
        },
        "privacy": "仅公开索引版本、统计量和制品哈希；省略笔记定位信息与正文。",
    }


def docker_image_summary(image: str, source_paths: list[str]) -> dict:
    inspect_code, inspect_out, inspect_error = _run(["docker", "image", "inspect", image])
    if inspect_code != 0:
        return {"available": False, "image": image, "reason": inspect_error or inspect_out}
    inspected = json.loads(inspect_out)[0]
    pip_code, pip_out, pip_error = _run([
        "docker", "run", "--rm", "--entrypoint", "python", image,
        "-m", "pip", "list", "--format", "json", "--disable-pip-version-check",
    ], timeout=60)
    script = (
        "import hashlib,json,pathlib; root=pathlib.Path('/app'); "
        "paths=json.loads(" + repr(json.dumps(source_paths)) + "); "
        "print(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}))"
    )
    source_code, source_out, source_error = _run([
        "docker", "run", "--rm", "--entrypoint", "python", image, "-c", script,
    ], timeout=60)
    packages = json.loads(pip_out) if pip_code == 0 else []
    embedded = json.loads(source_out) if source_code == 0 else {}
    host_hashes = {item["path"]: item["sha256"]
                   for item in hash_inventory(source_paths)["files"]}
    mismatches = sorted(path for path in source_paths if embedded.get(path) != host_hashes[path])
    config = inspected.get("Config") or {}
    env_keys = sorted(item.split("=", 1)[0] for item in config.get("Env") or [])
    return {
        "available": True,
        "image": image,
        "id": inspected.get("Id"),
        "repo_tags": inspected.get("RepoTags") or [],
        "repo_digests": inspected.get("RepoDigests") or [],
        "size_bytes": inspected.get("Size"),
        "created": inspected.get("Created"),
        "user": config.get("User"),
        "entrypoint": config.get("Entrypoint"),
        "cmd": config.get("Cmd"),
        "environment_keys": env_keys,
        "rootfs_diff_ids": (inspected.get("RootFS") or {}).get("Layers") or [],
        "packages": packages,
        "package_inventory_sha256": _canonical_hash(packages) if packages else None,
        "embedded_source_hashes": embedded,
        "source_matches_host": source_code == 0 and not mismatches,
        "source_mismatches": mismatches,
        "collection_errors": {
            "pip": None if pip_code == 0 else pip_error or pip_out,
            "source": None if source_code == 0 else source_error or source_out,
        },
        "limitations": (
            "本清单是制品盘点，不是镜像签名、构建证明或漏洞扫描；"
            "本地 tag 没有 registry digest 时不能证明远端身份。"
        ),
    }


def assert_public(manifest: dict, forbidden_values: list[str] | None = None):
    raw = json.dumps(manifest, ensure_ascii=False)
    forbidden = [str(PROJECT_ROOT.resolve()), "vault_dir", "source_file", "documents.json\": ["]
    forbidden.extend(value for value in (forbidden_values or []) if value)
    windows_user_path = re.search(r"[A-Za-z]:\\\\Users\\\\[^\"\\]+", raw, re.IGNORECASE)
    hits = [value for value in forbidden if value in raw]
    if windows_user_path or hits:
        raise ValueError(f"公开清单隐私检查失败；命中={hits or [windows_user_path.group(0)]}")


def build_manifest(store_root: Path, image: str) -> dict:
    runtime_sources = runtime_source_files()
    source = hash_inventory([
        *RECIPE_FILES, *runtime_sources, "src/index_backup.py", "src/recovery_drill.py",
        "src/release_manifest.py",
    ])
    manifest = {
        "schema": "rag-release-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": "obsidian-rag",
        "git": git_state(),
        "source": source,
        "host_environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.system(),
            **package_inventory(),
        },
        "index": public_index_summary(store_root),
        "container_image": docker_image_summary(image, runtime_sources),
        "scope": {
            "includes": ["source/recipe hashes", "package versions", "image identity", "index artifact hashes"],
            "excludes": ["secret values", "absolute private paths", "note filenames", "note content", "model weights"],
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_hash(manifest)
    assert_public(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--image", default="obsidian-rag:local")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.store, args.image)
    output = args.output or PROJECT_ROOT / "data/generated" / (
        "release_manifest_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "manifest": str(output),
        "payload_sha256": manifest["manifest_payload_sha256"],
        "source_files": len(manifest["source"]["files"]),
        "index": {key: manifest["index"].get(key) for key in ("available", "version", "stats")},
        "container_image": {key: manifest["container_image"].get(key)
                            for key in ("available", "id", "size_bytes", "source_matches_host")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
