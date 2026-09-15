"""随包提供的用户指南、发布说明与示例资料。

打包后资源位于 `_MEIPASS/resources/{docs,examples}`；开发态 `bundle_root()` 返回仓库根，
而仓库根的 `resources/` 是空的，内容仍在 `docs/` 与 `examples/` 下。两条路径都要找。
"""

from datetime import datetime, timezone
from pathlib import Path

from .paths import bundle_root


BUNDLED_DOCS = ("用户指南.md", "RELEASE_NOTES_0.2.0.md", "THIRD_PARTY_LICENSES.md")
BUNDLED_EXAMPLES = ("欢迎使用.md",)
BUNDLED = {"docs": BUNDLED_DOCS, "examples": BUNDLED_EXAMPLES}


def _resource_dirs(kind: str) -> list[Path]:
    root = bundle_root()
    return [root / "resources" / kind, root / kind]


def resolve_bundled(kind: str, name: str) -> Path | None:
    """按随包白名单解析资源；白名单同时挡掉路径穿越。"""
    if kind not in BUNDLED or name not in BUNDLED[kind]:
        return None
    for directory in _resource_dirs(kind):
        candidate = (directory / name).resolve()
        if candidate.is_file() and candidate.is_relative_to(directory.resolve()):
            return candidate
    return None


def _listing(kind: str) -> list[dict]:
    items = []
    for name in BUNDLED[kind]:
        path = resolve_bundled(kind, name)
        if path is None:
            continue
        items.append({
            "name": name,
            "size": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc).isoformat(),
        })
    return items


def bundled_docs() -> list[dict]:
    return _listing("docs")


def bundled_examples() -> list[dict]:
    return _listing("examples")