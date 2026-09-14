"""产品数据路径；安装目录与用户数据严格分离。"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys


def bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class ProductPaths:
    root: Path

    @classmethod
    def default(cls) -> "ProductPaths":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return cls((base / "ObsidianRAG").resolve())

    @property
    def database(self): return self.root / "workspace.sqlite3"
    @property
    def logs(self): return self.root / "logs"
    @property
    def snapshots(self): return self.root / "snapshots"
    @property
    def corpus(self): return self.root / "corpus"
    @property
    def indexes(self): return self.root / "indexes"
    @property
    def model_cache(self): return self.root / "model-cache"
    @property
    def exports(self): return self.root / "exports"
    @property
    def backups(self): return self.root / "backups"
    @property
    def runtime(self): return self.root / "runtime"

    def ensure(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (self.logs, self.snapshots, self.corpus, self.indexes,
                     self.model_cache, self.exports, self.backups, self.runtime):
            path.mkdir(exist_ok=True)
        return self
