"""产品检索模型的延迟下载、CPU 加载与最小完整性验证。"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading

from ..vector_retrieve import MODEL_NAME, MODEL_REVISION


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetrievalModelManager:
    """网页可以先启动；只有用户明确准备模型时才导入 PyTorch。"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.state_file = cache_dir / "retrieval-model.json"
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = {
            "status": "ready" if self.state_file.is_file() else "not_downloaded",
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "detail": None,
            "updated_at": None,
        }
        if self.state_file.is_file():
            try:
                saved = json.loads(self.state_file.read_text(encoding="utf-8"))
                if saved.get("model") == MODEL_NAME and saved.get("revision") == MODEL_REVISION:
                    self._state.update(saved)
                else:
                    self._state["status"] = "not_downloaded"
            except (OSError, ValueError, TypeError):
                self._state["status"] = "not_downloaded"

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, allow_download: bool = True) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return dict(self._state)
            self._thread = threading.Thread(
                target=self.prepare, kwargs={"allow_download": allow_download},
                name="retrieval-model-setup", daemon=True,
            )
            self._thread.start()
            return dict(self._state)

    def _update(self, status: str, detail: str | None = None, **extra):
        with self._lock:
            self._state.update(status=status, detail=detail, updated_at=_now(), **extra)

    def prepare(self, allow_download: bool = True) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._update("downloading" if allow_download else "loading",
                         "正在获取固定版本的 multilingual-e5-small" if allow_download
                         else "正在检查本机模型缓存")
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                MODEL_NAME, revision=MODEL_REVISION, device="cpu",
                cache_folder=str(self.cache_dir), local_files_only=not allow_download,
            )
            self._update("verifying", "正在执行向量维度和数值检查")
            vector = model.encode(
                "query: 检索模型完整性检查", normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            dimension = int(model.get_embedding_dimension())
            norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            if len(vector) != dimension or dimension != 384 or not math.isfinite(norm):
                raise ValueError(f"模型输出异常：dimension={dimension}, norm={norm}")
            if not 0.999 <= norm <= 1.001:
                raise ValueError(f"归一化验证失败：norm={norm}")
            ready = {
                "status": "ready", "model": MODEL_NAME, "revision": MODEL_REVISION,
                "dimension": dimension, "device": "cpu", "detail": "模型验证通过",
                "updated_at": _now(),
            }
            temporary = self.state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.state_file)
            with self._lock:
                self._state = ready
        except Exception as error:
            self._update("failed", str(error)[:500])
        return self.status()


class MemoryRetrievalModelManager:
    """测试替身，不下载或加载真实模型。"""

    def __init__(self, status="not_downloaded"):
        self._state = {"status": status, "model": MODEL_NAME, "revision": MODEL_REVISION,
                       "detail": None, "updated_at": None}

    def status(self):
        return dict(self._state)

    def start(self, allow_download=True):
        self._state.update(status="ready", dimension=384, device="cpu",
                           detail="测试模型验证通过", updated_at=_now())
        return self.status()


def build_cpu_encoder(cache_dir: Path):
    """用本机缓存里的固定快照装一个 CPU 编码器，供向量索引使用。

    只读本地文件：产品承诺不因为检索而联网。
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME, revision=MODEL_REVISION, device="cpu",
        cache_folder=str(cache_dir), local_files_only=True,
    )
