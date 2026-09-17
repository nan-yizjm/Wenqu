"""非秘密配置；路径相对配置文件解析，错误在加载模型前报告。"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import tomllib

from .config import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "rag.toml"


@dataclass(frozen=True)
class Settings:
    vault_dir: Path
    store_dir: Path
    device: str = "cuda"
    max_tokens: int = 480
    overlap_tokens: int = 32
    provider: str = "ollama"
    retriever: str = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    rerank: bool = False
    rerank_top_n: int = 20
    rerank_device: str = "cpu"
    context_policy: str = "blocks"
    max_context_chars: int = 2200
    host: str = "127.0.0.1"
    port: int = 8000
    request_timeout: int = 90
    max_question_chars: int = 2000
    max_queue_size: int = 0
    queue_wait_timeout: int = 15

    def __post_init__(self):
        for name, choices in {
            "device": ("cuda", "cpu"), "rerank_device": ("cuda", "cpu"),
            "provider": ("ollama", "deepseek"), "retriever": ("bm25", "vector", "hybrid"),
            "context_policy": ("legacy", "blocks"), "host": ("127.0.0.1",),
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} 只能为 {choices}")
        if type(self.rerank) is not bool:
            raise ValueError("rerank 必须为 true / false")
        for name in ("max_tokens", "overlap_tokens", "top_k", "candidate_k", "rrf_k", "rerank_top_n",
                     "max_context_chars", "port", "request_timeout", "max_question_chars",
                     "max_queue_size", "queue_wait_timeout"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} 必须为整数")
        if not 32 <= self.max_tokens <= 512 or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("要求 32 <= max_tokens <= 512，0 <= overlap_tokens < max_tokens")
        if not 1 <= self.top_k <= self.candidate_k <= 200:
            raise ValueError("要求 1 <= top_k <= candidate_k <= 200")
        if not 1 <= self.rrf_k <= 1000:
            raise ValueError("要求 1 <= rrf_k <= 1000")
        if not 1 <= self.rerank_top_n <= self.candidate_k:
            raise ValueError("rerank_top_n 必须在 1～candidate_k 之间")
        if self.rerank and self.rerank_top_n < self.top_k:
            raise ValueError("重排候选数不能少于 top_k")
        if not (128 <= self.max_context_chars <= 32000 and 1 <= self.port <= 65535
                and 1 <= self.request_timeout <= 300 and 1 <= self.max_question_chars <= 20000):
            raise ValueError("上下文预算、端口、超时或问题长度超出允许范围")
        if not 0 <= self.max_queue_size <= 100 or not 1 <= self.queue_wait_timeout <= 300:
            raise ValueError("要求 0 <= max_queue_size <= 100，1 <= queue_wait_timeout <= 300")
        if self.store_dir.resolve().is_relative_to(self.vault_dir.resolve()):
            raise ValueError("索引目录不能放在笔记扫描目录内，避免把派生数据再次导入")

    def public_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_settings(path: Path = DEFAULT_CONFIG) -> Settings:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    unknown = set(values) - {field.name for field in fields(Settings)}
    if unknown:
        raise ValueError(f"未知配置项：{', '.join(sorted(unknown))}；密钥应放在 .env")
    for key in ("vault_dir", "store_dir"):
        value = values.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"缺少路径配置：{key}")
        values[key] = (path.parent / value).resolve()
    return Settings(**values)
