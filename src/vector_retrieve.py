"""E5 向量检索：持久化文档向量，查询仍需本地模型编码。"""

import argparse
import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .retrieve import load_chunks

MODEL_NAME = "intfloat/multilingual-e5-small"
# 固定本机已有的模型快照，防止同名模型更新后继续复用旧向量。
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data/generated/vector_index.npz"
PASSAGE_TEMPLATE = "passage: 标题路径：{heading_path}\n内容：{text}"
QUERY_PREFIX = "query: "


def chunks_fingerprint(chunks: list[dict[str, str]]) -> str:
    """包含顺序：矩阵第 i 行必须始终对应 chunks[i]。"""
    raw = json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class VectorIndex:
    def __init__(
        self, chunks: list[dict[str, str]],
        model_name: str = MODEL_NAME, device: str = "cuda",
        cache_path: Path | None = DEFAULT_CACHE, rebuild: bool = False,
        revision: str = MODEL_REVISION, batch_size: int = 16,
        max_length: int = 512, local_files_only: bool = True, encoder=None,
        reuse_from: tuple[list[dict], Path] | None = None, read_only: bool = False,
    ) -> None:
        if batch_size < 1 or not 1 <= max_length <= 512:
            raise ValueError("batch_size 必须为正，E5 max_length 必须在 1～512 之间")
        ids = [chunk["id"] for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk id 重复，无法可靠对应向量行与来源")
        self.chunks = chunks
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.batch_size = batch_size
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                model_name, revision=revision, device=device,
                local_files_only=local_files_only,
            )
        else:
            self.model = encoder  # 测试注入内存编码器，不下载模型。
        self.model.max_seq_length = max_length
        self.dimension = self.model.get_embedding_dimension()
        self.metadata = {
            "schema": 1, "model_name": model_name, "revision": revision,
            "fingerprint": chunks_fingerprint(chunks),
            "passage_template": PASSAGE_TEMPLATE, "query_prefix": QUERY_PREFIX,
            "max_length": max_length, "normalize_embeddings": True,
            "dimension": self.dimension, "dtype": "float32",
            "packages": {name: version(name) for name in (
                "torch", "sentence-transformers", "transformers", "numpy"
            )},
        }
        self.cache_status = "disabled" if self.cache_path is None else "missing"
        self.truncated_chunk_count = 0
        self.reused_count = 0
        self.encoded_count = 0
        self.reuse_status = "not_requested"
        cached = None if rebuild else self._load_cached_embeddings()
        if cached is not None:
            self.embeddings = cached
        else:
            if read_only:
                raise ValueError(f"版本索引只读，缓存不可用（{self.cache_status}）；请重新 build")
            if rebuild:
                self.cache_status = "forced_rebuild"
            self.embeddings = self._build_embeddings(reuse_from)

    def _valid_embeddings(self, matrix: np.ndarray) -> bool:
        if matrix.shape != (len(self.chunks), self.dimension):
            return False
        if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            return False
        return bool(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3))

    def _load_cached_embeddings(self) -> np.ndarray | None:
        if self.cache_path is None or not self.cache_path.is_file():
            return None
        try:
            with np.load(self.cache_path, allow_pickle=False) as cached:
                metadata = json.loads(str(cached["metadata"].item()))
                matrix = cached["embeddings"]
                truncated_count = int(cached["truncated_chunk_count"].item())
            if metadata != self.metadata:
                self.cache_status = "stale"
                return None
            if not self._valid_embeddings(matrix) or not 0 <= truncated_count <= len(self.chunks):
                raise ValueError("缓存的维度、数值或归一化状态异常")
        except (OSError, ValueError, TypeError, OverflowError, KeyError, EOFError, BadZipFile):
            self.cache_status = "invalid"
            return None
        self.truncated_chunk_count = truncated_count
        self.cache_status = "hit"
        return matrix

    def _reuse_rows(self, reuse_from) -> dict[str, np.ndarray]:
        if reuse_from is None:
            return {}
        old_chunks, old_path = reuse_from
        try:
            with np.load(old_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                matrix = archive["embeddings"]
            expected = {**self.metadata, "fingerprint": chunks_fingerprint(old_chunks)}
            if metadata != expected:
                self.reuse_status = "incompatible"
                return {}
            if (matrix.shape != (len(old_chunks), self.dimension) or matrix.dtype != np.float32
                    or not np.isfinite(matrix).all()
                    or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3)):
                raise ValueError("旧矩阵无效")
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
            self.reuse_status = "invalid"
            return {}
        self.reuse_status = "compatible"
        # 输入字符串完全相同才复用；来源行号变化不应触发重新编码。
        return {PASSAGE_TEMPLATE.format(**chunk): row for chunk, row in zip(old_chunks, matrix, strict=True)}

    def _build_embeddings(self, reuse_from=None) -> np.ndarray:
        passages = [PASSAGE_TEMPLATE.format(**chunk) for chunk in self.chunks]
        reusable = self._reuse_rows(reuse_from)
        missing = [i for i, passage in enumerate(passages) if passage not in reusable]
        matrix = np.empty((len(passages), self.dimension), dtype=np.float32)
        for i, passage in enumerate(passages):
            if passage in reusable:
                matrix[i] = reusable[passage]
        self.reused_count = len(passages) - len(missing)
        self.encoded_count = len(missing)
        if passages:
            for start in range(0, len(passages), self.batch_size):
                encoded = self.model.tokenizer(
                    passages[start:start + self.batch_size],
                    truncation=False, padding=False, verbose=False,
                )
                self.truncated_chunk_count += sum(
                    len(ids) > self.model.max_seq_length for ids in encoded["input_ids"]
                )
        if missing:
            matrix[missing] = np.asarray(self.model.encode(
                [passages[i] for i in missing], batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=True,
            ), dtype=np.float32)
        if not self._valid_embeddings(matrix):
            raise ValueError("编码器没有返回预期的有限、单位长度向量")
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # 写完临时文件才替换，进程中断不会破坏原缓存。
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.cache_path.parent, suffix=".npz", delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    np.savez_compressed(
                        handle, embeddings=matrix,
                        metadata=json.dumps(self.metadata, ensure_ascii=False, sort_keys=True),
                        truncated_chunk_count=self.truncated_chunk_count,
                    )
                os.replace(temporary_path, self.cache_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
        return matrix

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict[str, str], float]]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        if not query.strip() or not self.chunks:
            return []
        vector = np.asarray(self.model.encode(
            QUERY_PREFIX + query.strip(), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ), dtype=np.float32)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError("查询向量维度或数值异常")
        scores = self.embeddings @ vector
        positions = sorted(
            range(len(self.chunks)), key=lambda i: (-float(scores[i]), self.chunks[i]["id"])
        )
        return [(self.chunks[i], float(scores[i])) for i in positions[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="E5 原始余弦检索（未应用质量规则）。")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-download", action="store_true", help="首次在新机器上获取固定模型快照")
    parser.add_argument("--chunks-file", default="data/generated/chunks.json")
    args = parser.parse_args()
    chunks_path = Path(__file__).resolve().parent.parent / args.chunks_file
    chunks = load_chunks(chunks_path)
    index = VectorIndex(
        chunks, device=args.device, rebuild=args.rebuild,
        local_files_only=not args.allow_download,
        cache_path=chunks_path.parent / "vector_index.npz",
    )
    print(f"缓存状态：{index.cache_status}；向量矩阵：{index.embeddings.shape}")
    print(f"超过 {index.model.max_seq_length} 模型 token 的片段：{index.truncated_chunk_count}（本版截断编码）")
    for rank, (chunk, score) in enumerate(index.search(args.query, args.top_k), 1):
        print(f"\n[{rank}] 余弦相似度={score:.4f}\n{chunk['source_file']} → {chunk['heading_path']}")
        print(chunk["text"][:180].replace("\n", " "))


if __name__ == "__main__":
    main()
