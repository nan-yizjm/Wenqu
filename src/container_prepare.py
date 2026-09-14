"""显式准备或离线核对容器所需的固定 E5 模型缓存。"""

import argparse
import json
import os

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .vector_retrieve import MODEL_NAME, MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-check", action="store_true",
        help="只检查本地缓存，缺失时失败，不访问网络",
    )
    args = parser.parse_args()
    local_only = args.offline_check
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, local_files_only=local_only,
    )
    model = SentenceTransformer(
        MODEL_NAME, revision=MODEL_REVISION, device="cpu",
        local_files_only=local_only,
    )
    print(json.dumps({
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "mode": "offline_check" if local_only else "prepare_cache",
        "hf_home": os.getenv("HF_HOME"),
        "tokenizer_class": type(tokenizer).__name__,
        "embedding_dimension": model.get_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "ready": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
