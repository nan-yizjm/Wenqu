"""本地 CrossEncoder：对问题与候选联合打分，不能离线缓存单篇文档得分。"""

import argparse
import math
from importlib.metadata import version

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


class LocalReranker:
    def __init__(self, device="cuda", batch_size=8, max_length=512,
                 allow_download=False, encoder=None):
        if batch_size < 1 or not 64 <= max_length <= 512:
            raise ValueError("batch_size 必须为正；max_length 必须在 64～512 之间")
        if encoder is None:
            from sentence_transformers import CrossEncoder
            from torch import nn
            self.model = CrossEncoder(
                MODEL_NAME, revision=MODEL_REVISION, device=device,
                local_files_only=not allow_download, trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
                max_length=max_length, activation_fn=nn.Identity(),
            )
        else:
            self.model = encoder
        self.batch_size, self.max_length = batch_size, max_length
        self.metadata = {
            "model": MODEL_NAME, "revision": MODEL_REVISION, "device": device,
            "max_length": max_length, "batch_size": batch_size,
            "score_type": "raw_logit_not_probability", "activation": "Identity",
            "sentence_transformers": version("sentence-transformers"),
            "passage_template": "标题路径：{heading_path}\n内容：{text}",
        }

    def score(self, query: str, chunks: list[dict]) -> list[tuple[float, int]]:
        if not chunks:
            return []
        tokenizer = self.model.tokenizer
        query_tokens = len(tokenizer(query, truncation=False, verbose=False)["input_ids"])
        if query_tokens > self.max_length - 32:
            raise ValueError("重排序问题过长，没有足够空间联合读取候选；请缩短问题")
        passages = [self.metadata["passage_template"].format(**chunk) for chunk in chunks]
        counts = [len(ids) for ids in tokenizer([query] * len(passages), passages,
                  truncation=False, padding=False, verbose=False)["input_ids"]]
        values = self.model.predict(
            [(query, passage) for passage in passages], batch_size=self.batch_size,
            convert_to_numpy=True, show_progress_bar=False,
        )
        scores = [float(value) for value in values]
        if len(scores) != len(chunks) or not all(math.isfinite(value) for value in scores):
            raise ValueError("重排序返回数量或数值异常；没有静默降级成另一种排序")
        return list(zip(scores, counts, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="准备或验证本地重排序模型，不发送笔记到远程 API。")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    reranker = LocalReranker(device=args.device, allow_download=args.allow_download)
    results = reranker.score("RAG 如何利用知识库？", [
        {"heading_path": "RAG", "text": "先检索知识库，再将资料放入提示词，让模型据此回答。"},
        {"heading_path": "水果", "text": "苹果与香蕉是水果。"},
    ])
    print(reranker.metadata)
    print("相关与无关示例的分数及输入 token 数：", results)


if __name__ == "__main__":
    main()
