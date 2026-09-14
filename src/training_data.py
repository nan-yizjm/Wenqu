"""独立教学语料：先分组划分，再建训练词表和窗口；不读取 Obsidian 笔记。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256

PAD, BOS, EOS, UNK = 0, 1, 2, 3
IGNORE = -100
SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
DEFAULT_DATA = PROJECT_ROOT / "data/generated/tiny_lm_data_v1"


class CharTokenizer:
    def __init__(self, tokens):
        if tokens[:4] != SPECIAL or len(tokens) != len(set(tokens)):
            raise ValueError("字符词表的特殊符号或唯一性错误")
        self.tokens = list(tokens)
        self.to_id = {token: i for i, token in enumerate(tokens)}

    @classmethod
    def fit(cls, texts):
        return cls(SPECIAL + sorted(set("".join(texts))))

    def encode(self, text, *, document=False):
        ids = [self.to_id.get(char, UNK) for char in text]
        return [BOS, *ids, EOS] if document else ids

    def decode(self, ids):
        return "".join(self.tokens[index] for index in ids if index not in (PAD, BOS, EOS))


def synthetic_documents():
    """本项目编写的有限模板语料；同一情景的两种表述属于同一 group。"""
    documents = []
    for person in ("小林", "小周", "小陈", "小许"):
        for place in ("图书馆", "实验室", "教室", "工作室"):
            for item, task in (("读书笔记", "整理书页"), ("实验日志", "核对数据"),
                               ("训练曲线", "检查误差"), ("项目记录", "标记编号")):
                group = f"{person}-{place}-{item}"
                texts = [
                    f"在{place}里，{person}负责整理{item}。开始工作以前，{person}先检查材料是否齐全。"
                    f"确认没有遗漏以后，{person}开始{task}，把发现的问题记在纸上。工作结束时，"
                    f"{person}保存了整理结果，并说明下一次需要继续检查的内容。认真记录可以减少重复劳动。",
                    f"{person}来到{place}，准备处理今天的{item}。第一步是检查材料，第二步是{task}。"
                    f"遇到不清楚的地方，{person}会先记录问题，再寻找原因。完成任务以后，"
                    f"{person}把结果保存下来。这样下一次工作时，就能知道之前做了什么，还有什么没有完成。",
                ]
                for variant, text in enumerate(texts):
                    documents.append({"id": f"{group}-{variant}", "group_id": group, "text": text})
    return documents


def prepare_data(output=DEFAULT_DATA, documents=None, validation_fraction=0.2, seed=42):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("数据版本已存在；请换新 --output，保留原版本")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 (0,1)")
    supplied = documents is not None
    documents = synthetic_documents() if documents is None else documents
    ids, hashes, normalized = set(), set(), []
    for doc in documents:
        if any(not isinstance(doc.get(key), str) or not doc[key].strip() for key in ("id", "group_id", "text")):
            raise ValueError("每篇文档需要非空 id、group_id、text")
        text = doc["text"].replace("\r\n", "\n").strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if doc["id"] in ids or digest in hashes:
            raise ValueError("发现重复 ID 或完全相同文本；请先去重，不让重复内容跨分组泄漏")
        ids.add(doc["id"])
        hashes.add(digest)
        normalized.append({"id": doc["id"], "group_id": doc["group_id"], "text": text, "sha256": digest})
    groups = sorted({doc["group_id"] for doc in normalized},
                    key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest())
    if len(groups) < 2:
        raise ValueError("至少需要两个组才能划分训练/验证")
    validation_count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:validation_count])
    train = [doc for doc in normalized if doc["group_id"] not in validation_groups]
    validation = [doc for doc in normalized if doc["group_id"] in validation_groups]
    tokenizer = CharTokenizer.fit(doc["text"] for doc in train)
    for split in (train, validation):
        for doc in split:
            doc["ids"] = tokenizer.encode(doc["text"], document=True)
    diagnostics = {
        "train_documents": len(train), "validation_documents": len(validation),
        "train_groups": len(groups) - validation_count, "validation_groups": validation_count,
        "vocab_size": len(tokenizer.tokens), "train_tokens": sum(len(doc["ids"]) for doc in train),
        "validation_tokens": sum(len(doc["ids"]) for doc in validation),
        "validation_unk_tokens": sum(doc["ids"].count(UNK) for doc in validation),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="preparing-lm-") as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        save_report(staging / "dataset.json", {"train": train, "validation": validation})
        save_report(staging / "vocab.json", tokenizer.tokens)
        save_report(staging / "manifest.json", {
            "schema": "tiny-char-data-v1", "seed": seed, "validation_fraction": validation_fraction,
            "source": "user_supplied_json" if supplied else "project_authored_synthetic_templates_v1",
            "split_policy": "group_hash_before_vocab_and_windows", "diagnostics": diagnostics,
            "artifacts": {name: sha256(staging / name) for name in ("dataset.json", "vocab.json")},
            "limitations": ["合成数据模板高度重复，验证损失下降不证明通用语言能力", "没有独立测试集"]})
        os.rename(staging, output)
    return diagnostics


def load_data(path: Path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "tiny-char-data-v1" or set(manifest.get("artifacts", {})) != {"dataset.json", "vocab.json"}:
        raise ValueError("数据清单格式不正确")
    for name, expected in manifest["artifacts"].items():
        if sha256(path / name) != expected:
            raise ValueError("数据文件已变化；拒绝继续使用旧实验/检查点")
    dataset = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
    tokenizer = CharTokenizer(json.loads((path / "vocab.json").read_text(encoding="utf-8")))
    train, val = dataset["train"], dataset["validation"]
    for field in ("id", "group_id", "sha256"):
        if {doc[field] for doc in train} & {doc[field] for doc in val}:
            raise ValueError(f"训练/验证发生 {field} 泄漏")
    return dataset, tokenizer, manifest, sha256(path / "manifest.json")


def make_windows(documents, block_size):
    """窗口不跨文档；每个有效 next-token 目标仅计一次，尾部右填充。"""
    if block_size < 1:
        raise ValueError("block_size 必须大于 0")
    xs, ys, owners = [], [], []
    for doc in documents:
        ids = doc["ids"]
        for start in range(0, len(ids) - 1, block_size):
            piece = ids[start:start + block_size + 1]
            x = torch.full((block_size,), PAD, dtype=torch.long)
            y = torch.full((block_size,), IGNORE, dtype=torch.long)
            x[:len(piece) - 1] = torch.tensor(piece[:-1])
            y[:len(piece) - 1] = torch.tensor(piece[1:])
            xs.append(x)
            ys.append(y)
            owners.append(doc["id"])
    if not xs:
        raise ValueError("没有可训练的 next-token 对")
    return torch.stack(xs), torch.stack(ys), owners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--documents", type=Path, help="可选的已授权 JSON 文档列表，每项需要 id/group_id/text")
    args = parser.parse_args()
    documents = json.loads(args.documents.read_text(encoding="utf-8")) if args.documents else None
    print(json.dumps(prepare_data(args.output, documents), ensure_ascii=False, indent=2))
    print(f"数据版本：{args.output.resolve()}")


if __name__ == "__main__":
    main()
