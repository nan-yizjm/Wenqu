from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

QUERY = "如何缓解长上下文推理时的显存占用？"

PASSAGES = [
    (
        "KV Cache 架构优化",
        "GQA 和 MQA 通过减少 KV head 数量，"
        "MLA 通过压缩 KV 表示，降低 KV Cache 显存压力。",
    ),
    (
        "PagedAttention",
        "PagedAttention 像操作系统分页一样管理 KV Cache，"
        "减少显存碎片，提高推理吞吐。",
    ),
    (
        "LoRA",
        "LoRA 只训练低秩适配矩阵，用较低成本完成模型微调。",
    ),
]


def main() -> None:
    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    query_vector = model.encode(
        f"query: {QUERY}",
        normalize_embeddings=True,
    )

    passage_vectors = model.encode(
        [
            f"passage: {text}"
            for _title, text in PASSAGES
        ],
        normalize_embeddings=True,
    )

    scores = passage_vectors @ query_vector
    ranked_indexes = sorted(
        range(len(PASSAGES)),
        key=lambda index: scores[index],
        reverse=True,
    )

    print(f"模型：{MODEL_NAME}")
    print(f"设备：{model.device}")
    print(f"问题：{QUERY}")
    print(f"向量维度：{query_vector.shape[0]}")
    print("\n语义检索排序：")

    for rank, index in enumerate(ranked_indexes, start=1):
        title, text = PASSAGES[index]

        print(f"\n[{rank}] 相似度：{scores[index]:.4f}")
        print(f"标题：{title}")
        print(f"内容：{text}")


if __name__ == "__main__":
    main()