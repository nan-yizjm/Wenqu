REALTIME_DATA_PHRASES = (
    "今天",
    "明天",
    "当前",
    "现在",
    "实时",
    "最新",
    "天气",
    "空气质量",
    "价格",
    "汇率",
    "股价",
)

EXTERNAL_ACTION_PHRASES = (
    "预订",
    "订票",
    "购买",
    "下单",
    "支付",
    "发送邮件",
    "删除文件",
    "修改文件",
)

PERSONAL_CONTEXT_PHRASES = (
    "我的简历",
    "根据我",
    "适合我",
    "我的经历",
    "我的偏好",
)


def find_first_phrase(
    text: str,
    phrases: tuple[str, ...],
) -> str | None:
    """返回文本中命中的第一个短语。"""
    for phrase in phrases:
        if phrase in text:
            return phrase

    return None


def static_corpus_rejection_reason(query: str) -> str | None:
    """判断静态知识库是否不具备回答该问题的能力。"""
    normalized_query = query.lower()

    action_phrase = find_first_phrase(
        normalized_query,
        EXTERNAL_ACTION_PHRASES,
    )

    if action_phrase is not None:
        return (
            f"问题包含“{action_phrase}”，需要执行外部动作；"
            "当前项目只有只读知识库能力。"
        )

    personal_phrase = find_first_phrase(
        normalized_query,
        PERSONAL_CONTEXT_PHRASES,
    )

    if personal_phrase is not None:
        return (
            f"问题依赖“{personal_phrase}”相关的个人资料；"
            "当前知识库没有这些用户上下文。"
        )

    realtime_phrase = find_first_phrase(
        normalized_query,
        REALTIME_DATA_PHRASES,
    )

    if realtime_phrase is not None:
        return (
            f"问题包含“{realtime_phrase}”，需要实时或最新外部数据；"
            "当前知识库只包含静态笔记。"
        )

    return None