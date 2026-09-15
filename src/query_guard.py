"""判断静态知识库是否不具备回答该问题的能力。

实时类词分两档：硬词本身几乎只出现在“问此刻世界状态”的问句里；
软词（今天、最新、价格……）在技术提问中同样高频，因此只有在整句缺少静态语境时
才视为实时请求。“当前”“现在”刻意留在软词之外——它们在技术提问里太常见，
且真正问天气股价时硬词已经拦下，加进来只会制造误判。
判定只是启发式，误判由前端“仍然提问”兜底，不要为此继续加长词表。
"""

REALTIME_HARD_PHRASES = (
    "天气",
    "气温",
    "空气质量",
    "汇率",
    "股价",
    "比分",
    "航班",
    "限行",
    "热搜",
)

REALTIME_SOFT_PHRASES = (
    "今天",
    "明天",
    "最新",
    "实时",
    "价格",
)

STATIC_CONTEXT_PHRASES = (
    "最新论文",
    "最新研究",
    "最新版本",
    "最新进展",
    "最新工作",
    "实时推理",
    "实时计算",
    "实时系统",
    "实时性",
    "价格模型",
    "价格策略",
    "价格敏感",
    "定价",
    "我的资料",
    "我的笔记",
    "资料里",
    "笔记里",
    "文档里",
    "这份文档",
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


def realtime_rejection_reason(normalized_query: str) -> str | None:
    """判断问题是否在索取静态笔记之外的实时数据。"""
    phrase = find_first_phrase(
        normalized_query,
        REALTIME_HARD_PHRASES,
    )

    if phrase is None:
        phrase = find_first_phrase(
            normalized_query,
            REALTIME_SOFT_PHRASES,
        )
        if phrase is not None and find_first_phrase(
            normalized_query, STATIC_CONTEXT_PHRASES
        ):
            phrase = None

    if phrase is None:
        return None

    return (
        f"问题包含“{phrase}”，需要实时或最新外部数据；"
        "当前知识库只包含静态笔记。"
    )


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

    return realtime_rejection_reason(normalized_query)