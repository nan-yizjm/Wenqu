"""联网补充的接缝：只定义协议与一个空实现。

产品**本期不接任何真实搜索后端**。这里给出协议，等确定用哪家（自带 key 的商用
搜索、自建 SearXNG、别的）时替换 `NullSearchProvider` 即可，产品其余代码不动。

为什么不顺手接一家：联网会打破"数据不出机器"这条卖点，所以先要把**边界**做
出来并固定住，再谈接哪家。边界比后端更难改——后端换一家只改一个类，边界松了
就是产品性质变了。

四条硬约束（沿用品既有约定，不新开先例）：

1. **默认关**：`settings.web_enabled` 为 false 时，产品**根本不会调用**本协议
   （见 `chat.py` 的接线），不是"调用了但忽略结果"。
2. **只发查询词**：发给后端的就是那串检索查询，**绝不携带笔记正文、片段内容
   或对话历史**。这一条有测试固定（拿被检索到的片段原文去反查，必须不在实参里）。
3. **没有后端时也不出网**：`NullSearchProvider.configured` 为 false，`augment()`
   在调用它之前就返回，所以**即使开关打开也一个请求都不发**，并如实上报
   `unconfigured`——不假装搜过、也不把"没配后端"说成"没搜到"。
4. **降级必须显式**：后端失败时返回 `failed` 状态（附带异常类型名）交给界面标注
   "本次未能联网"，不允许静默降级成"看起来就像没开联网"。

**为什么不加本地缓存表**：缓存键的规范化与 TTL 策略取决于后端（关键词怎么归一、
新闻与常青内容的过期时间不一样），现在加表就是加一份未经任何写入方验证的
schema。等接了真实后端、能拿真实请求跑出命中率与过期行为，再加表。

接续扩展点：需要"网页正文抽取"时，`SearchResult` 加 `content` 字段；需要"抓取
时间 vs 发布时间"的时效判定时，`published_at` 与 `fetched_at` 已经在协议里了。
"""

from typing import Protocol, TypedDict, runtime_checkable
from urllib.parse import urlparse


class SearchResult(TypedDict):
    """一条搜索结果。

    `snippet` 是后端已经抽好的摘要——产品不自己去抓网页正文，那会把"发查询词"
    升级成"把整页内容拉回来解析"，是另一个量级的隐私与工程问题。

    `published_at` 与 `fetched_at` **都留着**：时效判定要的是"这条事实何时成立"
    与"我何时看到的"，两者可能差很远（一篇 2023 年的论文，今天才被搜到）。
    """

    id: str
    title: str
    url: str
    snippet: str
    published_at: str | None
    fetched_at: str


@runtime_checkable
class SearchProvider(Protocol):
    """搜索后端的协议。产品只依赖这两样东西。"""

    name: str
    # 后端能否真的发出请求。没有 key、没配自建实例时应当是 false——
    # 产品据此上报"未配置"而不是"没搜到"，这两件事对用户的含义完全不同。
    configured: bool

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """按查询词搜索。返回空列表表示没有结果，不是错误。"""
        ...


class NullSearchProvider:
    """默认实现：什么都不做，也不需要任何凭据。

    它不是临时占位，而是**出厂默认行为的一部分**——产品默认没有联网能力，
    所以这个实现必须与"完全没有联网功能"严格等价：不建连接、不读凭据、不读
    磁盘、不抛异常。
    """

    name = "none"
    configured = False

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return []


def _host(url: str) -> str:
    """面包屑位置放域名而不是路径：判断"这条来自哪儿"时，域名比路径有用。"""
    host = urlparse(url).netloc
    return host or "网络"


def source_record(result: SearchResult) -> dict:
    """把一条搜索结果转成与检索结果同形的来源条目。

    形状必须与笔记片段、记忆条目一致，否则来源表、引用编号与前端分层都要为
    "网络这一层"写第三套判断。`chunk_id` 加 `web:` 前缀、`locator.kind` 是
    `web`、`document_id`/`version_id` 是占位串——它没有本地文档，也没有版本。
    """
    snippet = result.get("snippet", "")
    return {
        "origin": "web",
        "chunk_id": f"web:{result['id']}",
        "document_id": "web",
        "version_id": "web",
        "title": result.get("title") or _host(result.get("url", "")),
        "media_type": "web",
        "heading_path": _host(result.get("url", "")),
        "locator": {"kind": "web", "url": result.get("url", ""),
                    "published_at": result.get("published_at"),
                    "retrieved_at": result.get("fetched_at")},
        "preview": snippet[:360],
        # 网络事实要能和笔记一样被引用，所以这里也得有可拼进提示词的正文。
        "text": snippet,
        "score": None,
        "matched_tokens": [],
        "channels": {},
    }


def augment(provider: SearchProvider, query: str, limit: int) -> tuple[list[dict], dict]:
    """问一次搜索后端，返回 `(来源条目, 状态)`。

    状态**始终有值**，界面按它如实标注，不允许"说不清这次到底联没连上"。可能的
    取值：`off`（开关关着，调用方自己判，不在这里）／`unconfigured`（没有后端）／
    `ok`／`empty`（问到了，但没有结果）／`failed`（后端报错）。

    这里不做异常上抛：联网是辅助通道，它坏了不该让用户连自己的笔记都问不了。
    但这**不等于静默**——调用方会把 `failed` 落进 `app_events` 并交给界面显示。
    """
    if not getattr(provider, "configured", True):
        return [], {"status": "unconfigured", "provider": getattr(provider, "name", "unknown")}
    try:
        results = provider.search(query, limit=limit)
    except Exception as error:
        return [], {"status": "failed", "provider": getattr(provider, "name", "unknown"),
                    "detail": type(error).__name__}
    return ([source_record(item) for item in results],
            {"status": "ok" if results else "empty", "provider": getattr(provider, "name", "unknown")})
