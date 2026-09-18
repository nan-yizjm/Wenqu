"""记忆系统的接缝：只定义协议与一个空实现。

产品本身**不实现记忆**。这里给出协议，等接入真正的记忆系统时替换
`NullMemoryProvider` 即可，产品其余代码不需要改。

为什么不顺手把存储做了：记忆要接哪一种（自研 / Mem0 / Zep 这类带事实抽取与
冲突消解的）还没定，而**接口形状定错的返工成本远高于先留一个空实现**。所以
P1 只交付协议 + 空实现，交付时行为与"没有记忆"完全一致。

四条硬约束（沿用产品既有约定，不新开先例）：

1. **默认关**：`settings.memory_enabled` 为 false 时，产品**根本不会调用**本协议
   （见 `chat.py` 的接线）。不是"调用了但结果被忽略"，而是不调用。
2. **落 `%LOCALAPPDATA%`**：实现只能写产品数据目录，与安装目录、仓库隔离。
3. **可查可删**：实现必须提供 `list` 与 `forget`，界面上能看到、能删——
   "记住了什么"必须是用户可审计的。
4. **只在本地**：任何实现都不许把记忆内容发往外部。

接续扩展点：若将来接入的系统要表达"新事实取代旧事实"，需要给 `MemoryItem`
加 `supersedes_id` 之类的字段；现在不加，因为还不知道要接哪个系统。
"""

from typing import Protocol, TypedDict, runtime_checkable


class MemoryItem(TypedDict):
    """一条可引用的记忆。

    `text` 要能直接拼进提示词并当作依据引用，所以是**一句事实**，不是整段对话。
    `expires_at` 是时效：过期即不再召回（`None` 表示不过期）。

    `derived_from` 是这条记忆的来源（哪次对话、哪份笔记）。**它刻意不叫 `origin`**：
    来源分层里已经有一个 `origin` 表示"这条依据属于哪一层"（笔记/记忆/网络），
    两者同时出现在一条来源记录里，同名会让"记忆来自哪次对话"和"这是记忆层"分不清。
    """

    id: str
    text: str
    derived_from: str
    created_at: str
    expires_at: str | None


@runtime_checkable
class MemoryProvider(Protocol):
    """记忆提供者的协议。产品只依赖这四个方法。"""

    name: str

    def recall(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """按查询召回记忆。返回空列表表示没有相关记忆，不是错误。"""
        ...

    def remember(self, items: list[MemoryItem]) -> None:
        """写入记忆。实现方决定抽取与冲突消解策略。"""
        ...

    def forget(self, item_id: str) -> None:
        """删除一条记忆。不存在时应当安静返回，不抛异常。"""
        ...

    def list(self, limit: int = 100) -> list[MemoryItem]:
        """列出记忆，供用户查看与删除。"""
        ...


class NullMemoryProvider:
    """默认实现：不做任何事。

    它不是临时占位，而是**出厂默认行为的一部分**——产品默认没有记忆，所以这个
    实现必须与"完全没有记忆系统"严格等价：不建文件、不建目录、不抛异常、不
    返回任何条目。以下写入方法刻意是空操作而非抛 `NotImplementedError`：
    接缝存在的意义是让产品能安全地调用它，任何一条路径抛异常都会让问答崩掉，
    而"接了记忆但实现有 bug"是最不该演变成"整个产品用不了"的情形。
    """

    name = "none"

    def recall(self, query: str, limit: int = 5) -> list[MemoryItem]:
        return []

    def remember(self, items: list[MemoryItem]) -> None:
        return None

    def forget(self, item_id: str) -> None:
        return None

    def list(self, limit: int = 100) -> list[MemoryItem]:
        return []
