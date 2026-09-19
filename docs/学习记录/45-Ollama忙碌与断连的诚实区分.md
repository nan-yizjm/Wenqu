# 学习记录 45：Ollama 忙碌与断连的诚实区分

> 日期：2026-09-19。41 号 §四.1 的遗留："双实例并发抢同一个 Ollama 撞 120 秒超时"。本次解剖后发现它不只没修，还在**两层报错里说谎**：服务好端端地运行着，用户却被引导去"确认服务已启动且模型已下载"。修法是一个异常类型 + 三条话术的边界，`timeout=120` 本身不动。

## 一、说谎链的两层

超时的真因是**排队**：本地 Ollama 的生成槽被另一个请求占着（另一个工作台实例、一条长生成），第二个请求连上了 TCP、发出去了 HTTP，只是在队列里等数据，等满 120 秒读超时。而用户看到的是：

1. **第一层（llm.py）**：urllib 在等待响应头阶段的读超时把 `socket.timeout` 包装成 `URLError(TimeoutError)`——落进 `except URLError` 分支，报 **"无法连接到 Ollama。请确认 Ollama 已安装并正在运行。"** 连接明明建立了。
2. **第二层（safe_error）**：`"Ollama" in text` 模糊匹配 → 错误码 `ollama_unavailable` → **"无法使用本机 Ollama，请确认服务已启动且模型已下载。"** 服务在运行、模型在显存里。

还有第三条隐路径：响应头回来之后、流式读行之间的停滞超时是**裸 `TimeoutError`**（不经 urllib 包装，不是 `URLError` 子类）——直接冒泡进 `generation_failed` "生成暂时失败，请检查模型设置后重试"，同样没说真因。

41 号记录的 `ollama_unavailable` 就是第一层+第二层的合成物。

## 二、修法：一个类型，三条话术

- `src/llm.py` 新增 **`OllamaBusy(RuntimeError)`**，统一话术由 `_busy()` 生成：**"本机 Ollama 在 120 秒内没有返回数据：它可能正被其他任务占用（比如另一个工作台窗口正在生成长内容）。等它空闲后重试。"**——说观察到的事实（120 秒没数据）、最可能的原因、可操作的建议，不让人去检查没坏的东西。
- 触发点三个，都归到 `OllamaBusy`：
  1. `chat()` / `stream_chat()` 里 `URLError.reason` 是 `TimeoutError`（等响应头超时）——`isinstance(error.reason, TimeoutError)` 区分，`ConnectionRefusedError` 等仍然走"无法连接"；
  2. 响应体读取阶段的裸 `TimeoutError`（行间停滞）——新增 `except TimeoutError`；
- `src/product/chat.py` 的 `safe_error` 在 `"Ollama" in text` 模糊匹配**之前**加 `isinstance(error, OllamaBusy)` 分支：错误码 `ollama_busy`，消息**原样透传**（它本来就是给人看的）。三条话术的边界从此清晰：忙（连得上没数据）/ 断连（连不上）/ HTTP 错误（服务端给了状态码）。

**`timeout=120` 不调**。41 号给过两个方向："调 timeout 或加排队提示"。调大让用户干等更久，而且双实例下重试照样排队——报错诚实是更对的那个默认。今天这台机器上排队通常几十秒内消化（见 §四），干等 300 秒换一个"其实早就能失败"的结果，不值得。

## 三、测试与短路

- 后端 **344 项**全过（新增 6：`OllamaBusyTests` 4 条——排队超时、行间停滞、两种拒连保持原话；`SafeErrorTests` 2 条——忙透传、断连保持 `ollama_unavailable`）。
- 短路 3 点（`.workbuddy/shortcircuit_ollama_busy.py`）：safe_error 去掉 busy 分支 → 落回 `ollama_unavailable`，透传测试挂；URLError 分类去掉超时判断 → 排队超时误报断连，挂；`except TimeoutError` 改原样上抛 → 行间停滞测试挂。全部"破坏后必挂"。

## 四、真实验证：41 号的场景被 Ollama 并行化部分消解（本篇的意外发现）

两轮真实复现（`.workbuddy/ollama_busy_repro.py`：占用者直调 Ollama API 强制生成 ~8500 token ≈ 130 秒，同时产品客户端发短消息）：

| 轮次 | 占用者 | 排队者结果 |
| --- | --- | --- |
| 1 占用 + 1 排队 | finished | **14.5 秒正常完成**（并行跑了） |
| 3 占用 + 1 排队 | 全部 finished | **14.2 秒正常完成**（仍并行） |

**这台机器的 Ollama 并行度 ≥4**——41 号实证的"排队 120 秒"在当前版本上需要并发超过并行 slot 数才可能复现，4 条以内全部并行消化，排队超时事实上被 Ollama 的并行化消解了大半。没有再往上堆并发（VRAM 和使用中的模型槽都不适合拿来造事故）。

所以这次修复的价值定位要说准：**它保证无论使用者的 Ollama 版本/配置如何，只要 120 秒读超时真的发生（旧版单并发排队、读停滞、显存交换导致的极慢），话说得都对**；顺带把 41 号场景在新版 Ollama 下的现状（并行消化）记录在案。

## 五、边界

- DeepSeek 侧的 `URLError` 分支有同构问题（超时也说"无法连接"），但它是远端 API：没有"被本地别的窗口占用"的场景，"请检查网络连接"对超时大体成立。保持现状，不为对称而对称。
- `safe_error` 的透传依赖 `isinstance` 而不是字符串匹配——`OllamaBusy` 是 `RuntimeError` 子类，跨层传递类型不丢失（生成器与异常链都保真）。
