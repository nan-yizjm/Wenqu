# 阶段 23：真实生成 token 预算与完整证据阶梯

日期：2026-09-14。接续[阶段 22](22-流式输出与真实取消.md)。当前上下文一直用 `max_context_chars=2200` 控制，但模型的上下文窗口按 tokenizer token 计算。字符不越界不能证明 prompt 不越界，而且直接开启 tokenizer truncation 可能静默切掉问题、来源或引用标签。

## 1. 项目里至少有三种“长度”

```text
字符数：上下文打包器便于快速规划完整 Markdown 块
E5 token：向量编码模型的 512-token 输入限制
Qwen token：chat template 后，生成模型真正接收的输入长度
```

它们不能互换。中文一个字符可能接近一个 token，也可能被不同词表拆分；英文技术词、路径、标点和 chat template 特殊 token 又会改变比例。

本阶段约束：

```text
actual_prompt_tokens <= max_prompt_tokens
max_prompt_tokens + max_new_tokens <= model_context_tokens
```

Qwen2.5-1.5B 的模型配置提供上下文窗口；实验服务默认保留 2048 个 prompt tokens 和 160 个生成 tokens。它不是模型理论最大值，而是当前本机 RAG 实验的受控上限。

## 2. 为什么不直接用 truncation

若在 tokenizer 上写 `truncation=True`，虽然张量能进入模型，但系统未必知道具体丢了什么：

- 问题可能被截断成不完整要求；
- 来源头存在，关键证据正文却消失；
- Markdown 代码块或列表在中间断开；
- 模型仍可能输出一个看似正常、带合法标签的答案。

当前方案先按完整证据块打包，再用 Qwen 自己的 chat template 计数。超预算时，保留同一次检索结果，降低字符预算并重新打包；每个候选方案重新计数。最终只接受真实 prompt token 数不越界的完整块组合。

## 3. RAGAnswerer 的预算流程

```text
检索（只执行一次）
   ↓
按配置字符预算打包
   ↓
构造 system/user chat messages
   ↓
Qwen tokenizer 真实计数
   ├─ 未超：直接使用
   └─ 超出：在更小字符预算中二分寻找安全组合
              ├─ 至少一个完整可引用证据能放入：生成
              └─ 一个都放不下：明确预算不足，零模型调用
```

`PreparedAnswer` 和最终 `AnswerResult` 新增：

- `prompt_tokens`：本次实际 prompt token 数；
- `prompt_token_budget`：允许上限；
- `context_char_budget`：最终打包时采用的字符预算；
- diagnostics：每次候选预算、上下文字符数、token 数和是否含来源。

检索不在每个预算候选里重跑，因此不会因预算收紧得到另一套候选，也避免把多次向量检索时间混入准备阶段。

## 4. 第一次真实实验为什么全部显示 2 tokens

第一次报告：

```text
data/generated/token_budget_lab_20260914_210422_710038.json
```

四档预算都显示 `prompt_tokens=2`，而实际生成客户端报告约千 tokens，检查失败。根因不是 tokenizer，而是当前 Transformers 版本：

```python
tokenized = tokenizer.apply_chat_template(..., tokenize=True)
type(tokenized)  # BatchEncoding
tokenized.keys() # input_ids, attention_mask
len(tokenized)   # 2：字段数量，不是 token 数
```

修复后读取 `tokenized["input_ids"]` 的长度，并兼容 list、嵌套 batch 和 Tensor。新增检查专门防止以后再次把 BatchEncoding 字段数当 token 数。

这个错误说明：变量名叫 `token_ids`、数值是整数，并不证明度量语义正确。观测系统也需要与真实生成路径交叉核对。

## 5. 当前索引上的真实对照

命令：

```powershell
.\.venv\Scripts\python.exe -m src.token_budget_lab `
  --adapter .\data\generated\qlora_runs\20260914_200637_007529\best_adapter
```

最终报告：

```text
data/generated/token_budget_lab_20260914_210526_416395.json
```

固定问题：

```text
PagedAttention 是什么？它解决什么问题？
```

结果：

| prompt 上限 | 实际 prompt | 打包字符预算 | 实际上下文字符 | 来源数 | 是否生成 |
|---:|---:|---:|---:|---:|---|
| 2048 | 1048 | 2200 | 1676 | 3 | 是 |
| 1024 | 1021 | 1675 | 1633 | 3 | 否，只检查准备 |
| 768 | 501 | 1612 | 748 | 2 | 否，只检查准备 |
| 512 | 501 | 1612 | 748 | 2 | 是 |

5 项检查全部通过：所有可用 prompt 均不越界、完整预算不缩减、至少一档发生重打包、生成回答引用标签合法、生成客户端计数与准备阶段完全一致。

## 6. 为什么 768 和 512 得到同一个 501

完整块策略会产生离散台阶：

```text
当前两条来源组合 = 501 tokens
再加入下一完整证据块 > 768 tokens
```

所以 512～768 之间的额外空间不足以容纳下一整块，两档保留相同上下文。字符预算从 1612 再增加一点，并不会逐字符填满；一旦跨过块门槛，prompt 可能突然跳到约千 tokens。

这正是保结构与充分利用窗口之间的取舍。若追求连续填满，可以在块内按句子二次切分，但会再次引入列表、代码与语义关系被切断的问题。

## 7. 两个生成结果

2048 档：

```text
结论：PagedAttention 是一种优化技术，主要用于管理键值缓存（KV Cache），
通过像操作系统分页一样管理 KV Cache，减少碎片，提高缓存的利用率。
依据：[S2]
```

- prompt 1048 tokens；
- 生成 46 tokens；
- 生成约 3356.57 ms；
- 峰值 allocated 约 1801.97 MiB。

512 档：

```text
结论：PagedAttention 是一种优化技术，用于管理键值缓存，
以减少碎片和提高性能。
依据：[S2]
```

- prompt 501 tokens；
- 生成 31 tokens；
- 生成约 1975.22 ms；
- 峰值 allocated 约 1691.10 MiB。

低预算更快，但不能把差值全部解释成 prefill 缩短：低预算回答本身也少生成了 15 tokens，decode 时间同时变化。若要分离 prefill/decode，应固定输出 token 数或使用 profiler。

两次引用结构都合法，但回答都只引用 S2；“减少碎片”的最直接来源其实是 S1。token 预算保证输入可执行，不负责判断某条陈述是否由所引来源蕴含。阶段 20 的引用支持问题仍然存在。

## 8. 接入方式

常驻 adapter 服务新增：

```powershell
python -m src.serve_adapter `
  --adapter <adapter目录> `
  --max-prompt-tokens 2048 `
  --max-new-tokens 160
```

`/v1/index` 返回模型窗口与两类预算；普通和流式回答返回本题的实际 prompt tokens、预算和最终字符预算。一次性 `src.adapter_rag` 也使用同样规则。默认 Ollama/DeepSeek 没有强行套用 Qwen tokenizer，原入口保持原行为。

## 9. 当前限制

- 二分搜索依赖“字符预算越大，token 大致越多”的工程近似；完整块重排和不同 token 密度可能不严格单调，因此结果保证安全，但不保证是所有组合中的全局最大利用率；
- 每个超预算请求会重新打包和 tokenize 约 10～12 次，当前规模很轻，但高并发下应缓存或改成 token-aware packer；
- 当前只限制输入和最大输出之和，不动态根据答案所需长度分配；
- 预算失败是本地能力边界，不等同于知识库 OOD；
- 生成模型 token 预算解决不了 E5 输入限制，两者仍分别控制；
- 只对本地固定 Qwen tokenizer 做了严格计数，远端 provider 的实际 tokenizer 需要供应商对应实现；
- 完整块导致窗口剩余空间无法利用，后续可研究“语义单元 token 长度预计算 + 背包选择”。

## 10. 练习与参考答案

### 练习 1

为什么 `max_context_chars=2200` 不能推出 prompt 小于 2200 tokens？

参考答案：prompt 还包含 system、问题、来源头和 chat template 特殊 token；字符与 token 也不是一一对应。只能用实际生成模型 tokenizer 对最终 messages 计数。

### 练习 2

为什么预算不足时不应让 tokenizer 自动截断？

参考答案：自动截断只保证张量长度，不保证证据、问题和引用契约完整。系统可能在不知道证据已丢失的情况下继续生成有引用外观的答案。

### 练习 3

512 和 768 都得到 501 tokens，剩余空间是不是浪费？

参考答案：是有意识的结构性空余。下一完整块放不下，而当前策略不拆块。它用利用率换取列表、代码和语义单元完整。

### 练习 4

低预算生成更快，能否直接得出 prefill 加速了约 1.38 秒？

参考答案：不能。两次输入长度不同，输出也从 46 降到 31 tokens；总生成时间同时含 prefill 和逐 token decode。需要固定输出或单独 profiler 才能归因。
