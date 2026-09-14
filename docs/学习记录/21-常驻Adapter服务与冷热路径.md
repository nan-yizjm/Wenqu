# 阶段 21：常驻 adapter 服务、冷热路径与取消边界

日期：2026-09-14。接续[阶段 20](20-Adapter接入真实RAG与上下文故障.md)。前一阶段的 `src.adapter_rag` 每回答一题都会重新加载基座、adapter 和 E5，适合保存单次实验，却不是正常服务的运行方式。本阶段把同一个实验 adapter 放进常驻进程，连续处理请求，并实际区分启动、首次生成、热生成和不经过模型的拒答。

## 1. 本阶段解决什么问题

原单次命令的总耗时混在一起：

```text
加载 4-bit 基座和 adapter
        ↓
加载 chunk、BM25、E5 向量
        ↓
检索、上下文、生成
        ↓
进程退出，显存释放
```

如果每题都重复加载，就无法回答这些常见部署问题：模型常驻后一次请求到底多慢；第一题与后续题是否有 CUDA 首次执行差异；OOD 守卫是否真的绕过 GPU；服务繁忙时是否会让两个生成同时挤进 8 GB 显存；HTTP 超时是否等于模型已停止。

## 2. 新增的运行方式

新增三个入口：

- `src/adapter_runtime.py`：固定当前索引、Qwen2.5-1.5B commit 和 adapter，一次初始化后重复调用；
- `src/serve_adapter.py`：复用已有 FastAPI 的本机 Host/Origin 限制、请求大小、单 GPU 席位、超时和错误清理；
- `src/adapter_service_probe.py`：只接受 `hf-qlora-experimental` 服务，连续发送两次相同问题和一次 OOD 问题，保存可复查报告。

启动命令：

```powershell
.\.venv\Scripts\python.exe -m src.serve_adapter `
  --adapter .\data\generated\qlora_runs\20260914_200637_007529\best_adapter `
  --max-new-tokens 160
```

另开 PowerShell 执行：

```powershell
.\.venv\Scripts\python.exe -m src.adapter_service_probe
```

完成后在服务窗口按 `Ctrl+C`。本入口不会修改 `rag.toml` 的默认 `provider = 'ollama'`，也不会调用 DeepSeek。

## 3. 生命周期和请求数据流

```text
进程启动（只执行一次）
  ├─ 读取 current 索引 manifest
  ├─ 加载固定 4-bit Qwen 基座 + adapter
  ├─ 加载 911 个 chunk、BM25 与 E5 向量
  └─ /health/ready = true

每个 /v1/answer
  ├─ 本机 Host / Origin / JSON / 长度校验
  ├─ 尝试获取唯一生成席位；失败返回 429
  ├─ 静态 OOD 守卫
  ├─ hybrid 检索与块上下文
  ├─ adapter 生成（OOD 时跳过）
  ├─ 引用标签校验
  └─ 返回固定索引版本、请求序号和阶段指标
```

模型没有在请求结束时卸载。服务关闭后显存从常驻时约 2795 MiB 回落到约 740 MiB。

## 4. 真实运行结果

最终报告：

```text
data/generated/adapter_service_probe_20260914_203346_936237.json
```

固定对象：

| 项目 | 值 |
|---|---|
| 索引版本 | `20260912T062623795056Z-2d6e3a20` |
| 索引片段 | 911 |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4c` |
| adapter 权重 SHA256 | `6aa483bc5cc01be7d35ff2e9461e79bb9b7b8d96beb8c1fbb1d18b3f4e48fef9` |
| 索引 manifest SHA256 | `54d0cb1596f346503c37d1e6dff678d45ae5a3c7c688118d25918095d666155a` |

时延：

| 阶段 | 实测 |
|---|---:|
| 服务启动总耗时 | 5605.11 ms |
| 其中基座 + adapter 加载 | 3569.51 ms |
| 索引、检索器及其余初始化 | 2035.60 ms |
| 首次回答请求 | 3750.74 ms |
| 首次请求中模型生成 | 3543.32 ms |
| 第二次热回答请求 | 3094.18 ms |
| 第二次请求中模型生成 | 3082.63 ms |
| OOD 运行时处理 | 0.01 ms |

首题和热题返回完全相同的 PagedAttention 答案，`request_sequence` 分别为 1、2；OOD 是第 3 个请求，`rejected=true`、`generation_calls=0`，且 `local_model_metrics={}`。自动检查 5/5 通过。

热题比首题少约 657 ms。这里只测了一个固定问题的两次运行，不能直接称为普遍加速比。它能证明模型没有重新加载，并且热路径确实与启动阶段分离。

## 5. 修复的指标污染问题

`HFQLoRAClient.last_metrics` 保存最近一次真实生成。若前一题调用了模型，后一题被静态守卫拒答，而运行时不先清空该字段，那么后一题会错误携带上一题的生成耗时和显存。

现在每题开始前先清空指标：普通题由 `chat()` 写入本题指标；OOD 不调用 `chat()`，指标保持空字典。这类错误不会影响答案文本，却会让监控得出“OOD 也调用了 GPU”的假结论。工程评测不仅要检查模型输出，也要检查观测数据是否属于当前请求。

## 6. 911 与旧实验的 757 为什么不同

先前直接运行 `src.vector_retrieve` 时看到 757 个 chunk，那是旧的非版本化实验语料。当前服务严格读取 `IndexStore.current()`，对应版本化索引，共 911 个 chunk。

因此比较两次检索或生成时，必须同时固定：

```text
模型 + adapter + 索引版本 + 配置 + prompt 契约 + 源码
```

只说“都用了 E5”不足以归因。语料切分数从 757 变为 911，本身就可能改变候选、上下文和时延。

## 7. 为什么暂时没有直接加流式输出

当前模型客户端调用的是一次性 `model.generate()`。FastAPI 把它交给线程，并不意味着生成可取消：

- HTTP 的 504 只是请求方停止等待；
- Python 工作线程仍在 `generate()` 内运行；
- 唯一 GPU 席位要等真实生成结束才释放；
- 强行把 Future 标为 cancelled 不能中断 CUDA kernel。

正确的流式/取消边界应落在模型生成层：

1. `TextIteratorStreamer` 逐段产生文本；
2. `StoppingCriteria` 每个 decode 步检查一个取消事件；
3. HTTP 层发现客户端断开或超时后设置事件；
4. 生成线程真实退出后才释放 GPU 席位；
5. 流中途断开时，不把半截文本当成完整、已引用校验的答案。

这不是在 HTTP 返回类型上改成流就能解决的问题。下一阶段应先做本地可取消生成实验，再接 SSE/NDJSON；否则只是“看起来流式”，后台仍不可控。

## 8. 已知限制

- 服务仍是单进程、单生成请求，第二个并发请求返回 429，没有持久队列；
- `request_timeout` 后底层生成仍继续，尚无 token 级取消；
- adapter 仍有阶段 19/20 记录的部分证据过度作答问题；
- 引用校验只验证标签，不验证每条陈述是否被来源蕴含；
- `non_generation_ms` 是总耗时减生成耗时，只是聚合差值，不是精确 profiler；
- 本机接口无账号认证，只绑定 `127.0.0.1`，不应开放公网；
- Starlette 测试客户端仍有 `httpx` 弃用警告，是后续依赖迁移债务，不影响实际 uvicorn 服务。

## 9. 练习与参考答案

### 练习 1

为什么常驻服务中仍要区分“首次请求”和“热请求”？

参考答案：模型虽然已经加载，但第一次实际计算还可能初始化 CUDA kernel、内存池或内部缓存。稳定运行的请求通常更接近热路径，因此不能把启动完成后的第一题直接当长期延迟。

### 练习 2

为什么 HTTP 504 后仍返回 429 是合理的？

参考答案：504 只说明客户端等待超时，不证明 GPU 生成已停止。若立即释放席位，第二次生成会与仍运行的第一次重入 GPU，可能导致显存溢出。应等底层任务真正退出后再释放。

### 练习 3

为什么 OOD 请求的模型指标必须是空，而不是 0 ms？

参考答案：空表示根本没有发生模型调用；0 ms 容易被理解为调用发生但耗时恰好为零。缺失事件与数值为零是不同语义。

### 练习 4

为什么不能拿 757-chunk 实验与 911-chunk 服务的答案直接判断 adapter 变好或变差？

参考答案：输入语料和切分已变化，检索候选与上下文也会变化。模型不是唯一自变量，因而无法把答案差异归因于 adapter。
