# 阶段 12：从命令行程序到本机 HTTP 服务

日期：2026-09-12。入口：`src/serve.py`；共享运行时：`src/runtime.py`；真实检查：`src/service_probe.py`。

## 1. 本阶段做了什么

把已有 RAGAnswerer 包装成固定索引版本的 HTTP 接口。算法流程没换，但程序开始需要处理生命周期、输入验证、并发、超时、错误与日志。

当前只服务本机，监听 `127.0.0.1:8000`。没有公开网站、局域网开放、防火墙修改、后台自启动或付费 API 自动调用。没有引入 Docker、向量数据库或复杂队列。

本机缺少 FastAPI / Uvicorn / Pydantic，所以安装了服务依赖，版本记录在 `requirements-service.txt`。没有升级或替换已经可用的 CUDA PyTorch。安装后 `pip check` 没有依赖冲突。

## 2. 启动、调用、停止

终端 A，先进入项目目录：

```powershell
Set-Location -LiteralPath '<项目目录>'
.\.venv\Scripts\python.exe -X utf8 -m src.index_store build
.\.venv\Scripts\python.exe -X utf8 -m src.serve
```

等到 `Application startup complete` 后，在终端 B 调用：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health/ready'
Invoke-RestMethod 'http://127.0.0.1:8000/v1/index'

$payload = @{ question = 'PagedAttention 是什么？它解决什么问题？' } | ConvertTo-Json
$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
$result = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/v1/answer' -ContentType 'application/json; charset=utf-8' -Body $bodyBytes
$result | ConvertTo-Json -Depth 8
```

中文请求显式编码成 UTF-8 字节，避免不同版本 PowerShell 的默认编码差异。读取保存的 JSON 也用 `Get-Content -Encoding UTF8 -Raw`。

终端 A 按 `Ctrl+C` 停止服务。本轮已经实际启动、调用并正常停止，交付时没有留下这个 HTTP 服务进程。Ollama 是你原有的服务，没有顺带关闭或卸载。

完整 HTTP 冒烟检查（需要终端 A 正在运行；会执行两次本地生成）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.service_probe
```

它先确认服务供应商是 Ollama，避免不小心对 DeepSeek 跑自动生成检查。报告放到被忽略的 `data/generated/`，其中有答案和来源，不应直接公开上传。

## 3. 接口与返回值

| 接口 | 作用 | 注意点 |
|---|---|---|
| `GET /health/live` | HTTP 应用能回应 | 不代表生成模型能用；启动加载未结束时还不能响应 |
| `GET /health/ready` | 运行时已加载，并检查 Ollama 模型是否已安装 | 不是一次真实生成；不保证下一次无 OOM |
| `GET /v1/index` | 查看进程固定的版本、模型和检索配置 | 它可能与刚更新的磁盘 current 不同 |
| `POST /v1/answer` | `{ "question": "…" }` → 带来源的回答 | 不接受客户端指定服务器文件路径或覆盖供应商 |
| `GET /docs` | 自动生成的交互接口说明 | 默认 Swagger UI 资源来自 CDN，离线时可直接用 PowerShell |

问答响应包含：`request_id`、`index_version`、`answer`、`sources`、`generation_calls`、`citation_validation` 和 `elapsed_ms` 等字段。每个来源有 chunk ID、笔记相对路径、标题路径和快照行号。**这些行号属于回答使用的快照，不承诺等于正在编辑的最新文件行号。**

标签合法只说明引用编号存在于上下文，不等于它支持回答的每个结论。`rejected=true` 表示现有静态能力守卫拒绝；模型因资料不足而用自然语言拒答时，不一定会设这个标志。

DeepSeek 模式的 readiness 只表示本地客户端配置已建立，不发收费探测请求、不声称远端当前可用。Ollama 暂时不通时 readiness 返回 503；独立的 OOD 问题仍可能在 POST 内被守卫直接拒绝，不需要实际生成。

## 4. 生命周期：模型应在什么时候加载

实现用 FastAPI 的 `lifespan` 管理启动和退出：启动读取配置、验证 current、固定版本、构建 BM25，并按配置加载 E5/重排模型；退出等待已有计算结束。多个请求复用同一运行时。这个组织方式依据 [FastAPI 生命周期文档](https://fastapi.tiangolo.com/advanced/events/)。

没有把所有重模型放到模块 import 时加载，否则只是导入一个 HTTP 测试也会占用 GPU。没有每个请求创建 RAGAnswerer，否则延迟会反复包含模型加载。

服务设置 `workers=1`，不启用 reload。多 worker 是多个进程，不是免费获得并发，每个进程会建立各自模型实例；在 8GB GPU 上尤其需要先算资源。监听、worker 和日志开关可对照 [Uvicorn 设置文档](https://uvicorn.dev/settings/)。

索引更新与模型计算分开：服务运行中可以用另一个进程构建新版本，但新构建也会占 GPU；本机空间紧张时，建议先停服务再构建，或把构建设备改为 CPU。**单请求锁只限制本服务的计算，不限制外部 Ollama 客户端或别的 Python 进程。**

## 5. 最容易误解的问题：超时不等于取消计算

现有模型客户端使用同步 HTTP 调用。HTTP 接口等待 90 秒后可以给用户返回 504，但 Python 不能因此安全地强杀正在执行的线程；Ollama 端也可能仍在生成。

本版的 `SingleFlight` 采用：

```text
请求 A 获得席位 → 后台线程计算
        HTTP 等待超时 → A 收到 504，但席位仍占用
请求 B 到来 → 429 busy
        A 的后台线程真正结束 → 释放席位
下一请求 → 可以计算
```

通过 `asyncio.shield` 避免请求等待被取消时顺带取消 Future；锁在后台函数的 `finally` 中释放，不在 HTTP 超时分支释放。后台异常会被消费，不把错误详情或密钥写到响应中。

用受控阻塞的假模型验证了这条时间线：计算时 429，1 秒 HTTP 超时后 504，后台尚未放行时下一请求仍 429，真正结束后恢复 200。这些是故障注入验证，不是声称真实模型刚好在 1 秒超时。

局限：没有真正的生成取消、流式 token 或“停止生成”按钮。HTTP 超时不会释放 Ollama 的计算资源；客户端现有 120 秒网络 timeout 也不是对所有计算情况的严格总时限。退出等待后台完成，异常卡住时仍需要人工排查进程；不能把 `async def` 当作可抢占计算。

## 6. 输入、错误和日志

| HTTP 状态 | 本版含义 | 常见排查 |
|---|---|---|
| 200 | 得到回答或守卫拒绝结果 | 检查 rejected、引用与版本，不只看状态码 |
| 403 | 非本机 Host 或不允许的 Origin | 用配置端口的 localhost/127.0.0.1；不要跨站调用 |
| 413 | 实际请求体字节超限 | 缩短输入，不上传整篇长文 |
| 415 | POST 不是 application/json | 设置 Content-Type |
| 422 | 空问题、类型不对、未知字段或问题太长 | 只传 question 字符串 |
| 429 | 已有一个问答在计算 | 等待，参考 Retry-After，不密集重试 |
| 502 | 后端计算失败 | 检查 Ollama、模型安装、显存；错误响应不回显底层详情 |
| 503 | readiness 检查失败 | 检查生成服务；本状态来自健康检查 |
| 504 | HTTP 等待超时，后台仍可能工作 | 不立即并发补发很多请求 |

默认问题最多 2000 字符，实际请求体最多 `4 * max_question_chars + 1024` 字节。后者按收到的字节检查，不信任 Content-Length。它是请求限制，不是 E5/生成模型的 token 保证；极长问题仍可能被检索编码器截断，这个限制要在后续 token 预算阶段改进。

Host / Origin 限制是本机服务的防护，不是用户身份认证。有本机访问能力的程序仍能调用；不要把它直接反向代理成公开 API。本版不开放任意文件读取、工具执行或更新索引 HTTP 路由。

结构化日志记录请求 ID、固定版本、状态和总耗时；模型失败只记录异常类型。不记录问题全文、检索上下文或 API Key。关闭默认访问日志，避免用户把内容放到 URL 查询串后被原样记录。响应带 `X-Request-ID` 和 `Cache-Control: no-store`。诊断 CLI 报告可能包含答案，但它与服务日志是两种不同用途的材料。

## 7. 真实运行结果

报告：`data/generated/service_probe_20260912_142755_755055.json`。

- 服务版本：`20260912T062623795056Z-2d6e3a20`。
- Ollama `qwen2.5:7b`，hybrid，blocks，未开启重排，911 chunks。
- 10 项真实 HTTP 合约检查通过：索引信息、存活、就绪、两道知识库问题、OOD、空问题、未知字段、超大请求、跨站请求。
- RAG 流程题约 5476 ms；PagedAttention 题约 919 ms；天气守卫约 3 ms、0 次生成调用。
- 两个实际答案都使用了合法的 S1 引用。RAG 答案仍偏简略，并没有独立展开上下文注入等全部步骤；不能据此宣称答案质量问题已解决。
- 服务与生成模型共存时，`nvidia-smi` 一次采样为 5790 / 8151 MiB。它包含整张卡上当时各进程的占用，不是某一个模型的精确显存，更不是并发负载下的峰值。
- 正常 Ctrl+C 停止，退出日志包含 `Application shutdown complete`；随后确认 8000 没有监听。

本轮离线回归为 **69 个测试通过**，包括增量更新、数据损坏、模型失败、并发和超时场景。编译检查通过，`pip check` 无冲突。测试客户端提示 Starlette 对旧 httpx 的弃用警告：当前仍兼容并通过，暂未为了消除提示更换整套测试传输库；这不是运行请求失败。注入假模型错误时出现一条只带 RuntimeError 类型的日志，也是预期检查。

没有重跑所有生成评测来追求涨分，HTTP 的 10/10 也**不是答案正确率 100%**。新语料与旧语料不一致，之前 13/18 等数字仅保留作历史实验记录。

## 8. 学习时建议这样追代码

先从 `create_app` 看四个业务路由，再看 lifespan 如何建立 runtime，然后看 `SingleFlight.submit` 为什么不排队。最后沿 `runtime.answer → RAGAnswerer.answer` 回到已有检索与生成流程。

练习与参考答案：

1. **ready=true，但第一道题很慢，矛盾吗？** 不矛盾。readiness 检查 Ollama 模型已安装，不会提前收费/耗时生成；生成模型可能还需加载，首题也可能有额外初始化。
2. **build 成功后，为什么 HTTP 仍返回旧版本？** 进程固定版本，重启才切换。这避免同一请求检索旧 chunks 却拼接新 sources。
3. **把 workers 改成 4 能提高四倍吞吐吗？** 不能这样推断。模型实例、显存、Ollama 队列和 GPU 计算都是约束；要先测并发和资源，再改架构。
4. **答案有 S1 就一定有证据吗？** 标签检查只验证来源编号合法。仍需查看具体来源是否支持每个结论。

下一阶段可以先做 CPU/GPU 分工、冷启动与热请求的资源实验，再转入手写 attention / KV Cache。流式输出、真正取消、细粒度阶段耗时、模型 token 预算、证据级评测和公开服务认证都还是待办。
