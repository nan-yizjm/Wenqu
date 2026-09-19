# Obsidian RAG

一个面向个人 Obsidian 知识库的、可引用且可评测的问答助手。

## 当前开发状态：个人知识工作台（2026-09-15）

项目主线已经从命令行 RAG 实验切换到可安装的 Windows 产品。阶段 28～31 完成产品骨架、Markdown/PDF 资料库、证据阅读、流式问答、会话、收藏、反馈和 Markdown 导出。阶段 32 已加入完整备份/校验恢复、数据库迁移失败安全模式、脱敏诊断、用户指南、第三方许可证清单和示例资料。Markdown 引用保留章节与行号，PDF 引用保留页码并通过按需加载的 PDF.js 阅读。

```powershell
# 开发模式
npm --prefix web run build
.\.venv\Scripts\python.exe -X utf8 -m src.product_entry --no-browser

# Python 3.12 CPU 发布环境：构建 onedir 和 Inno Setup 安装程序
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_product.ps1
```

产品数据独立保存在 `%LOCALAPPDATA%\ObsidianRAG`；安装目录、开发仓库的 `.env`、旧索引和私人笔记不会被产品入口自动读取。生成仍支持本机 Ollama 与 DeepSeek，DeepSeek 密钥只保存到 Windows 凭据管理器。完整实现与故障记录见[产品阶段学习记录](docs/学习记录/README.md)。

0.2.1 安装包已在开发机完成安装、覆盖升级、数据保留、卸载保留和发布包自检验收（明细见 [0.2.1 发布说明](docs/RELEASE_NOTES_0.2.1.md)；`retrieval_model.py` 的模型准备路径自 0.2.0 起未变，真实 CPU E5 准备验收沿用 0.2.0 记录）。需要留意的是：**该安装包构建于 2026-09-15，不含 2026-09-16 的界面自适应、删除会话与应用内文件夹选择三项修复**；`dist/ObsidianRAG` 便携版已带这三项重打，是否升版本重打安装包见[改进记录第 23 节](改进记录.md)。阶段 33 的[朋友试用清单](docs/朋友试用与反馈.md)、[反馈台账](docs/试用反馈台账.md)和 GitHub Issue 模板已就绪；至少两位朋友的独立电脑试用仍是待办，不计为已完成。训练、QLoRA 和 adapter 保留为实验支线，不再决定产品主线排期。

阶段 34～36 新增三个入口，都按"默认关闭、关着时不产生任何副作用"来做：**来源分层**（每条依据标明来自笔记／记忆／网络；记忆系统挡在协议与空实现之后）、**联网补充**（默认关闭，且本期**没有接任何搜索后端**，所以打开开关也一个请求都不发）、**产出**（把资料整理成学习指南或思维导图）。产出这一块的重点不是"能生成"，而是给一个可核实的数字：`docs/` 语料 + 本机 `qwen2.5:7b` 实测**回链命中率 38.2%**（55 句陈述里 21 句带了来源，4 个主题中有 2 个交了 0%）。**这个数字偏低是结论，不是事故**——它量的是引用覆盖率、不是事实正确性，读之前请看[学习记录 36](docs/学习记录/36-产出与回链命中率.md) §四。这个缺口后来关上了：产出切换为**逐句 JSON**（模型输出逐句带来源编号的结构化 JSON，程序只换形状、不补链），产品链路真模型回归 **53/53 = 100%**，落地过程抓到的"文法约束下模型无限复读、必须配生成上限"见[学习记录 44](docs/学习记录/44-逐句JSON与失控兜底.md)。

阶段 37 的**图片产出**做成 Studio 的一个**导出器**，不是第三种产出类型——所以**没有数据库迁移**，也没有动产出列表。它把一份产出的来源分布、章节结构、回链命中率与导图结构画成一张 PNG，渲染由本机 Edge / Chrome 无头完成，**不联网、不调模型**：图里没有一个字是生成的，每个编号都指向一份真实片段。版面高度是**算出来的**（所有文本行不换行），所以出的图不裁切也不拖白边；真机实测 Edge **909 毫秒**出图 2160×2108。浏览器不可用时**不算故障**——HTML 照常导出，你可以自己打开或打印成图片。过程（含一个只有把 PNG 打开看才会发现的缺陷）见[学习记录 37](docs/学习记录/37-信息图与无头渲染.md)。

阶段 38 补上一个真实缺口：**产品此前在正常使用下没有退出入口**。它是个常驻进程，没有控制台窗口也没有托盘，而**关掉浏览器标签并不会让它退出**——所以侧栏现在常驻一个"退出工作台"，并把这句话直接写在按钮旁边。两处口径照旧守诚实：没有接启动器时接口回 **503 并说明原因**，而不是假装"正在退出"（与联网那块"未配置就说未配置"同一条线）；"退出不删数据"由接口自己在响应里签字。退出走的是正常路径而非被杀进程——验收验到浏览器外面：端口不再应答，且启动器日志里有 `server_stopped`。**心跳 + 闲置自动退出**是看过、想过、**决定不做**的（后台标签定时器被节流、长任务会被误杀、自动化脚本一个心跳都不发、关标签本就不损坏），四条理由写在[学习记录 38](docs/学习记录/38-退出入口与诚实响应.md) §六。

阶段 39 补上联网那一块剩下的口径缺口：**联网状态原来只活在一帧流式事件里**，答完重取一次会话就没了——于是"这次没联上"被读成"今天没什么可网的"。DB 升到 **v10**（`messages.web_state_json`），生成时与来源一起落库、读消息时带出来，界面在刷新之后仍然标注。关键判断是 **`NULL` 不等于 `off`**：v10 之前的旧回答、以及被能力守卫拦下（根本没走到联网那层）的回答，都留 `NULL`，不补默认值——**不替历史编结论**；这与 v8 给 `origin` 补 `'note'`（那是已知的历史事实）是故意相反的两个取舍。顺便删掉前端那张与消息重复的组件状态表。证据：真实数据根就地升级、真模型端到端、以及"把提示改挂错位置 → 界面验证挂 4 项"的反向对照，见[学习记录 39](docs/学习记录/39-联网状态入库.md)。

## 历史状态：RAG 工程与小模型训练学习（2026-09-14）

最新学习推进：在[独立挑战集与泛化差距](docs/学习记录/17-独立挑战集与泛化差距.md)之后，已完成 [SFT 目标、数据与训练前基线](docs/学习记录/18-SFT目标数据与训练前基线.md)、[QLoRA 训练与适配器评测](docs/学习记录/19-QLoRA训练与适配器评测.md)，并用[显式 adapter 入口接入真实 RAG](docs/学习记录/20-Adapter接入真实RAG与上下文故障.md)。固定 Qwen2.5-1.5B-Instruct commit，以 4-bit NF4 基座和 9,232,384 个 LoRA 参数在本机 8 GB GPU 完成 68 次更新。

同基座的已观察回归集从 3/15 提升到 8/15，训练前冻结技术 holdout 从 2/10 提升到 4/10；但两组各有 2 道“部分证据应拒答”题退化。adapter 已学到动态引用、冲突和部分多跳行为，也出现无依据补全，所以保留为实验产物，**没有自动替换默认 Ollama `qwen2.5:7b`**。

```powershell
# 配置与数据说明见阶段 18/19；产物位于被 Git 忽略的 data/generated
.\.venv\Scripts\python.exe -X utf8 -m src.train_qlora
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_hf_sft `
  --adapter data/generated/qlora_runs/20260914_200637_007529/best_adapter
# 显式实验：复用当前索引，不修改默认 Ollama 配置
.\.venv\Scripts\python.exe -X utf8 -m src.adapter_rag `
  "PagedAttention 是什么？它解决什么问题？" `
  --adapter data/generated/qlora_runs/20260914_200637_007529/best_adapter
```

QLoRA 完整进程耗时 82.8 秒，PyTorch 训练峰值分配约 5270 MiB，`nvidia-smi` 整卡采样峰值约 6860 MiB。模型权重、adapter 和含回答的报告均未纳入 Git；新机器需先按阶段 18/19 准备固定基座与训练数据。

此前的小型 Decoder 阶段仍保留：443904 参数模型的原选择验证 NLL 是 0.1291，但新句序、新格式、技术领域的已知字符 NLL 分别为 2.6046、2.8934、4.8987，说明模板内低损失没有转化成通用问答能力。

教学训练配置独立放在 [tiny_lm.toml](tiny_lm.toml)，没有用 Obsidian 笔记训练，没有替换 RAG 的 Qwen，也没有提交或推送 Git。本机可直接体验已有训练产物：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.train_tiny_lm --generate '.\data\generated\tiny_lm_runs\20260912_151041_274119\best.pt' --prompt '小周来到'
```

这是本机已生成、未纳入 Git 的检查点；新机器需先按阶段 15/16 准备数据和训练，不能仅克隆代码就假设权重存在。当前全量离线检查 **344 项通过**（2026-09-19 实测；另有前端 `tsc -b` 零错误、130 项 vitest、设计 token 门禁、57 项真实界面验收——其中"确认退出"那条要显式开启，见[学习记录 38](docs/学习记录/38-退出入口与诚实响应.md) §五；39 那次与 40 那次的界面证据各在[学习记录 39](docs/学习记录/39-联网状态入库.md) §八.2 与[学习记录 40](docs/学习记录/40-批量删除.md) §七）；SFT/QLoRA 与泛化入口均已在 CUDA 上实际运行并保存冻结报告，未调用远程生成 API。

前一阶段已完成 [CPU/GPU 部署实测](docs/学习记录/13-CPU与GPU部署实验.md) 和 [手写 Attention / KV Cache](docs/学习记录/14-手写Attention与KVCache.md)。部署报告将加载、生成与检索分开；张量实验验证缓存结果，并复现 mask 对齐和 view 存储问题。

```powershell
# 固定版本、独立设备组合实验，包含两次本地 Ollama 请求
.\.venv\Scripts\python.exe -X utf8 -m src.deployment_lab
# 不调用任何生成 API 的手写张量实验
.\.venv\Scripts\python.exe -X utf8 -m src.attention_lab --device cpu
.\.venv\Scripts\python.exe -X utf8 -m src.attention_lab --device cuda
```

最新已完成普通参数 TOML 配置、增量导入/向量复用、索引版本与回退、本机 HTTP 服务。

语料口径（2026-09-16 实测，此前文档里的"56 篇 / 911 片段"只算 Markdown，已过时）：

| 口径 | 规模 |
|---|---|
| 真实笔记目录 | **101 篇** = 56 Markdown + 44 Notebook + 1 PDF |
| 实验线语料 `data/generated/token_v1/chunks.json` | 52 个来源 / **822 片段** |
| 产品切分（800 字符）· 仅 Markdown | **980 片段** |
| 产品切分（800 字符）· 完整目录 | **2014 片段** |

旧快照与评测数据未覆盖。没有提交或推送 Git。

日常使用：编辑 [rag.toml](rag.toml)，然后在项目根目录执行：

```powershell
# 更新笔记快照；无变化时不加载编码模型
.\.venv\Scripts\python.exe -X utf8 -m src.index_store build
# 使用当前版本连续提问
.\.venv\Scripts\python.exe -X utf8 -m src.runtime
# 或启动本机 HTTP 服务（Ctrl+C 停止）
.\.venv\Scripts\python.exe -X utf8 -m src.serve
```

服务监听 `127.0.0.1:8000`；`POST /v1/answer` 接收 `{"question":"你的问题"}`，返回答案、来源和索引版本。完整启动/请求命令及 429/504 故障说明见 [HTTP 使用记录](docs/学习记录/12-本机HTTP服务与故障处理.md)。依赖见 [requirements-service.txt](requirements-service.txt)。本次真实 HTTP 合约检查 10/10 通过，不代表答案正确率 100%；临时服务已停止，没有后台自启动。

更新或激活版本后，已经运行的 CLI/HTTP 仍固定原版本；**重启才切换**。新增入口读取 TOML，下面旧实验入口仍保留原默认参数，不会隐式迁移。

### 上一阶段：证据层改进

已接入 E5 向量缓存、BM25/向量/RRF 混合检索切换、连续提问、实际上下文与检索轨迹报告。当前未提交或发布新版本；后文 v0.1 数字保留为历史记录，不代表最新运行。

最新一轮增加 Token 感知切分、本地 CrossEncoder 重排、完整块上下文与不调用生成模型的上下文解释入口。同一快照生成 822 个片段，E5 超限由 70 降为 0；三组真实生成仍均为 13/18，没有宣称总体质量全面提高。新功能显式启用，旧默认保留。[最新阶段验收](docs/学习记录/09-证据层阶段验收.md)

学习入口：[阶段学习记录](docs/学习记录/README.md)。里面按阶段记录改动、原理、遇到的问题、复现命令、真实结果、尚未解决的限制和参考练习。

### 在本机体验

```powershell
Set-Location -LiteralPath '<项目目录>'
.\.venv\Scripts\python.exe -X utf8 -m src.chat --provider ollama --retriever hybrid --chunks-file data/generated/token_v1/chunks.json --rerank --context-policy blocks
```

输入完整问题；`:method bm25`、`:method vector`、`:method hybrid` 切换，`:quit` 退出。问题之间不共享历史，暂不支持“它呢”这样的指代追问。

该命令要求已有 `data/generated/token_v1/chunks.json`、可用的 PyTorch 环境、E5 和重排模型的本地快照、运行中的 Ollama 和 qwen2.5:7b。本机这些前置条件已验证。依赖记录在 [requirements-vector.txt](requirements-vector.txt)，它不是完整环境锁；新机器先按平台配置 PyTorch，再安装该文件，不要覆盖当前可用的 CUDA 安装。

新机器缺少快照时可运行一次 `python -m src.vector_retrieve "RAG 是什么？" --allow-download`；未生成 chunks 时，先检查 `src/ingest.py` 中的笔记路径，再运行 `python -m src.ingest`。详细生命周期见[缓存说明](docs/学习记录/01-向量缓存与数据生命周期.md)。

要准备新版语料，再运行 `python -m src.token_chunking`（拒绝覆盖已有 token_v1）；重排模型准备命令为 `python -m src.reranker --allow-download`。省略 `--chunks-file`、`--rerank`、`--context-policy` 则仍走旧语料与旧上下文，不会隐式迁移数据。

### 常用命令

```powershell
# 查看两路排序和融合贡献
.\.venv\Scripts\python.exe -X utf8 -m src.hybrid_retrieve "PagedAttention 是什么？"
# 单次回答；省略 --retriever 则仍用 BM25
.\.venv\Scripts\python.exe -X utf8 -m src.answer --provider ollama --retriever hybrid "RAG 的基本流程是什么？"
# 固定题集比较检索，不调用生成模型
.\.venv\Scripts\python.exe -X utf8 -m src.compare_retrievers
# 新语料上对比三种检索器及其重排版本
.\.venv\Scripts\python.exe -X utf8 -m src.compare_retrievers --chunks-file data/generated/token_v1/chunks.json --include-rerank
# 仅查看实际上下文，不调用生成模型
.\.venv\Scripts\python.exe -X utf8 -m src.explain_context "PagedAttention 是什么？" --chunks-file data/generated/token_v1/chunks.json --rerank
# 真实生成评测；新报告使用时间戳，保护历史结果
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_generation --provider ollama --retriever hybrid
# 小规模离线回归检查，无需真实 LLM
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

BM25 仍是默认。上一轮旧语料的完整结果见[实验结果与取舍](docs/学习记录/04-实验结果与取舍.md)，本轮新语料与重排的完整结果见[证据层阶段验收](docs/学习记录/09-证据层阶段验收.md)。两轮都存在收益与退化并存的情况，不将一次小样本结果当成普遍结论。

索引和报告保存在被忽略的 `data/generated/`；报告可能含笔记原文，不应直接公开上传。DeepSeek 选项仍可用，但使用时问题与上下文会发送到远程 API，可能产生费用。本轮仅调用本地 Ollama。

## 学习目标

- 亲手实现 Markdown 解析、检索、上下文组织与引用回答。
- 通过小型评测集定位数据、检索、上下文和模型层的问题。
- 在实践中学习基础部署、日志与可靠性设计。

## v0.1 范围

- 只读取指定的 Markdown 笔记目录。
- 回答必须给出对应笔记来源。
- 当资料不足时，明确拒绝编造。
- 记录检索结果、回答、耗时和异常。

## 暂不实现

- 多 Agent、网页搜索、自动修改笔记、复杂前端、云端部署。


## 检索实验记录

### E01：关键词 + IDF + 查询预处理

- 语料：大模型笔记目录，共 757 个 Markdown 片段。
- Query 预处理：移除“什么是”“有哪些”“解决什么问题”等常见疑问表达。
- 排序方法：按匹配 token 的 IDF 分数排序，标题路径匹配额外加权。
- 开发集：30 条问题，其中知识库内问题 26 条、知识库外问题 4 条。

结果：

| 指标 | 结果 |
|---|---:|
| Source Recall@5 | 92.3% |
| Source Recall@1 | 50.0% |
| MRR | 0.670 |
| OOD Rejection Rate | 0.0% |

主要观察：

- 稀有技术词经过 IDF 加权后，PagedAttention、Prompt Injection 等问题的召回明显改善。
- 正确来源经常进入 Top-5，但不一定排在第一。
- 多概念问题容易被总览页、MOC 或索引型片段抢占第一名。
- 知识库外问题会被示例文本、同名工具说明等词面相似内容误召回。

### E02：BM25 + 查询预处理

- 语料、Chunking、开发集与 E01 相同。
- Query 预处理：保留 E01 的常见疑问表达清理。
- 排序方法：BM25，参数为 `k1 = 1.5`、`b = 0.75`。
- 标题路径在 BM25 索引中重复一次，作为轻量标题加权。

结果：

| 指标 | 结果 |
|---|---:|
| Source Recall@5 | 100.0% |
| Source Recall@1 | 80.8% |
| MRR | 0.879 |
| OOD Rejection Rate | 0.0% |

与 E01 对比：

- Source Recall@5：92.3% → 100.0%
- Source Recall@1：50.0% → 80.8%
- MRR：0.670 → 0.879
- OOD 拒答能力未改善。

主要观察：

- BM25 的词频饱和与长度归一化改善了长总览页、MOC 型片段的排序问题。
- 当前知识库内来源检索已达到可用水平。
- 后续重点从“召回正确资料”转向“知识库外拒答与回答依据判断”。

### E02-H：BM25 Holdout 基线

- 评测集：`data/eval_holdout.json`
- 总问题数：12
- 知识库内问题：8
- 知识库外问题：4

| 指标 | 结果 |
|---|---:|
| Source Recall@5 | 87.5% |
| Source Recall@1 | 75.0% |
| MRR | 0.792 |
| OOD Rejection Rate | 0.0% |

说明：

- Holdout 集只用于记录基线与最终验证。
- 后续不应根据 Holdout 中某一条问题调整规则或参数。

### E03：静态知识库能力守卫

- 能力边界：
  - 只读静态知识库；
  - 不提供实时数据；
  - 不执行外部操作；
  - 不假设用户未提供的个人资料。
- 拒答实现：基于实时性、外部动作、个人上下文短语的可解释规则。

开发集结果：

| 指标 | 结果 |
|---|---:|
| Source Recall@5 | 100.0% |
| Source Recall@1 | 80.8% |
| MRR | 0.879 |
| OOD Rejection Rate | 100.0% |

限制：

- 规则只覆盖当前定义的能力边界；
- 不等同于通用语义 OOD 检测；
- 后续需要结合检索置信度和生成阶段的依据检查。

### E03-H：静态知识库能力守卫 Holdout 验证

| 指标 | 结果 |
|---|---:|
| Source Recall@5 | 87.5% |
| Source Recall@1 | 75.0% |
| MRR | 0.792 |
| OOD Rejection Rate | 100.0% |

结论：

- 静态知识库守卫在 Holdout 的 4 条知识库外问题上均正确拒答。
- BM25 在 8 条 Holdout 知识库内问题中召回了 7 条正确来源。
- 失败问题属于多概念比较问题，表明后续需要 Multi-Query Retrieval，而不是继续为单一关键词排序调参。

### E04：Multi-Query Retrieval 对比实验

- 方法：仅对包含“分别”的列举型问题拆分为多个子查询。
- 合并方法：RRF（Reciprocal Rank Fusion）。
- 对比对象：E02 的 BM25 + 查询预处理。
- 评测集：开发集 `eval_set.json`。

结果：

| 指标 | BM25 | Multi-Query |
|---|---:|---:|
| Source Recall@5 | 100.0% | 100.0% |
| Source Recall@1 | 80.8% | 69.2% |
| MRR | 0.879 | 0.812 |

逐题对比：

- 改善题数：0
- 退化题数：4
- 排名不变题数：22

结论：

- 对当前结构化技术笔记，多个概念往往需要综合章节解释。
- 将问题机械拆分为子查询会破坏原问题的整体语义。
- 当前版本保留 BM25 作为默认检索方案；Multi-Query 仅保留为实验实现。

## v0.1 基线评测（2026-08）

本版本实现了一个面向 Obsidian Markdown 笔记的本地 RAG 问答系统，包含：

- 标题感知的 Markdown 切分与索引
- BM25 检索与中文/英文技术词处理
- 知识库外与实时问题拦截
- 基于 `[S1]`、`[S2]` 的来源引用
- Ollama 本地模型与 DeepSeek API 两种生成后端
- 检索评测、保留集评测与生成质量评测

### 生成评测结果

评测集共 18 题，覆盖单概念、多概念、综合问答、资料不足和知识库外问题。

| Provider | 通过率 | 引用有效率 | OOD 拒绝率 | 平均耗时 |
| --- | ---: | ---: | ---: | ---: |
| Ollama（qwen2.5:7b） | 88.9%（16/18） | 100.0% | 100.0% | 1726 ms |
| DeepSeek | 66.7%（12/18） | 78.6% | 100.0% | 2275 ms |

### 当前边界

- 对单概念和多概念的知识库问答，系统已有较稳定表现。
- 系统可以拒绝天气、实时价格、最新发布等依赖外部实时信息的问题。
- 综合型问题仍可能召回不足或无法组织出完整答案，例如“基础 RAG 的完整流程”和“如何判断系统是否变好”。
- 知识库未覆盖底层实现细节时，系统应明确说明资料不足，而不是补充无依据的内容。

这些结果是 v0.1 的基线，而不是最终能力上限。后续将围绕混合检索、重排序、知识库覆盖和更严格的生成评测继续迭代。

## 实验 QLoRA 常驻服务

训练后的 1.5B adapter 与默认 Ollama 服务保持隔离。本地 adapter 产物存在时，可启动固定索引、单进程的实验服务：

```powershell
.\.venv\Scripts\python.exe -m src.serve_adapter `
  --adapter .\data\generated\qlora_runs\20260914_200637_007529\best_adapter
```

另一个终端运行 `python -m src.adapter_service_probe`。该检查会拒绝连接非 adapter 后端，比较首次/热请求，并验证 OOD 请求没有触发模型。启动、延迟、显存与取消限制见[阶段 21](docs/学习记录/21-常驻Adapter服务与冷热路径.md)。它仍是 localhost 实验，不替换 `provider = 'ollama'`。

流式入口是 `POST /v1/answer/stream`，返回 `application/x-ndjson` 的 `metadata → token → final/cancelled` 事件。运行 `python -m src.adapter_stream_probe` 可验证客户端断开会真正停止模型、释放 GPU 席位并允许下一次恢复请求。协议、竞态修复和限制见[阶段 22](docs/学习记录/22-流式输出与真实取消.md)。

adapter 入口还支持 `--max-prompt-tokens`。系统使用实际 Qwen chat template 计数，超预算时重新选择完整证据块，不依赖 tokenizer 静默截断。运行 `python -m src.token_budget_lab --adapter <目录>` 可比较不同预算；详见[阶段 23](docs/学习记录/23-真实Token预算与证据阶梯.md)。

实验服务默认仍在 GPU 忙时立即返回 429。需要观察短突发时，可加 `--queue-size 2 --queue-wait-timeout 20`，形成 1 个 active 加 2 个 waiting 的有界 FIFO；成功响应会报告初始位置和排队耗时，`/health/ready` 会报告 scheduler 状态。运行 `python -m src.adapter_queue_probe` 可复现排队断线移除、FIFO 和队满背压；首轮真实失败与修复见[阶段 24](docs/学习记录/24-有界队列与背压.md)。

成功回答还会返回 `phase_timings_ms`，区分守卫、检索、证据打包、token 计数、生成、收尾、排队和接口总时间。`GET /v1/metrics` 聚合最近 100 条数字诊断，不保存问题、回答或证据正文。运行 `python -m src.observability_probe` 可验证计时层级、OOD 零生成和聚合口径；详见[阶段 25](docs/学习记录/25-阶段耗时与轻量监控.md)。

## Docker 本机基线

当前 Docker 镜像运行 CPU E5 检索与 RAG 服务，生成请求转给宿主机 Ollama；不会把 Vault、索引、`.env` 或模型权重装进镜像。先阅读 [container/README](container/README.md)，显式准备/核对 E5 缓存并构建容器专属索引，再启动服务。Compose 只发布宿主机 `127.0.0.1:8000`；最终镜像以非 root 用户运行。构建超时、Windows pyc 泄漏、最小源码、路径校验和真实问答见[阶段 26](docs/学习记录/26-容器边界与端到端运行.md)。

## 索引备份、恢复与供应链清单

索引备份含有笔记正文，只能放在受保护且被 Git 忽略的位置。创建和校验：

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffffff'
$backup = ".\data\generated\index-backup-$stamp.zip"
.\.venv\Scripts\python.exe -X utf8 -m src.index_backup create --store .\data\generated\index_store --output $backup
.\.venv\Scripts\python.exe -X utf8 -m src.index_backup verify $backup
```

恢复命令只接受不存在的新目录，不覆盖当前索引。`python -m src.recovery_drill --store .\data\generated\index_store --runtime-config .\rag.toml` 还会构造一个 CRC 正常但哈希错误的备份，验证拒绝后再恢复有效版本，并用恢复出的索引执行一次真实问答。

`python -m src.release_manifest --store .\data\generated\index_store` 生成不含私人路径、笔记文件名、正文或密钥值的供应链清单。Docker 可用时它还核对镜像身份、容器内包版本及镜像内源码哈希；daemon 不响应时在 20 秒后降级为明确的 unavailable，不会无限挂住。格式、真实结果、边界和练习见[阶段 27](docs/学习记录/27-供应链清单与索引恢复.md)。
