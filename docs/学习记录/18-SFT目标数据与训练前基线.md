# 阶段 18：把“微调”变成一个可以判断对错的 RAG 行为任务

日期：2026-09-14。接续[独立挑战集与泛化差距](17-独立挑战集与泛化差距.md)。本章记录已经完成的数据、基线和训练准备；QLoRA 的实际训练结果另记在下一章。

## 1. 这次到底要让模型学什么

目标不是把 Obsidian 笔记背进模型，而是让模型在**已经给定证据**时稳定遵守以下协议：

1. 证据充分：输出两行“结论/依据”，并引用真正支持结论的标签；
2. 问题有多个要求，但证据只覆盖一部分：整题拒答，不用外部知识补齐；
3. 证据互相冲突且没有优先级：指出冲突并引用冲突双方；
4. 需要两段关系才能作答：完成多跳组合，并同时引用两段证据；
5. 证据中出现“忽略系统提示”之类文字：把它当资料内容，不执行。

这是一项有限的行为微调。知识更新仍应通过检索和索引完成；LoRA adapter 不能替代 RAG 知识库。

## 2. 先做简单版，为什么反而暴露了问题

`src/sft_data.py` 先生成 v1：60 个互相隔离的事实主体，每个主体 3 道题，先按主体划分，再形成：

| split | 主体数 | 样本数 |
|---|---:|---:|
| train | 48 | 144 |
| validation | 6 | 18 |
| test | 6 | 18 |

每条记录保存逻辑 `system/user` messages 和单独的 assistant `response`，没有提前拼成某个模型专用字符串。Ollama `qwen2.5:7b` 在 v1 test 上未经训练就得到 **18/18**。

这不是“模型已经完美”，而是说明任务太容易：固定标签、固定句式和简单单事实检索不足以测出训练收益。这个报告仍保留，作为“评测饱和”的反例：

```text
data/generated/sft_baseline_qwen2.5_7b_test_20260914_191237_895223.json
SHA256 5d9d011628eddf4d7632d78f629d83fd061ed96c5ab5cb226141966952944858
```

## 3. 冻结更难的回归集

随后在 `data/rag_sft_challenge_v1.json` 冻结 15 题，每类 3 题。其 SHA256 为：

```text
997966f238bd1217786df5f8d82a45b79943ed82f6ce65810b8ca02a2c9161a2
```

`qwen2.5:7b` 的训练前结果为 **7/15（46.7%）**：

| 类别 | 通过 |
|---|---:|
| dynamic_citation | 1/3 |
| partial_evidence | 3/3 |
| conflict | 3/3 |
| multi_hop | 0/3 |
| evidence_injection | 0/3 |

具体失败很有代表性：

- 有充分证据时仍错误拒答；
- 答案事实正确，但把引用写成 `S3, S8`，没有遵守 `[S3][S8]`；
- 多跳答案只引用终点事实，漏掉中间关系；
- 把证据内的命令文字当成事实冲突，说明“内容”和“指令”的边界没有稳定学会。

报告保存在：

```text
data/generated/sft_challenge_qwen2.5_7b_20260914_191552_499227.json
SHA256 0d7e631a52b479639c36b17bbbc62ce4130962b3ff6657c549ae6d37c465df82
```

## 4. v2 训练数据怎样对应真实失败

`src/sft_data_v2.py` 生成 60 个新的合成事实主体，每个主体覆盖上述 5 种行为，共 300 条：

| split | 主体数 | 样本数 | 每类样本数 |
|---|---:|---:|---:|
| train | 54 | 270 | 54 |
| validation | 6 | 30 | 6 |

设计要点：

- 主体先分组，再进入 train/validation，避免同一设备事实跨 split；
- 标签位置会变化，不能靠“总引用 S1”投机；
- 多跳目标明确要求两条引用；
- 证据内指令使用不同措辞，训练的是边界规则而不是一个固定坏句子；
- 验证集只用于选择 epoch，不进入反向传播。

v2 没有复制那 15 道挑战题，但必须诚实承认：我们已经观察过挑战集失败，并据此选择了五种训练类别。因此这 15 题应称为**冻结回归集**，不是完全盲测。

为修正这个实验设计问题，QLoRA 开始前又冻结了 `data/rag_sft_holdout_v1.json`：10 道技术领域题，每类 2 道，SHA256 为 `8513f2ef86554713d4e657e28afcb26e51e606745a434ffd0b88a022640f950c`。它使用 PagedAttention、KV Cache、QLoRA 和验证集等主题，内容由题内证据自洽支持；后续不根据它的结果修改本轮数据、prompt、评分器或超参数。这样才能把“修复已知失败”和“迁移到未训练事实/措辞”分开看。

## 5. 为什么先存 messages，再应用 chat template

不同 instruct 模型对对话边界的特殊 token 约定不同。数据如果提前写死成某一种字符串，换模型时容易出现：

- 重复添加 BOS/EOS；
- assistant 开始标记错位；
- 把 system/user 内容也当成回答目标；
- 推理时模板与训练时模板不一致。

当前流程是：

```text
逻辑 messages + assistant response
          │
          ▼  Qwen tokenizer.apply_chat_template
完整 token 序列
          │
          ├─ system/user/assistant header → label = -100
          └─ assistant 正文与结束标记   → label = token id
```

`-100` 是 PyTorch 交叉熵默认的 ignore index。模型仍会读取 prompt token 作为条件，但它们不直接贡献本次 loss；这就是 response-only SFT。

实际 token 化结果：

| split | 总长度 | prompt 长度 | 回答监督长度 | 监督 token 占全部 token |
|---|---:|---:|---:|---:|
| train | 191～211 | 174～192 | 13～29 | 10.86% |
| validation | 191～211 | 174～192 | 13～29 | 10.87% |

这也解释了为什么“序列有约 200 tokens”不等于“每条样本有 200 个训练目标”。梯度累积、loss 平均和验证 NLL 都必须围绕真正的监督 token 理解。

## 6. 实际遇到的兼容问题

### Transformers 5.x 的返回值变化

原实现假设 `apply_chat_template(tokenize=True)` 返回 token ID 列表；本机 Transformers 5.15.1 返回了 `BatchEncoding`，直接切片会失败。

修复不是硬编码某个版本分支，而是统一提取 `input_ids`，再把 tensor、二维单样本或普通列表归一成一维 ID 列表。随后显式检查“完整训练序列是否以 generation prompt 为前缀”，避免模板错位后继续静默训练。

### 超长样本不静默截断

当前最大长度配置为 512，而实际最长为 211。若未来样本超过 512，代码直接报错；它不会悄悄截掉问题、证据或目标答案，因为这种截断可能让标签与证据不再对应。

### Windows 下载链路

固定 commit 的 tokenizer 小文件已正常缓存。首次下载约 3.09 GB 的模型权重时，Python Hub 下载器在大文件 CDN 响应体上发生 read timeout，`.incomplete` 文件几乎不增长；检查 URL 后确认 CDN 可访问，改用 `curl -L -C -` 的断点续传、低速超时和自动重试下载到同一缓存。问题属于下载传输，不是 CUDA、量化或模型结构错误。

### 4-bit 环境不能只看“import 成功”

本机实际安装并验证：

```text
torch 2.13.0+cu132
transformers 5.15.1
peft 0.20.0
accelerate 1.15.0
bitsandbytes 0.50.2
GPU NVIDIA GeForce RTX 5060 Laptop GPU
```

预检创建了真实 `Linear4bit(NF4)` 层，在 CUDA 上完成 forward 和 backward，且输入梯度有限。`triton not found` 只使 FLOP 统计不可用，不代表 bitsandbytes 计算失败。

## 7. 为什么选 Qwen2.5-1.5B-Instruct

固定模型为：

```text
Qwen/Qwen2.5-1.5B-Instruct
commit 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
license apache-2.0（模型卡标注）
```

它比本地 Ollama 的 7B 小，主要是为了在 8 GB 显存上同时容纳 4-bit 基座、LoRA 参数、梯度、优化器状态和短序列激活。7B Ollama 的 46.7% 只能说明任务难度，不能作为 1.5B adapter 的严格前测；公平比较必须是同一个 1.5B commit、同一种 4-bit 加载方式，在 adapter 前后各跑一次。

Ollama 中的量化推理模型也不能直接当成当前 Hugging Face/PEFT 训练基座。推理打包格式、量化实现、训练梯度和 adapter 保存方式是不同层的问题。

## 8. 复现入口

```powershell
Set-Location -LiteralPath '<项目目录>'

# 重新生成数据时必须换输出目录，程序拒绝覆盖旧版本
.\.venv\Scripts\python.exe -X utf8 -m src.sft_data_v2 --output data/generated/rag_sft_v2_rebuild

# 检查实际 chat template、长度和 response-only 标签
.\.venv\Scripts\python.exe -X utf8 -m src.sft_tokenization

# 同一 1.5B 基座的训练前回归集与 holdout（权重已缓存后无需 --allow-download）
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_hf_sft
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_hf_sft --challenge data/rag_sft_holdout_v1.json

# QLoRA 配置集中在 qlora.toml；训练结果写入被 Git 忽略的 data/generated
.\.venv\Scripts\python.exe -X utf8 -m src.train_qlora
```

## 9. 本阶段没有证明什么

- 没有证明合成设备数据能提高所有真实 RAG 问答；
- 没有把知识库内容训练进模型；
- 没有证明 1.5B 一定优于 7B，也没有做模型规模公平比较；
- 没有把验证集、已观察回归集和未调参 holdout 混成一个分数；
- 当前 adapter 保存不含优化器状态，不能伪装成可恢复的完整训练检查点；
- 固定随机种子不保证 bitsandbytes/CUDA 跨机器逐位一致。

## 10. 练习与参考答案

### 练习 1

为什么输入证据也很重要，却把对应 label 设为 `-100`？

参考答案：输入 token 仍参与前向计算，决定 assistant token 的条件分布；`-100` 只表示不要求模型在这些位置复现 prompt。当前目标是学习“看到证据后怎样回答”，不是继续训练模型照抄 system/user 文本。

### 练习 2

如果 270 条训练样本随机切行，而不是先按设备主体分组，会发生什么？

参考答案：同一设备的负责人、位置或模式可能同时出现在训练和验证中。验证分数会混入事实记忆收益，无法区分模型学会了行为规则，还是见过同一主体。

### 练习 3

为什么回归集从 7/15 提升到很高仍不足以证明泛化？

参考答案：它的失败模式已经影响了 v2 类别设计，属于开发反馈。它适合防止已知问题回归；训练前另行冻结、之后不调参的技术 holdout 才更接近本轮独立测试。

### 练习 4

QLoRA 的 4-bit 指什么？LoRA 参数也必须以 4-bit 训练吗？

参考答案：这里主要指冻结基座权重以 4-bit NF4 形式加载和参与计算；新加的低秩 adapter 需要可训练梯度，并不按同样的 4-bit 冻结权重方式训练。量化基座节省了最大头的权重显存，但激活、梯度和优化器状态仍要占显存。

### 练习 5

为什么必须测“同一个 1.5B 基座的 adapter 前后”，而不能只比较 Ollama 7B 与微调后 1.5B？

参考答案：两者同时改变了参数规模、权重版本、量化/运行时和是否微调，结果无法归因。控制其他变量后，前后差值才主要反映 adapter 和本轮训练数据的影响。
