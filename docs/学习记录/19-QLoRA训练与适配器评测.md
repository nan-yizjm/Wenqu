# 阶段 19：在 8 GB GPU 上完成 QLoRA，并保留它的回归

日期：2026-09-14。接续[SFT 目标、数据与训练前基线](18-SFT目标数据与训练前基线.md)。本章记录已经实际运行的 QLoRA，不是安装命令或预期结果。

## 1. 先看最终结论

固定的 `Qwen/Qwen2.5-1.5B-Instruct` 基座在 4-bit NF4 下完成 2 个 epoch、68 次优化器更新。训练没有 OOM，最佳 adapter 为 epoch 2。

| 指标 | 训练前 | 最佳 adapter | 变化 |
|---|---:|---:|---:|
| 合成验证集 response NLL | 0.585474 | 0.000054 | -0.585420 |
| 已观察冻结回归集 | 3/15（20.0%） | 8/15（53.3%） | +33.3 pct |
| 训练前冻结技术 holdout | 2/10（20.0%） | 4/10（40.0%） | +20.0 pct |

总体提升是真实的，但 adapter **还不适合替换默认 RAG 模型**：两组评测各有 2 道原本正确的“部分证据拒答”发生退化，模型开始只回答有证据的一半，甚至补出不存在的内容。

## 2. QLoRA 实际加载和训练了什么

配置位于 `qlora.toml`：

```text
base              Qwen2.5-1.5B-Instruct，固定 commit
base load         4-bit NF4 + double quantization
compute dtype     BF16
LoRA targets      all-linear
r / alpha         8 / 16
dropout           0.05
micro-batch       2
gradient accum.   4
nominal batch     8 examples / optimizer update
epochs            2
learning rate     2e-4，5 步 warmup，cosine 降至 2e-5
```

可以把一次训练理解成：

```text
冻结 4-bit 基座权重
        │
每个目标线性层旁加入 A、B 两个低秩矩阵
        │
前向：基座输出 + 缩放后的 B(A(x))
        │
response-only loss
        │
只为 LoRA 参数计算/更新梯度
```

实际可训练参数为 **9,232,384**。运行时枚举到的参数元素总数为 897,848,832，可训练占比显示为 1.028%。这里不能误读：bitsandbytes 会打包 4-bit 权重，量化参数的 `numel()`/存储布局不再等同于模型卡所说的约 1.54B 原始参数，也不等于实际显存字节数。报告因此把字段命名为 `total_parameter_elements_reported`，没有把它写成模型真实参数规模。

## 3. 为什么是 68 次更新，而不是 270 × 2

训练集 270 条，micro-batch 为 2：

```text
每个 epoch 的 micro-batch 数 = ceil(270 / 2) = 135
每 4 个 micro-batch 更新一次 = ceil(135 / 4) = 34 updates
2 epochs = 68 updates
```

最后一组只有 3 个 micro-batch，即 6 条样本。代码按这一组真实大小除 loss；如果仍机械除以 4，最后一次更新的梯度会被额外缩小 25%。

学习率也按 **optimizer update** 计数，而不是按每次 forward 计数。梯度累积改变了 forward 次数与更新次数的关系，这是自己写训练循环时很常见的错误源。

## 4. response-only loss 怎样进入模型

每个 batch 有三个张量：

- `input_ids`：system、user、assistant header、assistant answer 的完整 token；
- `attention_mask`：真实 token 为 1，右侧 padding 为 0；
- `labels`：prompt 与 padding 为 `-100`，assistant 正文和结束标记为真实 token ID。

Hugging Face causal LM 在内部做 next-token shift，并忽略 `-100`。因此验证 NLL 的分母必须是真正受监督的 response token 数，而不是样本数、序列总长度或 batch 数。当前 validation 共有 660 个监督 token。

## 5. smoke run 先发现了什么

正式训练前运行 2 次更新：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.train_qlora --max-updates 2
```

结果被明确标记为 `role=smoke`、`complete=false`，没有混入正式实验：

| 项目 | 结果 |
|---|---:|
| 初始 validation NLL | 0.585474 |
| 2 updates 后 validation NLL | 0.457589 |
| PyTorch 训练峰值分配 | 5075 MiB |
| `nvidia-smi` 整卡采样峰值 | 6763 MiB |
| 完整进程耗时 | 12.3 s |

smoke 还触发了一个 PyTorch 警告：日志用 `float(raw_loss)` 直接转换带梯度 tensor。它不会改变已完成的反向传播，但容易形成不必要的同步，也可能让人误以为日志值仍在计算图内。正式训练前改为 `float(raw_loss.detach())`。

这个小问题说明 smoke 的价值不只是“看会不会 OOM”：它能在完整实验前暴露真实模型、真实量化层、真实保存路径才会触发的问题。

## 6. 正式训练曲线与资源

运行命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.train_qlora
```

| 阶段 | train response NLL | validation NLL | 累计 updates | 耗时 |
|---|---:|---:|---:|---:|
| 初始 | - | 0.585474 | 0 | - |
| epoch 1 | 0.076894 | 0.000542 | 34 | 36.1 s |
| epoch 2 | 0.000260 | 0.000054 | 68 | 36.3 s |

完整进程耗时 82.8 秒。梯度裁剪前范数在 epoch 1 的最大值为 15.489，而阈值为 1.0；这证明裁剪确实被触发，不是只写了一个从未生效的配置。

显存口径：

| 口径 | 结果 |
|---|---:|
| 模型加载后 PyTorch allocated | 1581 MiB |
| 训练阶段 PyTorch peak allocated | 5270 MiB |
| 训练阶段 PyTorch reserved（结束时） | 5942 MiB |
| `nvidia-smi` 离散采样整卡峰值 | 6860 MiB / 8151 MiB |

PyTorch 数字只描述当前进程的 allocator；`nvidia-smi` 还包含 CUDA context、驱动、桌面和其他进程。离散采样也可能漏过极短峰值。两组数字用途不同，不能互相覆盖。

正式报告：

```text
data/generated/qlora_runs/20260914_200637_007529/summary.json
SHA256 c44de6681e4e131bb57c4ed2ce7c454cdd0336d28eaf54e7eb10aa78dd56de37
```

最佳 adapter：

```text
adapter_config.json       1,207 bytes
SHA256 ca2bac8c23e541b54ad5605e90d3c2b60f66ba4b7994bbb864da14086b7f60ad

adapter_model.safetensors 36,981,072 bytes
SHA256 6aa483bc5cc01be7d35ff2e9461e79bb9b7b8d96beb8c1fbb1d18b3f4e48fef9
```

adapter 只有约 35.3 MiB，但推理时仍需相同基座。它不是一个可以单独回答问题的完整模型。

## 7. 受控比较：哪些题修好了，哪些退化了

比较器会拒绝模型名、commit、量化方式、题集 SHA、评估器 SHA 或生成参数不同的两份报告。前后都满足同一条件后，结果如下。

### 已观察冻结回归集

| 类别 | 基座 | adapter |
|---|---:|---:|
| dynamic_citation | 0/3 | 2/3 |
| partial_evidence | 3/3 | 1/3 |
| conflict | 0/3 | 2/3 |
| multi_hop | 0/3 | 2/3 |
| evidence_injection | 0/3 | 1/3 |
| 总计 | 3/15 | 8/15 |

逐题变化：修复 7、退化 2、持续通过 1、持续失败 5。

### 训练前冻结的技术 holdout

| 类别 | 基座 | adapter |
|---|---:|---:|
| dynamic_citation | 0/2 | 0/2 |
| partial_evidence | 2/2 | 0/2 |
| conflict | 0/2 | 2/2 |
| multi_hop | 0/2 | 1/2 |
| evidence_injection | 0/2 | 1/2 |
| 总计 | 2/10 | 4/10 |

逐题变化：修复 4、退化 2、持续通过 0、持续失败 4。

比较报告：

```text
data/generated/sft_adapter_comparison_20260914_200935_786182.json  # 回归集
data/generated/sft_adapter_comparison_20260914_200935_786181.json  # holdout
```

## 8. 失败输出告诉了我们什么

### 真实回归：模型学成了“尽量回答”

部分证据题问“模式 + 周期”，只给周期；adapter 输出：

```text
结论：设备B202使用月度模式，多久检查一次：10天。
依据：[S3]
```

“月度模式”没有证据。另一道技术 holdout 只说 LoRA 适配所有线性层，未给 rank，adapter 却声称 `r=1`。这两条属于真正的无依据生成，不能用总分提升掩盖。

可能原因不是“拒答样本数量少”：五类训练样本数量相同。更可能的结构问题包括：

- 训练中的 partial 问法和未知字段类型仍太单一；
- 大量 answer/conflict 样本强化了套格式作答，模型在领域措辞变化时倾向完成答案；
- validation 与 train 来自相同生成模板，接近零的 NLL 无法暴露这种迁移失败。

本轮不根据已经看过的 holdout 继续修改 v2 并重跑。如果要做 v3，应先定义新的训练变化，并再冻结一份未用于调参的测试切片。

### 引用正确性不只是“标签存在”

holdout 中有回答引用 `[S9][S6]`，但真正支持 4-bit QLoRA 的只有 `[S9]`；`[S6]` 只是描述全参数微调。所有标签都存在，仍然不是精确证据引用。当前评估同时检查可用标签和要求标签顺序，正是为了区分这两类问题。

### 多跳仍会漏中间来源

模型能得到“21 天”的最终答案，却只引用周期来源，漏掉“设备由某团队维护”的关系来源。事实字符串正确，不等于推理链有完整证据。

### 证据内指令只部分改善

一部分样本已能忽略假命令并引用正常事实；另一部分仍把命令文字和事实当成冲突，甚至输出命令指定的错误负责人。adapter 降低了失败率，但没有构成可靠的 prompt injection 防线。

### 评分器也有边界

回归集中有一题正确输出“检查频率记录不一致”，参考要求词是“检查周期”，因此 `must_mention` 失败。这是词面评分的假阴性。因为题集和评分器已经冻结，本轮保留原结果，不在看到输出后放宽规则。未来可在新评测版本加入等价词组或模型裁判，但必须保留旧报告可解释性。

## 9. adapter 保存不等于训练检查点

当前目录保存：

- epoch 1 adapter；
- epoch 2 adapter；
- `best_adapter`；
- `final_adapter`；
- `summary.json`。

它没有保存 AdamW 一阶/二阶矩、当前梯度累积位置或 DataLoader 状态，所以不能从 adapter 文件恢复成“与连续训练相同”的后半程。阶段 16 已用小模型验证完整恢复需要哪些状态；本阶段刻意聚焦 k-bit 训练、LoRA 和评测，不重复伪造一个不完整 resume。

## 10. 复现和查看结果

```powershell
Set-Location -LiteralPath '<项目目录>'

# 训练前/后回归集
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_hf_sft
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_hf_sft `
  --adapter data/generated/qlora_runs/20260914_200637_007529/best_adapter

# 训练前/后技术 holdout
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_hf_sft `
  --challenge data/rag_sft_holdout_v1.json
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate_hf_sft `
  --challenge data/rag_sft_holdout_v1.json `
  --adapter data/generated/qlora_runs/20260914_200637_007529/best_adapter

# 严格检查是不是同条件，再看逐题变化
.\.venv\Scripts\python.exe -X utf8 -m src.compare_sft_adapters `
  data/generated/hf_sft_base_challenge_20260914_200443_779631.json `
  data/generated/hf_sft_adapter_challenge_20260914_200849_311457.json
```

这些报告和 adapter 位于 `.gitignore` 覆盖的 `data/generated/`。它们没有自动上传到 GitHub；代码、配置和学习记录可审阅后再由用户决定是否提交。

## 11. 下一步取舍

本轮先不替换 `qwen2.5:7b`，原因不是 adapter 没有学到东西，而是它同时引入了“证据不完整仍作答”的危险回归。合理后续是：

1. 提供显式的本地 adapter 实验入口，与默认 Ollama 路径隔离；
2. 对真实检索上下文做少量人工审阅，检查合成行为能否迁移；
3. 若做数据 v3，扩大“缺哪个字段”的表达和领域范围，并新建未观察测试，不能继续调当前 holdout；
4. 再进入流式输出、取消、队列/批处理、容器与监控等部署问题。

## 12. 练习与参考答案

### 练习 1

QLoRA 已把验证 NLL 降到接近 0，为什么 holdout 仍只有 40%？

参考答案：验证集与训练集来自同一生成器和模板，只隔离了事实主体；模型可以非常准确地学会模板内映射。holdout 同时换成技术领域、不同未知字段和新措辞，测的是行为迁移。低同分布验证 loss 不等于域外可靠性。

### 练习 2

micro-batch 2、累积 4 是否总等于 batch 8？

参考答案：大多数更新等于 8 条样本，但每个 epoch 有 135 个 micro-batch，最后一组只有 3 个，即 6 条样本。代码必须按最后组的真实数量缩放 loss。

### 练习 3

为什么 36.98 MB 的 adapter 推理时仍需要约 3 GB 权重文件？

参考答案：adapter 只保存低秩增量，不包含冻结基座。前向需要基座线性变换与 LoRA 增量共同计算，因此必须加载匹配模型名和 commit 的基座。

### 练习 4

`nvidia-smi` 6860 MiB 和 PyTorch peak allocated 5270 MiB 哪个是错的？

参考答案：都不一定错。前者是整卡视角，包含非 PyTorch/非张量开销及其他进程；后者是当前进程 allocator 统计。两者测量边界不同。

### 练习 5

下一轮能否直接把当前 holdout 失败题改写成训练样本，再用同一 holdout 宣称提升？

参考答案：可以把失败作为开发反馈设计新数据，但此后当前 holdout 已参与调参，只能降级为回归集。要声称新的独立泛化结果，必须在训练变化确定后再冻结一份未用于调整的新测试集。
