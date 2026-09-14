# 阶段 14：亲手实现 Attention，并证明 KV Cache 没算错

日期：2026-09-12。实现：`src/attention_lab.py`。这是一个可以逐行阅读的小张量实验，不是下载新的大模型，也不是训练好的 Chatbot。

## 1. 本阶段到底做了什么

- 用 `nn.Linear`、矩阵乘法、mask 和 softmax 写一个因果多头自注意力层。
- 写“每步重算整个前缀”和“保留 K/V，每步只投影新输入”两条推理路径。
- 用相同权重和输入，比较两条路径与一次完整计算的结果。
- 与 PyTorch 的 SDPA 实现交叉核对。
- 复现矩形 causal mask 的错误，并修复单 token view 留住整个 QKV 存储的问题。
- 在 CPU 和 GPU 上运行，保存形状、实际工作量、误差、缓存字节数与计时。

没有实现 tokenizer、位置编码、完整 Transformer、语言模型 head 或训练。输入是固定随机张量，输出是隐状态向量，不会生成自然语言。这里的“decode 32 步”是固定下一步输入，排除随机采样导致输入不同的干扰。

## 2. 先运行，然后带着输出看代码

```powershell
Set-Location -LiteralPath '<项目目录>'
.\.venv\Scripts\python.exe -X utf8 -m src.attention_lab --device cpu
.\.venv\Scripts\python.exe -X utf8 -m src.attention_lab --device cuda
```

默认：batch=1，前缀 64 个位置，继续输入 32 个位置，d_model=128，4 个头，每头 32 维。每组计时先热身，再重复 5 次。

学习顺序建议：`project → causal_mask → attend → forward → decode_fixed_inputs`。不要一开始就读保存报告和命令行解析代码。

## 3. 从输入形状到 Attention

以一次处理 64 个位置为例：

| 对象 | 形状 | 意义 |
|---|---|---|
| X | `[1, 64, 128]` | batch、位置数、隐藏维度 |
| 合并的 QKV 投影 | `[1, 64, 384]` | 同一次线性层输出三组向量 |
| Q、K、V 各自 | `[1, 4, 64, 32]` | 分成 4 个 head |
| QKᵀ | `[1, 4, 64, 64]` | 每个 query 对各 key 的打分 |
| Attention 输出 | `[1, 4, 64, 32]` | 对 V 加权求和 |
| 合并 head 后 | `[1, 64, 128]` | 再经过输出线性层 |

核心计算：

```python
scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
scores = scores.masked_fill(~allowed, float('-inf'))
weights = torch.softmax(scores, dim=-1)
attended = weights @ value
```

这里 `nn.Linear(128, 384)` 的权重张量形状是 `[384, 128]`，相当于输入右乘权重转置。缩放用 `sqrt(head_dim)`，本例是 `sqrt(32)`，不是 `sqrt(128)`。softmax 的最后一个维度是 key 位置，不是 head 维度。

本实现 `allowed=True` 表示可以关注。PyTorch 不同注意力接口的布尔 mask 约定并不完全相同，不能把 padding mask 原封不动搬到这里。代码与本机安装的 torch 2.13.0 文档字符串核对过；在线 [SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) 也说明了掩码语义和非方形情况下的对齐方式，在线版本可能比本机更新。

## 4. 为什么需要因果 mask

当第 i 个位置预测后续内容时，不应读取未来位置。矩阵中允许 `key_position <= query_position`，其他位置置为负无穷，经 softmax 后权重变成 0。

验证方法不是只看三角形长得像不像：实际修改后半段输入，确认前半段输出不变。这个“未来输入不能改变过去输出”的检查，比只验证最终张量形状更有意义。

注意本例没有 padding；所有输入都是真实位置，也不会出现某个 query 整行全部被 mask 的情况。若后续添加 padding，需要重新考虑全 mask 行、数值稳定和 batch 内不同长度。

## 5. KV Cache 保存什么，为什么不保存旧 Q

先处理 64 个位置时，一次性得到每个位置的 K/V，保存：

```text
K cache: [1, 4, 64, 32]
V cache: [1, 4, 64, 32]
```

第 65 个位置到来时：

1. 只对这个新位置计算 Q_new、K_new、V_new。
2. 把 K_new、V_new 接到缓存后面，长度从 64 增到 65。
3. 计算 `Q_new @ K_allᵀ`，分数形状是 `[1, 4, 1, 65]`。
4. 对全部 V 加权，得到最新位置的输出。

本次只需要新位置的输出，旧 Q 不参与新位置的计算。因果注意力保证旧位置不会因为后来的输入而改变，因此可以不重算旧位置。这个推理依赖权重固定、前缀一致、位置处理正确和推理模式；不是任意序列模型都能这样缓存。

**仍然要读取历史 K/V**。缓存减少重复投影和旧位置计算，不代表生成第 1000 个 token 与第 10 个 token 的注意力工作相同。历史长度增长，缓存存储和新 Q 对历史 K/V 的读取也会增长。

## 6. 常见错误一：decode 不是重新从第 0 个位置开始

最容易写错的是：

```python
allowed = torch.ones(query_length, key_length, dtype=torch.bool).tril()
```

完整 prefill 时 Q/K 长度相等，这往往看起来正常。但 decode 时 Q 长度是 1、K 长度是 65；从左上角做 tril，只剩第一个 key 可见。

正确做法使用绝对位置：

```python
query_positions = past_length + torch.arange(query_length)
key_positions = torch.arange(key_length)
allowed = key_positions[None, :] <= query_positions[:, None]
```

实际复现：本应看到 65 个 key，错误 mask 只看到 1 个，注意力输出最大误差约 **1.55（CPU）/1.59（GPU）**。不是浮点误差，而是语义完全错了。

多 token 追加也要正确。例如已有 3 个位置，再输入 2 个位置，mask 应是：

```text
1 1 1 1 0
1 1 1 1 1
```

代码同时验证了单 token 和多 token 追加。不要在矩形 Q/K 上不加思考地用 `is_causal=True`；它具体如何对齐要看接口约定。对这里的缓存场景，显式位置 mask 更容易理解和核对。

## 7. 常见错误二：小 view 会留住大存储

初版认为 `.contiguous()` 能保证缓存只拥有自己的 K/V 数据。单 token 场景实际检查失败：

```text
单个 K 的逻辑大小：128 bytes
K 引用的底层 storage：384 bytes
```

原因：QKV 是合并投影，K/V 是其中的切片。单 token 时切片可能已经连续，`contiguous()` 可以直接返回原 view，仍引用包含 Q 的整块存储。

修复：**首次建立缓存时显式 clone K/V**。后续 `torch.cat` 得到新的存储，不继续依赖原合并 QKV。

注意 K 和 V 可能引用同一块底层存储，不能把两个 384 简单相加当作实际占用。逻辑张量大小、底层 storage 和 CUDA allocator 分配量也不是同一个口径。这个问题说明“shape 很小”不必然表示“持有的内存很小”。

对应检查：`test_single_token_cache_does_not_retain_fused_qkv_storage`。先实际复现失败，再修复，通过后才保存最终实验报告；旧实验报告保留，没有覆盖成新数字。

## 8. 数值一致性与真实工作量

最终报告：

- `data/generated/attention_cpu_20260912_144853_841033.json`
- `data/generated/attention_cuda_20260912_144856_745162.json`

同设备、相同权重和相同输入下：

| 检查 | CPU 最大绝对误差 | GPU 最大绝对误差 |
|---|---:|---:|
| 完整计算 vs 带缓存逐步计算 | 8.20e-8 | 4.47e-8 |
| 手写 Attention vs PyTorch SDPA | 1.34e-7 | 1.51e-7 |

两种设备分别验证通过。CPU 和 CUDA 的随机数生成不保证产生逐元素相同输入，所以这里不是跨设备同一随机张量的误差比较。浮点求和顺序、底层 kernel 不同，也不应该要求所有结果 bitwise 相同。

本例记录 QKV 实际处理的位置数：

```text
完整前缀重算：64 + 65 + ... + 96 = 2640
带缓存：64 + 32 = 96
```

本例 batch=1。batch 增大时计数相应增加；这不是权重参数量或唯一 token 的数量，而是包含重复工作的“被投影的位置数”。

手写稠密 score 张量元素累计：856768 → 26688，包含 mask 前计算的元素。这不是精确 GPU FLOP 计数，也不是现代融合 attention kernel 的内存分配预测。

## 9. 缓存会占多少空间

单层、普通多头注意力的逻辑 K/V 字节数：

```text
2 × batch × KV_heads × cached_length × head_dim × bytes_per_element
```

本例 FP32：`2 × 1 × 4 × 96 × 32 × 4 = 98304 bytes = 96 KiB`，与真实 K/V tensor 的元素计数一致。多层 Transformer 还要对各层求和；同规格时再乘层数。GQA/MQA 要使用 KV head 数，而不是直接使用 query head 数。

它不包含模型权重、Attention 中间结果、输出、allocator 预留或 CUDA context。因此不能把这个公式算出的数字当成整个模型的显存需求。

本例每步用 `torch.cat` 追加，会分配新缓存并复制旧缓存。它适合理解语义，不是生产级实现；预分配缓存、分块缓存、分页管理就是后续工程优化的切入点。

## 10. 速度：减少计算，不等于按计数比例加速

最终热计时中位数，prefix=64，decode=32：

| 设备 | 前缀重算 | 带缓存 | 重算/缓存时间比 |
|---|---:|---:|---:|
| CPU，4 线程 | 4.893 ms | 2.153 ms | 2.27 |
| GPU | 7.867 ms | 6.618 ms | 1.19 |

QKV 位置数减少了 27.5 倍，实际时间没有相应减少 27.5 倍；这里还有 Python 循环、mask 创建、动态分配、缓存复制和 kernel 启动等工作。

这个小张量下 CPU 比 GPU 更快的观察，与小工作量难以摊薄 GPU 启动/同步开销相符；尚未用 profiler 分离每一项原因，不能说已经证明某一项占了多少百分比，更不能推广为大模型应该在 CPU 上运行。

GPU 计时在边界同步，计入宿主调度和同步，不只是 kernel 时间。本例记录的额外 CUDA 分配峰值约为重算 582144 bytes、缓存 266240 bytes，属于这段特定基准的临时分配观察，不是上面的缓存公式，也不包含 Ollama 服务占用。

## 11. 与当前 RAG 项目有什么关系

| 概念 | 缓存/保存的东西 | 主要避免什么 |
|---|---|---|
| 文档向量 NPZ | 笔记的 embedding | 每次启动重新编码全部笔记 |
| 模型权重驻留 | 模型参数 | 每次请求重新加载模型 |
| KV Cache | 每层已处理 token 的 K/V | decode 重复计算旧前缀 |
| 跨请求前缀复用 | 兼容的公共前缀中间状态 | 对重复前缀再次 prefill |

这四者不要混成一个“缓存开了就快”。本章实现的是同一条因果序列内的 KV Cache，没有实现跨请求缓存命中。

也不要直接把因果解码缓存塞进 E5 文档编码：E5 编码器与生成解码器的注意力条件不同。本项目中 NPZ 复用与 Qwen 生成时的 KV Cache 是不同层次的事情。

## 12. 留给你的练习与参考答案

**练习 1：用 prefix=8、decode=4，先手算 QKV 位置数，再运行。**

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.attention_lab --device cpu --prefix 8 --decode 4
```

答案：batch=1 时重算 `8+9+10+11+12=50`，缓存 `8+4=12`。不要只报最终长度 12，而忘记重算中重复处理的位置。

**练习 2：batch 翻倍，缓存逻辑大小如何变化？**

答案：其他量固定时翻倍。若长度也翻倍，则变为四倍。速度不一定按同样比例变化。

**练习 3：为什么只比较最后一个位置还不够？**

答案：中间位置可能出错但末位刚好相近。要比较所有输出，也要验证未来输入不影响过去、单/多 token decode、mask 位置和 cache 兼容性。

**练习 4：`model.eval()` 与 `torch.inference_mode()` 是同一个开关吗？**

答案：不是。eval 改变 dropout 等层的训练/推理行为；inference_mode 控制梯度和推理相关开销。本实验没有 dropout，仍显式区分；缓存分支只允许在无梯度推理环境使用。

下一步：在理解这个 attention 层后，再搭建带位置编码、残差、LayerNorm、前馈层和词表输出的小型 decoder；用明确划分的训练/验证数据观察损失，不急着直接跑大规模微调。
