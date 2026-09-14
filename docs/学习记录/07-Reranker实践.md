# 阶段 7：本地 Reranker 实践

日期：2026-09-12。

## 做了什么

新增 `src/reranker.py`，把本地 CrossEncoder 接入统一检索引擎。支持 BM25、vector、hybrid 三种候选来源，是否重排由 `--rerank` 显式控制，不替换原来的默认行为。

模型：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`；固定 revision 为 `1427fd652930e4ba29e8149678df786c240d8825`。它以问题和段落的联合输入做相关性排序，模型卡说明了多语言 MMARCO 训练来源。[官方模型卡](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)

只下载必要的 safetensors 权重和配置/tokenizer 文件，权重约 471 MB，未下载重复的 PyTorch bin、ONNX/OpenVINO 导出或训练脚本。下载完成后默认本地读取，未使用远程推理服务。

## 与 E5 的区别

E5 可以分别编码问题和文档，所以文档向量能预先计算。CrossEncoder 联合读取这次问题与候选，相关性分数依赖这一对输入，不能把某篇笔记的重排序分数永久缓存后用于所有问题。[CrossEncoder 官方接口](https://sbert.net/docs/package_reference/cross_encoder/model.html)

因此先召回，再只重排少量候选。本版默认重排前 20 个，返回前 5 个。它无法挽回没进入候选集合的证据。

## 数据经过了哪些步骤

1. 检索器产生候选；Hybrid 内部先做两路 RRF。
2. 沿用已有低信息标题规则产生重排前顺序。
3. 取前 20 个候选，组成 `(完整问题, 标题路径 + 正文)`。
4. CrossEncoder 输出原始 logit。
5. 按 logit 降序排列，保留重排前名次、检索分、重排分、联合 token 数及是否截断。

Reranker 最终分不再与 BM25 分、余弦或 RRF 分相加，也不再次乘 0.35。特别是负 logit 乘 0.35 反而会变大，可能把应降权的片段抬上去。

## 遇到的问题

### Windows 下载警告

HF 提示未登录和当前缓存目录不支持符号链接，但下载正常完成。未登录不是私有模型权限错误，符号链接警告也不等于无法运行。没有为了消除警告修改系统安全设置、开启开发者模式或要求管理员运行。

### 分数是负数

这是正常的原始 logit。本轮明确使用 Identity 激活，避免不同默认设置造成分数口径混乱。示例中相关段落约 4.42、无关段落约 -9.11，但这些数不是“答对概率”，不能机械设置 0 分拒答阈值。

### E5 不超限，重排序仍可能超限

重排输入多了问题，而且 tokenizer/格式可能不同。因此另外统计每个候选的联合 token 数。本轮三种候选方式在两个检索题集上均未出现联合输入超限；这不是对未来任意长问题的保证。

### 并不是所有排名都改善

在新的 822 个片段上，开发集重排后三种候选方式的 Source Hit@5 都为 26/26；保留集 vector+rerank 为 8/8，而 bm25+rerank、hybrid+rerank 为 7/8。没有因为这些结果继续改模型、改候选数或针对单题加规则。

热查询延迟从不重排的大约 3～10 ms 增至重排后的约 82～91 ms，具体随方法与题集变化。它是额外模型计算带来的成本，不应只记录排名收益。

## 如何体验

```powershell
# 本机已准备好模型；仅在新机器首次准备时添加 --allow-download。
.\.venv\Scripts\python.exe -X utf8 -m src.reranker
.\.venv\Scripts\python.exe -X utf8 -m src.hybrid_retrieve "PagedAttention 是什么？它解决什么问题？" --chunks-file data/generated/token_v1/chunks.json --rerank
```

若显存紧张，可用 `--rerank-device cpu` 单独把重排模型放到 CPU；`--device cpu` 控制向量模型，未指定独立设备时也会被重排模型继承。Ollama 的设备仍由它自己的服务管理。

模型加载或推理失败会明确报错，不悄悄换回不重排的结果；否则你可能以为在测 Reranker，实际测的是另一条路径。

## 练习与参考答案

问：为何重排时保留 pre_rerank_rank？答：用于确认好证据原本是否在候选中，以及是被重排提升还是压低；没有它就难区分召回失败和排序失败。

问：重排序分数很高，能说明引用事实正确吗？答：不能，它判断相关性，不负责验证结论是否由证据逻辑支持。
