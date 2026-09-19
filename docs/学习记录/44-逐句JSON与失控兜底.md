# 学习记录 44：逐句 JSON 产出与失控兜底

> 日期：2026-09-19。回链命中率 38.2%（学习记录 36）的改进落地：产出从"自由 Markdown"切换为**逐句 JSON**（Ollama 文法约束），产品链路真模型回归 **53/53 = 100%**（三主题、全带源、零编造编号）。落地过程中抓到一个比命中率本身更危险的故障：**`format=json` 下模型无限复读**——没有生成上限就永不停止。

## 一、实验结论（36 §七.1 的延续）

三方案 × 三主题（`.workbuddy/backlink_experiment.py`，真模型 qwen2.5:7b + docs 语料快照，评分直接 import `backlink_report`——三方案同一把尺子）：

| 方案 | 加权命中率 | 断言句 | 备注 |
| --- | --- | --- | --- |
| baseline（产品原提示） | 22.7% | 22/97 | 与 36 的 38.2% 不可比：语料快照已不同 |
| few-shot 格式示范 | 50.0% | 18/36 | "检索评测口径"主题仍 0%——不稳定 |
| **逐句 JSON** | **100%** | 57/57 | 三主题全中，无 invalid_labels、无 missing |

逐句 JSON = 提示词只描述 JSON 形状（`{"sections":[{"h":..}|{"s":..,"src":[..]}]}`）+ `format=json` 文法约束 + 程序渲染。**src 完全来自模型，渲染器只换形状**——36 §七.1 否决"程序补链"的那条线还守着。使用者拍板"逐句 JSON（推荐）"。

## 二、落地形状

- `src/llm.py`：`OllamaClient` 加 `json_mode` 参数——`chat()`/`stream_chat()` 的 payload 在 json_mode 时加 `"format": "json"`、`temperature=0.8`、`num_predict=3072`（§三）。
- `src/product/studio.py`：
  - `GUIDE_SYSTEM_PROMPT` 换成 JSON 版（保留注入防护句）。
  - 新增 `render_json_guide(raw)`：剥 ```json 围栏 → `json.loads` → 转 Markdown；**非整数 src 直接丢弃不猜**（模型给 `["1"]` 渲染器不会"猜它是 S1"）；失败返回 `(None, 原因)`——`json_parse_failed` / `json_shape_failed` / `json_empty`。
  - `stream()` 生成段攒 `raw`（原始 JSON 碎片）而不是攒正文；完成后渲染；解析失败走 **`invalid_model_output` 失败路径**（原始输出留在正文里供排查，不把散文当指南存）；**停止路径**"能渲染就渲染、渲染不出就如实存原文"。
- DeepSeek 没有文法约束可用：走提示词 + 渲染器围栏容错，解析失败如实报错重试。**未实测真模型**，行为有测试兜着。

## 三、真模型回归抓出的失控复读（本篇最重要的教训）

第一次全链路回归：第一主题 13 分钟无输出。Ollama server.log 给出答案：`n_gen = 57541`、64 t/s、持续 `slot context shift, n_discard = 4093`——**模型在 format=json 下无限复读**，靠丢弃旧上下文无限续命。

**根因是结构性的**：文法约束排除了"格式错了就停"这种自然停止点；复读环里模型不给 EOS；Ollama 默认 `num_predict` 无限 → 永不停止。实验没暴露它，是因为那批请求恰好没有进入复读环——这是概率问题，不是"实验过了就安全"。

两步修复，都先做对照再改：

1. **`JSON_MODE_NUM_PREDICT = 3072`**（硬停兜底）。上限按实验标定：三主题成品最长 1245 字符（34 句），连同 JSON 结构开销约 2000 token，留余量取 3072。正常单篇实测 387–834 token，碰不到上限；顶满 = 复读，截断的 JSON 走 `invalid_model_output` 如实失败。
2. **`JSON_MODE_TEMPERATURE = 0.8`**（对照归因）。同主题同证据：产品统一口径 0.2 下"检索评测口径"**稳定复读**（3013 token 顶满上限被硬停、如实失败），换 0.8 后 7.1 秒自然结束、11/11 全带源。低温把模型压进复读环。温度是这条线的参数而不是全局参数：**问答仍用 0.2**——自由文本没有这个问题，复读会自然结束。

## 四、最终回归数字（产品全链路）

`StudioService.stream()` 真链路：真提示词 + 文法约束 + 0.8 + 3072 + 渲染 + 落库 + `get_artifact` 现算（`.workbuddy/backlink_regression.py`，明细 `backlink_regression_result.json`）：

| 主题 | 秒 | 生成 token | 命中率 |
| --- | --- | --- | --- |
| 回链命中率 | 14.7 | 834 | 28/28 |
| 检索评测口径 | 8.6 | 471 | 14/14 |
| 数据库迁移备份 | 8.6 | 489 | 11/11 |
| **合计** | 32 | — | **53/53 = 100%** |

无 invalid_labels。抽查正文是实质陈述（迁移失败恢复流程、备份包含范围、恢复校验清单），不是凑数的空话——样张在 `.workbuddy/backlink_probe_content.md`。

对照组（改温度前）的失败实例本身就是兜底路径的实证：3013 token 顶满 → 截断 → `invalid_model_output` → 失败记录可查看、可重试，界面如实报"模型输出无法解析为结构化指南"。

## 五、前端

- **生成中不再把原始 JSON 当 Markdown 打字机渲染**——那会闪出半截 JSON。改为进度提示"正在逐句整理资料…（已收到 N 字）"，N 是真实收到的字符数；正文等 final 渲染完成再显示。
- **`stopped`/`error` 事件带回的正文也替换草稿**（此前只有 `final` 替换）——否则停止后界面上残留的是原始 JSON 碎片，而落库的是渲染后的正文，两边对不上。
- 新测试两条：生成中显示进度且不渲染正文（用受控 Promise 把流停在 token 后）；停止后界面无 `"sections"` 残留。

## 六、测试与验证

- 后端 **338 项**全过（新增 9：渲染器直测 4、流路径 3——解析失败/停止两副面孔/json_mode 只在 ollama 开、llm payload 断言 1、夹具适配若干）；前端 vitest **130 项**（16 文件）；`tsc -b` + build 过；CSS 变量门禁等效检查过（44 个变量浅色/深色成对、无字面量残留；本篇未动 styles.css）。
- 短路验证 3 点（`.workbuddy/shortcircuit_studio_json.py`）：不设 json_mode → ollama 测试挂；渲染器对解析失败放行 → 失败路径测试挂；停止路径不渲染 → "完整 JSON 停止仍渲染"挂。全部"破坏后必挂"。
- 实验与回归脚本都在 `.workbuddy/`（gitignore，不进库）：`backlink_experiment.py`（三方案）、`backlink_regression.py`（产品全链路）、`backlink_probe.py`（单主题对照，argv[2] 可覆盖温度）、`shortcircuit_studio_json.py`。

## 七、还没解决 / 边界

- DeepSeek 路径未实测真模型（提示词约束 + 围栏容错）；它的输出不稳定时走 `invalid_model_output` 如实失败，风险是失败率而不是撒谎。
- 温度归因是**单主题单次对照**：0.8 是"实验与回归全过"的配置，不是"证明了 0.2 必然复读"。`num_predict` 兜底保证最坏情况是如实失败而不是卡死。
- "检索评测口径"在 0.2 下那次回归里存了 5909 字符的截断 JSON 进正文（失败记录按设计保留原始输出）——临时库里的记录，不影响真实数据。
- 0.2.2 冻结版不含本次改动（正常：未发版）。
