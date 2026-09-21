# Obsidian RAG

一个可在 Windows 本机运行的个人知识工作台：把 Markdown、文字型 PDF 和 Jupyter Notebook 接入本地资料库，用 Ollama 或 DeepSeek 搜索、追问、核对来源，并把有用结论收藏或导出为 Markdown。

当前产品版本：**0.2.2**。项目仍是个人学习与小范围试用项目，不是云端 SaaS。

## 现在能做什么

- **资料接入**：连接多个只读 Markdown 文件夹；上传 Markdown、文字型 PDF 和 .ipynb。
- **资料管理**：查看导入状态、失败原因、重试、更新和断开文件夹；断开不会删除原文件。
- **检索**：默认使用 BM25；可选本地 E5 向量和混合检索。检索模型在 CPU 上也能运行。
- **问答**：Ollama 本地模型和 DeepSeek 两条生成路径，支持流式输出、停止、重试、会话保存和基本追问。
- **证据阅读**：回答带 [S1] 等来源标签；Markdown 定位到章节/行号，PDF 定位到页码，并保留文档快照和索引版本。
- **知识整理**：收藏回答、编辑标题和备注、批量删除、导出 UTF-8 Markdown；Studio 可生成指南、思维导图和信息图。
- **可靠性**：备份/恢复、数据库迁移保护、诊断导出、模型不可用提示、重启和退出入口。
- **隐私边界**：联网补充和记忆接口已预留但默认关闭；没有账号、云端同步、自动更新或后台遥测。

## 产品边界

这条主线关注“资料接入 → 检索 → 有证据的回答 → 保存成果 → 可交付安装包”。预训练、SFT、QLoRA、adapter、复杂 Agent 自主执行和训练基础设施保留在实验区，不阻塞产品开发。

当前限制：

- PDF 首版只处理可提取文字的页面，不包含 OCR、图表理解和复杂表格重建；暂不支持 PPTX。
- 扫描件、加密 PDF 和无法解析的页面会显示失败或待处理状态。
- 基本追问有长度上限；每次回答都会重新检索，历史回答不会自动当作事实证据。
- Web 搜索只有接口接缝，默认没有真实搜索后端，也不会在未配置时发起请求。
- 使用 DeepSeek 时，问题、必要的对话内容和检索片段会发送到远端服务。
- 没有多用户、云同步、自动更新和自动修改 Obsidian 笔记。

## 使用者快速开始

仓库的 dist-installer/ 保存了开发机生成的 Windows x64 安装包：

```text
dist-installer/ObsidianRAG-Setup-0.2.2-win-x64.exe
SHA-256: 209770effe5f82a04eeae9374a93da8c7657e1840b7ee321485b5195f6c77fd4
```

安装后：

1. 打开产品，在设置页完成首次准备。
2. 连接 Markdown 文件夹，或上传 Markdown/PDF/Notebook。
3. 等待导入完成；首次使用 E5 时按提示下载模型，也可以先使用默认 BM25。
4. 搜索或提问，点击 [S1] 打开章节、行号或 PDF 页码。
5. 收藏满意的回答，补充备注后导出 Markdown，直接放入 Obsidian。

个人数据默认在 %LOCALAPPDATA%\ObsidianRAG，与安装目录分离。连接文件夹只读扫描，上传文件复制到产品数据目录；卸载默认保留个人资料。DeepSeek 密钥保存到 Windows 凭据管理器，界面只显示配置状态。

### 生成后端

本地模型可使用 Ollama：

```powershell
ollama serve
ollama pull qwen2.5:7b
```

然后在设置中选择 Ollama 和模型名称。检索可以在 CPU 上运行，生成速度取决于本机模型和硬件。选择 DeepSeek 时在设置页填写 API Key，产品不会读取开发仓库的 .env。

## 开发环境与本地运行

要求：Windows 10/11 x64、Python、Node.js/npm 和 Git。开发仓库使用 .venv，发布脚本会准备独立的 Python 3.12 CPU 环境。

```powershell
Set-Location -LiteralPath 'C:\Users\zjm\Desktop\file\项目\obsidian-rag'

npm --prefix web install
.\.venv\Scripts\python.exe -m pip install -r requirements-product.txt

npm --prefix web run build
.\.venv\Scripts\python.exe -X utf8 -m src.product_entry --no-browser
```

打开 http://127.0.0.1:8765。端口被占用时：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.product_entry --no-browser --port 8876
```

启动器支持 --data-root 指定开发数据目录；产品 API 位于 /api/v1，历史实验接口 /v1 保留给旧脚本。

前端命令：

```powershell
npm --prefix web run dev
npm --prefix web run build
npm --prefix web run test -- --run
```

前端使用 React、TypeScript 和 Vite；生产构建由 FastAPI 同源提供。界面包含纸张、石板、沙色三套主题，以及浅色/深色模式。

## 构建与验证

构建 Windows 安装包：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_product.ps1
```

输出为 dist\ObsidianRAG\ 便携目录和 dist-installer\ObsidianRAG-Setup-<version>-win-x64.exe。发布前运行安装验收：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_installer.ps1
```

它覆盖安装、启动、升级/恢复和卸载保留等关键路径。完整安装验收是否针对当前版本重跑，以对应发布说明为准。

最小离线检查：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
npm --prefix web run build
npm --prefix web run test -- --run
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_css_tokens.ps1
```

产品验收还覆盖中文路径、无 Python/显卡、端口占用、模型未配置、Markdown/PDF 导入与重试、引用定位、流式停止、追问、刷新恢复、收藏导出、升级数据保留和恢复后的实际问答。

## 代码结构与数据流

```text
src/product/                 # 产品主线：API、数据库、导入、检索、聊天、Studio、备份恢复
src/product_entry.py         # 产品启动器
web/                          # React + TypeScript + Vite 前端
tests/                        # Python 产品/实验测试
data/                         # 题集和报告；私人生成物默认被 Git 忽略
docs/                         # 产品文档、评测、发布说明、学习记录
scripts/                      # 构建、安装验收和发布辅助脚本
installer/                   # Inno Setup 配置和随包资源
src/                          # 旧 RAG、BM25/E5、评测、推理/训练实验
examples/                     # 示例资料和最小演示
```

```text
浏览器 -> FastAPI /api/v1 -> SQLite
       -> 文档快照 / 索引 -> BM25 或 E5/混合检索
       -> 证据预算 -> Ollama / DeepSeek 流式生成
       -> 来源阅读、收藏、Markdown 导出
```

产品索引与开发仓库中的旧索引分开。文件更新先产生新快照和新索引，验证完成后再切换；历史回答保存来源和索引版本，避免资料修改后引用静默指向另一份内容。

## 评测结论

早期 RAG 和生成评测仍保留，用来学习检索、证据和拒答边界，不是产品使用必需项。默认 BM25 是因为当前私人中文技术语料下它更易解释、启动更快；E5 和混合检索用于对照实验，不宣称对所有问题都更好。

Studio 指南使用逐句结构化输出，渲染器只显示模型实际给出的句子和来源；思维导图由标题层级确定树结构，不依赖模型。引用标签有效只表示标签对应本次上下文中的来源，不等于事实正确性已被自动证明。

## 文档入口

- [文档索引](docs/README.md)：哪些文档生效、哪些只是历史或参考。
- [学习记录索引](docs/学习记录/README.md)：从 RAG 实验到产品阶段 28–49 的实现与故障记录。
- [用户指南](docs/用户指南.md)：面向安装后使用者。
- [0.2.2 发布说明](docs/RELEASE_NOTES_0.2.2.md)：当前版本功能、修复、限制和验收状态。
- [发布检查清单](docs/发布检查清单.md)：升版、打包、验收和版本引用点。
- [检索评测复现指南](docs/检索评测复现指南.md)：复现实验方法和口径。
- [第三方许可证](docs/THIRD_PARTY_LICENSES.md)：随包依赖的许可清单。

## Git 工作方式

产品和学习记录按阶段小步提交，保持每天可回退、可解释：

```powershell
git status
git diff --stat
git add <本阶段文件>
git commit -m "feat: <阶段目标>"
git push origin main
```

不要提交 .env、API Key、私人笔记、%LOCALAPPDATA%\ObsidianRAG 数据和本地生成报告。仓库当前没有声明开源许可证；如要对外分发或允许他人修改，先补充明确的 License、第三方资源许可和隐私说明。
