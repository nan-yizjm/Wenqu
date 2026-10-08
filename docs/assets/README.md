# 展示素材

本目录截图来自实际运行的 Wenqu 0.2.3 React 界面，使用 `examples/demo/` 合成资料和独立产品数据目录，浅色纸张主题、1440×1000 视口。

本机 Ollama 服务可启动但没有已下载模型，因此本次使用 `scripts/demo_product.py` 注入的**固定测试响应**。问答正文、模型名称与指南标题均标注“测试演示”；这些截图展示界面与来源/收藏链路，不作为真实模型回答或质量评测。检索、文档版本、来源定位、收藏和导出仍调用真实产品服务。

复现：

```powershell
.\.venv-product\Scripts\python.exe -X utf8 -m scripts.demo_product
```

打开 `http://127.0.0.1:8892`。使用临时浏览器配置和 Playwright 截图，不读取用户浏览器会话。演示数据默认写入被忽略的 `output/release/showcase-data`。

`architecture.svg` 是可编辑矢量图，准确标注本地与 DeepSeek 远端边界，不包含默认关闭接缝的虚构实现。
