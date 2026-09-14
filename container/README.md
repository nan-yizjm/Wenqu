# 本机容器基线

这个镜像只运行 Ollama provider + CPU E5 检索。模型仍由宿主机 Ollama 管理，Vault 只读挂载；索引和 Hugging Face 缓存写入 `data/generated/`，不会进入 Git。

1. 复制 `container/.env.container.example` 为 `container/.env.container.local`，只修改 Vault 路径。不要添加 API Key。
2. 启动 Docker Desktop，显式准备固定 revision 的 E5 缓存。这是唯一会下载模型权重的步骤；后面的索引和服务只读本地缓存：

```powershell
docker compose --env-file .\container\.env.container.local `
  --profile tools run --rm rag-model-cache
```

如果宿主机已有同一 Hugging Face 缓存，可把 `RAG_HF_CACHE_DIR` 指向它，并先运行不联网检查：

```powershell
docker compose --env-file .\container\.env.container.local `
  --profile tools run --rm rag-model-cache `
  python -m src.container_prepare --offline-check
```

3. 构建独立的 Linux 索引：

```powershell
docker compose --env-file .\container\.env.container.local `
  --profile tools run --rm rag-index
```

Windows 上已有的索引 manifest 记录 Windows Vault 路径，不能直接冒充容器内 `/data/vault` 的索引。首次容器运行必须重建；之后未修改语料时会复用缓存。

4. 启动服务：

```powershell
docker compose --env-file .\container\.env.container.local up --build rag-service
```

5. 另一个终端检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/index
```

6. 停止并删除容器网络：

```powershell
docker compose --env-file .\container\.env.container.local down
```

不要使用 `-v`，除非你明确要删除 Compose 管理的卷。本配置使用项目内被忽略目录保存索引和模型缓存；删除前仍应先核对路径。

限制：Docker Desktop 需要允许容器访问宿主机 Ollama；若 Ollama 只接受 127.0.0.1 或被防火墙阻止，ready 会失败。本基线不包含 GPU QLoRA、远程 DeepSeek 密钥、公开网络认证或私人数据镜像。
