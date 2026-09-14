# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=8 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# 只安装 Ollama + CPU E5 服务需要的依赖。QLoRA/CUDA 有独立硬件与镜像约束，
# 不在这个基线镜像中假装可用。
COPY requirements-service.txt requirements-vector.txt ./
RUN python -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir \
      -r requirements-service.txt -r requirements-vector.txt

COPY src/__init__.py \
     src/answer.py \
     src/bm25.py \
     src/citations.py \
     src/config.py \
     src/container_prepare.py \
     src/context.py \
     src/evidence_context.py \
     src/hybrid_retrieve.py \
     src/index_store.py \
     src/ingest.py \
     src/llm.py \
     src/markdown_structure.py \
     src/query_guard.py \
     src/reranker.py \
     src/retrieve.py \
     src/runtime.py \
     src/serve.py \
     src/settings.py \
     src/token_chunking.py \
     src/vector_retrieve.py \
     ./src/
COPY container/rag.container.toml ./container/rag.container.toml

RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /data/index_store /cache/huggingface \
    && chown -R rag:rag /app /data/index_store /cache/huggingface

USER rag
EXPOSE 8000

CMD ["python", "-m", "src.serve", "--config", "/app/container/rag.container.toml", "--bind-host", "0.0.0.0"]
