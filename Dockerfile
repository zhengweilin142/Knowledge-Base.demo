# 企业知识库 Demo —— 容器镜像（Hugging Face Spaces / 任意 Docker 环境均可用）
#
# 体积与启动权衡说明：
#   模型合计约 1.2GB，**不烤进镜像**（否则镜像臃肿、构建慢且易超平台上限）。
#   改为容器启动时下载到可写缓存；平台磁盘缓存通常在重启间保留，
#   首次冷启动约 1~3 分钟，之后为秒级。详见 tools/preload.py。

FROM python:3.12-slim

# - PYTHONUNBUFFERED：让日志实时输出，便于在平台日志里观察预热进度
# - TOKENIZERS_PARALLELISM：避免 tokenizers 的多进程告警
# - HF_HUB_DISABLE_XET：本机实测该传输后端会报 I/O 错误，回退标准 HTTP
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    ANONYMIZED_TELEMETRY=False \
    KB_DATA_DIR=/tmp/kb-data \
    KB_MODELS_DIR=/tmp/kb-models \
    KB_LOG_DIR=/tmp/kb-logs \
    PORT=7860

# libgomp1 是 faiss-cpu 的运行时依赖；curl 供健康检查使用
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖：这一层能被 Docker 缓存，改代码时不必重装
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

# 再拷代码
COPY app ./app
COPY tools ./tools
COPY verify.py ./
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 以非 root 用户运行（平台安全要求，也避免容器内权限过大）
RUN useradd -m -u 1000 appuser \
 && mkdir -p /tmp/kb-data /tmp/kb-models /tmp/kb-logs \
 && chown -R appuser:appuser /app /tmp/kb-data /tmp/kb-models /tmp/kb-logs
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]
