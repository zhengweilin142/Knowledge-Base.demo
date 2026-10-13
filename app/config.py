"""集中配置：所有可调参数都在这里，便于迁移到服务器时只改 .env。

注意：本文件在导入任何向量化/网络库之前被加载（app/embedding.py 首行 import config），
因此下面设置的环境变量能可靠生效。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"          # 原始文档存放处
INDEX_DIR = DATA_DIR / "index"            # FAISS 索引 + 分块元数据
MODELS_DIR = BASE_DIR / "models"          # 本地模型缓存
LOG_DIR = BASE_DIR / "logs"

for _d in (SOURCE_DIR, INDEX_DIR, MODELS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """极简 .env 读取，避免额外依赖；已存在的环境变量优先。"""
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# ---- 嵌入模型 ----
# fastembed 使用 Qdrant 转换版权重，原生支持中文，512 维，约 90MB。
EMBED_MODEL_NAME = os.getenv("KB_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
# 该版本已内置查询侧指令，前缀留空；若换成官方 BAAI 权重再考虑填
# "为这个句子生成表示以用于检索相关文章："
EMBED_QUERY_PREFIX = os.getenv("KB_QUERY_PREFIX", "")
EMBED_BATCH_SIZE = int(os.getenv("KB_EMBED_BATCH", "32"))

# ---- 切分参数 ----
CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "500"))       # 每块目标字符数
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "80"))  # 相邻块重叠字符数
MIN_CHUNK_SIZE = int(os.getenv("KB_MIN_CHUNK", "40"))     # 小于此长度丢弃

# ---- 检索参数 ----
VECTOR_TOP_K = int(os.getenv("KB_VECTOR_TOPK", "20"))     # 向量召回条数
BM25_TOP_K = int(os.getenv("KB_BM25_TOPK", "20"))         # 关键词召回条数
FINAL_TOP_K = int(os.getenv("KB_FINAL_TOPK", "5"))        # 送入大模型的条数
RRF_K = int(os.getenv("KB_RRF_K", "60"))                  # RRF 融合平滑常数
# 相关度门槛（余弦相似度）。向量检索永远返回"最近邻"，即使问的问题与知识库
# 完全无关也会返回一堆结果——这是 RAG 最常见的失效模式：用户看到无关引用，
# 模型在此之上编造答案。低于此门槛的结果一律丢弃。
# 中文 BGE 模型在 4 份示例文档上的实测分布：
#   真正相关  0.45 ~ 0.78
#   无关问题  0.30 ~ 0.64   ← 与相关区间重叠，单一门槛无法干净区分（见 rerank.py）
# 因此这是**兜底门槛**：重排器可用时，最终判据是 RERANK_MIN_SCORE。
MIN_RELEVANCE = float(os.getenv("KB_MIN_RELEVANCE", "0.40"))

# 关键词路（BM25）命中，但语义相似度低于此值时丢弃该命中。
# 仅在重排器不可用时作为兜底（重排器可用时由重排分做主判据）。
BM25_MIN_VECTOR = float(os.getenv("KB_BM25_MIN_VECTOR", "0.55"))

# ---- 重排（Cross-Encoder）----
# 第二阶段精排，解决"关键词高分污染语义检索"的问题。见 app/rerank.py 的完整说明。
RERANK_ENABLED = os.getenv("KB_RERANK", "1").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("KB_RERANK_MODEL", "BAAI/bge-reranker-base")  # 约 1.04GB
# 重排分门槛。实测：真相关 0.909~4.895，无关 -10.19~-0.043，取中间值即可。
RERANK_MIN_SCORE = float(os.getenv("KB_RERANK_MIN_SCORE", "0.5"))
# 重排阶段的余弦硬底线：低于此值连重排都不送，省算力
RERANK_COSINE_FLOOR = float(os.getenv("KB_RERANK_COSINE_FLOOR", "0.25"))
# 送进重排器的候选数（第一阶段广召回，第二阶段精排）
RERANK_CANDIDATES = int(os.getenv("KB_RERANK_CANDIDATES", "20"))

# ---- 大模型（回答生成）----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))

SYSTEM_PROMPT = os.getenv(
    "KB_SYSTEM_PROMPT",
    "你是一个企业内部知识库助手。请严格依据提供的参考资料回答问题。\n"
    "规则：\n"
    "1. 只使用参考资料中的信息，不要编造，不要引入外部知识。\n"
    "2. 每处结论后面必须标注来源编号，格式如 [1]、[2]。\n"
    "3. 如果参考资料不足以回答，直接说明「根据现有资料无法回答」，并指出还缺什么信息。\n"
    "4. 用简体中文回答，条理清晰；涉及数字、日期、金额要原样引用。",
)

# ---- 服务 ----
WEB_HOST = os.getenv("KB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("KB_PORT", "8000"))

# 模型下载镜像（内网/受限网络可指向私有镜像，如 https://hf-mirror.com）
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
