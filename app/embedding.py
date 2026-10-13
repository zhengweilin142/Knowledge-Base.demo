"""向量化：fastembed（ONNX Runtime）加载 BGE 中文模型。

选型经过三步实测，记录在此以便你日后复用同样的判断过程：

1) sentence-transformers  → 效果最好，但会拉入 PyTorch（约 2GB 磁盘、运行时数 GB 内存），
   本机 7.6G 内存 / C 盘仅剩 14G，不合适。
2) ChromaDB 内置 ONNX 版  → 轻量，但已确认它的默认函数**硬编码英文模型**
   all-MiniLM-L6-v2（下载路径写死在 ~/.cache/chroma 且不读任何环境变量），
   中文检索效果差，且实测 ChromaDB 1.5.9 在 Windows 上建库直接报错。
3) fastembed（本方案）    → Qdrant 出品的 ONNX 嵌入库，原生支持 BAAI/bge-small-zh-v1.5，
   磁盘约 90MB、内存约 200MB，中文语义效果实测良好（见 verify.py 的语义一致性检查）。

注意：fastembed 用的是 Qdrant 转换版权重（Qdrant/bge-small-zh-v1.5），
该版本已内置查询侧指令，**不需要**再手工加「为这个句子生成表示…」前缀，
加了反而会降低相似度。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Sequence

from . import config

# ---- 必须在导入 fastembed / huggingface_hub 之前设置缓存目录 ----
# 否则模型会下载到 C:\Users\<用户>\.cache，本机 C 盘只剩十几 GB。
_FASTEMBED_CACHE = config.MODELS_DIR / "fastembed"
_HF_CACHE = config.MODELS_DIR / "hf"
for _d in (_FASTEMBED_CACHE, _HF_CACHE / "hub"):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("FASTEMBED_CACHE_PATH", str(_FASTEMBED_CACHE))
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# 关掉 Xet 传输（本机实测会报 I/O error: os error 5），回退标准 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if config.HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)


def model_cache_dir() -> Path:
    """模型落盘位置（完全离线部署时整体拷贝此目录即可）。"""
    return _FASTEMBED_CACHE


class Embedder:
    """惰性加载、进程内共享、线程安全的嵌入器。"""

    _instance: "Embedder | None" = None
    _cls_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._load_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "Embedder":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(
                        model_name=config.EMBED_MODEL_NAME,
                        cache_dir=str(_FASTEMBED_CACHE),
                    )
        return self._model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        return [list(map(float, vec)) for vec in model.embed(list(texts))]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """文档侧向量（按批处理，避免一次性占用过多内存）。"""
        out: list[list[float]] = []
        batch = max(1, config.EMBED_BATCH_SIZE)
        for i in range(0, len(texts), batch):
            out.extend(self._embed(list(texts[i : i + batch])))
        return out

    def embed_query(self, text: str) -> list[float]:
        """查询侧向量。BGE 中文需加指令前缀；Qdrant 转换版已内置，故前缀默认留空。"""
        prefixed = f"{config.EMBED_QUERY_PREFIX}{text}" if config.EMBED_QUERY_PREFIX else text
        return self._embed([prefixed])[0]

    def warmup(self) -> None:
        """预热：把模型加载与首次推理放到启动阶段，避免污染首次查询延迟。"""
        self.embed_query("预热")

    def name(self) -> str:
        return f"fastembed-onnx:{config.EMBED_MODEL_NAME}"

    # 兼容可能的 embedding_function 协议调用
    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return self.embed_documents(input)


embedder = Embedder.instance()
