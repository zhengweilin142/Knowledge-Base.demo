"""重排（Cross-Encoder Rerank）：RAG 里提升精度的关键一环。

为什么必须有它
--------------
第一阶段检索（向量 + BM25）本质是**双塔**结构：问题和文档分别编码成向量，
只用一次点积比较。它快，但无法建模问题与文档之间的细粒度交互，
因此存在两类顽固错误（本机在 4 份示例文档上实测到的真实数据）：

  问题                       余弦相似度   是否相关
  「出差住酒店能报多少钱」      0.7130      相关
  「北京到上海高铁多久」        0.6370      无关  ← 因为文档里有「高铁乘坐二等座」，
                                                   BM25 的稀有词高分把它顶了上来

两段区间**重叠**，所以无论把余弦阈值设在 0.45 还是 0.64，都必然在
「漏拦假阳性」和「误杀真相关」之间二选一。实测扫描证明了这一点。

Cross-Encoder 把「问题 + 文档」拼成一个序列送进模型联合编码，
能真正判断二者是否在回答同一个问题。同一批数据上的重排分：

  相关：0.909 ~ 4.895        无关：-10.19 ~ -0.043       ← 完全不重叠

因此正确做法是：第一阶段广召回（保证相关文档不漏），第二阶段重排（保证精度）。
这也是生产级 RAG 的标准两段式架构。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Sequence

from . import config

_CACHE = config.MODELS_DIR / "fastembed"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XATTR", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "hf"))


class Reranker:
    """惰性加载的 Cross-Encoder 重排器；加载失败时自动降级，不阻断服务。"""

    _instance: "Reranker | None" = None
    _cls_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._error: str | None = None
        self._warned = False

    @classmethod
    def instance(cls) -> "Reranker":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        return config.RERANK_ENABLED

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _get_model(self) -> Any:
        if self._model is not None or self._error is not None:
            return self._model
        with self._load_lock:
            if self._model is None and self._error is None:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(
                        model_name=config.RERANK_MODEL,
                        cache_dir=str(_CACHE),
                    )
                except Exception as exc:
                    self._error = f"{type(exc).__name__}: {exc}"
                    print(f"[rerank] 重排模型不可用，降级为纯余弦门槛: {self._error}")
        return self._model

    def warmup(self) -> bool:
        return self._get_model() is not None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """返回与 documents 等长、顺序对应的相关性得分（越大越相关）。

        注意 fastembed 的 rerank() 返回的是 **float 迭代器**，不是 (索引, 分数) 元组
        ——按位置与输入文档一一对应。返回长度不符时视为失败，交由上层降级。
        """
        model = self._get_model()
        if model is None or not documents:
            return []
        try:
            scores = [float(s) for s in model.rerank(query, list(documents))]
        except Exception as exc:
            self._warn_once(f"重排推理失败，本次降级: {type(exc).__name__}: {exc}")
            return []
        if len(scores) != len(documents):
            self._warn_once(
                f"重排返回数量不符（{len(scores)} != {len(documents)}），本次降级"
            )
            return []
        return scores

    def _warn_once(self, message: str) -> None:
        """重排不可用时每个查询都会失败，只提示一次，避免刷屏。"""
        with self._load_lock:
            if self._warned:
                return
            self._warned = True
        print(f"[rerank] {message}")


reranker = Reranker.instance()
