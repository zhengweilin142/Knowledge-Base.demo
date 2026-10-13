"""混合检索：向量召回（FAISS）+ BM25 关键词召回 + RRF 融合。

单靠向量检索在企业场景有两个典型短板：
  1. 专有名词、编号（如「HT-2024-087」「ISO27001」）向量召回经常漏；
  2. 纯向量对精确数字/条款号不敏感。
因此叠加一路 BM25 关键词召回，再用 RRF（倒数排名融合）合并，
不依赖两路分数量纲，工程上比加权求和稳定得多。
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from . import config
from .embedding import embedder

_ASCII_WORD = re.compile(r"[A-Za-z0-9_\-]+")


@dataclass
class Hit:
    """一条召回结果。

    score 的语义必须区分清楚，否则前端显示的"相关度"会毫无意义：
      score        —— 可解释的相关度。向量路命中时为余弦相似度（-1~1）；
                      仅被关键词路命中时为 None（没有语义相似度可言）。
      fusion_score —— RRF 融合分，只用于**排序**，跨查询不可比较（量级约 0.016）。
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    fusion_score: float = 0.0
    score: float | None = None
    rerank_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None

    @property
    def source_label(self) -> str:
        meta = self.metadata or {}
        parts = [str(meta.get("source") or "未知文件")]
        if meta.get("section"):
            parts.append(str(meta["section"]))
        page = meta.get("page")
        if isinstance(page, int) and page > 0:
            parts.append(f"第{page}页")
        return " · ".join(parts)


# --------------------------------------------------------------------------
# 中文分词（BM25 用）：ASCII 词整体保留 + 中文二元组
# 二元组无需词典即可工作，对「年假」「报销」这类词组召回效果好，
# 且不引入 jieba 这类额外依赖。
# --------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    tokens: list[str] = [w.lower() for w in _ASCII_WORD.findall(text)]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class _BM25:
    """纯 Python BM25（Okapi），k1/b 取通用默认值。"""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_len = [len(t) for t in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for tokens in corpus_tokens:
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            for tok in counts:
                self.df[tok] = self.df.get(tok, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: Iterable[str], top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            if term not in self.df:
                continue
            idf = self._idf(term)
            for idx, counts in enumerate(self.tf):
                freq = counts.get(term)
                if not freq:
                    continue
                dl = self.doc_len[idx] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] = scores.get(idx, 0.0) + idf * freq * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


class _BM25Cache:
    """缓存 BM25 索引：只有分块数量与最大序号变化时才重建。"""

    def __init__(self) -> None:
        self.bm25: _BM25 | None = None
        self.fingerprint: tuple[int, float] | None = None

    def ensure(self, records: Sequence[Any]) -> None:
        max_ord = max((int((r.metadata or {}).get("ordinal", -1)) for r in records), default=-1)
        fp = (len(records), float(max_ord))
        if self.fingerprint == fp and self.bm25 is not None:
            return
        self.bm25 = _BM25([tokenize(r.text) for r in records]) if records else None
        self.fingerprint = fp


_cache = _BM25Cache()
_cache_lock = threading.Lock()


def invalidate_index() -> None:
    """入库/删除后调用，强制下次检索重建 BM25 索引。"""
    with _cache_lock:
        _cache.fingerprint = None


# --------------------------------------------------------------------------
# 检索主流程
# --------------------------------------------------------------------------
def retrieve_records(
    records: Sequence[Any],
    embed_fn: Any,
    vector_search: Any,
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> list[Hit]:
    """纯函数版检索，便于测试：不直接依赖 kb 模块。

    records      : 具备 .chunk_id / .text / .metadata 的对象序列
    embed_fn     : (text) -> list[float]
    vector_search: (embedding, k, source_filter) -> [(记录下标, 相似度)]
    """
    top_k = top_k or config.FINAL_TOP_K
    if not records:
        return []

    # ---- 路 1：向量召回 ----
    vector_hits: list[Hit] = []
    try:
        neighbors = vector_search(embed_fn(query), config.VECTOR_TOP_K, source_filter)
        for rank, (pos, sim) in enumerate(neighbors, start=1):
            rec = records[pos]
            vector_hits.append(
                Hit(
                    chunk_id=rec.chunk_id,
                    text=rec.text,
                    metadata=rec.metadata or {},
                    fusion_score=1.0 / (config.RRF_K + rank),
                    score=sim,
                    vector_rank=rank,
                    vector_score=sim,
                )
            )
    except Exception as exc:  # 向量路失败不应让整个查询崩掉
        print(f"[retrieve] 向量召回失败: {exc}")

    # ---- 路 2：BM25 关键词召回 ----
    bm25_hits: list[Hit] = []
    with _cache_lock:
        _cache.ensure(records)
        if _cache.bm25 is not None:
            for rank, (pos, _raw) in enumerate(
                _cache.bm25.search(tokenize(query), config.BM25_TOP_K), start=1
            ):
                rec = records[pos]
                if source_filter and str((rec.metadata or {}).get("source")) != source_filter:
                    continue
                bm25_hits.append(
                    Hit(
                        chunk_id=rec.chunk_id,
                        text=rec.text,
                        metadata=rec.metadata or {},
                        fusion_score=1.0 / (config.RRF_K + rank),
                        score=None,  # 关键词命中没有余弦相似度
                        bm25_rank=rank,
                    )
                )

    # ---- RRF 融合：同一块在两路中的贡献相加 ----
    merged: dict[str, Hit] = {}
    for hit in vector_hits + bm25_hits:
        cur = merged.get(hit.chunk_id)
        if cur is None:
            merged[hit.chunk_id] = hit
        else:
            cur.fusion_score += hit.fusion_score
            if cur.vector_rank is None and hit.vector_rank is not None:
                cur.vector_rank = hit.vector_rank
                cur.vector_score = hit.vector_score
                cur.score = hit.score
            if cur.bm25_rank is None:
                cur.bm25_rank = hit.bm25_rank

    # 排序只看融合分；score 仅作为对外展示的相关度
    ranked = sorted(merged.values(), key=lambda h: h.fusion_score, reverse=True)

    # 单文件配额：避免一份文档霸占全部上下文
    picked: list[Hit] = []
    per_source: dict[str, int] = {}
    cap = max(2, top_k // 2 + 1)
    for hit in ranked:
        src = str((hit.metadata or {}).get("source") or "")
        if per_source.get(src, 0) >= cap:
            continue
        per_source[src] = per_source.get(src, 0) + 1
        picked.append(hit)
        if len(picked) >= top_k:
            break
    if len(picked) < top_k:  # 配额限制导致不足时补齐
        seen = {h.chunk_id for h in picked}
        for hit in ranked:
            if hit.chunk_id not in seen:
                picked.append(hit)
                seen.add(hit.chunk_id)
            if len(picked) >= top_k:
                break
    return picked


def retrieve(query: str, top_k: int | None = None, source_filter: str | None = None) -> list[Hit]:
    """对外便捷入口：第一阶段混合召回 → 第二阶段 Cross-Encoder 精排。"""
    from . import kb

    candidates = retrieve_records(
        kb.all_records(),
        embedder.embed_query,
        kb.search,
        query,
        max(config.RERANK_CANDIDATES, top_k or config.FINAL_TOP_K),
        source_filter,
    )
    return rerank_hits(query, candidates, top_k)


def rerank_hits(query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    """第二阶段精排：用 Cross-Encoder 联合编码「问题+文档」重新打分。

    重排器不可用（未安装/加载失败/已关闭）时**原样返回**，
    由 gate_by_relevance 里的余弦门槛兜底，不会阻断服务。
    """
    if not hits or not config.RERANK_ENABLED:
        return hits

    # 先按余弦硬底线过滤，省掉明显无关片段的算力
    candidates = [
        h for h in hits
        if h.score is None or h.score >= config.RERANK_COSINE_FLOOR
    ]
    if not candidates:
        return hits

    from .rerank import reranker  # 局部导入，避免无重排需求时的额外依赖加载

    scores = reranker.score(query, [h.text for h in candidates])
    if len(scores) != len(candidates):
        return hits  # 重排不可用或推理失败

    for hit, sc in zip(candidates, scores):
        hit.rerank_score = sc
    ranked = sorted(candidates, key=lambda h: h.rerank_score or 0.0, reverse=True)
    return ranked[: (top_k or config.FINAL_TOP_K)]


def gate_by_relevance(hits: list[Hit], min_relevance: float | None = None) -> list[Hit]:
    """相关性闸门：丢弃与问题无关的召回结果。

    判据分两种情况，**优先使用重排分**：

    ① 重排器可用（生产推荐）：
       以 Cross-Encoder 的重排分为主判据。实测在示例语料上
         真相关问题 0.909~4.895，无关问题 -10.19~-0.043，区间完全不重叠，
       因此单一门槛即可干净区分，无需在误杀与漏拦之间取舍。

    ② 重排器不可用（降级）：
       退回余弦相似度 + 「两路一致」双重约束。这是有损的兜底——
       实测「北京到上海高铁多久」会被放行（余弦 0.637），
       因为文档里有「高铁乘坐二等座」被 BM25 当作稀有词命中。

    返回顺序：有重排分时按其降序，否则按余弦降序。
    """
    if not hits:
        return []

    reranked = [h for h in hits if h.rerank_score is not None]
    if reranked:
        return [h for h in reranked if (h.rerank_score or 0.0) >= config.RERANK_MIN_SCORE]

    threshold = config.MIN_RELEVANCE if min_relevance is None else min_relevance
    kept: list[Hit] = []
    for hit in hits:
        if hit.score is None or hit.score < threshold:
            continue
        if hit.bm25_rank is not None and hit.score < config.BM25_MIN_VECTOR:
            continue  # 关键词命中但语义不够像：判定为关键词路造成的假阳性
        kept.append(hit)
    return sorted(kept, key=lambda h: h.score or 0.0, reverse=True)
