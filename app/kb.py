"""向量库存储层：FAISS 索引 + JSON 元数据。

选型说明（为什么不是 ChromaDB）：
  本机实测 ChromaDB 1.5.9（Rust 内核）在 Windows 上建库即报
  `unable to open database file`，而降级到 0.5.x 需要现场编译 chroma-hnswlib。
  FAISS 官方提供 Windows 预编译 wheel，且存储完全显式：
    data/index/kb.index  —— 向量索引
    data/index/kb.json   —— 分块文本 + 元数据 + 原始向量
  删掉这两个文件就是彻底重置，没有任何隐藏状态，便于审计与迁移。
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import config
from .ingest import Chunk, iter_source_files, prepare_chunks
from .retrieval import invalidate_index

INDEX_PATH = config.DATA_DIR / "index" / "kb.index"
META_PATH = config.DATA_DIR / "index" / "kb.json"

_lock = threading.RLock()
_index: Any = None
_chunks: list[dict[str, Any]] = []
_loaded = False


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class Record:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector: list[float] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.chunk_id, "text": self.text, "meta": self.metadata, "vec": self.vector}


def _from_json(raw: dict[str, Any]) -> Record:
    return Record(
        chunk_id=raw["id"],
        text=raw.get("text", ""),
        metadata=raw.get("meta") or {},
        vector=raw.get("vec") or [],
    )


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------
def _import_faiss():
    import faiss  # 只在真正用到时导入，便于无依赖时给出清晰报错

    return faiss


def _build_index(vectors: list[list[float]]):
    import numpy as np

    faiss = _import_faiss()
    if not vectors:
        return None
    arr = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(arr)  # 归一化后内积即余弦相似度
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


def load() -> None:
    """从磁盘载入索引与元数据（进程内只做一次）。"""
    global _index, _chunks, _loaded
    with _lock:
        if _loaded:
            return
        if META_PATH.exists():
            raw = json.loads(META_PATH.read_text(encoding="utf-8"))
            _chunks = [_from_json(r) for r in raw.get("records", [])]
        else:
            _chunks = []
        if INDEX_PATH.exists():
            _index = _import_faiss().read_index(str(INDEX_PATH))
        else:
            _index = _build_index([r.vector for r in _chunks])
        _loaded = True
        invalidate_index()


def save() -> None:
    """原子落盘：先写临时文件再替换，避免中途失败损坏索引。"""
    global _index
    with _lock:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        _index = _build_index([r.vector for r in _chunks])
        if _index is not None:
            tmp = INDEX_PATH.with_suffix(".index.tmp")
            _import_faiss().write_index(_index, str(tmp))
            tmp.replace(INDEX_PATH)
        elif INDEX_PATH.exists():
            INDEX_PATH.unlink()
        payload = {"dim": len(_chunks[0].vector) if _chunks else 0,
                   "records": [r.to_json() for r in _chunks]}
        tmp_meta = META_PATH.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_meta.replace(META_PATH)
        invalidate_index()


# --------------------------------------------------------------------------
# 查询与统计
# --------------------------------------------------------------------------
def stats() -> dict[str, Any]:
    load()
    return {
        "backend": "faiss(IndexFlatIP, cosine)",
        "chunks": len(_chunks),
        "sources": len(list_sources()),
        "index_path": str(INDEX_PATH),
        "embed_model": config.EMBED_MODEL_NAME,
        "dim": len(_chunks[0].vector) if _chunks else 0,
    }


def list_sources() -> list[dict[str, Any]]:
    load()
    counts: dict[str, int] = {}
    for rec in _chunks:
        src = str(rec.metadata.get("source") or "未知")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": k, "chunks": v} for k, v in sorted(counts.items())]


def search(embedding: list[float], top_k: int, source_filter: str | None = None) -> list[tuple[int, float]]:
    """返回 [(记录下标, 余弦相似度)]，按相似度降序。"""
    import numpy as np

    load()
    if _index is None or not _chunks:
        return []
    k = min(max(top_k, 1) * (5 if source_filter else 1), len(_chunks))
    query = np.asarray([embedding], dtype="float32")
    _import_faiss().normalize_L2(query)
    sims, idxs = _index.search(query, k)
    out: list[tuple[int, float]] = []
    for pos, sim in zip(idxs[0].tolist(), sims[0].tolist()):
        if pos < 0 or pos >= len(_chunks):
            continue
        rec = _chunks[pos]
        if source_filter and str(rec.metadata.get("source")) != source_filter:
            continue
        out.append((pos, float(sim)))
        if len(out) >= top_k:
            break
    return out


def get_record(pos: int) -> Record:
    load()
    return _chunks[pos]


def all_records() -> list[Record]:
    load()
    return list(_chunks)


def add_records(records: Iterable[Record]) -> int:
    load()
    with _lock:
        existing = {r.chunk_id for r in _chunks}
        added = 0
        for rec in records:
            if rec.chunk_id in existing:  # 同内容重复入库时覆盖旧向量
                for i, old in enumerate(_chunks):
                    if old.chunk_id == rec.chunk_id:
                        _chunks[i] = rec
                        break
            else:
                _chunks.append(rec)
                existing.add(rec.chunk_id)
            added += 1
        save()
    return added


def delete_source(source: str) -> int:
    global _chunks
    load()
    with _lock:
        before = len(_chunks)
        _chunks = [r for r in _chunks if str(r.metadata.get("source")) != source]
        removed = before - len(_chunks)
        if removed:
            # 记录顺序变了，重建 ordinal，避免检索配额逻辑读错下标
            for i, rec in enumerate(_chunks):
                rec.metadata["ordinal"] = i
            save()
        return removed


def reset_all() -> None:
    global _chunks, _index
    with _lock:
        _chunks = []
        _index = None
        for path in (INDEX_PATH, META_PATH):
            if path.exists():
                path.unlink()
        invalidate_index()


# --------------------------------------------------------------------------
# 入库流程
# --------------------------------------------------------------------------
def _to_records(chunks: list[Chunk], vectors: list[list[float]]) -> list[Record]:
    return [
        Record(chunk_id=c.chunk_id, text=c.text, metadata=c.metadata, vector=v)
        for c, v in zip(chunks, vectors)
    ]


def ingest_paths(paths: list[Path], on_progress: Any | None = None) -> dict[str, Any]:
    """解析 → 切分 → 向量化 → 入库。逐文件返回结果，单文件失败不影响其他文件。"""
    from .embedding import embedder

    results: list[dict[str, Any]] = []
    total = 0
    for path in paths:
        entry: dict[str, Any] = {"file": path.name, "chunks": 0, "status": "ok"}
        try:
            chunks = prepare_chunks(path)
            if not chunks:
                entry["status"] = "empty"
                entry["message"] = "未解析出文本（可能是扫描件或空文件）"
            else:
                delete_source(path.name)  # 同一文件重建，先清旧块
                vectors = embedder.embed_documents([c.text for c in chunks])
                added = add_records(_to_records(chunks, vectors))
                entry["chunks"] = added
                total += added
                if on_progress:
                    on_progress(entry)
        except Exception as exc:
            entry["status"] = "error"
            entry["message"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
    return {"files": results, "total_chunks": total}


def ingest_source_dir() -> dict[str, Any]:
    files = list(iter_source_files())
    return ingest_paths(files) | {"scanned": [f.name for f in files]}
