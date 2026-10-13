"""FastAPI 应用：知识库管理 + 混合检索问答（SSE 流式）。"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, kb, rag
from .retrieval import gate_by_relevance, retrieve

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="企业知识库 Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(rag.MissingKeyError)
async def missing_key_handler(_request: Any, exc: rag.MissingKeyError) -> JSONResponse:
    """未配置大模型 Key 时给出可读提示，而不是 500。"""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request: Any, exc: RuntimeError) -> JSONResponse:
    """大模型接口报错等运行时问题，转成 502 并保留原因。"""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_handler(_request: Any, exc: Exception) -> JSONResponse:
    print(f"[error] 未处理异常: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务内部错误：{type(exc).__name__}: {exc}"},
    )

_SAFE_NAME = re.compile(r"[^\w\u4e00-\u9fff.\-()\[\] ]+")

# 启动时间与模型状态，供 /api/health 反馈
_state: dict[str, Any] = {
    "started_at": time.time(),
    "model_ready": False,
    "model_error": None,
    "rerank_ready": False,
    "rerank_error": None,
}


def _sanitize(filename: str) -> str:
    name = Path(filename or "未命名").name
    name = _SAFE_NAME.sub("_", name).strip() or "未命名"
    return name[:120]


def _write_env(updates: dict[str, str]) -> None:
    """把设置写回 .env（保留注释与其他键）。"""
    env_file = config.BASE_DIR / ".env"
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    env_file.write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# 请求/响应模型
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None
    source: str | None = None
    history: list[dict[str, str]] | None = None


class SettingsRequest(BaseModel):
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    final_top_k: int | None = None


# --------------------------------------------------------------------------
# 基础接口
# --------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    kb_stats: dict[str, Any] = {}
    try:
        kb_stats = kb.stats()
    except Exception as exc:
        kb_stats["error"] = str(exc)
    return {
        "ok": True,
        "uptime_seconds": round(time.time() - _state["started_at"], 1),
        "model_ready": _state["model_ready"],
        "model_error": _state["model_error"],
        "embed_model": config.EMBED_MODEL_NAME,
        "rerank_enabled": config.RERANK_ENABLED,
        "rerank_ready": _state.get("rerank_ready", False),
        "rerank_error": _state.get("rerank_error"),
        "rerank_model": config.RERANK_MODEL if config.RERANK_ENABLED else None,
        "rerank_min_score": config.RERANK_MIN_SCORE if config.RERANK_ENABLED else None,
        "llm_configured": bool(config.LLM_API_KEY),
        "llm_model": config.LLM_MODEL,
        "kb": kb_stats,
    }


@app.get("/api/documents")
async def documents() -> dict[str, Any]:
    return {"sources": kb.list_sources(), "stats": kb.stats()}


@app.post("/api/documents/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    saved: list[Path] = []
    for item in files:
        name = _sanitize(item.filename or "未命名")
        target = config.SOURCE_DIR / name
        target.write_bytes(await item.read())
        saved.append(target)
    # 解析+向量化是 CPU 密集的，放线程池避免阻塞事件循环
    result = await asyncio.to_thread(kb.ingest_paths, saved)
    result["stats"] = kb.stats()
    return result


@app.post("/api/documents/rebuild")
async def rebuild() -> dict[str, Any]:
    """把 data/source 目录下的全部文档重新入库（新增文件、改了切分参数后用）。"""
    result = await asyncio.to_thread(kb.ingest_source_dir)
    result["stats"] = kb.stats()
    return result


@app.delete("/api/documents/{source}")
async def delete_document(source: str) -> dict[str, Any]:
    removed_chunks = kb.delete_source(source)
    target = config.SOURCE_DIR / Path(source).name
    if target.exists():
        target.unlink()
    return {"removed_chunks": removed_chunks, "stats": kb.stats()}


@app.post("/api/reset")
async def reset() -> dict[str, Any]:
    kb.reset_all()
    return {"ok": True, "stats": kb.stats()}


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    return {
        "llm_configured": bool(config.LLM_API_KEY),
        "llm_model": config.LLM_MODEL,
        "llm_base_url": config.LLM_BASE_URL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "final_top_k": config.FINAL_TOP_K,
        "embed_model": config.EMBED_MODEL_NAME,
        "vector_top_k": config.VECTOR_TOP_K,
        "bm25_top_k": config.BM25_TOP_K,
    }


@app.post("/api/settings")
async def update_settings(req: SettingsRequest) -> dict[str, Any]:
    updates: dict[str, str] = {}
    if req.llm_api_key is not None and req.llm_api_key.strip():
        config.LLM_API_KEY = req.llm_api_key.strip()
        updates["LLM_API_KEY"] = config.LLM_API_KEY
    if req.llm_model:
        config.LLM_MODEL = req.llm_model.strip()
        updates["LLM_MODEL"] = config.LLM_MODEL
    if req.llm_base_url:
        config.LLM_BASE_URL = req.llm_base_url.strip()
        updates["LLM_BASE_URL"] = config.LLM_BASE_URL
    if req.chunk_size:
        config.CHUNK_SIZE = int(req.chunk_size)
        updates["KB_CHUNK_SIZE"] = str(config.CHUNK_SIZE)
    if req.chunk_overlap is not None:
        config.CHUNK_OVERLAP = int(req.chunk_overlap)
        updates["KB_CHUNK_OVERLAP"] = str(config.CHUNK_OVERLAP)
    if req.final_top_k:
        config.FINAL_TOP_K = int(req.final_top_k)
        updates["KB_FINAL_TOPK"] = str(config.FINAL_TOP_K)
    if updates:
        _write_env(updates)
    return {"saved": list(updates), "settings": await get_settings()}


# --------------------------------------------------------------------------
# 检索与问答
# --------------------------------------------------------------------------
def _observe_relevance_rule() -> str:
    """返回当前实际生效的相关性判据描述，让页面提示与真实行为一致。"""
    if config.RERANK_ENABLED and _state.get("rerank_ready"):
        return f"重排分门槛 {config.RERANK_MIN_SCORE}（Cross-Encoder 重排）"
    return f"余弦相似度门槛 {config.MIN_RELEVANCE}（重排未启用）"


def _do_retrieve(req: ChatRequest) -> list[Any]:
    """检索并过相关度门槛；返回空列表表示知识库中没有相关内容。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    hits = retrieve(req.question, req.top_k, req.source)
    if not hits:
        raise HTTPException(
            status_code=404,
            detail="知识库还是空的，请先在左侧上传文档。",
        )
    return gate_by_relevance(hits, config.MIN_RELEVANCE)


@app.post("/api/search")
async def search(req: ChatRequest) -> dict[str, Any]:
    hits = await asyncio.to_thread(_do_retrieve, req)
    _, citations = rag.build_context(hits)
    return {
        "citations": citations,
        "count": len(citations),
        "min_relevance": config.MIN_RELEVANCE,
        "matched": bool(citations),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """SSE 流式问答：先推 sources，再推 delta，最后 done。

    未检索到足够相关的内容时直接停止，不调用大模型——
    这是防止幻觉最有效的一道闸门。
    """
    hits = await asyncio.to_thread(_do_retrieve, req)

    async def event_stream():
        if not hits:
            payload = {
                "message": (
                    "知识库中没有找到与问题相关的内容"
                    f"（{_observe_relevance_rule()}）。"
                    "请确认相关文档已上传，或换一种问法。"
                )
            }
            yield f"event: norefs\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        _, citations = rag.build_context(hits)
        yield f"event: sources\ndata: {json.dumps({'citations': citations}, ensure_ascii=False)}\n\n"
        try:
            async for piece in rag.stream_answer(req.question, hits, req.history):
                yield f"event: delta\ndata: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"
        except rag.MissingKeyError as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': f'生成失败：{exc}'}, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest) -> dict[str, Any]:
    """非流式问答，便于接口联调与自动化测试。"""
    hits = await asyncio.to_thread(_do_retrieve, req)
    if not hits:
        return {
            "answer": (
                "知识库中没有找到与问题相关的内容"
                f"（{_observe_relevance_rule()}）。请确认相关文档已上传，或换一种问法。"
            ),
            "citations": [],
        }
    _, citations = rag.build_context(hits)
    answer = await rag.complete_answer(req.question, hits, req.history)
    return {"answer": answer, "citations": citations}


# --------------------------------------------------------------------------
# 启动
# --------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    def warm() -> None:
        # 向量模型：加载失败则检索完全不可用，页面会明确提示
        try:
            from .embedding import embedder

            embedder.warmup()
            _state["model_ready"] = True
        except Exception as exc:
            _state["model_error"] = str(exc)

        # 重排模型：加载失败只降级精度，不影响服务可用
        if config.RERANK_ENABLED:
            try:
                from .rerank import reranker

                reranker.warmup()
                _state["rerank_ready"] = reranker.ready
                _state["rerank_error"] = reranker.error
            except Exception as exc:
                _state["rerank_error"] = str(exc)

    await asyncio.to_thread(warm)
    if _state["model_ready"]:
        print(f"[startup] 嵌入模型就绪: {config.EMBED_MODEL_NAME}")
    else:
        print(f"[startup] 嵌入模型加载失败: {_state['model_error']}")
    if not config.RERANK_ENABLED:
        print("[startup] 重排已关闭（KB_RERANK=0），仅使用余弦门槛")
    elif _state.get("rerank_ready"):
        print(f"[startup] 重排模型就绪: {config.RERANK_MODEL} (门槛 {config.RERANK_MIN_SCORE})")
    else:
        print(f"[startup] 重排模型不可用，已降级: {_state.get('rerank_error')}")
    print(f"[startup] 向量库: FAISS @ {config.DATA_DIR / 'index'}")
    print(f"[startup] 大模型: {config.LLM_MODEL} 已配置Key={bool(config.LLM_API_KEY)}")
