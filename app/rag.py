"""RAG 问答：拼接带编号的参考资料 → 调用 OpenAI 兼容接口 → 流式返回。"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

import httpx

from . import config
from .retrieval import Hit


class MissingKeyError(RuntimeError):
    pass


def build_context(hits: Iterable[Hit]) -> tuple[str, list[dict[str, Any]]]:
    """把召回结果编号化，返回（上下文文本, 引用清单）。"""
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        blocks.append(f"[{idx}] 来源：{hit.source_label}\n{hit.text.strip()}")
        citations.append(
            {
                "index": idx,
                "source": hit.source_label,
                "file": (hit.metadata or {}).get("source"),
                "page": (hit.metadata or {}).get("page"),
                "section": (hit.metadata or {}).get("section"),
                # relevance 为余弦相似度（可为 None）；fusion 仅用于排序，跨查询不可比较；
                # rerank 为 Cross-Encoder 重排分，是最终相关性判据（越大越相关）
                "relevance": round(hit.score, 4) if hit.score is not None else None,
                "fusion": round(hit.fusion_score, 5),
                "rerank": round(hit.rerank_score, 4) if hit.rerank_score is not None else None,
                "vector_rank": hit.vector_rank,
                "bm25_rank": hit.bm25_rank,
                "text": hit.text.strip(),
            }
        )
    return "\n\n---\n\n".join(blocks), citations


def build_messages(question: str, hits: list[Hit], history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    context, _ = build_context(hits)
    user = (
        f"参考资料如下：\n\n{context}\n\n"
        f"=====\n\n请依据上述参考资料回答这个问题：{question}"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


def api_key() -> str:
    if not config.LLM_API_KEY:
        raise MissingKeyError(
            "尚未配置大模型 API Key。请在页面「设置」里填入 DeepSeek API Key，"
            "或直接写入 E:\\deepseek\\kb\\.env 的 LLM_API_KEY。"
        )
    return config.LLM_API_KEY


async def stream_answer(question: str, hits: list[Hit], history: list[dict[str, str]] | None = None) -> AsyncIterator[str]:
    """以 SSE 文本增量形式产出回答。"""
    key = api_key()
    payload = {
        "model": config.LLM_MODEL,
        "messages": build_messages(question, hits, history),
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"大模型接口返回 {resp.status_code}：{body[:500]}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece


async def complete_answer(question: str, hits: list[Hit], history: list[dict[str, str]] | None = None) -> str:
    """非流式版本，供脚本/接口调用与自动化测试使用。"""
    key = api_key()
    payload = {
        "model": config.LLM_MODEL,
        "messages": build_messages(question, hits, history),
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"大模型接口返回 {resp.status_code}：{resp.text[:500]}")
        data = resp.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
