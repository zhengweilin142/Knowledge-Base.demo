"""演示用大模型：OpenAI 兼容接口，从参考资料里抽取原文作答并标注引用编号。

用途
----
公网演示部署不能挂私人 API Key（会被陌生人刷爆额度）。本服务让公开访问者
在**没有任何 Key** 的情况下也能完整体验「检索 → 重排 → 引用溯源」全流程：

  * 它不"编造"内容，而是从检索到的参考资料中挑选与问题最相关的句子，
    逐条附上来源编号 [1][2]，因此展示的引用是真实的、可核对的；
  * 它明确标注自己是规则式演示实现，不冒充大模型效果。

想体验真实大模型的人，可在页面「设置」中填入自己的 API Key——
此时请求会走真实接口，本服务不再参与。

实现要点：参考资料由 app/rag.py 以 `[n] 来源：xxx` 的格式拼入 user 消息，
这里解析回结构化片段，再用字符级重合度挑句子（无需任何模型依赖，启动快、内存低）。
"""
from __future__ import annotations

import json
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="知识库演示模型")

# 解析 app/rag.py 拼出的参考资料块
BLOCK_RE = re.compile(r"^\[(\d+)\]\s*来源：(.+?)\n(.*?)(?=\n\n---\n\n|\Z)", re.S | re.M)
QUESTION_RE = re.compile(r"请依据上述参考资料回答这个问题：(.*?)\s*$", re.S)


def parse_context(text: str) -> tuple[str, list[dict]]:
    """从 user 消息里解析出（问题, 参考资料列表）。"""
    match = QUESTION_RE.search(text)
    question = (match.group(1).strip() if match else "").strip()
    pieces = text.split("请依据上述参考资料回答这个问题：")[0]
    refs: list[dict] = []
    for m in BLOCK_RE.finditer(pieces):
        refs.append({"index": int(m.group(1)), "source": m.group(2).strip(), "text": m.group(3).strip()})
    return question, refs


def _keywords(question: str) -> set[str]:
    """问题里的关键词：中文二元组 + ASCII 词。"""
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9_\-]{2,}", question)}
    for run in re.findall(r"[\u4e00-\u9fff]+", question):
        words.update(run[i : i + 2] for i in range(len(run) - 1))
    return words


_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def _normalize(text: str) -> str:
    """归一化：去掉所有标点、空白、表格分隔符，只留文字与数字。

    这样「600元」与「600 元」、「[表格 1]」与「表格1」都能正确匹配。
    """
    return _PUNCT_RE.sub("", text)


def _is_content_sentence(sentence: str) -> bool:
    """过滤标题、页码、表格表头等无实质信息的碎片。"""
    s = sentence.strip()
    if len(s) < 10:
        return False
    # 纯标题：以「一、」「（三）」「第二章」等开头且很短
    if len(s) < 22 and re.match(r"^(\d+[、.）)]|[一二三四五六七八九十]+[、.]|[（(][一二三四五六七八九十\d]+[）)]|第[一二三四五六七八九十\d]+[章节条])", s):
        return False
    # 表格行：竖线超过 2 个说明是整行表格，缺乏上下文，价值低
    if s.count("|") > 2:
        return False
    # 至少要含一个中文实义字符
    if not re.search(r"[\u4e00-\u9fff]{2,}", s):
        return False
    return True


def _bigrams(text: str) -> set[str]:
    norm = _normalize(text)
    if len(norm) < 2:
        return {norm} if norm else set()
    return {norm[i : i + 2] for i in range(len(norm) - 1)}


def _similar(a: str, b: str, threshold: float = 0.45) -> bool:
    """判断两句是否在讲同一件事。

    子串匹配不够用：实测「超标部分须事先取得…」与「超标部分需事先取得…」
    只差一个字，互为子串判定为 False，同一事实被列了两遍。

    阈值由实测数据确定（字符二元组 Jaccard）：
        同一事实的不同措辞      0.519
        同一事实的不同长度表述  0.793
        不同事实                0.080
    取 0.45 有充分安全余量；同时保留互为子串判定以覆盖"长句含短句"的情况。
    """
    na, nb = _normalize(a), _normalize(b)
    if na and nb and (na in nb or nb in na):
        return True
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga or not gb:
        return False
    return (len(ga & gb) / len(ga | gb)) >= threshold


def _best_sentences(text: str, question: str, limit: int = 2) -> list[str]:
    """按与问题的字符重合度挑选最相关的原文句子。

    分句只用句末标点与换行，**不用分号**：
    制度类文本里分号常连接并列条款（如「年假 5 天；10 天；15 天」），
    用分号切会把完整规定截成半句，丢掉关键信息。
    """
    keys = _keywords(question)
    raw = re.split(r"(?<=[。！？!?])|\n+", text)
    scored: list[tuple[float, int, str]] = []
    for order, sentence in enumerate(raw):
        s = sentence.strip()
        if not _is_content_sentence(s):
            continue
        score = sum(1 for k in keys if k in s)
        # 略微偏好靠前的句子（文档通常先讲结论）
        scored.append((score - order * 0.01, -order, s))
    scored.sort(reverse=True)
    picked: list[str] = []
    for _, _, s in scored:
        if any(_similar(s, p) for p in picked):
            continue
        picked.append(s)
        if len(picked) >= limit:
            break
    return picked


def compose_answer(question: str, refs: list[dict]) -> str:
    """合成回答：按事实聚合，同一事实的多个来源合并成 [1][2] 形式。

    刻意避免两种劣质输出：
      1. 把参考资料原文照抄一遍（读者要自己找答案）；
      2. 同一事实来自两份文档时重复列两遍（看起来像凑数）。
    """
    if not refs:
        return "知识库中没有检索到与问题相关的内容。"

    keys = _keywords(question)
    chosen: list[str] = []             # 已输出的事实
    lines: list[str] = []

    for ref in refs:
        for sentence in _best_sentences(ref["text"], question, limit=2):
            if not _normalize(sentence):
                continue
            # 跨资料来源去重：同一事实换个措辞出现时只保留一次
            if any(_similar(sentence, prev) for prev in chosen):
                continue
            chosen.append(sentence)
            lines.append(sentence)

    # 为每条要点标注实际来源编号：同一事实出现在多份文档时全部标出
    ref_norms = {r["index"]: _normalize(r["text"]) for r in refs}
    rendered: list[str] = []
    for i, sentence in enumerate(lines):
        compact = _normalize(sentence)
        sources = sorted(idx for idx, text in ref_norms.items() if compact and compact in text)
        if not sources:  # 归一化后仍匹配不上（如跨句拼接），退化为按顺序标注
            sources = [refs[min(i, len(refs) - 1)]["index"]]
        marks = "".join(f"[{s}]" for s in sources[:3])
        rendered.append(f"{i + 1}. {sentence} {marks}")

    head = (
        "【演示模式】以下回答由内置规则式模型生成（未连接真实大模型）："
        "它只从检索到的原文中抽取句子并标注来源编号，不做改写与推理，因此不会编造内容。\n\n"
        f"针对「{question}」，从 {len(refs)} 条命中的原文中整理出以下要点：\n"
    )
    body = "\n".join(rendered) if rendered else "（命中片段较短，未能整理出完整要点）"
    tail = (
        "\n\n—— 每条要点后的编号对应上方「命中原文」，点击即可展开核对。\n"
        "需要真正的归纳、对比与推理能力，请在右上角「设置」中填入你自己的 API Key。"
    )
    return head + "\n" + body + tail


@app.get("/")
async def root():
    return {"service": "knowledge-base demo model", "mode": "rule-based", "endpoints": ["/v1/chat/completions"]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})

    messages = body.get("messages") or []
    user_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    question, refs = parse_context(user_text or "")
    answer = compose_answer(question or "（未识别到问题）", refs)

    if not body.get("stream"):
        return {
            "id": "demo-completion",
            "object": "chat.completion",
            "model": body.get("model", "demo-rule-based"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(user_text), "completion_tokens": len(answer), "total_tokens": len(user_text) + len(answer)},
        }

    async def gen():
        # 按句切分流式输出，模拟真实模型的逐字返回
        step = 8
        for i in range(0, len(answer), step):
            chunk = answer[i : i + step]
            payload = {"choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            time.sleep(0.02)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
