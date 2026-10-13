"""临时的 OpenAI 兼容 mock 服务，仅用于端到端自测（不参与正式运行）。

存在的意义：在尚未配置真实大模型 Key 的情况下，验证
「检索 → 重排 → 组 prompt → 流式 SSE → 前端渲染引用」这条链路真的通了。
它会把收到的 prompt 中参考资料条数解析出来，据此生成带 [1][2] 引用的回答。
"""
from __future__ import annotations

import json
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    user_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    question = user_text.split("请依据上述参考资料回答这个问题：")[-1].strip()

    # 解析参考资料条数，用于生成对应数量的引用
    refs = sorted({int(n) for n in re.findall(r"^\[(\d+)\] 来源：", user_text, flags=re.M)})
    refs = refs[:3] or [1]
    cite = "".join(f"[{n}]" for n in refs)

    answer = (
        f"（这是本地 mock 模型的回答，用于验证链路，不代表真实大模型效果）\n"
        f"针对你的问题「{question}」，根据检索到的参考资料 {cite}，"
        f"共命中 {len(refs)} 条相关原文。\n"
        f"回答会在这里逐字流出，并在句末标注来源编号 {cite}，"
        f"点击编号可回看对应原文片段。"
    )

    if not body.get("stream"):
        return {
            "choices": [{"message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": len(user_text), "completion_tokens": len(answer)},
        }

    async def gen():
        for i in range(0, len(answer), 4):
            piece = answer[i : i + 4]
            yield f"data: {json.dumps({'choices':[{'delta':{'content':piece}}]}, ensure_ascii=False)}\n\n"
            time.sleep(0.01)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
