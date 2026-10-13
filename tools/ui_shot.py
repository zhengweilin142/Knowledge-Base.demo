"""驱动无头 Chrome 在页面上真实提一个问题，并截图。

为什么需要它：仅靠"代码看起来对"无法证明前端的
「输入 → SSE 流式接收 → 引用面板渲染 → 逐字显示」真的可用。
本脚本通过 Chrome DevTools Protocol 打开页面、在输入框里填入问题、
触发发送、等待回答流式渲染完成，然后截图留证。

用法：
    python tools/ui_shot.py <问题> <输出png路径>
需要 Chrome 以 --remote-debugging-port=9222 --headless=new 启动。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

from websockets.sync.client import connect

CDP_HTTP = "http://127.0.0.1:9222"


def page_target() -> str:
    with urllib.request.urlopen(f"{CDP_HTTP}/json", timeout=10) as resp:
        targets = json.loads(resp.read().decode("utf-8"))
    for t in targets:
        if t.get("type") == "page":
            return t["webSocketDebuggerUrl"]
    raise RuntimeError("没有找到可用的 page 目标")


class CDP:
    def __init__(self, url: str) -> None:
        self.ws = connect(url, max_size=64 * 1024 * 1024, open_timeout=20)
        self.seq = 0

    def call(self, method: str, params: dict | None = None, timeout: float = 60.0):
        self.seq += 1
        msg_id = self.seq
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.ws.recv(timeout=max(1.0, deadline - time.time()))
            data = json.loads(raw)
            if data.get("id") == msg_id:
                if "error" in data:
                    raise RuntimeError(f"{method} 失败: {data['error']}")
                return data.get("result", {})
        raise TimeoutError(f"{method} 超时")

    def eval(self, expression: str, timeout: float = 60.0):
        res = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            timeout,
        )
        return res.get("result", {}).get("value")

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "一线城市差旅住宿标准是多少"
    out = sys.argv[2] if len(sys.argv) > 2 else r"E:\deepseek\kb\logs\ui-chat.png"

    cdp = CDP(page_target())
    try:
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")

        # 等待页面脚本就绪（init() 会先拉 health 与文档列表）
        for _ in range(40):
            ready = cdp.eval(
                "typeof ask === 'function' && "
                "!!document.getElementById('docList').querySelector('.doc, .empty')"
            )
            if ready:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("页面未就绪")

        docs = cdp.eval("document.querySelectorAll('#docList .doc').length")
        print(f"  页面已就绪，文档列表 {docs} 项")

        cdp.eval(
            "(() => { const q=document.getElementById('q');"
            f" q.value={json.dumps(question, ensure_ascii=False)};"
            " q.dispatchEvent(new Event('input')); return q.value; })()"
        )
        print(f"  已填入问题: {question}")

        cdp.eval("ask()")
        # 等待 .answer 里出现非空且不再是"检索中"
        for _ in range(60):
            state = cdp.eval(
                "(() => { const turns=document.querySelectorAll('.turn');"
                " const last=turns[turns.length-1];"
                " if(!last) return {len:0,srcs:0};"
                " const a=last.querySelector('.answer');"
                " const s=last.querySelectorAll('.srcs .src').length;"
                " return {len:(a?a.textContent.length:0), srcs:s}; })()"
            )
            if state and state.get("len", 0) > 60 and state.get("srcs", 0) > 0:
                break
            time.sleep(0.5)
        print(f"  回答渲染完成：文本 {state.get('len')} 字符，引用卡片 {state.get('srcs')} 个")

        # 展开第一条原文，展示溯源能力
        cdp.eval(
            "(() => { const turns=document.querySelectorAll('.turn');"
            " const last=turns[turns.length-1];"
            " const t=last.querySelector('.toggle'); if(t) t.click(); return true; })()"
        )
        time.sleep(1.0)

        shot = cdp.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        import base64

        with open(out, "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        print(f"  截图已保存: {out}")
        return 0
    finally:
        cdp.close()


if __name__ == "__main__":
    raise SystemExit(main())
