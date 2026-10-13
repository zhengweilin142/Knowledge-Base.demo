"""生成一份带中文的多页 PDF，用于验证 PDF 解析与「页码溯源」。

用 Windows 自带中文字体 + reportlab 生成（reportlab 是 office 技能的自带依赖，
若环境没有则退化为英文 PDF，并在输出里说明）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "source"
SRC.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("星海科技信息安全规范", [
        "第一条 本规范适用于公司全体员工及外包人员。",
        "第二条 公司资料分为公开、内部、秘密、绝密四个等级。",
        "第三条 秘密级及以上文件禁止通过个人邮箱、微信、QQ 等外部工具传输。",
    ]),
    ("第二章 账号与口令", [
        "第四条 系统口令长度不得少于12位，须包含大小写字母、数字与符号。",
        "第五条 口令每90天更换一次，且不得与前5次使用过的口令重复。",
        "第六条 禁止在便签、聊天记录中留存口令，禁止多人共用同一账号。",
    ]),
    ("第三章 违规处理", [
        "第七条 违反本规范造成信息泄露的，视情节给予警告直至解除劳动合同。",
        "第八条 涉嫌违法犯罪的，移交司法机关处理，公司保留追偿权利。",
        "第九条 本规范由信息安全部负责解释，自发布之日起施行。",
    ]),
]


def main() -> int:
    out = SRC / "信息安全规范.pdf"
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError:
        print("[跳过] 未安装 reportlab，无法生成中文 PDF")
        return 1

    font_candidates = [
        (r"C:\Windows\Fonts\msyh.ttc", "MSYH"),
        (r"C:\Windows\Fonts\simsun.ttc", "SimSun"),
        (r"C:\Windows\Fonts\simhei.ttf", "SimHei"),
    ]
    registered = None
    for path, name in font_candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                registered = name
                break
            except Exception:
                continue
    if not registered:
        print("[跳过] 未找到可嵌入的中文字体")
        return 1

    c = canvas.Canvas(str(out), pagesize=A4)
    for title, lines in PAGES:
        c.setFont(registered, 16)
        c.drawString(70, 780, title)
        c.setFont(registered, 11)
        y = 740
        for line in lines:
            c.drawString(70, y, line)
            y -= 26
        c.setFont(registered, 9)
        c.drawString(70, 60, "星海科技信息安全部")
        c.showPage()
    c.save()
    print(f"[生成] PDF  : {out.name}  3 页 / {out.stat().st_size:,} 字节（中文，用于验证页码溯源）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
