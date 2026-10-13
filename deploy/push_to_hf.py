"""一键部署到 Hugging Face Spaces。

用法：
    set HF_TOKEN=hf_xxxxxxxx
    .venv\\Scripts\\python.exe deploy\\push_to_hf.py --space <用户名>/<空间名>

脚本做四件事：
  1. 校验 Token 与空间名；
  2. 把 Space 所需的文件**临时复制**到一个 staging 目录（不污染项目目录）；
  3. 写入 Space 专用的 README（含 YAML frontmatter，这是 HF Spaces 识别应用的必需文件）；
  4. 上传并轮询构建状态，最后打印公开访问地址。

设计说明：本脚本只读取项目文件、只写入 staging 目录，不修改、不删除任何项目文件。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://huggingface.co/api"

# 需要上传到 Space 的文件与目录（相对项目根目录）
INCLUDE_DIRS = ["app", "tools", "data/source", "deploy"]
INCLUDE_FILES = ["requirements.txt", "Dockerfile", ".dockerignore", "verify.py"]

SPACE_README = """---
title: 企业知识库 Demo
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: 中文 RAG 知识库：混合检索 + Cross-Encoder 重排 + 答案溯源
---

# 企业知识库 Demo（在线体验）

一套**完全自建、可解释**的中文 RAG 知识库：上传文档 → 提问 → 得到**带原文出处**的回答。

完整源码与设计文档：[GitHub - Knowledge-Base.demo](https://github.com/zhengweilin142/Knowledge-Base.demo)

## 怎么玩

1. 左侧已内置 4 份示例文档（员工手册、差旅制度、信息安全规范、供应商清单）；
   也可以自己上传 PDF / Word / Excel / Markdown。
2. 在下方输入问题，例如：
   - `年假有多少天`
   - `一线城市差旅住宿标准是多少`
   - `涉密文件能不能用微信传`
   - `SUP-002 是哪家公司`
3. 回答里每个 `[编号]` 都可以点击，展开对应原文核对。

试试问一个**知识库里没有的问题**（如「今天食堂吃什么菜」）——
系统会明确拒答，而不是编造。这是设计目标，不是故障。

## 关于回答质量

本空间**内置一个规则式演示模型**（不冒充实测大模型效果）：它只从检索到的原文里
抽取句子并标注来源，因此引用是真实可核对的，但不会做归纳与推理。

想体验真实大模型的归纳能力：点右上角「设置」→ 填入你自己的
[DeepSeek](https://platform.deepseek.com/) 或任意 OpenAI 兼容接口的 API Key。
Key 只保存在本空间的运行实例内，不经过任何第三方。

## 技术架构

```
提问 → 向量检索(FAISS+BGE中文) + BM25关键词检索 → RRF融合
     → Cross-Encoder重排(BAAI/bge-reranker-base) → 相关性闸门
     → 带编号参考资料 → 大模型 → 带[编号]引用的回答
```

**两段式检索是精度达标的关键**：实测中无关问题「北京到上海高铁多久」的余弦相似度
（0.637）反而高于真相关问题「ISO27001是什么认证」（0.454），单一阈值无法区分；
加入重排后两组分数完全不重叠。详见仓库 README 第四节的实测数据。

## 注意

- 免费实例在闲置后会自动休眠，再次访问需等待约 30 秒唤醒。
- 容器重启后索引会重建（示例文档约 30 秒），**用户上传的文档不会保留**。
"""


def hf_request(path: str, token: str, method: str = "GET", data: bytes | None = None, ctype: str | None = None):
    req = urllib.request.Request(
        API + path,
        method=method,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "kb-deploy-script",
            **( {"Content-Type": ctype} if ctype else {} ),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(body) if body.strip().startswith(("{", "[")) else body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


import json  # noqa: E402  （放在 hf_request 之后仅为可读性）


def whoami(token: str) -> str | None:
    status, data = hf_request("/whoami-v2", token)
    if status == 200 and isinstance(data, dict):
        return data.get("name")
    print(f"  ✗ Token 校验失败 (HTTP {status}): {str(data)[:200]}")
    return None


def ensure_repo(space: str, token: str) -> bool:
    status, data = hf_request(f"/spaces/{space}", token)
    if status == 200:
        print(f"  空间已存在：{space}")
        return True
    if status != 404:
        print(f"  ! 查询空间返回 HTTP {status}: {str(data)[:200]}")

    print(f"  创建空间 {space} ...")
    payload = json.dumps({
        "name": space.split("/")[-1],
        "type": "space",
        "sdk": "docker",
        "private": False,
        "hardware": "cpu-basic",
    }).encode()
    status, data = hf_request("/repos/create", token, "POST", payload, "application/json")
    if status in (200, 201):
        print("  ✓ 空间创建成功")
        return True
    print(f"  ✗ 创建失败 (HTTP {status}): {str(data)[:300]}")
    return False


def stage(space: str) -> tuple[Path, list[Path]]:
    """把要上传的文件复制到临时目录，返回（临时目录, 已创建条目列表）。"""
    staging = ROOT / ".deploy-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    created: list[Path] = []
    for name in INCLUDE_DIRS:
        src = ROOT / name
        if not src.exists():
            print(f"  ! 跳过不存在的目录: {name}")
            continue
        dst = staging / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        created.append(dst)
        count = sum(1 for _ in dst.rglob("*") if _.is_file())
        print(f"  + {name}/  ({count} 个文件)")

    for name in INCLUDE_FILES:
        src = ROOT / name
        if not src.exists():
            print(f"  ! 跳过不存在的文件: {name}")
            continue
        dst = staging / name
        shutil.copy2(src, dst)
        created.append(dst)
        print(f"  + {name}")

    # Space 专用 README（含 YAML frontmatter，HF 靠它识别应用配置）
    (staging / "README.md").write_text(SPACE_README, encoding="utf-8")
    created.append(staging / "README.md")
    print("  + README.md（Space 专用，含 YAML 配置头）")

    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"  暂存完成，共 {total/1024:.1f} KB")
    return staging, created


def upload(space: str, token: str) -> bool:
    """用 HfApi 上传（比手写 multipart 可靠，且能自动处理 LFS）。"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  ! 未安装 huggingface_hub，正在安装 ...")
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(ROOT / ".deploy-staging"),
        repo_id=space,
        repo_type="space",
        commit_message="deploy: 企业知识库 Demo（两段式 RAG）",
    )
    return True


def poll_build(space: str, token: str, timeout: float = 1500.0) -> str:
    """轮询 Space 运行状态，返回最终 stage。"""
    print("\n  轮询构建状态（首次构建需安装依赖，通常 3~8 分钟）...")
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        status, data = hf_request(f"/spaces/{space}", token)
        if status == 200 and isinstance(data, dict):
            stage = data.get("runtime", {}).get("stage", "UNKNOWN")
            if stage != last:
                print(f"    [{int(time.time()-t0):>4}s] {stage}")
                last = stage
            if stage in ("RUNNING", "RUNNING_BUILDING", "BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR", "PAUSED"):
                return stage
        time.sleep(15)
    return "TIMEOUT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="形如 用户名/空间名")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--no-poll", action="store_true", help="上传后不等待构建")
    args = parser.parse_args()

    token = args.token.strip()
    if not token:
        print("✗ 未提供 Token。请设置环境变量 HF_TOKEN 或传入 --token")
        return 1
    if "/" not in args.space:
        print("✗ 空间名格式应为 用户名/空间名")
        return 1

    print("=" * 62)
    print("部署到 Hugging Face Spaces")
    print("=" * 62)

    print("\n[1/5] 校验 Token")
    user = whoami(token)
    if not user:
        return 1
    print(f"  ✓ 登录账号: {user}")

    owner = args.space.split("/")[0]
    if owner.lower() != user.lower():
        print(f"  ! 空间所有者 ({owner}) 与登录账号 ({user}) 不同，通常是组织空间；继续。")

    print("\n[2/5] 确认空间存在")
    if not ensure_repo(args.space, token):
        return 1

    print("\n[3/5] 暂存待上传文件")
    staging, created = stage(args.space)

    try:
        print("\n[4/5] 上传")
        upload(args.space, token)
        print("  ✓ 上传完成")

        if not args.no_poll:
            print("\n[5/5] 等待构建")
            stage_name = poll_build(args.space, token)
            icon = {"RUNNING": "✓", "RUNNING_BUILDING": "…"}.get(stage_name, "✗")
            print(f"  {icon} 最终状态: {stage_name}")

        print("\n" + "=" * 62)
        print(f"  访问地址: https://huggingface.co/spaces/{args.space}")
        print(f"  直接访问: https://{args.space.replace('/', '-').lower()}.hf.space")
        print(f"  构建日志: https://huggingface.co/spaces/{args.space}?logs=container")
        print("=" * 62)
    finally:
        # 清理暂存目录（只删本脚本自己创建的 .deploy-staging，绝不触碰项目文件）
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
            print(f"\n  已清理暂存目录 {staging.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
