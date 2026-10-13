"""容器启动预热：确保模型已就位、示例文档已入库。

为什么需要
----------
模型合计约 1.2GB（向量 90MB + 重排 1.07GB）。若不预热：
  * 第一个访问者要等 1~3 分钟（下载 + 加载），体验极差，且容易误判为"服务挂了"；
  * 重排模型加载失败会让检索精度静默降级。
所以在启动 HTTP 服务之前，先把模型拉全、把示例文档入库。

设计原则：**预热失败不阻断启动**。模型下载可能因网络波动失败，
此时服务仍应起来（页面会明确显示降级状态），而不是整个容器退出。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402


def step(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def seed_source_documents() -> int:
    """把镜像自带的示例文档播种到数据目录。

    为什么需要：部署时 KB_DATA_DIR 会被指向可写数据卷（如 /data），
    数据卷会**遮挡**镜像里原本的 data/source 目录，导致示例文档"消失"、
    公开演示变成空知识库。这里在启动时补齐，使部署自愈。

    只在目标目录为空时播种，绝不影响用户自己上传的文档。
    """
    repo_source = config.BASE_DIR / "data" / "source"
    target = config.SOURCE_DIR
    if repo_source.resolve() == target.resolve():
        return 0  # 本地开发：数据目录就是仓库目录，无需播种

    existing = list(target.glob("*")) if target.exists() else []
    if existing:
        print(f"  数据目录已有 {len(existing)} 个文件，跳过播种", flush=True)
        return 0
    if not repo_source.exists():
        print(f"  ! 镜像内未找到示例文档目录 {repo_source}", flush=True)
        return 0

    target.mkdir(parents=True, exist_ok=True)
    import shutil

    copied = 0
    for src in sorted(repo_source.iterdir()):
        if src.is_file():
            shutil.copy2(src, target / src.name)
            copied += 1
            print(f"  + 播种示例文档 {src.name}", flush=True)
    print(f"  已播种 {copied} 个示例文档到 {target}", flush=True)
    return copied


def main() -> int:
    step("知识库启动预热")
    print(f"  数据目录 : {config.DATA_DIR}", flush=True)
    print(f"  模型目录 : {config.MODELS_DIR}", flush=True)
    print(f"  文档目录 : {config.SOURCE_DIR}", flush=True)

    ok = True

    # ---- 1. 向量模型 ----
    step("[1/3] 加载向量模型")
    t0 = time.time()
    try:
        from app.embedding import embedder

        embedder.warmup()
        print(f"  ✓ {config.EMBED_MODEL_NAME} 就绪，耗时 {time.time() - t0:.1f}s", flush=True)
    except Exception as exc:
        ok = False
        print(f"  ✗ 向量模型加载失败: {type(exc).__name__}: {exc}", flush=True)

    # ---- 2. 重排模型 ----
    step("[2/3] 加载重排模型")
    if not config.RERANK_ENABLED:
        print("  已通过 KB_RERANK=0 关闭，跳过", flush=True)
    else:
        t0 = time.time()
        try:
            from app.rerank import reranker

            reranker.warmup()
            if reranker.ready:
                print(f"  ✓ {config.RERANK_MODEL} 就绪，耗时 {time.time() - t0:.1f}s", flush=True)
            else:
                ok = False
                print(f"  ✗ 重排模型不可用（将降级为余弦门槛）: {reranker.error}", flush=True)
        except Exception as exc:
            ok = False
            print(f"  ✗ 重排模型加载异常: {type(exc).__name__}: {exc}", flush=True)

    # ---- 3. 播种示例文档 + 入库 ----
    step("[3/3] 检查示例文档索引")
    seed_source_documents()
    try:
        from app import kb
        from app.ingest import iter_source_files

        files = list(iter_source_files())
        current = kb.stats()
        print(f"  文档目录: {len(files)} 个文件；当前索引: {current['chunks']} 块", flush=True)

        if current["chunks"] == 0 and files:
            t0 = time.time()
            result = kb.ingest_paths(files)
            print(f"  ✓ 首次入库 {result['total_chunks']} 块，耗时 {time.time() - t0:.1f}s", flush=True)
            for entry in result["files"]:
                if entry["status"] != "ok":
                    print(f"      ! {entry['file']}: {entry.get('message', entry['status'])}", flush=True)
        elif current["chunks"] == 0:
            print("  ! 没有任何示例文档，页面会提示先上传", flush=True)
        else:
            print("  ✓ 索引已存在，跳过入库", flush=True)
    except Exception as exc:
        print(f"  ✗ 入库检查失败: {type(exc).__name__}: {exc}", flush=True)

    step("预热结束" + ("" if ok else "（存在失败项，服务将以降级模式启动）"))
    return 0  # 始终返回 0，保证服务能起来


if __name__ == "__main__":
    raise SystemExit(main())
