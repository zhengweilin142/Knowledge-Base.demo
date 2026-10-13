"""端到端自检：模型 → 解析切分 → 入库 → 混合检索 → 问答。

用法（项目根目录）：
    .venv\\Scripts\\python.exe verify.py            # 在现有数据上巡检（安全，默认）
    .venv\\Scripts\\python.exe verify.py --reset    # 先清空向量库再自检（会丢数据）
    .venv\\Scripts\\python.exe verify.py --no-llm   # 跳过需要 API Key 的检查
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = "[PASS]", "[FAIL]"
_results: list[tuple[bool, str]] = []


def check(name: str, fn):
    try:
        detail = fn()
        _results.append((True, name))
        print(f"{PASS} {name}" + (f"  ->  {detail}" if detail else ""))
        return True
    except Exception as exc:
        _results.append((False, name))
        print(f"{FAIL} {name}  ->  {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="先清空向量库再自检（会丢失现有索引，慎用）")
    parser.add_argument("--no-llm", action="store_true", help="跳过需要 API Key 的问答检查")
    args = parser.parse_args()

    from app import config

    print("=" * 74)
    print("企业知识库 Demo —— 端到端自检")
    print("=" * 74)
    print(f"项目目录 : {config.BASE_DIR}")
    print(f"索引目录 : {config.INDEX_DIR}")
    print(f"文档目录 : {config.SOURCE_DIR}")
    print(f"嵌入模型 : {config.EMBED_MODEL_NAME}")
    print(f"大模型   : {config.LLM_MODEL}  (Key 已配置={bool(config.LLM_API_KEY)})")
    print("-" * 74)

    print("\n【1】依赖与向量化")
    check(
        "导入 fastapi / faiss / fastembed / pypdf / docx / openpyxl",
        lambda: (
            __import__("fastapi"), __import__("faiss"), __import__("fastembed"),
            __import__("pypdf"), __import__("docx"), __import__("openpyxl"),
        ) and "ok",
    )

    from app.embedding import embedder, model_cache_dir

    check(
        "加载 ONNX 中文嵌入模型并预热",
        lambda: (embedder.warmup(), f"{config.EMBED_MODEL_NAME} @ {model_cache_dir()}")[-1],
    )

    def dim_check():
        vec = embedder.embed_query("测试")
        assert vec and all(isinstance(x, float) for x in vec[:5])
        return f"向量维度 {len(vec)}"

    check("查询向量维度与类型正确", dim_check)

    def semantic_check():
        """语义相近的句子相似度应显著高于无关句子——这是检索质量的地基。"""
        import math

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

        docs = embedder.embed_documents(
            ["员工入职满一年后享有带薪年假。", "差旅住宿标准：一线城市每晚不超过600元。"]
        )
        q_leave = embedder.embed_query("年假有多少天")
        q_hotel = embedder.embed_query("出差住酒店能报多少钱")
        pairs = [
            ("年假问题", cos(q_leave, docs[0]), cos(q_leave, docs[1])),
            ("住宿问题", cos(q_hotel, docs[1]), cos(q_hotel, docs[0])),
        ]
        for label, hit, miss in pairs:
            assert hit > miss, f"{label} 语义排序错误 {hit:.4f} <= {miss:.4f}"
        return "  ".join(f"{l}:{h:.4f}>{m:.4f}" for l, h, m in pairs)

    check("语义相似度排序合理", semantic_check)

    print("\n【2】文档解析与切分")
    from app.ingest import parse_file, prepare_chunks

    sample = config.SOURCE_DIR / "员工手册.md"
    if not sample.exists():
        print(f"     提示：{sample} 不存在，跳过解析检查")
    else:
        check(f"解析 {sample.name}", lambda: f"{len(parse_file(sample).text)} 字符")

        def chunk_check():
            chunks = prepare_chunks(sample)
            assert chunks, "没有切出任何分块"
            lens = [len(c.text) for c in chunks]
            assert max(lens) <= config.CHUNK_SIZE * 2, f"存在异常长分块 {max(lens)}"
            for c in chunks:
                assert c.metadata.get("source") == sample.name
                assert "ordinal" in c.metadata
            return f"{len(chunks)} 块，长度 {min(lens)}~{max(lens)} 字符"

        check("切分结果与元数据完整", chunk_check)

    print("\n【3】向量库写入与混合检索")
    from app import kb
    from app.retrieval import retrieve

    if args.reset:
        check("重置索引（--reset）", lambda: (kb.reset_all(), "已清空")[1])
    else:
        print("     保留现有索引（如需清空请加 --reset）")

    if sample.exists():
        def ingest_check():
            out = kb.ingest_paths([sample])
            assert out["total_chunks"] > 0, f"入库失败：{out}"
            return f"写入 {out['total_chunks']} 块"

        check("入库示例文档", ingest_check)

    check("向量库统计", lambda: (
        lambda s: f"{s['chunks']} 块 / {s['sources']} 份文档 / {s['backend']}" if s["chunks"] else (_ for _ in ()).throw(AssertionError("索引为空"))
    )(kb.stats()))

    def roundtrip_check():
        """落盘后重新载入，验证索引与元数据一致（重启不丢数据的保证）。"""
        before = kb.stats()["chunks"]
        kb._loaded = False
        kb._index = None
        kb.load()
        after = kb.stats()["chunks"]
        assert before == after, f"重载后数量不一致 {before} -> {after}"
        return f"重载 {after} 块，索引与元数据一致"

    check("索引落盘/重载一致性", roundtrip_check)

    cases = [
        ("年假有多少天", ["年假"]),
        ("报销超过多少钱需要总监审批", ["5000", "总监"]),
        ("涉密文件能不能用微信传", ["微信"]),
        ("ISO27001", ["ISO27001"]),
    ]

    def retrieval_quality():
        lines, ok = [], 0
        for question, expects in cases:
            hits = retrieve(question, top_k=3)
            assert hits, f"「{question}」没有召回任何内容"
            blob = " ".join(h.text for h in hits)
            matched = any(e in blob for e in expects)
            ok += 1 if matched else 0
            top = hits[0]
            lines.append(
                f"      {question}  ->  {top.source_label}  "
                f"(向量#{top.vector_rank} 关键词#{top.bm25_rank})  {'命中' if matched else '未命中关键词'}"
            )
        print("\n".join(lines))
        assert ok >= 3, f"仅 {ok}/{len(cases)} 命中预期关键词，检索质量不足"
        return f"{ok}/{len(cases)} 命中预期"

    check("检索质量抽查", retrieval_quality)

    def hybrid_check():
        hits = retrieve("年假 报销 审批", top_k=5)
        both = sum(1 for h in hits if h.vector_rank and h.bm25_rank)
        assert hits, "未召回"
        return f"两路同时命中的块数 {both}/{len(hits)}"

    check("RRF 融合可用", hybrid_check)

    def score_semantics_check():
        """score 必须是可解释的余弦相似度，而不是不可跨查询比较的 RRF 融合分。"""
        hits = retrieve("年假有多少天", top_k=3)
        top = hits[0]
        assert top.vector_score is not None, "向量命中却没有余弦相似度"
        assert -1.0 <= top.score <= 1.0, f"相关度超出余弦范围: {top.score}"
        assert top.fusion_score < 0.1, f"融合分应当很小，实际 {top.fusion_score}"
        assert top.score > top.fusion_score, "score 疑似仍是融合分"
        for h in hits:
            assert h.score is None or -1.0 <= h.score <= 1.0
        extra = f"，重排分 {top.rerank_score:.4f}" if top.rerank_score is not None else ""
        return f"余弦 {top.score:.4f} ≠ 融合分 {top.fusion_score:.5f}{extra}"

    check("相关度语义正确", score_semantics_check)

    def relevance_gate_check():
        """无关问题必须被闸门挡住——这是防止模型编造答案的关键闸门。"""
        from app.retrieval import gate_by_relevance

        relevant = gate_by_relevance(retrieve("年假有多少天", top_k=5))
        assert relevant, "相关问题被误杀，门槛过高"
        irrelevant = gate_by_relevance(retrieve("今天食堂午餐吃什么菜", top_k=5))
        assert not irrelevant, f"无关问题仍召回了 {len(irrelevant)} 条，门槛失效"
        mode = "重排分门槛" if any(h.rerank_score is not None for h in relevant) else "余弦门槛(降级)"
        return f"相关问题保留 {len(relevant)} 条，无关问题保留 0 条（{mode}）"

    check("相关性闸门拦截无关问题", relevance_gate_check)

    def hard_negative_check():
        """最难的一个假阳性：文档里有「高铁乘坐二等座」，
        问「北京到上海高铁多久」时 BM25 会给它极高分数。
        纯余弦门槛无法拦住（实测 0.637），重排器应能正确判定为无关。"""
        from app.retrieval import gate_by_relevance

        hits = retrieve("北京到上海高铁多久", top_k=5)
        if not any(h.rerank_score is not None for h in hits):
            return "重排未启用，跳过（余弦门槛下此处会漏拦）"
        kept = gate_by_relevance(hits)
        top_before = max((h.rerank_score for h in hits if h.rerank_score is not None), default=0.0)
        assert not kept, f"仍召回了 {len(kept)} 条（最高重排分 {top_before:.4f}）"
        return f"重排分最高仅 {top_before:.4f}，正确判定为无关"

    check("关键词污染的硬负例被拦截", hard_negative_check)

    def rerank_quality_check():
        """重排器应把真正相关的段落排到最前。"""
        hits = retrieve("差旅住宿标准是多少", top_k=3)
        if not any(h.rerank_score is not None for h in hits):
            return "重排未启用，跳过"
        top = hits[0]
        assert "住宿" in top.text or "600" in top.text, f"重排后首条不相关: {top.text[:40]}"
        return f"首条 {top.source_label} 重排分 {top.rerank_score:.4f}"

    check("重排结果相关性正确", rerank_quality_check)

    def source_filter_check():
        if not sample.exists():
            return "跳过"
        hits = retrieve("年假", top_k=3, source_filter=sample.name)
        assert hits, "限定来源后没有召回"
        assert all(h.metadata.get("source") == sample.name for h in hits), "来源过滤失效"
        return f"限定「{sample.name}」召回 {len(hits)} 块"

    check("按来源过滤检索", source_filter_check)

    print("\n【4】答案生成（需要 API Key）")
    if args.no_llm or not config.LLM_API_KEY:
        print("     跳过：未配置 LLM_API_KEY 或指定了 --no-llm")
        print("     填好 Key 后可重跑本脚本，验证「带引用的回答」这一环。")
    else:
        import asyncio

        from app import rag

        def answer_check():
            hits = retrieve("一线城市差旅住宿标准是多少钱", top_k=3)
            answer = asyncio.run(rag.complete_answer("一线城市差旅住宿标准是多少钱", hits))
            assert answer.strip(), "大模型返回空内容"
            assert "[" in answer, "回答里没有引用编号，溯源约束可能失效"
            print(f"      回答节选：{answer.strip().replace(chr(10),' ')[:160]}")
            return f"{len(answer)} 字符，含引用编号"

        check("调用大模型生成带引用的回答", answer_check)

    print("\n" + "=" * 74)
    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"结果：{passed}/{total} 项通过")
    if passed != total:
        print("未通过项：" + "、".join(name for ok, name in _results if not ok))
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
