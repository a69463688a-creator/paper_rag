"""
首字延迟(TTFT) + 端到端延迟 + 分阶段耗时 基准测试

用法（在 paperrag-app 容器内）:
    cd /app && python scripts/benchmark_latency.py

输出:
    - ttft_s: 查询发出 -> 首个 answer token 到达（首字延迟）
    - total_s: 端到端总耗时（含完整流式生成）
    - 分阶段: bert_classify / strategy_select / retrieve_rerank / llm_ttft(推导)
    - 结果写入 benchmark_result.json
"""
import time, json, statistics, os, sys
# 让脚本从任意目录运行都能 import new_main（指向 integrated_qa_system/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_main import IntegratedQASystem

# 覆盖 4 种检索策略 + 图表/表格召回的测试集
QUERIES = [
    "Transformer 的 self-attention 机制是如何工作的？",      # 直接检索
    "自注意力机制相比 RNN 有哪些优势？",                      # HyDE
    "比较 Transformer 和 LSTM 在机器翻译上的性能差异",        # 子查询
    "如果把 Transformer 编码器从 6 层增加到 12 层会怎样？",   # 回溯
    "Figure 1 展示了什么实验结果？",                          # 图表
    "Table 2 的实验数据对比结果是什么？",                     # 表格
]


def measure_one(qa, query):
    """单独测一次 BM25 缓存检索耗时（管线第 2 阶段）"""
    t0 = time.perf_counter()
    qa.bm25_search.search(query, threshold=0.85)
    return time.perf_counter() - t0


def run_query(qa, query):
    bm25_s = measure_one(qa, query)

    t0 = time.perf_counter()
    ttft = total = None
    for item in qa.query(query, session_id=None):
        token, is_complete, _ = item if len(item) == 3 else (item[0], item[1], None)
        if token and ttft is None:
            ttft = time.perf_counter() - t0   # 首个真实 token 到达
        if is_complete:
            total = time.perf_counter() - t0
            break

    stages = dict(getattr(qa.rag_system, "last_stage_times", {}))
    # LLM 首 token = TTFT - (前面各阶段的叠加)
    known = bm25_s + stages.get("bert_classify", 0) + stages.get("strategy_select", 0) + stages.get("retrieve_rerank", 0)
    llm_ttft = max(0.0, (ttft or 0) - known)

    return {
        "query": query,
        "bm25_s": round(bm25_s, 3),
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3) if total else None,
        "stages": {k: round(v, 3) for k, v in stages.items()},
        "llm_ttft_s": round(llm_ttft, 3),
    }


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {}
    return {"avg": round(statistics.mean(xs), 3),
            "p50": round(xs[len(xs) // 2], 3),
            "p95": round(xs[int(len(xs) * 0.95)], 3),
            "max": round(max(xs), 3)}


def main():
    qa = IntegratedQASystem()
    print("预热中（加载/预热 GPU 模型 + 建立连接）...")
    run_query(qa, QUERIES[0])

    rows = []
    for q in QUERIES:
        r = run_query(qa, q)
        rows.append(r)
        print(f"[{q[:22]:<24}] TTFT={r['ttft_s']}s  总耗时={r['total_s']}s  阶段={r['stages']}")

    result = {
        "num_queries": len(QUERIES),
        "ttft_s": stats([r["ttft_s"] for r in rows]),
        "total_s": stats([r["total_s"] for r in rows]),
        "stage_avg_s": {
            "bm25": round(statistics.mean(r["bm25_s"] for r in rows), 3),
            "bert_classify": round(statistics.mean(r["stages"].get("bert_classify", 0) for r in rows), 3),
            "strategy_select": round(statistics.mean(r["stages"].get("strategy_select", 0) for r in rows), 3),
            "retrieve_rerank": round(statistics.mean(r["stages"].get("retrieve_rerank", 0) for r in rows), 3),
            "llm_first_token": round(statistics.mean(r["llm_ttft_s"] for r in rows), 3),
        },
        "detail": rows,
    }
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))
    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n结果已保存: benchmark_result.json")


if __name__ == "__main__":
    main()
