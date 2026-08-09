"""
混合检索权重网格搜索 — 使用 Ragas 寻找最优 dense/sparse 权重

用法:
    cd integrated_qa_system
    D:/conda_envs/PaperRAG-GPU/python rag_qa/core/grid_search_weights.py

输出:
    最优 (dense_weight, sparse_weight) 组合及对应 context_relevancy 分数
"""

import os, sys, json, itertools, time
from datetime import datetime
from collections import defaultdict

_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_qa_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_rag_qa_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base.config import Config
from base.logger import logger
from rag_qa.core.vector_store import VectorStore

# ── 论文库专用查询 ──
# 每条查询以问句形式，目标命中特定论文的内容
# expected_paper: 预期应该检索到的 arXiv ID
PAPER_QUERIES = [
    # Transformer (1706.03762) - Attention Is All You Need
    {
        "question": "Transformer 的 self-attention 机制如何替代 RNN 的序列建模？",
        "paper_id": "1706.03762",
    },
    {
        "question": "Scaled Dot-Product Attention 的公式是什么？为什么需要除以 sqrt(dk)？",
        "paper_id": "1706.03762",
    },
    # WGAN (1701.07875)
    {
        "question": "WGAN 相比原始 GAN 在训练稳定性上做了哪些改进？",
        "paper_id": "1701.07875",
    },
    {
        "question": "Wasserstein 距离在 WGAN 中是如何近似计算的？",
        "paper_id": "1701.07875",
    },
    # GPT-3 (2005.14165)
    {
        "question": "GPT-3 的 few-shot learning 能力如何随模型参数规模变化？",
        "paper_id": "2005.14165",
    },
    {
        "question": "GPT-3 使用了多大的训练数据集？",
        "paper_id": "2005.14165",
    },
    # GAN (1406.2661)
    {
        "question": "GAN 的生成器和判别器分别使用什么损失函数？",
        "paper_id": "1406.2661",
    },
    # ResNet (1512.03385)
    {
        "question": "ResNet 的残差块是如何解决深层网络梯度消失问题的？",
        "paper_id": "1512.03385",
    },
    # BERT (1810.04805)
    {
        "question": "BERT 的 Masked Language Model 预训练任务具体如何实现？",
        "paper_id": "1810.04805",
    },
    # ViT (2010.11929)
    {
        "question": "Vision Transformer 如何将二维图像转换为一维序列输入？",
        "paper_id": "2010.11929",
    },
    # DQN (1312.5602)
    {
        "question": "DQN 如何使用经验回放和 target network 来稳定训练？",
        "paper_id": "1312.5602",
    },
    # Adam (1412.6980)
    {
        "question": "Adam 优化器结合了哪两种优化方法的优点？",
        "paper_id": "1412.6980",
    },
    # YOLO (1506.02640)
    {
        "question": "YOLO 如何将目标检测转化为一个回归问题？",
        "paper_id": "1506.02640",
    },
    # Faster R-CNN (1506.01497)
    {
        "question": "Faster R-CNN 的 Region Proposal Network 是如何工作的？",
        "paper_id": "1506.01497",
    },
    # VGG (1409.1556)
    {
        "question": "VGG 网络为什么使用多个小卷积核而不是一个大卷积核？",
        "paper_id": "1409.1556",
    },
    # CycleGAN (1703.10593)
    {
        "question": "CycleGAN 的 cycle consistency loss 如何实现无配对图像翻译？",
        "paper_id": "1703.10593",
    },
    # DDPM/Diffusion (2006.11239)
    {
        "question": "DDPM 扩散模型的前向过程和反向过程分别是什么？",
        "paper_id": "2006.11239",
    },
    # Image Transformer (1802.05751)
    {
        "question": "Image Transformer 如何将自注意力机制应用于图像生成？",
        "paper_id": "1802.05751",
    },
    # EfficientNet (1905.11946)
    {
        "question": "EfficientNet 如何通过复合缩放方法来平衡网络深度、宽度和分辨率？",
        "paper_id": "1905.11946",
    },
    # GAN paper also - DCGAN
    {
        "question": "DCGAN 使用哪些技术来稳定 GAN 的训练过程？",
        "paper_id": "1406.2661",
    },
]

conf = Config()


def run_grid_search():
    vs = VectorStore()
    grid = {
        "dense_weight": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "sparse_weight": [0.1, 0.3, 0.5, 0.7, 1.0],
    }

    # 搜索组合数
    combinations = list(itertools.product(grid["dense_weight"], grid["sparse_weight"]))
    # 过滤：权重比极端偏离的组合意义不大，但全部保留以观测趋势
    total = len(combinations) * len(PAPER_QUERIES)
    logger.info(f"网格: dense {grid['dense_weight']}, sparse {grid['sparse_weight']}")
    logger.info(f"共 {len(combinations)} 种组合 × {len(PAPER_QUERIES)} 条查询 = {total} 次搜索")

    # ── 收集所有检索结果 ──
    # { (dense_w, sparse_w): [(question, paper_ids_from_topK), ...] }
    all_results = {}

    start_time = datetime.now()
    search_count = 0

    for dw, sw in combinations:
        key = (dw, sw)
        all_results[key] = []

        for q_item in PAPER_QUERIES:
            q = q_item["question"]
            try:
                docs = vs.hybrid_search_with_custom_weights(q, k=conf.RETRIEVAL_K, dense_weight=dw, sparse_weight=sw)
                # 提取 top-K 结果中的 paper_id
                paper_ids = [d.metadata.get("paper_id", "") for d in docs]
                paper_ids = [p for p in paper_ids if p]  # 过滤空值
            except Exception as e:
                logger.warning(f"搜索失败 (dw={dw}, sw={sw}, q={q[:30]}...): {e}")
                paper_ids = []

            all_results[key].append((q, paper_ids))
            search_count += 1

    search_elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"检索阶段完成: {search_count} 次搜索, 耗时 {search_elapsed:.1f}s")

    # ── 评估: Paper-level Hit Rate ──
    # 每个查询标注了预期论文，检查预期论文是否在 top-K 检索结果中
    # 指标1: Hit@1 (top-1 命中)
    # 指标2: Hit@K (top-K 中至少1篇命中)
    # 指标3: MRR (Mean Reciprocal Rank of expected paper)
    scores = {}
    eval_start = datetime.now()

    for (dw, sw), items in all_results.items():
        key = (dw, sw)
        hit1 = 0
        hitK = 0
        mrr_sum = 0.0
        total = 0

        for (question, paper_ids), q_item in zip(items, PAPER_QUERIES):
            expected = q_item["paper_id"]
            total += 1

            if paper_ids and paper_ids[0] == expected:
                hit1 += 1
            if expected in paper_ids:
                hitK += 1
            try:
                rank = paper_ids.index(expected) + 1
                mrr_sum += 1.0 / rank
            except ValueError:
                pass

        hit1_rate = hit1 / total if total > 0 else 0
        hitK_rate = hitK / total if total > 0 else 0
        mrr = mrr_sum / total if total > 0 else 0

        # 综合分数: 30% Hit@1 + 30% Hit@K + 40% MRR
        composite = 0.3 * hit1_rate + 0.3 * hitK_rate + 0.4 * mrr
        scores[key] = {
            "hit1": round(hit1_rate, 4),
            "hitK": round(hitK_rate, 4),
            "mrr": round(mrr, 4),
            "composite": round(composite, 4),
        }

    eval_elapsed = (datetime.now() - eval_start).total_seconds()
    logger.info(f"评估阶段完成, 耗时 {eval_elapsed:.1f}s")

    # ── 汇总 ──
    total_elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 80)
    print("网格搜索结果 (Paper-Level Hit Rate)")
    print("=" * 80)
    print(f"{'dense':>6}  {'sparse':>6}  {'Hit@1':>7}  {'Hit@K':>7}  {'MRR':>7}  {'Comp':>7}")
    print("-" * 48)

    # 按 composite 排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)
    for (dw, sw), s in sorted_scores:
        marker = " ← BEST" if (dw, sw) == sorted_scores[0][0] else ""
        print(
            f"{dw:>6.1f}  {sw:>6.1f}  "
            f"{s['hit1']:>7.4f}  {s['hitK']:>7.4f}  "
            f"{s['mrr']:>7.4f}  {s['composite']:>7.4f}{marker}"
        )

    # 保存完整结果
    best_key = sorted_scores[0][0]
    output = {
        "metric": "paper_hit_rate",
        "metrics_detail": {
            "hit1": "Hit@1 — expected paper is top-1 result",
            "hitK": "Hit@K — expected paper is in top-K results",
            "mrr": "Mean Reciprocal Rank of expected paper",
            "composite": "0.3*Hit@1 + 0.3*Hit@K + 0.4*MRR",
        },
        "grid": grid,
        "k": conf.RETRIEVAL_K,
        "num_queries": len(PAPER_QUERIES),
        "num_combinations": len(combinations),
        "scores": {
            f"dw={dw}_sw={sw}": {
                "hit1": s["hit1"], "hitK": s["hitK"],
                "mrr": s["mrr"], "composite": s["composite"],
            }
            for (dw, sw), s in sorted_scores
        },
        "best": {
            "dense_weight": best_key[0],
            "sparse_weight": best_key[1],
            **sorted_scores[0][1],
        },
        "search_time_s": round(search_elapsed, 1),
        "eval_time_s": round(eval_elapsed, 1),
        "total_time_s": round(total_elapsed, 1),
    }
    output_path = os.path.join(os.path.dirname(__file__), "grid_search_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    return output


if __name__ == "__main__":
    run_grid_search()
