"""
重建 paper_rag_figures Collection — 使用 qwen3-vl:8b 重新生成所有图表描述

用法:
    cd integrated_qa_system
    D:/conda_envs/PaperRAG-GPU/python rag_qa/core/rebuild_figures.py

流程:
    1. 清空 paper_rag_figures Collection
    2. 遍历 data/paper_data/*.pdf
    3. 提取图表 → qwen3-vl:8b 描述 → upsert
"""

import os, sys
from datetime import datetime

_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_qa_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_rag_qa_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base.config import Config
from base.logger import logger
from rag_qa.core.vector_store import VectorStore
from rag_qa.paper_data.figure_extractor import FigureExtractor, VISION_MODEL

conf = Config()


def rebuild_figures():
    logger.info("=" * 60)
    logger.info(f"图表重建开始 — 视觉模型: {VISION_MODEL}")
    logger.info("=" * 60)

    vs = VectorStore()
    extractor = FigureExtractor()
    data_dir = os.path.join(conf.DATA_DIR, "paper_data")

    pdf_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pdf")])
    logger.info(f"发现 {len(pdf_files)} 篇论文 PDF")

    # ── 1. 清空 figures collection ──
    try:
        count_before = _count_figures(vs)
        result = vs.client.delete(
            collection_name=vs.figure_collection_name,
            filter="figure_id != ''"
        )
        deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info(f"已清空 figures collection: 删除 {deleted} 条 (原 {count_before} 条)")
    except Exception as e:
        logger.error(f"清空 figures collection 失败: {e}")
        return

    # ── 2. 逐篇重建 ──
    total_figures = 0
    total_failed = 0
    start_time = datetime.now()

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(data_dir, pdf_file)
        paper_id = os.path.splitext(pdf_file)[0]

        try:
            logger.info(f"[{i}/{len(pdf_files)}] 提取: {paper_id}")

            figures = extractor.extract_from_pdf(pdf_path)
            if not figures:
                logger.info(f"  └─ 无图表，跳过")
                continue

            logger.info(f"  └─ 提取到 {len(figures)} 张图，正在描述...")

            # qwen3-vl:8b 生成描述
            figures = extractor.describe_figures(figures)

            # 入库
            vs.add_figures(figures)
            total_figures += len(figures)

            # 统计描述成功数
            described = sum(1 for f in figures if f.get("description") != f.get("caption", ""))
            logger.info(f"     ✓ 入库 {len(figures)} 张 (描述成功 {described})")

        except Exception as e:
            logger.error(f"  ✗ {paper_id} 失败: {e}")
            total_failed += 1

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(
        f"重建完成: {total_figures} 张图, "
        f"失败 {total_failed} 篇, "
        f"耗时 {elapsed:.0f}s"
    )
    logger.info("=" * 60)

    return total_figures, total_failed


def _count_figures(vs: VectorStore) -> int:
    try:
        results = vs.client.query(
            collection_name=vs.figure_collection_name,
            filter="figure_id != ''",
            output_fields=["figure_id"],
            limit=10000
        )
        return len(results)
    except Exception:
        return -1


if __name__ == "__main__":
    rebuild_figures()
