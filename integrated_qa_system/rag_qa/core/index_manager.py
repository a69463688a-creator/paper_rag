"""
增量索引管理器

支持在不重跑全量数据的情况下，仅对新论文进行分块和入库。
同时支持覆盖式重索引（删除旧数据后重新处理同一篇论文）。

用法:
    from rag_qa.core.vector_store import VectorStore
    from rag_qa.core.index_manager import IndexManager

    vs = VectorStore()
    mgr = IndexManager(vs)
    result = mgr.incremental_index("path/to/paper_data")
    print(result)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

# 确保路径可用
_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_qa_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_rag_qa_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base.config import Config
from base.logger import logger
from rag_qa.core.document_processor import (
    get_paper_ids_from_directory,
    load_documents_from_directory,
    process_documents,
    _detect_language,
)
from rag_qa.core.vector_store import VectorStore
from rag_qa.text_spliter import ChineseRecursiveTextSplitter
from rag_qa.paper_data.paper_pdf_loader import PaperPDFLoader
from rag_qa.paper_data.figure_extractor import FigureExtractor
from rag_qa.paper_data.table_extractor import TableExtractor

# AliTextSplitter: 可选语义分块
try:
    from rag_qa.text_spliter.model_text_spliter import AliTextSplitter
except ImportError:
    AliTextSplitter = None

conf = Config()


class IndexManager:
    """增量索引管理器"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def incremental_index(self, data_dir: str) -> Dict:
        """
        增量索引入口：扫描 data_dir 中的 PDF，仅处理未入库的新论文

        :param data_dir: 论文 PDF 目录
        :return: 统计信息 dict
        """
        disk_ids = get_paper_ids_from_directory(data_dir)
        if not disk_ids:
            logger.info("[增量索引] 磁盘上未找到 PDF 文件")
            return {"new": 0, "skipped": 0, "errors": 0, "details": []}

        indexed_ids = self.vector_store.get_indexed_paper_ids()
        new_ids = disk_ids - indexed_ids

        if not new_ids:
            logger.info(
                f"[增量索引] 无新论文 (磁盘 {len(disk_ids)} 篇, "
                f"已索引 {len(indexed_ids)} 篇)"
            )
            return {"new": 0, "skipped": len(disk_ids), "errors": 0, "details": []}

        logger.info(
            f"[增量索引] 发现 {len(new_ids)} 篇新论文: {sorted(new_ids)}"
        )

        stats = {"new": 0, "skipped": len(indexed_ids), "errors": 0, "details": []}

        for paper_id in sorted(new_ids):
            pdf_path = os.path.join(data_dir, f"{paper_id}.pdf")
            if not os.path.exists(pdf_path):
                logger.warning(f"[增量索引] PDF 不存在，跳过: {pdf_path}")
                stats["errors"] += 1
                continue

            try:
                detail = self.index_single_paper(pdf_path)
                stats["new"] += 1
                stats["details"].append(detail)
                logger.info(
                    f"[增量索引] ✓ {paper_id}: "
                    f"文本 {detail['text_chunks']} 块, "
                    f"图表 {detail['figures']} 张, "
                    f"表格 {detail['tables']} 个"
                )
            except Exception as e:
                logger.error(f"[增量索引] ✗ {paper_id} 处理失败: {e}")
                stats["errors"] += 1

        logger.info(
            f"[增量索引] 完成: 新增 {stats['new']}, "
            f"跳过 {stats['skipped']}, 失败 {stats['errors']}"
        )
        return stats

    def index_single_paper(self, pdf_path: str) -> Dict:
        """
        处理并入库单篇论文的完整流程：
        文本分块 + 图表提取 + 表格提取

        :param pdf_path: PDF 文件路径
        :return: 处理统计 dict
        """
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        source = os.path.basename(os.path.dirname(pdf_path)).replace("_data", "")

        # ---- 1. 加载文本 ----
        loader = PaperPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            logger.warning(f"[增量索引] PDF 无文本内容: {pdf_path}")
            return {"paper_id": paper_id, "text_chunks": 0, "figures": 0, "tables": 0}

        doc = docs[0]
        doc.metadata["source"] = source
        doc.metadata["file_path"] = pdf_path
        doc.metadata["paper_id"] = paper_id
        doc.metadata["timestamp"] = datetime.now().isoformat()

        # ---- 2. 分块 ----
        child_chunks = self._chunk_single_document(doc)
        logger.info(f"[增量索引] {paper_id}: 切分得到 {len(child_chunks)} 个子块")

        # ---- 3. 入库文本 ----
        self.vector_store.add_documents(child_chunks)
        text_count = len(child_chunks)

        # ---- 4. 提取 & 入库图表 ----
        figure_count = 0
        try:
            fig_extractor = FigureExtractor()
            figures = fig_extractor.extract_from_pdf(pdf_path)
            if figures:
                figures = fig_extractor.describe_figures(figures)
                self.vector_store.add_figures(figures)
                figure_count = len(figures)
        except Exception as e:
            logger.warning(f"[增量索引] {paper_id} 图表提取失败: {e}")

        # ---- 5. 提取 & 入库表格 ----
        table_count = 0
        try:
            tbl_extractor = TableExtractor()
            tables = tbl_extractor.extract_from_pdf(pdf_path)
            if tables:
                self.vector_store.add_tables(tables)
                table_count = len(tables)
        except Exception as e:
            logger.warning(f"[增量索引] {paper_id} 表格提取失败: {e}")

        return {
            "paper_id": paper_id,
            "text_chunks": text_count,
            "figures": figure_count,
            "tables": table_count,
        }

    def _chunk_single_document(self, doc: Document) -> List[Document]:
        """
        对单篇论文 Document 执行 parent-child 分块

        复用 process_documents 中的分块器选择和 chunk 逻辑
        """

        # 初始化分块器（与 process_documents 保持一致）
        parent_splitter = ChineseRecursiveTextSplitter(
            chunk_size=conf.PARENT_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP
        )
        child_splitter = ChineseRecursiveTextSplitter(
            chunk_size=conf.CHILD_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP
        )
        english_parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=conf.PARENT_CHUNK_SIZE,
            chunk_overlap=conf.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        english_child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=conf.CHILD_CHUNK_SIZE,
            chunk_overlap=conf.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        file_path = doc.metadata.get("file_path", "")
        file_extension = os.path.splitext(file_path)[1].lower()

        # 选择分块器
        if file_extension == ".md":
            parent_st = MarkdownTextSplitter(
                chunk_size=conf.PARENT_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP
            )
            child_st = MarkdownTextSplitter(
                chunk_size=conf.CHILD_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP
            )
            splitter_label = "markdown"
        else:
            lang = _detect_language(doc.page_content)
            if lang == "en":
                parent_st = english_parent_splitter
                child_st = english_child_splitter
                splitter_label = "english"
            elif conf.USE_SEMANTIC_SPLITTER and AliTextSplitter is not None:
                parent_st = AliTextSplitter(pdf=(file_extension == ".pdf"))
                child_st = child_splitter
                splitter_label = "semantic_chinese"
            else:
                parent_st = parent_splitter
                child_st = child_splitter
                splitter_label = "chinese"

        logger.info(
            f"[增量索引] 论文: {file_path}, 切分器: {splitter_label}"
        )

        # Parent-child chunking
        doc_idx = 0  # single doc
        child_chunks = []
        parent_docs = parent_st.split_documents([doc])

        for j, parent_doc in enumerate(parent_docs):
            parent_id = f"doc_{doc_idx}_parent_{j}"
            parent_doc.metadata["parent_id"] = parent_id
            parent_doc.metadata["content"] = parent_doc.page_content

            sub_chunks = child_st.split_documents([parent_doc])
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                child_chunks.append(sub_chunk)

        return child_chunks

    def reindex_paper(self, pdf_path: str) -> Dict:
        """
        覆盖式重索引：先删除旧数据，再重新处理

        :param pdf_path: PDF 文件路径
        :return: 处理统计 dict
        """
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info(f"[增量索引] 覆盖重索引: {paper_id}")
        self.vector_store.delete_paper_data(paper_id)
        return self.index_single_paper(pdf_path)


# ==================== CLI ====================

if __name__ == "__main__":
    data_dir = os.path.join(conf.DATA_DIR, "paper_data")
    vs = VectorStore()
    mgr = IndexManager(vs)
    result = mgr.incremental_index(data_dir)
    print(f"\n结果: {result}")
