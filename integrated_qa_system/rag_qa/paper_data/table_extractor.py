"""
论文 PDF 表格结构化提取模块

从论文 PDF 中提取表格，支持两种表格类型：
1. 内嵌文本表格 → pdfplumber 直接提取
2. 图片型表格 → PaddleOCR PPStructure 识别

提取结果统一转为 Markdown 格式存储，便于后续检索和 LLM 理解。

依赖:
    - pdfplumber: PDF 内嵌表格提取
    - PaddleOCR PPStructure: 图片型表格识别
"""

import os
import hashlib
import json
from typing import List, Dict, Optional
from base.logger import logger


class TableExtractor:
    """
    论文表格提取器

    用法:
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf("paper.pdf")
        for t in tables:
            print(t["markdown"])
    """

    def __init__(self):
        """初始化提取器，pdfplumber 按需加载"""
        self._pdfplumber = None

    @property
    def pdfplumber(self):
        """延迟导入 pdfplumber"""
        if self._pdfplumber is None:
            try:
                import pdfplumber
                self._pdfplumber = pdfplumber
            except ImportError:
                logger.error("pdfplumber 未安装，请执行: pip install pdfplumber")
                raise
        return self._pdfplumber

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从 PDF 中提取所有表格

        :param pdf_path: PDF 文件路径
        :return: 表格信息列表，每项包含 table_id, paper_id, markdown, page_num
        """
        paper_id = self._paper_id_from_path(pdf_path)
        all_tables = []

        try:
            embedded_tables = self._extract_embedded_tables(pdf_path, paper_id)
            all_tables.extend(embedded_tables)
            logger.info(f"内嵌表格: {pdf_path} → {len(embedded_tables)} 个")
        except Exception as e:
            logger.warning(f"pdfplumber 提取失败 ({pdf_path}): {e}")

        # 方案二：PaddleOCR PPStructure 处理图片型表格
        # 注意：此方法较慢，仅在 pdfplumber 未提取到表格或需要补充时使用
        # try:
        #     image_tables = self._extract_image_tables(pdf_path, paper_id)
        #     all_tables.extend(image_tables)
        #     logger.info(f"图片表格: {pdf_path} → {len(image_tables)} 个")
        # except Exception as e:
        #     logger.warning(f"PPStructure 提取失败 ({pdf_path}): {e}")

        logger.info(f"表格提取完成: {pdf_path} → 共 {len(all_tables)} 个")
        return all_tables

    def _extract_embedded_tables(self, pdf_path: str, paper_id: str) -> List[Dict]:
        """
        使用 pdfplumber 提取 PDF 内嵌的文本型表格

        :param pdf_path: PDF 路径
        :param paper_id: 论文 ID
        :return: 表格列表
        """
        tables = []
        with self.pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for tbl_idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue

                    cleaned = [
                        [cell if cell is not None else "" for cell in row]
                        for row in table_data
                    ]

                    markdown = self._to_markdown(cleaned)
                    table_hash = hashlib.md5(
                        json.dumps(cleaned, ensure_ascii=False).encode()
                    ).hexdigest()[:16]

                    tables.append({
                        "table_id": table_hash,
                        "paper_id": paper_id,
                        "markdown": markdown,
                        "page_num": page_num + 1,
                        "row_count": len(cleaned),
                        "col_count": len(cleaned[0]) if cleaned else 0,
                    })

        return tables

    def _extract_image_tables(self, pdf_path: str, paper_id: str) -> List[Dict]:
        """
        使用 PaddleOCR PPStructure 识别图片型表格

        :param pdf_path: PDF 路径
        :param paper_id: 论文 ID
        :return: 表格列表
        """
        try:
            from paddleocr import PPStructure
        except ImportError:
            logger.warning("PaddleOCR PPStructure 不可用，跳过图片表格提取")
            return []

        engine = PPStructure(show_log=False, lang="en")
        tables = []

        import fitz
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img_array = pix.samples

            result = engine(img_array)
            for item in result:
                if item["type"] == "table":
                    html = item["res"]["html"]
                    markdown = self._html_to_markdown(html)

                    table_hash = hashlib.md5(html.encode()).hexdigest()[:16]
                    tables.append({
                        "table_id": table_hash,
                        "paper_id": paper_id,
                        "html": html,
                        "markdown": markdown,
                        "page_num": page_num + 1,
                        "bbox": item.get("bbox", []),
                    })

        doc.close()
        return tables

    @staticmethod
    def _to_markdown(data: List[List[str]]) -> str:
        """
        将二维数组转为 Markdown 表格字符串

        :param data: 二维数组（第一行为表头）
        :return: Markdown 表格
        """
        if not data:
            return ""
        lines = []
        # 表头行
        lines.append("| " + " | ".join(str(cell) for cell in data[0]) + " |")
        # 分隔行
        lines.append("| " + " | ".join(["---"] * len(data[0])) + " |")
        # 数据行
        for row in data[1:]:
            # 确保行长度与表头一致
            padded = row + [""] * (len(data[0]) - len(row))
            lines.append("| " + " | ".join(str(cell) for cell in padded[:len(data[0])]) + " |")
        return "\n".join(lines)

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """
        简单 HTML 表格 → Markdown 转换

        :param html: HTML 表格字符串
        :return: Markdown 表格
        """
        try:
            import pandas as pd
            dfs = pd.read_html(html)
            if dfs:
                return TableExtractor._to_markdown(
                    dfs[0].fillna("").astype(str).values.tolist()
                )
        except Exception:
            pass
        # 回退：直接保存 HTML
        return html

    @staticmethod
    def _paper_id_from_path(pdf_path: str) -> str:
        """从 PDF 文件路径提取论文 ID

        兼容 Windows（``\\``）与 Linux（``/``）两种路径分隔符：
        Windows 上传的 PDF 路径带反斜杠，若直接用 os.path.basename 在 Linux
        上无法识别 ``\\``，会把整条路径当成文件名。
        """
        basename = os.path.basename(pdf_path.replace("\\", "/"))
        return os.path.splitext(basename)[0]


if __name__ == "__main__":
    import sys
    from base.config import Config

    conf = Config()
    data_dir = os.path.join(conf.DATA_DIR, "paper_data")
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"请先将论文 PDF 放入 {data_dir}")
        sys.exit(1)

    pdf_path = os.path.join(data_dir, pdf_files[0])
    print(f"测试文件: {pdf_path}")

    extractor = TableExtractor()
    tables = extractor.extract_from_pdf(pdf_path)
    print(f"\n提取到 {len(tables)} 个表格:")
    for t in tables:
        print(f"\n--- [{t['table_id']}] page {t['page_num']} ---")
        print(t["markdown"][:500])
