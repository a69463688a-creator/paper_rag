"""
论文 PDF 专用加载器

arXiv 论文是数字原生 PDF，使用 PyMuPDF 内置 text 提取即可，无需 OCR。
相比 OCRPDFLoader，本加载器不依赖 rapidocr/onnxruntime，轻量且快速。
"""

import fitz  # PyMuPDF
from typing import Iterator
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from tqdm import tqdm
from base.logger import logger


class PaperPDFLoader(BaseLoader):
    """
    论文 PDF 加载器 —— 仅用 PyMuPDF 内置提取，不依赖 OCR

    用法:
        loader = PaperPDFLoader("paper.pdf")
        docs = loader.load()
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        text = self._pdf2text()
        yield Document(page_content=text, metadata={"source": self.file_path})

    def _pdf2text(self) -> str:
        """使用 PyMuPDF 内置文本提取，跳过 OCR 步骤"""
        doc = fitz.open(self.file_path)
        resp = ""

        b_unit = tqdm(total=doc.page_count, desc=f"PaperPDFLoader: {self.file_path[-40:]}")
        for i, page in enumerate(doc):
            b_unit.set_description(f"PaperPDFLoader page {i+1}/{doc.page_count}")
            b_unit.refresh()
            text = page.get_text("text")
            if text:
                resp += text + "\n"
            b_unit.update(1)

        doc.close()
        logger.info(f"PaperPDFLoader 完成: {self.file_path} -> {len(resp)} 字符")
        return resp


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base.config import Config

    conf = Config()
    data_dir = os.path.join(conf.DATA_DIR, "paper_data")
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if pdf_files:
        pdf_path = os.path.join(data_dir, pdf_files[0])
        loader = PaperPDFLoader(pdf_path)
        doc = loader.load()
        print(f"文件: {pdf_path}")
        print(f"总字符数: {len(doc[0].page_content)}")
        print(f"\n前 500 字符:\n{doc[0].page_content[:500]}")
