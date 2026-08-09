"""测试文档处理器"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLanguageDetection:
    """中英文检测"""

    def test_english_paper_detected(self):
        """英文论文文本 → 'en'"""
        from rag_qa.core.document_processor import _detect_language
        text = ("We propose a new simple network architecture, the Transformer, "
                "based solely on attention mechanisms, dispensing with recurrence "
                "and convolutions entirely. Experiments on two machine translation "
                "tasks show these models to be superior in quality while being more "
                "parallelizable and requiring significantly less time to train.")
        assert _detect_language(text) == "en"

    def test_chinese_text_detected(self):
        """中文文本 → 'zh'"""
        from rag_qa.core.document_processor import _detect_language
        text = ("本文提出了一种全新的网络架构——Transformer，它完全基于注意力机制，"
                "摒弃了传统的循环和卷积结构。在机器翻译任务上的实验表明，该模型"
                "在翻译质量上显著优于现有方法，同时具有更好的并行性。")
        assert _detect_language(text) == "zh"

    def test_mixed_text_defaults_chinese(self):
        """中英混合但中文为主 → 'zh'"""
        from rag_qa.core.document_processor import _detect_language
        text = "本文提出Transformer模型，使用Self-Attention机制处理序列数据"
        assert _detect_language(text) == "zh"

    def test_empty_text(self):
        """空文本 → 'zh'（默认）"""
        from rag_qa.core.document_processor import _detect_language
        assert _detect_language("") == "zh"


class TestDocumentLoaders:
    """文档加载器注册"""

    def test_pdf_loader_registered(self):
        """PDF 加载器已注册"""
        from rag_qa.core.document_processor import document_loaders
        assert ".pdf" in document_loaders

    def test_txt_loader_registered(self):
        """TXT 加载器已注册"""
        from rag_qa.core.document_processor import document_loaders
        assert ".txt" in document_loaders

    def test_md_loader_registered(self):
        """Markdown 加载器已注册"""
        from rag_qa.core.document_processor import document_loaders
        assert ".md" in document_loaders

    def test_img_loaders_registered(self):
        """图片加载器已注册"""
        from rag_qa.core.document_processor import document_loaders
        assert ".jpg" in document_loaders
        assert ".png" in document_loaders
