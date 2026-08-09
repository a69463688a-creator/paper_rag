"""测试表格提取器"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMarkdownConversion:
    """二维数组 → Markdown 表格转换"""

    def test_simple_table(self):
        """简单表格转换"""
        from rag_qa.paper_data.table_extractor import TableExtractor
        data = [
            ["Name", "Score", "Rank"],
            ["Alice", "95", "1"],
            ["Bob", "87", "2"],
        ]
        result = TableExtractor._to_markdown(data)
        assert "| Name | Score | Rank |" in result
        assert "| --- | --- | --- |" in result
        assert "| Alice | 95 | 1 |" in result
        assert "| Bob | 87 | 2 |" in result

    def test_single_row_table(self):
        """单行（仅表头）表格"""
        from rag_qa.paper_data.table_extractor import TableExtractor
        data = [["Metric", "Value"]]
        result = TableExtractor._to_markdown(data)
        assert "| Metric | Value |" in result
        assert "| --- | --- |" in result

    def test_empty_table(self):
        """空表格 → 空字符串"""
        from rag_qa.paper_data.table_extractor import TableExtractor
        assert TableExtractor._to_markdown([]) == ""

    def test_table_with_none_values(self):
        """含 None 单元格的表格"""
        from rag_qa.paper_data.table_extractor import TableExtractor
        data = [
            ["A", None, "C"],
            ["1", "2", None],
        ]
        result = TableExtractor._to_markdown(data)
        # None → "None" 字符串，因为到这一步前已经清洗过
        assert "| A |" in result


class TestPaperIDExtraction:
    """论文 ID 提取"""

    def test_arxiv_id_from_filename(self):
        """从文件名提取 arXiv ID"""
        from rag_qa.paper_data.table_extractor import TableExtractor
        path = "/some/dir/1706.03762.pdf"
        assert TableExtractor._paper_id_from_path(path) == "1706.03762"

    def test_paper_id_from_full_path(self):
        """从完整路径提取"""
        from rag_qa.paper_data.table_extractor import TableExtractor
        path = "E:\\Workspace\\rag_project\\data\\paper_data\\2103.00020.pdf"
        assert TableExtractor._paper_id_from_path(path) == "2103.00020"
