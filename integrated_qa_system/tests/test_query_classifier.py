"""测试查询意图分类器"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestQueryClassifier:
    """BERT 查询分类器单元测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from rag_qa.core.query_classifier import QueryClassifier
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rag_qa", "models", "bert_query_classifier"
        )
        # 模型文件未提交到 git（.gitignore 忽略），CI 环境无模型则跳过；
        # 本地有模型时正常跑这 8 个分类器测试
        if not os.path.exists(model_path):
            pytest.skip("BERT 分类器模型未提交到 git，CI 环境跳过")
        self.classifier = QueryClassifier(model_path=model_path)

    def test_paper_query_transformer(self):
        """论文查询: Transformer 架构"""
        result = self.classifier.predict_category(
            "Transformer 论文中的多头注意力机制是如何实现的？"
        )
        assert result == "论文学术咨询"

    def test_paper_query_bert(self):
        """论文查询: BERT 预训练"""
        result = self.classifier.predict_category(
            "BERT 的 MLM 预训练任务中 mask 比例是多少？"
        )
        assert result == "论文学术咨询"

    def test_paper_query_comparison(self):
        """论文查询: 模型对比"""
        result = self.classifier.predict_category(
            "ResNet 和 ViT 在 ImageNet 上的性能对比如何？"
        )
        assert result == "论文学术咨询"

    def test_generic_query_math(self):
        """通用查询: 数学计算"""
        result = self.classifier.predict_category("100 的二进制表示是多少？")
        assert result == "通用问答"

    def test_generic_query_coding(self):
        """通用查询: 编程问题"""
        result = self.classifier.predict_category("用 Python 写一个快速排序算法")
        assert result == "通用问答"

    def test_generic_query_daily(self):
        """通用查询: 日常问题"""
        result = self.classifier.predict_category("怎么做红烧肉？")
        assert result == "通用问答"

    def test_label_map(self):
        """标签映射正确"""
        assert self.classifier.label_map == {"通用问答": 0, "论文学术咨询": 1}

    def test_empty_query_handling(self):
        """空查询不崩溃"""
        result = self.classifier.predict_category("")
        assert result in ("通用问答", "论文学术咨询")
