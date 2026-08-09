"""测试提示词模板"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRAGPrompt:
    """RAG 主提示词"""

    def test_prompt_renders_with_context(self):
        """包含上下文时正常渲染"""
        from rag_qa.core.prompts import RAGPrompts
        prompt = RAGPrompts.rag_prompt()
        result = prompt.format(
            context="Transformer 是一种基于注意力机制的架构。",
            history="",
            question="什么是 Transformer？",
            phone=""
        )
        assert "Transformer" in result
        assert "什么是 Transformer？" in result
        assert "学术论文研究助手" in result

    def test_prompt_renders_with_history(self):
        """包含对话历史时正常渲染"""
        from rag_qa.core.prompts import RAGPrompts
        prompt = RAGPrompts.rag_prompt()
        result = prompt.format(
            context="BERT 使用 MLM 和 NSP 预训练。",
            history="Q: 什么是预训练？\nA: 预训练是在大规模语料上训练模型的过程。",
            question="BERT 的预训练任务有哪些？",
            phone=""
        )
        assert "BERT" in result
        assert "预训练" in result
        assert "什么是预训练？" in result


class TestHyDEPrompt:
    """HyDE 假设文档提示词"""

    def test_hyde_prompt_contains_query(self):
        """HyDE prompt 包含原始查询"""
        from rag_qa.core.prompts import RAGPrompts
        prompt = RAGPrompts.hyde_prompt()
        result = prompt.format(query="Vision Transformer 的图像分块策略")
        assert "Vision Transformer" in result


class TestSubqueryPrompt:
    """子查询分解提示词"""

    def test_subquery_prompt_contains_query(self):
        """子查询 prompt 包含原始查询"""
        from rag_qa.core.prompts import RAGPrompts
        prompt = RAGPrompts.subquery_prompt()
        result = prompt.format(query="Transformer 和 CNN 的优缺点对比")
        assert "Transformer" in result
        assert "子查询" in result


class TestBacktrackingPrompt:
    """回溯简化提示词"""

    def test_backtracking_prompt_contains_query(self):
        """回溯 prompt 包含原始查询"""
        from rag_qa.core.prompts import RAGPrompts
        prompt = RAGPrompts.backtracking_prompt()
        result = prompt.format(query="ResNet 中残差连接的数学原理及其在梯度传播中的作用")
        assert "ResNet" in result
        assert "简化问题" in result
