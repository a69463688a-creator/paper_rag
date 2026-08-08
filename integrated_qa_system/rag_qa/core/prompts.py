from langchain_core.prompts import PromptTemplate


class RAGPrompts:
    # @staticmethod
    # def rag_prompt():
    #     return PromptTemplate(
    #         template="""
    #         你是一个智能助手，帮助用户回答问题。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         上下文: {context}
    #         问题: {question}
    #
    #         如果无法回答，请回复：“信息不足，无法回答。建议尝试调整搜索关键词或查询相关论文。”
    #         回答:
    #         """,
    #         input_variables=["context", "question", "phone"],
    #     )
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
        你是一个学术论文研究助手，负责帮助用户理解和分析学术论文。请按照以下步骤处理：

        1. **分析问题和上下文**：
           - 基于提供的上下文（如果有）和你的知识回答问题。
           - 如果答案来源于检索到的论文内容，请在回答中明确引用，例如：”根据《Attention Is All You Need》论文，……”。涉及专业术语时，首次出现需标注中英文，如”自注意力机制（Self-Attention）”。

        2. **评估对话历史**：
           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
           - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。

        3. **生成回答**：
           - 提供清晰、准确、结构化的回答，避免无关信息。如果涉及多篇论文的对比，使用结构化方式呈现（如列表或表格）。
           - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答。建议尝试调整搜索关键词或查询相关论文。”

        **上下文**: {context}
        **对话历史**:
        {history}
        **问题**: {question}

        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )

    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

if __name__ == '__main__':
    rag_prompt = RAGPrompts.rag_prompt()
    result=rag_prompt.format(content='')
    print()