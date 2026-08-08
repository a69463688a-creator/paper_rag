from langchain_core.prompts import PromptTemplate
from base.config import Config
from base.logger import logger
from openai import OpenAI


class StrategySelector:
    def __init__(self):
        self.client = OpenAI(api_key=Config().DASHSCOPE_API_KEY,
                             base_url=Config().DASHSCOPE_BASE_URL)
        self.strategy_prompt_template = self._get_strategy_prompt()

    def call_dashscope(self, prompt):
        # 调用 DashScope API
        try:
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            logger.error(f"DashScope API 调用失败: {e}")
            return "直接检索"


    def _get_strategy_prompt(self):
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从论文知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：Transformer 论文中提出的多头注意力机制是如何工作的？
                        * 策略：直接检索
                    * 查询：BERT 模型在 GLUE 基准上的实验结果是多少？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：自注意力机制相比循环神经网络有哪些优势？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多篇论文或需要对比分析的问题，例如：
                    * 示例：
                        * 查询：比较 Transformer 和 LSTM 在机器翻译任务上的性能和架构差异。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：如果把 Transformer 的编码器层数从 6 层增加到 12 层，同时把注意力头数也翻倍，会对模型性能和训练时间产生什么影响？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    def select_strategy(self, query):
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    ss = StrategySelector()
    ss.select_strategy('如何养成孩子的时间观念')