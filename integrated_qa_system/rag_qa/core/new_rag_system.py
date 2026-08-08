import sys,os

from torch.backends.opt_einsum import strategy

from langchain_core.documents import Document
from rag_qa.core.prompts import RAGPrompts
import time

current_dir=os.path.dirname(os.path.abspath(__file__))
rag_qa_path=os.path.dirname(current_dir)
sys.path.insert(0,rag_qa_path)
project_root=os.path.dirname(rag_qa_path)
sys.path.insert(0,project_root)

from base.config import Config
from base.logger import logger
from rag_qa.core.query_classifier import QueryClassifier, current_dir
from rag_qa.core.strategy_selector import StrategySelector
from rag_qa.core.vector_store import VectorStore


from transformers import BertTokenizer,BertModel
local_model_path=r'E:\Workspace\rag_project\integrated_qa_system\rag_qa\models\bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model=BertModel.from_pretrained(local_model_path)
from openai import OpenAI

conf=Config()

class RAGSystem:
    def __init__(self,vector_store,llm):
        """

        :param vector_store: 数据库对象
        :param llm: 大模型调用函数
        """
        self.vector_store=vector_store
        self.llm=llm
        self.rag_prompt=RAGPrompts.rag_prompt()

        classifier_path=os.path.join(rag_qa_path,'models','bert_query_classifier')
        self.query_classifier=QueryClassifier(classifier_path)
        self.strategy_selector=StrategySelector()

    def _retrieve_with_hyde(self, query,source_filter=None):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")

            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=conf.RETRIEVAL_K,source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    def _retrieve_with_subqueries(self, query):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            #   返回去重后的文档，限制数量 (是否需要在此处限制? retrieve_and_merge 末尾会限制)
            # return unique_docs[: Config.CANDIDATE_M]
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    def _retrieve_with_backtracking(self, query):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K  # 使用 K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    def retrieve_and_merge(self, query,source_filter=None,strategy=None):
        """
        统一入口
        :param query:
        :param source_filter:
        :param strategy:
        :return:
        """
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )
        logger.info(f'策略{strategy}检索到{len(ranked_sub_chunks)}个候选文档')

        # 三路检索合并：文本 + 图表 + 表格
        try:
            figure_results = self.vector_store.search_figures(query, k=2)
            if figure_results:
                for fig in figure_results:
                    fig_content = f"[图表] {fig.get('caption', '')}\n{fig.get('description', '')}"
                    ranked_sub_chunks.append(Document(
                        page_content=fig_content,
                        metadata={"type": "figure", "paper_id": fig.get("paper_id", ""),
                                   "image_path": fig.get("image_path", "")}
                    ))
                logger.info(f'图表检索补充 {len(figure_results)} 个结果')
        except Exception as e:
            logger.warning(f"图表检索跳过: {e}")

        try:
            table_results = self.vector_store.search_tables(query, k=2)
            if table_results:
                for tbl in table_results:
                    tbl_content = f"[表格] (page {tbl.get('page_num', '?')})\n{tbl.get('markdown', '')}"
                    ranked_sub_chunks.append(Document(
                        page_content=tbl_content,
                        metadata={"type": "table", "paper_id": tbl.get("paper_id", "")}
                    ))
                logger.info(f'表格检索补充 {len(table_results)} 个结果')
        except Exception as e:
            logger.warning(f"表格检索跳过: {e}")

        final_content_docs=ranked_sub_chunks[:conf.CANDIDATE_M]
        logger.info(f'最终上下文文档数量: {len(final_content_docs)}')
        return final_content_docs

    def generate_answer(self,query,source_filter=None,history=None):
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")

        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        query_category = self.query_classifier.predict_category(query)

        # 关键词规则兜底：包含论文相关关键词的问题强制按学术咨询处理
        paper_keywords = [
            "论文", "文献", "arxiv", "模型架构", "自注意力", "预训练",
            "Figure", "Table", "图表", "公式", "实验", "对比", "综述",
            "Transformer", "BERT", "GPT", "ResNet", "ViT", "LSTM", "CNN",
            "消融", "参数量", "贡献", "架构图", "encoder", "decoder",
        ]
        if query_category == "通用问答" and any(kw.lower() in query.lower() for kw in paper_keywords):
            query_category = "论文学术咨询"
            logger.info(f"关键词规则兜底：查询分类改为论文学术咨询 (查询: '{query}')")

        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        if query_category == "通用问答":
            logger.info("查询为通用问答，直接调用 LLM")
            prompt_input = self.rag_prompt.format(
                context="", history=history_context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
            )
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                answer = f"抱歉，处理您的通用问答问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
            processing_time = time.time() - start_time
            logger.info(
                f"通用问答查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return answer

        logger.info("查询为论文学术咨询，执行 RAG 流程")
        strategy = self.strategy_selector.select_strategy(query)

        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )

        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...")
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")



        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE
        )
        # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = f"抱歉，处理您的论文学术咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

        #   记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer



if __name__ == '__main__':
    vector_store=VectorStore()
    llm=StrategySelector().call_dashscope
    rag_system=RAGSystem(vector_store, llm)
    answer=rag_system.generate_answer('原神是什么',source_filter='ai')
    print(answer)








