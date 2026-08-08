from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.bm25_search import BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base.config import Config
from base.logger import logger
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time

class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY, base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise

        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

    def call_dashscope(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个靠谱的助手，根据信息好好回答问题。"},
                    {"role": "user", "content": prompt},
                ]
            )
            return completion.choices[0].message.content if completion.choices else "错误：无效的 LLM 响应"
        except Exception as e:
            self.logger.error(f"LLM 调用失败: {e}")
            return f"错误：LLM 调用失败 - {e}"

    def query(self,query,source_filter=None):
        start_time = time.time()
        self.logger.info(f'开始处理{query}问题,过滤条件为{source_filter or "不限"}')

        answer,need_rag=self.bm25_search.search(query,threshold=0.85)
        if answer:
            self.logger.info(f'BM25找到了靠谱答案:{answer[:50]}')
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成,用时{processing_time:.3f}秒')
            return answer

        elif need_rag:
            self.logger.info(f'BM25答案不靠谱,使用RAG系统处理')
            answer = self.rag_system.generate_answer(query,source_filter=source_filter)
            self.logger.info(f'RAG系统生成答案:{answer[:50]}')
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成,用时{processing_time:.3f}秒')
            return answer

        else:
            self.logger.info(f'BM25没找到任何答案')
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成,用时{processing_time:.3f}秒')
            return '没有找到与问题相关的答案'

def main():
    qa_system = IntegratedQASystem()
    qa_system.logger.info('系统初始化完成,开始接受问题')
    try:
        print('\n-----------集成问答系统----------')
        print(f'支持的来源:{qa_system.config.VALID_SOURCES}')
        print('用法:输入问题按回车键查看答案,输入exit按回车键退出系统')

        while True:
            query=input('\n请录入您的问题:').strip()
            if query.lower() == 'exit':
                qa_system.logger.info('用户输入exit,退出系统准备')
                print('系统退出')
                break
            source_filter = input(f'请输入来源过滤(可选,支持:{"./".join(qa_system.config.VALID_SOURCES)},直接回车表示不限)').strip()
            if source_filter:
                if source_filter.lower() not in qa_system.config.VALID_SOURCES:
                    qa_system.logger.warning(f'用户输入了无效的来源:{source_filter}')
                    source_filter = None
                else:
                    qa_system.logger.info(f'用户选择了来源过滤:{source_filter}')
            answer = qa_system.query(query,source_filter=source_filter)
            print(f'\n答案:{answer}')


    except Exception as e:
        qa_system.logger.error(e)
        print(f'处理问题时出错{e}')
    finally:
        qa_system.mysql_client.close()
        qa_system.logger.info('MYSQL连接关闭,系统退出')

if __name__ == '__main__':
    main()

