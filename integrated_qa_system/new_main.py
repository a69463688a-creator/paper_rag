#历史对话 流式输出

from mysql_qa import RedisClient,MySQLClient,BM25Search

from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore

from base.config import Config
from base.logger import logger
from base.retry import retry_with_backoff

from openai import OpenAI

import time
import pymysql #处理异常捕获curd
import uuid #生成唯一会话id

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

        #new
        self.init_conversation_table()

    def init_conversation_table(self):
        self.mysql_client.ensure_connection()
        try:
            self.mysql_client.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(36) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                INDEX idx_session_id (session_id) 
);""")
            self.mysql_client.connection.commit()
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"初始化对话历史表失败: {e}")
            raise

    def _fetch_recent_history(self,session_id):
        self.mysql_client.ensure_connection()
        try:
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                    LIMIT %s
                """, (session_id, 5))
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []

    def get_session_history(self,session_id):
        return self._fetch_recent_history(session_id)

    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        self.mysql_client.ensure_connection()
        try:
            self.mysql_client.cursor.execute("""
                 INSERT INTO conversations (session_id, question, answer, timestamp)
                 VALUES (%s, %s, %s, NOW())
                 """, (session_id, question, answer))
            history = self._fetch_recent_history(session_id)
            self.mysql_client.cursor.execute("""
                DELETE
                FROM conversations
                WHERE session_id = %s
                  AND id NOT IN (SELECT id
                                 FROM (SELECT id
                                       FROM conversations
                                       WHERE session_id = %s
                                       ORDER BY timestamp DESC
                                           LIMIT %s) AS sub)
                """, (session_id, session_id, 5))
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史更新成功")
            return history
        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            raise
        except Exception as e:
            self.logger.error(f"更新会话历史意外错误: {e}")
            self.mysql_client.connection.rollback()
            raise

    def clear_session_history(self, session_id: str) -> bool:
        self.mysql_client.ensure_connection()
        try:
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"清除会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            return False
    

    def call_dashscope(self, prompt):
        """
        调用 LLM 流式生成，带指数退避重试

        连接建立阶段使用 retry_with_backoff 自动重试；
        流式传输阶段的错误不重试（避免重复生成）。
        """

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def _create_completion():
            """创建流式请求，仅连接建立阶段出错时重试"""
            return self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个靠谱的助手，根据信息好好回答问题。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True
            )

        try:
            completion = _create_completion()
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"LLM 调用失败（含重试后）: {e}")
            yield f"抱歉，LLM 服务暂时不可用，请稍后重试。（错误: {str(e)[:100]}）"
        
    

    def query(self,query,source_filter=None,session_id=None):
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        history = self.get_session_history(session_id) if session_id else []
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        if answer:
            self.logger.info(f"MySQL答案: {answer}")
            if session_id:
                self.update_session_history(session_id, query, answer)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield answer, True, None
        elif need_rag:
            self.logger.info("无可靠MySQL答案，回退到RAG")
            collected_answer = ""
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                collected_answer += token
                yield token, False, None
            # 附加来源信息
            sources = self.rag_system.last_sources
            if sources:
                yield "", False, sources
            if session_id:
                self.update_session_history(session_id, query, collected_answer)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "", True, None
        else:
            self.logger.info("未找到答案")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True, None

def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用集成问答系统！")
    print(f"会话ID: {session_id}")
    print(f"支持的论文领域：{qa_system.config.VALID_SOURCES}")
    print("输入查询进行问答，输入 'exit' 退出。")
    try:
        while True:
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break
            source_filter = input(
                f"请输入论文领域 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效的论文领域 '{source_filter}'，将不过滤")
                source_filter = None
            print("\n答案: ", end="", flush=True)
            answer = ""
            for item in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                token, is_complete, sources = item if len(item) == 3 else (item[0], item[1], None)
                if token:
                    print(token, end="", flush=True)
                    answer += token
                if sources:
                    print("\n\n📚 参考来源:")
                    for src in sources:
                        print(f"  • {src.get('paper_id', '未知')} ({src.get('type', 'text')})")
                if is_complete:
                    print()
                    break
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
            time.sleep(0.5)
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.mysql_client.close()

if __name__ == '__main__':
    main()