from db.mysql_client import MySQLClient
from cache.redis_client import RedisClient
from retrieval.bm25_search import BM25Search
from base import logger
import time

class MySQLQASystem:
    def __init__(self):
        self.logger = logger
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_client = BM25Search(redis_client=self.redis_client,mysql_client=self.mysql_client)

    def query(self,query):
        start_time = time.time()
        self.logger.info(f'query: {query}')
        answer,_=self.bm25_client.search(query,threshold=0.85)
        if answer:
            self.logger.info(f'mysql answer: {answer}')
        else:
            self.logger.info('SQL中未找到答案,调用RAG系统')
            answer='SQL中未找到答案'
        process_time = time.time() - start_time
        self.logger.info(f'process_time: {process_time:.3f}')
        return answer

def main():
    mysql_qa = MySQLQASystem()
    try:
        print('\n欢迎使用MySQL问答系统:')
        print('输入查询进行回答,输入exit退出!')

        while True:
            query=input('query: ').strip()
            if query.lower() == 'exit':
                logger.info('exit')
                print('再见')
                break
            answer=mysql_qa.query(query)
            print(f'mysql answer: {answer}')

    except Exception as e:
        logger.error(f'system error:{e}')
    finally:
        mysql_qa.mysql_client.close()

if __name__ == '__main__':
    main()