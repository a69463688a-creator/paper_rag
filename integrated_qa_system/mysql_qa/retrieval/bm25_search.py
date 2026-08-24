import time
from rank_bm25 import BM25Okapi
import numpy as np

from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient

from base import Config,logger


class BM25Search:
    def __init__(self,redis_client,mysql_client):
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.bm25=None
        self.questions=None
        self.original_questions=None
        self._last_load_ts = 0                       # 上次加载时间戳
        self._corpus_ttl = Config().BM25_CORPUS_TTL  # 语料缓存 TTL（默认 300s）
        self._load_data(force=True)                  # 首次强制加载

    def _load_data(self, force=False):
        # 懒重建：已加载且未超 TTL，直接短路（O(1) 时间戳比较，零开销）
        if not force and self._last_load_ts and \
                (time.time() - self._last_load_ts) < self._corpus_ttl:
            return

        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        self.original_questions=self.redis_client.get_data(original_key)
        # print("original_questions:",self.original_questions)
        tokenized_questions=self.redis_client.get_data(tokenized_key)
        # print("tokenized_questions:",tokenized_questions)

        if not self.original_questions or not tokenized_questions:
            self.original_questions=self.mysql_client.fetch_questions()

            if not self.original_questions:
                self.logger.warning('No questions found in MySQL database')
                return

            tokenized_questions=[preprocess_text(q[0]) for q in self.original_questions]

            # 语料缓存加 TTL，过期自动回源 MySQL
            self.redis_client.set_data(original_key,[q[0] for q in self.original_questions], ttl=self._corpus_ttl)
            self.redis_client.set_data(tokenized_key,tokenized_questions, ttl=self._corpus_ttl)

        self.questions=tokenized_questions
        self.bm25=BM25Okapi(self.questions)
        self._last_load_ts = time.time()
        self.logger.info('BM25 search data loaded')

    def reload(self):
        """主动重建：清 Redis 语料缓存 + 强制重建进程内 BM25（供写路径调用）"""
        self.redis_client.flush_qa_corpus()
        self._load_data(force=True)


    def _softmax(self,scores):
        exp_scores=np.exp(scores-np.max(scores)) #防溢出
        return exp_scores/np.sum(exp_scores)

    def search(self,query,threshold=0.85):
        """

        :param query:
        :param threshold:
        :return: 匹配成功返回（答案，False） 失败 （None True）  True为新查询 False为命中缓存
        """
        self._load_data()   # 懒重建：每次查询前检查，TTL 过期自动回源 MySQL
        if not query or not isinstance(query,str):
            self.logger.error('无效查询，非字符串或空')
            return None,True

        cached_answer=self.redis_client.get_answer(query)
        if cached_answer:
            self.logger.info(f'缓存redis中获取答案成功:{query}')
            return cached_answer, False
        try:
            query_tokens=preprocess_text(query)
            scores=self.bm25.get_scores(query_tokens)
            # print(f'scores:{scores}')
            softmax_score=self._softmax(scores)
            # print(softmax_score)
            best_idx=softmax_score.argmax()
            best_score=softmax_score[best_idx]
            if best_score >= threshold:
                original_question=self.original_questions[best_idx]
                answer=self.mysql_client.fetch_answer(original_question)
                if answer:
                    self.redis_client.set_answer(query, answer)
                    self.logger.info(f'搜索成功,相似度:{best_score:.3f}')
                    return answer,False
            self.logger.info(f'未找到可靠答案,最高匹配度:{best_score:.3f}低于阈值{threshold:.3f}')
            return None,True

        except Exception as e:
            self.logger.error(e)
            return None,True




if __name__ == '__main__':
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25 = BM25Search(redis_client,mysql_client)

    result=bm25.search(query='afawfwafawf') #怎么获取MongoDB中某个集合中所有文档的键的名字
    print(f'{result}')
