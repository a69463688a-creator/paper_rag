import redis
import json
import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
project_root=os.path.dirname(module_dir)
sys.path.insert(0,project_root)

from base import Config,logger

class RedisClient:
    def __init__(self):
        self.logger=logger
        self.config=Config()
        try:
            self.client=redis.StrictRedis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                password=self.config.REDIS_PASSWORD,
                db=self.config.REDIS_DB,
                decode_responses=True
            )
        except redis.RedisError as e:
            self.logger.error(e)
            raise
        self.logger.info("Redis client successfully initialized")

    def set_data(self,key,value,ttl=None):
        try:
            self.client.set(key,json.dumps(value,ensure_ascii=False),ex=ttl)
            self.logger.info("Set key %s to %s successfully"%(key,value))
        except redis.RedisError as e:
            self.logger.error(e)
            raise

    def get_data(self,key):
        try:
            data= self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(e)
            return None

    @staticmethod
    def _normalize_key(query):
        """规范化缓存键：小写 + 压缩连续空白，让大小写/空格不同的等价问题共享缓存"""
        return " ".join(query.lower().split())

    def get_answer(self,query):
        key=f'answer:{self._normalize_key(query)}'
        try:
            answer=self.client.get(key)
            if answer:
                self.logger.info("Get answer from redis successfully")
                return answer
            return None

        except redis.RedisError as e:
            self.logger.error(f'Redis get answer error: {e}')
            return None

    def set_answer(self,query,answer,ttl=None):
        """缓存 FAQ 答案，TTL 过期自动失效；键经规范化，避免大小写/空格差异导致重复缓存"""
        if ttl is None:
            ttl=self.config.REDIS_CACHE_TTL
        key=f'answer:{self._normalize_key(query)}'
        try:
            self.client.set(key,answer,ex=ttl)
            self.logger.info(f"Cache answer for key {key} successfully")
        except redis.RedisError as e:
            self.logger.error(f'Redis set answer error: {e}')

    def delete_key(self, key):
        """主动失效：删除指定缓存 key"""
        try:
            self.client.delete(key)
            self.logger.info(f"Delete key {key}")
        except redis.RedisError as e:
            self.logger.error(f'Redis delete error: {e}')

    def flush_qa_corpus(self):
        """失效语料缓存，下次 _load_data 会回源 MySQL 重建"""
        self.delete_key('qa_original_questions')
        self.delete_key('qa_tokenized_questions')

    def invalidate_answer(self, query):
        """失效某条 FAQ 答案缓存"""
        self.delete_key(f'answer:{self._normalize_key(query)}')

    def invalidate_all_answers(self):
        """失效所有 FAQ 答案缓存（用 scan_iter 避免 keys 阻塞）"""
        try:
            for key in self.client.scan_iter(match='answer:*'):
                self.client.delete(key)
            self.logger.info('All answer caches invalidated')
        except redis.RedisError as e:
            self.logger.error(f'invalidate answers error: {e}')

if __name__ == '__main__':
    redCli=RedisClient()
    questions=redCli.get_answer('qa_original_questions')
    print(questions)
