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
        try:
            self.client=redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DB,
                decode_responses=True
            )
        except redis.RedisError as e:
            self.logger.error(e)
            raise
        self.logger.info("Redis client successfully initialized")

    def set_data(self,key,value):
        try:
            self.client.set(key,json.dumps(value,ensure_ascii=False))
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

    def get_answer(self,query):
        try:
            answer=self.client.get(f'answer:{query}')
            if answer:
                self.logger.info("Get answer from redis successfully")
                return answer
            return None

        except redis.RedisError as e:
            self.logger.error(f'Redis get answer error: {e}')
            return None

if __name__ == '__main__':
    redCli=RedisClient()
    questions=redCli.get_answer('qa_original_questions')
    print(questions)
