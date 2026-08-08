import os
import configparser

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)

project_root=os.path.dirname(current_dir_path)

config_file_path=os.path.join(project_root,'config.ini')

class Config:
    def __init__(self,config_file=config_file_path):
        # self.config=configparser.ConfigParser()
        # self.config.read(config_file,encoding='utf-8')
        #
        # self.MYSQL_HOST=self.config.get('mysql','host',fallback='localhost')
        # self.MYSQL_USER=self.config.get('mysql','user',fallback='root')
        # self.MYSQL_PASSWORD=self.config.get('mysql','password',fallback='123456')
        # self.MYSQL_DATABASE=self.config.get('mysql','database',fallback='paper_rag')
        #
        # self.REDIS_HOST=self.config.get('redis', 'host',fallback='localhost')
        # self.REDIS_PORT=self.config.get('redis', 'port',fallback=6379)
        # self.REDIS_PASSWORD=self.config.get('redis', 'password',fallback='1234')
        # self.REDIS_DB=self.config.get('redis', 'db',fallback=0)
        #
        # self.LOG_FILE=self.config.get('logger','log_file',fallback='logs/app.log')
        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 如果没有提供配置文件路径，则使用默认路径

        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件
        self.config.read(config_file,encoding='utf-8')

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        # MySQL 密码
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        # MySQL 数据库名
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='paper_rag'))

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        # Redis 密码
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        # Redis 数据库编号
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='paper_rag'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name', fallback='paper_rag'))

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope/LLM API 密钥 — config.ini 优先，环境变量可覆盖
        _api_key_from_file = self.config.get('llm', 'dashscope_api_key', fallback='')
        self.DASHSCOPE_API_KEY = _api_key_from_file if _api_key_from_file else os.getenv('DASHSCOPE_API_KEY', '')
        # Base URL
        _base_url_from_file = self.config.get('llm', 'dashscope_base_url', fallback='')
        self.DASHSCOPE_BASE_URL = _base_url_from_file if _base_url_from_file else os.getenv('DASHSCOPE_BASE_URL',
            'https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')

        # 路径配置


if __name__ == '__main__':
    config_file='../config.ini'
    conf=Config(config_file)
    print(f'{conf.MYSQL_HOST}{conf.MYSQL_USER}{conf.MYSQL_PASSWORD}{conf.MYSQL_DATABASE}')
    print(f'{conf.REDIS_HOST}{conf.REDIS_PORT}{conf.REDIS_PASSWORD}{conf.REDIS_DB}')
    print(f'{conf.LOG_FILE}')
