import pymysql
import pandas as pd
import sys,os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)

project_root=os.path.dirname(module_dir)
sys.path.insert(0,project_root)

from base import Config,logger

class MySQLClient:
    def __init__(self):
        self.logger=logger
        self._config = {
            'host': Config().MYSQL_HOST,
            'user': Config().MYSQL_USER,
            'password': Config().MYSQL_PASSWORD,
            'database': Config().MYSQL_DATABASE,
            'autocommit': True,
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 30,
        }
        self._connect()

    def _connect(self):
        """建立 MySQL 连接"""
        try:
            self.connection = pymysql.connect(**self._config)
            self.cursor = self.connection.cursor()
            self.logger.info("mysql client successfully initialized")
        except Exception as e:
            self.logger.error(f'MySQL connect error: {e}')
            raise

    def ensure_connection(self):
        """确保连接有效，断开则自动重连（公开方法，供外部调用）"""
        try:
            self.connection.ping(reconnect=True)
        except Exception:
            self.logger.warning("MySQL 连接断开，正在重连...")
            self._connect()

    # 内部别名
    _ensure_connection = ensure_connection

    def create_table(self):
        self._ensure_connection()
        create_table_query="""
        CREATE TABLE IF NOT EXISTS jpkb (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_name VARCHAR(20),
            question VARCHAR(1000),
            answer VARCHAR(1000))
        """
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info(f'MySQL table successfully created')
        except pymysql.MySQLError as e:
            self.logger.error(f'MySQL create table error: {e}')
            raise
    def insert_data(self,csv_path):
        self._ensure_connection()
        try:
            data =pd.read_csv(csv_path)
            # print(data.head())
            for _ ,row in data.iterrows():
                insert_query="insert into jpkb (subject_name,question,answer) values (%s,%s,%s)"
                self.cursor.execute(insert_query,(row['学科名称'],row['问题'],row['答案']))
            self.connection.commit()
            self.logger.info(f'MySQL table data successfully inserted')

        except Exception as e:
            self.logger.error(f'MySQL insert data error: {e}')
            self.connection.rollback()
            raise

    def fetch_questions(self):
        self._ensure_connection()
        try:
            self.cursor.execute('select question from jpkb')
            results=self.cursor.fetchall()
            self.logger.info(f'MySQL questions successfully fetched')
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f'MySQL fetch questions error: {e}')
            return []

    def fetch_answer(self,question):
        self._ensure_connection()
        try:
            self.cursor.execute('select answer from jpkb where question=%s',(question,))
            result=self.cursor.fetchone()
            self.logger.info(f'MySQL answer successfully fetched')
            return result[0] if result else None
        except pymysql.MySQLError as e:
            self.logger.error(f'MySQL fetch answer error: {e}')
            return None

    def close(self):
        try:
            self.connection.close()
            self.logger.info(f'MySQL client successfully closed')
        except pymysql.MySQLError as e:
            self.logger.error(f'MySQL close error: {e}')


if __name__ == '__main__':
    mysql_client = MySQLClient()
    # mysql_client.create_table()
    # mysql_client.insert_data(csv_path='../data/JP学科知识问答.csv')

    # questions = mysql_client.fetch_questions()
    # print(questions)

    answer=mysql_client.fetch_answer('用上下文管理器实现函数运行时间的计算?')
    print(answer)

    mysql_client.close()

