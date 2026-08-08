import jieba

from base import logger

def preprocess_text(text):
    logger.info('preprocess text start')
    try:
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(e)
        return []

if __name__ == '__main__':
    print(preprocess_text('这是一个测试文本'))