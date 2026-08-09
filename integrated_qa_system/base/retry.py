"""
LLM 调用重试工具

对瞬时性失败（网络超时、服务端 5xx、连接断开等）自动重试，
使用指数退避策略避免雪崩效应。

用法:
    from base.retry import retry_with_backoff

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def call_llm(prompt):
        ...
"""

import time
import random
import functools
from base.logger import logger


def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0):
    """
    指数退避重试装饰器

    重试间隔: base_delay * (2 ** attempt) + random(0, 1) 秒
    每次重试的延迟递增: ~1s → ~2s → ~4s → ~8s

    :param max_retries: 最大重试次数（不含首次调用）
    :param base_delay: 基础延迟（秒）
    :param max_delay: 单次延迟上限（秒）
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (2**attempt) + random.uniform(0, 1),
                            max_delay,
                        )
                        logger.warning(
                            f"[重试 {attempt + 1}/{max_retries}] {func.__name__} "
                            f"失败: {e}，{delay:.1f}s 后重试"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[重试耗尽] {func.__name__} 已重试 {max_retries} 次，"
                            f"最终错误: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator
