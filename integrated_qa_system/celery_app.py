"""
Celery 应用配置 — PaperRAG 增量索引异步化

作用：把「论文增量索引」这个耗时任务从请求主流程中解耦出来，
      扔进消息队列，由后台 Worker 异步执行。

架构：
    生产者(提交任务) → Broker(Redis 消息队列) → Worker(后台进程执行) → Backend(Redis 存结果)
                        celery_app.delay()            celery worker             result.get()

运行 Worker（在 integrated_qa_system 目录下）：
    # Windows 必须用 solo/threads pool（prefork 不支持）
    celery -A celery_app worker --loglevel=info --pool=solo

提交任务（示例）：
    from tasks import incremental_index_task
    incremental_index_task.delay("E:/Workspace/rag_project/data/paper_data")
"""
import os
import sys

# 让 celery_app 从任意目录运行都能 import base.config
_project_root = os.path.dirname(os.path.abspath(__file__))  # = integrated_qa_system/
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from celery import Celery
from base.config import Config

conf = Config()

# ── Broker / Backend：复用项目已有的 Redis（docker-compose 中已启动）──
# broker = 消息队列，传递任务；backend = 结果后端，存储任务返回值
_redis_pw = f":{conf.REDIS_PASSWORD}@" if conf.REDIS_PASSWORD else ""
_redis_url = f"redis://{_redis_pw}{conf.REDIS_HOST}:{conf.REDIS_PORT}/{conf.REDIS_DB}"

celery_app = Celery(
    "paperrag",
    broker=_redis_url,    # 任务队列（消息中间件）
    backend=_redis_url,   # 结果后端（存任务返回值，可选）
)

celery_app.conf.update(
    # 序列化：用 JSON（跨语言、安全），不用 pickle（有反序列化注入风险）
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # 时区
    timezone="Asia/Shanghai",
    enable_utc=True,

    # 任务状态可追踪（PENDING → STARTED → SUCCESS/FAILURE）
    task_track_started=True,

    # 任务执行完才 ack（手动确认）：Worker 崩溃时任务不会丢，会重新投递
    task_acks_late=True,

    # 每次只预取 1 个任务：长任务场景避免一个 Worker 囤积多个任务
    worker_prefetch_multiplier=1,

    # 超时保护：软超时 30 分钟（抛异常可捕获），硬超时 1 小时（强杀）
    task_soft_time_limit=1800,
    task_time_limit=3600,
)
