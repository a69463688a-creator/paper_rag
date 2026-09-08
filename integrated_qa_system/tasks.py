"""
Celery 任务定义 — 增量索引的异步化封装

关键点：Celery Worker 是独立进程（prefork/solo），模型和数据库连接
        不能跨进程共享。所以每个 task 内部要自己实例化 VectorStore 和
        IndexManager，而不是复用主进程的实例。

提交方式：
    from tasks import incremental_index_task, index_paper_task

    # 异步提交（不阻塞，立刻返回）
    async_result = incremental_index_task.delay("path/to/paper_data")
    print(async_result.id)          # 任务 ID
    print(async_result.get())       # 阻塞等待，拿最终统计结果

依赖安装：
    pip install celery   # requirements.txt 尚未包含，需手动安装
"""
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from celery_app import celery_app
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.index_manager import IndexManager


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def index_paper_task(self, pdf_path: str):
    """
    异步索引单篇论文（文本分块 + 图表提取 + 表格提取 + 入库）

    bind=True → 绑定 self，可调用 self.retry() 重试
    """
    try:
        # Worker 进程内独立实例化：模型/连接不跨进程共享
        vs = VectorStore()
        mgr = IndexManager(vs)
        return mgr.index_single_paper(pdf_path)
    except Exception as exc:
        # 失败自动重试，指数退避：60s → 120s → 240s
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def incremental_index_task(self, data_dir: str):
    """
    异步增量索引：扫描目录，只处理未入库的新论文（免全量重建）
    """
    try:
        vs = VectorStore()
        mgr = IndexManager(vs)
        return mgr.incremental_index(data_dir)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(bind=True, max_retries=2)
def reindex_paper_task(self, pdf_path: str):
    """
    异步覆盖式重索引：先删除旧数据，再重新处理（用于论文更新）
    """
    try:
        vs = VectorStore()
        mgr = IndexManager(vs)
        return mgr.reindex_paper(pdf_path)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
