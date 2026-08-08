# PaperRAG — 学术论文阅读助手

基于 RAG 的学术论文智能问答系统，应届生 Agent 开发岗位求职项目。

## 快速启动

```bash
# 1. Docker
cd E:\Software\milvus_redis && docker compose up -d

# 2. App
cd integrated_qa_system
D:/conda_envs/EduRAG-GPU/python app.py
# → http://localhost:8080
```

## 架构

```
用户提问 → BERT分类器 → BM25(MySQL) → RAG(Milvus+BGE-M3+Reranker+LLM)
                              ↓ 命中就返回
                         三路检索: 文本 + 图表 + 表格 → DeepSeek 生成 → WebSocket流式
```

## 关键模块

| 文件 | 功能 |
|------|------|
| `new_main.py` | IntegratedQASystem 主入口 |
| `rag_qa/core/new_rag_system.py` | RAG主流程+检索策略+三路检索 |
| `rag_qa/core/vector_store.py` | Milvus 3个Collection |
| `rag_qa/core/query_classifier.py` | BERT 意图分类 |
| `rag_qa/core/document_processor.py` | 文档加载+中英文分块 |
| `rag_qa/paper_data/` | arXiv加载器+图表提取+表格提取 |
| `app.py` | FastAPI WebSocket API |
| `base/config.py` | 配置管理(config.ini优先) |

## 数据

- 19篇经典论文 (arXiv) → 1358文本块 + 73图表 + 122表格
- Milvus: `paper_rag` / `paper_rag_figures` / `paper_rag_tables`
- MySQL: `paper_rag` 数据库 (conversations + jpkb 表)
