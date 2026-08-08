# paper_rag

论文阅读智能助手 —— 基于 RAG（检索增强生成）的学术论文问答系统。

## 核心能力

- **论文检索**：arXiv + Semantic Scholar API，支持中英文混合查询
- **图表理解**：PDF 图表抽取 + 视觉模型描述
- **表格解析**：结构化提取论文中的实验数据表格
- **多策略 RAG**：HyDE / 子查询分解 / 回溯检索 / 引用链追踪
- **混合检索**：BM25 精确匹配 + BGE-M3 稠密/稀疏双路向量 + Reranker 重排序
- **流式输出**：WebSocket 实时打字机式回答

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | FastAPI + LangChain |
| 向量库 | Milvus |
| Embedding | BGE-M3 (dense + sparse) |
| Reranker | BGE-Reranker-Large |
| LLM | DeepSeek API |
| 视觉模型 | Ollama llama3.2-vision:11b |
| 查询分类 | BERT (bert-base-chinese) |
| OCR | PaddleOCR / RapidOCR |
| 结构化提取 | pdfplumber / PaddleOCR PPStructure |
| 缓存 | Redis |
| 存储 | MySQL + Milvus |

## 快速开始

### 1. 环境准备

```bash
pip install -r integrated_qa_system/requirements.txt
```

### 2. 配置

```bash
cp integrated_qa_system/config.ini.example integrated_qa_system/config.ini
# 编辑 config.ini，填入你的 API Key 和数据库连接信息
```

### 3. 下载模型

从 HuggingFace 下载以下模型到 `integrated_qa_system/rag_qa/models/`：

- `BAAI/bge-m3` → `models/bge-m3/`
- `BAAI/bge-reranker-large` → `models/bge-reranker-large/`
- `google-bert/bert-base-chinese` → `models/bert-base-chinese/`

### 4. 启动服务

```bash
# 启动 Milvus + MySQL + Redis
docker-compose up -d

# 启动问答系统
cd integrated_qa_system
python app.py
```

## 项目结构

```
paper_rag/
├── integrated_qa_system/
│   ├── app.py                          # FastAPI 主入口
│   ├── new_main.py                     # 集成问答系统核心
│   ├── base/
│   │   ├── config.py                   # 配置管理
│   │   └── logger.py                   # 日志
│   ├── rag_qa/
│   │   ├── core/
│   │   │   ├── rag_system.py           # RAG 主流程
│   │   │   ├── vector_store.py         # Milvus 向量存储
│   │   │   ├── document_processor.py   # 文档加载 + 分块
│   │   │   ├── query_classifier.py     # BERT 查询分类
│   │   │   └── strategy_selector.py    # 检索策略选择
│   │   ├── edu_document_loaders/       # PDF/PPT/图片加载器
│   │   ├── edu_text_spliter/           # 中文递归分块
│   │   └── rag_assessment/             # RAGAS 评估
│   ├── mysql_qa/                       # MySQL + BM25 检索
│   └── static/                         # 前端页面
└── .gitignore
```

## License

MIT
