# PaperRAG — 学术论文阅读助手 技术详解

> 适用于简历项目描述、面试准备、技术复盘

---

## 一、项目概览

| 维度 | 说明 |
|------|------|
| **项目定位** | 基于 RAG 的学术论文智能问答系统，支持论文检索、对比分析、图表问答 |
| **技术栈** | Python / PyTorch / FastAPI / Milvus / MySQL / Redis / Docker / LangChain |
| **核心能力** | 多策略检索、三路召回（文本+图表+表格）、BERT 意图分类、流式生成 |
| **数据规模** | 19 篇经典 arXiv 论文 → 1,033 文本块 (283 父块) + 69 图表 + 97 表格 |
| **硬件环境** | NVIDIA RTX 5070 (Blackwell sm_120, CUDA 12.8) / 本地 GPU 推理 |

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (static HTML)                    │
│                    WebSocket 流式 / REST API                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI (app.py)                            │
│           CORS / 问候检测 / 会话管理 / 健康检查                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                 IntegratedQASystem (new_main.py)                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ BM25 检索 │───▶│ BERT 意图分类 │───▶│ RAG 管线 (4 策略)     │   │
│  │ (MySQL)  │    │ 通用/学术     │    │ 文本+图表+表格 三路   │   │
│  └──────────┘    └──────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────▼──────┐  ┌───────▼───────┐  ┌──────▼──────┐
│    Milvus      │  │    MySQL      │  │   Redis     │
│  paper_rag     │  │ conversations │  │  缓存/Session│
│  paper_rag_    │  │ jpkb (BM25)   │  │             │
│  figures       │  │               │  │             │
│  paper_rag_    │  └───────────────┘  └─────────────┘
│  tables        │
└────────────────┘
```

### 数据流全链路

```
用户提问 "Transformer 相比 LSTM 有什么优势？"
    │
    ▼
[Redis 缓存 answer:{query}] ── 命中 → 直接返回 (<5ms) ──┐
    │ 未命中                                              │
    ▼                                                    │
[BM25 + MySQL jpkb] ── 命中(≥0.85) → 写Redis → 返回 ──┤
    │ 未命中                                              │
    ▼                                                    │
[BERT 分类器] "论文学术咨询"                                │
    │                                                    │
    ▼                                                    │
[策略选择] DeepSeek 分析 → 选择"子查询检索"                  │
    │                                                     │
    ├─▶ 子查询1: "Transformer 架构特点" ──▶ Milvus 混合检索   │
    ├─▶ 子查询2: "LSTM 架构特点"       ──▶ Milvus 混合检索   │
    └─▶ 子查询3: "Transformer vs LSTM 对比"                  │
    │                                                     │
    ▼                                                     │
[三路召回合并] 文本 + 图表 + 表格 → 配额保底 + 统一重排 → top-5 上下文
    │
    ▼
[BGE-Reranker 统一重排序] Cross-Encoder 跨模态公平打分
    │
    ▼
[DeepSeek 生成] + 引用标注 ──▶ WebSocket 流式逐 token 推送  ◀─┘
```

---

## 三、核心模块详解

### 3.1 两阶段检索架构

| 阶段 | 技术 | 说明 |
|------|------|------|
| **第一阶段** | BM25 + Redis 缓存 (MySQL jpkb 表) | 论文高频 FAQ 精确匹配，threshold=0.85，命中直接返回 |
| **第二阶段** | RAG (Milvus + BGE-M3 + Reranker + DeepSeek) | BM25 未命中时启动，多策略 + 多模态检索 |

**设计动机**：BM25 在精确匹配场景下延迟 <50ms（含 Redis 缓存命中时 <5ms），RAG 管线延迟 ~2-3s。两级检索兼顾了性能和覆盖面。

#### BM25 + Redis 缓存机制

```
用户提问 "ResNet 解决了什么问题"
    │
    ▼
┌─ Redis 查 answer:{query} ── 命中 → 直接返回 (<5ms)
│   │ 未命中
│   ▼
│ jieba 分词 → BM25Okapi 打分 → softmax 归一化
│   │
│   ├─ top-1 ≥ 0.85 → MySQL jpkb 查答案 → 写入 Redis answer:{query} → 返回 (<50ms)
│   └─ top-1 < 0.85 → 进入 RAG 管线
│
└─ Redis 未命中但 BM25 命中 → 写入 Redis，下次相同 query 直接缓存返回
```

**数据存储**：

| 层级 | 存储 | Key/Table | 内容 |
|------|------|-----------|------|
| L1 缓存 | Redis | `answer:{query}` | 查询→答案 键值对，首次 BM25 命中后写入 |
| L2 预热 | Redis | `qa_original_questions` / `qa_tokenized_questions` | MySQL 全量问题的原文 + jieba 分词结果，避免每次重启重复分词 |
| L3 持久 | MySQL | `jpkb` 表 (subject_name, question, answer) | 26 条论文 FAQ 数据，覆盖 nlp/cv/ai 三个领域 |

**数据示例**（`jpkb` 表，26 条）：

| 领域 | 示例问题 |
|------|---------|
| NLP | Transformer 是什么、BERT 是什么、自注意力机制、多头注意力 |
| CV | ResNet 解决了什么问题、YOLO 核心思想、ViT 是什么、GAN 原理 |
| AI | Adam 优化器、GPT-3 贡献、DDPM 扩散模型、预训练与微调 |

**BM25 命中实测**：

```
✅ "ResNet 解决了什么问题"     → 相似度 0.999 → BM25 返回，<50ms
✅ "GAN 的原理是什么"          → 相似度 0.998 → BM25 返回
✅ "YOLO 和 Faster R-CNN 区别" → 相似度 0.856 → BM25 返回
❌ "Transformer 注意力机制细节"  → 相似度 0.196 → RAG 接管
❌ "今天天气怎么样"             → 相似度 0.038 → RAG 接管（正确拒绝无关问题）
```

### 3.2 BERT 查询意图分类器

| 参数 | 值 |
|------|-----|
| 基座模型 | `bert-base-chinese` (12层 Transformer) |
| 分类类别 | 2 类：`通用问答` / `论文学术咨询` |
| 训练数据 | 自定义标注数据 (`classify_data/fulldata.txt`) |
| 训练配置 | epoch=3, batch_size=8, max_length=128, fp16=True |
| 优化器 | AdamW, warmup=50 steps, weight_decay=0.01 |
| 评估策略 | 每 epoch 评估，load_best_model_at_end |
| 推理设备 | CUDA (GPU) / CPU fallback |

**分类后处理**：关键词规则兜底——若 BERT 判为"通用问答"但 query 含 `Transformer/BERT/GPT/ResNet` 等论文关键词，强制修正为"论文学术咨询"。

### 3.3 四种检索增强策略

策略选择由 **DeepSeek LLM** 根据查询特征自动决策（`strategy_selector.py`）：

| 策略 | 触发场景 | 核心逻辑 |
|------|---------|---------|
| **直接检索** | 查询意图明确、信息型问题 | user query → Milvus 混合检索 |
| **HyDE** (假设文档嵌入) | 查询抽象、难以直接匹配 | LLM 生成假设答案 → 用假设答案检索 |
| **子查询检索** | 跨论文对比、复合问题 | LLM 拆分为 N 个子查询 → 分别检索 → 合并去重 |
| **回溯问题检索** | 问题过于复杂/具体 | LLM 简化为更基础的学术问题 → 用简化问题检索 |

### 3.4 混合检索 + 重排序

```
Dense Vector (BGE-M3, 1024-dim, IVF_FLAT, nprobe=10, IP)
        +
Sparse Vector (BGE-M3 lexical token weights, SPARSE_INVERTED_INDEX)
        │
        ▼
WeightedRanker (dense=0.7, sparse=1.0) → top-K 候选
        │
        ▼
BGE-Reranker-Large (Cross-Encoder) → 精排 → top-M 最终上下文
```

**为什么用混合检索？** 纯稠密检索对专业术语查全率高但查准率低，纯稀疏检索对精确术语匹配好但语义覆盖差。BGE-M3 同时输出稠密+稀疏向量，加权融合取两者之长。

### 3.5 文档处理与分块

| 参数 | 值 | 说明 |
|------|-----|------|
| Parent chunk size | 1,800 | 大粒度保留上下文完整性（优化后，原 1,200） |
| Child chunk size | 400 | 小粒度提升检索精度（优化后，原 300） |
| Chunk overlap | 50 | 跨块信息不丢失 |
| 英文分块器 | `RecursiveCharacterTextSplitter` (tiktoken, cl100k_base) | 按 `\n\n` / `\n` / `. ` 递归切分 |
| 中文分块器 | `ChineseRecursiveTextSplitter` (自研) | 中文语义边界切分 |
| 语言检测 | ASCII 字母占比 >60% → 英文, 否则 → 中文 | 前 500 字符采样 |

**Parent-Child 设计**：子块用于检索（粒度细、匹配准），检索到子块后返回其父块完整内容，保证 LLM 获得充足上下文。

### 3.6 三路多模态检索

| 通道 | Collection | 数据量 | Embedding | 检索方式 |
|------|-----------|--------|-----------|---------|
| **文本** | `paper_rag` | 1,033 块 (283 父块) | BGE-M3 dense + sparse | 混合检索 + Reranker |
| **图表** | `paper_rag_figures` | 69 张 | BGE-M3 dense (caption+description) | ANN (IP, nprobe=10) |
| **表格** | `paper_rag_tables` | 97 个 | BGE-M3 dense (markdown) | ANN (IP, nprobe=10) |

**合并策略（方案C — 配额保底 + 统一重排）**：

```
文本 top-K(6) → Reranker 预排 → 进竞争池
图表 top-1 ──────────────────→ 保底入围（不入围则让给文本）
表格 top-1 ──────────────────→ 保底入围（不入围则让给文本）
额外图表 + 额外表格 ──────────→ 进竞争池

竞争池 → Cross-Encoder 统一打分 → 填满剩余槽位 → 全局 top-5
```

**设计思路**：
- 保底机制确保论文中关键的图表/表格至少 1 张进入 LLM 上下文
- 剩余槽位由文本 + 额外图表 + 额外表格经 Reranker 跨模态公平竞争
- 若某类型无结果，配额自动让给其他类型，零侵蚀正常场景
- 相比旧版 `[:CANDIDATE_M]` 硬截断（图表表格永远被挤占），按类型配额 + 统一打分兼顾了多模态覆盖和质量择优

### 3.7 论文数据管线

| 模块 | 文件 | 功能 | 关键技术 |
|------|------|------|---------|
| **论文获取** | `arxiv_fetcher.py` | arXiv API 检索 + PDF 下载 | `urllib` + `xml.etree` (零外部依赖), 3s 频率控制 |
| **PDF 加载** | `paper_pdf_loader.py` | 数字原生 PDF 文本提取 | PyMuPDF (无 OCR, 轻量) |
| **图表提取** | `figure_extractor.py` | PDF → 图片 + 标题 | PyMuPDF 图片提取 + 近邻标题匹配 + Ollama 视觉描述 |
| **表格提取** | `table_extractor.py` | PDF 内嵌表格 → Markdown | pdfplumber (PaddleOCR PPStructure 备用) |

### 3.8 RAG Prompt 设计

```
你是一个学术论文研究助手，负责帮助用户理解和分析学术论文。请按照以下步骤处理：

1. 分析问题和上下文：
   - 基于提供的上下文（如果有）和你的知识回答问题。
   - 如果答案来源于检索到的论文内容，请在回答中明确引用。
   - 涉及专业术语时，首次出现需标注中英文。

2. 评估对话历史：
   - 检查对话历史是否与当前问题相关。
   - 如果相关，结合历史信息生成更准确的回答。
   - 如果无关，忽略历史，仅基于上下文和问题回答。

3. 生成回答：
   - 提供清晰、准确、结构化的回答，避免无关信息。
   - 如果涉及多篇论文的对比，使用结构化方式呈现。
   - 如果上下文和历史消息均不足以回答问题，回复信息不足提示。
```

### 3.9 LLM 调用容错

| 机制 | 参数 |
|------|------|
| 重试策略 | 指数退避 + 随机抖动 (`base/retry.py` → `@retry_with_backoff` 装饰器) |
| 最大重试次数 | 3 |
| 退避算法 | `min(1s × 2^attempt + random(0,1), 30s)` |
| 超时 | 30s (stream=True) |
| Fallback | 所有重试耗尽后返回友好错误提示 |
| 模型 | `deepseek-v4-pro` (API: `api.deepseek.com`) |

> **v2.4 优化**：`call_dashscope()` 原有手写重试逻辑替换为 `base/retry.py` 的 `@retry_with_backoff` 装饰器。连接建立阶段错误自动重试（最多 3 次），流式传输中的错误不重试（避免 token 浪费和重复输出）。 |

---

## 四、技术参数速查表

### 模型规格

| 模型 | 用途 | 维度/规格 |
|------|------|----------|
| BGE-M3 | 文本/图表/表格 Embedding | Dense: 1024-dim, Sparse: 词汇级权重 |
| BGE-Reranker-Large | Cross-Encoder 重排序 | ~560M 参数 |
| BERT-Base-Chinese | 查询意图分类 | 12层, 768-hidden, 110M 参数 |
| DeepSeek-v4-pro | 生成/策略选择/HyDE | API 调用, 上下文窗口 128K |

### 检索参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `retrieval_k` | 6 | 混合检索返回数量（优化后，原 3） |
| `candidate_m` | 5 | 重排序后最终上下文数量（优化后，原 2） |
| `parent_chunk_size` | 1,800 | 父块大小（优化后，原 1,200） |
| `child_chunk_size` | 400 | 子块大小（优化后，原 300） |
| `nprobe` | 10 | IVF 索引搜索的聚类数 |
| Dense weight | 0.7 | 稠密检索在混合检索中的权重 |
| Sparse weight | 1.0 | 稀疏检索在混合检索中的权重 |
| BM25 threshold | 0.85 | MySQL 精确匹配阈值 |

### 基础设施

| 组件 | 版本 | 端口 | 用途 |
|------|------|------|------|
| Milvus | 2.4.4 | 19530 | 向量存储与检索 |
| MySQL | 8.0 | 3306/3307 | 对话历史 + BM25 全文索引 |
| Redis | latest | 6379 | BM25 数据预热 + 查询结果缓存 (L1/L2) |
| etcd | v3.5.5 | 2379 | Milvus 元数据协调 |
| MinIO | latest | 9000/9001 | Milvus 对象存储 |
| FastAPI | latest | 8080 | WebSocket + REST API |

### Milvus 索引配置

| 索引 | 类型 | 参数 |
|------|------|------|
| dense_vector | IVF_FLAT | nlist=128, metric_type=IP |
| sparse_vector | SPARSE_INVERTED_INDEX | drop_ratio_build=0.2 |
| 主键模式 | MD5(page_content) hex → VARCHAR(100) | 支持 upsert, 去重 |

---

## 五、项目迭代历程

### v1.0 — IT 教育智能问答系统 (初始版本)

- **背景**：面向 IT 培训机构的学科知识问答
- **架构**：MySQL BM25 → BERT 分类 → RAG (Milvus + LLM) → 流式输出
- **数据**：IT 学科教材/讲义（AI、Java、测试、运维、大数据 5 个领域）
- **局限**：领域耦合强，无法直接复用到论文场景

### v2.0 — 重构为论文阅读 Agent (PaperRAG)

**迭代决策**：保留核心 RAG 管道，替换领域层并新增论文独有模块。

| 改动维度 | 具体内容 |
|---------|---------|
| **数据源** | IT 教材 → 19 篇 arXiv 经典论文 (cs.AI / cs.CV / cs.CL / cs.LG) |
| **配置切换** | database/collection → `paper_rag`, 领域标签 → `["cs","nlp","cv","ai"]` |
| **数据获取** | 新增 `arxiv_fetcher.py` — arXiv API 搜索 + PDF 批量下载 |
| **文档加载** | 新增 `paper_pdf_loader.py` — PyMuPDF 轻量 PDF 加载 (学术论文无需 OCR) |
| **分块优化** | 新增英文分块器 — tiktoken `cl100k_base` 编码, 自动中/英文检测 |
| **图表模块** | 新增 `figure_extractor.py` — PyMuPDF 提取图片 + Ollama 视觉描述 |
| **表格模块** | 新增 `table_extractor.py` — pdfplumber 提取内嵌表格 → Markdown |
| **三路检索** | 文本 + 图表 + 表格 三路并行检索 → 合并 → Reranker 精排 |
| **策略提示词** | 示例从 "AI学科学费" → "Transformer 核心贡献是什么" |
| **RAG 提示词** | 从"智能助手" → "学术论文研究助手"，新增术语中英标注、论文引用规则 |
| **前端** | 标题/欢迎语/快捷提问全部切换为论文场景 |
| **分类器** | 重新标注训练数据、重训练 BERT 模型 (通用问答 / 论文学术咨询) |

### v2.1 — GPU 适配与 Docker 部署

| 迭代点 | 详情 |
|--------|------|
| **GPU 适配** | RTX 5070 (Blackwell sm_120) 需要 CUDA ≥ 12.8 → 选择 `pytorch:2.7.0-cuda12.8-cudnn9-runtime` 基础镜像 |
| **Docker Compose** | 6 服务编排 (app + mysql + redis + milvus + etcd + minio)，GPU 直通 (`deploy.resources.reservations.devices`) |
| **配置安全** | API Key 从 `config.ini` 迁移到 `.env` → `load_dotenv()` → `.gitignore` 排除 |
| **代码安全** | `eval()` → `json.loads()` 解析 JSON, CORS 限制 `localhost:8080` |
| **端口隔离** | MySQL 容器端口映射 `3307:3306` (避免与宿主机 MySQL 冲突) |

### v2.2 — Source Citation 修复

| 问题 | 根因 | 修复 |
|------|------|------|
| 前端始终显示"参考来源 1 篇" | `paper_id` 未写入 Milvus + `_doc_form_hit()` 未提取 | `document_processor.py` 从文件名提取 arXiv ID → `add_documents()` 写入 → `output_fields` 包含 → `_doc_form_hit()` 携带 |

### v2.3 — 检索参数优化 + 三路合并重构

| 迭代点 | 详情 |
|--------|------|
| **参数优化** | `candidate_m` 2→5, `retrieval_k` 3→6, `parent_chunk_size` 1200→1800, `child_chunk_size` 300→400 |
| **子块碎片化改善** | 更大子块 (400) 减少同父块扎堆，总数从 1,404 降至 1,033 (↓26%) |
| **LLM 上下文扩充** | 最终上下文从 ~2,400 字增至 ~9,000 字 |
| **三路合并重构（方案C）** | 图表 top-1 + 表格 top-1 保底入围 → 剩余槽位统一 Cross-Encoder 重排竞争 |
| **效果** | 图表/表格从"永远被挤占"变为"100% 至少 1 张入围"，跨模态公平打分 |
| **jpkb 论文 FAQ 改造** | 从 19 篇论文中提炼 26 条高频问答（nlp/cv/ai 三领域），写入 jpkb 表，BM25 + Redis 缓存链路重新激活 |

### v2.4 — 废弃模块清理 + 重试逻辑集成

| 迭代点 | 详情 |
|--------|------|
| **删除旧入口点** | 移除 `old_main.py`（v1.0 入口）、`rag_qa/main.py`（独立 CLI） |
| **删除旧 RAG 系统** | 移除 `rag_qa/core/rag_system.py`（v1.0 旧版，已被 `new_rag_system.py` 替代） |
| **删除旧加载器** | 移除 `pdf_loader.py`（OCR 加载器）、`doc_loader.py`（DOCX）、`ppt_loader.py`（PPT）—— 论文场景不需要 |
| **保留 OCR 模块** | `ocr.py` 保留，仍被 `img_loader.py` 使用（图片型问答场景） |
| **清理引用** | 更新 `document_processor.py` 移除条件导入、`document_loaders/__init__.py` 精简为仅 `img_loader` |
| **重试逻辑集成** | `new_main.py` 中的 `call_dashscope()` 从手写重试逻辑改为使用 `base/retry.py` 的 `@retry_with_backoff` 装饰器 |
| **效果** | 项目更轻量，去除非论文场景的冗余模块，重试逻辑集中管理便于维护 |

---

## 六、关键设计决策与权衡

| 决策 | 选择 | 原因 |
|------|------|------|
| 嵌入模型 | BGE-M3 (非 OpenAI Embedding) | 中英双语原生支持、免费本地推理、同时输出 dense+sparse |
| Milvus vs FAISS | Milvus | 持久化存储、分布式支持、混合检索原生支持、生产级 |
| Parent-Child vs 固定分块 | Parent-Child | 检索精度（小粒度匹配）+ 上下文完整性（大粒度喂 LLM）|
| 两阶段检索 (BM25+RAG) | 保留 BM25 + Redis 缓存 | 论文 FAQ 精确匹配延迟 <5ms（Redis 命中）或 <50ms（BM25），远低于 RAG 的 2-3s；两级缓存（Redis → BM25 → RAG）逐级降级保证可用性 |
| PyMuPDF vs OCR | PyMuPDF (数字原生) | 学术论文 99% 是数字原生 PDF，OCR 不仅慢且无必要 |
| Ollama 视觉 vs API | Ollama 本地 | 成本考虑，`llama3.2-vision:11b` 免费本地推理 |
| DeepSeek vs GPT | DeepSeek API | 性价比高、中文能力强、128K 上下文 |
| WebSocket vs SSE | WebSocket | 双向通信、逐 token 流式推送、打字机效果 |
| Docker vs 纯 Conda | Docker | 环境一致性、一键部署、GPU 直通支持 |
| 三路合并策略 | 配额保底 + 统一重排 | 保证多模态覆盖（图表/表格不丢失）+ Reranker 跨模态公平择优 |

---

## 七、可面试讨论的延伸方向

1. **混合检索权重调优**：dense=0.7/sparse=1.0 如何通过离线评估（RAGAS）确定最优值？
2. **Reranker 的替代方案**：Cross-Encoder vs ColBERT vs LLM-as-Reranker 的延迟/效果 trade-off
3. **图表理解的升级路径**：Ollama 本地视觉模型 → GPT-4V API → 多模态 Embedding (如 Jina CLIP)
4. **查询分类的扩展**：二分类 → 多分类（论文检索/图表问答/公式解释/实验复现）
5. **增量索引**：新论文到达时如何做增量入库而不全量重跑？
6. **评估体系**：RAGAS 指标 (faithfulness/answer_relevancy/context_precision/context_recall) 的自动化 pipeline
7. **对话记忆**：当前保留最近 5 轮 → 可升级为摘要记忆 / 向量记忆
8. **Agent 化**：当前是单轮 RAG → 可扩展为 ReAct Agent（论文检索 + 公式推导 + 实验复现工具链）
9. **多模态合并策略演进**：固定配额 → 统一重排 → 配额+重排混合 → 查询自适应分配（根据 query 类型动态调整图文表比例）
10. **分块参数调优方法论**：如何通过 RAGAS 评估 + A/B 测试确定最优 parent/child chunk size？
11. **两级缓存架构演进**：Redis L1 → BM25 L2 → RAG L3 的逐级降级策略，缓存预热、失效与更新机制

---

## 八、部署与运行

### 本地 Conda 开发

```bash
cd integrated_qa_system
python app.py
# → http://localhost:8080
```

### Docker Compose 生产部署

```bash
docker compose up -d
# → http://localhost:8080
```

### 环境变量优先级

```
.env 文件 → 系统环境变量 → config.ini 文件
```

---

*文档生成时间: 2026-08-09 | 项目分支: main | 最新 commit: 23b3b15*
