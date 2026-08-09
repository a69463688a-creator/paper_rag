# PaperRAG — 学术论文阅读助手
# 宿主机需装 NVIDIA 驱动 + Docker Desktop（已自带 GPU 支持）
# 启动方式: docker compose up -d (自动启用 GPU)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.title="PaperRAG"
LABEL org.opencontainers.image.description="RAG-based academic paper Q&A system"

# pip 镜像加速（国内环境）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

WORKDIR /app

# 先复制干净的依赖文件（UTF-8，不含 torch，基础镜像已自带）
COPY integrated_qa_system/requirements-docker.txt requirements.txt

# 安装系统依赖（独立层，网络失败不影响后续 pip 层缓存）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 生成 torch 版本约束文件，防止 pip 将 torch 从 2.6.0 升级到 2.13.0（会下载 2GB+ CUDA 包）
RUN python -c "import torch; v=torch.__version__; print(f'torch=={v}')" > /tmp/constraints.txt \
    && python -c "import torchvision; print(f'torchvision=={torchvision.__version__}')" >> /tmp/constraints.txt \
    && python -c "import torchaudio; print(f'torchaudio=={torchaudio.__version__}')" >> /tmp/constraints.txt \
    && echo "=== constraints ===" && cat /tmp/constraints.txt

# 安装 Python 依赖（先升级 pip 修复 local version 解析 bug，约束 torch 版本防止被升级到 2.13）
RUN pip install --no-cache-dir --upgrade pip \
    && PIP_CONSTRAINT=/tmp/constraints.txt pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY integrated_qa_system/ .

# 模型和论文数据通过 volume 挂载，不打包进镜像
# - /app/rag_qa/models/   → BERT 模型文件
# - /app/rag_qa/data/     → 论文数据

EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "app.py"]
