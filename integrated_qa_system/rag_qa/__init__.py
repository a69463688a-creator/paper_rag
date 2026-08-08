import os, sys

current_dir = os.path.abspath(__file__)
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, rag_qa_path)

from core.prompts import RAGPrompts

# 以下模块依赖外部服务（Milvus 等），仅在使用时导入
# from core.vector_store import VectorStore
# from core.new_rag_system import RAGSystem
