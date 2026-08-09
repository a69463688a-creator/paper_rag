import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from datetime import datetime
from rag_qa.text_spliter import ChineseRecursiveTextSplitter
from rag_qa.document_loaders import OCRIMGLoader

# 论文 PDF 使用轻量加载器（无需 OCR），扫描件回退到 OCRPDFLoader
from rag_qa.paper_data.paper_pdf_loader import PaperPDFLoader

# 以下 loader 依赖 python-docx / python-pptx，仅在可用时导入
try:
    from rag_qa.document_loaders.doc_loader import OCRDOCLoader
except ImportError:
    OCRDOCLoader = None

try:
    from rag_qa.document_loaders.ppt_loader import OCRPPTLoader
except ImportError:
    OCRPPTLoader = None

from base.config import Config
from base.logger import logger
# import nltk
# nltk.download('punkt_tab')

conf = Config()
# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 PaperPDFLoader（数字原生 PDF，无需 OCR）
    ".pdf": PaperPDFLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# 仅在 loader 可用时注册
if OCRDOCLoader is not None:
    document_loaders[".docx"] = OCRDOCLoader
if OCRPPTLoader is not None:
    document_loaders[".ppt"] = OCRPPTLoader
    document_loaders[".pptx"] = OCRPPTLoader

def load_documents_from_directory(directory_path):
    """
    递归的加载路径内所有文件
    :param directory_path:
    :return: 加载后文档列表，每个元素为LangChain Document对象，含page_content和metadata
    """
    documents = []
    supported_extensions=document_loaders.keys()
    source=os.path.basename(directory_path).replace('_data','')
    for root,_,files in os.walk(directory_path):
        for file in files:
            file_path=os.path.join(root,file)
            # print(file_path)

            file_extension=os.path.splitext(file)[1].lower()
            # print(file_extension)
            if file_extension in supported_extensions:
                try:
                    loader_class = document_loaders[file_extension]
                    if file_extension == '.txt':
                        loader = loader_class(file_path, encoding='utf-8')
                    else:
                        loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    # print(loaded_docs)
                    for doc in loaded_docs:
                        doc.metadata['source'] = source  # 论文领域
                        doc.metadata['file_path'] = file_path
                        # 从文件名提取论文 ID（arXiv ID），如 "1706.03762.pdf" → "1706.03762"
                        doc.metadata['paper_id'] = os.path.splitext(os.path.basename(file_path))[0]
                        doc.metadata['timestamp'] = datetime.now().isoformat()
                    documents.extend(loaded_docs)
                    logger.info(f'成功加载文件:{file_path}')
                except Exception as e:
                    logger.error(f'加载文件失败:{file_path}')
            else:
                logger.warning(f'不支持文件类型:{file_path}')
    return documents

def _detect_language(text: str, sample_size: int = 500) -> str:
    """
    检测文本语言，用于选择合适的分块器

    取文本前 sample_size 个字符，统计 ASCII 字母占比：
      > 60% → 英文（使用 RecursiveCharacterTextSplitter）
      否则 → 中文（使用 ChineseRecursiveTextSplitter）

    :param text: 待检测文本
    :param sample_size: 采样长度
    :return: "en" 或 "zh"
    """
    sample = text[:sample_size] if len(text) > sample_size else text
    if not sample:
        return "zh"  # 空文本默认中文
    ascii_letters = sum(1 for c in sample if c.isascii() and c.isalpha())
    ratio = ascii_letters / len(sample)
    return "en" if ratio > 0.6 else "zh"


def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP):
    """
    加载文档并切分 先切大块再子块，子块关联父块元数据
    :param directory_path: 文档目录路径
    :param parent_chunk_size: 大粒度
    :param child_chunk_size: 小粒度
    :param chunk_overlap: 子块重叠度 确保关联
    :return: child_chunks：切分后的子块列表，每个都含有父块信息
    """
    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档数量: {len(documents)}")

    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 英文论文分块器: 使用 tiktoken 编码，按段落/句子边界递归切分
    english_parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    english_child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    child_chunks = []

    for i,doc in enumerate(documents):
        file_extension=os.path.splitext(doc.metadata.get('file_path',''))[1].lower()
        is_markdown = (file_extension =='.md')
        # 检测文档语言：Markdown 文件保持用 Markdown 分块器；其他文件按语言选择中/英文分块器
        if is_markdown:
            parent_splitter_to_use = markdown_parent_splitter
            child_splitter_to_use = markdown_child_splitter
            splitter_label = "markdown"
        else:
            lang = _detect_language(doc.page_content)
            if lang == "en":
                parent_splitter_to_use = english_parent_splitter
                child_splitter_to_use = english_child_splitter
                splitter_label = "english"
            else:
                parent_splitter_to_use = parent_splitter
                child_splitter_to_use = child_splitter
                splitter_label = "chinese"
        logger.info(f'处理文档:{doc.metadata["file_path"]},使用切分器:{splitter_label}')

        parent_docs=parent_splitter_to_use.split_documents([doc])#函数需要可迭代类型 返回Document对象

        for j ,parent_doc in enumerate(parent_docs):
            parent_id=f'doc_{i}_parent_{j}'
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata['content']=parent_doc.page_content
            # print(f'父块:{parent_doc.metadata["file_path"]},索引:{j},唯一id:{parent_id}\n') 输出: 父块:../data/paper_data\LLM基础知识.pdf,索引:0,唯一id:doc_0_parent_0


            sub_chunks=child_splitter_to_use.split_documents([parent_doc])
            for k,sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata['parent_id']=parent_id
                sub_chunk.metadata['parent_content']=parent_doc.page_content

                sub_chunk.metadata['id']=f'{parent_id}_child_{k}'

                child_chunks.append(sub_chunk)

    logger.info(f'切分后子块总数为:{len(child_chunks)}')
    return child_chunks





if __name__ == '__main__':
    # directory_path='../data/paper_data'
    # documents=load_documents_from_directory(directory_path)
    # print(len(documents))
    # print(documents[0]if documents else'未加载到')

    chunks=process_documents('../data/paper_data',conf.PARENT_CHUNK_SIZE,conf.CHILD_CHUNK_SIZE,conf.CHUNK_OVERLAP)
    print(len(chunks))
    print(chunks[0])