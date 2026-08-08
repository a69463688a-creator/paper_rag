import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter
from rag_qa.edu_text_spliter import AliTextSplitter
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base.config import Config
from base.logger import logger
# import nltk
# nltk.download('punkt_tab')

conf = Config()
# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

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
                        doc.metadata['source'] = source  # 学科
                        doc.metadata['file_path'] = file_path
                        doc.metadata['timestamp'] = datetime.now().isoformat()
                    documents.extend(loaded_docs)
                    logger.info(f'成功加载文件:{file_path}')
                except Exception as e:
                    logger.error(f'加载文件失败:{file_path}')
            else:
                logger.warning(f'不支持文件类型:{file_path}')
    return documents

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

    child_chunks = []

    for i,doc in enumerate(documents):
        file_extension=os.path.splitext(doc.metadata.get('file_path',''))[1].lower()
        is_markdown = (file_extension =='.md')
        parent_splitter_to_use=markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use=markdown_child_splitter if is_markdown else child_splitter
        logger.info(f'处理文档:{doc.metadata["file_path"]},使用切分器:{"markdown" if is_markdown else "ChineseRecursive"}')

        parent_docs=parent_splitter_to_use.split_documents([doc])#函数需要可迭代类型 返回Document对象

        for j ,parent_doc in enumerate(parent_docs):
            parent_id=f'doc_{i}_parent_{j}'
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata['content']=parent_doc.page_content
            # print(f'父块:{parent_doc.metadata["file_path"]},索引:{j},唯一id:{parent_id}\n') 输出: 父块:../data/ai_data\LLM基础知识.pdf,索引:0,唯一id:doc_0_parent_0


            sub_chunks=child_splitter_to_use.split_documents([parent_doc])
            for k,sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata['parent_id']=parent_id
                sub_chunk.metadata['parent_content']=parent_doc.page_content

                sub_chunk.metadata['id']=f'{parent_id}_child_{k}'

                child_chunks.append(sub_chunk)

    logger.info(f'切分后子块总数为:{len(child_chunks)}')
    return child_chunks





if __name__ == '__main__':
    # directory_path='../data/ai_data'
    # documents=load_documents_from_directory(directory_path)
    # print(len(documents))
    # print(documents[0]if documents else'未加载到')

    chunks=process_documents('../data/ai_data',conf.PARENT_CHUNK_SIZE,conf.CHILD_CHUNK_SIZE,conf.CHUNK_OVERLAP)
    print(len(chunks))
    print(chunks[0])