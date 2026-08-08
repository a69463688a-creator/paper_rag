import torch.cuda

from milvus_model.hybrid import BGEM3EmbeddingFunction
from  pymilvus import MilvusClient,DataType,AnnSearchRequest,WeightedRanker
from  langchain_core.documents import Document
#用于重排和NLI判断 优化结果相关性排序
from sentence_transformers import CrossEncoder

#作为主键
import hashlib

import sys,os

from sqlalchemy.testing.suite.test_reflection import metadata


local_path=os.path.abspath(os.path.dirname(__file__))

rag_qa_path=os.path.abspath(os.path.dirname(local_path))
# print(rag_qa_path)

project_root=os.path.dirname(rag_qa_path)
# print(project_root)

sys.path.insert(0,project_root)

from .document_processor import *
from base.config import Config
from base.logger import logger

conf=Config()

class VectorStore:
    def __init__(
            self,
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME
    ):
        self.collection_name=collection_name
        self.host=host
        self.port=port
        self.database=database

        self.logger=logger
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备:{self.device}')


        reranker_path=os.path.join(rag_qa_path,'models','bge-reranker-large')
        # print(reranker_path)

        self.reranker=CrossEncoder(reranker_path,device=self.device)

        m3_path=os.path.join(rag_qa_path,'models','bge-m3')
        self.embedding_function=BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            device=self.device,
            use_fp16=False #gpu时使用半精度计算 减少内存消耗 加速
        )

        self.dense_dim = self.embedding_function.dim['dense']
        # print(self.dense_dim)



        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}",db_name=self.database)

        self._create_or_load_collection()

    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    def add_documents(self, documents):
        texts=[doc.page_content for doc in documents]

        embeddings=self.embedding_function(texts)

        data=[]
        for i,doc in enumerate(documents):
            #字符串转为字节流  MD5哈希需要字节
            text_hash=hashlib.md5(doc.page_content.encode('utf-8')).hexdigest() #十六进制 三十二位
            sparse_vector={}

            row=embeddings['sparse'][[i]]
            # print(row)
            # print(row.shape)
            #(0, 202296)	0.18399202823638916   (1, 250002)

            indices=row.indices
            # print(indices)

            values=row.data #非零元素的值的列表

            for idx,value in zip(indices, values):
                sparse_vector[idx]=value
                # print(sparse_vector)
                # print(embeddings['dense'][i])
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f'插入{len(data)}条数据到集合{self.collection_name}')




    def hybrid_search_with_rerank(self,query,k=conf.RETRIEVAL_K,source_filter=None):
        """

        :param query:
        :param k: topk子块
        :param source_filter:
        :return: topm父块
        """
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])
        # 获取查询的稠密向量
        dense_query_vector = query_embeddings["dense"][0]
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # 获取查询稀疏向量的第 0 行数据 仅一个查询
        row = query_embeddings["sparse"][[0]]
        # 获取稀疏向量的非零值索引
        indices = row.indices
        # 获取稀疏向量的非零值
        values = row.data
        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        filter_expr=f"source=='{source_filter}'"if source_filter else ""
        # print(filter_expr)

        dense_request=AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={'metric_type': 'IP','params': {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )

        sparse_request=AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={'metric_type': 'IP','params': {}},
            limit=k,
            expr=filter_expr
        )

        ranker = WeightedRanker(0.7, 1.0)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]
        # print(results)
        # print(type(results))
        # print(len(results))

        sub_chunks=[self._doc_form_hit(hit["entity"])for hit in results]
        # print(sub_chunks)
        # print(len(sub_chunks))

        parent_docs=self._get_unique_parent_docs(sub_chunks)
        # print(parent_docs)
        # print(len(parent_docs))

        if len(parent_docs)<2:
            return parent_docs[:conf.CANDIDATE_M]

        if parent_docs:
            pairs=[[query,doc.page_content] for doc in parent_docs]
            scores=self.reranker.predict(pairs)
            # print(scores)
            ranked_parent_docs=[doc for _,doc in sorted(zip(scores,parent_docs),key=lambda x:x[0],reverse=True)] #******排序问题 score一致会排序document导致错误
        else:
            ranked_parent_docs=[]
        return ranked_parent_docs[:conf.CANDIDATE_M]




    def _get_unique_parent_docs(self,sub_chunks):
        parent_contents=set()
        unique_docs=[]
        for chunk in sub_chunks:
            parent_content=chunk.metadata.get('parent_content',chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                unique_docs.append(Document(page_content=parent_content,metadata=chunk.metadata))
                parent_contents.add(parent_content)
        return unique_docs

    def _doc_form_hit(self,hit):
        return Document(
            page_content=hit.get('text'),
            metadata={
                'parent_id': hit.get('parent_id'),
                'parent_content': hit.get('parent_content'),
                'source': hit.get('source'),
                'timestamp': hit.get('timestamp')
            }
        )





if __name__ == '__main__':
    vector_store=VectorStore()

    # directory_path='../data/ai_data'
    # # print(vector_store.embedding_function.dim)
    #
    # documents=process_documents(directory_path)
    # vector_store.add_documents(documents)


    query='AI学科的课程内容是什么'
    results=vector_store.hybrid_search_with_rerank(query,source_filter='ai')

    print(results)
    print(len(results))