# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入langchain_openai的嵌入模型和聊天模型，用于评估时的语义计算和推理
import json

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_ollama import ChatOllama,OllamaEmbeddings

with open('rag_evaluate_data.json','r',encoding='utf-8') as f:
    data = json.load(f)
eval_data={
    'questions':[item['question'] for item in data],
    'answers':[item['answer'] for item in data],
    'contexts':[item['context'] for item in data],
    'ground_truth':[item['ground_truth'] for item in data]
}
dataset=Dataset.from_dict(eval_data)

llm=ChatOllama(model='qwen2.5:7b',base_url='http://localhost:11434')
embeddings=OllamaEmbeddings(model='qwen2.5:7b',base_url='http://localhost:11434')

result=evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=embeddings
)
print(f'RAGAS评估结果为{result}')

