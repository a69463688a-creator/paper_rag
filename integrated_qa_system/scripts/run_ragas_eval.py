"""
RAGAS 评估（ragas 0.2.x）：
    忠实度 faithfulness / 答案相关性 answer_relevancy /
    上下文精确度 context_precision / 上下文召回率 context_recall

用法（在 paperrag-app 容器内）:
    cd /app && python scripts/run_ragas_eval.py

流程:
    1. 真实走一遍检索管线 -> 拿到 answer + 实际检索到的 contexts
    2. 用 DeepSeek(生产 LLM) + 本地 BGE-M3 作为 ragas 的 judge/embeddings
    3. 输出四项分数并保存 ragas_result.json
"""
import os, json, sys
# 让脚本从任意目录运行都能 import base / new_main（指向 integrated_qa_system/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall
)
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings


# ── 补丁：修复 ragas 0.2.6 fix_output_format 重试 bug ──
# 原实现解析失败后走 fix_output_format 重试，但直接把返回的 StringIO 当结果，
# 导致下游访问 .statements/.question 时 AttributeError。这里改为重新解析回 OutputModel。
from ragas.prompt import pydantic_prompt as _rp
from ragas.prompt.utils import extract_json as _extract_json
from langchain_core.exceptions import OutputParserException as _OPE

async def _parse_output_string_fixed(self, output_string, prompt_value, llm, callbacks, retries_left=1):
    callbacks = callbacks or []
    try:
        jsonstr = _extract_json(output_string)
        result = super(_rp.RagasOutputParser, self).parse(jsonstr)
    except _OPE:
        if retries_left != 0:
            retry_rm, retry_cb = _rp.new_group(
                name="fix_output_format",
                inputs={"output_string": output_string},
                callbacks=callbacks,
            )
            fixed = await _rp.fix_output_format_prompt.generate(
                llm=llm,
                data=_rp.OutputStringAndPrompt(
                    output_string=output_string,
                    prompt_value=prompt_value.to_string(),
                ),
                callbacks=retry_cb,
                retries_left=retries_left - 1,
            )
            retry_rm.on_chain_end({"fixed_output_string": fixed})
            # 修复点：从 StringIO.text 重新提取 JSON 并解析，而不是直接返回 StringIO
            try:
                result = super(_rp.RagasOutputParser, self).parse(_extract_json(fixed.text))
            except _OPE:
                raise _rp.RagasOutputParserException()
        else:
            raise _rp.RagasOutputParserException()
    return result

_rp.RagasOutputParser.parse_output_string = _parse_output_string_fixed

from base.config import Config
from new_main import IntegratedQASystem

# 论文库专属问题 + 人工标注标准答案要点（ground_truth）
EVAL_QUESTIONS = [
    {"question": "Transformer 的 self-attention 机制是如何工作的？",
     "ground_truth": "注意力机制通过 Query、Key、Value 计算加权求和，多头注意力并行关注不同子空间，完全摒弃循环和卷积结构。"},
    {"question": "BERT 的预训练任务有哪些？",
     "ground_truth": "掩码语言模型（MLM）和下一句预测（NSP）两个无监督预训练任务。"},
    {"question": "ResNet 如何解决深层网络的梯度消失问题？",
     "ground_truth": "引入残差学习框架，通过捷径连接（shortcut/skip connection）使梯度直接流过恒等映射。"},
    {"question": "GAN 由哪两个网络组成，作用分别是什么？",
     "ground_truth": "生成器捕捉数据分布并生成样本，判别器判断样本真伪，二者对抗训练。"},
    {"question": "Vision Transformer 如何将图像转换为序列输入？",
     "ground_truth": "将图像切分为固定大小 patch，线性投影为 patch embedding 并加位置编码后输入 Transformer。"},
    {"question": "WGAN 相比原始 GAN 在训练上做了哪些改进？",
     "ground_truth": "用 Wasserstein 距离替代 JS 散度度量分布差异，缓解训练不稳定和模式坍塌。"},
    {"question": "GPT-3 的 few-shot 学习能力有什么特点？",
     "ground_truth": "1750 亿参数，仅通过上下文提示进行 few-shot 学习，无需对下游任务微调。"},
    {"question": "Adam 优化器结合了哪两种优化方法？",
     "ground_truth": "结合动量（Momentum）与 RMSProp 的自适应学习率。"},
    {"question": "YOLO 如何将目标检测转化为回归问题？",
     "ground_truth": "单次前向传播同时预测边界框和类别概率，将检测转化为回归。"},
    {"question": "DQN 如何稳定强化学习训练？",
     "ground_truth": "使用经验回放（experience replay）和目标网络（target network）缓解样本相关性和目标漂移。"},
    {"question": "VGG 为什么使用多个小卷积核？",
     "ground_truth": "堆叠 3x3 小卷积核在相同感受野下增加深度和非线性，参数量更少。"},
    {"question": "CycleGAN 如何实现无配对图像翻译？",
     "ground_truth": "通过循环一致性损失（cycle consistency loss）约束两个生成器互为逆映射。"},
]


class LocalBGEM3(Embeddings):
    """把已加载的本地 BGE-M3 包装成 LangChain 兼容 embeddings，复用 qa 的模型避免二次加载"""

    def __init__(self, ef):
        self.ef = ef

    def embed_documents(self, texts):
        return [v.tolist() for v in self.ef(list(texts))["dense"]]

    def embed_query(self, text):
        return self.ef([text])["dense"][0].tolist()


def build_records(qa):
    records = []
    for item in EVAL_QUESTIONS:
        q = item["question"]
        answer_parts = []
        for tup in qa.query(q, session_id=None):
            token, _, _ = tup if len(tup) == 3 else (tup[0], tup[1], None)
            if token:
                answer_parts.append(token)
        contexts = getattr(qa.rag_system, "last_contexts", []) or [""]
        records.append({
            "user_input": q,
            "response": "".join(answer_parts),
            "retrieved_contexts": contexts,
            "reference": item["ground_truth"],
        })
        print(f"  完成: {q[:20]}... (contexts={len(contexts)})")
    return records


def main():
    qa = IntegratedQASystem()
    conf = Config()
    print("生成 answer + 检索 context ...")
    records = build_records(qa)

    dataset = Dataset.from_dict({
        "user_input": [r["user_input"] for r in records],
        "response": [r["response"] for r in records],
        "retrieved_contexts": [r["retrieved_contexts"] for r in records],
        "reference": [r["reference"] for r in records],
    })

    # judge 用生产 DeepSeek（deepseek-v4-pro，稳定快速 + JSON 可靠），本地 BGE-M3 作 embeddings
    llm = ChatOpenAI(model=conf.LLM_MODEL, api_key=conf.DASHSCOPE_API_KEY,
                     base_url=conf.DASHSCOPE_BASE_URL, temperature=0)
    embeddings = LocalBGEM3(qa.vector_store.embedding_function)

    # 拉长单条超时，避免慢速本地模型被误判为 Timeout
    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(timeout=600, max_retries=10)
    except Exception:
        run_config = None

    print("运行 RAGAS 评估 ...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm, embeddings=embeddings,
        **({"run_config": run_config} if run_config else {}),
    )
    print(result)

    try:
        scores = {k: float(v) for k, v in result.items()}
    except AttributeError:
        # ragas 0.2.x 返回 EvaluationResult，用 to_pandas 取各指标均值
        df = result.to_pandas()
        scores = {c: float(df[c].mean()) for c in df.columns
                  if c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    with open("ragas_result.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print("结果已保存: ragas_result.json")


if __name__ == "__main__":
    main()
