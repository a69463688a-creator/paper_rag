from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, FileResponse

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from starlette.websockets import WebSocketDisconnect

import os

from pydantic import BaseModel

import asyncio

import json

import uuid

from typing import Optional, List, Dict, Any

import time

import re

from new_main import IntegratedQASystem

app=FastAPI(title="论文阅读助手API",description='集成MySQL和RAG的学术论文问答系统')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境需限制）
    allow_credentials=True,  # 允许凭证 Cookie
    allow_methods=["*"],  # 允许所有 HTTP 方法 GET/POST等
    allow_headers=["*"],  # 允许所有头部
)
os.makedirs("static",exist_ok=True)

qa_system = IntegratedQASystem()

GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "你好！我是论文阅读助手，专注于学术论文检索与分析，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是论文阅读助手，致力于提供学术论文相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在！我是论文阅读助手，随时为你解答问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，随时为你解答学术论文相关的问题！有什么我可以帮你的？"
    }
]

class QueryRequest(BaseModel):
    query: str  # 查询内容，必填
    source_filter: Optional[str] = None  # 论文领域过滤，可选
    session_id: Optional[str] = None  # 会话 ID，可选

class QueryResponse(BaseModel):
    answer: str  # 答案内容
    is_streaming: bool  # 是否流式响应
    session_id: str  # 会话 ID
    processing_time: float  # 处理时间

#静态资源挂载
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  # 生成唯一会话 ID
    return {"session_id": session_id}  # 返回会话 ID



@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录
        history = qa_system.get_session_history(session_id)
        # 返回会话 ID 和历史记录
        #{"data": {"session_id": session_id, "history": history} ,"errno":0,"errmsg":报错信息, "log_id":日志编号}
        return {"session_id": session_id, "history": history}
    except Exception as e:
        # 抛出 HTTP 异常，包含错误信息
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    # 清除指定会话的历史记录
    success = qa_system.clear_session_history(session_id)
    if success:
        # 返回成功状态
        return {"status": "success", "message": "历史记录已清除"}
    else:
        # 抛出 HTTP 异常
        raise HTTPException(status_code=500, detail="清除历史记录失败")



def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()  # 去除首尾空格
    for pattern_info in GREETING_PATTERNS:
        # 使用正则匹配，忽略大小写, 兼容中英文大小写混用场景.
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]  # 返回匹配的回复
    return None  # 无匹配返回 None


# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()  # 记录开始时间
    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())
    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        # 返回问候回复
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 执行 BM25 搜索
    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        # 需要 RAG，提示使用 WebSocket
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 返回 MySQL 答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }


# 流式查询 WebSocket 接口
# WebSocket长连接流式问答接口, 逐Token分段推送答案, 实现: 打字机流失效果.
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受 WebSocket 连接
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            request_data = json.loads(data)  # 解析 JSON 数据
            # 获取查询参数
            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()  # 记录开始时间
            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送问候回复
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    # 发送结束标志
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break
            # 调用问答系统，流式处理查询
            collected_answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token  # 累积答案
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)  # 控制流式输出的速度
    except WebSocketDisconnect as e:
        # 记录 WebSocket 断开信息
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 记录错误信息
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            # 发送错误消息
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            print(f"Error closing WebSocket: {str(e)}")


@app.get("/health")
async def health_check():
    # 返回健康状态标记, k8s调用该接口返回200则判断服务正常运行.
    return {"status": "healthy"}  # 返回健康状态

# 获取有效论文领域接口
@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}


# 主程序入口
if __name__ == "__main__":
    # springboot = springcore + spirngmvc + tomcat
    # fastapi = springmvc (url -> 方法调用)
    # uvicorn = tomcat (服务容器，负责处理多线程、高并发等)

    import uvicorn      # uvicorn库 -> 异步web服务容器, 用于启动FastAPI应用.
    import os

    # 从环境变量获取主机和端口，默认值为 0.0.0.0:8080
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))

    # 运行 FastAPI 应用，监听指定的主机和端口
    # reload=False 关闭热重载, 生产环境禁用, 避免性能损耗.
    uvicorn.run("app:app", host=host, port=port, reload=False)