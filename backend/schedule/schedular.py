'''
Author: lpz 1565561624@qq.com
Date: 2025-07-29 14:26:11
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-07-29 16:08:30
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/schedule/schedular.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
from collections import deque
from typing import List, Optional


class Request:
    def __init__(
        self,
        user_id: str,
        query: str,
        vector_retrieval: str,
        graph_retrieval: str,
        state: str = None,
        timestamp: float = None,
        flag: bool = None
    ):
        self.user_id = user_id                  # 用户 ID
        self.query = query                      # 查询内容
        self.vector_retrieval = vector_retrieval  # 是否启用向量检索
        self.graph_retrieval = graph_retrieval    # 是否启用图检索
        self.state = state                      # 当前状态（如 init / processed / failed）
        self.timestamp = timestamp or time.time()  # 时间戳，默认为当前时间
        self.flag = flag

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "query": self.query,
            "vector_retrieval": self.vector_retrieval,
            "graph_retrieval": self.graph_retrieval,
            "state": self.state,
            "timestamp": self.timestamp,
        }

    def __repr__(self):
        return f"Request({self.to_dict()})"
    

    class Schedular:
        def __init__(self):
            self.waiting_queue = deque()   # 等待处理的请求
            self.processing_queue = deque() #正在处理的请求
            self.processing_queue = deque() #处理完准备发给 llm的请求
            self.finished = []    # 完成的request，准备返回给用户的
        def add_request(self, request):
            """添加新请求"""
            self.waiting_queue.append(request)
            print(f"[{time.strftime('%H:%M:%S')}] New request queued: {request.user_id} → {request.query}")